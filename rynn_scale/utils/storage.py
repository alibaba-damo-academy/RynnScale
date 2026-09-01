import ctypes
import importlib
import io
import json
import os
import posixpath
import shutil
import struct
import tempfile
import threading
import warnings
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union, cast

import oss2
import safetensors.torch
import torch
from oss2.credentials import EnvironmentVariableCredentialsProvider
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint.filesystem import (
    FileSystemBase,
    FileSystemReader,
    _metadata_fn,
)
from torch.distributed.checkpoint.planner import LoadItemType, LoadPlan, LoadPlanner
from torch.futures import Future
from tqdm import tqdm
from transformers import CONFIG_MAPPING, AutoProcessor, PreTrainedConfig, ProcessorMixin
from transformers.utils import cached_file

from .logging import get_logger

logger = get_logger(__name__)

_auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
_buckets = {}


def is_oss(path: Union[str, os.PathLike]) -> bool:
    return str(path).startswith("oss://")


def _get_bucket(bucket_name: str):
    global _buckets
    if bucket_name not in _buckets:
        _buckets[bucket_name] = oss2.Bucket(
            auth=_auth,
            endpoint=os.getenv("OSS_ENDPOINT"),
            bucket_name=bucket_name,
        )
    return _buckets[bucket_name]


def _parse_oss_path(oss_path: str):
    assert oss_path.startswith("oss://"), f"oss path must start with oss://, but got {oss_path}"
    splits = oss_path.replace("oss://", "").split("/", maxsplit=1)
    assert len(splits) == 2, f"oss path must be in format oss://{{bucket}}/{{object}}, but got {oss_path}"
    return splits[0], splits[1]


def _exec_with_retry(func: Callable, retry: int = 5):
    while True:
        try:
            return func()
        except oss2.exceptions.RequestError as e:
            retry = retry - 1
            if retry < 0:
                raise e


def clear_cache():
    global _buckets
    _buckets.clear()


def _object_exists(oss_path: str):
    bucket_name, object_name = _parse_oss_path(oss_path)
    return _get_bucket(bucket_name).object_exists(object_name)


def _get_object(oss_path: str, byte_range: Optional[Tuple[int, int]] = None, retry: int = 5):
    bucket_name, object_name = _parse_oss_path(oss_path)
    bucket = _get_bucket(bucket_name)
    func = partial(bucket.get_object, object_name, byte_range=byte_range)
    return _exec_with_retry(func, retry)


def _put_object(oss_path: str, obj: Any, retry: int = 5):
    bucket_name, object_name = _parse_oss_path(oss_path)
    bucket = _get_bucket(bucket_name)
    func = partial(bucket.put_object, object_name, obj)
    return _exec_with_retry(func, retry)


def _oss_sign_url(oss_path):
    bucket_name, object_name = _parse_oss_path(oss_path)
    return _get_bucket(bucket_name).sign_url("GET", object_name, 3600)


def _get_object_to_file(oss_path: str, local_path: str, retry: int = 5):
    bucket_name, object_name = _parse_oss_path(oss_path)
    bucket = _get_bucket(bucket_name)
    func = partial(bucket.get_object_to_file, object_name, local_path)
    return _exec_with_retry(func, retry)


def _put_object_from_file(oss_path: str, local_path: str, retry: int = 5):
    bucket_name, object_name = _parse_oss_path(oss_path)
    bucket = _get_bucket(bucket_name)
    func = partial(bucket.put_object_from_file, object_name, local_path)
    return _exec_with_retry(func, retry)


def _oss_load_processor(
    oss_path: str,
    processor_class: Optional[type[ProcessorMixin]] = None,
    **kwargs,
):
    if processor_class is None:
        processor_class = AutoProcessor

    temp_dir = tempfile.mkdtemp()

    bucket_name, object_name = _parse_oss_path(oss_path)
    bucket = _get_bucket(bucket_name)

    prefix = os.path.join(object_name, "")
    for obj in oss2.ObjectIteratorV2(bucket=bucket, prefix=prefix, delimiter="/"):
        if obj.is_prefix() or obj.key == prefix:
            continue
        if obj.key.endswith(".safetensors") or obj.key.endswith(".bin"):
            continue
        bucket.get_object_to_file(
            obj.key,
            os.path.join(temp_dir, os.path.basename(obj.key)),
        )

    processor = processor_class.from_pretrained(temp_dir, **kwargs)
    shutil.rmtree(temp_dir)

    return processor


def _oss_isdir(path):
    bucket_name, prefix = _parse_oss_path(path)
    bucket = _get_bucket(bucket_name)
    for obj in oss2.ObjectIterator(bucket, prefix=prefix, delimiter="/"):
        if obj.key == prefix:
            continue
        return True
    return False


def _oss_listdir(path):
    path = os.path.join(path, "")
    bucket_name, prefix = _parse_oss_path(path)
    bucket = _get_bucket(bucket_name)
    continuation_token = ""

    outputs = []
    while True:
        result = bucket.list_objects_v2(
            prefix=prefix,
            delimiter="/",
            continuation_token=continuation_token,
        )

        for obj_or_prefix in result.object_list + result.prefix_list:
            if hasattr(obj_or_prefix, "key"):
                abs_path = obj_or_prefix.key
            else:
                abs_path = obj_or_prefix
            if abs_path == prefix:
                continue
            rel_path = os.path.relpath(abs_path, prefix)
            outputs.append(rel_path)

        if result.is_truncated:
            continuation_token = result.next_continuation_token
        else:
            break

    return outputs


def _oss_walk(path: str):
    path = os.path.join(path, "")
    bucket_name, prefix = _parse_oss_path(path)
    bucket = _get_bucket(bucket_name)
    continuation_token = ""

    dirs, files = [], []
    while True:
        result = bucket.list_objects_v2(
            prefix=prefix,
            delimiter="/",
            continuation_token=continuation_token,
        )

        for d in result.prefix_list:
            if d == prefix:
                continue
            dirs.append(os.path.relpath(d, prefix))

        for obj in result.object_list:
            if obj.key == prefix:
                continue
            files.append(os.path.relpath(obj.key, prefix))

        if result.is_truncated:
            continuation_token = result.next_continuation_token
        else:
            break

    yield path, dirs, files

    for d in dirs:
        yield from _oss_walk(os.path.join(path, d))


def _oss_rmtree(path: str):
    path = os.path.join(path, "")
    bucket_name, prefix = _parse_oss_path(path)
    bucket = _get_bucket(bucket_name)
    for obj in oss2.ObjectIterator(bucket, prefix=prefix):
        bucket.delete_object(obj.key)


def _oss_torch_save(obj: object, path: str, retry: int = 5):
    fd, tmp_path = tempfile.mkstemp()
    os.close(fd)

    try:
        torch.save(obj, tmp_path)
        _put_object_from_file(
            oss_path=path,
            local_path=tmp_path,
            retry=retry,
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


_TORCH_DTYPE_TO_SAFETENSORS = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.float8_e5m2: "F8_E5M2",
}
_SAFETENSORS_DTYPE_TO_TORCH = {v: k for k, v in _TORCH_DTYPE_TO_SAFETENSORS.items()}


class open_safetensors:
    def __init__(self, path: str, framework: str = "pt", device: str = "cpu"):
        assert framework == "pt", f"only framework='pt' is supported, got {framework!r}"
        assert device == "cpu", f"only device='cpu' is supported, got {device!r}"
        self._path = path
        self._data_offset: Optional[int] = None
        self._tensor_infos: Optional[Dict[str, Dict[str, Any]]] = None
        self._user_metadata: Optional[Dict[str, str]] = None

    def __enter__(self):
        self._load_header()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

    def _read_range(self, start: int, end: int) -> bytes:
        if is_oss(self._path):
            with _get_object(self._path, byte_range=(start, end - 1)) as stream:
                return stream.read()
        with open(self._path, "rb") as f:
            f.seek(start)
            return f.read(end - start)

    def _load_header(self):
        if self._data_offset is not None:
            return
        header_len = struct.unpack("<Q", self._read_range(0, 8))[0]
        header = json.loads(self._read_range(8, 8 + header_len).decode("utf-8"))
        self._user_metadata = header.pop("__metadata__", None)
        self._tensor_infos = header
        self._data_offset = 8 + header_len

    def keys(self) -> List[str]:
        assert self._tensor_infos is not None, "open_safetensors must be used as a context manager"
        return list(self._tensor_infos.keys())

    def metadata(self) -> Optional[Dict[str, str]]:
        assert self._data_offset is not None, "open_safetensors must be used as a context manager"
        return self._user_metadata

    def get_tensor(self, name: str) -> torch.Tensor:
        assert self._tensor_infos is not None, "open_safetensors must be used as a context manager"
        info = self._tensor_infos[name]
        dtype = _SAFETENSORS_DTYPE_TO_TORCH[info["dtype"]]
        shape = info["shape"]
        start, end = info["data_offsets"]
        buf = self._read_range(self._data_offset + start, self._data_offset + end)
        return torch.frombuffer(bytearray(buf), dtype=dtype).view(shape)


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.contiguous()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    nbytes = tensor.numel() * tensor.element_size()
    if nbytes == 0:
        return b""
    ptr = tensor.data_ptr()
    if ptr == 0:
        return b""
    raw = (ctypes.c_ubyte * nbytes).from_address(ptr)
    return bytes(raw)


def _oss_save_safetensors(
    state_dict: Dict[str, torch.Tensor],
    path: str,
    metadata: Optional[Dict[str, str]] = None,
    max_workers: int = 8,
    target_part_size: int = 64 * 1024 * 1024,
    retry: int = 5,
):
    """Save a state_dict to OSS as a single safetensors file using a
    multipart upload. Tensors are uploaded across parts concurrently from
    a ThreadPoolExecutor.

    The file layout matches the safetensors spec:
        [ u64 header_length ][ JSON header ][ raw tensor data ]
    """
    # Build header JSON and compute per-tensor offsets.
    header_dict: Dict[str, Any] = {}
    if metadata:
        header_dict["__metadata__"] = metadata

    tensors = list(state_dict.items())
    nbytes_list: List[int] = []
    cursor = 0
    for name, tensor in tensors:
        if tensor.dtype not in _TORCH_DTYPE_TO_SAFETENSORS:
            raise TypeError(f"Unsupported safetensors dtype for {name}: {tensor.dtype}")
        nbytes = tensor.numel() * tensor.element_size()
        header_dict[name] = {
            "dtype": _TORCH_DTYPE_TO_SAFETENSORS[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [cursor, cursor + nbytes],
        }
        nbytes_list.append(nbytes)
        cursor += nbytes

    header_json = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")
    # Pad the header to an 8-byte boundary so tensor data is aligned.
    padding = (8 - len(header_json) % 8) % 8
    if padding:
        header_json = header_json + b" " * padding
    header_prefix = struct.pack("<Q", len(header_json)) + header_json

    # Group segments (header + tensors, in file order) into multipart parts.
    # OSS requires every non-last part to be >= 100KB.
    MIN_PART_SIZE = 100 * 1024
    segments_sizes = [len(header_prefix)] + nbytes_list  # segment 0 = header
    parts: List[List[int]] = []
    current: List[int] = []
    current_size = 0
    for idx, size in enumerate(segments_sizes):
        if current and current_size + size > target_part_size and current_size >= MIN_PART_SIZE:
            parts.append(current)
            current = []
            current_size = 0
        current.append(idx)
        current_size += size
    if current:
        parts.append(current)

    # If we ended up with multiple parts and the last one is too small to
    # stand alone but the penultimate one is large, OSS allows it; the min
    # only applies to non-last parts. Our packing above already guarantees
    # non-last parts >= MIN_PART_SIZE when possible.

    bucket_name, object_name = _parse_oss_path(path)
    bucket = _get_bucket(bucket_name)
    init_result = _exec_with_retry(
        partial(bucket.init_multipart_upload, object_name),
        retry=retry,
    )
    upload_id = init_result.upload_id

    def _segment_bytes(seg_idx: int) -> bytes:
        if seg_idx == 0:
            return header_prefix
        _, tensor = tensors[seg_idx - 1]
        return _tensor_to_bytes(tensor)

    def _upload_one(part_number: int, seg_indices: List[int]):
        if len(seg_indices) == 1:
            payload = _segment_bytes(seg_indices[0])
        else:
            payload = b"".join(_segment_bytes(i) for i in seg_indices)
        res = _exec_with_retry(
            partial(bucket.upload_part, object_name, upload_id, part_number, payload),
            retry=retry,
        )
        return oss2.models.PartInfo(part_number, res.etag)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_upload_one, i + 1, seg_indices) for i, seg_indices in enumerate(parts)]
            part_infos = [f.result() for f in futures]
        part_infos.sort(key=lambda p: p.part_number)
        _exec_with_retry(
            partial(bucket.complete_multipart_upload, object_name, upload_id, part_infos),
            retry=retry,
        )
    except Exception:
        try:
            bucket.abort_multipart_upload(object_name, upload_id)
        except Exception:
            logger.warning(f"Failed to abort multipart upload {upload_id} for {path}")
        raise


def _oss_torch_load(
    path: str,
    map_location: Optional[str] = None,
    weights_only: Optional[bool] = None,
    retry: int = 5,
):
    with _get_object(path, retry=retry) as result:
        buffer = io.BytesIO(result.read())
        state_dict = torch.load(
            buffer,
            map_location=map_location,
            weights_only=weights_only,
        )
        buffer.close()
    return state_dict


class _TemporaryDirectory(tempfile.TemporaryDirectory):
    def __init__(
        self,
        oss_path: str,
        mode: str = "download",
        include: Optional[List[str]] = None,
        ignore_cleanup_errors: bool = False,
        delete: bool = True,
    ):
        assert mode in ["download", "upload"], f"mode must be 'download' or 'upload', but got {mode}"
        self.oss_path = oss_path
        self.mode = mode
        self.include = include

        self.name = tempfile.mkdtemp()
        self._ignore_cleanup_errors = ignore_cleanup_errors
        self._delete = delete
        self._finalizer = weakref.finalize(
            self,
            self._cleanup,
            oss_path=self.oss_path,
            mode=self.mode,
            include=self.include,
            name=self.name,
            warn_message="Implicitly cleaning up {!r}".format(self),
            ignore_errors=self._ignore_cleanup_errors,
            delete=self._delete,
        )

        if mode == "download":
            self._download(self.oss_path, self.include, self.name)

    @staticmethod
    def _download(oss_path: str, include: Optional[List[str]], name: str):
        for root, _, files in _oss_walk(oss_path):
            for f in files:
                remote_path = os.path.join(root, f)
                rel_path = os.path.relpath(remote_path, oss_path)
                local_path = os.path.join(name, rel_path)
                if include is not None and rel_path not in include:
                    continue
                logger.debug(f"Downloading {remote_path} to {local_path} ...")
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                _get_object_to_file(remote_path, local_path)

    @staticmethod
    def _upload(oss_path: str, include: Optional[List[str]], name: str):
        for root, _, files in os.walk(name):
            for f in files:
                local_path = os.path.join(root, f)
                rel_path = os.path.relpath(local_path, name)
                remote_path = os.path.join(oss_path, rel_path)
                if include is not None and rel_path not in include:
                    continue
                logger.debug(f"Uploaded {local_path} to {remote_path} ...")
                _put_object_from_file(remote_path, local_path)

    def cleanup(self):
        if self._finalizer.detach() or os.path.exists(self.name):
            if self.mode == "upload":
                self._upload(self.oss_path, self.include, self.name)
            self._rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)

    @classmethod
    def _cleanup(cls, oss_path, mode, include, name, warn_message, ignore_errors=False, delete=True):
        if mode == "upload":
            cls._upload(oss_path, include, name)
        if delete:
            cls._rmtree(name, ignore_errors=ignore_errors)
            warnings.warn(warn_message, ResourceWarning)


class _OSSReadableFile(io.IOBase):
    def __init__(self, oss_path: str):
        super().__init__()
        self._oss_path = oss_path
        self._size: Optional[int] = None
        self._pos = 0

    def _ensure_size(self) -> int:
        if self._size is None:
            bucket_name, object_name = _parse_oss_path(self._oss_path)
            bucket = _get_bucket(bucket_name)
            head = _exec_with_retry(partial(bucket.head_object, object_name))
            self._size = head.content_length
        return self._size

    def seek(self, offset: int, whence: int = io.SEEK_SET, /) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos = self._pos + offset
        elif whence == io.SEEK_END:
            self._pos = self._ensure_size() + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        return self._pos

    def tell(self) -> int:
        return self._pos

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._ensure_size() - self._pos
        if size <= 0:
            return b""
        end = self._pos + size - 1
        with _get_object(self._oss_path, byte_range=(self._pos, end)) as stream:
            data = stream.read()
        self._pos += len(data)
        return data

    def readinto(self, buf) -> int:
        data = self.read(len(buf))
        n = len(data)
        buf[:n] = data
        return n


class _OSSFileSystem(FileSystemBase):
    """Read-only ``FileSystemBase`` backed by OSS byte-range gets."""

    @contextmanager
    def create_stream(self, path: Union[str, os.PathLike], mode: str) -> Generator[io.IOBase, None, None]:
        assert mode == "rb", f"_OSSFileSystem only supports reading, got mode={mode!r}"
        path_str = str(path)
        # ``.metadata`` is small but ``pickle.load`` makes many tiny reads
        # while parsing it — on an OSS stream each becomes a full RTT, so a
        # single MB-sized metadata can cost seconds × world_size. Pull the
        # whole file once and let pickle read against memory instead.
        if path_str.endswith(_metadata_fn):
            with _get_object(path_str) as stream:
                buf = io.BytesIO(stream.read())
            try:
                yield buf
            finally:
                buf.close()
            return
        f = _OSSReadableFile(path_str)
        try:
            yield f
        finally:
            f.close()

    def concat_path(self, path: Union[str, os.PathLike], suffix: str) -> str:
        return posixpath.join(str(path), suffix)

    def init_path(self, path: Union[str, os.PathLike]) -> str:
        return str(path)

    def exists(self, path: Union[str, os.PathLike]) -> bool:
        return _object_exists(str(path))

    def rename(self, path: Union[str, os.PathLike], new_path: Union[str, os.PathLike]) -> None:
        raise NotImplementedError("_OSSFileSystem is read-only")

    def mkdir(self, path: Union[str, os.PathLike]) -> None:
        raise NotImplementedError("_OSSFileSystem is read-only")

    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        raise NotImplementedError("_OSSFileSystem is read-only")

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return str(checkpoint_id).startswith("oss://")


class OSSFileSystemReader(FileSystemReader):
    """Read a DCP checkpoint directly from OSS via byte-range gets, skipping
    the full-checkpoint download that ``FileSystemReader`` would otherwise need.

    For each ``.distcp`` file the reader sorts the requested chunks by offset
    and merges chunks whose gap is below ``coalesce_gap_bytes`` (capped at
    ``coalesce_max_bytes`` per merged range to bound memory). Each merged
    range becomes one large OSS GET issued from a thread pool of size
    ``num_workers``; per-chunk decode happens against an in-memory buffer.
    This collapses the "one tensor → one range request" pattern that
    dominates wall time on resharded loads (each RTT is ~100ms on OSS).

    ``num_workers=0`` falls back to upstream's serial ``FileSystemReader``
    path, which is still useful for debugging.
    """

    def __init__(
        self,
        path: str,
        num_workers: int = 8,
        coalesce_gap_bytes: int = 1 << 20,
        coalesce_max_bytes: int = 256 << 20,
        show_progress: bool = False,
    ):
        super().__init__(path)
        self.fs = _OSSFileSystem()
        self.path = self.fs.init_path(path)
        self.num_workers = num_workers
        self.coalesce_gap_bytes = coalesce_gap_bytes
        self.coalesce_max_bytes = coalesce_max_bytes
        self.show_progress = show_progress

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return _OSSFileSystem.validate_checkpoint_id(checkpoint_id)

    def _build_fetch_groups(self, plan: LoadPlan) -> List[Tuple[str, int, int, List[Tuple[Any, int, int]]]]:
        per_file: Dict[str, List[Tuple[Any, int, int]]] = {}
        for req in plan.items:
            md = self.storage_data[req.storage_index]
            per_file.setdefault(md.relative_path, []).append((req, int(md.offset), int(md.length)))

        groups: List[Tuple[str, int, int, List[Tuple[Any, int, int]]]] = []
        for path, items in per_file.items():
            items.sort(key=lambda x: x[1])
            cur_start: Optional[int] = None
            cur_end: int = 0
            cur_items: List[Tuple[Any, int, int]] = []
            for req, off, length in items:
                if length <= 0:
                    # zero-length still goes through decode (empty tensor /
                    # empty bytes); attach to the current group at off=off.
                    if cur_start is None:
                        cur_start, cur_end, cur_items = off, off, [(req, off, length)]
                    else:
                        cur_items.append((req, off, length))
                    continue
                end = off + length
                if cur_start is None:
                    cur_start, cur_end, cur_items = off, end, [(req, off, length)]
                    continue
                gap = off - cur_end
                new_end = max(cur_end, end)
                if gap <= self.coalesce_gap_bytes and (new_end - cur_start) <= self.coalesce_max_bytes:
                    cur_end = new_end
                    cur_items.append((req, off, length))
                else:
                    groups.append((path, cur_start, cur_end, cur_items))
                    cur_start, cur_end, cur_items = off, end, [(req, off, length)]
            if cur_start is not None:
                groups.append((path, cur_start, cur_end, cur_items))
        return groups

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        if self.num_workers <= 0:
            return super().read_data(plan, planner)

        groups = self._build_fetch_groups(plan)

        # ``planner.{resolve,commit}_tensor`` and ``load_bytes`` mutate the
        # planner's state_dict; serialize those calls. The OSS gets and the
        # tensor decode/copy still happen concurrently across groups.
        planner_lock = threading.Lock()

        def process_group(
            relative_path: str,
            fstart: int,
            fend: int,
            items: List[Tuple[Any, int, int]],
        ) -> None:
            full_path = self.fs.concat_path(self.path, relative_path)
            if fend > fstart:
                with _get_object(full_path, byte_range=(fstart, fend - 1)) as stream:
                    buf = stream.read()
            else:
                buf = b""
            view = memoryview(buf)
            for req, off, length in items:
                local = off - fstart
                sub = io.BytesIO(bytes(view[local : local + length]))
                md = self.storage_data[req.storage_index]
                transform_from = self.transforms.transform_load_stream(
                    req,
                    md.transform_descriptors or (),
                    sub,
                )

                if req.type == LoadItemType.BYTE_IO:
                    read_bytes = io.BytesIO(transform_from.read(-1))
                    read_bytes.seek(0)
                    with planner_lock:
                        planner.load_bytes(req, read_bytes)
                else:
                    if transform_from.seekable():
                        seekable = transform_from
                    else:
                        seekable = io.BytesIO(transform_from.read(-1))
                        seekable.seek(0)
                    tensor = cast(
                        torch.Tensor,
                        torch.load(seekable, map_location="cpu", weights_only=True),
                    )
                    tensor = narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
                    with planner_lock:
                        target_tensor = planner.resolve_tensor(req).detach()
                        assert target_tensor.size() == tensor.size(), (
                            f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                        )
                        target_tensor.copy_(tensor)
                        planner.commit_tensor(req, target_tensor)

        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            futures = [ex.submit(process_group, *g) for g in groups]
            iterator = as_completed(futures)
            if self.show_progress:
                total_bytes = sum(max(0, fend - fstart) for _, fstart, fend, _ in groups)
                iterator = tqdm(
                    iterator,
                    total=len(futures),
                    desc=f"Load checkpoint ({total_bytes / (1 << 30):.2f} GiB)",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                )
            for f in iterator:
                f.result()

        fut: Future = Future()
        fut.set_result(None)
        return fut


# ---------------------------------------------------------------------------
# Unified interface — dispatches on the ``oss://`` prefix so callers don't
# need to special-case local vs OSS paths.
# ---------------------------------------------------------------------------


def exists(path: str) -> bool:
    if is_oss(path):
        return _object_exists(path)
    return os.path.exists(path)


def isdir(path: str) -> bool:
    if is_oss(path):
        return _oss_isdir(path)
    return os.path.isdir(path)


def listdir(path: str) -> List[str]:
    if is_oss(path):
        return _oss_listdir(path)
    return os.listdir(path)


def walk(path: str):
    if is_oss(path):
        yield from _oss_walk(path)
    else:
        yield from os.walk(path)


def rmtree(path: str):
    if is_oss(path):
        _oss_rmtree(path)
    else:
        shutil.rmtree(path, ignore_errors=True)


def makedirs(path: str):
    if is_oss(path):
        return
    os.makedirs(path, exist_ok=True)


def sign_url(path: str) -> str:
    if is_oss(path):
        return _oss_sign_url(path)
    return path


class _OSSWriteBuffer(io.BytesIO):
    """In-memory write buffer that uploads its contents to OSS on ``close``."""

    def __init__(self, oss_path: str):
        super().__init__()
        self._oss_path = oss_path
        self._uploaded = False

    def close(self):
        if not self.closed and not self._uploaded:
            _put_object(self._oss_path, self.getvalue())
            self._uploaded = True
        super().close()


def open_file(path: str, mode: str = "rb", *, stream: bool = False) -> io.IOBase:
    """Open a local or OSS path and return a file object, like the builtin ``open``.

    The returned object supports ``read``/``write``/``seek`` and the context
    manager protocol, so ``with storage.open_file(...) as f:`` works and the
    destination is finalized on close (OSS writes upload then).

    ``stream`` selects the OSS *read* strategy (ignored for local paths and writes):

    * ``False`` (default): fetch the whole object once into a seekable in-memory
      buffer — one GET, best when the file is read in full (pickle/json/image).
    * ``True``: read lazily via a seekable reader backed by byte-range GETs — for
      random access or partial reads of large files (e.g. video decoding).
    """
    writing = "w" in mode or "a" in mode or "x" in mode
    if not is_oss(path):
        return open(path, mode)
    if writing:
        return _OSSWriteBuffer(path)
    if stream:
        return _OSSReadableFile(path)
    with _get_object(path) as resp:
        return io.BytesIO(resp.read())


def torch_save(obj: object, path: str, retry: int = 5):
    if is_oss(path):
        _oss_torch_save(obj, path, retry=retry)
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(obj, path)


def torch_load(
    path: str,
    map_location: Optional[str] = None,
    weights_only: Optional[bool] = None,
    retry: int = 5,
):
    if is_oss(path):
        return _oss_torch_load(path, map_location=map_location, weights_only=weights_only, retry=retry)
    return torch.load(path, map_location=map_location, weights_only=weights_only)


def save_safetensors(
    state_dict: Dict[str, torch.Tensor],
    path: str,
    metadata: Optional[Dict[str, str]] = None,
    **kwargs,
):
    if is_oss(path):
        _oss_save_safetensors(state_dict, path, metadata=metadata, **kwargs)
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        safetensors.torch.save_file(state_dict, path, metadata=metadata)


def load_config(path: str, model_type: Optional[str] = None, **kwargs):
    """Load a config, resolving model types transformers does not know about.

    A model type implemented in this repo is absent from transformers' registry
    until its package registers it, and that cannot happen before the config
    names the package.

    ``model_type`` overrides the type named in the checkpoint, which is how a
    model is initialized from a checkpoint of a different (base) type -- e.g. a
    ``rynn_brain_vla`` from a ``qwen3_vl`` VLM.
    """
    if is_oss(path):
        with tempfile.NamedTemporaryFile() as tmp_file:
            _get_object_to_file(os.path.join(path, "config.json"), tmp_file.name)
            return load_config(tmp_file.name, model_type=model_type, **kwargs)

    if model_type is None:
        if os.path.isdir(path):
            config_file = os.path.join(path, "config.json")
        elif os.path.isfile(path):
            config_file = path
        else:
            config_file = cached_file(path, "config.json")
        with open(config_file) as f:
            model_type = json.load(f).get("model_type")

    if model_type in CONFIG_MAPPING:
        return CONFIG_MAPPING[model_type].from_pretrained(path, **kwargs)

    package_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", str(model_type))
    if os.path.isfile(os.path.join(package_dir, "__init__.py")):
        module = importlib.import_module(f"..models.{model_type}", package=__package__)
        for value in vars(module).values():
            if (
                isinstance(value, type)
                and issubclass(value, PreTrainedConfig)
                and getattr(value, "model_type", None) == model_type
            ):
                # Register right away: transformers' Auto* classes go on resolving this
                # type on their own later (AutoTokenizer silently falls back to the base
                # config and warns when it is unknown) and they have no way back here.
                CONFIG_MAPPING.register(model_type, value, exist_ok=True)
                return value.from_pretrained(path, **kwargs)

    raise ValueError(
        f"The checkpoint you are trying to load has model type `{model_type}` but "
        "transformers does not recognize this architecture. This could be because of "
        "an issue with the checkpoint, or because your version of transformers is out "
        f"of date.\n\nModel type `{model_type}` is not implemented under "
        f"`rynn_scale/models` either. Add `rynn_scale/models/{model_type}` or upgrade "
        "transformers."
    )


def load_processor(
    path: str,
    processor_class: Optional[type[ProcessorMixin]] = None,
    **kwargs,
):
    if is_oss(path):
        return _oss_load_processor(path, processor_class=processor_class, **kwargs)
    if processor_class is None:
        processor_class = AutoProcessor
    return processor_class.from_pretrained(path, **kwargs)


def get_storage_reader(path: str, num_workers: int = 8, show_progress: bool = False):
    if is_oss(path):
        return OSSFileSystemReader(path, num_workers=num_workers, show_progress=show_progress)
    return FileSystemReader(path)


@contextmanager
def writable_dir(path: str):
    """Yield a local directory to write into, uploading to ``path`` on exit
    when ``path`` is an OSS location."""
    if is_oss(path):
        tmp_dir = _TemporaryDirectory(oss_path=path, mode="upload")
        try:
            yield tmp_dir.name
        finally:
            tmp_dir.cleanup()
    else:
        os.makedirs(path, exist_ok=True)
        yield path


@contextmanager
def readable_dir(path: str, include: Optional[List[str]] = None):
    """Yield a local directory mirroring ``path``, downloading from OSS first
    when ``path`` is an OSS location."""
    if is_oss(path):
        tmp_dir = _TemporaryDirectory(oss_path=path, mode="download", include=include)
        try:
            yield tmp_dir.name
        finally:
            tmp_dir.cleanup()
    else:
        yield path
