import os
import shutil
import tempfile
import warnings
import weakref
from functools import partial
from typing import Any, Callable, Optional, List

import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
from transformers import AutoConfig, AutoProcessor

from .logging import get_logger

logger = get_logger(__name__)

_auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
_buckets = {}


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


def object_exists(oss_path: str):
    bucket_name, object_name = _parse_oss_path(oss_path)
    return _get_bucket(bucket_name).object_exists(object_name)


def get_object(oss_path: str, retry: int = 5):
    bucket_name, object_name = _parse_oss_path(oss_path)
    bucket = _get_bucket(bucket_name)
    func = partial(bucket.get_object, object_name)
    return _exec_with_retry(func, retry)


def put_object(oss_path: str, obj: Any, retry: int = 5):
    bucket_name, object_name = _parse_oss_path(oss_path)
    bucket = _get_bucket(bucket_name)
    func = partial(bucket.put_object, object_name, obj)
    return _exec_with_retry(func, retry)


def sign_url(oss_path):
    bucket_name, object_name = _parse_oss_path(oss_path)
    return _get_bucket(bucket_name).sign_url("GET", object_name, 3600)


def get_object_to_file(oss_path: str, local_path: str, retry: int = 5):
    bucket_name, object_name = _parse_oss_path(oss_path)
    bucket = _get_bucket(bucket_name)
    func = partial(bucket.get_object_to_file, object_name, local_path)
    return _exec_with_retry(func, retry)


def put_object_from_file(oss_path: str, local_path: str, retry: int = 5):
    bucket_name, object_name = _parse_oss_path(oss_path)
    bucket = _get_bucket(bucket_name)
    func = partial(bucket.put_object_from_file, object_name, local_path)
    return _exec_with_retry(func, retry)


def load_config(oss_path: str):
    with tempfile.NamedTemporaryFile() as tmp_file:
        get_object_to_file(
            os.path.join(oss_path, "config.json"),
            tmp_file.name,
        )
        config = AutoConfig.from_pretrained(tmp_file.name)
    return config


def load_processor(oss_path: str):
    temp_dir = tempfile.mkdtemp()

    bucket_name, object_name = _parse_oss_path(oss_path)
    bucket = _get_bucket(bucket_name)

    prefix = os.path.join(object_name, "")
    for obj in oss2.ObjectIteratorV2(
        bucket=bucket, prefix=prefix, delimiter="/"
    ):
        if obj.is_prefix() or obj.key == prefix:
            continue
        if obj.key.endswith(".safetensors") or obj.key.endswith(".bin"):
            continue
        bucket.get_object_to_file(
            obj.key,
            os.path.join(temp_dir, os.path.basename(obj.key)),
        )

    processor = AutoProcessor.from_pretrained(temp_dir)
    shutil.rmtree(temp_dir)

    return processor


def isdir(path):
    bucket_name, prefix = _parse_oss_path(path)
    bucket = _get_bucket(bucket_name)
    for obj in oss2.ObjectIterator(bucket, prefix=prefix, delimiter="/"):
        if obj.key == prefix:
            continue
        return True
    return False


def listdir(path):
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

        for obj_or_prefix in (result.object_list + result.prefix_list):
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


def walk(path: str):
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
        yield from walk(os.path.join(path, d))


def rmtree(path: str):
    path = os.path.join(path, "")
    bucket_name, prefix = _parse_oss_path(path)
    bucket = _get_bucket(bucket_name)
    for obj in oss2.ObjectIterator(bucket, prefix=prefix):
        bucket.delete_object(obj.key)


class TemporaryDirectory(tempfile.TemporaryDirectory):
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
            ignore_errors=self._ignore_cleanup_errors, delete=self._delete
        )

        if mode == "download":
            self._download(self.oss_path, self.include, self.name)

    @staticmethod
    def _download(oss_path: str, include: Optional[List[str]], name: str):
        for root, _, files in walk(oss_path):
            for f in files:
                remote_path = os.path.join(root, f)
                rel_path = os.path.relpath(remote_path, oss_path)
                local_path = os.path.join(name, rel_path)
                if include is not None and rel_path not in include:
                    continue
                logger.debug(f"Downloading {remote_path} to {local_path} ...")
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                get_object_to_file(remote_path, local_path)

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
                put_object_from_file(remote_path, local_path)

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
