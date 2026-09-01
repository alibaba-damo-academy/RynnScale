import functools
import io
import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from typing import Dict, Iterator, List, Mapping

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ...constants import RobotType, RotationRepresentation
from ...registry import DATASET_REGISTRY
from ...utils.robot import Arm, Position, RobotAction, RobotState
from ...utils.robot import Rotation as EefRotation
from .base import BaseVLADataset, EpisodeMetadata

INDEX_NAME = "episodes.jsonl"


SOURCE_FPS = 15.0  # DROID is captured at fixed 15Hz across all shards.
DROID_PROBE_KEY = b"steps/discount"

FLOAT_FIELDS = {
    b"steps/observation/cartesian_position": ("cartesian_position", 6),
    b"steps/observation/joint_position": ("joint_position", 7),
    b"steps/observation/gripper_position": ("gripper_position", 1),
    b"steps/action_dict/cartesian_position": ("action_cartesian_position", 6),
    b"steps/action_dict/joint_position": ("action_joint_position", 7),
    b"steps/action_dict/gripper_position": ("action_gripper_position", 1),
}

BYTES_FIELDS = {
    b"steps/language_instruction": "instruction",
    b"steps/observation/wrist_image_left": "wrist_images",
    b"steps/observation/exterior_image_1_left": "exterior_image_1",
    b"steps/observation/exterior_image_2_left": "exterior_image_2",
}


# ── TFRecord low-level utilities (reused by other TFDS-format datasets) ──
def _decode_varint(buf, pos):
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        result |= (b & 0x7F) << shift
        pos += 1
        if not (b & 0x80):
            return result, pos
        shift += 7


class _PreadReader:
    """os.pread-based on-demand reader: seeks across large entries cost no I/O."""

    __slots__ = ("_fd", "_pos", "_buf", "_buf_start")

    def __init__(self, fd):
        self._fd = fd
        self._pos = 0
        self._buf = b""
        self._buf_start = 0

    def seek(self, pos):
        self._pos = pos

    @property
    def pos(self):
        return self._pos

    def _fill(self, n):
        rel = self._pos - self._buf_start
        if 0 <= rel <= len(self._buf) - n:
            return
        self._buf = os.pread(self._fd, max(n, 4096), self._pos)
        self._buf_start = self._pos

    def read_varint(self):
        self._fill(10)
        val, end = _decode_varint(self._buf, self._pos - self._buf_start)
        self._pos = self._buf_start + end
        return val

    def read(self, n):
        self._fill(n)
        off = self._pos - self._buf_start
        self._pos += n
        return self._buf[off : off + n]

    def read_u64le(self):
        self._fill(8)
        off = self._pos - self._buf_start
        self._pos += 8
        return struct.unpack_from("<Q", self._buf, off)[0]


def _scan_shard(shard_path, probe_key: bytes):
    """Scan one TFRecord shard via protobuf length-prefix skipping.

    Only decodes far enough to count frames per record by reading ``probe_key``
    — a per-step packed FloatList with exactly **one** float per step (so
    ``packed_len // 4 == num_frames``). Skips over large value blobs (images)
    with zero I/O via seek.
    """
    offsets = []
    frame_counts = []
    file_size = os.path.getsize(shard_path)

    fd = os.open(shard_path, os.O_RDONLY)
    try:
        r = _PreadReader(fd)
        pos = 0
        while pos < file_size:
            r.seek(pos)
            record_len = r.read_u64le()
            r.read(4)
            data_start = pos + 12
            offsets.append(pos)

            r.seek(data_start)
            r.read_varint()
            features_len = r.read_varint()
            features_end = r.pos + features_len

            while r.pos < features_end:
                r.read_varint()
                entry_len = r.read_varint()
                entry_end = r.pos + entry_len

                r.read_varint()
                key_len = r.read_varint()
                key = r.read(key_len)

                if key == probe_key:
                    r.read_varint()
                    r.read_varint()
                    r.read_varint()
                    r.read_varint()
                    r.read_varint()
                    packed_len = r.read_varint()
                    frame_counts.append(packed_len // 4)
                    break
                r.seek(entry_end)
            else:
                raise ValueError(f"{probe_key!r} not found at offset {pos}")

            pos = data_start + record_len + 4
    finally:
        os.close(fd)

    return offsets, frame_counts


def _parse_episode(
    path: str,
    offset: int,
    float_fields: Mapping[bytes, tuple],
    bytes_fields: Mapping[bytes, str],
    include_images: bool = True,
    instruction_key: str = "instruction",
):
    """Hand-rolled protobuf skip-parse for one TFDS-style episode record.

    The two field maps drive what gets extracted::

        float_fields: {tf_key_bytes: (out_name, dim_per_step)}
        bytes_fields: {tf_key_bytes: out_name}

    When ``include_images=False`` only the entry whose ``out_name`` matches
    ``instruction_key`` is kept; every other BytesList is seek-skipped at zero
    payload I/O. The entry tagged with ``instruction_key`` is returned as the
    decoded first item (single string), all others are returned as the raw
    bytes lists.
    """
    if include_images:
        bytes_targets = dict(bytes_fields)
    else:
        bytes_targets = {k: v for k, v in bytes_fields.items() if v == instruction_key}

    floats: Dict[str, tuple] = {}
    bytes_lists: Dict[str, list] = {}

    fd = os.open(path, os.O_RDONLY)
    try:
        r = _PreadReader(fd)
        r.seek(offset)
        record_len = r.read_u64le()
        r.read(4)
        data_start = offset + 12

        r.seek(data_start)
        r.read_varint()
        features_len = r.read_varint()
        features_end = r.pos + features_len

        n_want = len(float_fields) + len(bytes_targets)
        while r.pos < features_end and (len(floats) + len(bytes_lists)) < n_want:
            r.read_varint()
            entry_len = r.read_varint()
            entry_end = r.pos + entry_len

            r.read_varint()
            key_len = r.read_varint()
            key = r.read(key_len)

            ftarget = float_fields.get(key)
            btarget = bytes_targets.get(key)

            if ftarget is not None:
                name, dim = ftarget
                r.read_varint()
                r.read_varint()
                r.read_varint()
                r.read_varint()
                r.read_varint()
                packed_len = r.read_varint()
                raw = r.read(packed_len)
                floats[name] = (np.frombuffer(raw, dtype=np.float32).copy(), dim)
            elif btarget is not None:
                r.read_varint()
                r.read_varint()
                r.read_varint()
                list_len = r.read_varint()
                list_end = r.pos + list_len
                items = []
                while r.pos < list_end:
                    r.read_varint()
                    item_len = r.read_varint()
                    items.append(bytes(r.read(item_len)))
                bytes_lists[btarget] = items

            r.seek(entry_end)
    finally:
        os.close(fd)

    missing_f = {v[0] for v in float_fields.values()} - set(floats)
    missing_b = set(bytes_targets.values()) - set(bytes_lists)
    if missing_f or missing_b:
        raise ValueError(f"Missing fields at {path}:{offset}: floats={missing_f}, bytes={missing_b}")

    # Derive num_frames from any float field via its (array, dim) tuple.
    any_arr, any_dim = next(iter(floats.values()))
    n = len(any_arr) // any_dim

    out = {"num_frames": n}
    for k, (arr, d) in floats.items():
        out[k] = arr.reshape(n, d)
    for name, items in bytes_lists.items():
        if name == instruction_key:
            out[name] = items[0].decode("utf-8") if items else ""
        else:
            out[name] = items
    return out


def scan_rlds_episodes(
    data_path: str,
    probe_key: bytes,
    fps: float,
    robot_type: RobotType,
    desc: str,
    split: str = "train",
    max_workers: int = 64,
) -> List[EpisodeMetadata]:
    """Threaded scan of every ``*.tfrecord-*`` shard under ``data_path``,
    returning a flat ``List[EpisodeMetadata]`` ready for ``_load_metadata``.

    When a TFDS-style ``dataset_info.json`` is present, its declared shard
    lengths are cross-checked against the scanner's counts.
    """
    info_path = os.path.join(data_path, "dataset_info.json")
    if os.path.isfile(info_path):
        info = json.load(open(info_path))
        split_info = next((s for s in info["splits"] if s["name"] == split), None)
        assert split_info is not None, (
            f"split '{split}' not found in {info_path}, available: {[s['name'] for s in info['splits']]}"
        )
        shard_lengths = [int(x) for x in split_info["shardLengths"]]
    else:
        shard_lengths = None

    shard_files = sorted(glob(os.path.join(data_path, f"*-{split}.tfrecord-*")))
    if not shard_files:
        shard_files = sorted(glob(os.path.join(data_path, "*.tfrecord-*")))
    if shard_lengths is not None:
        assert len(shard_files) == len(shard_lengths), (
            f"shard count mismatch: info={len(shard_lengths)}, files={len(shard_files)}"
        )

    scan = functools.partial(_scan_shard, probe_key=probe_key)
    shard_results: List = [None] * len(shard_files)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_idx = {ex.submit(scan, p): i for i, p in enumerate(shard_files)}
        for fut in tqdm(as_completed(fut_to_idx), total=len(shard_files), desc=desc, dynamic_ncols=True):
            shard_results[fut_to_idx[fut]] = fut.result()

    if shard_lengths is not None:
        for i, (expected, (offsets, _)) in enumerate(zip(shard_lengths, shard_results)):
            assert expected == len(offsets), f"Shard {i}: expected {expected}, scanned {len(offsets)}"

    out: List[EpisodeMetadata] = []
    for path, (offsets, lengths) in zip(shard_files, shard_results):
        for off, n in zip(offsets, lengths):
            out.append(
                EpisodeMetadata(
                    length=n,
                    fps=fps,
                    robot_type=robot_type,
                    extras={"path": path, "offset": off},
                )
            )
    assert out, f"No episodes found under {data_path}"
    return out


# ── Dataset ──────────────────────────────────────────────────
@DATASET_REGISTRY.register()
class DroidDataset(BaseVLADataset):
    """Single-arm Franka; one TFRecord record = one episode; state/action carry
    6D cartesian + 7D joint + 1D gripper. Streaming-only — random-access loaders
    raise ``NotImplementedError`` because each episode lives behind a contiguous
    protobuf chunk that has to be parsed in one shot."""

    def _load_metadata(self) -> List[EpisodeMetadata]:
        index_path = os.path.join(self.data_path, INDEX_NAME)
        if not os.path.isfile(index_path):
            raise FileNotFoundError(
                f"{index_path} not found. Generate it with:\n  python -m {__name__} --root {self.data_path}"
            )
        out: List[EpisodeMetadata] = []
        with open(index_path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                e = json.loads(line)
                path = e["path"]
                if not os.path.isabs(path):
                    path = os.path.join(self.data_path, path)
                out.append(
                    EpisodeMetadata(
                        length=int(e["length"]),
                        fps=SOURCE_FPS,
                        robot_type=RobotType.FRANKA,
                        extras={"path": path, "offset": int(e["offset"])},
                    )
                )
        assert out, f"No episodes in {index_path}"
        return out

    def _load_action(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotAction:
        raise NotImplementedError("DroidDataset is streaming-only; use iter_episode / _iter_episode.")

    def _load_state(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotState:
        raise NotImplementedError("DroidDataset is streaming-only; use iter_episode / _iter_episode.")

    def _load_images(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("DroidDataset is streaming-only; use iter_episode / _iter_episode.")

    def _load_instruction(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> List[str]:
        raise NotImplementedError("DroidDataset is streaming-only; use iter_episode / _iter_episode.")

    def _iter_episode(
        self,
        episode_index: int,
        source_ranges: List[tuple],
        include_images: bool = True,
    ) -> Iterator[Dict]:
        meta = self._metadata[episode_index]
        ep = _parse_episode(
            meta.extras["path"],
            meta.extras["offset"],
            FLOAT_FIELDS,
            BYTES_FIELDS,
            include_images=include_images,
        )

        cart_s = torch.from_numpy(ep["cartesian_position"]).float()
        joint_s = torch.from_numpy(ep["joint_position"]).float()
        cart_a = torch.from_numpy(ep["action_cartesian_position"]).float()
        joint_a = torch.from_numpy(ep["action_joint_position"]).float()

        # DROID stores gripper as 0=open, 1=closed; flip to our convention (1=open, 0=closed).
        grip_s = 1.0 - torch.from_numpy(ep["gripper_position"]).float()
        grip_a = 1.0 - torch.from_numpy(ep["action_gripper_position"]).float()

        full_state = RobotState(
            left_arm=Arm(
                joint_position=Position(joint_s),
                eef_position=Position(cart_s[:, :3]),
                eef_rotation=EefRotation(
                    cart_s[:, 3:6],
                    representation=RotationRepresentation.ROT_VEC,
                ),
            ),
            left_gripper=Position(grip_s, allow_relative=False),
        )
        full_action = RobotAction(
            left_arm=Arm(
                joint_position=Position(joint_a),
                eef_position=Position(cart_a[:, :3]),
                eef_rotation=EefRotation(
                    cart_a[:, 3:6],
                    representation=RotationRepresentation.ROT_VEC,
                ),
            ),
            left_gripper=Position(grip_a, allow_relative=False),
        )

        instr = ep["instruction"]
        if include_images:
            wrist_blob = ep["wrist_images"]
            ext1_blob = ep["exterior_image_1"]
            ext2_blob = ep["exterior_image_2"]

        for start, end in source_ranges:
            images = None
            if include_images:
                images = {
                    "wrist": torch.from_numpy(np.asarray(Image.open(io.BytesIO(wrist_blob[start])))),
                    "exterior_1": torch.from_numpy(np.asarray(Image.open(io.BytesIO(ext1_blob[start])))),
                    "exterior_2": torch.from_numpy(np.asarray(Image.open(io.BytesIO(ext2_blob[start])))),
                }
            yield {
                "state": full_state[start : start + 1],
                "action": full_action[start:end],
                "instruction": instr,
                "images": images,
            }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Build DROID episodes.jsonl index (offline scan).",
    )
    ap.add_argument("--root", required=True, help="DROID data_path (directory containing *.tfrecord-* shards)")
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    episodes = scan_rlds_episodes(
        args.root,
        DROID_PROBE_KEY,
        SOURCE_FPS,
        RobotType.FRANKA,
        "Scanning DROID",
        max_workers=args.workers,
    )
    output = os.path.join(args.root, INDEX_NAME)
    with open(output, "w") as fout:
        for ep in episodes:
            fout.write(
                json.dumps(
                    {
                        "path": os.path.relpath(ep.extras["path"], args.root),
                        "offset": ep.extras["offset"],
                        "length": ep.length,
                    }
                )
                + "\n"
            )
    print(f"Wrote {len(episodes)} episodes -> {output}")
