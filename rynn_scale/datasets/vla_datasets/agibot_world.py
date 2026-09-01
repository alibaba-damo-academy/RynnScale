import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from typing import Dict, Iterator, List, Tuple

import h5py
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from ...constants import CACHE_DIR, RobotType, RotationRepresentation
from ...registry import DATASET_REGISTRY
from ...utils.robot import Arm, Position, RobotAction, RobotState, Rotation
from .base import BaseVLADataset, EpisodeMetadata
from .utils import SequentialVideoReader, VideoReader, fork_safe_cache, mt_process, suppress_hf_progress

# Camera keys produced by both AgiBot sources, so the two are interchangeable
# downstream (the processor only sorts and concatenates camera keys). For the
# 2026 LeRobot layout these double as the ``observation.images.<key>`` video
# suffix; for the Beta layout they map to mp4 basenames (see AGIBOT_BETA_CAMERAS).
AGIBOT2026_CAMERAS = ("top_head", "hand_left", "hand_right")

# AgiBot World Beta: output camera key -> mp4 basename under videos/.
AGIBOT_BETA_CAMERAS: Dict[str, str] = {
    "top_head": "head_color",
    "hand_left": "hand_left_color",
    "hand_right": "hand_right_color",
}

BETA_SOURCE_FPS = 30.0


@DATASET_REGISTRY.register()
class AgiBotWorld2026Dataset(BaseVLADataset):
    @staticmethod
    def _scan_task(task_path: str) -> Tuple[Dict, List[Tuple[int, int]]]:
        with open(os.path.join(task_path, "meta/info.json"), "r") as f:
            info = json.load(f)

        seg_lookup: Dict[int, List[Tuple[int, int, str]]] = {}
        for ep_key, segs in info.get("instruction_segments", {}).items():
            ep_idx = int(ep_key)
            seg_lookup[ep_idx] = [
                (int(s["start_frame_index"]), int(s["end_frame_index"]), s["instruction"])
                for s in segs
                if s["track"] == "default"
            ]

        task_meta = {
            "chunks_size": int(info["chunks_size"]),
            "data_path": info["data_path"],
            "video_path": info["video_path"],
            "offsets": AgiBotWorld2026Dataset._extract_offsets(info),
            "segments": seg_lookup,
            "fps": float(info.get("fps", 30.0)),
        }

        episodes: List[Tuple[int, int]] = []
        with open(os.path.join(task_path, "meta/episodes.jsonl"), "r") as f:
            for line in f:
                entry = json.loads(line)
                n = int(entry["length"])
                if n <= 0:
                    continue
                episodes.append((int(entry["episode_index"]), n))

        return task_meta, episodes

    @staticmethod
    def _extract_offsets(info: Dict) -> Dict:
        sf = info["features"]["observation.state"]["field_descriptions"]
        af = info["features"]["action"]["field_descriptions"]

        s_arm_orient = sf["state/end/arm_orientation"]["indices"]
        s_arm_pos = sf["state/end/arm_position"]["indices"]
        s_l_grip = sf["state/left_effector/position"]["indices"]
        s_r_grip = sf["state/right_effector/position"]["indices"]
        s_head = sf["state/head/position"]["indices"]
        s_waist = sf["state/waist/position"]["indices"]
        s_joint = sf["state/joint/position"]["indices"]
        a_pos = af["action/end/position"]["indices"]
        a_orient = af["action/end/orientation"]["indices"]
        a_l_grip = af["action/left_effector/position"]["indices"]
        a_r_grip = af["action/right_effector/position"]["indices"]
        a_head = af["action/head/position"]["indices"]
        a_waist = af["action/waist/position"]["indices"]
        a_joint = af["action/joint/position"]["indices"]

        return {
            "state": {
                "left_gripper": slice(s_l_grip[0], s_l_grip[-1] + 1),
                "right_gripper": slice(s_r_grip[0], s_r_grip[-1] + 1),
                "left_orient": slice(s_arm_orient[0], s_arm_orient[0] + 4),
                "right_orient": slice(s_arm_orient[0] + 4, s_arm_orient[0] + 8),
                "left_pos": slice(s_arm_pos[0], s_arm_pos[0] + 3),
                "right_pos": slice(s_arm_pos[0] + 3, s_arm_pos[0] + 6),
                "left_joint": slice(s_joint[0], s_joint[0] + 7),
                "right_joint": slice(s_joint[0] + 7, s_joint[0] + 14),
                "head": slice(s_head[0], s_head[-1] + 1),
                "waist": slice(s_waist[0], s_waist[-1] + 1),
            },
            "action": {
                "left_gripper": slice(a_l_grip[0], a_l_grip[-1] + 1),
                "right_gripper": slice(a_r_grip[0], a_r_grip[-1] + 1),
                "left_pos": slice(a_pos[0], a_pos[0] + 3),
                "right_pos": slice(a_pos[0] + 3, a_pos[0] + 6),
                "left_orient": slice(a_orient[0], a_orient[0] + 4),
                "right_orient": slice(a_orient[0] + 4, a_orient[0] + 8),
                "left_joint": slice(a_joint[0], a_joint[0] + 7),
                "right_joint": slice(a_joint[0] + 7, a_joint[0] + 14),
                "head": slice(a_head[0], a_head[-1] + 1),
                "waist": slice(a_waist[0], a_waist[-1] + 1),
            },
        }

    def _load_metadata(self) -> List[EpisodeMetadata]:
        task_paths = sorted(
            os.path.dirname(os.path.dirname(p))
            for p in glob(os.path.join(self.data_path, "**/meta/info.json"), recursive=True)
        )
        assert len(task_paths) > 0, f"No tasks found under {self.data_path}"

        with ThreadPoolExecutor(max_workers=32) as executor:
            scanned = list(
                tqdm(
                    executor.map(self._scan_task, task_paths),
                    total=len(task_paths),
                    desc="Loading episodes",
                )
            )

        out: List[EpisodeMetadata] = []
        for task_path, (task_meta, episodes) in zip(task_paths, scanned):
            offsets = task_meta["offsets"]
            seg_lookup = task_meta["segments"]
            for parquet_ep_index, n in episodes:
                episode_chunk = parquet_ep_index // task_meta["chunks_size"]
                parquet_path = os.path.join(
                    task_path,
                    task_meta["data_path"].format(
                        episode_chunk=episode_chunk,
                        episode_index=parquet_ep_index,
                    ),
                )
                video_tpl = task_meta["video_path"]
                out.append(
                    EpisodeMetadata(
                        length=n,
                        fps=float(task_meta["fps"]),
                        robot_type=RobotType.AGIBOT_G2,
                        extras={
                            "parquet_path": parquet_path,
                            "video_dir": task_path,
                            "video_tpl": video_tpl,
                            "episode_chunk": episode_chunk,
                            "parquet_ep_index": parquet_ep_index,
                            "offsets": offsets,
                            "segments": seg_lookup.get(parquet_ep_index, []),
                        },
                    )
                )
        return out

    def _video_path(self, extras: Dict, cam_name: str) -> str:
        return os.path.join(
            extras["video_dir"],
            extras["video_tpl"].format(
                episode_chunk=extras["episode_chunk"],
                video_key=f"observation.images.{cam_name}",
                episode_index=extras["parquet_ep_index"],
            ),
        )

    @fork_safe_cache
    def _get_parquet(self, parquet_path: str):
        with suppress_hf_progress():
            return load_dataset(
                "parquet",
                data_files=parquet_path,
                split="train",
            ).with_format("torch")

    @fork_safe_cache
    def _get_video_reader(self, video_path: str):
        if not os.path.exists(video_path):
            return None
        return VideoReader(video_path)

    @staticmethod
    def _lookup_instruction(segments: List[Tuple[int, int, str]], frame: int) -> str:
        for start, end, text in segments:
            if start <= frame < end:
                return text
        return ""

    # Gripper range in the raw data: -0.91 (fully closed) .. 0.0 (fully open).
    _GRIPPER_CLOSED = -0.91

    @staticmethod
    def _normalize_gripper(raw: torch.Tensor) -> torch.Tensor:
        return (-raw / (-AgiBotWorld2026Dataset._GRIPPER_CLOSED)).clamp(0.0, 1.0)

    @classmethod
    def _build_robot(cls, values: torch.Tensor, offsets: Dict, robot_cls):
        return robot_cls(
            left_arm=Arm(
                eef_position=Position(values[:, offsets["left_pos"]]),
                eef_rotation=Rotation(
                    values[:, offsets["left_orient"]],
                    representation=RotationRepresentation.QUAT_XYZW,
                ),
                joint_position=Position(values[:, offsets["left_joint"]]),
            ),
            right_arm=Arm(
                eef_position=Position(values[:, offsets["right_pos"]]),
                eef_rotation=Rotation(
                    values[:, offsets["right_orient"]],
                    representation=RotationRepresentation.QUAT_XYZW,
                ),
                joint_position=Position(values[:, offsets["right_joint"]]),
            ),
            left_gripper=Position(
                cls._normalize_gripper(values[:, offsets["left_gripper"]]),
                allow_relative=False,
            ),
            right_gripper=Position(
                cls._normalize_gripper(values[:, offsets["right_gripper"]]),
                allow_relative=False,
            ),
            head=Position(values[:, offsets["head"]]),
            torso=Position(values[:, offsets["waist"]]),
        )

    @staticmethod
    def _gather_column(ds, name: str, idx: List[int]) -> torch.Tensor:
        return torch.stack([ds[i][name] for i in idx], dim=0).float()

    def _load_action(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotAction:
        extras = self._metadata[episode_index].extras
        ds = self._get_parquet(extras["parquet_path"])
        actions = self._gather_column(ds, "action", frame_index)
        return self._build_robot(actions, extras["offsets"]["action"], RobotAction)

    def _load_state(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotState:
        extras = self._metadata[episode_index].extras
        ds = self._get_parquet(extras["parquet_path"])
        states = self._gather_column(ds, "observation.state", frame_index)
        return self._build_robot(states, extras["offsets"]["state"], RobotState)

    def _load_images(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> Dict[str, torch.Tensor]:
        extras = self._metadata[episode_index].extras
        images: Dict[str, torch.Tensor] = {}
        for cam_name in AGIBOT2026_CAMERAS:
            reader = self._get_video_reader(self._video_path(extras, cam_name))
            if reader is None:
                continue
            images[cam_name] = reader.read(frame_index)
        return images

    def _load_instruction(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> List[str]:
        segments = self._metadata[episode_index].extras["segments"]
        return [self._lookup_instruction(segments, i) for i in frame_index]

    def _iter_episode(
        self,
        episode_index: int,
        source_ranges: List[tuple],
        include_images: bool = True,
    ) -> Iterator[Dict]:
        meta = self._metadata[episode_index]
        extras = meta.extras
        n_total = meta.length
        offsets = extras["offsets"]
        segments = extras["segments"]

        with suppress_hf_progress():
            ds = load_dataset("parquet", data_files=extras["parquet_path"], split="train").with_format("torch")

        all_states = self._gather_column(ds, "observation.state", list(range(n_total)))
        all_actions = self._gather_column(ds, "action", list(range(n_total)))
        full_state = self._build_robot(all_states, offsets["state"], RobotState)
        full_action = self._build_robot(all_actions, offsets["action"], RobotAction)

        readers = {}
        if include_images:
            for cam_name in AGIBOT2026_CAMERAS:
                path = self._video_path(extras, cam_name)
                if os.path.exists(path):
                    readers[cam_name] = SequentialVideoReader(path)

        try:
            for start, end in source_ranges:
                images = None
                if include_images:
                    images = {k: r.read(start) for k, r in readers.items()}
                yield {
                    "state": full_state[start : start + 1],
                    "action": full_action[start:end],
                    "instruction": self._lookup_instruction(segments, start),
                    "images": images,
                }
        finally:
            for r in readers.values():
                r.close()


@DATASET_REGISTRY.register()
class AgiBotWorldBetaDataset(BaseVLADataset):
    def _load_metadata(self) -> List[EpisodeMetadata]:
        self._obs_root = os.path.join(self.data_path, "observations")
        assert os.path.isdir(self._obs_root), f"observations/ not found under {self.data_path}"

        self._proprio_root = os.path.join(self.data_path, "proprio_stats")
        self._task_info_root = os.path.join(self.data_path, "task_info")
        episodes = self._load_or_build_index()
        assert episodes, f"No episodes found under {self._task_info_root}"

        out: List[EpisodeMetadata] = []
        for i, ep in enumerate(episodes):
            out.append(
                EpisodeMetadata(
                    length=ep["length"],
                    fps=BETA_SOURCE_FPS,
                    robot_type=RobotType.AGIBOT_G1,
                    extras={
                        "task_id": ep["task_id"],
                        "episode_id": ep["episode_id"],
                        "segments": ep.get("segments", []),
                    },
                )
            )
        return out

    # ── episode index ───────────────────────────────────────
    def _index_cache_path(self) -> str:
        digest = hashlib.sha256(os.path.abspath(self._obs_root).encode()).hexdigest()[:16]
        return os.path.join(CACHE_DIR, "indexes", f"agibot_world_beta_v3_{digest}.json")

    def _load_or_build_index(self) -> List[Dict]:
        cache_path = self._index_cache_path()
        if os.path.isfile(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)

        task_files = sorted(glob(os.path.join(self._task_info_root, "task_*.json")))
        assert task_files, f"No task_*.json found under {self._task_info_root}"

        per_task = mt_process(
            self._scan_task,
            task_files,
            max_workers=96,
            desc="Indexing AgiBot World Beta from task_info",
        )
        episodes = [ep for entries in per_task for ep in entries]
        episodes.sort(key=lambda e: (e["task_id"], e["episode_id"]))

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(episodes, f)
        return episodes

    def _scan_task(self, task_file: str) -> List[Dict]:
        task_id = os.path.basename(task_file)[len("task_") : -len(".json")]
        obs_task_dir = os.path.join(self._obs_root, task_id)
        present = set(os.listdir(obs_task_dir)) if os.path.isdir(obs_task_dir) else set()

        with open(task_file, "r") as f:
            records = json.load(f)

        entries: List[Dict] = []
        for rec in records:
            episode_id = str(rec["episode_id"])
            if episode_id not in present:
                continue
            action_config = rec.get("label_info", {}).get("action_config", [])
            if not action_config:
                continue
            segments = [[int(s["start_frame"]), int(s["end_frame"]), s["action_text"]] for s in action_config]
            length = max(seg[1] for seg in segments)
            if length <= 0:
                continue
            entries.append(
                {
                    "task_id": task_id,
                    "episode_id": episode_id,
                    "length": length,
                    "segments": segments,
                }
            )
        return entries

    def _video_path(self, episode_index: int, mp4_name: str) -> str:
        extras = self._metadata[episode_index].extras
        return os.path.join(self._obs_root, extras["task_id"], extras["episode_id"], "videos", f"{mp4_name}.mp4")

    @fork_safe_cache
    def _get_video_reader(self, episode_index: int, mp4_name: str):
        path = self._video_path(episode_index, mp4_name)
        return VideoReader(path) if os.path.exists(path) else None

    def _h5_path(self, episode_index: int) -> str:
        extras = self._metadata[episode_index].extras
        return os.path.join(self._proprio_root, extras["task_id"], extras["episode_id"], "proprio_stats.h5")

    @fork_safe_cache
    def _open_hdf5(self, path: str):
        return h5py.File(path, "r", locking=False, rdcc_nbytes=0)

    @staticmethod
    def _t(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(arr)).float()

    _GRIP_STATE_CLOSED = 34.85
    _GRIP_STATE_OPEN = 120.0

    def _build_robot(self, f: h5py.File, idxs: List[int], prefix: str, cls):
        end_pos = f[f"{prefix}/end/position"][idxs]
        end_ori = f[f"{prefix}/end/orientation"][idxs]
        joint = f[f"{prefix}/joint/position"][idxs]
        grip = self._t(f[f"{prefix}/effector/position"][idxs])
        head = f[f"{prefix}/head/position"][idxs]
        waist = f[f"{prefix}/waist/position"][idxs]

        is_action = prefix == "action"
        if not is_action:
            grip = ((grip - self._GRIP_STATE_CLOSED) / (self._GRIP_STATE_OPEN - self._GRIP_STATE_CLOSED)).clamp(
                0.0, 1.0
            )

        def arm(i: int) -> Arm:
            return Arm(
                eef_position=Position(self._t(end_pos[:, i, :])),
                eef_rotation=Rotation(
                    self._t(end_ori[:, i, :]),
                    representation=RotationRepresentation.QUAT_XYZW,
                ),
                joint_position=Position(self._t(joint[:, i * 7 : (i + 1) * 7])),
            )

        return cls(
            left_arm=arm(0),
            right_arm=arm(1),
            left_gripper=Position(grip[:, 0:1], allow_relative=False),
            right_gripper=Position(grip[:, 1:2], allow_relative=False),
            head=Position(self._t(head)),
            torso=Position(self._t(waist)),
        )

    @staticmethod
    def _lookup_instruction(segments: List, frame: int) -> str:
        for start, end, text in segments:
            if start <= frame < end:
                return text
        return ""

    def _load_action(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotAction:
        f = self._open_hdf5(self._h5_path(episode_index))
        return self._build_robot(f, frame_index, "action", RobotAction)

    def _load_state(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotState:
        f = self._open_hdf5(self._h5_path(episode_index))
        return self._build_robot(f, frame_index, "state", RobotState)

    def _load_images(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> Dict[str, torch.Tensor]:
        images: Dict[str, torch.Tensor] = {}
        for cam_key, mp4_name in AGIBOT_BETA_CAMERAS.items():
            reader = self._get_video_reader(episode_index, mp4_name)
            if reader is None:
                continue
            images[cam_key] = reader.read(frame_index)
        return images

    def _load_instruction(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> List[str]:
        segments = self._metadata[episode_index].extras.get("segments", [])
        return [self._lookup_instruction(segments, i) for i in frame_index]

    def _iter_episode(
        self,
        episode_index: int,
        source_ranges: List[tuple],
        include_images: bool = True,
    ) -> Iterator[Dict]:
        meta = self._metadata[episode_index]
        extras = meta.extras
        n_total = meta.length

        h5_path = self._h5_path(episode_index)
        with h5py.File(h5_path, "r", locking=False, rdcc_nbytes=0) as f:
            all_idx = list(range(n_total))
            full_state = self._build_robot(f, all_idx, "state", RobotState)
            full_action = self._build_robot(f, all_idx, "action", RobotAction)

        segments = extras.get("segments", [])

        readers = {}
        if include_images:
            for cam_key, mp4_name in AGIBOT_BETA_CAMERAS.items():
                path = self._video_path(episode_index, mp4_name)
                if os.path.exists(path):
                    readers[cam_key] = SequentialVideoReader(path)

        try:
            for start, end in source_ranges:
                images = None
                if include_images:
                    images = {k: r.read(start) for k, r in readers.items()}
                yield {
                    "state": full_state[start : start + 1],
                    "action": full_action[start:end],
                    "instruction": self._lookup_instruction(segments, start),
                    "images": images,
                }
        finally:
            for r in readers.values():
                r.close()
