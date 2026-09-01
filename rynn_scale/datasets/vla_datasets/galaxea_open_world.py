import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from scipy.spatial.transform import Rotation as ScipyRotation
from tqdm import tqdm

from ...constants import RobotType, RotationRepresentation
from ...registry import DATASET_REGISTRY
from ...utils.robot import Arm, Position, RobotAction, RobotState, Rotation
from .base import BaseVLADataset, EpisodeMetadata
from .utils import SequentialVideoReader, VideoReader, fork_safe_cache, suppress_hf_progress

CAMERA_NAMES = ("head_rgb", "head_right_rgb", "left_wrist_rgb", "right_wrist_rgb")


_ARM_BASE_OFFSET_LEFT = np.array([0.0, 0.335, 0.12306], dtype=np.float64)
_ARM_BASE_OFFSET_RIGHT = np.array([0.0, -0.335, 0.12306], dtype=np.float64)
_ARM_CHAIN: Tuple[Tuple[np.ndarray, np.ndarray], ...] = (
    (np.array([0.0, 0.0, 0.08605], dtype=np.float64), np.array([0.0, 0.0, 1.0], dtype=np.float64)),
    (np.array([0.0, 0.03075, 0.04925], dtype=np.float64), np.array([0.0, 1.0, 0.0], dtype=np.float64)),
    (np.array([-0.3, 0.00025004, 0.0], dtype=np.float64), np.array([0.0, 1.0, 0.0], dtype=np.float64)),
    (np.array([0.1747, 0.00049739, 0.075485], dtype=np.float64), np.array([0.0, 1.0, 0.0], dtype=np.float64)),
    (np.array([0.08, -0.031498, 0.0405], dtype=np.float64), np.array([0.0, 0.0, 1.0], dtype=np.float64)),
    (np.array([0.022503, 0.0, -0.0405], dtype=np.float64), np.array([1.0, 0.0, 0.0], dtype=np.float64)),
)


def _arm_fk(joints: np.ndarray, base_offset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = joints.shape[0]
    pos = np.broadcast_to(base_offset, (n, 3)).copy()
    rot = np.broadcast_to(np.eye(3, dtype=np.float64), (n, 3, 3)).copy()
    for i, (offset, axis) in enumerate(_ARM_CHAIN):
        pos = pos + np.einsum("nij,j->ni", rot, offset)
        r_joint = ScipyRotation.from_rotvec(joints[:, i : i + 1] * axis[None, :]).as_matrix()
        rot = rot @ r_joint
    quat_xyzw = ScipyRotation.from_matrix(rot).as_quat(scalar_first=False)
    return pos, quat_xyzw


@DATASET_REGISTRY.register()
class GalaxeaOpenWorldDataset(BaseVLADataset):
    @staticmethod
    def _scan_task(task_path: str) -> Tuple[Dict, List[str], List[Tuple[int, int]]]:
        with open(os.path.join(task_path, "meta/info.json"), "r") as f:
            info = json.load(f)

        if info.get("robot_type") != "r1lite":
            return None, [], []

        task_meta = {
            "chunks_size": int(info["chunks_size"]),
            "data_path": info["data_path"],
            "video_path": info["video_path"],
            "fps": float(info.get("fps", 30.0)),
        }

        tasks_map: Dict[int, str] = {}
        with open(os.path.join(task_path, "meta", "tasks.jsonl"), "r") as f:
            for line in f:
                entry = json.loads(line)
                text = entry["task"]
                if "@" in text:
                    text = text.split("@", 1)[1]
                tasks_map[int(entry["task_index"])] = text
        n_tasks = max(tasks_map) + 1 if tasks_map else 0
        task_texts = [tasks_map.get(i, "") for i in range(n_tasks)]

        episodes: List[Tuple[int, int]] = []
        with open(os.path.join(task_path, "meta", "episodes.jsonl"), "r") as f:
            for line in f:
                entry = json.loads(line)
                n = int(entry["length"])
                if n <= 0:
                    continue
                episodes.append((int(entry["episode_index"]), n))

        return task_meta, task_texts, episodes

    def _load_metadata(self) -> List[EpisodeMetadata]:
        outer_dirs = sorted(d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d)))
        task_paths: List[str] = []
        for d in outer_dirs:
            inner = os.path.join(self.data_path, d)
            if os.path.isdir(os.path.join(inner, "meta")):
                task_paths.append(inner)
            elif os.path.isdir(os.path.join(inner, d, "meta")):
                task_paths.append(os.path.join(inner, d))
        assert len(task_paths) > 0, f"No tasks found under {self.data_path}"

        max_workers = min(32, max(1, len(task_paths)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            scanned = list(
                tqdm(
                    executor.map(self._scan_task, task_paths),
                    total=len(task_paths),
                    desc="Loading episodes",
                )
            )

        out: List[EpisodeMetadata] = []
        for task_path, (task_meta, task_texts, episodes) in zip(task_paths, scanned):
            if task_meta is None:
                continue
            for parquet_ep_index, n in episodes:
                episode_chunk = parquet_ep_index // task_meta["chunks_size"]
                parquet_path = os.path.join(
                    task_path,
                    task_meta["data_path"].format(
                        episode_chunk=episode_chunk,
                        episode_index=parquet_ep_index,
                    ),
                )
                out.append(
                    EpisodeMetadata(
                        length=n,
                        fps=float(task_meta["fps"]),
                        robot_type=RobotType.GALAXEA_R1_LITE,
                        extras={
                            "parquet_path": parquet_path,
                            "video_dir": task_path,
                            "video_tpl": task_meta["video_path"],
                            "episode_chunk": episode_chunk,
                            "parquet_ep_index": parquet_ep_index,
                            "task_texts": task_texts,
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
    def _gather_column(ds, name: str, idx: List[int]) -> torch.Tensor:
        out = torch.stack([ds[i][name] for i in idx], dim=0).float()
        if out.ndim == 1:
            out = out.unsqueeze(-1)
        return out

    def _build_action(self, ds, idx: List[int]) -> RobotAction:
        left_joint = self._gather_column(ds, "action.left_arm", idx)
        right_joint = self._gather_column(ds, "action.right_arm", idx)
        left_grip = self._gather_column(ds, "action.left_gripper", idx) / 100.0
        right_grip = self._gather_column(ds, "action.right_gripper", idx) / 100.0

        left_pos, left_quat = _arm_fk(
            left_joint.numpy().astype(np.float64),
            _ARM_BASE_OFFSET_LEFT,
        )
        right_pos, right_quat = _arm_fk(
            right_joint.numpy().astype(np.float64),
            _ARM_BASE_OFFSET_RIGHT,
        )

        torso = self._gather_column(ds, "observation.state.torso", idx)[:, :3]

        return RobotAction(
            left_arm=Arm(
                joint_position=Position(left_joint),
                eef_position=Position(torch.from_numpy(left_pos).float()),
                eef_rotation=Rotation(
                    torch.from_numpy(left_quat).float(),
                    representation=RotationRepresentation.QUAT_XYZW,
                ),
            ),
            right_arm=Arm(
                joint_position=Position(right_joint),
                eef_position=Position(torch.from_numpy(right_pos).float()),
                eef_rotation=Rotation(
                    torch.from_numpy(right_quat).float(),
                    representation=RotationRepresentation.QUAT_XYZW,
                ),
            ),
            left_gripper=Position(left_grip, allow_relative=False),
            right_gripper=Position(right_grip, allow_relative=False),
            torso=Position(torso),
        )

    def _build_state(self, ds, idx: List[int]) -> RobotState:
        left_joint = self._gather_column(ds, "observation.state.left_arm", idx)
        right_joint = self._gather_column(ds, "observation.state.right_arm", idx)
        left_grip = self._gather_column(ds, "observation.state.left_gripper", idx) / 100.0
        right_grip = self._gather_column(ds, "observation.state.right_gripper", idx) / 100.0
        left_eef = self._gather_column(ds, "observation.state.left_ee_pose", idx)
        right_eef = self._gather_column(ds, "observation.state.right_ee_pose", idx)
        torso = self._gather_column(ds, "observation.state.torso", idx)[:, :3]
        return RobotState(
            left_arm=Arm(
                joint_position=Position(left_joint),
                eef_position=Position(left_eef[:, :3]),
                eef_rotation=Rotation(left_eef[:, 3:7], representation=RotationRepresentation.QUAT_XYZW),
            ),
            right_arm=Arm(
                joint_position=Position(right_joint),
                eef_position=Position(right_eef[:, :3]),
                eef_rotation=Rotation(right_eef[:, 3:7], representation=RotationRepresentation.QUAT_XYZW),
            ),
            left_gripper=Position(left_grip, allow_relative=False),
            right_gripper=Position(right_grip, allow_relative=False),
            torso=Position(torso),
        )

    def _load_action(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotAction:
        extras = self._metadata[episode_index].extras
        ds = self._get_parquet(extras["parquet_path"])
        return self._build_action(ds, frame_index)

    def _load_state(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotState:
        extras = self._metadata[episode_index].extras
        ds = self._get_parquet(extras["parquet_path"])
        return self._build_state(ds, frame_index)

    def _load_images(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> Dict[str, torch.Tensor]:
        extras = self._metadata[episode_index].extras
        images: Dict[str, torch.Tensor] = {}
        for cam_name in CAMERA_NAMES:
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
        extras = self._metadata[episode_index].extras
        ds = self._get_parquet(extras["parquet_path"])
        task_texts = extras["task_texts"]
        return [task_texts[int(ds[i]["task_index"])] for i in frame_index]

    def _iter_episode(
        self,
        episode_index: int,
        source_ranges: List[tuple],
        include_images: bool = True,
    ) -> Iterator[Dict]:
        meta = self._metadata[episode_index]
        extras = meta.extras
        n_total = meta.length
        task_texts = extras["task_texts"]

        with suppress_hf_progress():
            ds = load_dataset("parquet", data_files=extras["parquet_path"], split="train").with_format("torch")

        all_idx = list(range(n_total))
        full_action = self._build_action(ds, all_idx)
        full_state = self._build_state(ds, all_idx)

        readers = {}
        if include_images:
            for cam_name in CAMERA_NAMES:
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
                    "instruction": task_texts[int(ds[start]["task_index"])],
                    "images": images,
                }
        finally:
            for r in readers.values():
                r.close()
