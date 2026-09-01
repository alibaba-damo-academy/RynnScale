import json
import os
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from scipy.spatial.transform import Rotation as ScipyRotation
from tqdm import tqdm

from ...constants import RobotType, RotationRepresentation
from ...registry import DATASET_REGISTRY
from ...utils.robot import Arm, Position, RobotAction, RobotState, Rotation
from .base import BaseVLADataset, EpisodeMetadata
from .interndata_a1 import _X_AXIS, _Y_AXIS, _Z_AXIS, _serial_fk
from .utils import SequentialVideoReader, VideoReader, fork_safe_cache, suppress_hf_progress

_CAMERAS: Sequence[Tuple[str, str]] = (
    ("head", "observation.images.rgb.head"),
    ("hand_left", "observation.images.rgb.left_wrist"),
    ("hand_right", "observation.images.rgb.right_wrist"),
)

# Action (23-dim) layout for R1Pro – follows OmniGibson raw_controller_order:
#   [0:3]   base velocity
#   [3:5]   torso (2 joints: torso_joint1..2)
#   [5:7]   head (2 joints: head_joint1..2)
#   [7:14]  left arm joint targets (7 joints: left_arm_joint1..7)
#   [14]    left gripper binary {-1, 1}
#   [15:22] right arm joint targets (7 joints: right_arm_joint1..7)
#   [22]    right gripper binary {-1, 1}
_ACTION_TORSO = slice(3, 5)
_ACTION_HEAD = slice(5, 7)
_ACTION_LEFT_ARM = slice(7, 14)
_ACTION_LEFT_GRIPPER = slice(14, 15)
_ACTION_RIGHT_ARM = slice(15, 22)
_ACTION_RIGHT_GRIPPER = slice(22, 23)

# State (256-dim) – positions interleaved with velocities for arm joints:
_STATE_TORSO = slice(6, 8)
_STATE_HEAD = slice(8, 10)
_STATE_LEFT_ARM = [10, 12, 14, 16, 18, 20, 22]
_STATE_RIGHT_ARM = slice(197, 204)
_STATE_LEFT_GRIPPER = slice(24, 25)
_STATE_RIGHT_GRIPPER = slice(193, 194)


def _normalize_gripper_binary(g: torch.Tensor) -> torch.Tensor:
    """Map gripper binary {-1, 1} -> [0, 1] (0=close, 1=open)."""
    return (g + 1.0) * 0.5


_GRIPPER_STATE_MAX = 0.05


def _normalize_gripper_state(g: torch.Tensor) -> torch.Tensor:
    """Map gripper joint position [0, 0.05] -> [0, 1] (0=close, 1=open)."""
    return (g / _GRIPPER_STATE_MAX).clamp(0.0, 1.0)


# ── R1Pro 7-DOF arm FK chains (in torso_link4 frame) ──────────────────────
# Derived from the R1 Pro URDF.
# All joint origins have rpy="0 0 0", so fixed rotations are identity.
# Joint axes: Y, X, Z, Y, Z, Y, X for both arms.

_I3 = np.eye(3, dtype=np.float64)

_R1PRO_LEFT_ARM_CHAIN: Tuple[Tuple[np.ndarray, np.ndarray], ...] = (
    # torso_link4 -> left_arm_base_link (fixed, xyz="-0.00048618 0.097234 0.30302")
    #             -> left_arm_link1 (joint1, xyz="0 0.0735 0")
    (np.array([-0.00048618, 0.170734, 0.30302], dtype=np.float64), _I3),
    (np.array([0.025012, 0.081265, 0.0], dtype=np.float64), _I3),
    (np.array([-0.025012, 0.0, -0.1155], dtype=np.float64), _I3),
    (np.array([0.0, 0.035065, -0.1945], dtype=np.float64), _I3),
    (np.array([0.0, -0.035065, -0.095], dtype=np.float64), _I3),
    (np.array([0.0, -0.028, -0.16305], dtype=np.float64), _I3),
    (np.array([0.0295, 0.028, 0.0], dtype=np.float64), _I3),
)

_R1PRO_RIGHT_ARM_CHAIN: Tuple[Tuple[np.ndarray, np.ndarray], ...] = (
    # torso_link4 -> right_arm_base_link (fixed, xyz="-0.00048706 -0.097236 0.30302")
    #             -> right_arm_link1 (joint1, xyz="0 -0.0735 0")
    (np.array([-0.00048706, -0.170736, 0.30302], dtype=np.float64), _I3),
    (np.array([0.023988, -0.081265, 0.0], dtype=np.float64), _I3),
    (np.array([-0.023988, 0.0, -0.1155], dtype=np.float64), _I3),
    (np.array([0.0, 0.035058, -0.1945], dtype=np.float64), _I3),
    (np.array([0.0, -0.035058, -0.095], dtype=np.float64), _I3),
    (np.array([0.0, 0.028001, -0.16305], dtype=np.float64), _I3),
    (np.array([0.0295, -0.028001, 0.0], dtype=np.float64), _I3),
)

_R1PRO_ARM_AXES = (_Y_AXIS, _X_AXIS, _Z_AXIS, _Y_AXIS, _Z_AXIS, _Y_AXIS, _X_AXIS)


def _r1pro_arm_fk(
    joints: np.ndarray,
    chain: Tuple[Tuple[np.ndarray, np.ndarray], ...],
) -> Tuple[np.ndarray, np.ndarray]:
    return _serial_fk(joints.astype(np.float64), chain, joint_axes=_R1PRO_ARM_AXES)


def _arm_with_eef(joint_tensor: torch.Tensor, chain) -> Arm:
    joints_np = joint_tensor.numpy().astype(np.float64)
    pos, rot_mat = _r1pro_arm_fk(joints_np, chain)
    eef_rpy = ScipyRotation.from_matrix(rot_mat).as_euler("xyz").astype(np.float32)
    return Arm(
        joint_position=Position(joint_tensor),
        eef_position=Position(torch.from_numpy(pos.astype(np.float32))),
        eef_rotation=Rotation(
            torch.from_numpy(eef_rpy),
            representation=RotationRepresentation.EULER_XYZ,
        ),
    )


@DATASET_REGISTRY.register()
class Behavior1KDataset(BaseVLADataset):
    def _load_metadata(self) -> List[EpisodeMetadata]:
        info_path = os.path.join(self.data_path, "meta", "info.json")
        with open(info_path, "r") as f:
            info = json.load(f)

        chunks_size = int(info["chunks_size"])
        data_path_tpl = info["data_path"]
        video_path_tpl = info["video_path"]
        fps = float(info.get("fps", 30.0))

        tasks_map: Dict[int, str] = {}
        with open(os.path.join(self.data_path, "meta", "tasks.jsonl"), "r") as f:
            for line in f:
                entry = json.loads(line)
                tasks_map[int(entry["task_index"])] = entry["task"]
        n_tasks = max(tasks_map) + 1 if tasks_map else 0
        task_texts = [tasks_map.get(i, "") for i in range(n_tasks)]

        out: List[EpisodeMetadata] = []
        with open(os.path.join(self.data_path, "meta", "episodes.jsonl"), "r") as f:
            for line in tqdm(f, desc="Scanning BEHAVIOR-1K"):
                entry = json.loads(line)
                n = int(entry["length"])
                if n <= 0:
                    continue
                ep_idx = int(entry["episode_index"])
                episode_chunk = ep_idx // chunks_size
                parquet_path = os.path.join(
                    self.data_path,
                    data_path_tpl.format(
                        episode_chunk=episode_chunk,
                        episode_index=ep_idx,
                    ),
                )
                out.append(
                    EpisodeMetadata(
                        length=n,
                        fps=fps,
                        robot_type=RobotType.GALAXEA_R1_PRO,
                        extras={
                            "parquet_path": parquet_path,
                            "video_tpl": video_path_tpl,
                            "episode_chunk": episode_chunk,
                            "parquet_ep_index": ep_idx,
                            "task_texts": task_texts,
                        },
                    )
                )
        return out

    # ── helpers ────────────────────────────────────────────────

    def _video_path(self, extras: Dict, video_key: str) -> str:
        return os.path.join(
            self.data_path,
            extras["video_tpl"].format(
                episode_chunk=extras["episode_chunk"],
                video_key=video_key,
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
    def _bulk_column(ds, name: str) -> torch.Tensor:
        col = ds[:][name]
        if isinstance(col, list):
            col = torch.tensor(col)
        return col.float()

    @staticmethod
    def _make_action(action_col: torch.Tensor, idx: List[int]) -> RobotAction:
        a = action_col[idx]
        return RobotAction(
            left_arm=_arm_with_eef(a[:, _ACTION_LEFT_ARM], _R1PRO_LEFT_ARM_CHAIN),
            right_arm=_arm_with_eef(a[:, _ACTION_RIGHT_ARM], _R1PRO_RIGHT_ARM_CHAIN),
            left_gripper=Position(
                _normalize_gripper_binary(a[:, _ACTION_LEFT_GRIPPER]),
                allow_relative=False,
            ),
            right_gripper=Position(
                _normalize_gripper_binary(a[:, _ACTION_RIGHT_GRIPPER]),
                allow_relative=False,
            ),
            torso=Position(a[:, _ACTION_TORSO]),
            head=Position(a[:, _ACTION_HEAD]),
        )

    @staticmethod
    def _make_state(state_col: torch.Tensor, idx: List[int]) -> RobotState:
        s = state_col[idx]
        return RobotState(
            left_arm=_arm_with_eef(s[:, _STATE_LEFT_ARM], _R1PRO_LEFT_ARM_CHAIN),
            right_arm=_arm_with_eef(s[:, _STATE_RIGHT_ARM], _R1PRO_RIGHT_ARM_CHAIN),
            left_gripper=Position(
                _normalize_gripper_state(s[:, _STATE_LEFT_GRIPPER]),
                allow_relative=False,
            ),
            right_gripper=Position(
                _normalize_gripper_state(s[:, _STATE_RIGHT_GRIPPER]),
                allow_relative=False,
            ),
            torso=Position(s[:, _STATE_TORSO]),
            head=Position(s[:, _STATE_HEAD]),
        )

    # ── per-frame loaders ──────────────────────────────────────

    def _load_action(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotAction:
        extras = self._metadata[episode_index].extras
        ds = self._get_parquet(extras["parquet_path"])
        action_col = self._bulk_column(ds, "action")
        return self._make_action(action_col, frame_index)

    def _load_state(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotState:
        extras = self._metadata[episode_index].extras
        ds = self._get_parquet(extras["parquet_path"])
        state_col = self._bulk_column(ds, "observation.state")
        return self._make_state(state_col, frame_index)

    def _load_images(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> Dict[str, torch.Tensor]:
        extras = self._metadata[episode_index].extras
        images: Dict[str, torch.Tensor] = {}
        for out_name, video_key in _CAMERAS:
            reader = self._get_video_reader(self._video_path(extras, video_key))
            if reader is None:
                continue
            images[out_name] = reader.read(frame_index)
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

    # ── _iter_episode ──────────────────────────────────────────

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
            ds = load_dataset(
                "parquet",
                data_files=extras["parquet_path"],
                split="train",
            ).with_format("torch")

        all_idx = list(range(n_total))
        state_col = self._bulk_column(ds, "observation.state")
        action_col = self._bulk_column(ds, "action")
        full_state = self._make_state(state_col, all_idx)
        full_action = self._make_action(action_col, all_idx)

        ti_col = ds[:]["task_index"]
        all_task_indices = ti_col.tolist() if not isinstance(ti_col, list) else ti_col

        readers: Dict[str, SequentialVideoReader] = {}
        if include_images:
            for out_name, video_key in _CAMERAS:
                path = self._video_path(extras, video_key)
                if os.path.exists(path):
                    readers[out_name] = SequentialVideoReader(path)

        try:
            for start, end in source_ranges:
                images = None
                if include_images:
                    images = {k: r.read(start) for k, r in readers.items()}
                yield {
                    "state": full_state[start : start + 1],
                    "action": full_action[start:end],
                    "instruction": task_texts[int(all_task_indices[start])],
                    "images": images,
                }
        finally:
            for r in readers.values():
                r.close()
