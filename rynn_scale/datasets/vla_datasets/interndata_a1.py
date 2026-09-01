import json
import os
from concurrent.futures import ThreadPoolExecutor
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
from .utils import SequentialVideoReader, VideoReader, fork_safe_cache, suppress_hf_progress

_X_AXIS, _Y_AXIS, _Z_AXIS = 0, 1, 2
_NEG_Y_AXIS = 3
_NEG_Z_AXIS = 4


def _serial_fk(
    joints: np.ndarray,
    chain: Tuple[Tuple[np.ndarray, np.ndarray], ...],
    flange_offset: np.ndarray | None = None,
    joint_axes: Tuple[int, ...] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """FK for an N-DOF serial chain, vectorized over batch.

    Returns:
        pos: (N, 3) end-effector position.
        rot: (N, 3, 3) end-effector orientation.
    """
    n = joints.shape[0]
    pos = np.zeros((n, 3), dtype=np.float64)
    rot = np.broadcast_to(np.eye(3, dtype=np.float64), (n, 3, 3)).copy()

    for i, (offset, R_fixed) in enumerate(chain):
        pos += np.einsum("nij,j->ni", rot, offset)
        rot = rot @ R_fixed
        c = np.cos(joints[:, i])
        s = np.sin(joints[:, i])
        Rj = np.zeros((n, 3, 3), dtype=np.float64)
        ax = _Z_AXIS if joint_axes is None else joint_axes[i]
        if ax == _Z_AXIS:
            Rj[:, 0, 0] = c
            Rj[:, 0, 1] = -s
            Rj[:, 1, 0] = s
            Rj[:, 1, 1] = c
            Rj[:, 2, 2] = 1.0
        elif ax == _NEG_Z_AXIS:
            Rj[:, 0, 0] = c
            Rj[:, 0, 1] = s
            Rj[:, 1, 0] = -s
            Rj[:, 1, 1] = c
            Rj[:, 2, 2] = 1.0
        elif ax == _Y_AXIS:
            Rj[:, 0, 0] = c
            Rj[:, 0, 2] = s
            Rj[:, 1, 1] = 1.0
            Rj[:, 2, 0] = -s
            Rj[:, 2, 2] = c
        elif ax == _NEG_Y_AXIS:
            Rj[:, 0, 0] = c
            Rj[:, 0, 2] = -s
            Rj[:, 1, 1] = 1.0
            Rj[:, 2, 0] = s
            Rj[:, 2, 2] = c
        else:  # _X_AXIS
            Rj[:, 0, 0] = 1.0
            Rj[:, 1, 1] = c
            Rj[:, 1, 2] = -s
            Rj[:, 2, 1] = s
            Rj[:, 2, 2] = c
        rot = rot @ Rj

    if flange_offset is not None:
        pos += np.einsum("nij,j->ni", rot, flange_offset)
    return pos, rot


# ── AgileX Piper 6-DOF FK chain ────────────────────────────────────────────
# Derived from the AgileX Piper URDF.  Each tuple is
# (translation, fixed_rotation) from the parent joint frame to the child
# joint frame *before* the child's revolute rotation is applied.  All 6
# joints rotate about local Z.


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    return ScipyRotation.from_euler("xyz", [r, p, y]).as_matrix()


_PIPER_CHAIN: Tuple[Tuple[np.ndarray, np.ndarray], ...] = (
    # base_link -> joint1 frame
    (np.array([0.0, 0.0, 0.123], dtype=np.float64), _rpy(0, 0, -1.5708)),
    # link1 -> joint2 frame
    (np.array([0.0, 0.0, 0.0], dtype=np.float64), _rpy(1.5708, 0, -1.5708)),
    # link2 -> joint3 frame
    (np.array([0.28358, 0.028726, 0.0], dtype=np.float64), _rpy(0, 0, 0.10095)),
    # link3 -> joint4 frame
    (np.array([-0.24221, 0.068514, 0.0], dtype=np.float64), _rpy(-1.5708, 0, 1.3826)),
    # link4 -> joint5 frame
    (np.array([0.0, 0.0, 0.0], dtype=np.float64), _rpy(1.5708, 0, 0)),
    # link5 -> joint6 frame
    (np.array([0.0, 0.091, 0.0014165], dtype=np.float64), _rpy(-1.5708, -np.pi, 0)),
)
_PIPER_FLANGE_OFFSET = np.array([0.0, 0.0, 0.13503], dtype=np.float64)


def _piper_fk(joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Piper FK. joints: (N, 6) -> (pos (N,3), rot_mat (N,3,3))."""
    return _serial_fk(joints.astype(np.float64), _PIPER_CHAIN, _PIPER_FLANGE_OFFSET)


# ── ARX LIFT2 6-DOF FK chain ───────────────────────────────────────────────
# Derived from official ARX_Lift2 URDF.  Joint axes: Z, +Y, -Y, -Y, -Z, X.

_I3 = np.eye(3, dtype=np.float64)

_ARX_LIFT2_CHAIN: Tuple[Tuple[np.ndarray, np.ndarray], ...] = (
    # base_link -> link1 (joint1, Z)
    (np.array([0.0, 0.0, 0.0565], dtype=np.float64), _I3),
    # link1 -> link2 (joint2, +Y)
    (np.array([0.02, 0.0, 0.047], dtype=np.float64), _I3),
    # link2 -> link3 (joint3, -Y)
    (np.array([-0.264, 0.0, 0.0], dtype=np.float64), _I3),
    # link3 -> link4 (joint4, -Y)
    (np.array([0.245, 0.0, 0.06], dtype=np.float64), _I3),
    # link4 -> link5 (joint5, -Z)
    (np.array([0.068, 0.0, 0.085], dtype=np.float64), _I3),
    # link5 -> link6 (joint6, X)
    (np.array([0.029, 0.0, -0.085], dtype=np.float64), _I3),
)
_ARX_LIFT2_JOINT_AXES = (_Z_AXIS, _Y_AXIS, _NEG_Y_AXIS, _NEG_Y_AXIS, _NEG_Z_AXIS, _X_AXIS)
_ARX_LIFT2_FLANGE_OFFSET = np.array([0.087, 0.0, 0.0], dtype=np.float64)


def _arx_lift2_fk(joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ARX LIFT2 FK. joints: (N, 6) -> (pos (N,3), rot_mat (N,3,3))."""
    return _serial_fk(
        joints.astype(np.float64),
        _ARX_LIFT2_CHAIN,
        _ARX_LIFT2_FLANGE_OFFSET,
        joint_axes=_ARX_LIFT2_JOINT_AXES,
    )


# ── AgiBot G1 7-DOF arm FK ─────────────────────────────────────────────────
# Derived from assets/agibot_g1/g1.xml.  The G1 has a 2-DOF waist (prismatic
# lift + revolute pitch) feeding into mirrored 7-DOF arms (all revolute Z).
# The FK returns EEF (arm_l/r_end_link) position/rotation in the fixed
# base_link frame.  The waist stream in the dataset is [pitch, lift_height].


def _quat_to_mat(w: float, x: float, y: float, z: float) -> np.ndarray:
    return ScipyRotation.from_quat([x, y, z, w]).as_matrix()


_G1_SLIDE_OFFSET = 0.30

_G1_R_BODY2 = _quat_to_mat(0.70710678, -0.70710678, 0, 0)
_G1_R_ARM_BASE = _quat_to_mat(0.70710678, 0.70710678, 0, 0)
_G1_R_ARM_L_BASE = _quat_to_mat(0.70710678, -0.70710678, 0, 0)
_G1_R_ARM_R_BASE = _quat_to_mat(0, 0, -0.70710678, 0.70710678)

_G1_ARM_L_CHAIN: Tuple[Tuple[np.ndarray, np.ndarray], ...] = (
    (np.array([0.0, 0.0, 0.1859], dtype=np.float64), np.eye(3, dtype=np.float64)),
    (np.array([0.0, 0.0, 0.0], dtype=np.float64), _quat_to_mat(0.5, -0.5, -0.5, 0.5)),
    (np.array([0.0, -0.305, 0.0], dtype=np.float64), _quat_to_mat(-0.5, -0.5, -0.5, 0.5)),
    (np.array([0.0, 0.0, 0.0], dtype=np.float64), _quat_to_mat(0, 0, -0.70710678, 0.70710678)),
    (np.array([0.0, -0.1975, 0.0], dtype=np.float64), _quat_to_mat(0, 0, 0.70710678, -0.70710678)),
    (np.array([0.0, 0.0, 0.0], dtype=np.float64), _quat_to_mat(0.70710678, -0.70710678, 0, 0)),
    (np.array([0.0, -0.1805, 0.0], dtype=np.float64), _quat_to_mat(0.70710678, 0.70710678, 0, 0)),
)

_G1_ARM_R_CHAIN: Tuple[Tuple[np.ndarray, np.ndarray], ...] = (
    (np.array([0.0, 0.0, 0.188], dtype=np.float64), np.eye(3, dtype=np.float64)),
    (np.array([0.0, 0.0, 0.0], dtype=np.float64), _quat_to_mat(0.5, -0.5, 0.5, -0.5)),
    (np.array([0.0, -0.305, 0.0], dtype=np.float64), _quat_to_mat(0.5, 0.5, -0.5, 0.5)),
    (np.array([0.0, 0.0, 0.0], dtype=np.float64), _quat_to_mat(0, 0, -0.70710678, 0.70710678)),
    (np.array([0.0, -0.1975, 0.0], dtype=np.float64), _quat_to_mat(0, 0, 0.70710678, -0.70710678)),
    (np.array([0.0, 0.0, 0.0], dtype=np.float64), _quat_to_mat(0.70710678, -0.70710678, 0, 0)),
    (np.array([0.0, -0.1805, 0.0], dtype=np.float64), _quat_to_mat(0.70710678, 0.70710678, 0, 0)),
)


def _g1_arm_fk(
    waist: np.ndarray,
    arm_joints: np.ndarray,
    arm_chain: Tuple[Tuple[np.ndarray, np.ndarray], ...],
    arm_base_offset: np.ndarray,
    R_arm_base_link: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """G1 arm FK in base_link frame.

    Args:
        waist: (N, 2) — [pitch, lift_height].
        arm_joints: (N, 7) joint angles.
        arm_chain: per-joint chain for the specific arm (left or right).
        arm_base_offset: translation from arm_base_link to arm_l/r_base_link.
        R_arm_base_link: fixed rotation of arm_l/r_base_link.
    Returns:
        pos (N, 3), rot (N, 3, 3).
    """
    n = waist.shape[0]
    pitch = waist[:, 0]
    slide = waist[:, 1] - _G1_SLIDE_OFFSET

    pos = np.zeros((n, 3), dtype=np.float64)
    rot = np.broadcast_to(np.eye(3, dtype=np.float64), (n, 3, 3)).copy()

    # body_link1: prismatic lift
    pos[:, 2] += 0.6485 + slide

    # body_link2: fixed offset + rotation, then waist pitch (revolute Z)
    pos[:, 0] += 0.131
    rot = rot @ _G1_R_BODY2
    cp, sp = np.cos(pitch), np.sin(pitch)
    Rp = np.zeros((n, 3, 3), dtype=np.float64)
    Rp[:, 0, 0] = cp
    Rp[:, 0, 1] = -sp
    Rp[:, 1, 0] = sp
    Rp[:, 1, 1] = cp
    Rp[:, 2, 2] = 1.0
    rot = rot @ Rp

    # arm_base_link (fixed)
    pos += np.einsum("nij,j->ni", rot, np.array([0.0, -0.305, 0.0], dtype=np.float64))
    rot = rot @ _G1_R_ARM_BASE

    # arm_l/r_base_link (fixed)
    pos += np.einsum("nij,j->ni", rot, arm_base_offset)
    rot = rot @ R_arm_base_link

    # 7-DOF arm chain
    arm_pos, arm_rot = _serial_fk(arm_joints, arm_chain)
    pos += np.einsum("nij,nj->ni", rot, arm_pos)
    rot = np.einsum("nij,njk->nik", rot, arm_rot)
    return pos, rot


_CAMERAS_A2D_REAL = (
    ("head", "observation.images.head"),
    ("hand_left", "observation.images.hand_left"),
    ("hand_right", "observation.images.hand_right"),
)
_CAMERAS_DUAL_RGB = (
    ("head", "images.rgb.head"),
    ("hand_left", "images.rgb.hand_left"),
    ("hand_right", "images.rgb.hand_right"),
)
_CAMERAS_FRANKA = (
    ("head", "images.rgb.head"),
    ("hand", "images.rgb.hand"),
)

# ── Gripper normalization (0 = closed, 1 = open) ───────────────────────────
# Each format records gripper differently; we map all to a unified [0, 1].
_GRIPPER_BOUNDS = {
    # raw values are in millimetres of physical opening; ~34.85=closed, ~120=open
    "a2d_real_state": (34.85, 120.0),
    # action is already normalized to [0, 1]
    "a2d_real_action": (0.0, 1.0),
    # genie1 sim: position correlates with openness (0.4=closed, 1.0=open)
    "a2d_sim": (0.4, 1.0),
    # split_aloha (AgileX Piper): metres, 0=closed, 0.10=open
    "dual_6dof_split_aloha": (0.0, 0.100),
    # lift2 (ARX R5A): metres, 0=closed, 0.088=open (4.4cm per finger)
    "dual_6dof_lift2": (0.0, 0.088),
    # franka: see _normalize_franka_gripper — the raw scale is ambiguous, so it
    # is resolved per-episode rather than with a fixed bound here.
}


def _normalize_gripper(x: torch.Tensor, key: str) -> torch.Tensor:
    """Map raw gripper value to [0, 1] where 0=closed, 1=open."""
    closed, open_val = _GRIPPER_BOUNDS[key]
    return ((x - closed) / (open_val - closed)).clamp(0.0, 1.0)


# InternData-A1's franka subset mixes two gripper encodings under *identical*
# feature keys, so _detect_format can't tell them apart:
#   1. physical opening width in metres — max ~0.08 (the Panda's 8 cm stroke);
#   2. an already-normalized [0, 1] value — max ~1.0.
# We classify each episode by its full-column max: a max within the metric
# stroke is metres (rescaled by _FRANKA_GRIPPER_OPEN_M), anything larger is
# taken as already normalized. The two clusters (~0.08 vs ~1.0) are far apart,
# so the threshold has wide margin.
_FRANKA_GRIPPER_OPEN_M = 0.08
_FRANKA_GRIPPER_METERS_MAX = 0.15


def _normalize_franka_gripper(gripper: torch.Tensor, full_column: torch.Tensor) -> torch.Tensor:
    """Normalize a franka gripper slice to [0, 1] (0=closed, 1=open).

    ``full_column`` is the whole episode's gripper column; its range decides
    which encoding this episode uses, independent of which frames ``gripper``
    covers (a single-frame slice can't reveal the scale on its own).
    """
    open_val = _FRANKA_GRIPPER_OPEN_M if float(full_column.float().max()) <= _FRANKA_GRIPPER_METERS_MAX else 1.0
    return (gripper / open_val).clamp(0.0, 1.0)


_PATH_SEGMENT_TO_ROBOT = {
    "genie1": RobotType.AGIBOT_G1,
    "lift2": RobotType.ARX_LIFT2,
    "split_aloha": RobotType.AGILEX_SPLIT_ALOHA,
    "franka": RobotType.FRANKA,
}


def _detect_format(features: Dict) -> str:
    keys = set(features.keys())
    if "observation.states.joint.position" in keys:
        return "a2d_real"
    if "states.left_joint.position" in keys:
        dim = features["states.left_joint.position"]["shape"][0]
        if dim == 7:
            return "a2d_sim"
        if dim == 6:
            return "dual_6dof"
        raise ValueError(f"Unexpected left_joint dim {dim}")
    if "states.joint.position" in keys:
        return "franka_single"
    raise ValueError(f"Cannot detect InternData-A1 format from feature keys: {sorted(keys)}")


def _cameras_for_format(fmt: str) -> Sequence[Tuple[str, str]]:
    if fmt == "a2d_real":
        return _CAMERAS_A2D_REAL
    if fmt in ("a2d_sim", "dual_6dof"):
        return _CAMERAS_DUAL_RGB
    if fmt == "franka_single":
        return _CAMERAS_FRANKA
    raise ValueError(f"Unknown format {fmt}")


def _robot_type_from_path(task_path: str) -> RobotType:
    parts = task_path.split(os.sep)
    for seg in reversed(parts):
        if seg in _PATH_SEGMENT_TO_ROBOT:
            return _PATH_SEGMENT_TO_ROBOT[seg]
    raise ValueError(f"Cannot infer robot type from path: {task_path}")


def _find_task_dirs(root: str) -> List[str]:
    results: List[str] = []
    for dirpath, dirnames, _ in os.walk(root):
        meta_info = os.path.join(dirpath, "meta", "info.json")
        if os.path.isfile(meta_info):
            results.append(dirpath)
            dirnames.clear()
        else:
            dirnames[:] = [d for d in dirnames if d not in ("data", "videos", "meta")]
    return sorted(results)


def _scan_task(task_path: str) -> List[EpisodeMetadata]:
    with open(os.path.join(task_path, "meta", "info.json"), "r") as f:
        info = json.load(f)
    if int(info.get("total_episodes") or 0) == 0:
        return []
    fmt = _detect_format(info["features"])
    cameras = _cameras_for_format(fmt)
    robot_type = _robot_type_from_path(task_path)
    chunks_size = int(info["chunks_size"])
    data_path_tpl = info["data_path"]
    video_path_tpl = info["video_path"]
    fps = float(info.get("fps", 30.0))

    tasks_map: Dict[int, str] = {}
    with open(os.path.join(task_path, "meta", "tasks.jsonl"), "r") as f:
        for line in f:
            entry = json.loads(line)
            tasks_map[int(entry["task_index"])] = entry["task"]
    n_tasks = max(tasks_map) + 1 if tasks_map else 0
    task_texts = [tasks_map.get(i, "") for i in range(n_tasks)]

    episodes: List[EpisodeMetadata] = []
    with open(os.path.join(task_path, "meta", "episodes.jsonl"), "r") as f:
        for line in f:
            entry = json.loads(line)
            n = int(entry["length"])
            if n <= 0:
                continue
            parquet_ep_index = int(entry["episode_index"])
            episode_chunk = parquet_ep_index // chunks_size
            parquet_path = os.path.join(
                task_path,
                data_path_tpl.format(
                    episode_chunk=episode_chunk,
                    episode_index=parquet_ep_index,
                ),
            )
            episodes.append(
                EpisodeMetadata(
                    length=n,
                    fps=fps,
                    robot_type=robot_type,
                    extras={
                        "parquet_path": parquet_path,
                        "video_dir": task_path,
                        "video_tpl": video_path_tpl,
                        "episode_chunk": episode_chunk,
                        "parquet_ep_index": parquet_ep_index,
                        "format": fmt,
                        "cameras": cameras,
                        "task_texts": task_texts,
                    },
                )
            )
    return episodes


@DATASET_REGISTRY.register()
class InternDataA1Dataset(BaseVLADataset):
    def _load_metadata(self) -> List[EpisodeMetadata]:
        subsets = ["sim_updated"]

        task_paths: List[str] = []
        for subset in subsets:
            root = os.path.join(self.data_path, subset)
            if os.path.isdir(root):
                task_paths.extend(_find_task_dirs(root))
        assert task_paths, f"No tasks found under {self.data_path}"

        out: List[EpisodeMetadata] = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            for episodes in tqdm(
                executor.map(_scan_task, task_paths), total=len(task_paths), desc="Scanning InternData-A1"
            ):
                out.extend(episodes)
        return out

    def _video_path(self, extras: Dict, video_key: str) -> str:
        return os.path.join(
            extras["video_dir"],
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
    def _gather_column(ds, name: str, idx: List[int]) -> torch.Tensor:
        """Bulk-read a column from a HuggingFace dataset.

        ``ds[:][name]`` reads the whole column as a single tensor (with
        ``with_format("torch")``), which is orders of magnitude faster than
        ``ds[i][name]`` per row when *idx* covers most/all of the episode.
        """
        col = ds[:][name]
        if isinstance(col, list):
            col = torch.tensor(col)
        out = col[idx].float() if len(idx) < len(col) else col.float()
        if out.ndim == 1:
            out = out.unsqueeze(-1)
        return out

    def _build_robot(self, ds, idx: List[int], fmt: str, prefix: str, cls, robot_type: RobotType = None):
        is_action = prefix.startswith("actions")
        if fmt == "a2d_real":
            joint = self._gather_column(ds, f"{prefix}joint.position", idx)
            effector = self._gather_column(ds, f"{prefix}effector.position", idx)
            head = self._gather_column(ds, f"{prefix}head.position", idx)
            waist = self._gather_column(ds, f"{prefix}waist.position", idx)
            grip_key = "a2d_real_action" if is_action else "a2d_real_state"
            effector = _normalize_gripper(effector, grip_key)
            left_arm = self._g1_arm_with_eef(
                waist,
                joint[:, :7],
                _G1_ARM_L_CHAIN,
                np.array([0.0, 0.025, 0.0], dtype=np.float64),
                _G1_R_ARM_L_BASE,
            )
            right_arm = self._g1_arm_with_eef(
                waist,
                joint[:, 7:14],
                _G1_ARM_R_CHAIN,
                np.array([0.0, -0.025, 0.0], dtype=np.float64),
                _G1_R_ARM_R_BASE,
            )
            return cls(
                left_arm=left_arm,
                right_arm=right_arm,
                left_gripper=Position(effector[:, 0:1], allow_relative=False),
                right_gripper=Position(effector[:, 1:2], allow_relative=False),
                head=Position(head),
                torso=Position(waist),
            )

        if fmt in ("a2d_sim", "dual_6dof"):
            left_joint = self._gather_column(ds, f"{prefix}left_joint.position", idx)
            right_joint = self._gather_column(ds, f"{prefix}right_joint.position", idx)
            left_grip = self._gather_column(ds, f"{prefix}left_gripper.position", idx)
            right_grip = self._gather_column(ds, f"{prefix}right_gripper.position", idx)
            if fmt == "dual_6dof":
                grip_key = "dual_6dof_lift2" if robot_type == RobotType.ARX_LIFT2 else "dual_6dof_split_aloha"
                left_grip = _normalize_gripper(left_grip, grip_key)
                right_grip = _normalize_gripper(right_grip, grip_key)
                fk_fn = _arx_lift2_fk if robot_type == RobotType.ARX_LIFT2 else _piper_fk
                left_arm = self._arm_with_eef(left_joint, fk_fn)
                right_arm = self._arm_with_eef(right_joint, fk_fn)
            else:
                left_grip = _normalize_gripper(left_grip, "a2d_sim")
                right_grip = _normalize_gripper(right_grip, "a2d_sim")
                # a2d_sim: G1 7-DOF arms without waist data; assume waist at rest
                n_frames = left_joint.shape[0]
                waist_zero = torch.zeros(n_frames, 2)
                waist_zero[:, 1] = _G1_SLIDE_OFFSET
                left_arm = self._g1_arm_with_eef(
                    waist_zero,
                    left_joint,
                    _G1_ARM_L_CHAIN,
                    np.array([0.0, 0.025, 0.0], dtype=np.float64),
                    _G1_R_ARM_L_BASE,
                )
                right_arm = self._g1_arm_with_eef(
                    waist_zero,
                    right_joint,
                    _G1_ARM_R_CHAIN,
                    np.array([0.0, -0.025, 0.0], dtype=np.float64),
                    _G1_R_ARM_R_BASE,
                )
            return cls(
                left_arm=left_arm,
                right_arm=right_arm,
                left_gripper=Position(left_grip, allow_relative=False),
                right_gripper=Position(right_grip, allow_relative=False),
            )

        if fmt == "franka_single":
            joint = self._gather_column(ds, f"{prefix}joint.position", idx)
            gripper = self._gather_column(ds, f"{prefix}gripper.position", idx)
            full_gripper = self._gather_column(ds, f"{prefix}gripper.position", list(range(len(ds))))
            gripper = _normalize_franka_gripper(gripper, full_gripper)
            pose = self._gather_column(ds, f"{prefix}gripper.pose", idx)
            return cls(
                left_arm=Arm(
                    joint_position=Position(joint),
                    eef_position=Position(pose[:, :3]),
                    eef_rotation=Rotation(pose[:, 3:6], representation=RotationRepresentation.EULER_XYZ),
                ),
                left_gripper=Position(gripper, allow_relative=False),
            )

        raise ValueError(f"Unknown format {fmt}")

    @staticmethod
    def _arm_with_eef(joint_tensor: torch.Tensor, fk_fn) -> Arm:
        joints_np = joint_tensor.numpy().astype(np.float64)
        pos, rot_mat = fk_fn(joints_np)
        eef_rpy = ScipyRotation.from_matrix(rot_mat).as_euler("xyz").astype(np.float32)
        return Arm(
            joint_position=Position(joint_tensor),
            eef_position=Position(torch.from_numpy(pos.astype(np.float32))),
            eef_rotation=Rotation(
                torch.from_numpy(eef_rpy),
                representation=RotationRepresentation.EULER_XYZ,
            ),
        )

    @staticmethod
    def _g1_arm_with_eef(
        waist_tensor: torch.Tensor,
        joint_tensor: torch.Tensor,
        arm_chain,
        arm_base_offset: np.ndarray,
        R_arm_base_link: np.ndarray,
    ) -> Arm:
        waist_np = waist_tensor.numpy().astype(np.float64)
        joints_np = joint_tensor.numpy().astype(np.float64)
        pos, rot_mat = _g1_arm_fk(waist_np, joints_np, arm_chain, arm_base_offset, R_arm_base_link)
        eef_rpy = ScipyRotation.from_matrix(rot_mat).as_euler("xyz").astype(np.float32)
        return Arm(
            joint_position=Position(joint_tensor),
            eef_position=Position(torch.from_numpy(pos.astype(np.float32))),
            eef_rotation=Rotation(
                torch.from_numpy(eef_rpy),
                representation=RotationRepresentation.EULER_XYZ,
            ),
        )

    def _load_action(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotAction:
        extras = self._metadata[episode_index].extras
        ds = self._get_parquet(extras["parquet_path"])
        return self._build_robot(
            ds,
            frame_index,
            extras["format"],
            "actions.",
            RobotAction,
            robot_type=self._metadata[episode_index].robot_type,
        )

    def _load_state(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotState:
        extras = self._metadata[episode_index].extras
        ds = self._get_parquet(extras["parquet_path"])
        prefix = "observation.states." if extras["format"] == "a2d_real" else "states."
        return self._build_robot(
            ds,
            frame_index,
            extras["format"],
            prefix,
            RobotState,
            robot_type=self._metadata[episode_index].robot_type,
        )

    def _load_images(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> Dict[str, torch.Tensor]:
        extras = self._metadata[episode_index].extras
        images: Dict[str, torch.Tensor] = {}
        for out_name, video_key in extras["cameras"]:
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
        fmt = extras["format"]
        task_texts = extras["task_texts"]

        with suppress_hf_progress():
            ds = load_dataset("parquet", data_files=extras["parquet_path"], split="train").with_format("torch")

        all_idx = list(range(n_total))
        full_action = self._build_robot(ds, all_idx, fmt, "actions.", RobotAction, robot_type=meta.robot_type)
        state_prefix = "observation.states." if fmt == "a2d_real" else "states."
        full_state = self._build_robot(ds, all_idx, fmt, state_prefix, RobotState, robot_type=meta.robot_type)

        # Preload task_index for the whole episode (one bulk read instead of
        # one ds[start]["task_index"] per step).
        ti_col = ds[:]["task_index"]
        all_task_indices = ti_col.tolist() if not isinstance(ti_col, list) else ti_col

        readers = {}
        if include_images:
            for out_name, video_key in extras["cameras"]:
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
