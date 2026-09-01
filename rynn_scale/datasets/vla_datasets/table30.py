"""RoboChallenge Table30 dataset (30 tasks × 4 embodiments).

Expected layout (assume tar parts already extracted offline):

    data_path/
        <task_name>/
            task_desc.json
            meta/task_info.json
            data/
                episode_000000/
                    meta/episode_meta.json
                    states/
                        states.jsonl         (single-arm: ARX5 / UR5 / FRANKA)
                        # or
                        left_states.jsonl    (ALOHA dual-arm)
                        right_states.jsonl
                    videos/*.mp4
                ...

Action semantics: `action[t] = state[t+1]` (per official convert_to_lerobot.py);
the last frame's action repeats the last state.
"""

import json
import os
from glob import glob
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch

from ...constants import RobotType, RotationRepresentation
from ...registry import DATASET_REGISTRY
from ...utils.robot import Arm, Position, RobotAction, RobotState, Rotation
from .base import BaseVLADataset, EpisodeMetadata
from .utils import decode_video_frames_pyav_by_timestamps, mt_process

_EMBODIMENT_TO_ROBOT = {
    "ARX5": RobotType.ARX_X5,
    "UR5": RobotType.UR_5,
    "FRANKA": RobotType.FRANKA,
    "ALOHA": RobotType.AGILEX_COBOT_MAGIC_2,
    "DOS-W1": RobotType.DEXMAL_DOS_W1,
}

# State field names differ between Table30 v1 (per-embodiment layouts) and v2
# (uniform joint_positions / ee_positions[quat] / gripper_width). Rather than
# key the schema on embodiment + dataset version, resolve each field by presence
# in the states dict. Candidates are ordered by preference; the first present
# wins. This reproduces every v1 mapping exactly and covers v2 as well:
#   ARX5   v1: end_effector_pose(euler)      v2: ee_positions(quat)
#   UR5    v1: gripper                        v2: gripper_width
#   ALOHA  v1: qpos, ee_pose_quaternion       v2: joint_positions, ee_positions
_JOINT_KEYS: Tuple[str, ...] = ("qpos", "joint_positions")
# (field name, rotation representation of the trailing rotation dims)
_EE_KEYS: Tuple[Tuple[str, RotationRepresentation], ...] = (
    ("end_effector_pose", RotationRepresentation.EULER_XYZ),  # xyz(3) + euler(3)
    ("ee_pose_quaternion", RotationRepresentation.QUAT_XYZW),  # xyz(3) + quat(4)
    ("ee_positions", RotationRepresentation.QUAT_XYZW),  # xyz(3) + quat(4)
)
_GRIPPER_KEYS: Tuple[str, ...] = ("gripper_width", "gripper")


def _pick_key(states: Dict[str, np.ndarray], candidates: Tuple[str, ...], kind: str) -> str:
    for k in candidates:
        if k in states:
            return k
    raise KeyError(f"Table30: no {kind} field among {list(candidates)}; available keys: {sorted(states.keys())}")


def _scan_task(task_path: str):
    task_info_path = os.path.join(task_path, "meta", "task_info.json")
    data_dir = os.path.join(task_path, "data")
    try:
        with open(task_info_path, "r", encoding="utf-8") as f:
            task_info = json.load(f)
        ep_names = sorted(os.listdir(data_dir))
    except (OSError, json.JSONDecodeError):
        return []
    tags = task_info["task_desc"]["task_tag"]
    tag_upper = {t.upper(): t for t in tags}
    embodiment = next((k for k in _EMBODIMENT_TO_ROBOT if k in tag_upper), None)
    if embodiment is None:
        return []
    task_meta = {
        "task_name": os.path.basename(task_path),
        "embodiment": embodiment,
        "robot_type": _EMBODIMENT_TO_ROBOT[embodiment].value,
        "prompt": task_info["task_desc"]["prompt"],
        "fps": float(task_info["video_info"]["fps"]),
    }
    return [(task_meta, os.path.join(data_dir, ep)) for ep in ep_names]


def _scan_episode(job):
    task_meta, ep_dir = job
    ep_meta_path = os.path.join(ep_dir, "meta", "episode_meta.json")
    try:
        with open(ep_meta_path, "r", encoding="utf-8") as f:
            ep_meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    n = int(ep_meta.get("frames", 0))
    if n <= 0:
        return None
    return {**task_meta, "episode_dir": ep_dir, "length": n}


def _load_jsonl_as_numpy(path: str) -> Dict[str, np.ndarray]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return {}
    out: Dict[str, np.ndarray] = {}
    for k in rows[0].keys():
        values = [r[k] for r in rows]
        dtype = np.float64 if k == "timestamp" else np.float32
        out[k] = np.asarray(values, dtype=dtype)
    return out


def _arm_from_states(
    states: Dict[str, np.ndarray],
    embodiment: str,
    idxs: List[int],
) -> Tuple[Arm, Position]:
    joint_key = _pick_key(states, _JOINT_KEYS, "joint_position")
    ee_key, ee_rot = next(((k, r) for k, r in _EE_KEYS if k in states), (None, None))
    if ee_key is None:
        raise KeyError(
            f"Table30: no eef_pose field among {[k for k, _ in _EE_KEYS]}; available keys: {sorted(states.keys())}"
        )
    gripper_key = _pick_key(states, _GRIPPER_KEYS, "gripper")
    jp = states[joint_key][idxs]
    ee = states[ee_key][idxs]
    g = states[gripper_key][idxs]
    if g.ndim == 1:
        g = g[:, None]
    arm = Arm(
        joint_position=Position(torch.from_numpy(jp.copy()).float()),
        eef_position=Position(torch.from_numpy(ee[:, :3].copy()).float()),
        eef_rotation=Rotation(
            torch.from_numpy(ee[:, 3 : 3 + ee_rot.dim].copy()).float(),
            representation=ee_rot,
        ),
    )
    gripper = Position(torch.from_numpy(g.copy()).float(), allow_relative=False)
    return arm, gripper


def _load_states(episode_dir: str, embodiment: str) -> Dict[str, Dict[str, np.ndarray]]:
    states_dir = os.path.join(episode_dir, "states")
    left_path = os.path.join(states_dir, "left_states.jsonl")
    if os.path.isfile(left_path):
        return {
            "left": _load_jsonl_as_numpy(left_path),
            "right": _load_jsonl_as_numpy(os.path.join(states_dir, "right_states.jsonl")),
        }
    return {"left": _load_jsonl_as_numpy(os.path.join(states_dir, "states.jsonl"))}


def _build_proprio(states: Dict, embodiment: str, idxs: List[int], cls):
    left_arm, left_grip = _arm_from_states(states["left"], embodiment, idxs)
    if "right" in states:
        right_arm, right_grip = _arm_from_states(states["right"], embodiment, idxs)
    else:
        right_arm = right_grip = None
    return cls(
        left_arm=left_arm,
        right_arm=right_arm,
        left_gripper=left_grip,
        right_gripper=right_grip,
    )


@DATASET_REGISTRY.register()
class Table30Dataset(BaseVLADataset):
    def _load_metadata(self) -> List[EpisodeMetadata]:
        task_paths = [
            os.path.join(self.data_path, t)
            for t in sorted(os.listdir(self.data_path))
            if os.path.isdir(os.path.join(self.data_path, t))
        ]
        nested = mt_process(_scan_task, task_paths, max_workers=32, desc="Listing Table30 tasks")
        jobs = [j for lst in nested for j in lst]
        episodes = [e for e in mt_process(_scan_episode, jobs, max_workers=64, desc="Scanning Table30 episodes") if e]
        episodes.sort(key=lambda e: (e["task_name"], e["episode_dir"]))

        assert episodes, f"No Table30 episodes found under {self.data_path}"

        out: List[EpisodeMetadata] = []
        for ep in episodes:
            out.append(
                EpisodeMetadata(
                    length=ep["length"],
                    fps=float(ep["fps"]),
                    robot_type=RobotType(ep["robot_type"]),
                    extras={
                        "episode_dir": ep["episode_dir"],
                        "embodiment": ep["embodiment"],
                        "prompt": ep["prompt"],
                    },
                )
            )
        return out

    # ── helpers ──────────────────────────────────────────────

    def _next_idxs(self, episode_index: int, idxs: List[int]) -> List[int]:
        n = self._metadata[episode_index].length
        return [min(i + 1, n - 1) for i in idxs]

    # ── _load_* ─────────────────────────────────────────────

    def _load_state(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotState:
        extras = self._metadata[episode_index].extras
        states = _load_states(extras["episode_dir"], extras["embodiment"])
        return _build_proprio(states, extras["embodiment"], frame_index, RobotState)

    def _load_action(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotAction:
        extras = self._metadata[episode_index].extras
        states = _load_states(extras["episode_dir"], extras["embodiment"])
        action_idxs = self._next_idxs(episode_index, frame_index)
        return _build_proprio(states, extras["embodiment"], action_idxs, RobotAction)

    def _load_images(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> Dict[str, torch.Tensor]:
        extras = self._metadata[episode_index].extras
        meta = self._metadata[episode_index]
        states = _load_states(extras["episode_dir"], extras["embodiment"])

        ts = states["left"]["timestamp"]
        if ts.ndim == 2 and ts.shape[1] == 1:
            ts = ts[:, 0]
        ts = ts.astype(np.float64)
        ts_rel = ts - ts[0]
        timestamps = ts_rel[frame_index].tolist()
        tol = 1.5 / meta.fps

        videos_dir = os.path.join(extras["episode_dir"], "videos")
        out: Dict[str, torch.Tensor] = {}
        for video_path in sorted(glob(os.path.join(videos_dir, "*.mp4"))):
            cam_key = os.path.splitext(os.path.basename(video_path))[0]
            cam_key = cam_key.replace("_realsense_rgb", "").replace("_rgb", "")
            frames = decode_video_frames_pyav_by_timestamps(
                video_path=video_path,
                timestamps=timestamps,
                tolerance_s=tol,
            )
            out[cam_key] = frames.contiguous()
        return out

    def _load_instruction(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> List[str]:
        return [self._metadata[episode_index].extras["prompt"]] * len(frame_index)

    # ── _iter_episode ────────────────────────────────────────

    def _iter_episode(
        self,
        episode_index: int,
        source_ranges: List[tuple],
        include_images: bool = True,
    ) -> Iterator[Dict]:
        meta = self._metadata[episode_index]
        extras = meta.extras
        n_total = meta.length
        embodiment = extras["embodiment"]
        prompt = extras["prompt"]

        states = _load_states(extras["episode_dir"], embodiment)
        all_idx = list(range(n_total))
        action_idx = [min(i + 1, n_total - 1) for i in all_idx]
        full_state = _build_proprio(states, embodiment, all_idx, RobotState)
        full_action = _build_proprio(states, embodiment, action_idx, RobotAction)

        img_loader = None
        if include_images:
            _decode = decode_video_frames_pyav_by_timestamps

            ts = states["left"]["timestamp"]
            if ts.ndim == 2 and ts.shape[1] == 1:
                ts = ts[:, 0]
            ts = ts.astype(np.float64)
            ts_rel = ts - ts[0]
            tol = 1.5 / meta.fps
            videos_dir = os.path.join(extras["episode_dir"], "videos")
            video_paths = sorted(glob(os.path.join(videos_dir, "*.mp4")))

            def _load_frame_images(k: int) -> Dict[str, torch.Tensor]:
                images = {}
                for vp in video_paths:
                    cam_key = os.path.splitext(os.path.basename(vp))[0]
                    cam_key = cam_key.replace("_realsense_rgb", "").replace("_rgb", "")
                    frame = _decode(
                        video_path=vp,
                        timestamps=[float(ts_rel[k])],
                        tolerance_s=tol,
                    )
                    images[cam_key] = frame[0].contiguous()
                return images

            img_loader = _load_frame_images

        for start, end in source_ranges:
            images = img_loader(start) if img_loader is not None else None
            yield {
                "state": full_state[start : start + 1],
                "action": full_action[start:end],
                "instruction": prompt,
                "images": images,
            }
