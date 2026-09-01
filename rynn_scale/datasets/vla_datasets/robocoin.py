"""RoboCOIN dataset (LeRobot v2.1, multi-embodiment).

Layout under ``data_path/``::

    <embodiment>_<task_name>/
        meta/info.json
        meta/episodes.jsonl
        meta/tasks.jsonl
        data/chunk-XXX/episode_XXXXXX.parquet
        videos/chunk-XXX/<video_key>/episode_XXXXXX.mp4

This adapter exposes a uniform schema across embodiments via ``eef_sim_pose_*``:
    RobotAction(left_arm=Arm(eef_position=…, eef_rotation=… EULER_XYZ),
                left_gripper=Position(...), right_arm=..., right_gripper=...)
"""

import json
import os
import re
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from datasets import load_dataset

from ...constants import RobotType, RotationRepresentation
from ...registry import DATASET_REGISTRY
from ...utils.robot import Arm, Position, RobotAction, RobotState, Rotation
from .base import BaseVLADataset, EpisodeMetadata
from .utils import SequentialVideoReader, VideoReader, fork_safe_cache, mt_process, suppress_hf_progress

_PREFIX_TO_ROBOT: Dict[str, RobotType] = {
    "AgiBot-g1": RobotType.AGIBOT_G1,
    "Cobot": RobotType.AGILEX_COBOT_MAGIC_2,
    "Agilex": RobotType.AGILEX_COBOT_MAGIC_2,
    "Split": RobotType.AGILEX_SPLIT_ALOHA,
    "R1": RobotType.GALAXEA_R1_LITE,
    "Galaxea": RobotType.GALAXEA_R1_LITE,
    "Galbot": RobotType.GALBOT_G1,
    "Realman": RobotType.REALMAN_RMC_AIDA,
    "RMC-AIDA-L": RobotType.REALMAN_RMC_AIDA,
    "G1edu-u3": RobotType.UNITREE_G1,
}

_ARM_JOINT_RE = re.compile(r"^(left|right)_arm_joint_(\d+)_rad$")

_EXPECTED_ARM_DOF: Dict[RobotType, int] = {
    RobotType.AGILEX_SPLIT_ALOHA: 6,
    RobotType.AGIBOT_G1: 7,
    RobotType.GALAXEA_R1_LITE: 6,
    RobotType.AGILEX_COBOT_MAGIC_2: 6,
    RobotType.GALBOT_G1: 7,
    RobotType.REALMAN_RMC_AIDA: 7,
    RobotType.UNITREE_G1: 7,
}


def _parse_arm_joint_cols(feat: Dict) -> Optional[Dict[str, List[int]]]:
    names = feat.get("names")
    if not isinstance(names, list):
        return None
    sides: Dict[str, List[Tuple[int, int]]] = {"left": [], "right": []}
    for i, n in enumerate(names):
        m = _ARM_JOINT_RE.match(n)
        if m:
            sides[m.group(1)].append((int(m.group(2)), i))
    left = [i for _, i in sorted(sides["left"])]
    right = [i for _, i in sorted(sides["right"])]
    if not left or not right or len(left) != len(right):
        return None
    return {"left": left, "right": right}


def _embodiment_prefix(task_name: str) -> str:
    for prefix in ("AgiBot-g1", "G1edu-u3", "RMC-AIDA-L", "Split"):
        if task_name.startswith(prefix):
            return prefix
    return task_name.split("_", 1)[0]


def _scan_task(task_dir: str) -> Optional[Dict]:
    info_path = os.path.join(task_dir, "meta", "info.json")
    eps_path = os.path.join(task_dir, "meta", "episodes.jsonl")
    if not (os.path.isfile(info_path) and os.path.isfile(eps_path)):
        return None
    try:
        with open(info_path) as f:
            info = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    feats = info.get("features", {})
    has_eef = "eef_sim_pose_state" in feats and "eef_sim_pose_action" in feats
    has_gripper = "gripper_open_scale_state" in feats and "gripper_open_scale_action" in feats
    if not has_eef:
        return None

    img_keys = sorted(k for k in feats if k.startswith("observation.images."))
    cam_names = [k[len("observation.images.") :] for k in img_keys]

    act_joint_cols = _parse_arm_joint_cols(feats.get("action", {}))
    state_joint_cols = _parse_arm_joint_cols(feats.get("observation.state", {}))

    episodes: List[Dict] = []
    try:
        with open(eps_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                n = int(e.get("length", 0))
                if n <= 0:
                    continue
                tasks = e.get("tasks") or [""]
                instruction = tasks[0] if isinstance(tasks, list) and tasks else ""
                episodes.append(
                    {
                        "episode_index": int(e["episode_index"]),
                        "length": n,
                        "instruction": instruction,
                    }
                )
    except (OSError, json.JSONDecodeError):
        return None

    if not episodes:
        return None

    task_name = os.path.basename(task_dir)
    prefix = _embodiment_prefix(task_name)
    if prefix not in _PREFIX_TO_ROBOT:
        return None
    # AgiBot g1 exposes several fisheye cameras that we don't use.
    if prefix == "AgiBot-g1":
        cam_names = [c for c in cam_names if "fisheye" not in c]
    return {
        "task_name": task_name,
        "task_dir": task_dir,
        "embodiment": prefix,
        "robot_type": _PREFIX_TO_ROBOT[prefix].value,
        "fps": float(info.get("fps", 30)),
        "chunks_size": int(info["chunks_size"]),
        "data_tpl": info["data_path"],
        "video_tpl": info["video_path"],
        "cam_names": cam_names,
        "has_gripper": has_gripper,
        "act_joint_cols": act_joint_cols,
        "state_joint_cols": state_joint_cols,
        "episodes": episodes,
    }


@DATASET_REGISTRY.register()
class RoboCOINDataset(BaseVLADataset):
    def __init__(self, *args, include_no_gripper: bool = False, **kwargs):
        self._include_no_gripper = include_no_gripper
        super().__init__(*args, **kwargs)

    def _load_metadata(self) -> List[EpisodeMetadata]:
        task_dirs = sorted(
            os.path.join(self.data_path, d)
            for d in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, d))
        )
        tasks = [
            t for t in mt_process(_scan_task, task_dirs, max_workers=32, desc="Scanning RoboCOIN") if t is not None
        ]
        tasks.sort(key=lambda t: t["task_name"])

        out: List[EpisodeMetadata] = []
        for task in tasks:
            if not self._include_no_gripper and not task["has_gripper"]:
                continue
            for ep in task["episodes"]:
                episode_chunk = ep["episode_index"] // task["chunks_size"]
                parquet_path = os.path.join(
                    task["task_dir"],
                    task["data_tpl"].format(
                        episode_chunk=episode_chunk,
                        episode_index=ep["episode_index"],
                    ),
                )
                out.append(
                    EpisodeMetadata(
                        length=ep["length"],
                        fps=float(task["fps"]),
                        robot_type=RobotType(task["robot_type"]),
                        extras={
                            "parquet_path": parquet_path,
                            "task_dir": task["task_dir"],
                            "video_tpl": task["video_tpl"],
                            "episode_chunk": episode_chunk,
                            "episode_index": ep["episode_index"],
                            "cam_names": task["cam_names"],
                            "has_gripper": task["has_gripper"],
                            "act_joint_cols": task["act_joint_cols"],
                            "state_joint_cols": task["state_joint_cols"],
                            "instruction": ep["instruction"],
                        },
                    )
                )

        assert out, f"No usable RoboCOIN episodes under {self.data_path}"
        return out

    # ── parquet / video access ─────────────────────────────────

    def _video_path(self, extras: Dict, cam: str) -> str:
        return os.path.join(
            extras["task_dir"],
            extras["video_tpl"].format(
                episode_chunk=extras["episode_chunk"],
                video_key=f"observation.images.{cam}",
                episode_index=extras["episode_index"],
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

    # ── helpers ────────────────────────────────────────────────

    @staticmethod
    def _gather_column(ds, name: str, idx: List[int]) -> torch.Tensor:
        out = torch.stack([ds[i][name] for i in idx], dim=0).float()
        if out.ndim == 1:
            out = out.unsqueeze(-1)
        return out

    @staticmethod
    def _gather_joints(
        ds, col: str, idx: List[int], joint_cols: Dict[str, List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        full = torch.stack([ds[i][col] for i in idx], dim=0).float()
        return full[:, joint_cols["left"]], full[:, joint_cols["right"]]

    @staticmethod
    def _build_robot(
        eef: torch.Tensor,
        gripper: Optional[torch.Tensor],
        cls,
        joint_left: Optional[torch.Tensor] = None,
        joint_right: Optional[torch.Tensor] = None,
    ):
        T = eef.shape[0]
        if gripper is None:
            gripper = torch.zeros(T, 2)
        left_arm = Arm(
            eef_position=Position(eef[:, 0:3]),
            eef_rotation=Rotation(eef[:, 3:6], representation=RotationRepresentation.EULER_XYZ),
            joint_position=Position(joint_left) if joint_left is not None else None,
        )
        right_arm = Arm(
            eef_position=Position(eef[:, 6:9]),
            eef_rotation=Rotation(eef[:, 9:12], representation=RotationRepresentation.EULER_XYZ),
            joint_position=Position(joint_right) if joint_right is not None else None,
        )
        return cls(
            left_arm=left_arm,
            right_arm=right_arm,
            left_gripper=Position(gripper[:, 0:1], allow_relative=False),
            right_gripper=Position(gripper[:, 1:2], allow_relative=False),
        )

    def _joint_cols_for(self, extras: Dict, kind: str) -> Optional[Dict[str, List[int]]]:
        cols = extras["act_joint_cols" if kind == "act" else "state_joint_cols"]
        if cols is None:
            return None
        exp = _EXPECTED_ARM_DOF.get(self._metadata[0].robot_type)
        # Use per-episode robot_type from metadata
        return cols

    # ── _load_* ────────────────────────────────────────────────

    def _load_action(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotAction:
        extras = self._metadata[episode_index].extras
        ds = self._get_parquet(extras["parquet_path"])
        eef = self._gather_column(ds, "eef_sim_pose_action", frame_index)
        gripper = self._gather_column(ds, "gripper_open_scale_action", frame_index) if extras["has_gripper"] else None
        cols = extras["act_joint_cols"]
        jl, jr = self._gather_joints(ds, "action", frame_index, cols) if cols is not None else (None, None)
        return self._build_robot(eef, gripper, RobotAction, jl, jr)

    def _load_state(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotState:
        extras = self._metadata[episode_index].extras
        ds = self._get_parquet(extras["parquet_path"])
        eef = self._gather_column(ds, "eef_sim_pose_state", frame_index)
        gripper = self._gather_column(ds, "gripper_open_scale_state", frame_index) if extras["has_gripper"] else None
        cols = extras["state_joint_cols"]
        jl, jr = self._gather_joints(ds, "observation.state", frame_index, cols) if cols is not None else (None, None)
        return self._build_robot(eef, gripper, RobotState, jl, jr)

    @fork_safe_cache
    def _get_video_reader(self, video_path: str):
        if not os.path.exists(video_path):
            return None
        return VideoReader(video_path)

    def _load_images(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> Dict[str, torch.Tensor]:
        extras = self._metadata[episode_index].extras
        out: Dict[str, torch.Tensor] = {}
        for cam in extras["cam_names"]:
            reader = self._get_video_reader(self._video_path(extras, cam))
            if reader is None:
                continue
            out[cam] = reader.read(frame_index)
        return out

    def _load_instruction(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> List[str]:
        return [self._metadata[episode_index].extras["instruction"]] * len(frame_index)

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
        instruction = extras["instruction"]

        with suppress_hf_progress():
            ds = load_dataset("parquet", data_files=extras["parquet_path"], split="train").with_format("torch")

        all_idx = list(range(n_total))
        eef_action = self._gather_column(ds, "eef_sim_pose_action", all_idx)
        eef_state = self._gather_column(ds, "eef_sim_pose_state", all_idx)
        grip_action = self._gather_column(ds, "gripper_open_scale_action", all_idx) if extras["has_gripper"] else None
        grip_state = self._gather_column(ds, "gripper_open_scale_state", all_idx) if extras["has_gripper"] else None

        act_cols = extras["act_joint_cols"]
        state_cols = extras["state_joint_cols"]
        ajl, ajr = self._gather_joints(ds, "action", all_idx, act_cols) if act_cols else (None, None)
        sjl, sjr = self._gather_joints(ds, "observation.state", all_idx, state_cols) if state_cols else (None, None)

        full_action = self._build_robot(eef_action, grip_action, RobotAction, ajl, ajr)
        full_state = self._build_robot(eef_state, grip_state, RobotState, sjl, sjr)

        readers = {}
        if include_images:
            for cam in extras["cam_names"]:
                path = self._video_path(extras, cam)
                if os.path.exists(path):
                    readers[cam] = SequentialVideoReader(path)

        try:
            for start, end in source_ranges:
                images = None
                if include_images:
                    images = {k: r.read(start) for k, r in readers.items()}
                yield {
                    "state": full_state[start : start + 1],
                    "action": full_action[start:end],
                    "instruction": instruction,
                    "images": images,
                }
        finally:
            for r in readers.values():
                r.close()
