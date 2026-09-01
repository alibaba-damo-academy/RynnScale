import io
import json
import os
import re
from typing import Dict, Iterator, List

import h5py
import numpy as np
import torch
from PIL import Image

from ...constants import RobotType, RotationRepresentation
from ...registry import DATASET_REGISTRY
from ...utils.robot import Arm, Position, RobotAction, RobotState, Rotation
from .base import BaseVLADataset, EpisodeMetadata
from .utils import fork_safe_cache

CAMERA_KEYS = ("head", "left", "right")
CAMERA_OUTPUT_NAMES = {"head": "head", "left": "hand_left", "right": "hand_right"}

_EPISODE_RE = re.compile(r".*_episode_(\d+)\.hdf5$")


def _decode_jpeg_frame(rgb_data: np.ndarray, rgb_size: np.ndarray, frame_idx: int) -> np.ndarray:
    offset = int(rgb_size[:frame_idx].sum())
    size = int(rgb_size[frame_idx])
    jpeg_bytes = rgb_data[offset : offset + size].tobytes()
    img = Image.open(io.BytesIO(jpeg_bytes))
    return np.array(img)


def _find_task_dirs(root: str) -> List[str]:
    results: List[str] = []
    for entry in sorted(os.listdir(root)):
        d = os.path.join(root, entry)
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "task_info.json")):
            results.append(d)
    return results


@DATASET_REGISTRY.register()
class AstribotDemoDataset(BaseVLADataset):
    def _load_metadata(self) -> List[EpisodeMetadata]:
        task_dirs = _find_task_dirs(self.data_path)
        if not task_dirs:
            task_dirs = [self.data_path]

        out: List[EpisodeMetadata] = []
        for task_dir in task_dirs:
            info_path = os.path.join(task_dir, "task_info.json")
            if os.path.isfile(info_path):
                with open(info_path, "r") as f:
                    info = json.load(f)
                instruction = info.get("Description") or info.get("Name") or ""
            else:
                instruction = os.path.basename(task_dir)

            hdf5_files = sorted(f for f in os.listdir(task_dir) if _EPISODE_RE.match(f))
            for fname in hdf5_files:
                path = os.path.join(task_dir, fname)
                with h5py.File(path, "r") as f:
                    t = f["time"]
                    n = len(t)
                    if n <= 1:
                        continue
                    timestamps = t[:]
                    fps = (n - 1) / (timestamps[-1] - timestamps[0])

                out.append(
                    EpisodeMetadata(
                        length=n,
                        fps=float(fps),
                        robot_type=RobotType.ASTRIBOT,
                        extras={
                            "path": path,
                            "instruction": instruction,
                        },
                    )
                )

        assert out, f"No valid episodes found under {self.data_path}"
        return out

    # ── HDF5 access ──────────────────────────────────────────

    @fork_safe_cache
    def _open_hdf5(self, path: str):
        return h5py.File(path, "r")

    # ── helpers ──────────────────────────────────────────────

    def _build_robot(self, f: h5py.File, idx: List[int], pose_group: str, cls):
        def gather(dataset_path: str) -> torch.Tensor:
            data = f[dataset_path]
            out = torch.from_numpy(np.stack([data[i] for i in idx], axis=0)).float()
            if out.ndim == 1:
                out = out.unsqueeze(-1)
            return out

        arm_left = gather(f"{pose_group}/astribot_arm_left")
        arm_right = gather(f"{pose_group}/astribot_arm_right")
        gripper_left = gather(f"{pose_group}/astribot_gripper_left")
        gripper_right = gather(f"{pose_group}/astribot_gripper_right")
        torso = gather(f"{pose_group}/astribot_torso")
        head = gather(f"{pose_group}/astribot_head")

        return cls(
            left_arm=Arm(
                eef_position=Position(arm_left[:, :3]),
                eef_rotation=Rotation(
                    arm_left[:, 3:7],
                    representation=RotationRepresentation.QUAT_XYZW,
                ),
            ),
            right_arm=Arm(
                eef_position=Position(arm_right[:, :3]),
                eef_rotation=Rotation(
                    arm_right[:, 3:7],
                    representation=RotationRepresentation.QUAT_XYZW,
                ),
            ),
            left_gripper=Position(gripper_left, allow_relative=False),
            right_gripper=Position(gripper_right, allow_relative=False),
            torso=Position(torso),
            head=Position(head),
        )

    # ── _load_* ──────────────────────────────────────────────

    def _load_action(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotAction:
        f = self._open_hdf5(self._metadata[episode_index].extras["path"])
        return self._build_robot(f, frame_index, "command_poses_dict", RobotAction)

    def _load_state(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotState:
        f = self._open_hdf5(self._metadata[episode_index].extras["path"])
        return self._build_robot(f, frame_index, "poses_dict", RobotState)

    def _load_images(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> Dict[str, torch.Tensor]:
        f = self._open_hdf5(self._metadata[episode_index].extras["path"])
        images: Dict[str, torch.Tensor] = {}
        for cam_key in CAMERA_KEYS:
            grp_path = f"images_dict/{cam_key}"
            if grp_path not in f:
                continue
            rgb_data = f[f"{grp_path}/rgb"]
            rgb_size = f[f"{grp_path}/rgb_size"][:]
            frames = [_decode_jpeg_frame(rgb_data, rgb_size, i) for i in frame_index]
            images[CAMERA_OUTPUT_NAMES[cam_key]] = torch.from_numpy(np.stack(frames, axis=0))
        return images

    def _load_instruction(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> List[str]:
        return [self._metadata[episode_index].extras["instruction"]] * len(frame_index)

    # ── _iter_episode ────────────────────────────────────────

    def _iter_episode(
        self,
        episode_index: int,
        source_ranges: List[tuple],
        include_images: bool = True,
    ) -> Iterator[Dict]:
        meta = self._metadata[episode_index]
        n_total = meta.length
        instruction = meta.extras["instruction"]

        with h5py.File(meta.extras["path"], "r") as f:
            full_state = self._build_robot(f, list(range(n_total)), "poses_dict", RobotState)
            full_action = self._build_robot(f, list(range(n_total)), "command_poses_dict", RobotAction)

            rgb_cache = {}
            if include_images:
                for cam_key in CAMERA_KEYS:
                    grp_path = f"images_dict/{cam_key}"
                    if grp_path in f:
                        rgb_cache[cam_key] = (f[f"{grp_path}/rgb"], f[f"{grp_path}/rgb_size"][:])

            for start, end in source_ranges:
                images = None
                if include_images:
                    images = {}
                    for cam_key, (rgb_data, rgb_size) in rgb_cache.items():
                        frame = _decode_jpeg_frame(rgb_data, rgb_size, start)
                        images[CAMERA_OUTPUT_NAMES[cam_key]] = torch.from_numpy(frame)
                yield {
                    "state": full_state[start : start + 1],
                    "action": full_action[start:end],
                    "instruction": instruction,
                    "images": images,
                }
