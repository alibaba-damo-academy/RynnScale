import json
import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from typing import Dict, Iterator, List

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyRotation
from tqdm import tqdm

from ...constants import RobotType, RotationRepresentation
from ...registry import DATASET_REGISTRY
from ...utils.robot import (
    Arm,
    Position,
    RobotAction,
    RobotState,
    Rotation,
    _to_scipy_rotation,
)
from .base import BaseVLADataset, EpisodeMetadata
from .utils import fork_safe_cache

# robosuite default OSC_POSE controller (control_delta=True): the stored
# action is normalized to [-1, 1] and the controller scales it to
# output_max before applying it in the world frame relative to the current
# EEF pose. Position -> ±0.05 m, orientation axis-angle -> ±0.5 rad.
OSC_POS_SCALE = 0.05
OSC_ROT_SCALE = 0.5

# robosuite Panda gripper finger qpos open width (m); normalizes the raw
# gripper_qpos state to [0, 1] (0 = closed, 1 = open).
GRIPPER_OPEN_WIDTH = 0.04

# LIBERO places the robot base at this world-frame position (no rotation).
# Subtract from ee_states to get EEF pose in robot base_link frame, which
# is what real-robot data and the renderer expect.
BASE_POS = [-0.66, 0.0, 0.912]


@DATASET_REGISTRY.register()
class LiberoDataset(BaseVLADataset):
    # LIBERO uses Robosuite's default 20Hz control rate.
    _SOURCE_FPS = 20.0

    ALL_SUBSETS = [
        "libero_10",
        "libero_90",
        "libero_goal",
        "libero_object",
        "libero_spatial",
    ]

    def __init__(self, *args, subsets: List[str] = None, **kwargs):
        self.subsets = subsets if subsets is not None else self.ALL_SUBSETS
        super().__init__(*args, **kwargs)

    def _load_metadata(self) -> List[EpisodeMetadata]:
        data_files = []
        for subset in self.subsets:
            subset_dir = os.path.join(self.data_path, subset)
            files = sorted(glob(os.path.join(subset_dir, "*.hdf5")))
            assert len(files) > 0, f"no *.hdf5 files under {subset_dir}"
            data_files.extend(files)

        def _load_one(args):
            task_idx, path = args
            results = []
            with h5py.File(path, "r", locking=False) as f:
                for demo_idx in range(len(f["data"])):
                    n = len(f["data"][f"demo_{demo_idx}"]["actions"])
                    results.append(
                        EpisodeMetadata(
                            length=n,
                            fps=self._SOURCE_FPS,
                            robot_type=RobotType.FRANKA,
                            extras={
                                "data_path": path,
                                "task_index": task_idx,
                                "demo_index": demo_idx,
                            },
                        )
                    )
            return results

        out: List[EpisodeMetadata] = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            for results in tqdm(
                executor.map(_load_one, enumerate(data_files)),
                total=len(data_files),
                desc="Loading episodes",
            ):
                out.extend(results)
        return out

    @fork_safe_cache
    def _open_hdf5_file(self, path: str):
        return h5py.File(path, "r", locking=False)

    def _get_episode_group(self, episode_index: int):
        metadata = self._metadata[episode_index]
        demo_index = int(metadata.extras["demo_index"])
        return self._open_hdf5_file(metadata.extras["data_path"])["data"][f"demo_{demo_index}"]

    @fork_safe_cache
    def _task_instruction(self, episode_index: int) -> str:
        meta = self._metadata[episode_index]
        problem_info = self._open_hdf5_file(meta.extras["data_path"])["data"].attrs["problem_info"]
        return json.loads(problem_info)["language_instruction"]

    def _make_state(self, ee_states: torch.Tensor, gripper_state: torch.Tensor) -> RobotState:
        gripper_norm = (gripper_state / GRIPPER_OPEN_WIDTH).clamp(0.0, 1.0)
        ee_pos = ee_states[:, :3] - torch.tensor(BASE_POS, dtype=torch.float32)
        return RobotState(
            left_arm=Arm(
                eef_position=Position(ee_pos),
                eef_rotation=Rotation(ee_states[:, 3:], representation=RotationRepresentation.ROT_VEC),
            ),
            left_gripper=Position(gripper_norm, allow_relative=False),
        )

    def _make_action(self, actions: torch.Tensor, state: RobotState) -> RobotAction:
        # The stored action is the raw normalized OSC_POSE command. Recover the
        # absolute commanded pose exactly as the controller does: scale the
        # normalized deltas to metric units, then apply them in the world frame
        # relative to the current EEF pose. Position adds; orientation is a
        # world-frame (left-multiply) rotation: R_goal = R(delta) @ R_current.
        pos_delta = actions[:, :3] * OSC_POS_SCALE
        abs_pos = state.left_arm.eef_position.data + pos_delta

        rot_delta = ScipyRotation.from_rotvec((actions[:, 3:6] * OSC_ROT_SCALE).numpy())
        cur_rot = _to_scipy_rotation(state.left_arm.eef_rotation.data, RotationRepresentation.ROT_VEC)
        abs_rot = torch.from_numpy((rot_delta * cur_rot).as_rotvec()).float()

        # Gripper command: robosuite +1 = close, -1 = open -> [0, 1], 0 = closed.
        gripper = (1.0 - actions[:, 6:7]) * 0.5
        return RobotAction(
            left_arm=Arm(
                eef_position=Position(abs_pos),
                eef_rotation=Rotation(abs_rot, representation=RotationRepresentation.ROT_VEC),
            ),
            left_gripper=Position(gripper, allow_relative=False),
        )

    def _load_action(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotAction:
        episode = self._get_episode_group(episode_index)

        ee_states = torch.from_numpy(np.array(episode["obs"]["ee_states"][frame_index])).float()
        gripper_state = torch.from_numpy(np.array(episode["obs"]["gripper_states"][frame_index][:, :1])).float()
        state = self._make_state(ee_states, gripper_state)

        actions = torch.from_numpy(np.array(episode["actions"][frame_index])).float()
        return self._make_action(actions, state)

    def _load_state(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotState:
        episode = self._get_episode_group(episode_index)

        ee_states = torch.from_numpy(np.array(episode["obs"]["ee_states"][frame_index])).float()
        gripper_state = torch.from_numpy(np.array(episode["obs"]["gripper_states"][frame_index][:, :1])).float()
        return self._make_state(ee_states, gripper_state)

    def _load_images(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> Dict[str, torch.Tensor]:
        episode = self._get_episode_group(episode_index)

        return {
            "main": torch.from_numpy(np.array(episode["obs"]["agentview_rgb"][frame_index][:, ::-1].copy())),
            "wrist": torch.from_numpy(np.array(episode["obs"]["eye_in_hand_rgb"][frame_index])[:, ::-1].copy()),
        }

    def _load_instruction(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> List[str]:
        instruction = self._task_instruction(episode_index)
        return [instruction] * len(frame_index)

    def _iter_episode(
        self,
        episode_index: int,
        source_ranges: List[tuple],
        include_images: bool = True,
    ) -> Iterator[Dict]:
        meta = self._metadata[episode_index]

        with h5py.File(meta.extras["data_path"], "r", locking=False) as f:
            data_grp = f["data"]
            demo_idx = int(meta.extras["demo_index"])
            ep_grp = data_grp[f"demo_{demo_idx}"]
            obs = ep_grp["obs"]

            ee_states = torch.from_numpy(np.array(obs["ee_states"])).float()
            grip = torch.from_numpy(np.array(obs["gripper_states"][:, :1])).float()
            actions_raw = torch.from_numpy(np.array(ep_grp["actions"])).float()
            full_state = self._make_state(ee_states, grip)
            full_action = self._make_action(actions_raw, full_state)

            instr = self._task_instruction(episode_index)

            img_main_ds = obs["agentview_rgb"] if include_images else None
            img_wrist_ds = obs["eye_in_hand_rgb"] if include_images else None

            for start, end in source_ranges:
                images = None
                if include_images:
                    images = {
                        "main": torch.from_numpy(img_main_ds[start][::-1].copy()),
                        "wrist": torch.from_numpy(img_wrist_ds[start][::-1].copy()),
                    }
                yield {
                    "state": full_state[start : start + 1],
                    "action": full_action[start:end],
                    "instruction": instr,
                    "images": images,
                }
