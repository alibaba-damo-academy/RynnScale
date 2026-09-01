import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch

from ...constants import RobotType, RotationRepresentation
from ...registry import DATASET_REGISTRY
from ...utils.robot import (
    Arm,
    Position,
    RobotAction,
    RobotState,
    Rotation,
)
from .base import BaseVLADataset, EpisodeMetadata


@DATASET_REGISTRY.register()
class CalvinDataset(BaseVLADataset):
    _BASE_POS = torch.tensor([-0.34, -0.46, 0.24], dtype=torch.float32)
    _SOURCE_FPS = 30.0

    def _load_metadata(self) -> List[EpisodeMetadata]:
        ep_start_end_ids = np.load(os.path.join(self.data_path, "ep_start_end_ids.npy"))
        self._ep_start_end_ids = ep_start_end_ids

        text_data = np.load(
            os.path.join(self.data_path, "lang_annotations", "auto_lang_ann.npy"),
            allow_pickle=True,
        ).item()
        inst_start_end_ids = np.array(text_data["info"]["indx"])
        sorted_indices = inst_start_end_ids[:, 0].argsort()
        self._inst_start_end_ids = inst_start_end_ids[sorted_indices]
        self._instructions = [text_data["language"]["ann"][idx] for idx in sorted_indices]

        out: List[EpisodeMetadata] = []
        for i in range(len(ep_start_end_ids)):
            length = int(ep_start_end_ids[i][1] - ep_start_end_ids[i][0] + 1)
            out.append(
                EpisodeMetadata(
                    length=length,
                    fps=self._SOURCE_FPS,
                    robot_type=RobotType.FRANKA,
                    extras={"episode_start": int(ep_start_end_ids[i][0])},
                )
            )
        return out

    def _frame_path(self, episode_index: int, frame_index: int) -> str:
        global_id = self._metadata[episode_index].extras["episode_start"] + int(frame_index)
        return os.path.join(self.data_path, f"episode_{global_id:07d}.npz")

    def _load_frames(
        self,
        episode_index: int,
        frame_indices: List[int],
        keys: List[str],
    ) -> List[torch.Tensor]:
        def _load_one(fi: int) -> Tuple[np.ndarray, ...]:
            with np.load(self._frame_path(episode_index, fi)) as data:
                return tuple(np.asarray(data[k]) for k in keys)

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(_load_one, frame_indices))

        return [torch.from_numpy(np.stack([r[i] for r in results], axis=0)) for i in range(len(keys))]

    def _make_state(self, robot_obs: torch.Tensor) -> RobotState:
        robot_obs = robot_obs.float()
        gripper = robot_obs[:, 6:7] * 0.5
        ee_pos = robot_obs[:, :3] - self._BASE_POS
        return RobotState(
            left_arm=Arm(
                eef_position=Position(ee_pos),
                eef_rotation=Rotation(robot_obs[:, 3:6], representation=RotationRepresentation.EULER_XYZ),
            ),
            left_gripper=Position(gripper, allow_relative=False),
        )

    def _make_action(self, actions: torch.Tensor) -> RobotAction:
        actions = actions.float()
        gripper = (actions[:, 6:7] + 1.0) * 0.5
        ee_pos = actions[:, :3] - self._BASE_POS
        return RobotAction(
            left_arm=Arm(
                eef_position=Position(ee_pos),
                eef_rotation=Rotation(actions[:, 3:6], representation=RotationRepresentation.EULER_XYZ),
            ),
            left_gripper=Position(gripper, allow_relative=False),
        )

    def _load_action(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotAction:
        (actions,) = self._load_frames(episode_index, frame_index, ["actions"])
        return self._make_action(actions[:, :7])

    def _load_state(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotState:
        (robot_obs,) = self._load_frames(episode_index, frame_index, ["robot_obs"])
        return self._make_state(robot_obs[:, :15])

    def _load_images(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> Dict[str, torch.Tensor]:
        main, wrist = self._load_frames(episode_index, frame_index, ["rgb_static", "rgb_gripper"])
        return {"main": main, "wrist": wrist}

    def _lookup_instruction(self, global_idx: int) -> str:
        if global_idx > self._inst_start_end_ids[-1][1]:
            return ""
        inst_idx = int(np.searchsorted(self._inst_start_end_ids[:, 1], global_idx, side="right"))
        if global_idx < self._inst_start_end_ids[inst_idx][0]:
            return ""
        return self._instructions[inst_idx]

    def _load_instruction(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> List[str]:
        episode_start = self._metadata[episode_index].extras["episode_start"]
        return [self._lookup_instruction(episode_start + int(fi)) for fi in frame_index]

    def _iter_episode(
        self,
        episode_index: int,
        source_ranges: List[tuple],
        include_images: bool = True,
    ) -> Iterator[Dict]:
        meta = self._metadata[episode_index]
        n_total = meta.length
        episode_start = meta.extras["episode_start"]

        keys = ["actions", "robot_obs"]
        if include_images:
            keys += ["rgb_static", "rgb_gripper"]

        all_frames = self._load_frames(episode_index, list(range(n_total)), keys)
        all_actions = self._make_action(all_frames[0][:, :7])
        all_states = self._make_state(all_frames[1][:, :15])

        for start, end in source_ranges:
            images = None
            if include_images:
                images = {
                    "main": all_frames[2][start],
                    "wrist": all_frames[3][start],
                }
            yield {
                "state": all_states[start : start + 1],
                "action": all_actions[start:end],
                "instruction": self._lookup_instruction(episode_start + start),
                "images": images,
            }
