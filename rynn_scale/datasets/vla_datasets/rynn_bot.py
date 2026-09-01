import json
import os
from typing import Dict, Iterator, List

import torch
from datasets import load_dataset
from tqdm import tqdm

from ...constants import RobotType, RotationRepresentation
from ...registry import DATASET_REGISTRY
from ...utils.robot import Arm, Position, RobotAction, RobotState, Rotation
from .base import BaseVLADataset, EpisodeMetadata
from .utils import SequentialVideoReader, VideoReader, fork_safe_cache, gather_column, suppress_hf_progress

__all__ = ["RynnBotFrankaDataset", "RynnBotMarvinWujiDataset"]


CAMERA_KEYS = ("cam_main", "cam_side", "cam_arm")
MARVIN_WUJI_CAMERA_KEYS = ("head", "left_wrist", "right_wrist")


def _read_video_window(path: str, indices: List[int]):
    """Read specific frames from an mp4 via PyAV. Returns (T, H, W, 3) uint8 numpy."""
    import av
    import numpy as np

    container = av.open(path)
    stream = container.streams.video[0]
    stream.codec_context.thread_type = "AUTO"

    frames_out = []
    target_set = set(indices)
    idx = 0
    for frame in container.decode(stream):
        if idx in target_set:
            frames_out.append(frame.to_ndarray(format="rgb24"))
        idx += 1
        if len(frames_out) == len(indices):
            break
    container.close()

    return np.stack(frames_out, axis=0)


@DATASET_REGISTRY.register()
class RynnBotFrankaDataset(BaseVLADataset):
    """Single-arm Franka; one directory per episode."""

    def _load_metadata(self) -> List[EpisodeMetadata]:
        episode_dirs = sorted(
            os.path.join(self.data_path, d)
            for d in os.listdir(self.data_path)
            if d.startswith("episode_") and os.path.isdir(os.path.join(self.data_path, d))
        )
        assert episode_dirs, f"No episode_* directories found under {self.data_path}"

        out: List[EpisodeMetadata] = []
        for ep_dir in tqdm(episode_dirs, desc="Loading RynnBot episodes"):
            meta_path = os.path.join(ep_dir, "metadata.json")
            with open(meta_path, "r") as f:
                meta = json.load(f)
            n = int(meta.get("total_frames", 0))
            if n <= 0:
                continue
            out.append(
                EpisodeMetadata(
                    length=n,
                    fps=float(meta.get("fps", 30)),
                    robot_type=RobotType.FRANKA,
                    extras={
                        "ep_dir": ep_dir,
                        "prompt": meta["task_prompt"],
                    },
                )
            )
        assert out, f"No non-empty episodes under {self.data_path}"
        return out

    @fork_safe_cache
    def _get_parquet(self, parquet_path: str):
        with suppress_hf_progress():
            return load_dataset(
                "parquet",
                data_files=parquet_path,
                split="train",
            ).with_format("torch")

    def _parquet(self, episode_index: int):
        path = os.path.join(self._metadata[episode_index].extras["ep_dir"], "timeseries.parquet")
        return self._get_parquet(path)

    def _video_path(self, episode_index: int, cam_key: str) -> str:
        return os.path.join(
            self._metadata[episode_index].extras["ep_dir"],
            f"observation.images.{cam_key}.mp4",
        )

    def _load_action(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotAction:
        ds = self._parquet(episode_index)
        joint = gather_column(ds, "action.arm", frame_index)
        gripper = gather_column(ds, "action.gripper", frame_index)
        return RobotAction(
            left_arm=Arm(joint_position=Position(joint)),
            left_gripper=Position(gripper, allow_relative=False),
        )

    def _load_state(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotState:
        ds = self._parquet(episode_index)
        joint = gather_column(ds, "observation.state.arm", frame_index)
        gripper = gather_column(ds, "observation.state.gripper", frame_index)
        ee_pose = gather_column(ds, "observation.state.ee_pose", frame_index)
        return RobotState(
            left_arm=Arm(
                joint_position=Position(joint),
                eef_position=Position(ee_pose[:, :3]),
                eef_rotation=Rotation(ee_pose[:, 3:6], representation=RotationRepresentation.EULER_XYZ),
            ),
            left_gripper=Position(gripper, allow_relative=False),
        )

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
        out: Dict[str, torch.Tensor] = {}
        for cam_key in CAMERA_KEYS:
            reader = self._get_video_reader(self._video_path(episode_index, cam_key))
            if reader is None:
                continue
            out[cam_key] = reader.read(frame_index)
        return out

    def _load_instruction(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> List[str]:
        return [self._metadata[episode_index].extras["prompt"]] * len(frame_index)

    def _iter_episode(
        self,
        episode_index: int,
        source_ranges: List[tuple],
        include_images: bool = True,
    ) -> Iterator[Dict]:
        meta = self._metadata[episode_index]
        n_total = meta.length
        prompt = meta.extras["prompt"]

        ds = self._parquet(episode_index)
        all_idx = list(range(n_total))
        joint_a = gather_column(ds, "action.arm", all_idx)
        grip_a = gather_column(ds, "action.gripper", all_idx)
        joint_s = gather_column(ds, "observation.state.arm", all_idx)
        grip_s = gather_column(ds, "observation.state.gripper", all_idx)
        ee_s = gather_column(ds, "observation.state.ee_pose", all_idx)

        full_action = RobotAction(
            left_arm=Arm(joint_position=Position(joint_a)),
            left_gripper=Position(grip_a, allow_relative=False),
        )
        full_state = RobotState(
            left_arm=Arm(
                joint_position=Position(joint_s),
                eef_position=Position(ee_s[:, :3]),
                eef_rotation=Rotation(ee_s[:, 3:6], representation=RotationRepresentation.EULER_XYZ),
            ),
            left_gripper=Position(grip_s, allow_relative=False),
        )

        readers = {}
        if include_images:
            for cam_key in CAMERA_KEYS:
                path = self._video_path(episode_index, cam_key)
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
                    "instruction": prompt,
                    "images": images,
                }
        finally:
            for r in readers.values():
                r.close()


@DATASET_REGISTRY.register()
class RynnBotMarvinWujiDataset(BaseVLADataset):
    """Dual-arm dexterous-hand TianJi WuJi; one directory per episode."""

    def _load_metadata(self) -> List[EpisodeMetadata]:
        episode_dirs = sorted(
            os.path.join(self.data_path, d)
            for d in os.listdir(self.data_path)
            if d.startswith("episode_") and os.path.isdir(os.path.join(self.data_path, d))
        )
        assert episode_dirs, f"No episode_* directories found under {self.data_path}"

        out: List[EpisodeMetadata] = []
        for ep_dir in tqdm(episode_dirs, desc="Loading RynnBot TianJi WuJi episodes"):
            meta_path = os.path.join(ep_dir, "metadata.json")
            with open(meta_path, "r") as f:
                meta = json.load(f)
            n = int(meta.get("total_frames", 0))
            if n <= 0:
                continue
            out.append(
                EpisodeMetadata(
                    length=n,
                    fps=float(meta.get("fps", 30)),
                    robot_type=RobotType.MARVIN_WUJI,
                    extras={
                        "ep_dir": ep_dir,
                        "prompt": meta["task_prompt"],
                    },
                )
            )
        assert out, f"No non-empty episodes under {self.data_path}"
        return out

    @fork_safe_cache
    def _get_parquet(self, parquet_path: str):
        with suppress_hf_progress():
            return load_dataset(
                "parquet",
                data_files=parquet_path,
                split="train",
            ).with_format("torch")

    def _parquet(self, episode_index: int):
        path = os.path.join(self._metadata[episode_index].extras["ep_dir"], "timeseries.parquet")
        return self._get_parquet(path)

    def _video_path(self, episode_index: int, cam_key: str) -> str:
        return os.path.join(
            self._metadata[episode_index].extras["ep_dir"],
            f"observation.images.{cam_key}.mp4",
        )

    def _load_action(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotAction:
        ds = self._parquet(episode_index)
        arm_left = gather_column(ds, "action.arm_left", frame_index)
        arm_right = gather_column(ds, "action.arm_right", frame_index)
        hand_left = gather_column(ds, "action.hand_left", frame_index)
        hand_right = gather_column(ds, "action.hand_right", frame_index)
        return RobotAction(
            left_arm=Arm(joint_position=Position(arm_left)),
            right_arm=Arm(joint_position=Position(arm_right)),
            left_hand=Position(hand_left, allow_relative=False),
            right_hand=Position(hand_right, allow_relative=False),
        )

    def _load_state(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotState:
        ds = self._parquet(episode_index)
        arm_left = gather_column(ds, "observation.state.arm_pos_left", frame_index)
        arm_right = gather_column(ds, "observation.state.arm_pos_right", frame_index)
        hand_left = gather_column(ds, "observation.state.hand_pos_left", frame_index)
        hand_right = gather_column(ds, "observation.state.hand_pos_right", frame_index)
        return RobotState(
            left_arm=Arm(
                joint_position=Position(arm_left),
            ),
            right_arm=Arm(
                joint_position=Position(arm_right),
            ),
            left_hand=Position(hand_left),
            right_hand=Position(hand_right),
        )

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
        out: Dict[str, torch.Tensor] = {}
        for cam_key in MARVIN_WUJI_CAMERA_KEYS:
            reader = self._get_video_reader(self._video_path(episode_index, cam_key))
            if reader is None:
                continue
            out[cam_key] = reader.read(frame_index)
        return out

    def _load_instruction(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> List[str]:
        return [self._metadata[episode_index].extras["prompt"]] * len(frame_index)

    def _iter_episode(
        self,
        episode_index: int,
        source_ranges: List[tuple],
        include_images: bool = True,
    ) -> Iterator[Dict]:
        meta = self._metadata[episode_index]
        n_total = meta.length
        prompt = meta.extras["prompt"]

        ds = self._parquet(episode_index)
        all_idx = list(range(n_total))
        full_action = self._load_action(episode_index, all_idx)
        full_state = self._load_state(episode_index, all_idx)

        readers = {}
        if include_images:
            for cam_key in MARVIN_WUJI_CAMERA_KEYS:
                path = self._video_path(episode_index, cam_key)
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
                    "instruction": prompt,
                    "images": images,
                }
        finally:
            for r in readers.values():
                r.close()
