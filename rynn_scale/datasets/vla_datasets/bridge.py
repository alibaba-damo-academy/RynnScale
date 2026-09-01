import io
import json
import os
from typing import Dict, Iterator, List

import numpy as np
import torch
from PIL import Image

from ...constants import RobotType, RotationRepresentation
from ...registry import DATASET_REGISTRY
from ...utils.robot import Arm, Position, RobotAction, RobotState
from ...utils.robot import Rotation as EefRotation
from .base import BaseVLADataset, EpisodeMetadata
from .droid import INDEX_NAME, _parse_episode, scan_rlds_episodes

SOURCE_FPS = 5.0  # BridgeData V2 was teleoperated at 5 Hz.
BRIDGE_PROBE_KEY = b"steps/reward"  # scalar float32 per step → packed_len // 4 == num_frames.

# Per-step packed FloatList feature keys (name → (out_key, dim_per_step)).
# `action`: [dx, dy, dz, droll, dpitch, dyaw, gripper_abs] — first 6 are
#   per-step deltas in world frame, last entry is an absolute target gripper
#   command in [0, 1].
# `state`:  [x, y, z, roll, pitch, yaw, gripper] EEF proprio.
FLOAT_FIELDS = {
    b"steps/action": ("action", 7),
    b"steps/observation/state": ("state", 7),
}

# Per-step BytesList feature keys (name → out_key).
# `language_instruction` is repeated per-step but identical across steps for an
# episode; the parser only keeps and decodes the first item. image_0 is the
# primary view and always present; image_1/2/3 may carry a dummy fill when the
# per-episode `has_image_X` flag is False — surfaced raw, downstream chooses
# what to use.
BYTES_FIELDS = {
    b"steps/language_instruction": "instruction",
    b"steps/observation/image_0": "image_0",
    b"steps/observation/image_1": "image_1",
    b"steps/observation/image_2": "image_2",
    b"steps/observation/image_3": "image_3",
}


@DATASET_REGISTRY.register()
class BridgeV2Dataset(BaseVLADataset):
    """WidowX BridgeData V2; one TFRecord record = one episode. EEF state is
    [x, y, z, roll, pitch, yaw, gripper] in robot base frame (Euler XYZ); per-step
    action is the 6-DoF EEF delta + absolute gripper command. Streaming-only —
    each episode is a contiguous protobuf chunk, so random-access loaders raise
    ``NotImplementedError`` like droid."""

    def _load_metadata(self) -> List[EpisodeMetadata]:
        index_path = os.path.join(self.data_path, INDEX_NAME)
        if not os.path.isfile(index_path):
            raise FileNotFoundError(
                f"{index_path} not found. Generate it with:\n  python -m {__name__} --root {self.data_path}"
            )
        out: List[EpisodeMetadata] = []
        with open(index_path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                e = json.loads(line)
                path = e["path"]
                if not os.path.isabs(path):
                    path = os.path.join(self.data_path, path)
                out.append(
                    EpisodeMetadata(
                        length=int(e["length"]),
                        fps=SOURCE_FPS,
                        robot_type=RobotType.WIDOWX_250S,
                        extras={"path": path, "offset": int(e["offset"])},
                    )
                )
        assert out, f"No episodes in {index_path}"
        return out

    def _load_action(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotAction:
        raise NotImplementedError("BridgeV2Dataset is streaming-only; use iter_episode / _iter_episode.")

    def _load_state(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotState:
        raise NotImplementedError("BridgeV2Dataset is streaming-only; use iter_episode / _iter_episode.")

    def _load_images(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("BridgeV2Dataset is streaming-only; use iter_episode / _iter_episode.")

    def _load_instruction(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> List[str]:
        raise NotImplementedError("BridgeV2Dataset is streaming-only; use iter_episode / _iter_episode.")

    def _iter_episode(
        self,
        episode_index: int,
        source_ranges: List[tuple],
        include_images: bool = True,
    ) -> Iterator[Dict]:
        meta = self._metadata[episode_index]
        ep = _parse_episode(
            meta.extras["path"],
            meta.extras["offset"],
            FLOAT_FIELDS,
            BYTES_FIELDS,
            include_images=include_images,
        )
        n = ep["num_frames"]

        state_arr = torch.from_numpy(ep["state"]).float()
        action_arr = torch.from_numpy(ep["action"]).float()

        state_pos = state_arr[:, :3]
        state_euler = state_arr[:, 3:6]
        state_grip = state_arr[:, 6:7]

        # Action is stored as per-step world-frame deltas of xyz/euler plus an
        # absolute gripper command. Subclass contract is to return ABSOLUTE
        # action; the base re-derives the delta when use_delta_action=True.
        # Element-wise add for both xyz and euler matches the BridgeData V2
        # data-collection convention (target_pose = cur_pose + action[:6]).
        action_pos = state_pos + action_arr[:, :3]
        action_euler = state_euler + action_arr[:, 3:6]
        action_grip = action_arr[:, 6:7]

        full_state = RobotState(
            left_arm=Arm(
                eef_position=Position(state_pos),
                eef_rotation=EefRotation(
                    state_euler,
                    representation=RotationRepresentation.EULER_XYZ,
                ),
            ),
            left_gripper=Position(state_grip, allow_relative=False),
        )
        full_action = RobotAction(
            left_arm=Arm(
                eef_position=Position(action_pos),
                eef_rotation=EefRotation(
                    action_euler,
                    representation=RotationRepresentation.EULER_XYZ,
                ),
            ),
            left_gripper=Position(action_grip, allow_relative=False),
        )

        instr = ep["instruction"]
        if include_images:
            # image_1/2/3 carry an all-black dummy fill when the corresponding
            # camera view is absent for the episode. The per-episode
            # `has_image_X` flags that are supposed to gate this are unreliable
            # (all False in the TFDS build even for views that clearly exist),
            # so detect real views by content: decode each view's first frame
            # once and keep only those that aren't pure black. The dummy fill is
            # a constant zero image for the whole episode, so the first frame is
            # a sufficient probe; image_0 (primary view) is always present.
            present_blobs = {}
            for key in ("image_0", "image_1", "image_2", "image_3"):
                blob = ep[key]
                if np.asarray(Image.open(io.BytesIO(blob[0]))).max() > 0:
                    present_blobs[key] = blob

        for start, end in source_ranges:
            images = None
            if include_images:
                images = {
                    key: torch.from_numpy(np.asarray(Image.open(io.BytesIO(blob[start]))))
                    for key, blob in present_blobs.items()
                }
            yield {
                "state": full_state[start : start + 1],
                "action": full_action[start:end],
                "instruction": instr,
                "images": images,
            }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Build BridgeV2 episodes.jsonl index (offline scan).",
    )
    ap.add_argument("--root", required=True, help="BridgeV2 data_path (directory containing *.tfrecord-* shards)")
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    episodes = scan_rlds_episodes(
        args.root,
        BRIDGE_PROBE_KEY,
        SOURCE_FPS,
        RobotType.WIDOWX_250S,
        "Scanning BridgeV2",
        max_workers=args.workers,
    )
    output = os.path.join(args.root, INDEX_NAME)
    with open(output, "w") as fout:
        for ep in episodes:
            fout.write(
                json.dumps(
                    {
                        "path": os.path.relpath(ep.extras["path"], args.root),
                        "offset": ep.extras["offset"],
                        "length": ep.length,
                    }
                )
                + "\n"
            )
    print(f"Wrote {len(episodes)} episodes -> {output}")
