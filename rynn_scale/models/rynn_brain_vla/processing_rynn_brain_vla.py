from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_transforms import resize
from transformers.image_utils import get_image_size
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

from ...constants import RobotType, RotationRepresentation
from ...utils.robot import Arm, Position, RobotAction, RobotState, Rotation

ACTION_LAYOUT: Dict[Union[str, Tuple[str, str]], Tuple[int, int]] = {
    ("left_arm", "joint_position"): (0, 7),
    ("right_arm", "joint_position"): (7, 14),
    ("left_arm", "eef_position"): (14, 17),
    ("left_arm", "eef_rotation"): (17, 23),
    ("right_arm", "eef_position"): (23, 26),
    ("right_arm", "eef_rotation"): (26, 32),
    "left_gripper": (32, 33),
    "right_gripper": (33, 34),
    "left_hand": (34, 54),
    "right_hand": (54, 74),
    "torso": (74, 77),
    "head": (77, 80),
}
ACTION_DIM = 80


def _orthogonalize_rot_6d(tensor: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt orthogonalization for 6D rotation vectors (Zhou et al. 2019).

    This codebase stores rot_6d in **interleaved** layout produced by
    ``_matrix_to_rotation_6d``: [col0[0], col1[0], col0[1], col1[1], col0[2], col1[2]].
    """
    a1 = tensor[..., 0:6:2]
    a2 = tensor[..., 1:7:2]
    e1 = F.normalize(a1, dim=-1)
    e2 = a2 - (e1 * a2).sum(dim=-1, keepdim=True) * e1
    e2 = F.normalize(e2, dim=-1)
    out = torch.stack([e1, e2], dim=-1).flatten(-2)
    return out


def _any_leaf_data(action: Union[RobotAction, RobotState]) -> torch.Tensor:
    """Find any populated leaf tensor for dtype/device reference."""
    for _, field_value in action._fields():
        if isinstance(field_value, (Position, Rotation)):
            return field_value.data
        if isinstance(field_value, Arm):
            for _, sub_value in field_value._fields():
                return sub_value.data
    raise ValueError("RobotAction has no populated leaf field")


def get_rope_index(
    input_ids: torch.LongTensor,
    image_grid_thw: Optional[torch.LongTensor],
    video_grid_thw: Optional[torch.LongTensor],
    attention_mask: Optional[torch.Tensor],
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
) -> torch.Tensor:
    """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

    # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids


class RynnBrainVLAProcessor(Qwen3VLProcessor):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        schema=None,
        use_state=True,
        resolution=384,
        **kwargs,
    ):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs,
        )

        self.schema = schema
        self.use_state = use_state
        self.resolution = resolution

        if schema is not None:
            self._validate_rotation_repr(schema)

        self.state_token = "<|state_pad|>"
        self.tokenizer.add_tokens([self.state_token], special_tokens=True)
        self.state_token_id = self.tokenizer.convert_tokens_to_ids(self.state_token)

    def _validate_rotation_repr(self, schema: Dict) -> None:
        for section in ("action", "state"):
            if section not in schema:
                continue
            for robot_type, robot_schema in schema[section].items():
                self._check_rotation_leaves(robot_schema, path=f"{section}.{robot_type}")

    def _check_rotation_leaves(self, node: Dict, path: str) -> None:
        if node.get("type") == "Rotation":
            if node.get("representation") != "rot_6d":
                raise ValueError(
                    f"Schema rotation repr mismatch at '{path}': expected 'rot_6d', got '{node.get('representation')}'"
                )
            return
        for key, value in node.items():
            if isinstance(value, dict):
                self._check_rotation_leaves(value, path=f"{path}.{key}")

    def _process_image(self, image: Any):
        image = np.array(image)
        height, width = get_image_size(image)

        scale = self.resolution / min(height, width)
        target_size = (round(height * scale), round(width * scale))

        image = resize(image, target_size, resample=Image.Resampling.BICUBIC, return_numpy=False)
        return image

    def _process_action(
        self,
        action: Union[RobotAction, RobotState],
        schema_subtree: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flatten ``action`` into a fixed-layout tensor using ``schema_subtree``.

        The schema's stats drive normalization; the schema's field set drives
        which slots are filled. Returns ``(tensor, mask)`` with shape
        ``(chunk_size, ACTION_DIM)``; ``mask`` is True at filled positions.
        """
        action = action.convert_rotation(RotationRepresentation.ROT_6D)
        action = action.normalize(schema_subtree, norm_type="mean_std")

        chunk_size = len(action)
        ref = _any_leaf_data(action)
        out = torch.zeros(chunk_size, ACTION_DIM, dtype=ref.dtype, device=ref.device)
        mask = torch.zeros(chunk_size, ACTION_DIM, dtype=torch.bool, device=ref.device)

        # Pre-fill identity rotation into both eef_rotation slots so unused
        # slots stay valid in the configured representation.
        identity_rot = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=ref.dtype, device=ref.device)
        for arm_name in ("left_arm", "right_arm"):
            s, _ = ACTION_LAYOUT[(arm_name, "eef_rotation")]
            out[:, s : s + identity_rot.numel()] = identity_rot

        def _fill(slot: Tuple[int, int], value: torch.Tensor) -> None:
            start, end = slot
            d = value.size(1)
            assert d <= end - start, f"dim {d} exceeds slot size {end - start}"
            out[:, start : start + d] = value
            mask[:, start : start + d] = True

        for arm_name in ("left_arm", "right_arm"):
            arm = getattr(action, arm_name)
            if arm is None:
                continue
            # Prefer eef pose over joint position when both are present;
            # the joint slot stays as padding in that case.
            if arm.eef_position is not None:
                _fill(ACTION_LAYOUT[(arm_name, "eef_position")], arm.eef_position.data)
                _fill(ACTION_LAYOUT[(arm_name, "eef_rotation")], arm.eef_rotation.data)
            elif arm.joint_position is not None:
                _fill(ACTION_LAYOUT[(arm_name, "joint_position")], arm.joint_position.data)

        for name in ("left_gripper", "right_gripper", "left_hand", "right_hand", "torso", "head"):
            value = getattr(action, name)
            if value is not None:
                _fill(ACTION_LAYOUT[name], value.data)

        return out, mask

    def __call__(
        self,
        text: str,
        images: Dict[str, Any],
        robot_type: RobotType,
        action: Optional[RobotAction] = None,
        state: Optional[RobotState] = None,
        visual_instruction: Optional[Any] = None,
        return_tensors: str = "pt",
    ):
        image_list = []

        contents = [{"type": "text", "text": "INSTRUCTION:\n"}]
        if visual_instruction is not None:
            image_list.append(self._process_image(visual_instruction))
            contents.append({"type": "image"})
        contents.append({"type": "text", "text": text})

        contents.append({"type": "text", "text": "\n\nOBSERVATION:\n"})
        for key in sorted([k for k in images]):
            image_list.append(self._process_image(images[key]))
            contents.append({"type": "image"})

        if self.use_state:
            contents.append({"type": "text", "text": f"\n\nSTATE:\n{self.state_token}"})

        contents.append({"type": "text", "text": "\n\nWhat action should the robot take?"})
        conversation = [{"role": "user", "content": contents}]

        text = self.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        model_inputs = super().__call__(
            text=text,
            images=image_list,
            return_tensors="pt",
        )

        model_inputs["position_ids"] = get_rope_index(
            input_ids=model_inputs["input_ids"],
            image_grid_thw=model_inputs.get("image_grid_thw", None),
            video_grid_thw=model_inputs.get("video_grid_thw", None),
            attention_mask=model_inputs.get("attention_mask", None),
            spatial_merge_size=self.image_processor.merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
        )

        if action is not None:
            action_schema = self.schema["action"][robot_type.value]
            action_tensor, action_mask = self._process_action(action, action_schema)
            model_inputs["actions"] = action_tensor.unsqueeze(0)
            model_inputs["action_mask"] = action_mask.unsqueeze(0)

        if state is not None:
            state_schema = self.schema["state"][robot_type.value]
            state_tensor, _ = self._process_action(state, state_schema)
            model_inputs["states"] = state_tensor.unsqueeze(0)

        model_inputs = BatchFeature(
            model_inputs,
            tensor_type=return_tensors,
        )

        return model_inputs

    def get_action_mask(self, robot_type: RobotType, chunk_size: int) -> torch.Tensor:
        """Build the action_mask for a given robot type without needing action data."""
        action_schema = self.schema["action"][robot_type.value]
        mask = torch.zeros(chunk_size, ACTION_DIM, dtype=torch.bool)

        for arm_name in ("left_arm", "right_arm"):
            if arm_name not in action_schema:
                continue
            arm_schema = action_schema[arm_name]
            if "eef_position" in arm_schema:
                s, e = ACTION_LAYOUT[(arm_name, "eef_position")]
                mask[:, s : s + arm_schema["eef_position"]["dim"]] = True
                s, e = ACTION_LAYOUT[(arm_name, "eef_rotation")]
                mask[:, s : s + 6] = True
            elif "joint_position" in arm_schema:
                s, e = ACTION_LAYOUT[(arm_name, "joint_position")]
                mask[:, s : s + arm_schema["joint_position"]["dim"]] = True

        for name in ("left_gripper", "right_gripper", "left_hand", "right_hand", "torso", "head"):
            if name not in action_schema:
                continue
            s, _ = ACTION_LAYOUT[name]
            mask[:, s : s + action_schema[name]["dim"]] = True

        return mask

    def get_config_overrides(self):
        return {
            "action_dim": ACTION_DIM,
            "state_token_id": self.state_token_id,
        }

    def post_process(
        self,
        action: torch.Tensor,
        robot_type: RobotType,
        state: Optional[RobotState] = None,
    ) -> RobotAction:
        """Reverse of :meth:`_process_action`.

        The action's structure (which arms / joint-vs-eef / per-field dims and
        ``is_relative`` flags) is read from the schema rather than inferred
        from a state tensor. The optional ``state`` is only used to pick the
        final eef rotation representation; without it the rotation stays in
        ``self.eef_rotation_repr``.
        """
        assert action.ndim == 2

        eef_repr = RotationRepresentation.ROT_6D
        action_schema = self.schema["action"][robot_type.value]
        kwargs: Dict = {}

        for arm_name in ("left_arm", "right_arm"):
            if arm_name not in action_schema:
                continue
            arm_schema = action_schema[arm_name]
            arm_kwargs: Dict = {}

            # Mirror the eef-over-joint preference used in _process_action.
            if "eef_position" in arm_schema:
                pos_leaf = arm_schema["eef_position"]
                ps, _ = ACTION_LAYOUT[(arm_name, "eef_position")]
                arm_kwargs["eef_position"] = Position(
                    data=action[:, ps : ps + pos_leaf["dim"]],
                    is_relative=pos_leaf["is_relative"],
                    allow_relative=pos_leaf["allow_relative"],
                )
                rot_leaf = arm_schema["eef_rotation"]
                rs, _ = ACTION_LAYOUT[(arm_name, "eef_rotation")]
                arm_kwargs["eef_rotation"] = Rotation(
                    data=action[:, rs : rs + eef_repr.dim],
                    representation=eef_repr,
                    is_relative=rot_leaf["is_relative"],
                    allow_relative=rot_leaf["allow_relative"],
                )
            elif "joint_position" in arm_schema:
                joint_leaf = arm_schema["joint_position"]
                s, _ = ACTION_LAYOUT[(arm_name, "joint_position")]
                arm_kwargs["joint_position"] = Position(
                    data=action[:, s : s + joint_leaf["dim"]],
                    is_relative=joint_leaf["is_relative"],
                    allow_relative=joint_leaf["allow_relative"],
                )

            kwargs[arm_name] = Arm(**arm_kwargs)

        for name, (s, _) in ACTION_LAYOUT.items():
            if not isinstance(name, str):
                continue
            if name not in action_schema:
                continue
            leaf = action_schema[name]
            kwargs[name] = Position(
                data=action[:, s : s + leaf["dim"]],
                is_relative=leaf["is_relative"],
                allow_relative=leaf["allow_relative"],
            )

        formatted_action = RobotAction(**kwargs)
        formatted_action = formatted_action.denormalize(action_schema, norm_type="mean_std")

        for arm_name in ("left_arm", "right_arm"):
            arm = getattr(formatted_action, arm_name)
            if arm is not None and arm.eef_rotation is not None:
                arm.eef_rotation.data = _orthogonalize_rot_6d(arm.eef_rotation.data)

        if state is not None:
            formatted_action = formatted_action + state

            target_repr: Optional[RotationRepresentation] = None
            for arm_name in ("left_arm", "right_arm"):
                state_arm = getattr(state, arm_name, None)
                if state_arm is not None and state_arm.eef_rotation is not None:
                    target_repr = state_arm.eef_rotation.representation
                    break
            if target_repr is not None:
                formatted_action = formatted_action.convert_rotation(target_repr)

        return formatted_action
