import math
from typing import Dict, List, Optional

import torch
import transformers
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.processing_qwen3_vl import (
    Qwen3VLProcessor as _Qwen3VLProcessor,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import (
    Qwen3VLProcessorKwargs,
)
from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
from transformers.processing_utils import AllKwargsForChatTemplate, MultiModalData, Unpack

from ...utils.processing import load_multimodal_data


def _get_rope_index_qwen3_vl(
    input_ids: torch.LongTensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    image_spatial_merge_size: int,
    video_spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    **kwargs,
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
                    spatial_merge_size = image_spatial_merge_size
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    spatial_merge_size = video_spatial_merge_size
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


class Qwen3VLProcessor(_Qwen3VLProcessor):
    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        chat_template: Optional[str] = None,
        fps: Optional[int] = None,
        max_frames: Optional[int] = None,
        mm_max_length: Optional[int] = None,
        return_labels: bool = False,
        **kwargs: Unpack[AllKwargsForChatTemplate],
    ):
        # `size`/`max_pixels` are image-processing kwargs, not chat-template variables. As of
        # transformers 5.x, `apply_chat_template` requires these to be passed via the dedicated
        # `processor_kwargs` dict; forwarding them through `**kwargs` triggers a warning.
        processor_kwargs = dict(kwargs.pop("processor_kwargs", None) or {})
        for key in ("size", "max_pixels"):
            if key in kwargs:
                processor_kwargs[key] = kwargs.pop(key)

        if mm_max_length is not None:
            assert "max_pixels" not in processor_kwargs and "size" not in processor_kwargs, (
                "Please provide only one of `mm_max_length` and `max_pixels`."
            )
            num_images, num_videos = 0, 0
            for message in conversation:
                for content in message["content"]:
                    if content["type"] == "image":
                        num_images += 1
                    elif content["type"] == "video":
                        num_videos += 1
            processor_kwargs["size"] = {
                # FIXME: add an argument to control `shortest_edge`
                "shortest_edge": self.image_processor.size["shortest_edge"],
                "longest_edge": self._get_max_pixels(
                    num_images=num_images,
                    num_videos=num_videos,
                    mm_max_length=mm_max_length,
                ),
            }

        if not return_labels:
            return super().apply_chat_template(
                conversation,
                chat_template=chat_template,
                processor_kwargs=processor_kwargs,
                **kwargs,
            )

        assert kwargs.pop("return_tensors", None) == "pt", (
            "`return_tensors` must be set to `pt` when `return_labels` is True."
        )
        assert not kwargs.pop("add_generation_prompt", False), (
            "`add_generation_prompt` must be set to False when `return_labels` is True."
        )
        assert kwargs.pop("tokenize", True), "`tokenize` must be set to True when `return_labels` is True."
        assert kwargs.pop("return_dict", False), "`return_dict` must be set to True when `return_labels` is True."
        assert kwargs.pop("do_sample_frames", True), (
            "`do_sample_frames` must be set to True when `return_labels` is True."
        )

        prompt = super().apply_chat_template(
            conversation,
            chat_template=chat_template,
            add_generation_prompt=False,
            tokenize=False,
            processor_kwargs=processor_kwargs,
            **kwargs,
        )

        images, videos, video_metadatas = load_multimodal_data(
            conversation,
            fps=fps,
            max_frames=max_frames,
        )

        model_inputs = self(
            text=prompt,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            do_sample_frames=False,
            return_tensors="pt",
            **processor_kwargs,
            **kwargs,
        )

        start_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        assistant_token_id = self.tokenizer.convert_tokens_to_ids("assistant")

        generation_prompts = [
            self.tokenizer.encode(text, return_tensors="pt")[0]
            for text in ["<|im_start|>assistant\n<think>\n\n</think>\n\n", "<|im_start|>assistant\n"]
        ]

        batch_labels = []
        for i in range(len(model_inputs["input_ids"])):
            input_ids = model_inputs["input_ids"][i]
            start_indices = torch.nonzero(input_ids == start_token_id).squeeze(-1)
            end_indices = torch.nonzero(input_ids == end_token_id).squeeze(-1)
            assert start_indices.size(0) == end_indices.size(0)

            roles = input_ids[start_indices + 1]
            is_assistant_msg = roles == assistant_token_id
            assert is_assistant_msg.any()

            labels = torch.full_like(input_ids, fill_value=-100)
            for msg_idx in range(len(start_indices)):
                if is_assistant_msg[msg_idx]:
                    start_idx, end_idx = start_indices[msg_idx], end_indices[msg_idx]
                    for generation_prompt in generation_prompts:
                        prefix = input_ids[start_idx : start_idx + generation_prompt.size(0)]
                        if prefix.size(-1) == generation_prompt.size(-1) and torch.all(prefix == generation_prompt):
                            start_idx = start_idx + generation_prompt.size(0)
                            break
                    else:
                        raise ValueError("No generation prompt found in assistant message.")
                    labels[start_idx : end_idx + 1] = input_ids[start_idx : end_idx + 1]

            batch_labels.append(labels)

        model_inputs["labels"] = torch.stack(batch_labels, dim=0)

        # Qwen3VL uses timestamp-based mrope position ids, which depend on the tokenized
        # multimodal layout produced above, so compute them here in the processor.
        model_inputs["position_ids"] = _get_rope_index_qwen3_vl(
            input_ids=model_inputs["input_ids"],
            image_grid_thw=model_inputs.get("image_grid_thw"),
            video_grid_thw=model_inputs.get("video_grid_thw"),
            attention_mask=model_inputs.get("attention_mask"),
            image_spatial_merge_size=self.image_processor.merge_size,
            video_spatial_merge_size=self.video_processor.merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
        )

        return model_inputs

    def _get_max_pixels(
        self,
        num_images: int,
        num_videos: int,
        mm_max_length: Optional[int] = None,
    ):
        merge_size = max(self.image_processor.merge_size, self.video_processor.merge_size)
        if num_images > 0:
            merge_size = min(merge_size, self.image_processor.merge_size)
        if num_videos > 0:
            merge_size = min(merge_size, self.video_processor.merge_size)
        factor = self.image_processor.patch_size * merge_size
        return mm_max_length // max(num_images + num_videos, 1) * (factor**2)

    def _get_number_of_video_patches(self, num_frames: int, height: int, width: int, videos_kwargs=None):
        min_pixels = videos_kwargs.get("min_pixels", None) or self.video_processor.size["shortest_edge"]
        max_pixels = videos_kwargs.get("max_pixels", None) or self.video_processor.size["longest_edge"]
        patch_size = videos_kwargs.get("patch_size", None) or self.video_processor.patch_size
        merge_size = videos_kwargs.get("merge_size", None) or self.video_processor.merge_size
        temporal_patch_size = (
            videos_kwargs.get("temporal_patch_size", None) or self.video_processor.temporal_patch_size
        )

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            num_frames=num_frames,
            height=height,
            width=width,
            temporal_factor=temporal_patch_size,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        grid_t = math.ceil(num_frames / temporal_patch_size)
        return grid_t * grid_h * grid_w

    def _get_num_multimodal_tokens(
        self,
        image_sizes=None,
        video_sizes=None,
        mm_max_length: Optional[int] = None,
        **kwargs,
    ):
        if mm_max_length is not None:
            assert "max_pixels" not in kwargs, "Please provide only one of `mm_max_length` and `max_pixels`."
            kwargs["max_pixels"] = self._get_max_pixels(
                num_images=len(image_sizes) if image_sizes is not None else 0,
                num_videos=len(video_sizes) if video_sizes is not None else 0,
                mm_max_length=mm_max_length,
            )

        vision_data = {}
        if image_sizes is not None:
            images_kwargs = Qwen3VLProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get("merge_size", None) or self.image_processor.merge_size

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            num_image_tokens = [(num_patches // merge_size**2) for num_patches in num_image_patches]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        if video_sizes is not None:
            videos_kwargs = Qwen3VLProcessorKwargs._defaults.get("videos_kwargs", {})
            videos_kwargs.update(kwargs)
            merge_size = videos_kwargs.get("merge_size", None) or self.video_processor.merge_size

            fps = kwargs.pop("fps", 1)
            max_frames = kwargs.pop("max_frames", None)
            for video_size in video_sizes:
                num_frames = video_size[0] // fps
                if max_frames is not None:
                    num_frames = min(num_frames, max_frames)
                video_size[0] = num_frames

            num_video_patches = [
                self._get_number_of_video_patches(*video_size, videos_kwargs) for video_size in video_sizes
            ]
            num_video_tokens = [(num_patches // merge_size**2) for num_patches in num_video_patches]
            vision_data["num_video_tokens"] = num_video_tokens

        return MultiModalData(**vision_data)


def apply_monkey_patch():
    transformers.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor = Qwen3VLProcessor
    transformers.models.auto.processing_auto.PROCESSOR_MAPPING[Qwen3VLConfig] = Qwen3VLProcessor
