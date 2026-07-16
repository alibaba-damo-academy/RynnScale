import math
from typing import Dict, List, Optional

import torch
import transformers
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.processing_qwen3_vl import (
    Qwen3VLProcessor,
    Qwen3VLProcessorKwargs,
)
from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
from transformers.processing_utils import AllKwargsForChatTemplate, MultiModalData, Unpack

from ...utils.processing import load_multimodal_data


class _Qwen3VLProcessor(Qwen3VLProcessor):
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
        if mm_max_length is not None:
            assert "max_pixels" not in kwargs and "size" not in kwargs, (
                "Please provide only one of `mm_max_length` and `max_pixels`."
            )
            num_images, num_videos = 0, 0
            for message in conversation:
                for content in message["content"]:
                    if content["type"] == "image":
                        num_images += 1
                    elif content["type"] == "video":
                        num_videos += 1
            kwargs["size"] = {
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
        assert kwargs.pop("do_sample_frames", True), "`do_sample_frames` must be set to True when `return_labels` is True."

        prompt = super().apply_chat_template(
            conversation,
            chat_template=chat_template,
            add_generation_prompt=False,
            tokenize=False,
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
                    labels[start_idx:end_idx + 1] = input_ids[start_idx:end_idx + 1]

            batch_labels.append(labels)

        model_inputs["labels"] = torch.stack(batch_labels, dim=0)

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
    transformers.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor = _Qwen3VLProcessor
    transformers.models.auto.processing_auto.PROCESSOR_MAPPING[Qwen3VLConfig] = _Qwen3VLProcessor
