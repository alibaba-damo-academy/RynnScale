import math
from collections import defaultdict
from typing import Optional, List, Dict

import torch
import transformers
from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.processing_qwen3_vl import (
    Qwen3VLProcessor as _Qwen3VLProcessor,
    Qwen3VLProcessorKwargs,
)
from transformers.processing_utils import AllKwargsForChatTemplate, Unpack, BatchFeature, MultiModalData

from ...utils.processing import load_multimodal_data


class Qwen3VLProcessor(_Qwen3VLProcessor):
    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        chat_template: Optional[str] = None,
        mm_max_length: Optional[int] = None,
        return_labels: bool = False,
        **kwargs: Unpack[AllKwargsForChatTemplate],
    ):
        if return_labels:
            assert kwargs.get("return_tensors", None) == "pt", (
                "`return_tensors` must be set to `pt` when `return_labels` is True."
            )
            assert not kwargs.get("add_generation_prompt", False), (
                "`add_generation_prompt` must be set to False when `return_labels` is True."
            )
            assert kwargs.get("tokenize", True), "`tokenize` must be set to True when `return_labels` is True."
            assert kwargs.get("return_dict", False), "`return_dict` must be set to True when `return_labels` is True."

            pseudo_message = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
            prompt_tokens = super().apply_chat_template(
                pseudo_message, chat_template=chat_template, tokenize=True, add_generation_prompt=False
            )[0]
            conv_tokens = super().apply_chat_template(
                pseudo_message, chat_template=chat_template, tokenize=True, add_generation_prompt=True
            )[0]
            prompt_length = len(conv_tokens) - len(prompt_tokens)

            ignore_tokens = torch.as_tensor(
                [self.image_token_id, self.video_token_id, self.vision_start_token_id, self.vision_end_token_id]
            )[None, None]

        fps = kwargs.pop("fps", 1)
        max_frames = kwargs.pop("max_frames", None)
        tokenize = kwargs.pop("tokenize", True)
        return_dict = kwargs.pop("return_dict", False)
        return_tensors = kwargs.pop("return_tensors", None)
        add_generation_prompt = kwargs.pop("add_generation_prompt", False)
        kwargs.pop("do_sample_frames", False)

        if tokenize and return_dict:
            conversation = load_multimodal_data(
                conversation,
                fps=fps,
                max_frames=max_frames,
            )

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

        outputs = defaultdict(list)

        for i, message in enumerate(conversation):
            prompt = super().apply_chat_template(
                [message],
                chat_template=chat_template,
                tokenize=False,
                add_generation_prompt=add_generation_prompt and i == len(conversation) - 1,
            )

            if tokenize and return_dict:
                images, videos, video_metadatas = [], [], []
                if message["role"] != "assistant":
                    for content in message["content"]:
                        if content["type"] == "image":
                            images.append(content["image"])
                        elif content["type"] == "video":
                            videos.append(content["video"][0])
                            video_metadatas.append(content["video"][1])

                results = self(
                    text=prompt,
                    images=images if len(images) > 0 else None,
                    videos=videos if len(videos) > 0 else None,
                    video_metadata=video_metadatas if len(videos) > 0 else None,
                    return_tensors="pt",
                    do_sample_frames=False,
                    **kwargs,
                )

                if return_labels:
                    labels = torch.full_like(results["input_ids"], fill_value=-100, dtype=torch.long)
                    if message["role"] == "assistant":
                        valid_mask = torch.all(results["input_ids"][..., None] != ignore_tokens, dim=-1)
                        # prefix: <|im_start|>assistant\n
                        valid_mask[:, :prompt_length] = False
                        # postfix: \n
                        valid_mask[:, -1] = False
                        labels[valid_mask] = results["input_ids"][valid_mask]
                    results["labels"] = labels

                for key, value in results.items():
                    outputs[key].append(value)

            else:
                outputs["prompts"].append(prompt)

        if tokenize:
            mm_input_names = set(self.image_processor.model_input_names + self.video_processor.model_input_names)
            for k, v in outputs.items():
                if k in mm_input_names:
                    outputs[k] = torch.cat(v, dim=0)
                else:
                    outputs[k] = torch.cat(v, dim=1)
            outputs = BatchFeature(outputs, tensor_type=return_tensors)
            if return_dict:
                return outputs
            return outputs["input_ids"]

        return "".join(outputs["prompts"])

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
