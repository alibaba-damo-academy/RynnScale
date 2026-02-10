import numpy as np
from PIL import Image
from transformers import Qwen3VLProcessor, Qwen3VLForConditionalGeneration
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize as image_smart_resize
from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize as video_smart_resize
from transformers.image_utils import load_images
from transformers.image_utils import get_image_size
from transformers.image_transforms import resize
from transformers.feature_extraction_utils import BatchFeature

from .base import BaseInferenceWrapper
from ..registry import INFERENCE_WRAPPER_REGISTRY
from ..utils.processing import load_video
from ..utils import logging


logger = logging.get_logger(__file__)


@INFERENCE_WRAPPER_REGISTRY.register("qwen3_vl")
class Qwen3VLInferenceWrapper(BaseInferenceWrapper):
    def load_model(self):
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            device_map={"": "cuda:0"},
        )
        return model

    def load_processor(self):
        processor = Qwen3VLProcessor.from_pretrained(self.model_path)
        return processor

    def apply_chat_template(self, conversation, enable_thinking):
        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            tokenize=False,
        )
        return prompt

    def load_images(self, images, processing_params):
        factor = self.processor.image_processor.patch_size * self.processor.image_processor.merge_size
        if (min_pixels := processing_params.get("image_min_pixels", None)) is None:
            min_pixels = self.processor.image_processor.size["shortest_edge"]
        if (max_pixels := processing_params.get("image_max_pixels", None)) is None:
            max_pixels = self.processor.image_processor.size["longest_edge"]

        images = load_images(images)
        max_pixels = max_pixels // len(images)

        for i, image in enumerate(images):
            h, w = image_smart_resize(
                image.height,
                image.width,
                factor=factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            images[i] = image.resize(
                size=(w, h),
                resample=self.processor.image_processor.resample,
            )

        return images

    def load_videos(self, videos, processing_params):
        temporal_factor = self.processor.video_processor.temporal_patch_size
        factor = self.processor.video_processor.patch_size * self.processor.video_processor.merge_size

        if (min_pixels := processing_params.get("video_min_pixels", None)) is None:
            min_pixels = self.processor.video_processor.size["shortest_edge"]
        if (max_pixels := processing_params.get("video_max_pixels", None)) is None:
            max_pixels = self.processor.video_processor.size["longest_edge"]
        if (fps := processing_params.get("fps", None)) is None:
            fps = self.processor.video_processor.fps
        if (max_frames := processing_params.get("max_frames", None)) is None:
            max_frames = self.processor.video_processor.max_frames

        max_pixels = max_pixels // len(videos)

        output_videos, video_metadatas = [], []

        for video in videos:
            frames, video_metadata = load_video(video, fps=fps, max_frames=max_frames)
            if isinstance(frames, list) and isinstance(frames[0], Image.Image):
                frames = [np.array(frames) for frames in frames]

            height, width = get_image_size(frames[0])
            resized_height, resized_width = video_smart_resize(
                len(frames),
                height,
                width,
                temporal_factor=temporal_factor,
                factor=factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

            frames = [
                resize(
                    frame,
                    (resized_height, resized_width),
                    resample=self.processor.video_processor.resample,
                )
                for frame in frames
            ]

            output_videos.append(frames)
            video_metadatas.append(video_metadata)

        return output_videos, video_metadatas

    def process_images(self, images, processing_params):
        return self.processor.image_processor(
            images=images,
            return_tensors="pt",
        )

    def process_videos(self, videos, processing_params):
        videos, video_metadatas = videos
        return self.processor.video_processor(
            videos=videos,
            video_metadata=video_metadatas,
            do_sample_frames=False,
            return_metadata=True,
            return_tensors="pt",
        )

    def process_text(self, text, image_inputs={}, video_inputs={}):
        image_grid_thw = image_inputs.get("image_grid_thw", [])
        video_grid_thw = video_inputs.get("video_grid_thw", [])
        video_metadata = video_inputs.get("video_metadata", [])

        merge_length = self.processor.image_processor.merge_size**2
        image_index = 0
        while self.processor.image_token in text:
            num_image_tokens = image_grid_thw[image_index].prod() // merge_length
            text = text.replace(self.processor.image_token, "<|placeholder|>" * num_image_tokens, 1)
            image_index += 1
        text = text.replace("<|placeholder|>", self.processor.image_token)

        merge_length = self.processor.video_processor.merge_size**2
        video_index = 0
        while self.processor.video_token in text:
            metadata = video_metadata[video_index]
            if metadata.fps is None:
                logger.warning_once(
                    "Qwen3VL requires frame timestamps to construct prompts, but the `fps` of the input video could not be inferred. "
                    "Probably `video_metadata` was missing from inputs and you passed pre-sampled frames. "
                    "Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results."
                )
                metadata.fps = 24 if metadata.fps is None else metadata.fps

            # if timestamps are not provided, calculate them
            curr_timestamp = self.processor._calculate_timestamps(
                metadata.frames_indices,
                metadata.fps,
                self.processor.video_processor.merge_size,
            )

            video_placeholder = ""
            frame_seqlen = video_grid_thw[video_index][1:].prod() // merge_length
            for frame_idx in range(video_grid_thw[video_index][0]):
                curr_time = curr_timestamp[frame_idx]
                video_placeholder += f"<{curr_time:.1f} seconds>"
                video_placeholder += (
                    self.processor.vision_start_token
                    + "<|placeholder|>" * frame_seqlen
                    + self.processor.vision_end_token
                )
            if (
                f"{self.processor.vision_start_token}{self.processor.video_token}{self.processor.vision_end_token}"
                in text
            ):
                text = text.replace(
                    f"{self.processor.vision_start_token}{self.processor.video_token}{self.processor.vision_end_token}",
                    video_placeholder,
                    1,
                )
            else:
                # vllm may input video token directly
                text = text.replace(self.processor.video_token, video_placeholder, 1)
            video_index += 1

        text = text.replace("<|placeholder|>", self.processor.video_token)

        text_inputs = self.processor.tokenizer(text, return_tensors="pt")
        self.processor._check_special_mm_tokens([text], text_inputs, modalities=["image", "video"])

        model_inputs = {
            **text_inputs,
            **image_inputs,
            **{k: v for k, v in video_inputs.items() if k != "video_metadata"},
        }
        return BatchFeature(model_inputs, tensor_type="pt")

    def generate(self, model_inputs, sampling_params):
        output_ids = self.model.generate(
            **model_inputs,
            **sampling_params,
        )
        output_ids = output_ids[:, model_inputs["input_ids"].size(1) :]
        texts = self.processor.post_process_image_text_to_text(output_ids)
        return texts
