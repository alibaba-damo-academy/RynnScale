from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Union

import torch
from transformers import BatchFeature
from transformers.image_utils import ImageInput


class BaseInferenceWrapper(object, metaclass=ABCMeta):
    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype,
        attn_implementation: str,
    ):
        self.model_path = model_path
        self.dtype = dtype
        self.attn_implementation = attn_implementation

        self._model = None
        self._processor = None

    @property
    def model(self):
        if self._model is None:
            self._model = self.load_model()
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            self._processor = self.load_processor()
        return self._processor

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def load_processor(self):
        pass

    @abstractmethod
    def apply_chat_template(self, conversation: Dict[str, Any], enable_thinking: bool) -> str:
        pass

    @abstractmethod
    def load_images(
        self,
        images: ImageInput,
        processing_params: Dict[str, Any],
    ):
        pass

    @abstractmethod
    def load_videos(
        self,
        videos: Union[List[str], List[List[str]]],
        processing_params: Dict[str, Any],
    ):
        pass

    @abstractmethod
    def process_images(
        self,
        images: ImageInput,
        processing_params: Dict[str, Any],
    ):
        pass

    @abstractmethod
    def process_videos(
        self,
        videos: Union[List[str], List[List[str]]],
        processing_params: Dict[str, Any],
    ):
        pass

    @abstractmethod
    def process_text(
        self,
        text: str,
        image_inputs: Dict[str, Any],
        video_inputs: Dict[str, Any],
    ) -> BatchFeature:
        pass

    @abstractmethod
    def generate(
        self,
        model_inputs: Dict[str, Any],
        sampling_params: Dict[str, Any],
    ) -> List[str]:
        pass
