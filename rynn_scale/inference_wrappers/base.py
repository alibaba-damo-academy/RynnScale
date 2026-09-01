from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from accelerate.hooks import ModelHook, add_hook_to_module
from transformers import BatchFeature
from transformers.image_utils import ImageInput


class CudaDeviceHook(ModelHook):
    def __init__(self, execution_device: torch.device):
        self.execution_device = execution_device
        self._prev_device = None

    def pre_forward(self, module, *args, **kwargs):
        self._prev_device = torch.cuda.current_device()
        torch.cuda.set_device(self.execution_device)
        return args, kwargs

    def post_forward(self, module, output):
        torch.cuda.set_device(self._prev_device)
        return output


def restore_cuda_device_hook(module: torch.nn.Module) -> None:
    device = next(module.parameters()).device
    if device.type == "cuda":
        add_hook_to_module(module, CudaDeviceHook(device))


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


class BaseVLMInferenceWrapper(BaseInferenceWrapper, metaclass=ABCMeta):
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


class BaseVLAInferenceWrapper(BaseInferenceWrapper, metaclass=ABCMeta):
    @abstractmethod
    def process(
        self,
        text: str,
        images: Dict[str, ImageInput],
        state: Dict[str, Any],
        robot_type: str,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def collate(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def prefill(self, model_inputs: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def decode(
        self,
        model_inputs: Dict[str, Any],
        cache: Dict[str, Any],
        num_steps: int,
        robot_type: List[str],
        prev_actions: List[Optional[np.ndarray]],
        delay_steps: List[int],
    ) -> torch.Tensor:
        """Sample one action chunk per batch element.

        Everything the request carried arrives here, and **per element** wherever it can
        differ between callers -- ``@serve.batch`` groups independent ones, so a single
        value would be one caller's served to all of them. Only ``num_steps`` is scalar
        (it schedules the one shared forward pass); the dispatcher refuses a batch that
        disagrees on it rather than picking a winner.

        ``robot_type`` is each element's ``RobotType`` value string -- what
        ``get_action_mask`` needs to zero the action dimensions that robot does not have,
        which is how training fed the model and therefore how sampling has to.
        ``prev_actions`` / ``delay_steps`` are Real-Time Chunking: the flat
        ``(T, action_dim)`` actions the caller is still holding un-executed (``None`` when
        it holds none) and how many of those will have run by the time this answer lands.
        Ragged by nature, hence lists; a wrapper that wants them batched pads them itself.
        A returned chunk must be *longer* than that element's ``prev_actions`` -- it is
        aligned to where inference began, so one reaching no further re-plans only what
        the caller already had, and the caller rejects it.

        A wrapper with no use for a field still takes it. The parameter list is where
        "this policy ignores RTC" gets said out loud, instead of a dispatcher inferring it
        from a signature and dropping the data in silence.
        """

    @abstractmethod
    def post_process(
        self,
        action: torch.Tensor,
        state: Dict[str, Any],
        robot_type: str,
    ) -> Dict[str, Any]:
        pass
