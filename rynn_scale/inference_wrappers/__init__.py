from typing import Optional, Dict, Any

import torch

from .base import BaseInferenceWrapper
from .qwen3_vl import Qwen3VLInferenceWrapper
from .qwen3_vl_moe import Qwen3VLMoeInferenceWrapper
from ..registry import INFERENCE_WRAPPER_REGISTRY


def build_inference_wrapper(
    model_type: str,
    model_path: str,
    dtype: torch.dtype,
    attn_implementation: str,
) -> BaseInferenceWrapper:
    return INFERENCE_WRAPPER_REGISTRY[model_type](
        model_path=model_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
