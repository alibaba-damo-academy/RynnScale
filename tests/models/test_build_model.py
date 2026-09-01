import pytest
import torch
from torch.distributed.tensor import DTensor
from transformers import Qwen3VLForConditionalGeneration

from rynn_scale import parallel_state as mpu
from rynn_scale.models import build_model, init_weights


@pytest.mark.distributed(world_size=4)
def test_build_model(
    model_path: str = "Qwen/Qwen3-VL-2B-Instruct",
    dtype: torch.dtype = torch.bfloat16,
):
    mpu.initialize_model_parallel()

    ref_model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, dtype=dtype)
    ref_state_dict = ref_model.state_dict()

    model, _ = build_model(
        model_type="qwen3_vl",
        model_path=model_path,
        param_dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    init_weights(model, pretrained_model_name_or_path=model_path)

    for key, tensor in model.state_dict().items():
        assert key in ref_state_dict
        if isinstance(tensor, DTensor):
            tensor = tensor.full_tensor()
        ref_tensor = ref_state_dict[key].to(device=tensor.device, dtype=tensor.dtype)
        assert torch.all(tensor == ref_tensor), key
