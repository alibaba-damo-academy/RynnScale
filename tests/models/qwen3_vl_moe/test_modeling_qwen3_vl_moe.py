from copy import deepcopy

import torch
import transformers
from transformers import AutoModelForImageTextToText, Qwen3VLMoeConfig

from rynn_scale.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeForConditionalGeneration,
    apply_monkey_patch,
)
from tests.models.qwen3_vl.test_modeling_qwen3_vl import build_multimodal_inputs


def test_qwen3_vl_moe_model():
    ref_config = Qwen3VLMoeConfig(
        text_config=dict(
            vocab_size=128,
            hidden_size=64,
            moe_intermediate_size=16,
            num_experts_per_tok=8,
            num_experts=64,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
        ),
        vision_config=dict(
            depth=4,
            hidden_size=8,
            intermediate_size=32,
            num_heads=2,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=64,
            deepstack_visual_indexes=[0, 1],
        ),
        image_token_id=0,
        video_token_id=1,
        vision_start_token_id=2,
        vision_end_token_id=3,
        tie_word_embeddings=True,
    )

    config = deepcopy(ref_config)

    model_inputs, mm_token_type_ids, position_ids = build_multimodal_inputs(config)

    ref_model = transformers.Qwen3VLMoeForConditionalGeneration._from_config(
        ref_config,
        dtype=torch.bfloat16,
    )
    ref_model.cuda()

    ref_outputs = ref_model(
        **model_inputs,
        mm_token_type_ids=mm_token_type_ids,
        use_cache=False,
    )

    apply_monkey_patch()

    model = AutoModelForImageTextToText.from_config(
        config,
        dtype=torch.bfloat16,
    )
    assert isinstance(model, Qwen3VLMoeForConditionalGeneration)

    state_dict = {}
    for name, tensor in ref_model.state_dict().items():
        if ".experts." in name:
            tensor = tensor.transpose(1, 2)
        state_dict[name] = tensor

    model.cuda()
    model.load_state_dict(state_dict, convert=True)

    outputs = model(
        **model_inputs,
        use_cache=False,
        position_ids=position_ids,
    )

    # The bf16 MoE expert path (grouped experts + top-k routing) is not bit-exact
    # against the transformers reference (accumulation order and occasional routing
    # tie-breaks differ), so use a small tolerance instead of exact match.
    torch.testing.assert_close(outputs.logits, ref_outputs.logits, rtol=2e-2, atol=1e-2)
