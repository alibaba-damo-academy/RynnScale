import pytest
import torch
import transformers
from transformers import AutoModelForImageTextToText

from rynn_scale import parallel_state as mpu
from rynn_scale.models.qwen3_5 import apply_monkey_patch
from rynn_scale.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from rynn_scale.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from tests.models.qwen3_vl.test_modeling_qwen3_vl import build_multimodal_inputs


@pytest.mark.distributed(world_size=1)
def test_qwen3_5_model():
    mpu.initialize_model_parallel()

    config = Qwen3_5Config(
        text_config=dict(
            vocab_size=128,
            hidden_size=8,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
        ),
        vision_config=dict(
            depth=4,
            hidden_size=8,
            intermediate_size=32,
            num_heads=2,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=8,
        ),
        image_token_id=0,
        video_token_id=1,
        vision_start_token_id=2,
        vision_end_token_id=3,
        tie_word_embeddings=True,
    )

    model_inputs, mm_token_type_ids, position_ids = build_multimodal_inputs(config)

    ref_model = transformers.Qwen3_5ForConditionalGeneration._from_config(
        config,
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
    assert isinstance(model, Qwen3_5ForConditionalGeneration)

    model.cuda()
    model.load_state_dict(ref_model.state_dict())

    outputs = model(
        **model_inputs,
        use_cache=False,
        position_ids=position_ids,
    )

    torch.testing.assert_close(outputs.logits, ref_outputs.logits, rtol=0.0, atol=0.0)
