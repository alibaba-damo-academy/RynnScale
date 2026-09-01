import torch
import transformers
from transformers import AutoModelForImageTextToText, Qwen3VLConfig

from rynn_scale.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    apply_monkey_patch,
)
from rynn_scale.models.qwen3_vl.processing_qwen3_vl import _get_rope_index_qwen3_vl


def build_multimodal_inputs(config, batch_size=2):
    """Construct a shared multimodal batch for the Qwen3-VL family tests.

    The sequence contains one image followed by timestamp-separated video frames and
    trailing text, so the reference model can compute the multimodal position ids
    internally from ``mm_token_type_ids``. Returns the model inputs (on CUDA), the
    ``mm_token_type_ids`` expected by the reference model, and the rope ``position_ids``.
    """
    spatial_merge_size = config.vision_config.spatial_merge_size
    image_grid_thw = torch.as_tensor([[1, 2, 2]])
    video_grid_thw = torch.as_tensor([[2, 2, 2]])
    patch_dim = (
        config.vision_config.in_channels
        * config.vision_config.temporal_patch_size
        * config.vision_config.patch_size**2
    )
    pixel_values = torch.randn((image_grid_thw.prod(dim=1).sum(), patch_dim))
    pixel_values_videos = torch.randn((video_grid_thw.prod(dim=1).sum(), patch_dim))

    # Build a token sequence with timestamp-separated video frames, so the reference
    # model can compute the multimodal position ids internally from mm_token_type_ids.
    tokens = []
    tokens.append(config.vision_start_token_id)
    tokens += [config.image_token_id] * int(image_grid_thw[0].prod() // spatial_merge_size**2)
    tokens.append(config.vision_end_token_id)
    num_frames, frame_h, frame_w = video_grid_thw[0].tolist()
    tokens_per_frame = (frame_h // spatial_merge_size) * (frame_w // spatial_merge_size)
    for frame_idx in range(num_frames):
        tokens.append(config.vision_start_token_id)
        tokens += [config.video_token_id] * tokens_per_frame
        tokens.append(config.vision_end_token_id)
        if frame_idx != num_frames - 1:
            tokens.append(10)  # timestamp text token separating consecutive video frames
    tokens += [10] * 20  # trailing text tokens

    input_ids = torch.as_tensor(tokens).unsqueeze(0).repeat(batch_size, 1)
    # Randomize the pure-text positions while keeping the multimodal layout intact.
    text_mask = input_ids >= 10
    input_ids = torch.where(
        text_mask,
        torch.randint(10, config.text_config.vocab_size, input_ids.shape),
        input_ids,
    )

    pixel_values = pixel_values.repeat(batch_size, 1)
    pixel_values_videos = pixel_values_videos.repeat(batch_size, 1)
    image_grid_thw = image_grid_thw.repeat(batch_size, 1)
    video_grid_thw = video_grid_thw.repeat(batch_size, 1)

    model_inputs = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "pixel_values_videos": pixel_values_videos,
        "image_grid_thw": image_grid_thw,
        "video_grid_thw": video_grid_thw,
    }
    for k, v in model_inputs.items():
        model_inputs[k] = v.cuda()

    mm_token_type_ids = torch.zeros_like(input_ids)
    mm_token_type_ids[input_ids == config.image_token_id] = 1
    mm_token_type_ids[input_ids == config.video_token_id] = 2

    position_ids = _get_rope_index_qwen3_vl(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        image_spatial_merge_size=config.vision_config.spatial_merge_size,
        video_spatial_merge_size=config.vision_config.spatial_merge_size,
        image_token_id=config.image_token_id,
        video_token_id=config.video_token_id,
        vision_start_token_id=config.vision_start_token_id,
    ).cuda()

    return model_inputs, mm_token_type_ids, position_ids


def test_qwen3_vl_model():
    config = Qwen3VLConfig(
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
            deepstack_visual_indexes=[0, 1],
        ),
        image_token_id=0,
        video_token_id=1,
        vision_start_token_id=2,
        vision_end_token_id=3,
        tie_word_embeddings=True,
    )

    model_inputs, mm_token_type_ids, position_ids = build_multimodal_inputs(config)

    ref_model = transformers.Qwen3VLForConditionalGeneration._from_config(
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
    assert isinstance(model, Qwen3VLForConditionalGeneration)

    model.cuda()
    model.load_state_dict(ref_model.state_dict())

    outputs = model(
        **model_inputs,
        use_cache=False,
        position_ids=position_ids,
    )

    torch.testing.assert_close(outputs.logits, ref_outputs.logits, rtol=0.0, atol=0.0)
