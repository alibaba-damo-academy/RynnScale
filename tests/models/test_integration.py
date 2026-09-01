import gc
import os
import tempfile
from functools import partial
from typing import Dict

import numpy as np
import pytest
import torch

from rynn_scale.arguments import TrainingArguments
from rynn_scale.datasets import build_dataset
from rynn_scale.inference_wrappers import build_inference_wrapper
from rynn_scale.models import build_model, init_weights
from rynn_scale.ops import cross_entropy_loss
from rynn_scale.training import (
    DataCollator,
    Trainer,
)
from rynn_scale.utils.determinism import set_seed

PARALLEL_CONFIGS = {
    "pp1_ep1_cp1_dp4": {
        "pipeline_parallel_size": 1,
        "pipeline_parallel_schedule": None,
        "expert_parallel_size": 1,
        "context_parallel_size": 1,
        "encoder_context_parallel_size": 1,
        "gradient_accumulation_steps": 1,
        "reshard_after_forward": True,
    },
    "pp2_ep1_cp1_dp2": {
        "pipeline_parallel_size": 2,
        "pipeline_parallel_schedule": "1f1b",
        "expert_parallel_size": 1,
        "context_parallel_size": 1,
        "encoder_context_parallel_size": 1,
        "gradient_accumulation_steps": 2,
    },
    "pp1_ep1_cp2_dp2": {
        "pipeline_parallel_size": 1,
        "pipeline_parallel_schedule": None,
        "expert_parallel_size": 1,
        "context_parallel_size": 2,
        "encoder_context_parallel_size": 2,
        "gradient_accumulation_steps": 2,
    },
    "pp2_ep1_cp2_dp1": {
        "pipeline_parallel_size": 2,
        "pipeline_parallel_schedule": "1f1b",
        "expert_parallel_size": 1,
        "context_parallel_size": 2,
        "encoder_context_parallel_size": 2,
        "gradient_accumulation_steps": 4,
    },
    "pp1_ep4_cp1_dp4": {
        "pipeline_parallel_size": 1,
        "pipeline_parallel_schedule": None,
        "expert_parallel_size": 4,
        "context_parallel_size": 1,
        "encoder_context_parallel_size": 1,
        "gradient_accumulation_steps": 1,
    },
    "pp2_ep2_cp1_dp2": {
        "pipeline_parallel_size": 2,
        "pipeline_parallel_schedule": "1f1b",
        "expert_parallel_size": 2,
        "context_parallel_size": 1,
        "encoder_context_parallel_size": 1,
        "gradient_accumulation_steps": 2,
    },
}

MODEL_CONFIGS = {
    "qwen3_5": {
        "model_path": "Qwen/Qwen3.5-0.8B",
        "reference": {
            "pp1_ep1_cp1_dp4": {
                "loss": {"value": 2.474380, "rtol": 0.0001, "atol": 0.0},
                "grad_norm": {"value": 824.0, "rtol": 0.01, "atol": 0.0},
            },
            "pp2_ep1_cp1_dp2": {
                "loss": {"value": 2.474380, "rtol": 0.0001, "atol": 0.0},
                "grad_norm": {"value": 824.0, "rtol": 0.01, "atol": 0.0},
            },
            "pp1_ep1_cp2_dp2": {
                "loss": {"value": 2.474380, "rtol": 0.0001, "atol": 0.0},
                "grad_norm": {"value": 824.0, "rtol": 0.01, "atol": 0.0},
            },
            "pp2_ep1_cp2_dp1": {
                "loss": {"value": 2.474380, "rtol": 0.0001, "atol": 0.0},
                "grad_norm": {"value": 824.0, "rtol": 0.01, "atol": 0.0},
            },
        },
    },
    "qwen3_5_moe": {
        "model_path": "Qwen/Qwen3.5-35B-A3B",
        "reference": {
            "pp1_ep1_cp1_dp4": {
                "loss": {"value": 2.152884, "rtol": 0.0001, "atol": 0.0},
                "grad_norm": {"value": 126.5, "rtol": 0.01, "atol": 0.0},
            },
            "pp1_ep4_cp1_dp4": {
                "loss": {"value": 2.152884, "rtol": 0.0001, "atol": 0.0},
                "grad_norm": {"value": 127.0, "rtol": 0.01, "atol": 0.0},
            },
        },
    },
    "qwen3_vl": {
        "model_path": "Qwen/Qwen3-VL-2B-Instruct",
        "reference": {
            "pp1_ep1_cp1_dp4": {
                "loss": {"value": 1.717212, "rtol": 0.0001, "atol": 0.0},
                "grad_norm": {"value": 130.0, "rtol": 0.01, "atol": 0.0},
            },
            "pp2_ep1_cp1_dp2": {
                "loss": {"value": 1.717212, "rtol": 0.0001, "atol": 0.0},
                "grad_norm": {"value": 131.0, "rtol": 0.01, "atol": 0.0},
            },
            "pp1_ep1_cp2_dp2": {
                "loss": {"value": 1.717212, "rtol": 0.0001, "atol": 0.0},
                "grad_norm": {"value": 131.0, "rtol": 0.01, "atol": 0.0},
            },
            "pp2_ep1_cp2_dp1": {
                "loss": {"value": 1.717212, "rtol": 0.0001, "atol": 0.0},
                "grad_norm": {"value": 131.0, "rtol": 0.01, "atol": 0.0},
            },
        },
    },
    "qwen3_vl_moe": {
        "model_path": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "reference": {
            "pp1_ep1_cp1_dp4": {
                "loss": {"value": 1.553150, "rtol": 0.0001, "atol": 0.0},
                "grad_norm": {"value": 69.0, "rtol": 0.01, "atol": 0.0},
            },
            "pp1_ep4_cp1_dp4": {
                "loss": {"value": 1.553150, "rtol": 0.0001, "atol": 0.0},
                "grad_norm": {"value": 68.5, "rtol": 0.01, "atol": 0.0},
            },
        },
    },
}


def prepare_data(data_path):
    data_mixture: Dict[str, int] = {"text": 2, "single_image": 2, "multi_image": 2, "video": 2}
    with open(data_path, "w") as data_file:
        for modality, num_samples in data_mixture.items():
            with open(os.path.join("tests/assets", f"{modality}.jsonl")) as f:
                for i in range(num_samples):
                    data_file.write(f.readline() + "\n")


@pytest.mark.distributed(world_size=4)
@pytest.mark.parametrize(
    "model_name, config_name",
    [
        pytest.param(model_name, config_name, id=f"{model_name}_{config_name}")
        for model_name, model_config in MODEL_CONFIGS.items()
        for config_name in model_config["reference"]
    ],
)
def test_train(model_name: str, config_name: str):
    model_config = MODEL_CONFIGS[model_name]
    reference = model_config["reference"][config_name]
    parallel_config = PARALLEL_CONFIGS[config_name]

    temp_work_dir = tempfile.TemporaryDirectory()
    data_path = os.path.join(temp_work_dir.name, "data.jsonl")
    prepare_data(data_path)

    args = TrainingArguments(
        model_path=model_config["model_path"],
        optim="sgd",
        gradient_checkpointing=True,
        data_type="VLMDataset",
        data_path=data_path,
        model_max_length=16384,
        mm_max_length=10240,
        fps=2,
        max_frames=512,
        micro_batch_size=2,
        num_train_epochs=1,
        dataloader_num_workers=0,
        loss_reduction_scope="sequence",
        average_tokens_across_devices=True,
        full_determinism=True,
        save_strategy="no",
        logging_strategy="no",
        log_level="warning",
        log_level_replica="warning",
        cp_broadcast_data=True,
        pp_broadcast_data=True,
        master_param_dtype="bfloat16",
        output_dir=temp_work_dir.name,
        **parallel_config,
    )

    set_seed(args.seed, full_determinism=args.full_determinism)

    model, processor = build_model(
        model_type=args.model_type,
        model_path=args.model_path,
        param_dtype=args.param_dtype,
        attn_implementation=args.attn_implementation,
        vision_encoder_path=args.vision_encoder_path,
        reduced_layers_in_stage_zero=args.reduced_layers_in_stage_zero,
        master_param_dtype=args.master_param_dtype,
        reduce_dtype=args.reduce_dtype,
        reshard_after_forward=args.reshard_after_forward,
    )

    init_weights(model, pretrained_model_name_or_path=args.model_path)

    model.loss_function = partial(
        cross_entropy_loss,
        loss_reduction_scope=args.loss_reduction_scope,
    )

    train_dataset = build_dataset(args)
    train_dataset.processor = processor

    data_collator = DataCollator(
        processor=processor,
        sequence_packing=args.sequence_packing,
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        processing_class=processor,
    )

    try:
        output = trainer.train()
    finally:
        temp_work_dir.cleanup()

    loss = torch.as_tensor(output.metrics["train_loss"], device=args.device)
    torch.distributed.all_reduce(
        loss,
        op=torch.distributed.ReduceOp.AVG,
        group=args.dcp_group,
    )

    if args.global_rank == 0:
        torch.testing.assert_close(
            loss.cpu().float(),
            torch.as_tensor(reference["loss"]["value"]),
            rtol=reference["loss"].get("rtol", 0.0),
            atol=reference["loss"].get("atol", 0.0),
        )
        torch.testing.assert_close(
            torch.as_tensor(output.metrics["grad_norm"]),
            torch.as_tensor(reference["grad_norm"]["value"]),
            rtol=reference["grad_norm"].get("rtol", 0.0),
            atol=reference["grad_norm"].get("atol", 0.0),
        )


QA_CASES = {
    "text": [
        {"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]},
    ],
    "image": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "tests/assets/images/000000.jpg"},
                {"type": "text", "text": "Describe the image."},
            ],
        },
    ],
    "video": [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "tests/assets/videos/000000.mp4"},
                {"type": "text", "text": "Describe the video."},
            ],
        },
    ],
}


_WRAPPER_CACHE = {}


def _extract_media(conversation):
    images, videos = [], []
    for message in conversation:
        for content in message["content"]:
            if content["type"] == "image":
                images.append(content["image"])
            elif content["type"] == "video":
                videos.append(content["video"])
    return images, videos


def _infer_generate(model, model_inputs):
    prompt_len = model_inputs["input_ids"].size(1)
    # generate() caches rope_deltas; clear it so the two runs don't cross-talk
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "rope_deltas"):
        inner.rope_deltas = None
    with torch.inference_mode():
        output_ids = model.generate(**model_inputs, max_new_tokens=10, do_sample=False)
    return model_inputs["input_ids"], output_ids[:, prompt_len:]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "model_name, case_name",
    [
        pytest.param(model_name, case_name, id=f"{model_name}_{case_name}")
        for model_name in MODEL_CONFIGS
        for case_name in QA_CASES
    ],
)
def test_inference(model_name: str, case_name: str):
    conversation = QA_CASES[case_name]

    model_path = MODEL_CONFIGS[model_name]["model_path"]
    if model_path not in _WRAPPER_CACHE:
        for stale in list(_WRAPPER_CACHE):
            del _WRAPPER_CACHE[stale]
        gc.collect()
        torch.cuda.empty_cache()
        _WRAPPER_CACHE[model_path] = build_inference_wrapper(
            model_type=model_name,
            model_path=model_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    wrapper = _WRAPPER_CACHE[model_path]

    # wrapper path
    prompt = wrapper.apply_chat_template(conversation, enable_thinking=False)
    images, videos = _extract_media(conversation)

    image_inputs, video_inputs, decoded_videos = {}, {}, None
    if images:
        images = wrapper.load_images(images, processing_params={})
        image_inputs = wrapper.process_images(images, processing_params={})
    if videos:
        decoded_videos = wrapper.load_videos(videos, processing_params={})
        video_inputs = wrapper.process_videos(decoded_videos, processing_params={})

    model_inputs = wrapper.process_text(text=prompt, image_inputs=image_inputs, video_inputs=video_inputs).to(
        wrapper.model.device
    )

    wrap_input_ids, wrap_gen_ids = _infer_generate(wrapper.model, model_inputs)
    wrap_text = wrapper.processor.post_process_image_text_to_text(wrap_gen_ids)

    # reference path
    ref_conversation = [dict(m) for m in conversation]
    template_kwargs = {}

    if decoded_videos is not None:
        frames_list, metadatas = decoded_videos
        idx = 0
        for message in ref_conversation:
            content = []
            for item in message["content"]:
                if item["type"] == "video":
                    item = {**item, "video": np.stack([np.asarray(f) for f in frames_list[idx]])}
                    idx += 1
                content.append(item)
            message["content"] = content
        template_kwargs = {"do_sample_frames": False, "video_metadata": metadatas}

    ref_inputs = wrapper.processor.apply_chat_template(
        ref_conversation,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **template_kwargs,
    ).to(wrapper.model.device)

    off_input_ids, off_gen_ids = _infer_generate(wrapper.model, ref_inputs)
    off_text = wrapper.processor.post_process_image_text_to_text(off_gen_ids)

    assert wrap_input_ids.shape == off_input_ids.shape, (
        f"prompt length differs: wrapper={tuple(wrap_input_ids.shape)} official={tuple(off_input_ids.shape)}"
    )
    assert torch.equal(wrap_input_ids.cpu(), off_input_ids.cpu()), "prompt token ids differ"

    assert wrap_gen_ids.shape == off_gen_ids.shape, (
        f"generated length differs: wrapper={tuple(wrap_gen_ids.shape)} official={tuple(off_gen_ids.shape)}"
    )
    assert torch.equal(wrap_gen_ids.cpu(), off_gen_ids.cpu()), (
        f"generated token ids differ:\n  wrapper : {wrap_text}\n  official: {off_text}"
    )
    assert wrap_text == off_text, f"decoded text differs: wrapper={wrap_text} official={off_text}"
