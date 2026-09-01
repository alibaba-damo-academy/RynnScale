import torch
import transformers
from transformers import AutoProcessor

from rynn_scale.models.qwen3_vl.processing_qwen3_vl import (
    Qwen3VLProcessor,
    apply_monkey_patch,
)


def test_qwen3_5_processor(model_path: str = "Qwen/Qwen3-2B"):
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": "This is message 0."}]},
        {"role": "user", "content": [{"type": "text", "text": "This is message 1."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "This is message 2."}]},
        {"role": "user", "content": [{"type": "text", "text": "This is message 3."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "This is message 4."}]},
    ]

    ref_labels = torch.as_tensor(
        [
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            1919,
            369,
            1876,
            220,
            17,
            13,
            248046,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            -100,
            1919,
            369,
            1876,
            220,
            19,
            13,
            248046,
            -100,
        ]
    ).unsqueeze(0)

    ref_processor = transformers.Qwen3VLProcessor.from_pretrained(model_path)
    ref_model_inputs = ref_processor.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    apply_monkey_patch()

    processor = AutoProcessor.from_pretrained(model_path)
    assert isinstance(processor, Qwen3VLProcessor)

    model_inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        return_labels=True,
    )

    assert torch.all(model_inputs["input_ids"] == ref_model_inputs["input_ids"])
    assert torch.all(model_inputs["labels"] == ref_labels)
