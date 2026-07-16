from ..qwen3_vl import processing_qwen3_vl
from . import modeling_qwen3_5_moe


def apply_monkey_patch():
    modeling_qwen3_5_moe.apply_monkey_patch()
    processing_qwen3_vl.apply_monkey_patch()
