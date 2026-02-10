from . import modeling_qwen3_vl, processing_qwen3_vl


def apply_monkey_patch():
    modeling_qwen3_vl.apply_monkey_patch()
    processing_qwen3_vl.apply_monkey_patch()
