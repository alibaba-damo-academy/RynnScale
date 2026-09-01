import transformers
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config as _Qwen3_5Config,
)


class Qwen3_5Config(_Qwen3_5Config):
    def __init__(
        self,
        mtp_loss_weight=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mtp_loss_weight = mtp_loss_weight


def apply_monkey_patch():
    transformers.models.qwen3_5.configuration_qwen3_5.Qwen3_5Config = Qwen3_5Config
    transformers.models.auto.configuration_auto.CONFIG_MAPPING.register("qwen3_5", Qwen3_5Config, exist_ok=True)
