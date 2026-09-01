import transformers
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig as _Qwen3_5MoeConfig,
)


class Qwen3_5MoeConfig(_Qwen3_5MoeConfig):
    def __init__(
        self,
        mtp_loss_weight=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mtp_loss_weight = mtp_loss_weight


def apply_monkey_patch():
    transformers.models.qwen3_5_moe.configuration_qwen3_5_moe.Qwen3_5MoeConfig = Qwen3_5MoeConfig
    transformers.models.auto.configuration_auto.CONFIG_MAPPING.register("qwen3_5_moe", Qwen3_5MoeConfig, exist_ok=True)
