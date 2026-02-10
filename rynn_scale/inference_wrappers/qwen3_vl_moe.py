from transformers import Qwen3VLMoeForConditionalGeneration

from .qwen3_vl import Qwen3VLInferenceWrapper
from ..registry import INFERENCE_WRAPPER_REGISTRY


@INFERENCE_WRAPPER_REGISTRY.register("qwen3_vl_moe")
class Qwen3VLMoeInferenceWrapper(Qwen3VLInferenceWrapper):
    def load_model(self):
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            device_map={"": "cuda:0"},
        )
        return model
