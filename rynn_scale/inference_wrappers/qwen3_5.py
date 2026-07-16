from transformers import Qwen3_5ForConditionalGeneration

from ..registry import INFERENCE_WRAPPER_REGISTRY
from .qwen3_vl import Qwen3VLInferenceWrapper


@INFERENCE_WRAPPER_REGISTRY.register("qwen3_5")
class Qwen3_5InferenceWrapper(Qwen3VLInferenceWrapper):
    def load_model(self):
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            device_map="auto",
        )
        return model
