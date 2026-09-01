import torch
from transformers import Qwen3VLMoeForConditionalGeneration

from ..registry import INFERENCE_WRAPPER_REGISTRY
from .base import restore_cuda_device_hook
from .qwen3_vl import Qwen3VLInferenceWrapper


@INFERENCE_WRAPPER_REGISTRY.register("qwen3_vl_moe")
class Qwen3VLMoeInferenceWrapper(Qwen3VLInferenceWrapper):
    def load_model(self):
        from ..models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts

        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            device_map="auto",
        )

        with torch.no_grad():
            for layer in model.model.language_model.layers:
                if hasattr(layer.mlp, "experts"):
                    gate_up_proj = layer.mlp.experts.gate_up_proj.flatten(start_dim=0, end_dim=1)
                    down_proj = layer.mlp.experts.down_proj.flatten(start_dim=0, end_dim=1)

                    with torch.device("meta"):
                        new_module = Qwen3VLMoeTextExperts(config=model.model.language_model.config)

                    new_module.to_empty(device=gate_up_proj.device)
                    new_module.to(dtype=gate_up_proj.dtype)
                    new_module.gate_up_proj.copy_(gate_up_proj)
                    new_module.down_proj.copy_(down_proj)
                    layer.mlp.register_module("experts", new_module)
                    # register_module drops accelerate's device_map hook; restore
                    # a device-setting one so the experts' triton kernels launch
                    # on the right GPU when the model is sharded across devices.
                    restore_cuda_device_hook(new_module)

        return model
