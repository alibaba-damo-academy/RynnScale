from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.activations import ACT2FN
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeModel,
    Qwen3_5MoeTextModel,
    Qwen3_5MoeTextRotaryEmbedding,
    Qwen3_5MoeVisionModel,
)

from ...ops import grouped_linear
from ...utils.expert_parallel import BaseMoELayer, MoETokenDispatcher
from ..qwen3_5.modeling_qwen3_5 import (
    _Qwen3_5ForConditionalGeneration,
    _Qwen3_5Model,
    _Qwen3_5RMSNorm,
    _Qwen3_5TextModel,
    _Qwen3_5VisionModel,
)


class _Qwen3_5MoeVisionModel(Qwen3_5MoeVisionModel):
    forward = _Qwen3_5VisionModel.forward
    floating_point_ops = _Qwen3_5VisionModel.floating_point_ops

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_checkpointing_interval = None


class _Qwen3_5MoeExperts(BaseMoELayer):
    def __init__(self, config):
        super().__init__(config.num_experts)

        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size

        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts * self.intermediate_dim * 2, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts * self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

        self.token_dispatcher = MoETokenDispatcher(
            num_experts=self.num_experts,
            num_experts_per_token=config.num_experts_per_tok,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states, num_tokens_per_expert = self.token_dispatcher.dispatch(
            hidden_states, top_k_index, top_k_weights
        )

        gate_up = grouped_linear(
            input=hidden_states,
            weight=self.gate_up_proj,
            input_group_sizes=num_tokens_per_expert,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_states = up * self.act_fn(gate)

        hidden_states = grouped_linear(
            input=hidden_states,
            weight=self.down_proj,
            input_group_sizes=num_tokens_per_expert,
        )

        hidden_states = self.token_dispatcher.combine(hidden_states)

        return hidden_states


class _Qwen3_5MoeRMSNorm(_Qwen3_5RMSNorm):
    pass


class _Qwen3_5MoeTextModel(Qwen3_5MoeTextModel):
    forward = _Qwen3_5TextModel.forward

    def __init__(self, config):
        super(Qwen3_5MoeTextModel, self).__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleDict(
            {
                str(layer_idx): Qwen3_5MoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            }
        )
        self.norm = _Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def floating_point_ops(self, inputs: Dict[str, Any]):
        input_ids = inputs["input_ids"]

        hidden_size = self.config.text_config.hidden_size
        num_hidden_layers = self.config.text_config.num_hidden_layers
        num_attention_heads = self.config.text_config.num_attention_heads
        num_key_value_heads = self.config.text_config.num_key_value_heads
        intermediate_size = self.config.text_config.moe_intermediate_size * self.config.text_config.num_experts_per_tok
        head_dim = hidden_size // num_attention_heads

        if input_ids.size(0) == 1 and inputs.get("position_ids", None) is not None:
            position_ids = inputs["position_ids"]
            start_indices = torch.nonzero(position_ids[0, 0] == 0)[:, 0]
            end_indices = F.pad(start_indices[1:], (0, 1), value=position_ids.size(-1))
            seq_lens = end_indices - start_indices
        else:
            seq_lens = torch.tensor([input_ids.size(1)] * input_ids.size(0), device=input_ids.device)

        layer_flops = [
            # attention
            2 * seq_lens * hidden_size * hidden_size,  # q_proj
            2 * seq_lens * hidden_size * (num_key_value_heads * head_dim),  # k_proj
            2 * seq_lens * hidden_size * (num_key_value_heads * head_dim),  # v_proj
            2 * seq_lens * seq_lens * hidden_size,  # attention scores
            2 * seq_lens * seq_lens * hidden_size,  # attention output
            2 * seq_lens * hidden_size * hidden_size,  # out_proj
            # mlp
            2 * seq_lens * hidden_size * intermediate_size,  # gate_proj
            2 * seq_lens * hidden_size * intermediate_size,  # up_proj
            2 * seq_lens * intermediate_size * hidden_size,  # down_proj
        ]

        layer_flops = sum(layer_flops).sum().item()
        flops = num_hidden_layers * layer_flops

        return flops


class _Qwen3_5MoeModel(Qwen3_5MoeModel):
    get_multimodal_features = _Qwen3_5Model.get_multimodal_features
    get_placeholder_mask = _Qwen3_5Model.get_placeholder_mask
    forward = _Qwen3_5Model.forward
    floating_point_ops = _Qwen3_5Model.floating_point_ops
    apply_pipeline_parallel = _Qwen3_5Model.apply_pipeline_parallel

    def apply_expert_parallel(self, ep_world_size: int, ep_rank: int):
        assert self.config.text_config.num_experts % ep_world_size == 0
        for module in self.modules():
            if isinstance(module, _Qwen3_5MoeExperts):
                for name, param in module.named_parameters():
                    new_param = nn.Parameter(param.data.chunk(ep_world_size, dim=0)[ep_rank])
                    del param
                    module.register_parameter(name, new_param)


class _Qwen3_5MoeForConditionalGeneration(Qwen3_5MoeForConditionalGeneration):
    accepts_loss_kwargs = True

    forward = _Qwen3_5ForConditionalGeneration.forward
    floating_point_ops = _Qwen3_5ForConditionalGeneration.floating_point_ops
    apply_pipeline_parallel = _Qwen3_5ForConditionalGeneration.apply_pipeline_parallel

    def apply_expert_parallel(self, ep_world_size: int, ep_rank: int):
        self.model.apply_expert_parallel(
            ep_world_size=ep_world_size,
            ep_rank=ep_rank,
        )

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
        convert: bool = True,
    ):
        if convert:
            new_state_dict = {}
            for name, tensor in state_dict.items():
                if ".experts.gate_up_proj" in name or ".experts.down_proj" in name:
                    new_state_dict[name] = tensor.flatten(start_dim=0, end_dim=1)
                else:
                    new_state_dict[name] = tensor
            state_dict = new_state_dict
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def state_dict(self, *args, convert: bool = True, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        if convert:
            hidden_size = self.config.text_config.hidden_size
            moe_intermediate_size = self.config.text_config.moe_intermediate_size
            for name, tensor in state_dict.items():
                if ".experts.gate_up_proj" in name:
                    size = (-1, moe_intermediate_size * 2, hidden_size)
                    state_dict[name] = tensor.view(size)
                elif ".experts.down_proj" in name:
                    size = (-1, hidden_size, moe_intermediate_size)
                    state_dict[name] = tensor.view(size)
        return state_dict


def apply_monkey_patch():
    transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeVisionModel = _Qwen3_5MoeVisionModel
    transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeRMSNorm = _Qwen3_5MoeRMSNorm
    transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeExperts = _Qwen3_5MoeExperts
    transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeTextModel = _Qwen3_5MoeTextModel

    transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeModel = _Qwen3_5MoeModel
    transformers.models.auto.modeling_auto.MODEL_MAPPING[Qwen3_5MoeConfig] = _Qwen3_5MoeModel

    transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeForConditionalGeneration = (
        _Qwen3_5MoeForConditionalGeneration
    )
    transformers.models.auto.modeling_auto.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING[Qwen3_5MoeConfig] = (
        _Qwen3_5MoeForConditionalGeneration
    )
