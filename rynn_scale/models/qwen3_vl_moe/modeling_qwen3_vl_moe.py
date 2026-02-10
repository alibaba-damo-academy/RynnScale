from dataclasses import dataclass
from functools import partial
from typing import Optional, Union, Dict, Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.activations import ACT2FN
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeConfig,
    Qwen3VLMoeTextConfig,
)
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeModel as _Qwen3VLMoeModel,
    Qwen3VLMoeForConditionalGeneration as _Qwen3VLMoeForConditionalGeneration,
    Qwen3VLMoeTextModel as _Qwen3VLMoeTextModel,
    Qwen3VLMoeTextSparseMoeBlock,
    Qwen3VLMoeTextAttention,
    Qwen3VLMoeTextMLP,
    Qwen3VLMoeTextRMSNorm,
    Qwen3VLMoeTextRotaryEmbedding,
    create_causal_mask,
    BaseModelOutputWithPast,
)
from transformers.modeling_outputs import ModelOutput
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs, check_model_inputs

from ..qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel, Qwen3VLModel, Qwen3VLVisionBlock
from ...utils.expert_parallel import BaseMoELayer, MoETokenDispatcher
from ...ops import grouped_linear


@dataclass
class Qwen3VLMoeCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class Qwen3VLMoeTextExperts(BaseMoELayer):
    def __init__(self, config):
        super().__init__(config.num_experts)

        self.intermediate_size = config.moe_intermediate_size
        self.hidden_size = config.hidden_size

        self.gate_up_proj = nn.Parameter(
            torch.empty(config.num_experts * self.intermediate_size * 2, self.hidden_size)
        )
        self.down_proj = nn.Parameter(torch.empty((config.num_experts * self.hidden_size, self.intermediate_size)))
        self.act_fn = ACT2FN[config.hidden_act]

        self.token_dispatcher = MoETokenDispatcher(
            num_experts=self.num_experts,
            num_experts_per_token=config.num_experts_per_tok,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        router_indices: torch.Tensor,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]

        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        routing_weights = routing_weights.gather(dim=1, index=router_indices)
        routing_weights = routing_weights.view(-1, routing_weights.size(-1))
        router_indices = router_indices.view(-1, router_indices.size(-1))

        hidden_states, num_tokens_per_expert = self.token_dispatcher.dispatch(
            hidden_states, router_indices, routing_weights
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
        hidden_states = hidden_states.view(*input_shape, -1)

        return hidden_states


class Qwen3VLMoeTextDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3VLMoeTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.gradient_checkpointing = False
        self.selective_gradient_checkpointing = False

        self.self_attn = Qwen3VLMoeTextAttention(config, layer_idx)

        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3VLMoeTextSparseMoeBlock(config)
        else:
            self.mlp = Qwen3VLMoeTextMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = Qwen3VLMoeTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLMoeTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _forward_attention(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        return hidden_states

    def _forward_mlp(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # For the MoE layers, we need to unpack
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states
        return hidden_states

    def _forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        hidden_states = self._forward_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if self.gradient_checkpointing and self.training:
            if self.selective_gradient_checkpointing:
                return self._gradient_checkpointing_func(self._forward_mlp, hidden_states)
        return self._forward_mlp(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        args = dict(
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        if self.gradient_checkpointing and self.training:
            if not self.selective_gradient_checkpointing:
                return self._gradient_checkpointing_func(partial(self._forward, **args), hidden_states)
        return self._forward(hidden_states, **args)


class Qwen3VLMoeTextModel(_Qwen3VLMoeTextModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleDict(
            {
                str(layer_idx): Qwen3VLMoeTextDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            }
        )
        self.norm = Qwen3VLMoeTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLMoeTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        for layer_idx in range(self.config.num_hidden_layers):
            if str(layer_idx) not in self.layers:
                continue
            layer_outputs = self.layers[str(layer_idx)](
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        if hasattr(self, "norm"):
            hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Qwen3VLMoeModel(_Qwen3VLMoeModel):
    get_multimodal_features = Qwen3VLModel.get_multimodal_features
    forward = Qwen3VLModel.forward

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_channels = self.config.vision_config.in_channels
        patch_size = self.config.vision_config.patch_size
        temporal_patch_size = self.config.vision_config.temporal_patch_size
        self.patch_channels = patch_size * patch_size * in_channels * temporal_patch_size


class Qwen3VLMoeForConditionalGeneration(_Qwen3VLMoeForConditionalGeneration):
    accepts_loss_kwargs = True
    supports_selective_gradient_checkpointing = True

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLMoeCausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        loss, logits = None, None
        if hasattr(self, "lm_head"):
            if labels is not None:
                loss = self.loss_function(
                    hidden_states=hidden_states,
                    lm_head=self.lm_head,
                    position_ids=position_ids,
                    labels=labels,
                    num_items_in_batch=kwargs["num_items_in_batch"],
                )
            else:
                slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
                logits = self.lm_head(hidden_states[:, slice_indices, :])

        return Qwen3VLMoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            last_hidden_state=hidden_states,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def apply_pipeline_parallel(
        self,
        num_stages: int,
        stage_index: int,
        reduced_layers_in_stage_zero: int = 0,
    ):
        num_layers = len(self.model.language_model.layers)
        assert (num_layers + reduced_layers_in_stage_zero) % num_stages == 0
        num_local_layers = [
            (num_layers + reduced_layers_in_stage_zero) // num_stages - reduced_layers_in_stage_zero
            if i == 0
            else (num_layers + reduced_layers_in_stage_zero) // num_stages
            for i in range(num_stages)
        ]

        start_index = 0
        for i, local_layers in enumerate(num_local_layers):
            end_index = start_index + local_layers
            if i == stage_index:
                break
            start_index = end_index

        for layer_idx in list(self.model.language_model.layers.keys()):
            layer_idx = int(layer_idx)
            if layer_idx < start_index or layer_idx >= end_index:
                del self.model.language_model.layers[str(layer_idx)]

        if stage_index > 0:
            del self.model.visual
            if stage_index < num_stages - 1 or not self.config.text_config.tie_word_embeddings:
                del self.model.language_model.embed_tokens

        if stage_index < num_stages - 1:
            del self.model.language_model.norm, self.lm_head

    def apply_expert_parallel(self, ep_world_size: int, ep_rank: int):
        assert self.config.text_config.num_experts % ep_world_size == 0
        for module in self.modules():
            if isinstance(module, Qwen3VLMoeTextExperts):
                for name, param in module.named_parameters():
                    new_param = nn.Parameter(param.data.chunk(ep_world_size, dim=0)[ep_rank])
                    del param
                    module.register_parameter(name, new_param)

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

        decoder_layer_flops = [
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

        decoder_layer_flops = sum(decoder_layer_flops).sum().item()
        decoder_flops = num_hidden_layers * decoder_layer_flops

        grid_thw = [inputs[x] for x in ["image_grid_thw", "video_grid_thw"] if x in inputs]
        if len(grid_thw) > 0:
            encoder_flops = self.visual.floating_point_ops({"grid_thw": torch.cat(grid_thw, dim=0)})
        else:
            encoder_flops = 0

        return decoder_flops + encoder_flops

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
                    new_state_dict[name] = tensor.transpose(1, 2).reshape(-1, tensor.size(1))
                    del tensor
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
                    state_dict[name] = tensor.view(size).transpose(1, 2)
                elif ".experts.down_proj" in name:
                    size = (-1, hidden_size, moe_intermediate_size)
                    state_dict[name] = tensor.view(size).transpose(1, 2)
        return state_dict


def apply_monkey_patch():
    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeVisionBlock = Qwen3VLVisionBlock
    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextExperts = Qwen3VLMoeTextExperts
    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextDecoderLayer = Qwen3VLMoeTextDecoderLayer
    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeTextModel = Qwen3VLMoeTextModel
    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeVisionModel = Qwen3VLVisionModel
    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeModel = Qwen3VLMoeModel

    transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe.Qwen3VLMoeForConditionalGeneration = (
        Qwen3VLMoeForConditionalGeneration
    )
    transformers.models.auto.modeling_auto.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING[Qwen3VLMoeConfig] = (
        Qwen3VLMoeForConditionalGeneration
    )
