from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import Cache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, CausalLMOutputWithPast
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5DynamicCache,
    Qwen3_5GatedDeltaNet,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5RMSNorm,
    Qwen3_5TextModel,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5VisionModel,
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    eager_attention_forward,
    apply_rotary_pos_emb,
    apply_mask_to_padding_states,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_outputs import ModelOutput
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from ... import parallel_state as mpu
from ...utils import context_parallel


class _Qwen3_5GatedDeltaNet(Qwen3_5GatedDeltaNet):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Qwen3_5DynamicCache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_position is not None
        )

        # getting projected states from cache if it exists
        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        mixed_qkv = self.in_proj_qkv(hidden_states)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )

        query = query.view(batch_size, seq_len, -1, self.head_k_dim)
        key = key.view(batch_size, seq_len, -1, self.head_k_dim)
        value = value.view(batch_size, seq_len, -1, self.head_v_dim)

        query, key, value = context_parallel.ulysses_preprocess(query, key, value)
        query, key, value = query.flatten(2), key.flatten(2), value.flatten(2)
        mixed_qkv = torch.cat([query, key, value], dim=-1).transpose(1, 2)
        global_seq_len = mixed_qkv.size(2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        beta = context_parallel.ulysses_preprocess_single(beta)
        g = context_parallel.ulysses_preprocess_single(g)

        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()
        conv_weights = torch.split(
            self.conv1d.weight,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=0,
        )
        conv_weight = torch.cat([x.chunk(cp_size, dim=0)[cp_rank] for x in conv_weights], dim=0)

        if use_precomputed_states:
            # 2. Convolution sequence transformation
            # NOTE: the conv state is updated in `causal_conv1d_update`
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                conv_weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=conv_weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.conv1d(
                    mixed_qkv,
                    conv_weight,
                    self.conv1d.bias,
                    padding=self.conv_kernel_size - 1,
                    groups=mixed_qkv.size(1),
                )[:, :, :global_seq_len]
                mixed_qkv = F.silu(mixed_qkv)

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // cp_size,
                self.key_dim // cp_size,
                self.value_dim // cp_size,
            ],
            dim=-1,
        )

        query = query.reshape(batch_size, global_seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, global_seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, global_seq_len, -1, self.head_v_dim)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        core_attn_out = context_parallel.ulysses_postprocess(core_attn_out)

        # Update cache
        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output


class _Qwen3_5Attention(Qwen3_5Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape))
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        query_states, key_states, value_states = context_parallel.ulysses_preprocess(
            query_states, key_states, value_states,
        )

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = context_parallel.ulysses_postprocess(attn_output)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def get_seq_idx(cu_seqlens: torch.Tensor) -> torch.Tensor:
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    seq_idx = torch.repeat_interleave(
        torch.arange(lengths.size(0), device=cu_seqlens.device, dtype=torch.int32),
        lengths,
    )
    return seq_idx.unsqueeze(0)


class _Qwen3_5GatedDeltaNet(Qwen3_5GatedDeltaNet):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Qwen3_5DynamicCache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cu_seq_lens: torch.Tensor | None = None,
        seq_idx: torch.Tensor | None = None,
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_position is not None
        )

        # getting projected states from cache if it exists
        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        mixed_qkv = self.in_proj_qkv(hidden_states)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )

        query = query.view(batch_size, seq_len, -1, self.head_k_dim)
        key = key.view(batch_size, seq_len, -1, self.head_k_dim)
        value = value.view(batch_size, seq_len, -1, self.head_v_dim)

        query, key, value = context_parallel.ulysses_preprocess(query, key, value)
        query, key, value = query.flatten(2), key.flatten(2), value.flatten(2)
        mixed_qkv = torch.cat([query, key, value], dim=-1).transpose(1, 2)
        global_seq_len = mixed_qkv.size(2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        beta = context_parallel.ulysses_preprocess_single(beta)
        g = context_parallel.ulysses_preprocess_single(g)

        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()
        conv_weights = torch.split(
            self.conv1d.weight,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=0,
        )
        conv_weight = torch.cat([x.chunk(cp_size, dim=0)[cp_rank] for x in conv_weights], dim=0)

        if use_precomputed_states:
            # 2. Convolution sequence transformation
            # NOTE: the conv state is updated in `causal_conv1d_update`
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                conv_weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,
                weight=conv_weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=seq_idx,
            )

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // cp_size,
                self.key_dim // cp_size,
                self.value_dim // cp_size,
            ],
            dim=-1,
        )

        query = query.reshape(batch_size, global_seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, global_seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, global_seq_len, -1, self.head_v_dim)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seq_lens,
            )

        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        core_attn_out = context_parallel.ulysses_postprocess(core_attn_out)

        # Update cache
        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output


class _Qwen3_5DecoderLayer(Qwen3_5DecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Token Mixer
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                cu_seq_lens=kwargs.get("cu_seq_lens_q", None),
                seq_idx=kwargs.get("seq_idx_q", None),
            )
        elif self.layer_type == "full_attention":
            # Self Attention
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

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@dataclass
class Qwen3_5CausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class _Qwen3_5VisionModel(Qwen3_5VisionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_checkpointing_interval = None

    @merge_with_config_defaults
    @can_return_tuple
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        fake_forward = False
        if hidden_states.size(0) == 0:
            fake_forward = True
            hidden_states = hidden_states.new_zeros(
                (
                    self.spatial_merge_size * self.spatial_merge_size,
                    self.patch_size * self.patch_size * self.config.in_channels * self.config.temporal_patch_size,
                ),
            )
            grid_thw = grid_thw.new_tensor([[1, self.spatial_merge_size, self.spatial_merge_size]])

        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        dispatcher = context_parallel.EncoderContextDispatcher(grid_thw=grid_thw, merge_size=self.spatial_merge_size)
        hidden_states = dispatcher.dispatch(hidden_states)
        rotary_pos_emb = dispatcher.dispatch(rotary_pos_emb)

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        if dispatcher.cu_seqlens is not None:
            cu_seqlens = grid_thw.new_tensor(dispatcher.cu_seqlens, dtype=torch.int32)
        else:
            cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                dim=0,
                # Select dtype based on the following factors:
                #  - FA2 requires that cu_seqlens_q must have dtype int32
                #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
                # See https://github.com/huggingface/transformers/pull/34852 for more information
                dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
            )
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        def forward_chunk(hidden_states, start, end):
            for i in range(start, end):
                if i >= len(self.blocks):
                    break

                hidden_states = self.blocks[i](
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            return hidden_states

        start = 0

        while start < len(self.blocks):
            if self.gradient_checkpointing_interval is not None and self.training:
                end = start + self.gradient_checkpointing_interval
                hidden_states = checkpoint(
                    forward_chunk,
                    hidden_states,
                    start,
                    end,
                    use_reentrant=True,
                )
            else:
                end = start + len(self.blocks)
                hidden_states = forward_chunk(
                    hidden_states,
                    start,
                    end,
                )
            start = end

        merged_hidden_states = self.merger(hidden_states)
        merged_hidden_states = dispatcher.combine(merged_hidden_states)
        if fake_forward:
            merged_hidden_states = merged_hidden_states[:0]

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )

    def floating_point_ops(self, inputs: Dict[str, Any]):
        grid_thw = inputs.get("grid_thw")

        patch_size = self.config.patch_size
        temporal_patch_size = self.config.temporal_patch_size
        in_channels = self.config.in_channels
        patch_dim = patch_size * patch_size * in_channels * temporal_patch_size

        hidden_size = self.config.hidden_size
        intermediate_size = self.config.intermediate_size
        num_hidden_layers = self.config.depth

        seq_lens = grid_thw.prod(dim=1)
        embedding_flops = 2 * seq_lens * patch_dim * hidden_size

        layer_flops = [
            # attention
            2 * seq_lens * hidden_size * hidden_size,  # q_proj
            2 * seq_lens * hidden_size * hidden_size,  # k_proj
            2 * seq_lens * hidden_size * hidden_size,  # v_proj
            2 * seq_lens * seq_lens * hidden_size,  # attention scores
            2 * seq_lens * seq_lens * hidden_size,  # attention output
            2 * seq_lens * hidden_size * hidden_size,  # out_proj
            # mlp
            2 * seq_lens * hidden_size * intermediate_size,  # up_proj
            2 * seq_lens * intermediate_size * hidden_size,  # down_proj
        ]

        flops = torch.sum(embedding_flops + num_hidden_layers * sum(layer_flops)).item()

        return flops


class _Qwen3_5RMSNorm(Qwen3_5RMSNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            hidden_states,
            normalized_shape=[hidden_states.size(-1)],
            weight=self.weight + 1.0,
            eps=self.eps,
        )


class _Qwen3_5TextModel(Qwen3_5TextModel):
    def __init__(self, config: Qwen3_5TextConfig):
        super(Qwen3_5TextModel, self).__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleDict(
            {
                str(layer_idx): _Qwen3_5DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            }
        )
        self.norm = _Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if "cu_seq_lens_q" in kwargs:
            kwargs["seq_idx_q"] = get_seq_idx(kwargs["cu_seq_lens_q"])

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = Qwen3_5DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # mrope: the hard coded `4` is for text, temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx in range(self.config.num_hidden_layers):
            if str(layer_idx) not in self.layers:
                continue

            decoder_layer = self.layers[str(layer_idx)]
            layer_mask = linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        if hasattr(self, "norm"):
            hidden_states = self.norm(hidden_states)

        return Qwen3_5ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class _Qwen3_5Model(Qwen3_5Model):
    def get_multimodal_features(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ):
        in_channels = self.config.vision_config.in_channels
        patch_size = self.config.vision_config.patch_size
        temporal_patch_size = self.config.vision_config.temporal_patch_size
        patch_channels = patch_size * patch_size * in_channels * temporal_patch_size

        if pixel_values is None:
            pixel_values = torch.zeros(0, patch_channels, dtype=self.visual.dtype, device=self.visual.device)
            image_grid_thw = torch.zeros((0, 3), dtype=torch.long, device=self.visual.device)

        if pixel_values_videos is None:
            pixel_values_videos = torch.zeros(
                0, patch_channels, dtype=self.visual.dtype, device=self.visual.device
            )
            video_grid_thw = torch.zeros((0, 3), dtype=torch.long, device=self.visual.device)

        pixel_values = torch.cat([pixel_values, pixel_values_videos], dim=0).type(self.visual.dtype)
        grid_thw = torch.cat([image_grid_thw, video_grid_thw], dim=0)
        outputs = self.visual(pixel_values, grid_thw=grid_thw)
        visual_embeds = outputs.pooler_output

        num_image_tokens = image_grid_thw.prod(dim=1).sum() // self.visual.spatial_merge_size**2
        image_embeds = visual_embeds[:num_image_tokens]
        video_embeds = visual_embeds[num_image_tokens:]

        return image_embeds, video_embeds

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
    ):
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        special_image_mask = special_image_mask.unsqueeze(-1).to(inputs_embeds.device)
        special_video_mask = special_video_mask.unsqueeze(-1).to(inputs_embeds.device)

        return special_image_mask, special_video_mask

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen3_5ModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            sequence_splitter, kwargs = context_parallel.get_sequence_splitter(
                input_ids.size(1), attn_kwargs=kwargs,
            )
            inputs_embeds = self.get_input_embeddings()(input_ids[:, sequence_splitter])
        else:
            assert pixel_values is None and pixel_values_videos is None
            image_mask = video_mask = None

        if hasattr(self, "visual") and (self.training or pixel_values is not None or pixel_values_videos is not None):
            image_embeds, video_embeds = self.get_multimodal_features(
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )

            image_mask, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)

            image_splitter = context_parallel.get_multimodal_splitter(image_mask, sequence_splitter)
            video_splitter = context_parallel.get_multimodal_splitter(video_mask, sequence_splitter)
            image_mask, video_mask = image_mask[:, sequence_splitter], video_mask[:, sequence_splitter]
            image_embeds, video_embeds = image_embeds[image_splitter], video_embeds[video_splitter]

            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        else:
            image_embeds = video_embeds = None

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        return Qwen3_5ModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )

    def floating_point_ops(self, inputs: Dict[str, Any]):
        raise NotImplementedError
    
    def apply_pipeline_parallel(
        self,
        num_stages: int,
        stage_index: int,
        reduced_layers_in_stage_zero: int = 0,
    ):
        num_layers = len(self.language_model.layers)
        assert (num_layers + reduced_layers_in_stage_zero) % num_stages == 0
        assert not self.config.text_config.tie_word_embeddings

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

        for layer_idx in list(self.language_model.layers.keys()):
            layer_idx = int(layer_idx)
            if layer_idx < start_index or layer_idx >= end_index:
                del self.language_model.layers[str(layer_idx)]

        if stage_index > 0:
            del self.visual, self.language_model.embed_tokens

        if stage_index < num_stages - 1:
            del self.language_model.norm


class _Qwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen3_5CausalLMOutputWithPast:
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
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )

        hidden_states = outputs[0]

        loss, logits = None, None
        if hasattr(self, "lm_head"):
            if labels is not None:
                sequence_splitter, _ = context_parallel.get_sequence_splitter(
                    labels.size(1), attn_kwargs=kwargs,
                )
                loss = self.loss_function(
                    hidden_states=hidden_states,
                    lm_head=self.lm_head,
                    position_ids=position_ids,
                    labels=labels,
                    num_items_in_batch=kwargs["num_items_in_batch"],
                    sequence_splitter=sequence_splitter,
                )
            else:
                slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
                logits = self.lm_head(hidden_states[:, slice_indices, :])

        return Qwen3_5CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            last_hidden_state=hidden_states,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def floating_point_ops(self, inputs: Dict[str, Any]):
        return self.model.floating_point_ops(inputs)

    def apply_pipeline_parallel(
        self,
        num_stages: int,
        stage_index: int,
        reduced_layers_in_stage_zero: int = 0,
    ):
        self.model.apply_pipeline_parallel(
            num_stages=num_stages,
            stage_index=stage_index,
            reduced_layers_in_stage_zero=reduced_layers_in_stage_zero,
        )
        if stage_index < num_stages - 1:
            del self.lm_head


def apply_monkey_patch():
    transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5Attention = _Qwen3_5Attention
    transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet = _Qwen3_5GatedDeltaNet
    transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5RMSNorm = _Qwen3_5RMSNorm
    transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5DecoderLayer = _Qwen3_5DecoderLayer

    transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5VisionModel = _Qwen3_5VisionModel
    transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel = _Qwen3_5TextModel

    transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5Model = _Qwen3_5Model
    transformers.models.auto.modeling_auto.MODEL_MAPPING[Qwen3_5Config] = _Qwen3_5Model

    transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForConditionalGeneration = _Qwen3_5ForConditionalGeneration
    transformers.models.auto.modeling_auto.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING[Qwen3_5Config] = _Qwen3_5ForConditionalGeneration
