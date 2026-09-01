import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import fully_shard
from transformers.cache_utils import StaticCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
)
from transformers.processing_utils import Unpack
from transformers.utils import can_return_tuple
from transformers.utils.generic import TransformersKwargs

from .configuration_rynn_brain_vla import RynnBrainVLAConfig


@dataclass
class RynnBrainVLAModelOutputWithPast(Qwen3VLModelOutputWithPast):
    actions: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


class RynnBrainVLACache(StaticCache):
    _is_frozen = False

    def freeze(self):
        self._is_frozen = True

    def unfreeze(self):
        self._is_frozen = False

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._is_frozen:
            keys = torch.cat([self.layers[layer_idx].keys, key_states], dim=-2)
            values = torch.cat([self.layers[layer_idx].values, value_states], dim=-2)
            return keys, values

        # transformers>=5.x's StaticLayer.update ignores the passed `cache_position`
        # and instead appends using an internal `cumulative_length` counter. The
        # flow-matching decode loop re-runs the action tokens many times and must
        # *overwrite* the same action slots each iteration (via the explicit
        # `cache_position`), not append. So honor `cache_position` directly here,
        # reproducing the pre-5.x StaticCache overwrite semantics.
        layer = self.layers[layer_idx]
        if not layer.is_initialized:
            layer.lazy_initialization(key_states, value_states)

        cache_position = (cache_kwargs or {}).get("cache_position", None)
        if cache_position is None:
            cache_position = torch.arange(key_states.shape[-2], device=layer.keys.device) + int(
                layer.cumulative_length
            )
        cache_position = cache_position.to(layer.keys.device)

        layer.keys.index_copy_(2, cache_position, key_states)
        layer.values.index_copy_(2, cache_position, value_states)

        # Advance the cached-prefix length only on the prefill pass; decode
        # overwrites in place and must leave `get_seq_length` (== prefix length)
        # unchanged so every integration step keeps taking the decode branch.
        if int(layer.cumulative_length) == 0:
            prefix_len = int(cache_position[-1]) + 1
            if isinstance(layer.cumulative_length, torch.Tensor):
                layer.cumulative_length.fill_(prefix_len)
            else:
                layer.cumulative_length = prefix_len

        return layer.keys, layer.values


def _attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    num_action_tokens: int,
    past_key_values: Optional[RynnBrainVLACache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        if past_key_values.get_seq_length(self.layer_idx) == 0:
            # prefilling
            prefix_query_states = query_states
            prefix_key_states = key_states
            prefix_value_states = value_states
            prefix_attention_mask = attention_mask
            action_query_states = None
        else:
            # decoding
            prefix_query_states = None
            action_query_states = query_states
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
    else:
        # training
        assert query_states.size(2) == key_states.size(2)

        num_vlm_tokens = hidden_states.size(1) - num_action_tokens
        prefix_query_states = query_states[:, :, :num_vlm_tokens]
        prefix_key_states = key_states[:, :, :num_vlm_tokens]
        prefix_value_states = value_states[:, :, :num_vlm_tokens]

        if attention_mask is None:
            prefix_attention_mask = None
        elif attention_mask.ndim == 4:
            prefix_attention_mask = attention_mask[:, :, :num_vlm_tokens, :num_vlm_tokens]
        else:
            prefix_attention_mask = attention_mask[:, :num_vlm_tokens]

        if num_action_tokens == 0:
            action_query_states = None
        else:
            action_query_states = query_states[:, :, num_vlm_tokens:]

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output = None
    if prefix_query_states is not None:
        attn_output, _ = attention_interface(
            self,
            prefix_query_states,
            prefix_key_states,
            prefix_value_states,
            prefix_attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=False,
            is_causal=True,
            **kwargs,
        )

    if action_query_states is not None:
        if attention_mask is None:
            action_attention_mask = None
        elif attention_mask.ndim == 4:
            # 4D additive mask (B, 1, q_total, kv_total): slice action query rows
            # and zero out action-to-action region to allow bidirectional attention
            action_attention_mask = attention_mask[:, :, num_vlm_tokens:, :]
            action_attention_mask = action_attention_mask.clone()
            action_attention_mask[:, :, :, num_vlm_tokens:] = 0
        else:
            action_attention_mask = attention_mask

        action_attn_output, _ = attention_interface(
            self,
            action_query_states,
            key_states,
            value_states,
            action_attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=False,
            is_causal=False,
            **kwargs,
        )

        if attn_output is not None:
            attn_output = torch.cat([attn_output, action_attn_output], dim=1)
        else:
            attn_output = action_attn_output

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, _


def _create_sinusoidal_pos_embedding(
    time: torch.tensor,
    dimension: int,
    min_period: float = 0.004,
    max_period: float = 4.0,
    device: str = "cpu",
):
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


class RynnBrainVLAModel(Qwen3VLModel):
    config: RynnBrainVLAConfig
    model_type = "rynn_brain_vla"
    _default_key_mapping = {
        "model.visual": "visual",
        "model.language_model": "language_model",
    }

    def __init__(self, config: RynnBrainVLAConfig):
        super(Qwen3VLModel, self).__init__(config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

        self.state_proj = nn.Linear(config.action_dim, config.text_config.hidden_size)
        self.action_in_proj = nn.Linear(config.action_dim, config.text_config.hidden_size)
        self.action_time_proj = nn.Sequential(
            nn.Linear(config.text_config.hidden_size * 2, config.text_config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size),
        )
        self.action_out_proj = nn.Linear(config.text_config.hidden_size, config.action_dim)

        for layer in self.language_model.layers:
            layer.self_attn.forward = _attention_forward.__get__(layer.self_attn)

        # Initialize weights and apply final processing
        self.post_init()

    def sample_noise(self, shape):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=self.device,
        )
        return noise

    def sample_time(self, batch_size):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((batch_size,)).to(device=self.device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def preprocess_actions(
        self,
        actions: Optional[torch.FloatTensor] = None,
        times: Optional[torch.FloatTensor] = None,
    ):
        targets = None
        if self.training:
            assert actions is not None

            if times is None:
                times = self.sample_time(actions.size(0))

            expanded_times = times[:, None, None]
            noises = self.sample_noise(actions.shape)
            clean_actions = actions
            actions = expanded_times * noises + (1 - expanded_times) * clean_actions
            targets = noises - clean_actions

        return actions, targets, times

    def get_state_features(
        self,
        states: Optional[torch.FloatTensor] = None,
    ):
        assert states.ndim == 3
        assert states.size(-1) == self.config.action_dim
        return self.state_proj(states.type(self.dtype))

    def get_action_features(
        self,
        actions: Optional[torch.FloatTensor] = None,
        times: Optional[torch.FloatTensor] = None,
    ):
        assert actions.ndim == 3
        assert actions.size(-1) == self.config.action_dim

        action_embeds = self.action_in_proj(actions.type(self.dtype))
        time_embeds = _create_sinusoidal_pos_embedding(
            times.to(action_embeds.device),
            action_embeds.size(-1),
            device=action_embeds.device,
        ).type(dtype=self.dtype)
        time_embeds = time_embeds.unsqueeze(1).expand_as(action_embeds)
        action_embeds = torch.cat([action_embeds, time_embeds], dim=-1)
        action_embeds = self.action_time_proj(action_embeds)
        return action_embeds

    def get_state_token_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        features: Optional[torch.FloatTensor] = None,
    ):
        if input_ids is None:
            special_token_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.state_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_token_mask = special_token_mask.all(-1)
        else:
            special_token_mask = input_ids == self.config.state_token_id

        n_special_tokens = special_token_mask.sum()
        special_token_mask = special_token_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if features is not None and inputs_embeds[special_token_mask].numel() != features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_special_tokens}, features {features.shape[0]}"
            )

        return special_token_mask

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[RynnBrainVLACache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        action_mask: Optional[torch.BoolTensor] = None,
        states: Optional[torch.FloatTensor] = None,
        times: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, RynnBrainVLAModelOutputWithPast]:
        if inputs_embeds is None and input_ids is None:
            assert actions is not None
            inputs_embeds = torch.zeros(
                (actions.size(0), 0, self.config.text_config.hidden_size),
                dtype=self.dtype,
                device=self.get_input_embeddings().weight.device,
            )
        elif inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_outputs: BaseModelOutputWithDeepstackFeatures = self.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            )
            image_embeds = image_outputs.pooler_output
            deepstack_image_embeds = image_outputs.deepstack_features
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs: BaseModelOutputWithDeepstackFeatures = self.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True
            )
            video_embeds = video_outputs.pooler_output
            deepstack_video_embeds = video_outputs.deepstack_features
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        state_mask = self.get_state_token_mask(input_ids, inputs_embeds=inputs_embeds)
        if state_mask.any():
            state_embeds = self.get_state_features(states)
            state_embeds = state_embeds.view(-1, state_embeds.size(-1)).to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(state_mask, state_embeds)
        state_mask = state_mask[..., 0]

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        actions, targets, times = self.preprocess_actions(actions, times)

        if actions is not None and action_mask is not None:
            actions[~action_mask] = 0.0

        num_action_tokens = 0
        if actions is not None:
            action_embeds = self.get_action_features(actions, times).to(inputs_embeds.device, inputs_embeds.dtype)
            action_position_ids = (
                torch.arange(
                    start=1,
                    end=actions.size(1) + 1,
                    step=1,
                    dtype=position_ids.dtype,
                    device=position_ids.device,
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(3, actions.size(0), 1)
            )

            if "cu_seq_lens_q" in kwargs:
                kwargs["cu_seq_lens_action"] = torch.arange(
                    start=0,
                    end=actions.size(0) * actions.size(1) + 1,
                    step=actions.size(1),
                    dtype=torch.int32,
                    device=self.device,
                )
                kwargs["max_length_action"] = actions.size(1)
                action_embeds = action_embeds.flatten(0, 1).unsqueeze(0)
                action_position_ids += position_ids[:, 0, kwargs["cu_seq_lens_q"][1:] - 1].unsqueeze(-1)
                action_position_ids = action_position_ids.flatten(1, 2).unsqueeze(1)
            else:
                action_position_ids += position_ids[:, :, -1:]

            num_action_tokens = action_embeds.size(1)
            inputs_embeds = torch.cat([inputs_embeds, action_embeds], dim=1)

            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (0, num_action_tokens), value=1)

            if position_ids.size(-1) != inputs_embeds.size(1):
                position_ids = torch.cat([position_ids, action_position_ids], dim=-1)

            if visual_pos_masks is not None:
                visual_pos_masks = F.pad(
                    visual_pos_masks, (0, inputs_embeds.size(1) - visual_pos_masks.size(1)), value=False
                )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            num_action_tokens=num_action_tokens,
            **kwargs,
        )

        loss = pred_actions = None
        if actions is not None:
            action_hidden_states = outputs.last_hidden_state[:, -num_action_tokens:]
            pred_actions = self.action_out_proj(action_hidden_states)
            pred_actions = pred_actions.view(actions.shape)
            if targets is not None:
                if action_mask is not None:
                    pred_masked = pred_actions.type(targets.dtype)[action_mask]
                    targets_masked = targets[action_mask]
                    loss = F.mse_loss(pred_masked, targets_masked)
                else:
                    loss = F.mse_loss(pred_actions.type(targets.dtype), targets)

        return RynnBrainVLAModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=rope_deltas,
            actions=pred_actions,
            loss=loss,
        )

    def apply_fully_sharded_data_parallel(
        self,
        device_mesh: torch.distributed.DeviceMesh,
        expert_device_mesh: torch.distributed.DeviceMesh,
        mp_policy: torch.distributed.fsdp.MixedPrecisionPolicy,
        reshard_after_forward: bool = False,
    ):
        # This model is dense, so ``expert_device_mesh`` is accepted (build_model
        # always passes it) but unused, as in the dense qwen3 implementations.
        fsdp_config = {
            "reshard_after_forward": reshard_after_forward,
            "mp_policy": mp_policy,
        }

        if hasattr(self, "visual"):
            fully_shard(self.visual, mesh=device_mesh["dp"], **fsdp_config)

        if hasattr(self.language_model, "embed_tokens"):
            fully_shard(self.language_model.embed_tokens, mesh=device_mesh["dp"], **fsdp_config)

        # Upstream Qwen3VLTextModel.layers is an nn.ModuleList; the repo-local
        # qwen3_vl override uses an nn.ModuleDict. Accept both.
        layers = self.language_model.layers
        for layer in layers.values() if isinstance(layers, nn.ModuleDict) else layers:
            fully_shard(layer, mesh=device_mesh["dp"], **fsdp_config)

        if hasattr(self.language_model, "norm"):
            fully_shard(self.language_model.norm, mesh=device_mesh["dp"], **fsdp_config)

        # state_proj / action_in_proj / action_time_proj / action_out_proj are left
        # to the root group below: they are tiny and needed at both ends of forward,
        # so a per-module all-gather would only add latency. The root group ends up
        # holding exactly the action head.
        fully_shard(self, mesh=device_mesh["dp"], **fsdp_config)
