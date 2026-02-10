from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Union, Dict, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextModel as _Qwen3VLTextModel,
    Qwen3VLModel as _Qwen3VLModel,
    Qwen3VLForConditionalGeneration as _Qwen3VLForConditionalGeneration,
    Qwen3VLVisionModel as _Qwen3VLVisionModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLTextAttention,
    Qwen3VLTextMLP,
    Qwen3VLTextRMSNorm,
    Qwen3VLVisionAttention as _Qwen3VLVisionAttention,
    Qwen3VLVisionMLP,
    BaseModelOutputWithPast,
    Qwen3VLTextRotaryEmbedding,
    create_causal_mask,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.cache_utils import DynamicCache, Cache
from transformers.processing_utils import Unpack
from transformers.utils import is_torchdynamo_compiling
from transformers.utils.generic import TransformersKwargs, check_model_inputs

from ...utils.context_parallel import EncoderContextDispatcher


@dataclass
class Qwen3VLCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class Qwen3VLVisionAttention(_Qwen3VLVisionAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config._attn_implementation == "flash_attention_2":
            # Flash Attention 2: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config=config)
        self.mlp = Qwen3VLVisionMLP(config=config)
        self.gradient_checkpointing = False
        self.selective_gradient_checkpointing = False

    def _forward_attention(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        return hidden_states

    def _forward_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

    def _forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        hidden_states = self._forward_attention(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
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
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        args = dict(
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if self.gradient_checkpointing and self.training:
            if not self.selective_gradient_checkpointing:
                return self._gradient_checkpointing_func(partial(self._forward, **args), hidden_states)
        return self._forward(hidden_states, **args)


class Qwen3VLTextDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.gradient_checkpointing = False
        self.selective_gradient_checkpointing = False

        self.self_attn = Qwen3VLTextAttention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _forward_attention(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
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
            use_cache=use_cache,
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
        hidden_states = residual + hidden_states
        return hidden_states

    def _forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        hidden_states = self._forward_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
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
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        args = dict(
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        if self.gradient_checkpointing and self.training:
            if not self.selective_gradient_checkpointing:
                return self._gradient_checkpointing_func(partial(self._forward, **args), hidden_states)
        return self._forward(hidden_states, **args)


class Qwen3VLTextModel(_Qwen3VLTextModel):
    def __init__(self, config):
        super(_Qwen3VLTextModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleDict(
            {
                str(layer_idx): Qwen3VLTextDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            }
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

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


class Qwen3VLVisionModel(_Qwen3VLVisionModel):
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

        dispatcher = EncoderContextDispatcher(grid_thw=grid_thw, merge_size=self.spatial_merge_size)
        hidden_states = dispatcher.dispatch(hidden_states)
        rotary_pos_emb = dispatcher.dispatch(rotary_pos_emb)

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        if dispatcher.activated:
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

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature = dispatcher.combine(deepstack_feature)
                if fake_forward:
                    deepstack_feature = deepstack_feature[:0]
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)
        hidden_states = dispatcher.combine(hidden_states)
        if fake_forward:
            hidden_states = hidden_states[:0]

        return hidden_states, deepstack_feature_lists

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


class Qwen3VLModel(_Qwen3VLModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_channels = self.config.vision_config.in_channels
        patch_size = self.config.vision_config.patch_size
        temporal_patch_size = self.config.vision_config.temporal_patch_size
        self.patch_channels = patch_size * patch_size * in_channels * temporal_patch_size

    def get_multimodal_features(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ):
        if pixel_values is None:
            pixel_values = torch.zeros(0, self.patch_channels, dtype=self.visual.dtype, device=self.visual.device)
            image_grid_thw = torch.zeros((0, 3), dtype=torch.long, device=self.visual.device)

        if pixel_values_videos is None:
            pixel_values_videos = torch.zeros(
                0, self.patch_channels, dtype=self.visual.dtype, device=self.visual.device
            )
            video_grid_thw = torch.zeros((0, 3), dtype=torch.long, device=self.visual.device)

        pixel_values = torch.cat([pixel_values, pixel_values_videos], dim=0).type(self.visual.dtype)
        grid_thw = torch.cat([image_grid_thw, video_grid_thw], dim=0)
        visual_embeds, deepstack_visual_embeds = self.visual(pixel_values, grid_thw=grid_thw)

        num_image_tokens = image_grid_thw.prod(dim=1).sum() // self.visual.spatial_merge_size**2
        image_embeds = visual_embeds[:num_image_tokens]
        video_embeds = visual_embeds[num_image_tokens:]
        deepstack_image_embeds = [x[:num_image_tokens] for x in deepstack_visual_embeds]
        deepstack_video_embeds = [x[num_image_tokens:] for x in deepstack_visual_embeds]

        return image_embeds, video_embeds, deepstack_image_embeds, deepstack_video_embeds

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if hasattr(self, "visual") and (self.training or pixel_values is not None or pixel_values_videos is not None):
            image_embeds, video_embeds, deepstack_image_embeds, deepstack_video_embeds = self.get_multimodal_features(
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )

            image_mask, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        else:
            image_embeds = video_embeds = None
            deepstack_image_embeds = deepstack_video_embeds = None

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

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )


class Qwen3VLForConditionalGeneration(_Qwen3VLForConditionalGeneration):
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
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
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

        return Qwen3VLCausalLMOutputWithPast(
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

    def floating_point_ops(self, inputs: Dict[str, Any]):
        input_ids = inputs["input_ids"]

        hidden_size = self.config.text_config.hidden_size
        num_hidden_layers = self.config.text_config.num_hidden_layers
        num_attention_heads = self.config.text_config.num_attention_heads
        num_key_value_heads = self.config.text_config.num_key_value_heads
        intermediate_size = self.config.text_config.intermediate_size
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


def apply_monkey_patch():
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionBlock = Qwen3VLVisionBlock
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextDecoderLayer = Qwen3VLTextDecoderLayer
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextModel = Qwen3VLTextModel
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionModel = Qwen3VLVisionModel
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLModel = Qwen3VLModel

    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLForConditionalGeneration = Qwen3VLForConditionalGeneration
    transformers.models.auto.modeling_auto.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING[Qwen3VLConfig] = (
        Qwen3VLForConditionalGeneration
    )
