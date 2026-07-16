from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    BaseModelOutputWithDeepstackFeatures,
    BaseModelOutputWithPast,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionModel,
    create_causal_mask,
    Qwen3VLTextAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from ...utils import context_parallel


@dataclass
class Qwen3VLCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class _Qwen3VLVisionModel(Qwen3VLVisionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_checkpointing_interval = None

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithDeepstackFeatures:
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

        def forward_chunk(hidden_states, deepstack_feature_lists, start, end):
            for i in range(start, end):
                if i >= len(self.blocks):
                    break

                hidden_states = self.blocks[i](
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

                if i in self.deepstack_visual_indexes:
                    deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(i)](
                        hidden_states
                    )
                    deepstack_feature = dispatcher.combine(deepstack_feature)
                    if fake_forward:
                        deepstack_feature = deepstack_feature[:0]
                    deepstack_feature_lists.append(deepstack_feature)

            return hidden_states

        deepstack_feature_lists = []
        start = 0

        while start < len(self.blocks):
            if self.gradient_checkpointing_interval is not None and self.training:
                end = start + self.gradient_checkpointing_interval
                hidden_states = checkpoint(
                    forward_chunk,
                    hidden_states,
                    deepstack_feature_lists,
                    start,
                    end,
                    use_reentrant=True,
                )
            else:
                end = start + len(self.blocks)
                hidden_states = forward_chunk(
                    hidden_states,
                    deepstack_feature_lists,
                    start,
                    end,
                )
            start = end

        merged_hidden_states = self.merger(hidden_states)
        merged_hidden_states = dispatcher.combine(merged_hidden_states)
        if fake_forward:
            merged_hidden_states = merged_hidden_states[:0]

        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
            deepstack_features=deepstack_feature_lists,
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


class _Qwen3VLTextAttention(Qwen3VLTextAttention):
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

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
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
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class _Qwen3VLTextRMSNorm(Qwen3VLTextRMSNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            hidden_states,
            normalized_shape=[hidden_states.size(-1)],
            weight=self.weight,
            eps=self.variance_epsilon,
        )


class _Qwen3VLTextModel(Qwen3VLTextModel):
    def __init__(self, config):
        super(Qwen3VLTextModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleDict(
            {
                str(layer_idx): Qwen3VLTextDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            }
        )
        self.norm = _Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

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
        # args for deepstack
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | BaseModelOutputWithPast:
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

        # the hard coded `4` is for text, temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        attention_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
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


class _Qwen3VLModel(Qwen3VLModel):
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
        vision_output = self.visual(pixel_values, grid_thw=grid_thw)

        num_image_tokens = image_grid_thw.prod(dim=1).sum() // self.visual.spatial_merge_size**2
        image_embeds = vision_output.pooler_output[:num_image_tokens]
        video_embeds = vision_output.pooler_output[num_image_tokens:]
        deepstack_image_embeds = [x[:num_image_tokens] for x in vision_output.deepstack_features]
        deepstack_video_embeds = [x[num_image_tokens:] for x in vision_output.deepstack_features]

        return image_embeds, video_embeds, deepstack_image_embeds, deepstack_video_embeds

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
    ) -> tuple | Qwen3VLModelOutputWithPast:
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
            image_embeds, video_embeds, deepstack_image_embeds, deepstack_video_embeds = self.get_multimodal_features(
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

            for i, (img_embed, vid_embed) in enumerate(zip(deepstack_image_embeds, deepstack_video_embeds)):
                deepstack_image_embeds[i] = img_embed[image_splitter]
                deepstack_video_embeds[i] = vid_embed[video_splitter]

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
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )

    def floating_point_ops(self, inputs: Dict[str, Any]):
        decoder_flops = self.visual.floating_point_ops(inputs)

        grid_thw = [inputs[x] for x in ["image_grid_thw", "video_grid_thw"] if x in inputs]
        if len(grid_thw) > 0:
            encoder_flops = self.visual.floating_point_ops({"grid_thw": torch.cat(grid_thw, dim=0)})
        else:
            encoder_flops = 0

        return decoder_flops + encoder_flops

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


class _Qwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    accepts_loss_kwargs = True

    @can_return_tuple
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

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            last_hidden_state=hidden_states,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
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
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionModel = _Qwen3VLVisionModel
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextAttention = _Qwen3VLTextAttention
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextRMSNorm = _Qwen3VLTextRMSNorm
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextModel = _Qwen3VLTextModel

    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLModel = _Qwen3VLModel
    transformers.models.auto.modeling_auto.MODEL_MAPPING[Qwen3VLConfig] = _Qwen3VLModel

    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLForConditionalGeneration = _Qwen3VLForConditionalGeneration
    transformers.models.auto.modeling_auto.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING[Qwen3VLConfig] = (
        _Qwen3VLForConditionalGeneration
    )
