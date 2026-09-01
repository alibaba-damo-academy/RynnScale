import torch
from transformers import CONFIG_MAPPING

from ..constants import RobotType
from ..models.rynn_brain_vla import RynnBrainVLAConfig, RynnBrainVLAModel, RynnBrainVLAProcessor
from ..models.rynn_brain_vla.modeling_rynn_brain_vla import RynnBrainVLACache
from ..registry import INFERENCE_WRAPPER_REGISTRY
from ..utils.robot import RobotState
from .base import BaseVLAInferenceWrapper


@INFERENCE_WRAPPER_REGISTRY.register("rynn_brain_vla")
class RynnBrainVLAInferenceWrapper(BaseVLAInferenceWrapper):
    def load_model(self):
        assert self.attn_implementation == "flash_attention_2"
        model = RynnBrainVLAModel.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            device_map="auto",
        )
        return model

    def load_processor(self):
        CONFIG_MAPPING.register("rynn_brain_vla", RynnBrainVLAConfig, exist_ok=True)
        processor = RynnBrainVLAProcessor.from_pretrained(self.model_path)
        return processor

    def process(self, text, images, state, robot_type):
        return self.processor(
            text=text,
            robot_type=RobotType(robot_type),
            state=RobotState.from_dict(state),
            images=images,
        )

    def collate(self, batch):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [instance["input_ids"][0] for instance in batch],
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
            padding_side="right",
        )

        position_ids = torch.nn.utils.rnn.pad_sequence(
            [instance["position_ids"][:, 0].transpose(0, 1) for instance in batch],
            batch_first=True,
            padding_value=1,
            padding_side="right",
        ).permute(2, 0, 1)

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "states": torch.cat([instance["states"] for instance in batch], dim=0),
        }

        mm_input_names = set(
            self.processor.image_processor.model_input_names + self.processor.video_processor.model_input_names
        )

        for key in mm_input_names:
            data_list = [instance[key] for instance in batch if key in instance]
            if len(data_list) > 0:
                model_inputs[key] = torch.cat(data_list, dim=0)

        return model_inputs

    def prefill(self, model_inputs):
        cache_len = model_inputs["input_ids"].size(-1) + self.model.config.action_chunk_size
        past_key_values = RynnBrainVLACache(
            config=self.model.config,
            max_cache_len=cache_len,
        )
        self.model(
            **model_inputs,
            past_key_values=past_key_values,
        )
        return {"past_key_values": past_key_values}

    def decode(self, model_inputs, cache, num_steps, robot_type, prev_actions=None, delay_steps=None):
        # ``prev_actions`` / ``delay_steps`` are received and not used: this is a
        # flow-matching head with a fixed ``action_chunk_size`` and no Real-Time Chunking
        # conditioning yet, so it plans each chunk from the observation alone. Taking them
        # anyway is deliberate -- the caller's contract is that a chunk reaches past the
        # actions it is still holding, and this signature is where a reader can see that
        # this policy does not honour it on purpose. ``RobotAgent`` catches it if RTC is
        # switched on against this wrapper.
        batch_size = model_inputs["input_ids"].size(0)
        cache_length = model_inputs["input_ids"].size(-1)
        device = model_inputs["input_ids"].device
        action_chunk_size = self.model.config.action_chunk_size

        cache_position = torch.arange(action_chunk_size, device=device) + cache_length

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        # Zero the action dimensions each robot does not have, at every integration step.
        # Not optional: training feeds the model masked actions and scores only the masked
        # entries (``RynnBrainVLA.forward``), so sampling unmasked leaves those columns
        # carrying free noise where the model only ever saw exact zeros -- and they reach
        # the used columns through attention on the way. One mask per element, because
        # batched callers can be different robots.
        action_mask = torch.stack(
            [self.processor.get_action_mask(RobotType(rt), action_chunk_size) for rt in robot_type]
        ).to(device)

        actions_shape = (batch_size, action_chunk_size, self.model.config.action_dim)
        x_t = self.model.sample_noise(actions_shape)
        x_t[~action_mask] = 0.0
        times = torch.tensor(1.0, dtype=torch.float32, device=device).expand(batch_size)

        position_ids = torch.arange(1, action_chunk_size + 1).unsqueeze(0).unsqueeze(1)
        position_ids = position_ids.repeat(3, batch_size, 1)
        position_ids = position_ids.to(model_inputs["position_ids"]) + model_inputs["position_ids"][..., -1:]

        while times[0] >= -dt / 2:
            outputs = self.model(
                actions=x_t,
                times=times,
                past_key_values=cache["past_key_values"],
                cache_position=cache_position,
                position_ids=position_ids,
            )
            x_t += dt * outputs.actions
            x_t[~action_mask] = 0.0
            times = times + dt

        return x_t

    def post_process(self, action, state, robot_type):
        action = self.processor.post_process(
            action=action,
            state=RobotState.from_dict(state),
            robot_type=RobotType(robot_type),
        )
        return action.to_dict()
