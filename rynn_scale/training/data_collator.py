from typing import Any, Dict, List

import torch
from transformers import ProcessorMixin

from ..utils import context_parallel


class DataCollator(object):
    def __init__(
        self,
        processor: ProcessorMixin,
        sequence_packing: bool,
    ):
        self.processor = processor
        self.sequence_packing = sequence_packing

    def _collate_mm_inputs(self, instances):
        mm_input_names = set(
            self.processor.image_processor.model_input_names + self.processor.video_processor.model_input_names
        )

        mm_inputs = {}
        for key in mm_input_names:
            data_list = [instance[key] for instance in instances if key in instance]
            if len(data_list) > 0:
                mm_inputs[key] = torch.cat(data_list, dim=0)

        return mm_inputs

    def _collate_fn_packing(self, instances):
        input_ids_list, position_ids_list, labels_list = [], [], []

        cu_seq_lens = [0]
        max_length = 0

        for instance in instances:
            input_ids = instance.get("input_ids")
            position_ids = instance.get("position_ids", torch.arange(instance["input_ids"].size(-1)).unsqueeze(0))
            labels = instance.get("labels", None)

            if "labels" in instance:
                labels = instance["labels"].clone()
            else:
                labels = torch.full_like(input_ids, fill_value=-100, dtype=torch.long)
            labels[..., 0] = -100

            input_ids, _, position_ids, labels = context_parallel.pad_sequence(
                input_ids,
                position_ids=position_ids,
                labels=labels,
            )

            input_ids_list.append(input_ids)
            position_ids_list.append(position_ids)
            labels_list.append(labels)

            seq_len = input_ids.size(-1)
            cu_seq_lens.append(cu_seq_lens[-1] + seq_len)
            max_length = max(max_length, seq_len)

        cu_seq_lens = torch.as_tensor(cu_seq_lens, dtype=torch.int32)

        batch = {
            "input_ids": torch.cat(input_ids_list, dim=-1),
            "position_ids": torch.cat(position_ids_list, dim=-1),
            "labels": torch.cat(labels_list, dim=-1),
            **self._collate_mm_inputs(instances),
            "cu_seq_lens_q": cu_seq_lens,
            "cu_seq_lens_k": cu_seq_lens,
            "max_length_q": max_length,
            "max_length_k": max_length,
        }

        if "actions" in instances[0]:
            batch["actions"] = torch.cat([instance["actions"] for instance in instances], dim=0)

        if "action_mask" in instances[0]:
            batch["action_mask"] = torch.cat([instance["action_mask"] for instance in instances], dim=0)

        if "states" in instances[0]:
            batch["states"] = torch.cat([instance["states"] for instance in instances], dim=0)

        if "data_index" in instances[0]:
            batch["data_indices"] = [instance["data_index"] for instance in instances]

        return batch

    def _collate_fn_padding(self, instances):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [instance["input_ids"][0] for instance in instances],
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
            padding_side="left",
        )

        if "attention_mask" in instances[0]:
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [instance["attention_mask"][0] for instance in instances],
                batch_first=True,
                padding_value=0,
                padding_side="left",
            )
        else:
            attention_mask = input_ids != self.processor.tokenizer.pad_token_id

        if "position_ids" in instances[0]:
            if instances[0]["position_ids"].ndim == 3:
                position_ids = torch.nn.utils.rnn.pad_sequence(
                    [instance["position_ids"][:, 0].transpose(0, 1) for instance in instances],
                    batch_first=True,
                    padding_value=1,
                    padding_side="left",
                ).permute(2, 0, 1)
            else:
                position_ids = torch.nn.utils.rnn.pad_sequence(
                    [instance["position_ids"][0] for instance in instances],
                    batch_first=True,
                    padding_value=1,
                    padding_side="left",
                )
        else:
            assert attention_mask.ndim == 2
            position_ids = attention_mask.cumsum(-1) - 1

        if "labels" in instances[0]:
            labels = torch.nn.utils.rnn.pad_sequence(
                [instance["labels"][0] for instance in instances],
                batch_first=True,
                padding_value=-100,
                padding_side="left",
            )
        else:
            labels = torch.full_like(input_ids, fill_value=-100, dtype=torch.long)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels,
            **self._collate_mm_inputs(instances),
        }

        if "actions" in instances[0]:
            batch["actions"] = torch.cat([instance["actions"] for instance in instances], dim=0)

        if "action_mask" in instances[0]:
            batch["action_mask"] = torch.cat([instance["action_mask"] for instance in instances], dim=0)

        if "states" in instances[0]:
            batch["states"] = torch.cat([instance["states"] for instance in instances], dim=0)

        if "data_index" in instances[0]:
            batch["data_indices"] = [instance["data_index"] for instance in instances]

        return batch

    def __call__(self, instances: List[Dict[str, Any]]):
        if self.sequence_packing:
            batch = self._collate_fn_packing(instances)
        else:
            batch = self._collate_fn_padding(instances)
        batch["use_cache"] = False
        return batch
