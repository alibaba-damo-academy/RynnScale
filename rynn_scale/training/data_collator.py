from typing import Dict, List, Any

import torch
from transformers import ProcessorMixin


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
        input_ids, position_ids, labels = [], [], []

        cu_seq_lens = [0]
        max_length = 0

        for instance in instances:
            input_ids.append(instance["input_ids"])
            if "position_ids" in instance:
                position_ids.append(instance["position_ids"])
            else:
                position_ids.append(torch.arange(instance["input_ids"].size(-1)).unsqueeze(0))
            tmp_labels = instance["labels"].clone()
            tmp_labels[..., 0] = -100
            labels.append(tmp_labels)

            seq_len = instance["input_ids"].size(-1)
            cu_seq_lens.append(cu_seq_lens[-1] + seq_len)
            max_length = max(max_length, seq_len)

        cu_seq_lens = torch.as_tensor(cu_seq_lens, dtype=torch.int32)

        batch = {
            "data_indices": [instance["data_index"] for instance in instances],
            "input_ids": torch.cat(input_ids, dim=-1),
            "position_ids": torch.cat(position_ids, dim=-1),
            "labels": torch.cat(labels, dim=-1),
            "use_cache": False,
            **self._collate_mm_inputs(instances),
            "cu_seq_lens_q": cu_seq_lens,
            "cu_seq_lens_k": cu_seq_lens,
            "max_length_q": max_length,
            "max_length_k": max_length,
        }

        return batch

    def _collate_fn_padding(self, instances):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [instance["input_ids"] for instance in instances],
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [instance["labels"] for instance in instances],
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )

        attention_mask = torch.zeros_like(input_ids)
        position_ids = torch.ones_like(input_ids)
        for i, instance in enumerate(instances):
            seq_len = instance["input_ids"].size(-1)
            if "attention_mask" in instance:
                attention_mask[i, :seq_len] = instance["attention_mask"]
            else:
                attention_mask[i, :seq_len] = 1

            if "position_ids" in instance:
                position_ids[i, :seq_len] = instance["position_ids"]
            else:
                position_ids[i, :seq_len] = torch.arange(seq_len)

        batch = {
            "data_indices": [instance["data_index"] for instance in instances],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels,
            **self._collate_mm_inputs(instances),
        }

        return batch

    def __call__(self, instances: List[Dict[str, Any]]):
        if self.sequence_packing:
            return self._collate_fn_packing(instances)
        return self._collate_fn_padding(instances)
