import hashlib
import io
import json
import os
import pickle
import random
import traceback
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset, load_from_disk
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PretrainedConfig, ProcessorMixin

from ..registry import DATASET_REGISTRY
from ..utils import logging, oss
from .utils import get_rope_index

logger = logging.get_logger(__name__)


class SequenceLengthCalculator(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        processor: ProcessorMixin,
        mm_max_length: int,
        fps: int,
        max_frames: int,
    ):
        self.dataset = dataset
        self.processor = processor
        self.mm_max_length = mm_max_length
        self.fps = fps
        self.max_frames = max_frames

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        self.indices = list(range(rank, len(dataset), world_size))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = self.indices[index]
        data_dict = self.dataset[index]

        image_sizes, video_sizes = [], []
        for message in data_dict["conversation"]:
            for content in message["content"]:
                if content["type"] == "image":
                    image_sizes.append([content["height"], content["width"]])
                elif content["type"] == "video":
                    num_frames = min(int(content["duration"] * self.fps), self.max_frames)
                    video_sizes.append([num_frames, content["height"], content["width"]])

        try:
            info = self.processor._get_num_multimodal_tokens(
                image_sizes=image_sizes if len(image_sizes) else None,
                video_sizes=video_sizes if len(video_sizes) else None,
                mm_max_length=self.mm_max_length,
            )
            mm_sequence_length = 0
            if info.num_image_tokens is not None:
                mm_sequence_length += sum(info.num_image_tokens)
            if info.num_video_tokens is not None:
                mm_sequence_length += sum(info.num_video_tokens)
        except Exception:
            traceback.print_exc()
            mm_sequence_length = self.mm_max_length

        return index, data_dict["text_sequence_length"] + mm_sequence_length


@DATASET_REGISTRY.register()
class VLMDataset(Dataset):
    def __init__(
        self,
        model_config: PretrainedConfig,
        processor: ProcessorMixin,
        data_path: Optional[List[str]],
        data_mixture: Optional[str],
        model_max_length: int,
        mm_max_length: int,
        fps: int,
        max_frames: int,
        seed: int,
    ):
        if torch.distributed.is_initialized():
            seed = torch.tensor(seed, device="cuda")
            torch.distributed.broadcast(seed, src=0)
            seed = seed.item()

        self.model_config = model_config
        self.processor = processor
        self.data_path = data_path
        self.data_mixture = data_mixture
        self.model_max_length = model_max_length
        self.mm_max_length = mm_max_length
        self.fps = fps
        self.max_frames = max_frames
        self.seed = seed

        self._dataset = self._load_data()

    def get_sequence_lengths(self, num_workers, cache_dir):
        if torch.distributed.get_rank() == 0:
            pickled_bytes = pickle.dumps(self._dataset, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=False)
            md5_bytes = list(hashlib.md5(pickled_bytes).digest())
        else:
            md5_bytes = [0 for _ in range(16)]
        md5_bytes = torch.as_tensor(md5_bytes, dtype=torch.uint8, device="cuda")
        torch.distributed.broadcast(md5_bytes, src=0)

        uid = bytes(md5_bytes.tolist()).hex()
        cache_file = os.path.join(cache_dir, f"{uid}.pkl" )

        if cache_dir.startswith("oss://") and oss.object_exists(cache_file):
            with oss.get_object(cache_file) as result:
                buffer = io.BytesIO(result.read())
                lengths = pickle.load(buffer)
                buffer.close()
            return lengths
        elif os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                lengths = pickle.load(f)
            return lengths

        calculator = SequenceLengthCalculator(
            self._dataset,
            processor=self.processor,
            mm_max_length=self.mm_max_length,
            fps=self.fps,
            max_frames=self.max_frames,
        )

        dataloader = DataLoader(
            calculator,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: x[0],
        )

        lengths = [0 for _ in range(len(self._dataset))]
        for i, length in tqdm(
            dataloader,
            desc="Calculate sequence lengths",
            disable=torch.distributed.get_rank() > 0,
        ):
            lengths[i] = length

        lengths = torch.as_tensor(lengths, dtype=torch.int32, device="cuda")
        torch.distributed.all_reduce(lengths, op=torch.distributed.ReduceOp.SUM)
        lengths = lengths.tolist()

        assert len(lengths) == len(self._dataset)

        if torch.distributed.get_rank() == 0:
            if cache_dir.startswith("oss://"):
                with io.BytesIO() as buffer:
                    pickle.dump(lengths, buffer)
                    buffer.seek(0)
                    oss.put_object(cache_file, buffer)
            else:
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump(lengths, f)

        return lengths

    def _load_data(self):
        if self.data_mixture is not None:
            assert self.data_path is None, "`data_path` and `data_mixture` cannot be used simultaneously."
            with open(self.data_mixture, "r") as f:
                data_mixture = json.load(f)
            data_path = [x["data_path"] for x in data_mixture]
            sampling_ratios = [x.get("sampling_ratio", 1.0) for x in data_mixture]
        elif self.data_path is not None:
            data_path = self.data_path
            sampling_ratios = [1.0] * len(data_path)
        else:
            raise ValueError

        datasets = []
        for path, sampling_ratio in zip(data_path, sampling_ratios):
            if os.path.isdir(path):
                dataset = load_from_disk(path)
                assert sampling_ratio == 1.0
                datasets.append(dataset)
                continue

            if path.endswith(".csv"):
                data_format = "csv"
            elif path.endswith(".jsonl"):
                data_format = "json"
            elif path.endswith(".parquet"):
                data_format = "parquet"
            elif path.endswith(".arrow"):
                data_format = "arrow"
            elif path.endswith(".h5"):
                data_format = "hdf5"
            else:
                raise ValueError(f"Unsupported data format: {path}")

            dataset = load_dataset(data_format, data_files=path)["train"]

            if sampling_ratio < 1.0:
                generator = torch.Generator(device="cuda")
                generator.manual_seed(self.seed)
                num_samples = round(len(dataset) * sampling_ratio)
                sample_indices = torch.randperm(len(dataset), device="cuda", generator=generator)[:num_samples]
                if torch.distributed.is_initialized():
                    torch.distributed.broadcast(sample_indices, src=0)

                dataset = dataset.select(sample_indices.cpu())

            datasets.append(dataset)

        return concatenate_datasets(datasets)

    def _convert_conversation(self, conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        new_conversation = []
        for message in conversation:
            new_contents = []
            # TODO: add function call
            for content in message["content"]:
                if content["type"] == "image":
                    image = content["image"]
                    new_contents.append({"type": "image", "image": image})
                elif content["type"] == "video":
                    video = content["video"]
                    if video.startswith("[") and video.endswith("]"):
                        # Parse frame list from json string
                        video = json.loads(video)
                    new_contents.append({"type": "video", "video": video})
                else:
                    new_contents.append(
                        {
                            "type": content["type"],
                            content["type"]: content[content["type"]],
                        }
                    )
            message = {"role": message["role"], "content": new_contents}
            new_conversation.append(message)
        return new_conversation

    def _preprocess(self, data_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        conversation = self._convert_conversation(data_dict["conversation"])

        model_inputs = self.processor.apply_chat_template(
            conversation=conversation,
            mm_max_length=self.mm_max_length,
            fps=self.fps,
            max_frames=self.max_frames,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            return_labels=True,
        )

        assert model_inputs["input_ids"].size(-1) <= self.model_max_length, (
            f"Sequence length ({model_inputs['input_ids'].size(-1)}) exceeds model max length ({self.model_max_length})"
        )

        model_inputs["position_ids"] = get_rope_index(
            model_config=self.model_config,
            **model_inputs,
        )

        return model_inputs

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        try:
            data_dict = self._preprocess(self._dataset[index])
            data_dict["data_index"] = index
        except Exception:
            traceback.print_exc()
            # Ensuring deterministic for tp/pp
            local_rng = random.Random(index)
            backup_idx = local_rng.randint(0, len(self) - 1)
            logger.warning(f"Encounted error when process {index}-th example, use {backup_idx}-th example instead!!!")
            return self.__getitem__(backup_idx)
        return data_dict

    def __len__(self):
        return len(self._dataset)

    def __repr__(self):
        return self._dataset.__repr__()


@DATASET_REGISTRY.register()
class PseudoVLMDataset(VLMDataset):
    def _generate_sizes(self, generator: torch.Generator):
        while True:
            image_size = torch.randint(256, 1536, (2,), generator=generator).tolist()
            info = self.processor._get_num_multimodal_tokens(
                image_sizes=[image_size],
                mm_max_length=self.mm_max_length,
            )
            mm_sequence_length = sum(info.num_image_tokens)

            max_text_length = self.model_max_length - mm_sequence_length - 100
            if max_text_length > 0:
                break

        answer_length = torch.randint(0, min(max_text_length - 1, 512), size=(1,), generator=generator) + 1
        question_length = torch.randint(0, max_text_length - answer_length, size=(1,), generator=generator) + 1

        return [image_size], question_length.item(), answer_length.item()

    def _load_data(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        self.num_samples = 10000

        pbar = tqdm(
            total=self.num_samples,
            desc="Generating pseudo dataset",
            disable=torch.distributed.is_initialized() and torch.distributed.get_rank() != 0,
        )

        data_list = []

        for i in range(self.num_samples):
            image_sizes, question_length, answer_length = self._generate_sizes(generator=generator)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        *[
                            {"type": "image", "image": None, "height": image_size[0], "width": image_size[1]}
                            for image_size in image_sizes
                        ],
                        {"type": "text", "text": None, "length": question_length},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": None, "length": answer_length},
                    ],
                },
            ]

            data_list.append(
                {
                    "text_sequence_length": question_length + answer_length,
                    "conversation": conversation,
                }
            )

            pbar.update(1)

        pbar.close()

        return HFDataset.from_list(data_list)

    def _convert_conversation(self, conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        hash_obj = hashlib.sha256(json.dumps(conversation, sort_keys=True).encode("utf-8"))
        hash_hex = hash_obj.hexdigest()

        generator = torch.Generator()
        generator.manual_seed(int(hash_hex, 16) % (2**32))

        new_conversation = []
        for message in conversation:
            new_contents = []
            for content in message["content"]:
                if content["type"] == "image":
                    image = torch.randint(
                        0,
                        256,
                        (content["height"], content["width"], 3),
                        dtype=torch.uint8,
                        generator=generator,
                    )
                    image = Image.fromarray(image.numpy())
                    new_contents.append({"type": "image", "image": image})
                elif content["type"] == "text":
                    length = content["length"]
                    token_ids = torch.randint(0, 128, size=(length,), dtype=torch.long, generator=generator)
                    new_contents.append({"type": "text", "text": self.processor.decode(token_ids)})
                else:
                    raise ValueError(f"Unsupported content type: {content['type']}")
            new_conversation.append({"role": message["role"], "content": new_contents})

        return new_conversation

    def __repr__(self):
        return f"{self.__class__.__name__}(num_samples={len(self)})"


@DATASET_REGISTRY.register()
class PseudoVLMDatasetFixedLength(PseudoVLMDataset):
    def _generate_sizes(self, generator: torch.Generator):
        image_size = [512, 512]

        info = self.processor._get_num_multimodal_tokens(
            image_sizes=[image_size],
            mm_max_length=self.mm_max_length,
        )
        mm_sequence_length = sum(info.num_image_tokens)
        num_images = max(self.mm_max_length // mm_sequence_length, 1)

        max_text_length = self.model_max_length - mm_sequence_length * num_images - 100

        return [image_size] * num_images, max_text_length - 512, 512
