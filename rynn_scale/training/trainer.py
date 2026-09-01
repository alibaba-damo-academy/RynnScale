import contextlib
import functools
import gc
import inspect
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from huggingface_hub import split_torch_state_dict_into_shards
from packaging import version
from torch.distributed.checkpoint.filesystem import FileSystemWriter
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.tensor import DTensor
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import Trainer as _Trainer
from transformers.trainer import (
    DEFAULT_CALLBACKS,
    DEFAULT_PROGRESS_CALLBACK,
    SCHEDULER_NAME,
    TRAINER_STATE_NAME,
    BaseImageProcessor,
    CallbackHandler,
    DataCollator,
    ExportableState,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PrinterCallback,
    ProcessorMixin,
    TrainerCallback,
    TrainerControl,
    TrainerMemoryTracker,
    TrainOutput,
    get_model_param_count,
    get_reporting_integration_callbacks,
    seed_worker,
    speed_metrics,
)
from transformers.trainer import (
    TrainerState as _TrainerState,
)
from transformers.trainer_utils import SaveStrategy
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from ..arguments import TrainingArguments
from ..utils import logging, storage
from ..utils.determinism import set_seed
from ..utils.pipeline_parallel import ALL_PIPELINE_SCHEDULES, PipelineStage
from .sampler import DistributedBatchSampler

logger = logging.get_logger(__name__)

DEFAULT_MAX_SHARD_SIZE = 5 * 1024**3  # 5GB
SAFE_WEIGHTS_FILENAME_PATTERN = SAFE_WEIGHTS_NAME.replace(".safetensors", "{suffix}.safetensors")


def _global_shard_filename(global_index: int, total_shards: int) -> str:
    if total_shards <= 1:
        return SAFE_WEIGHTS_FILENAME_PATTERN.format(suffix="")
    return SAFE_WEIGHTS_FILENAME_PATTERN.format(suffix=f"-{global_index:05d}-of-{total_shards:05d}")


def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False
    except AttributeError:
        # Ray DataSets raises an AttributeError: https://github.com/ray-project/ray/blob/master/python/ray/data/dataset.py#L5616
        return False


def get_last_checkpoint(folder):
    content = storage.listdir(folder)
    pattern = re.compile("checkpoint" + r"\-(\d+)$")
    checkpoints = [path for path in content if pattern.search(path) is not None]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(pattern.search(x).groups()[0])))


def rotate_checkpoints(output_dir: str, save_total_limit: Optional[int] = None):
    if save_total_limit is None or save_total_limit <= 0:
        return

    content = storage.listdir(output_dir)

    pattern = re.compile("checkpoint" + r"\-(\d+)$")
    checkpoints = sorted(
        [path for path in content if pattern.search(path) is not None],
        key=lambda x: int(pattern.search(x).groups()[0]),
    )

    if len(checkpoints) <= save_total_limit:
        return

    for checkpoint in checkpoints[:-save_total_limit]:
        checkpoint = os.path.join(output_dir, checkpoint)
        storage.rmtree(checkpoint)


def safe_globals():
    # Starting from version 2.4 PyTorch introduces a check for the objects loaded
    # with torch.load(weights_only=True). Starting from 2.6 weights_only=True becomes
    # a default and requires allowlisting of objects being loaded.
    # See: https://github.com/pytorch/pytorch/pull/137602
    # See: https://pytorch.org/docs/stable/notes/serialization.html#torch.serialization.add_safe_globals
    # See: https://github.com/huggingface/accelerate/pull/3036
    if version.parse(torch.__version__).release < version.parse("2.6").release:
        return contextlib.nullcontext()

    np_core = np._core if version.parse(np.__version__) >= version.parse("2.0.0") else np.core
    allowlist = [np_core.multiarray._reconstruct, np.ndarray, np.dtype]
    # numpy >1.25 defines numpy.dtypes.UInt32DType, but below works for
    # all versions of numpy
    allowlist += [type(np.dtype(np.uint32))]

    return torch.serialization.safe_globals(allowlist)


@torch.no_grad()
def clip_grad_norm_(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    pp_group: torch.distributed.ProcessGroup,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    excluded_parameters: Iterable[torch.nn.Parameter] = (),
) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)

    excluded_param_ids = {id(p) for p in excluded_parameters}

    param_groups = defaultdict(list)
    for p in parameters:
        param_groups[p.device_mesh.mesh_dim_names].append(p)

    total_norms = []
    for param_group in param_groups.values():
        grads = [p.grad for p in param_group if p.grad is not None and id(p) not in excluded_param_ids]
        total_norm = torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite, foreach)

        if isinstance(total_norm, DTensor):
            total_norm = total_norm.full_tensor()

        total_norms.append(total_norm)

    if math.isinf(norm_type):
        total_norm = torch.amax(total_norms)
    else:
        total_norm = torch.sum(torch.stack(total_norms) ** norm_type)
        total_norm **= 1.0 / norm_type

    if torch.distributed.get_world_size(pp_group) > 1:
        if math.isinf(norm_type):
            torch.distributed.all_reduce(
                total_norm,
                op=torch.distributed.ReduceOp.MAX,
                group=pp_group,
            )
        else:
            total_norm **= norm_type
            torch.distributed.all_reduce(
                total_norm,
                op=torch.distributed.ReduceOp.SUM,
                group=pp_group,
            )
            total_norm **= 1.0 / norm_type

    for param_group in param_groups.values():
        torch.nn.utils.clip_grads_with_norm_(param_group, max_norm, total_norm, foreach)

    return total_norm


class LazyBatchLoader(object):
    _torch_dtype_map = {
        str(dtype): dtype
        for dtype in [
            torch.float,
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.long,
            torch.int64,
            torch.int32,
            torch.int16,
            torch.int8,
            torch.uint64,
            torch.uint32,
            torch.uint16,
            torch.uint8,
            torch.bool,
        ]
    }

    def __init__(
        self,
        epoch_iterator: Iterator,
        num_batches: int,
        training_args: TrainingArguments,
    ):
        self.epoch_iterator = epoch_iterator
        self.num_batches = num_batches
        self.args = training_args

        self._batch_samples = []

    def __len__(self):
        return self.num_batches

    def _load_one_batch(self):
        assert len(self._batch_samples) < self.num_batches

        if (not self.args.cp_broadcast_data or self.args.cp_rank == 0) and (
            not self.args.pp_broadcast_data or self.args.pp_rank == 0
        ):
            batch = next(self.epoch_iterator)
        else:
            batch = {}

        if self.args.cp_broadcast_data and (not self.args.pp_broadcast_data or self.args.pp_rank == 0):
            if self.args.cp_rank == 0:
                meta_data = defaultdict(list)
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        meta_data[str(value.dtype)].append((key, tuple(value.shape)))
                    else:
                        meta_data["others"].append((key, value))
            else:
                meta_data = None

            meta_data = [meta_data]
            torch.distributed.broadcast_object_list(
                meta_data,
                group=self.args.cp_group,
                group_src=0,
            )
            meta_data = meta_data[0]

            others = meta_data.pop("others", [])
            if self.args.cp_rank != 0:
                for key, value in others:
                    batch[key] = value

            for dtype, items in meta_data.items():
                dtype = self._torch_dtype_map[dtype]
                sizes = [math.prod(shape) for _, shape in items]

                if self.args.cp_rank == 0:
                    flattened_tensors = []
                    for key, _ in items:
                        batch[key] = batch[key].to(self.args.device)
                        flattened_tensors.append(batch[key].flatten())
                    buffer = torch.cat(flattened_tensors, dim=0)
                else:
                    buffer = torch.empty(sum(sizes), dtype=dtype, device=self.args.device)

                torch.distributed.broadcast(
                    buffer,
                    group=self.args.cp_group,
                    group_src=0,
                )

                if self.args.cp_rank != 0:
                    buffers = buffer.split(sizes, dim=0)
                    for (key, shape), tensor in zip(items, buffers):
                        batch[key] = tensor.view(shape)

        if self.args.pp_broadcast_data:
            cu_seq_lens = torch.empty(
                (self.args.micro_batch_size * self.args.dp_world_size + 2,),
                dtype=torch.int32,
                device=self.args.device,
            )

            if self.args.pp_rank == 0:
                cu_seq_lens[-1] = len(batch["cu_seq_lens_q"])
                cu_seq_lens[: len(batch["cu_seq_lens_q"])] = batch["cu_seq_lens_q"]

            torch.distributed.broadcast(
                cu_seq_lens,
                group=self.args.pp_group,
                group_src=0,
            )
            cu_seq_lens = cu_seq_lens[: cu_seq_lens[-1]]

            if self.args.pp_rank == 0:
                batch["position_ids"] = batch["position_ids"].to(self.args.device)
                assert batch["position_ids"].size() == (3, 1, cu_seq_lens[-1])
                assert batch["position_ids"].dtype == torch.long
                position_ids = batch["position_ids"]
                batch["labels"] = batch["labels"].to(self.args.device)
                assert batch["labels"].size() == (1, cu_seq_lens[-1])
                assert batch["labels"].dtype == torch.long
                labels = batch["labels"]
                batch["input_ids"] = batch["input_ids"].to(self.args.device)
                assert batch["input_ids"].size() == (1, cu_seq_lens[-1])
                assert batch["input_ids"].dtype == torch.long
                input_ids = batch["input_ids"]
            else:
                position_ids = torch.empty(
                    (3, 1, cu_seq_lens[-1]),
                    dtype=torch.long,
                    device=self.args.device,
                )
                labels = torch.empty(
                    (1, cu_seq_lens[-1]),
                    dtype=torch.long,
                    device=self.args.device,
                )
                input_ids = torch.empty(
                    (1, cu_seq_lens[-1]),
                    dtype=torch.long,
                    device=self.args.device,
                )

            torch.distributed.broadcast(
                position_ids,
                group=self.args.pp_group,
                group_src=0,
            )
            torch.distributed.broadcast(
                labels,
                group=self.args.pp_group,
                group_src=0,
            )
            torch.distributed.broadcast(
                input_ids,
                group=self.args.pp_group,
                group_src=0,
            )

            if self.args.pp_rank != 0:
                max_length = torch.amax(cu_seq_lens[1:] - cu_seq_lens[:-1]).item()
                batch["cu_seq_lens_q"] = cu_seq_lens
                batch["cu_seq_lens_k"] = cu_seq_lens
                batch["max_length_q"] = max_length
                batch["max_length_k"] = max_length
                batch["position_ids"] = position_ids
                batch["labels"] = labels
                batch["input_ids"] = input_ids
                batch["use_cache"] = False

        if self.args.synchronize_experts_before_forward:
            torch.distributed.barrier(group=self.args.ep_group)

        return batch

    def __getitem__(self, index: int):
        if index < 0 or index >= self.num_batches:
            raise IndexError(f"Index {index} is out of range")

        if index < len(self._batch_samples):
            return self._batch_samples[index]

        torch.cuda.nvtx.range_push("load_data")

        num_batches = index - len(self._batch_samples) + 1
        batch_samples = []

        for _ in range(num_batches):
            batch_samples.append(self._load_one_batch())

        num_items_in_batch = None
        count_num_items_in_batch = "labels" in batch_samples[0]

        if count_num_items_in_batch:
            if self.args.loss_reduction_scope == "batch":
                num_batches = self.num_batches - len(self._batch_samples) - len(batch_samples)
                for _ in range(num_batches):
                    batch_samples.append(self._load_one_batch())

                num_items_in_batch = sum((batch["labels"].ne(-100)).sum() for batch in batch_samples) / len(
                    batch_samples
                )
                if self.args.average_tokens_across_devices and self.args.dp_world_size > 1:
                    num_items_in_batch = num_items_in_batch.to(self.args.device)
                    torch.distributed.all_reduce(
                        num_items_in_batch,
                        op=torch.distributed.ReduceOp.SUM,
                        group=self.args.dp_group,
                    )
                    num_items_in_batch = num_items_in_batch / self.args.dp_world_size

            elif self.args.loss_reduction_scope == "sequence":
                num_items_in_batch = self.args.micro_batch_size

            else:
                raise ValueError(f"Unknown loss reduction scope: {self.args.loss_reduction_scope}")

        for batch in batch_samples:
            batch["num_items_in_batch"] = num_items_in_batch

        self._batch_samples.extend(batch_samples)

        torch.cuda.nvtx.range_pop()

        return self._batch_samples[index]


@dataclass
class TrainerState(_TrainerState):
    num_input_tokens_seen: float = 0.0
    running_time: float = 0.0


class Trainer(object):
    # Reuse some functions from huggingface transformers
    create_scheduler = _Trainer.create_scheduler
    get_optimizer_cls_and_kwargs = staticmethod(_Trainer.get_optimizer_cls_and_kwargs)
    _load_callback_state = _Trainer._load_callback_state
    _get_learning_rate = _Trainer._get_learning_rate

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        data_collator: DataCollator,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
    ):
        self.args = args
        # Seed must be set before instantiating the model when using model
        set_seed(self.args.seed, full_determinism=self.args.full_determinism)

        self.hp_name = None
        self.is_in_train = False
        self.is_deepspeed_enabled = False

        # memory metrics - must set up as early as possible
        self._memory_tracker = TrainerMemoryTracker()
        self._memory_tracker.start()

        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class

        self.model = model
        self.optimizer = None
        self.lr_scheduler = None

        # Check if the model has explicit setup for loss kwargs,
        # if not, check if `**kwargs` are in model.forward
        if hasattr(model, "accepts_loss_kwargs"):
            self.model_accepts_loss_kwargs = model.accepts_loss_kwargs
        else:
            forward_params = inspect.signature(model.forward).parameters
            self.model_accepts_loss_kwargs = any(
                k.kind == inspect.Parameter.VAR_KEYWORD for k in forward_params.values()
            )

        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.callback_handler.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        # Create distant repo and output directory if needed
        if self.args.global_rank == 0:
            storage.makedirs(self.args.output_dir)

        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            raise TypeError("The `data_collator` should be a simple callable (function, class with `__call__`).")

        if args.max_steps > 0 and args.num_train_epochs > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")

        if train_dataset is not None and not has_length(train_dataset) and args.max_steps <= 0:
            raise ValueError(
                "The train_dataset does not implement __len__, max_steps has to be specified. "
                "The number of steps needs to be known in advance for the learning rate scheduler."
            )

        self.control = TrainerControl()

        self.state = TrainerState(
            is_local_process_zero=self.args.local_rank == 0,
            is_world_process_zero=self.args.global_rank == 0,
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )

        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

        # very last
        self._memory_tracker.stop_and_update_metrics()

    @property
    def tokenizer(self) -> Optional[PreTrainedTokenizerBase]:
        logger.warning("Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.")
        return self.processing_class

    @tokenizer.setter
    def tokenizer(self, processing_class) -> None:
        logger.warning(
            "Trainer.tokenizer is now deprecated. You should use `Trainer.processing_class = processing_class` instead."
        )
        self.processing_class = processing_class

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        if isinstance(train_dataset, IterableDataset):
            train_dataset.shard(rank=self.args.dp_rank, world_size=self.args.dp_world_size)
            train_dataset.shuffle(
                seed=self.args.seed,
                episode_buffer_size=self.args.episode_iterator_buffer,
                shuffle_buffer_size=self.args.episode_iterator_shuffle_buffer,
            )

            def worker_init_fn(worker_id, num_workers, rank):
                seed_worker(worker_id, num_workers=num_workers, rank=rank)
                storage.clear_cache()

            return DataLoader(
                train_dataset,
                batch_size=self.args.micro_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                persistent_workers=self.args.dataloader_persistent_workers,
                prefetch_factor=self.args.dataloader_prefetch_factor,
                worker_init_fn=functools.partial(
                    worker_init_fn,
                    num_workers=self.args.dataloader_num_workers,
                    rank=self.args.dp_rank,
                ),
            )

        sampler_seed = torch.as_tensor(self.args.seed).cuda()
        torch.distributed.broadcast(sampler_seed, src=0)

        if self.args.decoder_load_balancing or self.args.dynamic_batching:
            assert hasattr(train_dataset, "get_sequence_lengths")
            sequence_lengths = train_dataset.get_sequence_lengths(
                num_workers=self.args.dataloader_num_workers,
                cache_dir=self.args.output_dir,
            )
        else:
            sequence_lengths = None

        batch_sampler = DistributedBatchSampler(
            train_dataset,
            sequence_lengths=sequence_lengths,
            num_replicas=self.args.dp_world_size,
            rank=self.args.dp_rank,
            micro_batch_size=self.args.micro_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            shuffle=True,
            seed=sampler_seed.item(),
            drop_last=self.args.dataloader_drop_last,
            decoder_load_balancing=self.args.decoder_load_balancing,
            dynamic_batching=self.args.dynamic_batching,
            dynamic_batching_window_size=self.args.dynamic_batching_window_size,
            model_max_length=self.args.model_max_length,
        )

        def worker_init_fn(worker_id, num_workers, rank):
            seed_worker(worker_id, num_workers=num_workers, rank=rank)
            storage.clear_cache()

        dataloader_params = {
            "batch_sampler": batch_sampler,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "worker_init_fn": functools.partial(
                worker_init_fn,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.dp_rank,
            ),
            "prefetch_factor": self.args.dataloader_prefetch_factor,
        }

        return DataLoader(train_dataset, **dataloader_params)

    def get_decay_parameter_names(self, model) -> list[str]:
        forbidden_layer_types = [nn.LayerNorm]
        forbidden_layer_names = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
        forbidden_layer_patterns = (
            [re.compile(pattern) for pattern in forbidden_layer_names] if forbidden_layer_names is not None else []
        )

        def get_decay_parameter_names(model):
            result = []
            for name, child in model.named_children():
                child_params = get_decay_parameter_names(child)
                result += [
                    f"{name}.{n}"
                    for n in child_params
                    if not isinstance(child, tuple(forbidden_layer_types))
                    and not any(pattern.search(f"{name}.{n}".lower()) for pattern in forbidden_layer_patterns)
                ]
            # Add model specific parameters that are not in any child
            result += [
                k
                for k in model._parameters
                if not any(pattern.search(k.lower()) for pattern in forbidden_layer_patterns)
            ]
            return result

        return get_decay_parameter_names(model)

    def resolve_param_learning_rates(self, model) -> Dict[str, float]:
        strategy = self.args.learning_rate_strategy
        if not strategy:
            return {n: self.args.learning_rate for n, _ in model.named_parameters()}

        compiled = [(pattern, re.compile(pattern), lr) for pattern, lr in strategy.items()]
        param_lrs: Dict[str, float] = {}
        for name, _ in model.named_parameters():
            matched = [(pattern, lr) for pattern, regex, lr in compiled if regex.match(name)]
            assert len(matched) <= 1, (
                f"Parameter '{name}' matches multiple learning_rate_strategy regexes: {[p for p, _ in matched]}"
            )
            param_lrs[name] = matched[0][1] if matched else self.args.learning_rate
        return param_lrs

    def create_optimizer(self):
        opt_model = self.model

        decay_parameters = set(self.get_decay_parameter_names(opt_model))
        param_lrs = self.resolve_param_learning_rates(opt_model)

        if self.args.learning_rate_strategy is not None:
            for n, p in opt_model.named_parameters():
                if param_lrs[n] == 0:
                    p.requires_grad_(False)
            lr_to_names: dict = {}
            for n, _ in opt_model.named_parameters():
                lr_to_names.setdefault(param_lrs[n], []).append(n)
            lr_summary = "\n".join(f"  lr={lr}: {names}" for lr, names in sorted(lr_to_names.items()))
            logger.info(f"Learning rate strategy (default lr={self.args.learning_rate}):\n{lr_summary}")

        groups: dict[tuple[float, float, str], dict] = {}
        for n, p in opt_model.named_parameters():
            if not p.requires_grad:
                continue
            lr = param_lrs[n]
            if lr == 0:
                continue
            decay = n in decay_parameters
            weight_decay = self.args.weight_decay if decay else 0.0
            key = (lr, weight_decay, "decay" if decay else "no_decay")
            if key not in groups:
                groups[key] = {
                    "name": f"{key[2]}_lr{lr}",
                    "params": [],
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            groups[key]["params"].append(p)

        optimizer_grouped_parameters = list(groups.values())

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

    def _save_distributed_checkpoint(self, state_dict: Dict[str, Any], path: str):
        with storage.writable_dir(path) as save_dir:
            dcp.save(state_dict, storage_writer=FileSystemWriter(save_dir))

    def _load_distributed_checkpoint(self, state_dict: Dict[str, Any], path: str):
        dcp.load(
            state_dict,
            storage_reader=storage.get_storage_reader(
                path,
                show_progress=self.args.global_rank == 0,
            ),
        )

    def _save_model(self, output_dir):
        state_dict = get_model_state_dict(
            model=self.model,
            options=StateDictOptions(full_state_dict=False),
        )
        self._save_distributed_checkpoint(state_dict, os.path.join(output_dir, "model"))

    def _load_model(self, checkpoint):
        state_dict = get_model_state_dict(
            model=self.model,
            options=StateDictOptions(full_state_dict=False),
        )
        self._load_distributed_checkpoint(state_dict, os.path.join(checkpoint, "model"))
        set_model_state_dict(
            model=self.model,
            model_state_dict=state_dict,
            options=StateDictOptions(full_state_dict=False),
        )

    def _save_optimizer(self, output_dir):
        state_dict = get_optimizer_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            options=StateDictOptions(full_state_dict=False),
        )
        self._save_distributed_checkpoint(state_dict, os.path.join(output_dir, "optimizer"))

    def _load_optimizer(self, checkpoint):
        state_dict = get_optimizer_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            options=StateDictOptions(full_state_dict=False),
        )
        self._load_distributed_checkpoint(state_dict, os.path.join(checkpoint, "optimizer"))
        set_optimizer_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            optim_state_dict=state_dict,
            options=StateDictOptions(full_state_dict=False),
        )

    def _save_scheduler(self, output_dir):
        if self.args.global_rank != 0:
            return

        save_path = os.path.join(output_dir, SCHEDULER_NAME)
        state_dict = self.lr_scheduler.state_dict()

        storage.torch_save(state_dict, save_path)

    def _load_scheduler(self, checkpoint):
        if checkpoint is None:
            return

        scheduler_file = os.path.join(checkpoint, SCHEDULER_NAME)
        state_dict = storage.torch_load(scheduler_file, weights_only=True)

        self.lr_scheduler.load_state_dict(state_dict)

    def _save_rng_state(self, output_dir):
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
            "cuda": torch.cuda.random.get_rng_state_all(),
        }
        save_path = os.path.join(output_dir, "rng", f"global_rank_{self.args.global_rank}.pt")
        storage.torch_save(rng_states, save_path)

    def _load_rng_state(self, checkpoint):
        if checkpoint is None:
            return

        rng_file = os.path.join(checkpoint, "rng", f"global_rank_{self.args.global_rank}.pt")
        with safe_globals():
            checkpoint_rng_state = storage.torch_load(rng_file)

        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])

    def _save_checkpoint(self):
        # Save model checkpoint
        checkpoint_folder = f"checkpoint-{self.state.global_step}"

        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

        self._save_model(output_dir)
        self._save_optimizer(output_dir)
        self._save_scheduler(output_dir)
        self._save_rng_state(output_dir)

        if self.args.global_rank == 0:
            with storage.writable_dir(output_dir) as save_dir:
                self.model.config.save_pretrained(save_dir)
                self.processing_class.save_pretrained(save_dir)

                # Save the Trainer state
                for cb in [
                    cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
                ]:
                    cb_name = cb.__class__.__name__
                    cb_state = cb.state()
                    if isinstance(self.state.stateful_callbacks[cb_name], list):
                        self.state.stateful_callbacks[cb_name].append(cb_state)
                    else:
                        self.state.stateful_callbacks[cb_name] = cb_state
                self.state.save_to_json(os.path.join(save_dir, TRAINER_STATE_NAME))

            rotate_checkpoints(
                output_dir=self.args.output_dir,
                save_total_limit=self.args.save_total_limit,
            )

        torch.distributed.barrier()

    def _load_checkpoint(self, checkpoint: Optional[str]):
        self._load_model(checkpoint)
        self._load_scheduler(checkpoint)
        self._load_optimizer(checkpoint)

    def _get_pp_shared_param_names(self) -> set[str]:
        if self.args.pp_world_size <= 1:
            return set()
        if self.args.pp_rank != self.args.pp_world_size - 1:
            return set()
        config = self.model.config
        tied = getattr(config, "tie_word_embeddings", False)
        mtp_enabled = getattr(config, "mtp_loss_weight", 0) > 0
        if not (tied or mtp_enabled):
            return set()
        embed = self.model.get_input_embeddings()
        if embed is None or embed.weight is None:
            return set()
        for name, param in self.model.named_parameters():
            if param is embed.weight:
                return {name}
        return set()

    def _save_full_model(self):
        output_dir = self.args.output_dir

        if "convert" in inspect.signature(self.model.state_dict).parameters:
            sharded_state_dict = self.model.state_dict(convert=True)
        else:
            sharded_state_dict = self.model.state_dict()

        pp_size = self.args.pp_world_size
        pp_rank = self.args.pp_rank

        excluded_names = self._get_pp_shared_param_names()
        for name in excluded_names:
            sharded_state_dict.pop(name, None)

        split = split_torch_state_dict_into_shards(
            sharded_state_dict,
            filename_pattern=SAFE_WEIGHTS_FILENAME_PATTERN,
            max_shard_size=DEFAULT_MAX_SHARD_SIZE,
        )
        local_filenames_in_order = list(split.filename_to_tensors.keys())
        local_total_bytes = split.metadata["total_size"]

        local_count = torch.tensor([len(local_filenames_in_order)], dtype=torch.long, device=self.args.device)
        if pp_size > 1:
            all_counts = torch.zeros(pp_size, dtype=torch.long, device=self.args.device)
            torch.distributed.all_gather_into_tensor(all_counts, local_count, group=self.args.pp_group)
            counts = all_counts.tolist()
        else:
            counts = [local_count.item()]

        shard_offset = sum(counts[:pp_rank])
        total_shards = sum(counts)

        local_rename = {
            local: _global_shard_filename(shard_offset + i + 1, total_shards)
            for i, local in enumerate(local_filenames_in_order)
        }
        local_shards = [(local_rename[local], split.filename_to_tensors[local]) for local in local_filenames_in_order]
        local_weight_map: Dict[str, str] = {
            name: local_rename[fname] for name, fname in split.tensor_to_filename.items()
        }

        keep = self.args.dp_rank == 0
        full_local_state_dict: Dict[str, torch.Tensor] = {}
        for name in list(sharded_state_dict.keys()):
            tensor = sharded_state_dict.pop(name)
            if isinstance(tensor, DTensor):
                tensor = tensor.full_tensor()
            if keep:
                full_local_state_dict[name] = tensor.cpu()
        del sharded_state_dict

        if keep:
            for fname, shard_tensors in local_shards:
                shard_state_dict = {name: full_local_state_dict.pop(name).contiguous() for name in shard_tensors}
                save_path = os.path.join(output_dir, fname)
                storage.save_safetensors(shard_state_dict, save_path, metadata={"format": "pt"})
                del shard_state_dict
        del full_local_state_dict

        torch.distributed.barrier()

        is_sharded = total_shards > 1
        payload = (local_weight_map, local_total_bytes)
        if pp_size > 1:
            gathered: Optional[List[Any]] = [None] * pp_size if pp_rank == 0 else None
            torch.distributed.gather_object(payload, gathered, group=self.args.pp_group, group_dst=0)
        else:
            gathered = [payload]

        if self.args.global_rank == 0:
            weight_map: Dict[str, str] = {}
            total_size = 0
            for m, sz in gathered:
                weight_map.update(m)
                total_size += sz

            with storage.writable_dir(output_dir) as local_dir:
                self.model.config.save_pretrained(local_dir)
                self.processing_class.save_pretrained(local_dir)

                if is_sharded:
                    index = {
                        "metadata": {"total_size": total_size},
                        "weight_map": weight_map,
                    }
                    with open(os.path.join(local_dir, SAFE_WEIGHTS_INDEX_NAME), "w", encoding="utf-8") as f:
                        json.dump(index, f, indent=2, sort_keys=True)

        torch.distributed.barrier()

    def log(self, logs: dict[str, float]) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.log_seen_tokens:
            num_tokens_tensor = torch.tensor(
                self.state.num_input_tokens_seen, dtype=torch.float32, device=self.args.device
            )
            torch.distributed.all_reduce(
                num_tokens_tensor, op=torch.distributed.ReduceOp.AVG, group=self.args.dcp_group
            )
            self.state.num_input_tokens_seen = num_tokens_tensor.item()
            logs["num_tokens_seen"] = self.state.num_input_tokens_seen * self.args.dp_world_size
            logs["throughput"] = self.state.num_input_tokens_seen / self.state.running_time
        if self.args.log_flops:
            flops_tensor = torch.tensor(self.state.total_flos, dtype=torch.float32, device=self.args.device)
            torch.distributed.all_reduce(flops_tensor, op=torch.distributed.ReduceOp.AVG, group=self.args.dcp_group)
            self.state.total_flos = flops_tensor.item()
            logs["tflops"] = self.state.total_flos / self.state.running_time

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, epoch, learning_rate=None):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: dict[str, float] = {}

            # get average loss over all processes
            torch.distributed.all_reduce(
                tr_loss,
                op=torch.distributed.ReduceOp.AVG,
                group=self.args.dcp_group,
            )
            tr_loss_scalar = tr_loss.item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            self.log(logs)

        # TODO
        # metrics = None
        # if self.control.should_evaluate:
        #     metrics = self._evaluate(trial, ignore_keys_for_eval)
        #     is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

        #     if self.args.save_strategy == SaveStrategy.BEST:
        #         self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint()
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def compare_trainer_and_checkpoint_args(self, training_args, trainer_state):
        attributes_map = {
            "logging_steps": "logging_steps",
            "eval_steps": "eval_steps",
            "save_steps": "save_steps",
        }

        has_warning = False
        warning_str = "Warning: The following arguments do not match the ones in the `trainer_state.json` within the checkpoint directory: "
        for arg_attr, state_attr in attributes_map.items():
            arg_value = getattr(training_args, arg_attr, None)
            state_value = getattr(trainer_state, state_attr, None)

            if arg_value is not None and state_value is not None and arg_value != state_value:
                warning_str += f"\n\t{arg_attr}: {arg_value} (from args) != {state_value} (from trainer_state.json)"
                has_warning = True

        # train bs is special as we need to account for multi-GPU
        train_bs_args = training_args.micro_batch_size
        train_bs_state = trainer_state.train_batch_size // max(1, training_args.dp_world_size)

        if train_bs_args != train_bs_state:
            warning_str += (
                f"\n\tmicro_batch_size: {train_bs_args} (from args) != {train_bs_state} (from trainer_state.json)"
            )
            has_warning = True

        if has_warning:
            logger.warning_once(warning_str)

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        **kwargs,
    ):
        args = self.args

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        train_dataloader = self.get_train_dataloader()

        total_train_batch_size = self.args.micro_batch_size * args.gradient_accumulation_steps * args.dp_world_size

        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.train_batch_size = self.args.micro_batch_size * args.dp_world_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)
            encoder = self.model.get_encoder(modality="image")
            if (
                args.encoder_gradient_checkpointing_interval is not None
                and encoder is not None
                and hasattr(encoder, "gradient_checkpointing_interval")
            ):
                encoder.gradient_checkpointing_disable()
                encoder.gradient_checkpointing_interval = args.encoder_gradient_checkpointing_interval

        self.model.train()

        pipeline_stage = PipelineStage(self.model, group=args.pp_group, dtype=args.param_dtype)

        pipeline_schedule = ALL_PIPELINE_SCHEDULES[args.pipeline_parallel_schedule](
            stages=[pipeline_stage],
        )

        # ckpt loading
        if resume_from_checkpoint is not None:
            self._load_checkpoint(resume_from_checkpoint)

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.micro_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(self.model, trainable_only=True):,}")

        self.state.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None:
            with storage.readable_dir(resume_from_checkpoint, include=[TRAINER_STATE_NAME]) as local_dir:
                self.state = TrainerState.load_from_json(os.path.join(local_dir, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()

            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
            steps_trained_in_current_epoch *= args.gradient_accumulation_steps

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            logger.info(
                f"  Will skip the first {epochs_trained} epochs then the first"
                f" {steps_trained_in_current_epoch} batches in the first epoch."
            )

        # Update the references
        for attr in ("model", "optimizer", "lr_scheduler"):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader

        self.state.init_training_references(self, max_steps, num_train_epochs, None)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0, device=args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._total_grad_norm_scaler = 0.0
        self._globalstep_last_logged = self.state.global_step
        self.optimizer.zero_grad()
        grad_norm: Optional[float] = None
        learning_rate = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # if args.eval_on_start:
        #     self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        start_time = time.time()

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            epoch_dataloader.batch_sampler.set_epoch(epoch)

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            step = -1
            update_step = -1
            rng_to_sync = False

            # Handle resumption from checkpoint
            if epoch == epochs_trained and resume_from_checkpoint is not None:
                if steps_trained_in_current_epoch > 0:
                    epoch_dataloader.batch_sampler.skip_first_batches(steps_trained_in_current_epoch)
                    step = steps_trained_in_current_epoch - 1
                    update_step = steps_trained_in_current_epoch // args.gradient_accumulation_steps - 1
                    rng_to_sync = True
                else:
                    self._load_rng_state(resume_from_checkpoint)

            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = steps_in_epoch % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps

            total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
                remainder < args.gradient_accumulation_steps
            )

            for _ in range(update_step + 1, total_updates):
                update_step += 1

                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples = LazyBatchLoader(
                    epoch_iterator=epoch_iterator,
                    num_batches=num_batches,
                    training_args=args,
                )
                step += num_batches

                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                losses = pipeline_schedule.step(batch_samples)
                tr_loss = tr_loss + losses.mean()

                if args.pp_rank == 0:
                    for inputs in batch_samples:
                        if args.log_seen_tokens:
                            main_input_name = getattr(self.model, "main_input_name", "input_ids")
                            if main_input_name not in inputs:
                                logger.warning(
                                    "Tried to track the number of tokens seen, however the current model is "
                                    "not configured properly to know what item is the input. To fix this, add "
                                    "a `main_input_name` attribute to the model class you are using."
                                )
                            else:
                                if "attention_mask" in inputs:
                                    input_tokens = inputs["attention_mask"].sum()
                                elif (
                                    self.processing_class is not None
                                    and hasattr(self.processing_class, "pad_token_id")
                                    and self.processing_class.pad_token_id is not None
                                ):
                                    input_tokens = (
                                        inputs[main_input_name] != self.processing_class.pad_token_id
                                    ).sum()
                                else:
                                    input_tokens = inputs[main_input_name].numel()

                                self.state.num_input_tokens_seen += input_tokens

                        if args.log_flops:
                            self.state.total_flos += float(self.model.floating_point_ops(inputs)) / 1e12 * 3

                if self.args.cleanup_before_optimizer_step:
                    del batch_samples
                    gc.collect()
                    torch.cuda.empty_cache()

                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    grad_norm = clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=args.max_grad_norm,
                        pp_group=args.pp_group,
                        excluded_parameters=pipeline_stage.get_pp_shared_params(),
                    ).item()
                    self._total_grad_norm_scaler += grad_norm

                self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)
                with torch.cuda.nvtx.range("optimizer_step"):
                    self.optimizer.step()
                self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                # get leaning rate before update
                learning_rate = self._get_learning_rate()

                if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step()

                self.optimizer.zero_grad()

                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / steps_in_epoch
                self.state.running_time += time.time() - start_time
                start_time = time.time()

                self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                self._maybe_log_save_evaluate(
                    tr_loss,
                    grad_norm,
                    self.model,
                    epoch,
                    learning_rate=learning_rate,
                )

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, self.model, epoch, learning_rate=learning_rate)

            if self.control.should_training_stop:
                break

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=self.state.num_input_tokens_seen,
        )
        metrics["train_loss"] = train_loss
        metrics["grad_norm"] = self._total_grad_norm_scaler / effective_global_step

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        if self.args.save_strategy != SaveStrategy.NO:
            self._save_full_model()

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def set_initial_training_values(
        self, args: TrainingArguments, dataloader: DataLoader, total_train_batch_size: int
    ):
        # Case 1: we rely on `args.max_steps` first
        max_steps = args.max_steps
        # If max_steps is negative, we use the number of epochs to determine the number of total steps later
        epoch_based = max_steps < 0
        len_dataloader = len(dataloader) if has_length(dataloader) else None

        # Case 2: We have a dataloader length and can extrapolate
        if len_dataloader is not None:
            num_update_steps_per_epoch = max(
                len_dataloader // args.gradient_accumulation_steps
                + int(len_dataloader % args.gradient_accumulation_steps > 0),
                1,
            )
            # Case 3: We have a length but are using epochs, we can extrapolate the number of steps
            if epoch_based:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)

        # Now we figure out `num_examples`, `num_train_epochs`, and `train_samples`
        if len_dataloader:
            num_examples = len(dataloader)
            if args.max_steps > 0:
                num_train_epochs = max_steps // num_update_steps_per_epoch + int(
                    max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = max_steps * total_train_batch_size
            else:
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = len(dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )
        return (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        )

    def is_local_process_zero(self):
        return self.args.local_rank == 0

    def is_world_process_zero(self):
        return self.args.global_rank == 0
