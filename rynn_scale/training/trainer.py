import contextlib
import functools
import gc
import inspect
import io
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, List, Optional, Union

import deepspeed
import numpy as np
import shutil
import torch
import torch.nn as nn
from deepspeed.runtime.checkpoint_engine import CheckpointCommitInfo
from deepspeed.runtime.zero.partition_parameters import GatheredParameters
from packaging import version
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
    enable_full_determinism,
    get_model_param_count,
    get_reporting_integration_callbacks,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.trainer import (
    TrainerState as _TrainerState,
)

from . import callbacks
from ..arguments import TrainingArguments
from ..utils import logging, oss
from ..utils.expert_parallel import BaseMoELayer, gather_ep_params
from ..utils.pipeline_parallel import ALL_PIPELINE_SCHEDULES, PipelineModule, PipelineStage, gather_pp_params
from .sampler import DistributedBatchSampler

logger = logging.get_logger(__name__)


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
    if folder.startswith("oss://"):
        content = oss.listdir(folder)
    else:
        content = os.listdir(folder)
    pattern = re.compile("checkpoint" + r"\-(\d+)$")
    checkpoints = [path for path in content if pattern.search(path) is not None]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(pattern.search(x).groups()[0])))


def rotate_checkpoints(output_dir: str, save_total_limit: Optional[int] = None):
    if save_total_limit is None or save_total_limit <= 0:
        return

    if output_dir.startswith("oss://"):
        content = oss.listdir(output_dir)
    else:
        content = os.listdir(output_dir)

    pattern = re.compile("checkpoint" + r"\-(\d+)$")
    checkpoints = sorted(
        [path for path in content if pattern.search(path) is not None],
        key=lambda x: int(pattern.search(x).groups()[0])
    )

    if len(checkpoints) <= save_total_limit:
        return

    for checkpoint in checkpoints[:-save_total_limit]:
        checkpoint = os.path.join(output_dir, checkpoint)
        if checkpoint.startswith("oss://"):
            oss.rmtree(checkpoint)
        else:
            shutil.rmtree(checkpoint, ignore_errors=True)


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


class LazyBatchLoader(object):
    _torch_dtype_map = {
        str(dtype): dtype for dtype in [
            torch.float, torch.float32, torch.float16, torch.bfloat16,
            torch.long, torch.int64, torch.int32, torch.int16, torch.int8,
            torch.uint64, torch.uint32, torch.uint16, torch.uint8, torch.bool,
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

        if (not self.args.cp_broadcast_data or self.args.cp_rank == 0) and \
            (not self.args.pp_broadcast_data or self.args.pp_rank == 0):
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
                cu_seq_lens[:len(batch["cu_seq_lens_q"])] = batch["cu_seq_lens_q"]

            torch.distributed.broadcast(
                cu_seq_lens,
                group=self.args.pp_group,
                group_src=0,
            )
            cu_seq_lens = cu_seq_lens[:cu_seq_lens[-1]]

            if self.args.pp_rank == 0:
                batch["position_ids"] = batch["position_ids"].to(self.args.device)
                assert batch["position_ids"].size() == (3, 1, cu_seq_lens[-1])
                assert batch["position_ids"].dtype == torch.long
                position_ids = batch["position_ids"]
                batch["labels"] = batch["labels"].to(self.args.device)
                assert batch["labels"].size() == (1, cu_seq_lens[-1])
                assert batch["labels"].dtype == torch.long
                labels = batch["labels"]
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

            if self.args.pp_rank != 0:
                max_length = torch.amax(cu_seq_lens[1:] - cu_seq_lens[:-1]).item()
                batch["cu_seq_lens_q"] = cu_seq_lens
                batch["cu_seq_lens_k"] = cu_seq_lens
                batch["max_length_q"] = max_length
                batch["max_length_k"] = max_length
                batch["position_ids"] = position_ids
                batch["labels"] = labels

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
        enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)

        self.hp_name = None
        self.deepspeed = None
        self.is_in_train = False
        self.model = model

        # memory metrics - must set up as early as possible
        self._memory_tracker = TrainerMemoryTracker()
        self._memory_tracker.start()

        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class

        self.is_deepspeed_enabled = True

        # later use `self.model is self.model_wrapped` to check if it's wrapped or not
        self.model_wrapped = model
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
        if self.args.global_rank == 0 and not self.args.output_dir.startswith("oss://"):
            os.makedirs(self.args.output_dir, exist_ok=True)

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
            oss.clear_cache()

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

    def create_optimizer(self):
        opt_model = self.model

        decay_parameters = set(self.get_decay_parameter_names(opt_model))

        expert_parameters = set()
        for module_name, module in opt_model.named_modules():
            if isinstance(module, BaseMoELayer):
                for name, _ in module.named_parameters():
                    expert_parameters.add(f"{module_name}.{name}")

        # Resolve a per-parameter learning rate from learning_rate_strategy: a parameter uses
        # the learning rate of the first regex pattern (matched with re.search) it matches, and
        # falls back to the default learning_rate when it matches none.
        def resolve_lr(name):
            for pattern, lr in self.args.learning_rate_strategy.items():
                if re.search(pattern, name):
                    return lr
            return self.args.learning_rate

        # Group parameters by (is_expert, is_decay, lr), preserving first-seen order.
        grouped_parameters = {}
        for n, p in opt_model.named_parameters():
            if not p.requires_grad:
                continue
            key = (n in expert_parameters, n in decay_parameters, resolve_lr(n))
            grouped_parameters.setdefault(key, []).append(p)

        optimizer_grouped_parameters = []
        for (is_expert, is_decay, lr), params in grouped_parameters.items():
            group = {
                "params": params,
                "lr": lr,
                "weight_decay": self.args.weight_decay if is_decay else 0.0,
            }
            if is_expert:
                # Preserve the exact name/flag DeepSpeed relies on for expert parallelism.
                group["name"] = f"ep_size_{self.args.ep_world_size}"
                group["moe"] = True
            else:
                group["name"] = "decay" if is_decay else "no_decay"
            optimizer_grouped_parameters.append(group)

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

    def _is_zero3(self):
        return self.deepspeed_engine.zero_optimization_stage() == 3

    def _save_model(self, output_dir, full: bool = False):
        if self.args.global_rank == 0:
            self.model.config.save_pretrained(output_dir)
            self.processing_class.save_pretrained(output_dir)

        kwargs = {}
        if "convert" in inspect.signature(self.model.state_dict).parameters:
            kwargs["convert"] = False

        if self._is_zero3():
            state_dict = {
                name: param.ds_tensor.clone().cpu()
                for name, param in self.model.named_parameters()
            }
            ckpt_name = f"model_zero_pp_rank_{self.args.dp_rank}.pt"
            torch.save(state_dict, os.path.join(output_dir, ckpt_name))
        else:
            if self.args.edp_rank != 0:
                return
            state_dict = self.model.state_dict(**kwargs)
            ckpt_name = f"model_pp_rank_{self.args.pp_rank:02d}_ep_rank_{self.args.ep_rank:02d}.pt"
            torch.save(state_dict, os.path.join(output_dir, ckpt_name))

    def _load_model(self, checkpoint):
        if self._is_zero3():
            ckpt_name = f"model_zero_pp_rank_{self.args.dp_rank}.pt"
            ckpt_path = os.path.join(checkpoint, ckpt_name)
            if ckpt_path.startswith("oss://"):
                with oss.get_object(ckpt_path) as result:
                    buffer = io.BytesIO(result.read())
                state_dict = torch.load(buffer, map_location="cpu")
                buffer.close()
            else:
                state_dict = torch.load(ckpt_path, map_location="cpu")

            for name, param in self.model.named_parameters():
                if name in state_dict:
                    param.ds_tensor.copy_(state_dict[name].to(param.ds_tensor.device))
            return

        if self.args.edp_rank == 0:
            ckpt_name = f"model_pp_rank_{self.args.pp_rank:02d}_ep_rank_{self.args.ep_rank:02d}.pt"
            ckpt_path = os.path.join(checkpoint, ckpt_name)
            if ckpt_path.startswith("oss://"):
                with oss.get_object(ckpt_path) as result:
                    buffer = io.BytesIO(result.read())
                state_dict = torch.load(buffer, map_location="cpu")
                buffer.close()
            else:
                state_dict = torch.load(ckpt_path, map_location="cpu")

            kwargs = {"strict": True}
            if "convert" in inspect.signature(self.model.state_dict).parameters:
                kwargs["convert"] = False
            self.model.load_state_dict(state_dict, **kwargs)

        self.model_wrapped._broadcast_model()

    def _save_optimizer_and_scheduler(self, output_dir, is_tmp_dir):
        if hasattr(self.optimizer, "checkpoint_event_prologue"):
            self.optimizer.checkpoint_event_prologue()

        tag = f"global_step{self.state.global_step}"
        save_dir = os.path.join(output_dir, tag)
        if self.args.global_rank == 0 or is_tmp_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.distributed.barrier()

        commit_info = CheckpointCommitInfo(tag=tag, save_dir=output_dir, save_latest=True)
        self.model_wrapped.checkpoint_engine.create(commit_info)

        if self.model_wrapped.save_zero_checkpoint:
            self.model_wrapped._create_zero_checkpoint_files(output_dir, tag)
            self.model_wrapped._save_zero_checkpoint(output_dir, tag)

        if hasattr(self.optimizer, "checkpoint_event_epilogue"):
            self.optimizer.checkpoint_event_epilogue()

        if self.args.global_rank == 0:
            torch.save(
                {
                    "skipped_steps": self.model_wrapped.skipped_steps,
                    "global_steps": self.model_wrapped.global_steps,
                    "global_samples": self.model_wrapped.global_samples,
                    "dp_world_size": self.model_wrapped.dp_world_size,
                    "mp_world_size": self.model_wrapped.mp_world_size,
                },
                os.path.join(output_dir, "deepspeed_state.pt"),
            )

        if not self.model_wrapped.checkpoint_engine.is_decoupled():
            self.model_wrapped.checkpoint_engine.commit(tag)
            if self.args.global_rank == 0:
                with open(os.path.join(output_dir, "latest"), "w") as fd:
                    fd.write(tag)

        if self.args.global_rank == 0:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))

        torch.distributed.barrier()

    def _load_optimizer_and_scheduler(self, checkpoint):
        if hasattr(self.optimizer, "checkpoint_event_prologue"):
            self.optimizer.checkpoint_event_prologue()

        latest_path = os.path.join(checkpoint, "latest")
        if latest_path.startswith("oss://"):
            with oss.get_object(latest_path) as result:
                tag = result.read().decode("utf-8").strip()
        else:
            with open(latest_path, "r") as fd:
                tag = fd.read().strip()

        deepspeed_state_path = os.path.join(checkpoint, "deepspeed_state.pt")
        if deepspeed_state_path.startswith("oss://"):
            with oss.get_object(deepspeed_state_path) as result:
                buffer = io.BytesIO(result.read())
            deepspeed_state = torch.load(buffer, map_location="cpu")
            buffer.close()
        else:
            deepspeed_state = torch.load(deepspeed_state_path)

        self.model_wrapped.global_steps = deepspeed_state["global_steps"]
        self.model_wrapped.global_samples = deepspeed_state["global_samples"]
        self.model_wrapped.skipped_steps = deepspeed_state["skipped_steps"]
        self.model_wrapped.loaded_checkpoint_dp_world_size = deepspeed_state["dp_world_size"]
        self.model_wrapped.loaded_checkpoint_mp_world_size = deepspeed_state["mp_world_size"]

        tmp_dir = None
        if checkpoint.startswith("oss://"):
            include = ["scheduler.pt"]

            for bf16_mode in [self.model_wrapped.bfloat16_enabled(), not self.model_wrapped.bfloat16_enabled()]:
                zero_ckpt_names = self.model_wrapped._get_all_zero_checkpoint_names(checkpoint, tag, bf16_mode)
                if zero_ckpt_names is not None:
                    for i, ckpt_name in enumerate(zero_ckpt_names):
                        if torch.distributed.get_rank(group=self.optimizer.dp_process_group) == i:
                            include.append(os.path.relpath(ckpt_name, checkpoint))

            tmp_dir = oss.TemporaryDirectory(
                oss_path=checkpoint,
                mode="download",
                include=include,
            )
            checkpoint = tmp_dir.name

        success = self.model_wrapped._load_zero_checkpoint(checkpoint, tag, load_optimizer_states=True)
        assert success

        if hasattr(self.optimizer, "checkpoint_event_epilogue"):
            self.optimizer.checkpoint_event_epilogue()

        self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME), weights_only=True))

        if tmp_dir is not None:
            tmp_dir.cleanup()

    def _save_rng_state(self, output_dir):
        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
            "cuda": torch.cuda.random.get_rng_state_all(),
        }
        torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.global_rank}.pth"))

    def _load_rng_state(self, checkpoint):
        # Load RNG states from `checkpoint`
        if checkpoint is None:
            return

        rng_file = os.path.join(checkpoint, f"rng_state_{self.args.global_rank}.pth")
        with safe_globals():
            if rng_file.startswith("oss://"):
                with oss.get_object(rng_file) as result:
                    buffer = io.BytesIO(result.read())
                checkpoint_rng_state = torch.load(buffer)
                buffer.close()
            else:
                checkpoint_rng_state = torch.load(rng_file)

        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])

    def _save_checkpoint(self):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"checkpoint-{self.state.global_step}"

        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        tmp_dir = None

        if output_dir.startswith("oss://"):
            tmp_dir = oss.TemporaryDirectory(oss_path=output_dir, mode="upload")
            output_dir = tmp_dir.name
        elif self.args.global_rank == 0:
            os.makedirs(output_dir, exist_ok=True)
        torch.distributed.barrier()

        self._save_model(output_dir, full=False)
        self._save_optimizer_and_scheduler(output_dir, is_tmp_dir=tmp_dir is not None)
        self._save_rng_state(output_dir)

        # Save the Trainer state
        if self.args.global_rank == 0:
            # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
            for cb in [
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]:
                cb_name = cb.__class__.__name__
                cb_state = cb.state()
                if isinstance(self.state.stateful_callbacks[cb_name], list):
                    self.state.stateful_callbacks[cb_name].append(cb_state)
                else:
                    self.state.stateful_callbacks[cb_name] = cb_state
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if self.args.global_rank == 0:
            rotate_checkpoints(
                output_dir=self.args.output_dir,
                save_total_limit=self.args.save_total_limit,
            )

        torch.distributed.barrier()

    def _load_checkpoint(self, checkpoint: Optional[str]):
        self._load_optimizer_and_scheduler(checkpoint)
        self._load_model(checkpoint)

    def _save_full_model(self):
        if self.args.save_full_model:
            if self._is_zero3():
                with GatheredParameters(self.model.parameters()):
                    if self.args.global_rank == 0:
                        kwargs = {}
                        if "convert" in inspect.signature(self.model.state_dict).parameters:
                            kwargs["convert"] = False
                        state_dict = self.model.state_dict(**kwargs)
                    else:
                        state_dict = None
            else:
                state_dict = gather_ep_params(self.model)
                state_dict = gather_pp_params(state_dict)
        else:
            state_dict = None
        if self.args.global_rank == 0:
            if state_dict is not None:
                self.model.save_pretrained(self.args.output_dir, state_dict=state_dict)
            else:
                self.model.config.save_pretrained(self.args.output_dir)
            self.processing_class.save_pretrained(self.args.output_dir)
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
            if args.encoder_gradient_checkpointing_interval is not None and encoder is not None and hasattr(encoder, "gradient_checkpointing_interval"):
                encoder.gradient_checkpointing_disable()
                encoder.gradient_checkpointing_interval = args.encoder_gradient_checkpointing_interval

        self.model.train()

        if self.args.pp_world_size > 1:
            module = PipelineModule(self.model)
            mpu = module.mpu()
        else:
            module = self.model
            mpu = None

        model = deepspeed.DeepSpeedEngine(
            args=args,
            model=module,
            optimizer=self.optimizer,
            mpu=mpu,
            config=args.deepspeed_config,
            config_class=deepspeed.DeepSpeedConfig(args.deepspeed_config, mpu=mpu),
        )

        self.optimizer = model.optimizer
        assert model.lr_scheduler is None

        self.model_wrapped = model
        self.deepspeed_engine = self.model_wrapped

        pipeline_stage = PipelineStage(
            self.model,
            deepspeed_engine=self.deepspeed_engine,
            group=args.pp_group,
        )

        pipeline_schedule = ALL_PIPELINE_SCHEDULES[args.pipeline_parallel_schedule](
            stages=[pipeline_stage],
            deepspeed_engine=self.deepspeed_engine,
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
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None:
            if resume_from_checkpoint.startswith("oss://"):
                tmp_dir = oss.TemporaryDirectory(
                    oss_path=resume_from_checkpoint,
                    mode="download",
                    include=[TRAINER_STATE_NAME],
                )
                trainer_state_path = os.path.join(tmp_dir.name, TRAINER_STATE_NAME)
            else:
                tmp_dir = None
                trainer_state_path = os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)

            self.state = TrainerState.load_from_json(trainer_state_path)
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()

            if tmp_dir is not None:
                tmp_dir.cleanup()

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
        model.zero_grad()
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
                    update_step = steps_trained_in_current_epoch // args.gradient_accumulation_steps
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

            for _ in range(total_updates):
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
                                    input_tokens = (inputs[main_input_name] != self.processing_class.pad_token_id).sum()
                                else:
                                    input_tokens = inputs[main_input_name].numel()

                                self.state.num_input_tokens_seen += input_tokens

                        if args.log_flops:
                            self.state.total_flos += (
                                float(self.model.floating_point_ops(inputs)) / 1e12 * 3
                            )

                if self.args.cleanup_before_optimizer_step:
                    del batch_samples
                    gc.collect()
                    torch.cuda.empty_cache()

                self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)
                with torch.cuda.nvtx.range("optimizer_step"):
                    if self.args.pp_world_size > 1:
                        self.optimizer.step()
                    else:
                        self.deepspeed_engine.step()
                self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    grad_norm = self.deepspeed_engine.get_global_grad_norm()
                    if grad_norm is None and hasattr(self.optimizer, '_global_grad_norm'):
                        grad_norm = self.optimizer._global_grad_norm
                    # In some cases the grad norm may not return a float
                    if hasattr(grad_norm, "item"):
                        grad_norm = grad_norm.item()
                    self._total_grad_norm_scaler += grad_norm

                # get leaning rate before update
                learning_rate = self._get_learning_rate()

                if not getattr(self.optimizer, "overflow", False):
                    # Delay optimizer scheduling until metrics are generated
                    if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step()

                self.deepspeed_engine.zero_grad()

                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / steps_in_epoch
                self.state.running_time += time.time() - start_time
                start_time = time.time()

                self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                self._maybe_log_save_evaluate(
                    tr_loss,
                    grad_norm,
                    model,
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
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, epoch, learning_rate=learning_rate)

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

        if self.control.should_save:
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
