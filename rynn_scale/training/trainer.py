import inspect
import contextlib
import os
import time
import functools
import random
import math
import re
import sys
from dataclasses import dataclass
from packaging import version
from typing import Any, Union, Optional, List, Dict, Iterator

import deepspeed
import numpy as np
import torch
import torch.nn as nn
from deepspeed.runtime.checkpoint_engine import CheckpointCommitInfo
from torch.utils.data import Dataset, IterableDataset, DataLoader
from transformers.trainer import (
    PreTrainedModel,
    DataCollator,
    PreTrainedTokenizerBase,
    BaseImageProcessor,
    FeatureExtractionMixin,
    ProcessorMixin,
    TrainerMemoryTracker,
    enable_full_determinism,
    set_seed,
    DEFAULT_CALLBACKS,
    get_reporting_integration_callbacks,
    TrainerCallback,
    CallbackHandler,
    TrainerControl,
    TrainerState as _TrainerState,
    ExportableState,
    get_model_param_count,
    TRAINER_STATE_NAME,
    SCHEDULER_NAME,
    speed_metrics,
    TrainOutput,
    seed_worker,
    PrinterCallback,
    DEFAULT_PROGRESS_CALLBACK,
)
from transformers import Trainer as _Trainer

from .sampler import DistributedBatchSampler
from ..arguments import TrainingArguments
from ..utils.pipeline_parallel import PipelineStage, PipelineModule, ALL_PIPELINE_SCHEDULES
from ..utils.expert_parallel import BaseMoELayer
from ..utils import logging


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
    content = os.listdir(folder)
    pattern = re.compile(r"^" + "checkpoint" + r"\-(\d+)$")
    checkpoints = [
        path for path in content if pattern.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(pattern.search(x).groups()[0])))


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
    _sorted_checkpoints = _Trainer._sorted_checkpoints
    _rotate_checkpoints = _Trainer._rotate_checkpoints

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
        if self.args.global_rank == 0:
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

    @torch.cuda.nvtx.range("load_data")
    def get_batch_samples(
        self, epoch_iterator: Iterator, num_batches: int, device: torch.device
    ) -> List[Dict[str, Any]]:
        batch_samples = []
        num_items_in_batch = None

        for _ in range(num_batches):
            try:
                batch_samples.append(next(epoch_iterator))
            except StopIteration:
                break

        count_num_items_in_batch = len(batch_samples) > 0 and "labels" in batch_samples[0]

        if count_num_items_in_batch:
            if self.args.loss_reduction_scope == "batch":
                num_items_in_batch = sum((batch["labels"].ne(-100)).sum() for batch in batch_samples) / len(
                    batch_samples
                )
                if self.args.average_tokens_across_devices and self.args.dp_world_size > 1:
                    num_items_in_batch = num_items_in_batch.to(device)
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

        return batch_samples

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        sampler_seed = torch.as_tensor(self.args.seed).cuda()
        torch.distributed.broadcast(sampler_seed, src=0)

        batch_sampler = DistributedBatchSampler(
            train_dataset,
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

        dataloader_params = {
            "batch_sampler": batch_sampler,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "worker_init_fn": functools.partial(
                seed_worker,
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

        optimizer_grouped_parameters = [
            {
                "name": "decay",
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if p.requires_grad and n in decay_parameters and n not in expert_parameters
                ],
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
            },
            {
                "name": "no_decay",
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if p.requires_grad and n not in decay_parameters and n not in expert_parameters
                ],
                "lr": self.args.learning_rate,
                "weight_decay": 0.0,
            },
        ]

        if len(expert_parameters) > 0:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "name": f"ep_size_{self.args.ep_world_size}",
                        "moe": True,
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if p.requires_grad and n in decay_parameters and n in expert_parameters
                        ],
                        "lr": self.args.learning_rate,
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "name": f"ep_size_{self.args.ep_world_size}",
                        "moe": True,
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if p.requires_grad and n not in decay_parameters and n in expert_parameters
                        ],
                        "lr": self.args.learning_rate,
                        "weight_decay": 0.0,
                    },
                ]
            )

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

    def _save_model(self, output_dir, full: bool = False):
        if self.args.edp_rank != 0:
            return
        ckpt_name = f"model_pp_rank_{self.args.pp_rank:02d}_ep_rank_{self.args.ep_rank:02d}.pt"
        torch.save(self.model.state_dict(), os.path.join(output_dir, ckpt_name))

    def _load_model(self, checkpoint):
        ckpt_name = f"model_pp_rank_{self.args.pp_rank:02d}_ep_rank_{self.args.ep_rank:02d}.pt"
        state_dict = torch.load(os.path.join(checkpoint, ckpt_name), map_location="cpu")
        self.model.load_state_dict(state_dict, strict=True)

    def _save_optimizer_and_scheduler(self, output_dir):
        if hasattr(self.optimizer, "checkpoint_event_prologue"):
            self.optimizer.checkpoint_event_prologue()

        tag = f"global_step{self.state.global_step}"
        save_dir = os.path.join(output_dir, tag)
        if self.args.global_rank == 0:
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

        torch.distributed.barrier()
        torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))

    def _load_optimizer_and_scheduler(self, checkpoint):
        if hasattr(self.optimizer, "checkpoint_event_prologue"):
            self.optimizer.checkpoint_event_prologue()

        latest_path = os.path.join(checkpoint, "latest")
        if os.path.isfile(latest_path):
            with open(latest_path, "r") as fd:
                tag = fd.read().strip()

        deepspeed_state = torch.load(os.path.join(checkpoint, "deepspeed_state.pt"))
        self.model_wrapped.global_steps = deepspeed_state["global_steps"]
        self.model_wrapped.global_samples = deepspeed_state["global_samples"]
        self.model_wrapped.skipped_steps = deepspeed_state["skipped_steps"]
        self.model_wrapped.loaded_checkpoint_dp_world_size = deepspeed_state["dp_world_size"]
        self.model_wrapped.loaded_checkpoint_mp_world_size = deepspeed_state["mp_world_size"]

        success = self.model_wrapped._load_zero_checkpoint(checkpoint, tag, load_optimizer_states=True)
        assert success

        if hasattr(self.optimizer, "checkpoint_event_epilogue"):
            self.optimizer.checkpoint_event_epilogue()

        self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME), weights_only=True))

    def _save_rng_state(self, output_dir):
        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
            "cuda": torch.cuda.random.get_rng_state_all(),
        }

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)
        torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.global_rank}.pth"))

    def _load_rng_state(self, checkpoint):
        # Load RNG states from `checkpoint`
        if checkpoint is None:
            return

        rng_file = os.path.join(checkpoint, f"rng_state_{self.args.global_rank}.pth")
        if not os.path.isfile(rng_file):
            raise ValueError(
                f"Didn't find an RNG file for process {self.args.global_rank}, if you are resuming a training that "
                "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
            )

        with safe_globals():
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
        if self.args.global_rank == 0:
            os.makedirs(output_dir, exist_ok=True)
        torch.distributed.barrier()

        self._save_model(output_dir, full=False)
        self._save_optimizer_and_scheduler(output_dir)
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

        # Maybe delete some older checkpoints.
        if self.args.global_rank == 0:
            # we use mtime as default, filesystems without mtime support will be detected in `_sorted_checkpoints`
            self._rotate_checkpoints(use_mtime=True, output_dir=self.args.output_dir)

    def _load_checkpoint(self, checkpoint: Optional[str]):
        self._load_optimizer_and_scheduler(checkpoint)
        self._load_model(checkpoint)

    def log(self, logs: dict[str, float]) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.log_seen_tokens:
            num_tokens_tensor = torch.tensor(
                self.state.num_input_tokens_seen, dtype=torch.float32, device=self.args.device
            )
            torch.distributed.all_reduce(
                num_tokens_tensor, op=torch.distributed.ReduceOp.AVG, group=self.args.dp_group
            )
            self.state.num_input_tokens_seen = num_tokens_tensor.item()
            logs["num_tokens_seen"] = self.state.num_input_tokens_seen * self.args.dp_world_size
            logs["throughput"] = self.state.num_input_tokens_seen / self.state.running_time
        if self.args.log_flops:
            flops_tensor = torch.tensor(self.state.total_flos, dtype=torch.float32, device=self.args.device)
            torch.distributed.all_reduce(flops_tensor, op=torch.distributed.ReduceOp.AVG, group=self.args.dp_group)
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
                group=self.args.dp_group,
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

        self.model.train()

        if self.args.pp_world_size > 1:
            module = PipelineModule(
                self.model,
                pipeline_model_parallel_size=args.pp_world_size,
                pipeline_model_parallel_rank=args.pp_rank,
                data_parallel_size=args.dp_world_size,
                data_parallel_rank=args.dp_rank,
            )
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
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
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
                batch_samples = self.get_batch_samples(epoch_iterator, num_batches, args.device)
                step += num_batches

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

                    if args.log_flops and args.pp_rank == 0:
                        self.state.total_flos += (
                            float(self.model.floating_point_ops(inputs)) / args.pp_world_size / 1e12 * 3
                        )

                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                losses = pipeline_schedule.step(batch_samples)

                tr_loss = tr_loss + losses.mean()

                self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)
                with torch.cuda.nvtx.range("optimizer_step"):
                    self.deepspeed_engine.step()
                self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    grad_norm = self.deepspeed_engine.get_global_grad_norm()
                    # In some cases the grad norm may not return a float
                    if hasattr(grad_norm, "item"):
                        grad_norm = grad_norm.item()

                # get leaning rate before update
                learning_rate = self._get_learning_rate()

                if not getattr(self.optimizer, "overflow", False):
                    # Delay optimizer scheduling until metrics are generated
                    if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step()

                model.zero_grad()

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

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

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
