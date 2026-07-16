import json
import math
import os
from dataclasses import dataclass, field, fields
from datetime import timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import deepspeed
import torch
from packaging import version
from transformers import AutoConfig
from transformers.trainer_utils import IntervalStrategy, SaveStrategy
from transformers.training_args import OptimizerNames, SchedulerType

from . import parallel_state as mpu
from .registry import DATASET_REGISTRY
from .utils import logging, oss
from .utils.pipeline_parallel import PipelineSchedule

logger = logging.get_logger(__name__)


@dataclass
class BaseArguments:
    def __post_init__(self):
        pass

    def to_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self) if field.init}

    def to_json_string(self):
        data_dict = self.to_dict()
        for key, value in data_dict.items():
            if isinstance(value, Enum):
                data_dict[key] = value.value
        return json.dumps(data_dict, indent=2)


@dataclass
class ModelArguments(BaseArguments):
    model_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None)
    vision_encoder_path: Optional[str] = field(default=None)

    attn_implementation: Optional[str] = field(default="flash_attention_2")

    fp16: bool = field(default=False)
    bf16: bool = field(default=True)

    use_token_compression: Optional[bool] = field(default=False)

    def __post_init__(self):
        super().__post_init__()
        assert self.model_path is not None

        if self.model_type is None:
            if self.model_path.startswith("oss://"):
                config = oss.load_config(self.model_path)
            else:
                config = AutoConfig.from_pretrained(self.model_path)
            self.model_type = config.model_type

        if self.bf16:
            self.dtype = torch.bfloat16
        elif self.fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32


@dataclass
class ParallelismArguments(BaseArguments):
    pipeline_parallel_size: int = field(default=1)
    pipeline_parallel_schedule: Optional[str] = field(
        default=None, metadata={"choices": [item.value for item in PipelineSchedule]}
    )
    reduced_layers_in_stage_zero: int = field(default=0)

    expert_parallel_size: int = field(default=1)

    context_parallel_size: int = field(default=1)
    encoder_context_parallel_size: int = field(default=1)

    pp_broadcast_data: bool = field(default=False)
    cp_broadcast_data: bool = field(default=False)

    ddp_timeout: int = field(default=7200)

    def __post_init__(self):
        super().__post_init__()

        self.local_rank = int(os.environ.get("LOCAL_RANK"))
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl",
                device_id=self.device,
                timeout=timedelta(seconds=self.ddp_timeout),
            )

        deepspeed.init_distributed(dist_backend="nccl")

        self.global_world_size = torch.distributed.get_world_size()
        self.global_rank = torch.distributed.get_rank()

        assert 1 <= self.pipeline_parallel_size <= torch.distributed.get_world_size()
        assert 1 <= self.expert_parallel_size <= torch.distributed.get_world_size()
        assert 1 <= self.context_parallel_size <= torch.distributed.get_world_size()
        assert 1 <= self.encoder_context_parallel_size <= torch.distributed.get_world_size()
        assert self.reduced_layers_in_stage_zero >= 0

        if self.pipeline_parallel_size > 1:
            assert self.pipeline_parallel_schedule is not None
        else:
            assert self.pipeline_parallel_schedule is None

        self.pipeline_parallel_schedule = PipelineSchedule(self.pipeline_parallel_schedule)

        mpu.initialize_model_parallel(
            pipeline_model_parallel_size=self.pipeline_parallel_size,
            expert_model_parallel_size=self.expert_parallel_size,
            context_parallel_size=self.context_parallel_size,
            encoder_context_parallel_size=self.encoder_context_parallel_size,
        )

        self.dp_group = mpu.get_data_parallel_group()
        self.dp_world_size = mpu.get_data_parallel_world_size()
        self.dp_rank = mpu.get_data_parallel_rank()

        self.dcp_group = mpu.get_data_parallel_group(with_context_parallel=True)
        self.dcp_world_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
        self.dcp_rank = mpu.get_data_parallel_rank(with_context_parallel=True)

        self.cp_group = mpu.get_context_parallel_group()
        self.cp_world_size = mpu.get_context_parallel_world_size()
        self.cp_rank = mpu.get_context_parallel_rank()

        self.pp_group = mpu.get_pipeline_model_parallel_group()
        self.pp_world_size = mpu.get_pipeline_model_parallel_world_size()
        self.pp_rank = mpu.get_pipeline_model_parallel_rank()

        self.ep_group = mpu.get_expert_model_parallel_group()
        self.ep_world_size = mpu.get_expert_model_parallel_world_size()
        self.ep_rank = mpu.get_expert_model_parallel_rank()

        self.edp_group = mpu.get_expert_data_parallel_group()
        self.edp_world_size = mpu.get_expert_data_parallel_world_size()
        self.edp_rank = mpu.get_expert_data_parallel_rank()


@dataclass
class DataArguments(BaseArguments):
    data_type: str = field(default=None)
    data_path: List[str] = field(default=None)
    data_mixture: Optional[str] = field(default=None)

    # Data Processing configs
    model_max_length: Optional[int] = field(default=16384)
    mm_max_length: Optional[int] = field(default=10240)
    fps: Optional[int] = field(default=1)
    max_frames: Optional[int] = field(default=180)

    def __post_init__(self):
        super().__post_init__()

        if self.data_mixture is not None:
            assert self.data_type is None and self.data_path is None
            assert os.path.isfile(self.data_mixture)

            with open(self.data_mixture, "r") as f:
                data_mixture = json.load(f)

            assert isinstance(data_mixture, list)
            for data_source in data_mixture:
                assert isinstance(data_source, dict)
                assert set(data_source.keys()) == {"data_type", "data_path", "sampling_rate"}

            logger.info(f"Using data mixture: {data_mixture}")

        else:
            assert self.data_type is not None
            assert self.data_type in DATASET_REGISTRY, f"Available data types: {DATASET_REGISTRY.keys()}"


@dataclass
class TrainingArguments(ModelArguments, ParallelismArguments, DataArguments, BaseArguments):
    # Efficiency-related configs
    deepspeed: str = field(default=None)

    gradient_checkpointing: bool = field(default=False)
    gradient_checkpointing_kwargs: Optional[Union[dict[str, Any], str]] = field(
        default=None,
        metadata={
            "help": "Gradient checkpointing key word arguments such as `use_reentrant`. Will be passed to `torch.utils.checkpoint.checkpoint` through `model.gradient_checkpointing_enable`."
        },
    )
    encoder_gradient_checkpointing_interval: Optional[int] = field(default=None)

    sequence_packing: bool = field(default=True)
    decoder_load_balancing: bool = field(default=False)

    dynamic_batching: bool = field(default=False)
    dynamic_batching_window_size: int = field(default=128)

    # Data configs
    micro_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)

    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )

    # Data loading configs
    dataloader_num_workers: int = field(default=0)
    dataloader_drop_last: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    dataloader_persistent_workers: bool = field(default=False)
    dataloader_prefetch_factor: Optional[int] = field(default=None)

    # Optimizer configs
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    frozen_parameters: Optional[List[str]] = field(default=None)

    lr_scheduler_type: Union[SchedulerType, str] = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )
    lr_scheduler_kwargs: Union[dict[str, Any], str] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Extra parameters for the lr_scheduler such as {'num_cycles': 1} for the cosine with hard restarts."
            )
        },
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    optim: Union[OptimizerNames, str] = field(
        default="adamw_torch_fused" if version.parse(torch.__version__) >= version.parse("2.8") else "adamw_torch",
        metadata={"help": "The optimizer to use.", "choices": [item.value for item in OptimizerNames]},
    )
    optim_args: Optional[str] = field(default=None, metadata={"help": "Optional arguments to supply to optimizer."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    # Loss configs
    loss_implementation: str = field(default="torch")
    loss_reduction_scope: str = field(default="sequence")
    average_tokens_across_devices: bool = field(default=True)

    # Eval configs
    eval_strategy: Union[IntervalStrategy, str] = field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
    )
    eval_steps: Optional[float] = field(default=None)

    # Log configs
    output_dir: str = field(default="outputs")
    log_flops: bool = field(default=False)
    log_seen_tokens: bool = field(default=False)
    report_to: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of integrations to report the results and logs to."}
    )

    logging_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    logging_steps: int = field(default=10)
    logging_first_step: bool = field(default=False, metadata={"help": "Log the first global_step"})
    log_level: str = field(default="info", metadata={"choices": [item.value for item in logging.LogLevel]})
    log_level_replica: str = field(default="warning", metadata={"choices": [item.value for item in logging.LogLevel]})
    disable_tqdm: bool = field(default=False)

    save_strategy: Union[SaveStrategy, str] = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    save_steps: int = field(default=1000)
    save_total_limit: Optional[int] = field(default=None)
    save_full_model: bool = field(default=True)

    restore_callback_states_from_checkpoint: bool = field(default=False)

    # Misc
    synchronize_experts_before_forward: bool = field(default=False)
    cleanup_before_optimizer_step: bool = field(default=False)

    # Reproducibility
    seed: int = field(default=42)
    full_determinism: bool = field(default=False)

    def __post_init__(self):
        super().__post_init__()

        if self.encoder_gradient_checkpointing_interval is not None:
            assert self.gradient_checkpointing
            assert self.encoder_gradient_checkpointing_interval > 0

        if self.sequence_packing:
            assert "flash_attention" in self.attn_implementation, "Sequence packing requires flash attention."

        if self.decoder_load_balancing:
            assert self.sequence_packing, "DP load balancing requires batch flattening."
            assert not self.dynamic_batching, "DP load balancing and dynamic batching cannot be used together."

        if self.dynamic_batching:
            assert self.sequence_packing, "Dynamic batching requires batch flattening."
            assert not self.decoder_load_balancing, "Dynamic batching and workload balancing cannot be used together."

        assert self.loss_reduction_scope in ["batch", "sequence"], (
            f"Unsupported loss reduction scope: {self.loss_reduction_scope}"
        )
        if self.loss_reduction_scope == "sequence":
            assert self.average_tokens_across_devices

        self.logging_dir = self.output_dir
        self.log_level = logging.LogLevel(self.log_level)
        self.log_level_replica = logging.LogLevel(self.log_level_replica)
        log_level = self.log_level if self.global_rank == 0 else self.log_level_replica
        logging.set_verbosity(log_level)

        self.eval_strategy = IntervalStrategy(self.eval_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = SaveStrategy(self.save_strategy)

        for attr in ["log_flops", "log_seen_tokens"]:
            if getattr(self, attr):
                logger.warn(f"The `{attr}` argument can only be used for debugging.")

        assert self.deepspeed is not None, "DeepSpeed config path is required."
        assert os.path.isfile(self.deepspeed)
        with open(self.deepspeed, "r") as f:
            deepspeed_config = json.load(f)
        self.deepspeed_config = self._process_deepspeed_config(deepspeed_config)

        zero_stage = self.deepspeed_config.get("zero_optimization", {}).get("stage", 0)
        if zero_stage == 3:
            assert self.pipeline_parallel_size == 1, "ZeRO-3 is incompatible with pipeline parallelism."
            assert self.expert_parallel_size == 1, "ZeRO-3 is incompatible with expert parallelism."

        if self.synchronize_experts_before_forward:
            assert self.ep_world_size > 1

    def _process_deepspeed_config(self, deepspeed_config: Dict[str, Any]):
        if self.model_path.startswith("oss://"):
            config = oss.load_config(self.model_path)
        else:
            config = AutoConfig.from_pretrained(self.model_path)
        hidden_size = config.get_text_config().hidden_size

        def _process_auto(config, prefix=""):
            config = config.copy()
            for key, value in config.items():
                global_key = prefix + key
                if isinstance(value, dict):
                    config[key] = _process_auto(value, prefix=global_key + ".")
                elif value == "auto":
                    if global_key == "train_micro_batch_size_per_gpu":
                        config[key] = self.micro_batch_size
                    elif global_key == "gradient_accumulation_steps":
                        config[key] = self.gradient_accumulation_steps
                    elif global_key == "gradient_clipping":
                        config[key] = self.max_grad_norm
                    elif global_key == "fp16.enabled":
                        config[key] = self.fp16
                    elif global_key == "bf16.enabled":
                        config[key] = self.bf16
                    elif global_key == "zero_optimization.reduce_bucket_size":
                        config[key] = hidden_size * hidden_size
                    elif global_key == "zero_optimization.stage3_prefetch_bucket_size":
                        config[key] = int(0.9 * hidden_size * hidden_size)
                    elif global_key == "zero_optimization.stage3_param_persistence_threshold":
                        config[key] = 10 * hidden_size
                    else:
                        raise ValueError(f"Unsupported auto config: {key}")
            return config

        return _process_auto(deepspeed_config)

    def get_warmup_steps(self, num_training_steps: int):
        warmup_steps = (
            self.warmup_steps if self.warmup_steps > 0 else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps


@dataclass
class EvaluationArguments(ModelArguments):
    benchmarks: List[str] = field(default=None)
    prompt_format: str = field(default=None)
    enable_thinking: bool = field(default=False)
    save_dir: str = field(default=None)

    backend: str = field(default="hf", metadata={"choices": ["hf", "sglang"]})
    num_processor_workers: int = field(default=8)

    image_min_pixels: int = field(default=16 * 32 * 32)
    image_max_pixels: int = field(default=16384 * 32 * 32)
    video_min_pixels: int = field(default=16 * 32 * 32)
    video_max_pixels: int = field(default=16384 * 32 * 32)

    fps: int = field(default=1)
    max_frames: int = field(default=180)

    max_new_tokens: int = field(default=128)
    temperature: float = field(default=0.0)
    top_p: float = field(default=0.95)
    top_k: int = field(default=50)
    repetition_penalty: Optional[float] = field(default=None)

    tensor_parallel_size: int = field(default=1)
    expert_parallel_size: int = field(default=1)
    pipeline_parallel_size: int = field(default=1)

    def __post_init__(self):
        super().__post_init__()

        assert self.benchmarks is not None
        assert self.save_dir is not None

        self.processing_params = {
            "image_max_pixels": self.image_max_pixels,
            "image_min_pixels": self.image_min_pixels,
            "video_max_pixels": self.video_max_pixels,
            "video_min_pixels": self.video_min_pixels,
            "fps": self.fps,
            "max_frames": self.max_frames,
        }

        self.sampling_params = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        if self.repetition_penalty is not None:
            self.sampling_params["repetition_penalty"] = self.repetition_penalty

        if self.backend == "hf":
            assert self.tensor_parallel_size == 1
            assert self.expert_parallel_size == 1

        self.parallel_params = {
            "tp_size": self.tensor_parallel_size,
            "ep_size": self.expert_parallel_size,
            "pp_size": self.pipeline_parallel_size,
        }
