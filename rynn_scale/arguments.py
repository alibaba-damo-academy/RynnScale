import json
import math
import os
from dataclasses import dataclass, field, fields
from datetime import timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch
from packaging import version
from transformers.trainer_utils import IntervalStrategy, SaveStrategy
from transformers.training_args import OptimizerNames, SchedulerType

from . import parallel_state as mpu
from .constants import RotationRepresentation
from .registry import DATASET_REGISTRY
from .utils import logging, storage
from .utils.pipeline_parallel import PipelineSchedule

logger = logging.get_logger(__name__)

# Supported dtype names -> torch dtype. Used both for the ``choices`` of the dtype
# arguments and for resolving their string values.
DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


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
            elif isinstance(value, torch.dtype):
                data_dict[key] = str(value)
        return json.dumps(data_dict, indent=2)


@dataclass
class ModelArguments(BaseArguments):
    model_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None)
    config_overrides: Union[Dict[str, Any], str, None] = field(default=None)
    processor_overrides: Union[Dict[str, Any], str, None] = field(default=None)
    vision_encoder_path: Optional[str] = field(default=None)

    attn_implementation: Optional[str] = field(default="flash_attention_2")

    master_param_dtype: str = field(default="float32", metadata={"choices": list(DTYPE_MAP)})
    param_dtype: str = field(default="bfloat16", metadata={"choices": list(DTYPE_MAP)})
    reduce_dtype: str = field(default="float32", metadata={"choices": list(DTYPE_MAP)})

    use_token_compression: Optional[bool] = field(default=False)

    def __post_init__(self):
        super().__post_init__()
        assert self.model_path is not None

        if isinstance(self.config_overrides, str):
            self.config_overrides = json.loads(self.config_overrides)
        elif self.config_overrides is None:
            self.config_overrides = {}

        if isinstance(self.processor_overrides, str):
            self.processor_overrides = json.loads(self.processor_overrides)
        elif self.processor_overrides is None:
            self.processor_overrides = {}

        if self.model_type is None:
            config = storage.load_config(self.model_path)
            self.model_type = config.model_type

        self._resolve_dtypes()

    def _resolve_dtypes(self):
        """Turn the dtype argument strings into ``torch.dtype`` values."""
        self.param_dtype = DTYPE_MAP[self.param_dtype]
        self.master_param_dtype = (
            torch.float32 if self.master_param_dtype is None else DTYPE_MAP[self.master_param_dtype]
        )
        self.reduce_dtype = torch.float32 if self.reduce_dtype is None else DTYPE_MAP[self.reduce_dtype]


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

    reshard_after_forward: bool = field(default=False)

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

        self.global_world_size = torch.distributed.get_world_size()
        self.global_rank = torch.distributed.get_rank()

        assert 1 <= self.pipeline_parallel_size <= torch.distributed.get_world_size()
        assert 1 <= self.expert_parallel_size <= torch.distributed.get_world_size()
        assert 1 <= self.context_parallel_size <= torch.distributed.get_world_size()
        assert 1 <= self.encoder_context_parallel_size <= torch.distributed.get_world_size()
        assert self.reduced_layers_in_stage_zero >= 0

        if self.pipeline_parallel_size > 1:
            assert self.pipeline_parallel_schedule is not None
            assert not self.reshard_after_forward, "reshard_after_forward is not supported with pipeline parallelism."
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
    data_path: str = field(default=None)
    data_mixture: List[Dict[str, Any]] | str | None = field(
        default=None,
        metadata={
            "help": "Either a path to a JSON file or an inline JSON string, encoding a list of data source dicts."
        },
    )

    # VLM processing configs
    model_max_length: Optional[int] = field(default=16384)
    mm_max_length: Optional[int] = field(default=10240)
    fps: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "VLM only: frames-per-second to sub-sample from input videos "
                "(processor extracts duration*fps frames, capped by max_frames). "
                "Ignored by VLA datasets — use target_fps for action/state resampling."
            )
        },
    )
    max_frames: Optional[int] = field(
        default=180,
        metadata={"help": "VLM only: hard cap on frames extracted per input video (pairs with fps)."},
    )

    # VLA processing configs
    action_chunk_size: int = field(default=20)
    use_delta_action: bool = field(default=True)
    eef_rotation_repr: Optional[str] = field(
        default=None, metadata={"choices": [item.value for item in RotationRepresentation]}
    )
    action_only: bool = field(default=False)
    target_fps: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "VLA only: resample action/state time-series to this rate "
                "(linear for joint/eef_pos, SLERP for rotation, nearest for gripper). "
                "None = use each episode's native fps (no resampling). "
                "Distinct from `fps`, which sub-samples VLM video frames."
            )
        },
    )

    # Episode iterator configs
    use_episode_iterator: bool = field(
        default=False,
        metadata={"help": "Use episode-streaming (IterableDataset) mode instead of map-style random access."},
    )
    episode_iterator_buffer: int = field(
        default=4,
        metadata={"help": "Number of concurrent episodes per worker for interleaving."},
    )
    episode_iterator_shuffle_buffer: int = field(
        default=1024,
        metadata={"help": "Reservoir shuffle buffer size per worker."},
    )

    def __post_init__(self):
        super().__post_init__()

        if self.data_mixture is not None:
            assert self.data_type is None and self.data_path is None
            if isinstance(self.data_mixture, str):
                if os.path.isfile(self.data_mixture):
                    with open(self.data_mixture, "r") as f:
                        data_mixture = json.load(f)
                else:
                    try:
                        data_mixture = json.loads(self.data_mixture)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"data_mixture {self.data_mixture!r} is neither an existing file nor a valid JSON string: {e}"
                        ) from e

                assert isinstance(data_mixture, list)
                for data_source in data_mixture:
                    assert isinstance(data_source, dict)

                self.data_mixture = data_mixture
                logger.info(f"Using data mixture: {data_mixture}")

        else:
            assert self.data_type is not None
            assert self.data_type in DATASET_REGISTRY, f"Available data types: {DATASET_REGISTRY.keys()}"

        if self.eef_rotation_repr is not None:
            self.eef_rotation_repr = RotationRepresentation(self.eef_rotation_repr)


@dataclass
class TrainingArguments(ModelArguments, ParallelismArguments, DataArguments, BaseArguments):
    # Efficiency-related configs
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
    learning_rate_strategy: Union[Dict[str, float], str, None] = field(
        default=None,
        metadata={
            "help": (
                "Per-parameter learning rates. Keys are regex patterns (like `frozen_parameters`); values are "
                "learning rates. A value of 0 freezes the matched parameters (requires_grad=False, excluded from "
                "the optimizer). Parameters not matched by any regex use `learning_rate`. Each parameter must "
                "match at most one regex, and all learning rates must be >= 0. Accepts a JSON string."
            )
        },
    )

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

    restore_callback_states_from_checkpoint: bool = field(default=False)

    # Misc
    synchronize_experts_before_forward: bool = field(default=False)
    cleanup_before_optimizer_step: bool = field(default=False)

    # Reproducibility
    seed: int = field(default=42)
    full_determinism: bool = field(default=False)

    def __post_init__(self):
        super().__post_init__()

        seed = torch.tensor(self.seed, device="cuda")
        torch.distributed.broadcast(seed, src=0)
        self.seed = seed.item()

        assert self.gradient_accumulation_steps >= self.pipeline_parallel_size

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

        if self.synchronize_experts_before_forward:
            assert self.ep_world_size > 1

        if isinstance(self.learning_rate_strategy, str):
            self.learning_rate_strategy = json.loads(self.learning_rate_strategy)
        if self.learning_rate_strategy is not None:
            assert isinstance(self.learning_rate_strategy, dict)
            for pattern, lr in self.learning_rate_strategy.items():
                assert isinstance(pattern, str)
                assert isinstance(lr, (int, float)) and lr >= 0, (
                    f"learning_rate_strategy['{pattern}'] must be a number >= 0, got {lr}"
                )

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
    save_rollout: bool = field(default=False)

    engine: str = field(default="vla", metadata={"choices": ["hf", "sglang", "vla"]})
    max_concurrent_episodes: int = field(default=128)
    max_running_requests: int = field(default=16)  # @serve.batch max batch size
    batch_wait_timeout_s: float = field(default=0.02)

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
        # Sizes the evaluator's episode semaphore verbatim -- 0 would admit nothing
        # and hang the run rather than report anything.
        assert self.max_concurrent_episodes >= 1, "max_concurrent_episodes must be >= 1"

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

        self.parallel_params = {
            "tp_size": self.tensor_parallel_size,
            "ep_size": self.expert_parallel_size,
            "pp_size": self.pipeline_parallel_size,
        }


@dataclass
class ServeArguments(ModelArguments):
    """Args for the HTTP serving entrypoint (``api/serve.py``).

    Reuses the serving knobs of :class:`EvaluationArguments` -- ``engine``
    selects the backend (vla/hf/sglang) and the request schema; the
    ``InferenceServer`` replicas fill every GPU (replica GPU count =
    ``tensor_parallel_size * pipeline_parallel_size``). Unlike eval it has no
    benchmarks/save_dir -- it just adds HTTP routes to those replicas on
    ``host``/``port``.
    """

    engine: str = field(default="vla", metadata={"choices": ["hf", "sglang", "vla"]})

    # Serving knobs (shared names with EvaluationArguments).
    max_running_requests: int = field(default=16)
    batch_wait_timeout_s: float = field(default=0.02)

    tensor_parallel_size: int = field(default=1)
    expert_parallel_size: int = field(default=1)
    pipeline_parallel_size: int = field(default=1)

    # Number of Model replicas to serve. Each replica takes
    # ``tensor_parallel_size * pipeline_parallel_size`` GPUs.
    num_model_replicas: int = field(default=1)

    # VLM image/video preprocessing (ignored by the vla engine).
    image_min_pixels: int = field(default=16 * 32 * 32)
    image_max_pixels: int = field(default=16384 * 32 * 32)
    video_min_pixels: int = field(default=16 * 32 * 32)
    video_max_pixels: int = field(default=16384 * 32 * 32)
    fps: int = field(default=1)
    max_frames: int = field(default=180)

    # Sampling (hf/sglang engines).
    max_new_tokens: int = field(default=128)
    temperature: float = field(default=0.0)
    top_p: float = field(default=0.95)
    top_k: int = field(default=50)
    repetition_penalty: Optional[float] = field(default=None)

    # HTTP ingress.
    host: str = field(default="0.0.0.0")
    port: int = field(default=8000)
    route_prefix: str = field(default="/")

    # Verbosity of the server replicas (``debug`` adds a line per forward pass).
    log_level: str = field(default="info", metadata={"choices": [item.value for item in logging.LogLevel]})

    def __post_init__(self):
        super().__post_init__()

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

        self.parallel_params = {
            "tp_size": self.tensor_parallel_size,
            "ep_size": self.expert_parallel_size,
            "pp_size": self.pipeline_parallel_size,
        }


@dataclass
class ControlArguments(BaseArguments):
    """Args for the deployment entrypoint (``api/control.py``).

    The policy is *not* deploy's to start: ``server_url`` names a running
    ``rynn_scale.api.serve`` server, and every knob that shapes it -- the checkpoint, the
    engine, the batch and the GPUs it takes -- belongs to that command line rather than
    to this one. So there is no ``--model_path`` here: this process holds a robot, a
    command loop and a GUI, reaches the policy over HTTP, and needs no Ray cluster, no
    GPU and no model stack of its own. Restarting the policy no longer means restarting
    the robot either.
    """

    # The policy, as a URL. ``request_timeout`` bounds one inference: a controller that
    # waits forever on a wedged server holds its robot at the last commanded pose with
    # nothing said about why.
    server_url: str = field(default="http://localhost:8000")
    request_timeout: float = field(default=120.0)

    # Deploy environment. ``controller`` names a registered ``BaseRobotEnvironment``
    # (a real-robot env, or ``Libero`` to drive a sim env through the same
    # control/GUI path); it becomes the agent's ``env_type``. Everything else the env
    # takes goes through ``env_config`` below -- the world *and* the clock a real robot
    # is built with (``command_freq`` / ``control_freq``). There are deliberately no
    # per-knob flags for those: a rate is a property of the robot, not
    # of the run, so each env declares its own default and a flag here would be a
    # second one to keep in step -- one that, being unconditionally merged in, also
    # reached the sim envs it means nothing to. There is no step budget: deploy runs
    # until commanded to stop.
    controller: str = field(default="")
    # REPLAY source, optional: a recorded VLA dataset the GUI plays back
    # inference-free. ``api/control.py`` builds the reader from these and hands the
    # env the live dataset, so an unregistered ``data_type`` or an unreadable path
    # fails in the CLI -- where the traceback is visible, and before the robot has been
    # connected to -- rather than half-way into building the env. No
    # ``--data_path`` means no reader, which means REPLAY is unavailable.
    # ``target_fps`` resamples the recording (unset = the episode's native rate). The
    # rotation encoding is deliberately *not* a flag: the recording keeps whatever it
    # was written in, and the env converts it to its own on the way to a flat command.
    data_type: Optional[str] = field(default=None)
    data_path: Optional[str] = field(default=None)
    target_fps: Optional[float] = field(default=None)
    # Real-Time Chunking, and therefore the *agent's* knobs rather than the env's:
    # how many actions of the chunk in hand to commit before re-inferring (``None``
    # disables RTC -- infer, then play the whole chunk), and how many recent
    # inferences the latency the GUI displays averages over. Both clocks chunk: on a
    # real one the answer is spliced in the moment it lands, on a sim one the latency
    # is *simulated* at exactly ``infer_interval`` actions so the episode stays
    # reproducible (see ``RobotAgent``).
    infer_interval: Optional[int] = field(default=None)
    latency_window: int = field(default=10)
    # JSON (inline or a file path) passed as ctor kwargs to the env, and the only
    # channel to it. Says what defines the world -- e.g. a Libero task:
    # '{"suite": "libero_spatial", "task_id": 0}' -- and, for a real robot, the clock
    # it runs on: '{"command_freq": 30, "control_freq": 200}'.
    # Keys are the env ctor's own parameter names, so a misspelled one is a
    # ``TypeError`` from the ctor rather than a value silently ignored.
    env_config: Optional[str] = field(default=None)
    # How long to wait for the agent's env to come up before giving up on it. The
    # loop builds it on its own thread, and nothing else -- the GUI's layout, its
    # snapshots, the startup commands -- means anything until it has (see
    # ``RobotAgent.wait_until_ready``). Generous by default because this covers a
    # simulator loading a scene, an EGL context, or a real robot being connected to.
    startup_timeout: float = field(default=300.0)

    # Commands / GUI. ``prompt`` is what the robot is asked to do -- one name for
    # it end to end (this flag, a command's ``extra_args["prompt"]``,
    # ``loop(prompt=...)``, the inference request), because it is one value with one
    # destination: the policy. It never reaches the env.
    prompt: str = field(default="")
    preset_commands: Optional[str] = field(default=None)
    gui: bool = field(default=False)
    gradio_host: str = field(default="0.0.0.0")
    gradio_port: int = field(default=7860)
    # The agent actor pushes GUI snapshots into shared memory under this prefix and the
    # GUI actor's widgets read them there. Two processes on the one node, so the prefix is
    # the whole of what they agree on -- the observation path carries no RPC. One live
    # controller per prefix, like the real robot's own segments: ``sweep_stale`` reclaims
    # everything under it at startup, which is also what cleans up after a hard-killed
    # prior run.
    gui_shm_prefix: str = field(default="rynn_gui")
    # Per-camera frame cap. Declared rather than derived because a segment cannot be
    # resized and nobody knows the resolution until the first frame arrives; a frame
    # over the cap asserts (with the size it wanted) instead of being truncated.
    gui_max_image_bytes: int = field(default=1280 * 720 * 3)


@dataclass
class ReplayArguments(ModelArguments, DataArguments):
    save_dir: Optional[str] = field(default=None)
    render_size: int = field(default=320)
    num_segments: int = field(
        default=1,
        metadata={
            "help": "Total number of segments to replay. Each segment is sampled by first drawing a random episode, then a random segment within it."
        },
    )
    num_inference_steps: int = field(default=10)
    seed: int = field(default=42)
    episode_indices: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "If set, replay exactly these flat episode indices (overrides random seed sampling and num_episodes)."
        },
    )
    action_source: str = field(
        default="joint",
        metadata={
            "help": "Renderer action source: 'joint' replays recorded joint angles; "
            "'eef' IK-solves recorded EEF poses. Falls back to the other "
            "source when the preferred one is absent.",
            "choices": ["eef", "joint"],
        },
    )
    max_segments_per_episode: int = field(
        default=0,
        metadata={
            "help": "If >0, cap each replayed episode to this many segments (useful to keep demo runs bounded for very long episodes)."
        },
    )
    robot_types: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "If set, only sample episodes whose robot_type matches one of these values (e.g. --robot_types franka ur5)."
        },
    )

    def __post_init__(self):
        # ``ModelArguments.__post_init__`` asserts ``model_path is not None``
        # and probes the checkpoint for ``model_type``. When replaying
        # training data only (no model), skip those steps.
        if self.model_path is not None:
            super().__post_init__()
            return

        DataArguments.__post_init__(self)

        if isinstance(self.config_overrides, str):
            self.config_overrides = json.loads(self.config_overrides)
        elif self.config_overrides is None:
            self.config_overrides = {}

        if isinstance(self.processor_overrides, str):
            self.processor_overrides = json.loads(self.processor_overrides)
        elif self.processor_overrides is None:
            self.processor_overrides = {}

        self._resolve_dtypes()
