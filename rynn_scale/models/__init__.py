import importlib
import inspect
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import reduce
from typing import Any, Dict, Optional

import torch
from torch.distributed.tensor import distribute_tensor
from tqdm import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
    PROCESSOR_MAPPING,
    AutoModel,
    AutoModelForImageTextToText,
    PreTrainedModel,
)
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    cached_file,
)

from .. import parallel_state as mpu
from ..utils import logging, storage

logger = logging.get_logger(__name__)


@contextmanager
def _init_empty_params():
    old_device = torch.get_default_device()

    def move_init_to_device(func):
        def decorator(self, *args, **kwargs):
            torch.set_default_device(old_device)
            func(self, *args, **kwargs)
            torch.set_default_device("meta")

        return decorator

    def apply_patch(cls):
        if hasattr(cls, "_orig_init"):
            return

        if "RotaryEmbedding" in cls.__name__:
            cls._orig_init = cls.__init__
            cls.__init__ = move_init_to_device(cls.__init__)

        if not hasattr(cls, "_orig_init_subclass"):
            cls._orig_init_subclass = cls.__init_subclass__

            @classmethod
            def patched_init_subclass(sub_cls, **kwargs):
                sub_cls._orig_init_subclass(**kwargs)
                apply_patch(sub_cls)

            cls.__init_subclass__ = patched_init_subclass

        for sub in cls.__subclasses__():
            if "__init__" in sub.__dict__:
                apply_patch(sub)

    def restore_patch(cls):
        if hasattr(cls, "_orig_init"):
            cls.__init__ = cls._orig_init
            del cls._orig_init

        if hasattr(cls, "_orig_init_subclass"):
            cls.__init_subclass__ = cls._orig_init_subclass
            del cls._orig_init_subclass

        for sub in cls.__subclasses__():
            restore_patch(sub)

    try:
        torch.set_default_device("meta")
        apply_patch(torch.nn.Module)
        yield
    finally:
        torch.set_default_device(old_device)
        restore_patch(torch.nn.Module)


def _get_local_path(
    pretrained_model_name_or_path: str,
    filename: str,
    _raise_exceptions_for_gated_repo: bool = True,
    _raise_exceptions_for_missing_entries: bool = True,
):
    local_path = os.path.join(pretrained_model_name_or_path, filename)
    if storage.exists(local_path):
        return local_path
    if storage.is_oss(local_path):
        return None
    return cached_file(
        pretrained_model_name_or_path,
        filename=filename,
        _raise_exceptions_for_gated_repo=_raise_exceptions_for_gated_repo,
        _raise_exceptions_for_missing_entries=_raise_exceptions_for_missing_entries,
    )


@torch.no_grad()
def _init_missing_param(model: PreTrainedModel, param_name: str, tensor: torch.Tensor) -> None:
    module_path, _, attr = param_name.rpartition(".")

    # Walk to the owning module, tracking the innermost enclosing `PreTrainedModel` on
    # the way down the same way transformers' `smart_apply` dispatches: each sub-model
    # brings its own `_init_weights` *and* its own config, so a vision tower must not
    # inherit the LLM's `initializer_range`.
    module = owner = model
    for part in module_path.split(".") if module_path else []:
        module = getattr(module, part)
        if isinstance(module, PreTrainedModel):
            owner = module

    # Direct holder write because `setattr` rejects a plain tensor in a parameter
    # slot; `nn.Parameter` shares storage, so writes reach `tensor`.
    is_param = attr in module._parameters
    holder = module._parameters if is_param else module._buffers
    original = holder[attr]
    holder[attr] = torch.nn.Parameter(tensor, requires_grad=original.requires_grad) if is_param else tensor

    # Sentinel: a bare custom `nn.Parameter` matches no `_init_weights` branch and
    # would otherwise keep the allocator's contents with no way to notice.
    sentinel = tensor.is_floating_point()
    tensor.fill_(float("nan") if sentinel else 0)
    try:
        owner._init_weights(module)
    finally:
        holder[attr] = original

    if sentinel and tensor.isnan().any():
        std = getattr(owner.config, "initializer_range", None)
        if std is None:
            std = getattr(owner.config.get_text_config(), "initializer_range", 0.02)
        logger.warning(
            f"'{param_name}' was not written by `{type(owner).__name__}._init_weights` "
            f"(module `{type(module).__name__}`); falling back to normal_(0, {std}). "
            f"Either no branch matches, or the branch copied instead of writing in place."
        )
        tensor.normal_(mean=0.0, std=std)


@torch.no_grad()
def init_weights(
    model: PreTrainedModel,
    pretrained_model_name_or_path: Optional[str] = None,
    num_workers: int = 4,
):
    head_key = "lm_head.weight"
    embedding_key = "model.language_model.embed_tokens.weight"

    if pretrained_model_name_or_path is None:
        weight_map = {}
    else:
        if torch.distributed.get_rank() == 0:
            index_file = _get_local_path(
                pretrained_model_name_or_path,
                filename=SAFE_WEIGHTS_INDEX_NAME,
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
            )
            if index_file is not None:
                with storage.open_file(index_file, "rb") as f:
                    weight_map = json.load(f)["weight_map"]
            else:
                single_file = _get_local_path(
                    pretrained_model_name_or_path,
                    filename=SAFE_WEIGHTS_NAME,
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_missing_entries=False,
                )
                assert single_file is not None, (
                    f"Neither {SAFE_WEIGHTS_INDEX_NAME} nor {SAFE_WEIGHTS_NAME} "
                    f"found for {pretrained_model_name_or_path}"
                )
                with storage.open_safetensors(single_file) as f:
                    weight_map = {key: SAFE_WEIGHTS_NAME for key in f.keys()}
        else:
            weight_map = None

        results = [weight_map]
        torch.distributed.broadcast_object_list(results, src=0)
        weight_map = results[0]

    if "convert" in inspect.signature(model.state_dict).parameters:
        meta_state_dict = model.state_dict(convert=True)
    else:
        meta_state_dict = model.state_dict()

    fsdp_rank = mpu.get_data_parallel_rank(with_context_parallel=True)
    fsdp_world_size = mpu.get_data_parallel_world_size(with_context_parallel=True)

    ep_world_size = mpu.get_expert_model_parallel_world_size()
    expert_dp_rank = mpu.get_expert_data_parallel_rank()
    expert_dp_world_size = mpu.get_expert_data_parallel_world_size()

    param_names = list(meta_state_dict.keys())
    if model.config.tie_word_embeddings and head_key in param_names:
        param_names.remove(head_key)
        if embedding_key not in param_names:
            param_names.append(embedding_key)

    def _is_expert_param(param):
        mesh = getattr(param, "device_mesh", None)
        if mesh is None or mesh.mesh_dim_names is None:
            return False
        return "ep" in mesh.mesh_dim_names

    ep_src_range = max(min(expert_dp_world_size, ep_world_size), 1)

    src_data_ranks = {}
    non_expert_idx = 0
    expert_idx = 0
    for name in param_names:
        if _is_expert_param(meta_state_dict[name]):
            src_data_ranks[name] = expert_idx % ep_src_range
            expert_idx += 1
        else:
            src_data_ranks[name] = non_expert_idx % fsdp_world_size
            non_expert_idx += 1

    local_param_names = []
    for name in param_names:
        src = src_data_ranks[name]
        if _is_expert_param(meta_state_dict[name]):
            if expert_dp_rank == src:
                local_param_names.append(name)
        else:
            if fsdp_rank == src:
                local_param_names.append(name)

    def load_weight(param_name):
        checkpoint_key = param_name
        if checkpoint_key not in weight_map and model.base_model_prefix:
            checkpoint_key = f"{model.base_model_prefix}.{param_name}"
        if checkpoint_key not in weight_map:
            return None
        local_checkpoint_file = _get_local_path(
            pretrained_model_name_or_path,
            filename=weight_map[checkpoint_key],
        )
        with storage.open_safetensors(local_checkpoint_file) as f:
            return f.get_tensor(checkpoint_key)

    missing_keys = set(meta_state_dict.keys())
    state_dict = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {param_name: executor.submit(load_weight, param_name) for param_name in local_param_names}

        for param_name in tqdm(param_names, desc="Init weights", disable=fsdp_rank != 0):
            src_data_rank = src_data_ranks[param_name]
            param = meta_state_dict[param_name]

            if param_name in futures:
                tensor = futures[param_name].result()
                if tensor is None:
                    tensor = torch.empty(param.shape, dtype=param.dtype, device="cuda")
                    _init_missing_param(model, param_name, tensor)
                else:
                    tensor = tensor.to(dtype=param.dtype, device="cuda")
                    missing_keys.remove(param_name)
            else:
                tensor = torch.empty(param.shape, dtype=param.dtype, device="cuda")

            dtensor = distribute_tensor(
                tensor,
                device_mesh=param.device_mesh,
                placements=param.placements,
                src_data_rank=src_data_rank,
            )
            state_dict[param_name] = torch.nn.Parameter(dtensor)

    if torch.distributed.get_rank() == 0:
        all_missing_keys = [None] * torch.distributed.get_world_size()
    else:
        all_missing_keys = None

    torch.distributed.gather_object(
        obj=missing_keys,
        object_gather_list=all_missing_keys,
        dst=0,
    )

    if pretrained_model_name_or_path is not None and torch.distributed.get_rank() == 0:
        missing_keys = reduce(set.intersection, all_missing_keys)
        if model.config.tie_word_embeddings and head_key in meta_state_dict and embedding_key not in missing_keys:
            missing_keys.remove(head_key)
        if missing_keys:
            logger.warning(
                f"Loaded checkpoint from '{pretrained_model_name_or_path}', initialized missing keys: {missing_keys}"
            )
        else:
            logger.info(f"Loaded checkpoint from '{pretrained_model_name_or_path}'")

    if model.config.tie_word_embeddings and head_key in meta_state_dict:
        state_dict[head_key] = state_dict[embedding_key]
        if embedding_key not in meta_state_dict:
            state_dict.pop(embedding_key)

    if "convert" in inspect.signature(model.load_state_dict).parameters:
        model.load_state_dict(state_dict, strict=True, assign=True, convert=True)
    else:
        model.load_state_dict(state_dict, strict=True, assign=True)

    return model


def _infer_model_type(
    model_path: str,
    model_type: Optional[str] = None,
):
    if model_type is not None:
        return model_type

    # Read the raw JSON rather than going through AutoConfig: a custom model type
    # is absent from transformers' CONFIG_MAPPING until the model package's
    # apply_monkey_patch() registers it, and that cannot run before we know which
    # package to import.
    with storage.open_file(os.path.join(model_path, "config.json")) as f:
        return json.loads(f.read())["model_type"]


def build_processor(
    model_type: Optional[str],
    model_path: str,
    processor_overrides: Optional[Dict[str, Any]] = None,
):
    processor_overrides = processor_overrides or {}
    model_type = _infer_model_type(model_path=model_path, model_type=model_type)

    module = importlib.import_module(f".{model_type}", package=__package__)
    assert hasattr(module, "apply_monkey_patch")
    logger.info(f"Apply monkey patch for `{model_type}` using {module.apply_monkey_patch}")
    module.apply_monkey_patch()

    processor_class = PROCESSOR_MAPPING[CONFIG_MAPPING[model_type]]
    return storage.load_processor(
        model_path,
        processor_class=processor_class,
        **processor_overrides,
    )


def build_model(
    model_type: str,
    model_path: str,
    param_dtype: torch.dtype,
    attn_implementation: str,
    config_overrides: Optional[Dict[str, Any]] = None,
    vision_encoder_path: Optional[str] = None,
    reduced_layers_in_stage_zero: int = 0,
    reshard_after_forward: bool = False,
    master_param_dtype: torch.dtype = torch.float32,
    reduce_dtype: torch.dtype = torch.float32,
    processor: Optional[Any] = None,
):
    config_overrides = config_overrides or {}

    # Resolve the model type and register the package before touching the config:
    # apply_monkey_patch() is what puts custom types into CONFIG_MAPPING.
    model_type = _infer_model_type(model_path=model_path, model_type=model_type)

    module_dir = os.path.join(os.path.dirname(__file__), model_type)
    assert os.path.isdir(module_dir)
    module = importlib.import_module(f".{model_type}", package=__package__)
    assert hasattr(module, "apply_monkey_patch")
    logger.info(f"Apply monkey patch for `{model_type}` using {module.apply_monkey_patch}")
    module.apply_monkey_patch()

    config = storage.load_config(model_path, model_type=model_type, **config_overrides)

    if processor is None:
        processor = storage.load_processor(model_path)
    with _init_empty_params():
        if type(config) in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING:
            auto_class = AutoModelForImageTextToText
        else:
            # Model packages that register into MODEL_MAPPING instead of the
            # image-text-to-text mapping (e.g. rynn_brain_vla) are reachable
            # only through AutoModel.
            auto_class = AutoModel
        model = auto_class.from_config(
            config=config,
            attn_implementation=attn_implementation,
            dtype=master_param_dtype,
        )

    pp_world_size = mpu.get_pipeline_model_parallel_world_size()
    pp_rank = mpu.get_pipeline_model_parallel_rank()

    if pp_world_size > 1:
        assert hasattr(model, "apply_pipeline_parallel")
        model.apply_pipeline_parallel(
            num_stages=pp_world_size,
            stage_index=pp_rank,
            reduced_layers_in_stage_zero=reduced_layers_in_stage_zero,
        )

    ep_world_size = mpu.get_expert_model_parallel_world_size()

    if ep_world_size > 1:
        assert hasattr(model, "apply_expert_parallel")
        model.apply_expert_parallel(
            expert_device_mesh=mpu.get_expert_device_mesh(),
        )

    model.apply_fully_sharded_data_parallel(
        device_mesh=mpu.get_device_mesh(),
        expert_device_mesh=mpu.get_expert_device_mesh(),
        mp_policy=torch.distributed.fsdp.MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cast_forward_inputs=False,
        ),
        reshard_after_forward=reshard_after_forward,
    )

    return model, processor
