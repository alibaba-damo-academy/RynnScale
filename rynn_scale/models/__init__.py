import gc
import importlib
import inspect
import io
import json
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Set

import torch
from safetensors import safe_open
from safetensors.torch import load
from tqdm import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
    PROCESSOR_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    AutoVideoProcessor,
    PreTrainedModel,
)
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    cached_file,
)

from .. import parallel_state as mpu
from ..utils import logging, oss

logger = logging.get_logger(__name__)


def _sync_expert_params(
    expert_param_tags: Dict[str, int],
    sharded_state_dict: Dict[str, torch.Tensor],
    state_dict: Dict[str, torch.Tensor],
):
    ep_group = mpu.get_expert_model_parallel_group()
    ep_world_size = mpu.get_expert_model_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank()

    if ep_world_size == 1:
        return

    if ep_rank == 0:
        sync_keys = [[key for key in sharded_state_dict if key in expert_param_tags]]
    else:
        sync_keys = [None]

    torch.distributed.broadcast_object_list(
        sync_keys,
        group=ep_group,
        group_src=0,
    )

    p2p_ops = []

    for key in sync_keys[0]:
        tag = expert_param_tags[key]
        if ep_rank == 0:
            tensor = sharded_state_dict[key].to(state_dict[key])
            shared_tensors = tensor.chunk(ep_world_size, dim=0)
            for dst_rank in range(1, ep_world_size):
                logger.debug(
                    f"Rank {ep_rank} (global rank {torch.distributed.get_rank()}), send {key}({shared_tensors[dst_rank].shape}) to rank {dst_rank}"
                )
                p2p_ops.append(
                    torch.distributed.P2POp(
                        torch.distributed.isend,
                        shared_tensors[dst_rank].contiguous(),
                        group=ep_group,
                        tag=tag,
                        group_peer=dst_rank,
                    )
                )
            state_dict[key].copy_(shared_tensors[0], non_blocking=True)
        else:
            logger.debug(
                f"Rank {ep_rank} (global rank {torch.distributed.get_rank()}), recv {key}({state_dict[key].shape}) from rank 0"
            )
            p2p_ops.append(
                torch.distributed.P2POp(
                    torch.distributed.irecv,
                    state_dict[key],
                    group=ep_group,
                    tag=tag,
                    group_peer=0,
                )
            )

    if len(p2p_ops) > 0:
        works = torch.distributed.batch_isend_irecv(p2p_ops)
        for work in works:
            work.wait()


def _get_local_path(
    pretrained_model_name_or_path: str,
    filename: str,
    _raise_exceptions_for_gated_repo: bool = True,
    _raise_exceptions_for_missing_entries: bool = True,
):
    local_path = os.path.join(pretrained_model_name_or_path, filename)
    if local_path.startswith("oss://"):
        if oss.object_exists(local_path):
            return local_path
        return None
    if os.path.exists(local_path):
        return local_path
    return cached_file(
        pretrained_model_name_or_path,
        filename=filename,
        _raise_exceptions_for_gated_repo=_raise_exceptions_for_gated_repo,
        _raise_exceptions_for_missing_entries=_raise_exceptions_for_missing_entries,
    )


def _load_checkpoint_file(
    pretrained_model_name_or_path: str,
    filename: str,
    expert_param_tags: Dict[str, int],
    state_dict: Dict[str, torch.Tensor],
    missing_keys: Set[str],
) -> None:
    local_checkpoint_file = _get_local_path(
        pretrained_model_name_or_path,
        filename=filename,
    )

    if mpu.get_expert_model_parallel_rank() == 0:
        if filename.endswith(".safetensors"):
            if local_checkpoint_file.startswith("oss://"):
                sharded_state_dict = load(oss.get_object(local_checkpoint_file).read())
            else:
                sharded_state_dict = {}
                with safe_open(local_checkpoint_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        sharded_state_dict[key] = f.get_tensor(key)
        else:
            if local_checkpoint_file.startswith("oss://"):
                buffer = io.BytesIO(oss.get_object(local_checkpoint_file).read())
                sharded_state_dict = torch.load(buffer, map_location="cpu")
            else:
                sharded_state_dict = torch.load(local_checkpoint_file, map_location="cpu")

        for key, tensor in state_dict.items():
            if key in sharded_state_dict:
                missing_keys.discard(key)
                if key not in expert_param_tags:
                    tensor.copy_(sharded_state_dict[key], non_blocking=True)

    else:
        sharded_state_dict = None

    _sync_expert_params(expert_param_tags, sharded_state_dict, state_dict)


def _load_checkpoint_files(
    pretrained_model_name_or_path: str,
    expert_param_tags: Dict[str, int],
    state_dict: Dict[str, torch.Tensor],
) -> List[str]:
    missing_keys = set(state_dict.keys())

    checkpoint_file = _get_local_path(
        pretrained_model_name_or_path,
        filename=SAFE_WEIGHTS_NAME,
        _raise_exceptions_for_gated_repo=False,
        _raise_exceptions_for_missing_entries=False,
    )

    checkpoint_name = None
    if checkpoint_file is not None:
        checkpoint_name = SAFE_WEIGHTS_NAME
    else:
        checkpoint_file = _get_local_path(
            pretrained_model_name_or_path,
            filename=WEIGHTS_NAME,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
        )
        if checkpoint_file is not None:
            checkpoint_name = WEIGHTS_NAME

    if checkpoint_name is not None:
        _load_checkpoint_file(
            pretrained_model_name_or_path,
            filename=checkpoint_name,
            expert_param_tags=expert_param_tags,
            state_dict=state_dict,
            missing_keys=missing_keys,
        )
        return list(missing_keys)

    index_file = _get_local_path(
        pretrained_model_name_or_path,
        filename=SAFE_WEIGHTS_INDEX_NAME,
        _raise_exceptions_for_gated_repo=False,
        _raise_exceptions_for_missing_entries=False,
    )

    if index_file is None:
        index_file = _get_local_path(
            pretrained_model_name_or_path,
            filename=WEIGHTS_INDEX_NAME,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
        )

    assert index_file is not None

    if SAFE_WEIGHTS_INDEX_NAME in index_file:
        if index_file.startswith("oss://"):
            weight_map = json.loads(oss.get_object(index_file).read())["weight_map"]
        else:
            with open(index_file, "r") as f:
                weight_map = json.load(f)["weight_map"]
    else:
        raise NotImplementedError

    for checkpoint_file in tqdm(
        set(weight_map[key] for key in state_dict.keys() if key in weight_map),
        desc="Loading checkpoint shards",
    ):
        _load_checkpoint_file(
            pretrained_model_name_or_path,
            filename=checkpoint_file,
            expert_param_tags=expert_param_tags,
            state_dict=state_dict,
            missing_keys=missing_keys,
        )

    return list(missing_keys)


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


def _load_pretrained_weights(
    model: PreTrainedModel,
    state_dict: Dict[str, torch.Tensor],
    pretrained_model_name_or_path: str,
):
    from ..utils.expert_parallel import BaseMoELayer

    expert_keys = set()
    for module_name, module in model.named_modules():
        if isinstance(module, BaseMoELayer):
            for param_name, _ in module.named_parameters():
                expert_keys.add(f"{module_name}.{param_name}")

    if mpu.get_expert_data_parallel_rank() == 0:
        expert_param_tags = {}
        for i, key in enumerate(sorted(expert_keys)):
            expert_param_tags[key] = i

        if pretrained_model_name_or_path.startswith("oss://"):
            original_config = oss.load_config(pretrained_model_name_or_path)
        else:
            original_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        tie_word_embeddings = original_config.tie_word_embeddings

        head_key = "lm_head.weight"
        # TODO: handle embedding keys for general models
        embedding_key = "model.language_model.embed_tokens.weight"

        if mpu.get_data_parallel_rank() == 0 and tie_word_embeddings and head_key in state_dict and embedding_key not in state_dict:
            state_dict[embedding_key] = state_dict[head_key].clone()

        missing_keys = _load_checkpoint_files(
            pretrained_model_name_or_path,
            expert_param_tags=expert_param_tags,
            state_dict=state_dict,
        )

        state_dict_args = {}
        if "convert" in inspect.signature(model.state_dict).parameters:
            state_dict_args["convert"] = True
        original_state_dict = model.state_dict(**state_dict_args)

        if mpu.get_data_parallel_rank() == 0 and tie_word_embeddings and head_key in original_state_dict:
            assert embedding_key not in missing_keys
            assert head_key in missing_keys
            missing_keys.remove(head_key)
            state_dict[head_key].copy_(state_dict[embedding_key])
            if embedding_key not in original_state_dict:
                state_dict.pop(embedding_key)

            logger.info(
                f"Loaded checkpoint from '{pretrained_model_name_or_path}', missing keys: {missing_keys}"
            )


def init_weights(
    model: PreTrainedModel,
    pretrained_model_name_or_path: Optional[str] = None,
):
    from ..utils.expert_parallel import BaseMoELayer

    state_dict_args = {}
    if "convert" in inspect.signature(model.state_dict).parameters:
        state_dict_args["convert"] = True
    original_state_dict = model.state_dict(**state_dict_args)

    # Ensuring continuous memory allocation
    state_dict = {
        key: torch.empty_like(tensor, memory_format=torch.contiguous_format, device="cuda")
        for key, tensor in original_state_dict.items()
    }

    if pretrained_model_name_or_path is not None:
        _load_pretrained_weights(
            model,
            state_dict=state_dict,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
        )

    state_dict_args = {"strict": True, "assign": True}
    if "convert" in inspect.signature(model.load_state_dict).parameters:
        state_dict_args["convert"] = True
    model.load_state_dict(state_dict, **state_dict_args)

    model.to("cuda")
    model.tie_weights()

    for module in model.modules():
        if isinstance(module, BaseMoELayer):
            module.mark_moe_parameters()

    torch.distributed.barrier()

    return model


def build_model(
    model_type: str,
    model_path: str,
    dtype: torch.dtype,
    attn_implementation: str,
    vision_encoder_path: Optional[str] = None,
    reduced_layers_in_stage_zero: int = 0,
):
    if model_path.startswith("oss://"):
        original_config = oss.load_config(model_path)
    else:
        original_config = AutoConfig.from_pretrained(model_path)

    if type(original_config) not in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING:
        assert model_type is not None

    tie_word_embeddings = original_config.tie_word_embeddings
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        tie_word_embeddings = False
    original_config.tie_word_embeddings = tie_word_embeddings
    original_config.get_text_config().tie_word_embeddings = tie_word_embeddings

    module_dir = os.path.join(os.path.dirname(__file__), model_type)
    assert os.path.isdir(module_dir)
    module = importlib.import_module(f".{model_type}", package=__package__)
    assert hasattr(module, "apply_monkey_patch")
    logger.info(f"Apply monkey patch for `{model_type}` using {module.apply_monkey_patch}")
    module.apply_monkey_patch()

    if model_path.startswith("oss://"):
        processor = oss.load_processor(model_path)
    else:
        processor = AutoProcessor.from_pretrained(model_path)
    with _init_empty_params():
        model = AutoModelForImageTextToText.from_config(
            config=original_config,
            dtype=dtype,
            attn_implementation=attn_implementation,
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
    ep_rank = mpu.get_expert_model_parallel_rank()

    if ep_world_size > 1:
        assert hasattr(model, "apply_expert_parallel")
        model.apply_expert_parallel(
            ep_world_size=ep_world_size,
            ep_rank=ep_rank,
        )

    return model, processor
