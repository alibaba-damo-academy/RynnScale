import os
import importlib
import inspect
from contextlib import contextmanager
from collections import defaultdict
from typing import Dict, Set, Optional

import json
import torch
from safetensors import safe_open
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoImageProcessor,
    AutoVideoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    CONFIG_MAPPING,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
    PROCESSOR_MAPPING,
)
from transformers.utils import (
    cached_file,
    WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
)

from ..utils import logging
from .. import parallel_state as mpu


logger = logging.get_logger(__name__)


def _recv_expert_params(
    expert_param_tags: Dict[str, int],
    state_dict: Dict[str, torch.Tensor],
):
    ep_group = mpu.get_expert_model_parallel_group()
    ep_world_size = mpu.get_expert_model_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank()

    if ep_world_size == 1:
        return

    recv_ops = []

    for key, tag in expert_param_tags.items():
        logger.debug(
            f"Rank {ep_rank} (global rank {torch.distributed.get_rank()}), recv {key}({state_dict[key].shape}) from rank 0"
        )
        recv_ops.append(
            torch.distributed.P2POp(
                torch.distributed.irecv,
                state_dict[key],
                group=ep_group,
                tag=tag,
                group_peer=0,
            )
        )

    works = torch.distributed.batch_isend_irecv(recv_ops)
    for work in works:
        work.wait()


def _send_expert_params(
    expert_param_tags: Dict[str, int],
    state_dict: Dict[str, torch.Tensor],
):
    ep_group = mpu.get_expert_model_parallel_group()
    ep_world_size = mpu.get_expert_model_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank()

    if ep_world_size == 1:
        return

    send_ops = []

    for key, tag in expert_param_tags.items():
        if key not in state_dict:
            continue
        shared_tensors = state_dict[key].cuda().chunk(ep_world_size, dim=0)
        for dst_rank in range(1, ep_world_size):
            logger.debug(
                f"Rank {ep_rank} (global rank {torch.distributed.get_rank()}), send {key}({shared_tensors[dst_rank].shape}) to rank {dst_rank}"
            )
            send_ops.append(
                torch.distributed.P2POp(
                    torch.distributed.isend,
                    shared_tensors[dst_rank],
                    group=ep_group,
                    tag=tag,
                    group_peer=dst_rank,
                )
            )
        state_dict[key] = shared_tensors[0]

    works = torch.distributed.batch_isend_irecv(send_ops)
    for work in works:
        work.wait()


def _get_local_path(
    pretrained_model_name_or_path: str,
    filename: str,
    _raise_exceptions_for_gated_repo: bool = True,
    _raise_exceptions_for_missing_entries: bool = True,
):
    local_path = os.path.join(pretrained_model_name_or_path, filename)
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
    keys: Set[str],
    expert_param_tags: Dict[str, int],
) -> Dict[str, torch.Tensor]:
    local_checkpoint_file = _get_local_path(
        pretrained_model_name_or_path,
        filename=filename,
    )

    if filename.endswith(".safetensors"):
        state_dict = {}
        with safe_open(local_checkpoint_file, framework="pt", device="cpu") as f:
            for key in keys:
                state_dict[key] = f.get_tensor(key)
    else:
        sharded_state_dict = torch.load(local_checkpoint_file, map_location="cpu")
        state_dict = {key: sharded_state_dict[key] for key in keys}

    _send_expert_params(expert_param_tags, state_dict)
    return state_dict


def _load_checkpoint_files(
    pretrained_model_name_or_path: str,
    keys_to_load: Set[str],
    expert_param_tags: Dict[str, int],
) -> Dict[str, torch.Tensor]:
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
        return _load_checkpoint_file(
            pretrained_model_name_or_path,
            filename=checkpoint_name,
            keys=keys_to_load,
            expert_param_tags=expert_param_tags,
        )

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
        with open(index_file, "r") as f:
            weight_map = json.load(f)["weight_map"]
    else:
        raise NotImplementedError

    checkpoint_keys_map = defaultdict(set)
    for key in keys_to_load:
        checkpoint_keys_map[weight_map[key]].add(key)

    state_dict = {}
    for checkpoint_file, loaded_keys in tqdm(
        checkpoint_keys_map.items(),
        desc="Loading checkpoint shards",
    ):
        state_dict.update(
            _load_checkpoint_file(
                pretrained_model_name_or_path,
                filename=checkpoint_file,
                keys=loaded_keys,
                expert_param_tags=expert_param_tags,
            )
        )

    return state_dict


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
    pretrained_model_name_or_path: str,
):
    from ..utils.expert_parallel import BaseMoELayer

    expert_keys = set()
    for module_name, module in model.named_modules():
        if isinstance(module, BaseMoELayer):
            for param_name, _ in module.named_parameters():
                expert_keys.add(f"{module_name}.{param_name}")

    expert_param_tags = {}
    for i, key in enumerate(sorted(expert_keys)):
        expert_param_tags[key] = i

    state_dict_args = {}
    if "convert" in inspect.signature(model.state_dict).parameters:
        state_dict_args["convert"] = True

    tie_word_embeddings = model.config.tie_word_embeddings or model.config.get_text_config().tie_word_embeddings
    original_keys = set(model.state_dict(**state_dict_args).keys())
    keys_to_load = original_keys.copy()

    head_key = "lm_head.weight"
    # TODO: handle embedding keys for general models
    embedding_key = "model.language_model.embed_tokens.weight"

    if tie_word_embeddings and head_key in original_keys:
        keys_to_load.discard(head_key)
        keys_to_load.add(embedding_key)

    if mpu.get_data_parallel_rank() == 0:
        state_dict = _load_checkpoint_files(
            pretrained_model_name_or_path,
            keys_to_load=keys_to_load,
            expert_param_tags=expert_param_tags,
        )
        if tie_word_embeddings and head_key in original_keys:
            state_dict[head_key] = state_dict[embedding_key].clone()
            if embedding_key not in original_keys:
                state_dict.pop(embedding_key)
    else:
        state_dict = {
            key: torch.empty_like(tensor, device="cuda") for key, tensor in model.state_dict(**state_dict_args).items()
        }
        if mpu.get_expert_data_parallel_rank() == 0:
            _recv_expert_params(expert_param_tags, state_dict)

    state_dict_args = {"strict": True, "assign": True}
    if "convert" in inspect.signature(model.load_state_dict).parameters:
        state_dict_args["convert"] = True

    incompatible_keys = model.load_state_dict(state_dict, **state_dict_args)
    logger.info(
        f"Loaded checkpoint from '{pretrained_model_name_or_path}', "
        f"missing keys: {incompatible_keys.missing_keys}, "
        f"unexpected keys: {incompatible_keys.unexpected_keys}"
    )

    model.to("cuda")

    if tie_word_embeddings:
        model.tie_weights()

    for module in model.modules():
        if isinstance(module, BaseMoELayer):
            module.mark_moe_parameters()

    torch.distributed.barrier()

    return model


def _check_chat_template(processor):
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "Hello!"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        conversation, tokenize=False, chat_template=processor.chat_template
    )
    prompt_local = processor.apply_chat_template(conversation, tokenize=False)
    assert prompt == prompt_local, "Chat template in local implementation does not match the processor."


def build_model(
    model_type: str,
    model_path: str,
    dtype: torch.dtype,
    attn_implementation: str,
    vision_encoder_path: Optional[str] = None,
    reduced_layers_in_stage_zero: int = 0,
):
    original_config = AutoConfig.from_pretrained(model_path)
    if type(original_config) not in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING:
        assert model_type is not None

    module_dir = os.path.join(os.path.dirname(__file__), model_type)
    assert os.path.isdir(module_dir)
    module = importlib.import_module(f".{model_type}", package=__package__)
    assert hasattr(module, "apply_monkey_patch")
    logger.info(f"Apply monkey patch for `{model_type}` using {module.apply_monkey_patch}")
    module.apply_monkey_patch()

    if type(original_config) in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING:
        processor = AutoProcessor.from_pretrained(model_path)
        with _init_empty_params():
            model = AutoModelForImageTextToText.from_config(
                config=original_config,
                dtype=dtype,
                attn_implementation=attn_implementation,
            )

    else:
        assert model_type is not None, "Please specify `model_type` when init from a LLM checkpoint."
        assert vision_encoder_path is not None, "Please specify `vision_encoder_path` when init from a LLM checkpoint."

        config_class = CONFIG_MAPPING[model_type]
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        image_processor = AutoImageProcessor.from_pretrained(vision_encoder_path)
        video_processor = AutoVideoProcessor.from_pretrained(vision_encoder_path)
        processor = PROCESSOR_MAPPING[config_class].from_pretrained(
            tokenizer=tokenizer,
            image_processor=image_processor,
            video_processor=video_processor,
        )

        vision_config = AutoConfig.from_pretrained(vision_encoder_path)
        config = config_class(
            text_config=original_config,
            vision_config=vision_config,
            image_token_id=processor.image_token_id,
            video_token_id=processor.video_token_id,
        )

        with _init_empty_params():
            model = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING[config_class].from_config(
                config=config,
                dtype=dtype,
                attn_implementation=attn_implementation,
            )

        vision_model = AutoModel.from_pretrained(vision_encoder_path, dtype=dtype)
        model.model.vision_model.load_state_dict(vision_model.state_dict())
        del vision_model

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

    _check_chat_template(processor)
    _load_pretrained_weights(model, model_path)

    return model, processor
