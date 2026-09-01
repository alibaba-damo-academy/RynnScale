from typing import Optional

import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

__all__ = [
    "initialize_model_parallel",
    "get_device_mesh",
    "get_data_parallel_group",
    "get_expert_model_parallel_group",
]


_DEVICE_MESH = None
_EXPERT_DEVICE_MESH = None

_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_WITH_CP = None
_PIPELINE_MODEL_PARALLEL_GROUP = None
_CONTEXT_PARALLEL_GROUP = None
_EXPERT_MODEL_PARALLEL_GROUP = None
_EXPERT_DATA_PARALLEL_GROUP = None
_ENCODER_CONTEXT_PARALLEL_GROUP = None

_EMBEDDING_GROUP = None


def initialize_model_parallel(
    pipeline_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    encoder_context_parallel_size: int = 1,
) -> None:
    assert encoder_context_parallel_size >= context_parallel_size

    global _DEVICE_MESH, _EXPERT_DEVICE_MESH

    if _DEVICE_MESH is not None or _EXPERT_DEVICE_MESH is not None:
        raise ValueError

    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    _DEVICE_MESH = init_device_mesh(
        "cuda",
        (pipeline_model_parallel_size, world_size // pipeline_model_parallel_size),
        mesh_dim_names=("pp", "dp"),
    )

    _EXPERT_DEVICE_MESH = init_device_mesh(
        "cuda",
        (
            pipeline_model_parallel_size,
            world_size // pipeline_model_parallel_size // expert_model_parallel_size,
            expert_model_parallel_size,
        ),
        mesh_dim_names=("pp", "dp", "ep"),
    )

    _DEVICE_MESH_WITH_CP = init_device_mesh(
        "cuda",
        (
            pipeline_model_parallel_size,
            world_size // pipeline_model_parallel_size // context_parallel_size,
            context_parallel_size,
        ),
        mesh_dim_names=("pp", "dp", "cp"),
    )

    _DEVICE_MESH_WITH_ENCODER_CP = init_device_mesh(
        "cuda",
        (
            pipeline_model_parallel_size,
            world_size // pipeline_model_parallel_size // encoder_context_parallel_size,
            encoder_context_parallel_size,
        ),
        mesh_dim_names=("pp", "dp", "cp"),
    )

    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _EXPERT_MODEL_PARALLEL_GROUP, _EXPERT_DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP, _DATA_PARALLEL_GROUP_WITH_CP, _CONTEXT_PARALLEL_GROUP
    global _ENCODER_CONTEXT_PARALLEL_GROUP
    global _EMBEDDING_GROUP

    _PIPELINE_MODEL_PARALLEL_GROUP = _DEVICE_MESH["pp"].get_group()
    _EXPERT_MODEL_PARALLEL_GROUP = _EXPERT_DEVICE_MESH["ep"].get_group()
    _EXPERT_DATA_PARALLEL_GROUP = _EXPERT_DEVICE_MESH["dp"].get_group()
    _DATA_PARALLEL_GROUP = _DEVICE_MESH_WITH_CP["dp"].get_group()
    _DATA_PARALLEL_GROUP_WITH_CP = _DEVICE_MESH["dp"].get_group()
    _CONTEXT_PARALLEL_GROUP = _DEVICE_MESH_WITH_CP["cp"].get_group()
    _ENCODER_CONTEXT_PARALLEL_GROUP = _DEVICE_MESH_WITH_ENCODER_CP["cp"].get_group()

    if pipeline_model_parallel_size > 1:
        global_ranks = torch.arange(world_size)
        pp_groups = global_ranks.view(pipeline_model_parallel_size, -1).transpose(0, 1)
        for pp_group_ranks in pp_groups:
            embedding_ranks = pp_group_ranks[[0, -1]].tolist()
            embedding_group = torch.distributed.new_group(ranks=embedding_ranks)
            if rank in embedding_ranks:
                _EMBEDDING_GROUP = embedding_group


def get_device_mesh() -> DeviceMesh:
    assert _DEVICE_MESH is not None
    return _DEVICE_MESH


def get_expert_device_mesh() -> DeviceMesh:
    assert _EXPERT_DEVICE_MESH is not None
    return _EXPERT_DEVICE_MESH


def get_data_parallel_group(with_context_parallel: bool = False) -> Optional[torch.distributed.ProcessGroup]:
    if with_context_parallel:
        assert _DATA_PARALLEL_GROUP_WITH_CP is not None
        return _DATA_PARALLEL_GROUP_WITH_CP
    assert _DATA_PARALLEL_GROUP is not None
    return _DATA_PARALLEL_GROUP


def get_data_parallel_world_size(with_context_parallel: bool = False) -> int:
    return torch.distributed.get_world_size(get_data_parallel_group(with_context_parallel=with_context_parallel))


def get_data_parallel_rank(with_context_parallel: bool = False) -> int:
    return torch.distributed.get_rank(get_data_parallel_group(with_context_parallel=with_context_parallel))


def get_pipeline_model_parallel_group() -> torch.distributed.ProcessGroup:
    assert _PIPELINE_MODEL_PARALLEL_GROUP is not None
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_world_size() -> int:
    return torch.distributed.get_world_size(get_pipeline_model_parallel_group())


def get_pipeline_model_parallel_rank() -> int:
    return torch.distributed.get_rank(get_pipeline_model_parallel_group())


def get_expert_model_parallel_group() -> Optional[torch.distributed.ProcessGroup]:
    if not torch.distributed.is_initialized():
        return None
    assert _EXPERT_MODEL_PARALLEL_GROUP is not None
    return _EXPERT_MODEL_PARALLEL_GROUP


def get_expert_model_parallel_world_size() -> int:
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(get_expert_model_parallel_group())


def get_expert_model_parallel_rank() -> int:
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank(get_expert_model_parallel_group())


def get_expert_data_parallel_group() -> torch.distributed.ProcessGroup:
    assert _EXPERT_DATA_PARALLEL_GROUP is not None
    return _EXPERT_DATA_PARALLEL_GROUP


def get_expert_data_parallel_world_size() -> int:
    return torch.distributed.get_world_size(get_expert_data_parallel_group())


def get_expert_data_parallel_rank() -> int:
    return torch.distributed.get_rank(get_expert_data_parallel_group())


def get_encoder_context_parallel_group() -> Optional[torch.distributed.ProcessGroup]:
    if not torch.distributed.is_initialized():
        return None
    assert _ENCODER_CONTEXT_PARALLEL_GROUP is not None
    return _ENCODER_CONTEXT_PARALLEL_GROUP


def get_encoder_context_parallel_world_size() -> int:
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(get_encoder_context_parallel_group())


def get_encoder_context_parallel_rank() -> int:
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank(get_encoder_context_parallel_group())


def get_context_parallel_group() -> Optional[torch.distributed.ProcessGroup]:
    if not torch.distributed.is_initialized():
        return None
    assert _CONTEXT_PARALLEL_GROUP is not None
    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_world_size() -> int:
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(get_context_parallel_group())


def get_context_parallel_rank() -> int:
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank(get_context_parallel_group())


def get_embedding_group() -> torch.distributed.ProcessGroup:
    assert _EMBEDDING_GROUP is not None
    return _EMBEDDING_GROUP
