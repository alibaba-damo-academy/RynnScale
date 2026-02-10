from typing import Optional

import torch
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh


__all__ = [
    "initialize_model_parallel",
    "get_device_mesh",
    "get_data_parallel_group",
    "get_expert_model_parallel_group",
]


_DEVICE_MESH = None
_EP_DEVICE_MESH = None

_DATA_PARALLEL_GROUP = None
_PIPELINE_MODEL_PARALLEL_GROUP = None
_EXPERT_MODEL_PARALLEL_GROUP = None
_EXPERT_DATA_PARALLEL_GROUP = None
_ENCODER_CONTEXT_PARALLEL_GROUP = None


def initialize_model_parallel(
    pipeline_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    encoder_context_parallel_size: int = 1,
) -> None:
    global _DEVICE_MESH, _EP_DEVICE_MESH

    if _DEVICE_MESH is not None or _EP_DEVICE_MESH is not None:
        raise ValueError

    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()

    assert world_size % (expert_model_parallel_size * pipeline_model_parallel_size) == 0

    data_parallel_size = world_size // pipeline_model_parallel_size

    _DEVICE_MESH = init_device_mesh(
        "cuda",
        (pipeline_model_parallel_size, data_parallel_size),
        mesh_dim_names=("pp", "dp"),
    )

    _EP_DEVICE_MESH = init_device_mesh(
        "cuda",
        (pipeline_model_parallel_size, data_parallel_size // expert_model_parallel_size, expert_model_parallel_size),
        mesh_dim_names=("pp", "edp", "ep"),
    )

    _ENCODER_CP_DEVICE_MESH = init_device_mesh(
        "cuda",
        (
            pipeline_model_parallel_size,
            data_parallel_size // encoder_context_parallel_size,
            encoder_context_parallel_size,
        ),
        mesh_dim_names=("pp", "cdp", "cp"),
    )

    global \
        _DATA_PARALLEL_GROUP, \
        _PIPELINE_MODEL_PARALLEL_GROUP, \
        _EXPERT_MODEL_PARALLEL_GROUP, \
        _EXPERT_DATA_PARALLEL_GROUP
    global _ENCODER_CONTEXT_PARALLEL_GROUP

    _DATA_PARALLEL_GROUP = _DEVICE_MESH["dp"].get_group()
    _PIPELINE_MODEL_PARALLEL_GROUP = _DEVICE_MESH["pp"].get_group()
    _EXPERT_MODEL_PARALLEL_GROUP = _EP_DEVICE_MESH["ep"].get_group()
    _EXPERT_DATA_PARALLEL_GROUP = _EP_DEVICE_MESH["edp"].get_group()
    _ENCODER_CONTEXT_PARALLEL_GROUP = _ENCODER_CP_DEVICE_MESH["cp"].get_group()


def get_device_mesh(with_ep: bool = False) -> Optional[DeviceMesh]:
    device_mesh = _EP_DEVICE_MESH if with_ep else _DEVICE_MESH
    if device_mesh is None:
        raise ValueError
    return device_mesh


def get_data_parallel_group() -> Optional[torch.distributed.ProcessGroup]:
    assert _DATA_PARALLEL_GROUP is not None
    return _DATA_PARALLEL_GROUP


def get_data_parallel_world_size() -> int:
    return torch.distributed.get_world_size(get_data_parallel_group())


def get_data_parallel_rank() -> int:
    return torch.distributed.get_rank(get_data_parallel_group())


def get_pipeline_model_parallel_group() -> torch.distributed.ProcessGroup:
    assert _PIPELINE_MODEL_PARALLEL_GROUP is not None
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_world_size() -> int:
    return torch.distributed.get_world_size(get_pipeline_model_parallel_group())


def get_pipeline_model_parallel_rank() -> int:
    return torch.distributed.get_rank(get_pipeline_model_parallel_group())


def get_expert_model_parallel_group() -> torch.distributed.ProcessGroup:
    assert _EXPERT_MODEL_PARALLEL_GROUP is not None
    return _EXPERT_MODEL_PARALLEL_GROUP


def get_expert_model_parallel_world_size() -> int:
    return torch.distributed.get_world_size(get_expert_model_parallel_group())


def get_expert_model_parallel_rank() -> int:
    return torch.distributed.get_rank(get_expert_model_parallel_group())


def get_expert_data_parallel_group() -> torch.distributed.ProcessGroup:
    assert _EXPERT_DATA_PARALLEL_GROUP is not None
    return _EXPERT_DATA_PARALLEL_GROUP


def get_expert_data_parallel_world_size() -> int:
    return torch.distributed.get_world_size(get_expert_data_parallel_group())


def get_expert_data_parallel_rank() -> int:
    return torch.distributed.get_rank(get_expert_data_parallel_group())


def get_encoder_context_parallel_group() -> torch.distributed.ProcessGroup:
    assert _ENCODER_CONTEXT_PARALLEL_GROUP is not None
    return _ENCODER_CONTEXT_PARALLEL_GROUP


def get_encoder_context_parallel_world_size() -> int:
    return torch.distributed.get_world_size(get_encoder_context_parallel_group())


def get_encoder_context_parallel_rank() -> int:
    return torch.distributed.get_rank(get_encoder_context_parallel_group())
