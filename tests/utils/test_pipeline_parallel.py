import pytest
import torch

from rynn_scale import parallel_state as mpu
from rynn_scale.utils.pipeline_parallel import (
    gather_pp_params,
)


@pytest.mark.distributed(world_size=4)
def test_gather_pp_params(
    dtype: torch.dtype = torch.bfloat16,
):
    mpu.initialize_model_parallel(
        pipeline_model_parallel_size=torch.distributed.get_world_size(),
    )

    pp_size = mpu.get_pipeline_model_parallel_world_size()
    pp_rank = mpu.get_pipeline_model_parallel_rank()

    original_state_dict = {
        f"{j}.{i}": torch.arange(i + 1, dtype=dtype, device="cuda") * j for i in range(5) for j in range(pp_size)
    }
    state_dict = {k: v for k, v in original_state_dict.items() if k.startswith(f"{pp_rank}.")}
    state_dict = gather_pp_params(state_dict)

    if pp_rank == 0:
        assert state_dict.keys() == original_state_dict.keys()
        for key in state_dict:
            assert torch.all(state_dict[key].cpu() == original_state_dict[key].cpu())
