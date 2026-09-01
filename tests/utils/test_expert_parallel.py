import pytest
import torch
import torch.nn as nn

from rynn_scale import parallel_state as mpu
from rynn_scale.utils.expert_parallel import (
    BaseMoELayer,
    gather_ep_params,
)


class FakeMoELayer(BaseMoELayer):
    def __init__(
        self,
        hidden_size: int = 16,
        moe_intermediate_size: int = 8,
        num_experts: int = 8,
    ):
        super().__init__(num_experts)
        expert_indices = torch.arange(num_experts, dtype=torch.float)
        self.up_proj = nn.Parameter(expert_indices[:, None, None].repeat(1, moe_intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(expert_indices[:, None, None].repeat(1, hidden_size, moe_intermediate_size) * 2.0)


class FakeMoEModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 16,
        moe_intermediate_size: int = 8,
        num_experts: int = 8,
    ):
        super().__init__()
        self.moe_layer = FakeMoELayer(
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            num_experts=num_experts,
        )
        self.dense_layer = nn.Linear(hidden_size, hidden_size)

    @torch.no_grad()
    def apply_expert_parallel(self):
        ep_size = mpu.get_expert_model_parallel_world_size()
        ep_rank = mpu.get_expert_model_parallel_rank()
        for name, param in self.moe_layer.named_parameters():
            self.moe_layer.register_parameter(name, torch.nn.Parameter(param.chunk(ep_size)[ep_rank]))


@pytest.mark.distributed(world_size=4)
def test_gather_ep_params(
    num_experts: int = 4,
    in_dim: int = 16,
    out_dim: int = 8,
    dtype: torch.dtype = torch.bfloat16,
):
    mpu.initialize_model_parallel(
        expert_model_parallel_size=torch.distributed.get_world_size(),
    )

    ep_size = mpu.get_expert_model_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank()
    assert num_experts % ep_size == 0

    model = FakeMoEModel(
        hidden_size=in_dim,
        moe_intermediate_size=out_dim,
        num_experts=num_experts,
    )
    model.to(dtype=dtype, device="cuda")
    original_state_dict = {k: v for k, v in model.state_dict().items()}

    model.apply_expert_parallel()
    state_dict = gather_ep_params(model)

    assert state_dict.keys() == original_state_dict.keys()
    if ep_rank == 0:
        for key in state_dict:
            assert torch.all(state_dict[key].cpu() == original_state_dict[key].cpu())
