from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Shard

from .. import parallel_state as mpu
from ..constants import MOE_DISPATCH_BACKEND
from ..ops import all_to_all, deepep_combine, deepep_dispatch, moe_token_permute, moe_token_unpermute


class MoETokenDispatcher(object):
    def __init__(
        self,
        num_experts: int,
        num_experts_per_token: int,
        backend: str = MOE_DISPATCH_BACKEND,
    ):
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token

        self.backend = backend.lower()
        assert self.backend in ["all_to_all", "deep_ep"]

        self.ep_group = mpu.get_expert_model_parallel_group()
        self.ep_world_size = mpu.get_expert_model_parallel_world_size()
        self.ep_rank = mpu.get_expert_model_parallel_rank()

        assert self.num_experts % self.ep_world_size == 0

        self.num_local_experts = num_experts // self.ep_world_size
        self.local_expert_slice = slice(
            self.ep_rank * self.num_local_experts, (self.ep_rank + 1) * self.num_local_experts
        )

    def _dispatch_a2a(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        routing_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int]]:
        assert hidden_states.ndim == 2
        assert expert_indices.ndim == 2

        num_tokens = torch.bincount(expert_indices.flatten(), minlength=self.num_experts)

        hidden_states, self.permuted_indices = moe_token_permute(
            hidden_states,
            routed_expert_indices=expert_indices,
            num_routed_tokens=num_tokens.tolist(),
            num_experts_per_token=self.num_experts_per_token,
        )

        if self.ep_group is not None:
            input_split_sizes = num_tokens.view(self.ep_world_size, -1).sum(1).tolist()

            global_num_tokens = [torch.zeros_like(num_tokens) for _ in range(self.ep_world_size)]
            torch.distributed.all_gather(
                global_num_tokens,
                num_tokens,
                group=self.ep_group,
            )
            global_num_tokens = torch.stack(global_num_tokens, dim=0)[:, self.local_expert_slice]
            output_split_sizes = global_num_tokens.sum(dim=1).tolist()
            num_tokens_list = global_num_tokens.sum(dim=0).tolist()

            hidden_states = all_to_all(
                hidden_states,
                output_split_sizes,
                input_split_sizes,
                group=self.ep_group,
            )

            global_token_indices = torch.arange(hidden_states.size(0), dtype=torch.long, device=hidden_states.device)
            splits = global_token_indices.split(global_num_tokens.flatten().tolist())
            global_token_indices = torch.cat(
                [
                    splits[i + j * self.num_local_experts]
                    for i in range(self.num_local_experts)
                    for j in range(self.ep_world_size)
                ]
            )
            hidden_states = hidden_states[global_token_indices]

            self.output_split_sizes = output_split_sizes
            self.input_split_sizes = input_split_sizes
            self.global_token_indices = global_token_indices

        else:
            num_tokens_list = num_tokens.tolist()

        self.routing_scores = routing_scores

        return hidden_states, num_tokens_list

    def _combine_a2a(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.ep_group is not None:
            hidden_states = torch.zeros_like(hidden_states).index_copy_(0, self.global_token_indices, hidden_states)
            hidden_states = all_to_all(
                hidden_states,
                self.input_split_sizes,
                self.output_split_sizes,
                group=self.ep_group,
            )

        outputs = moe_token_unpermute(
            hidden_states,
            probs=self.routing_scores,
            permuted_indices=self.permuted_indices,
            num_experts_per_token=self.num_experts_per_token,
        )

        self.permuted_indices = None
        self.global_token_indices = None
        self.routing_scores = None

        self.output_split_sizes = None
        self.input_split_sizes = None

        return outputs

    def _dispatch_deepep(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        routing_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int]]:
        if self.ep_world_size > 1:
            (
                hidden_states,
                expert_indices,
                routing_scores,
                num_tokens_list,
                handle,
            ) = deepep_dispatch(
                hidden_states,
                topk_idx=expert_indices,
                topk_weights=routing_scores,
                num_experts=self.num_experts,
                group=self.ep_group,
            )
            self.handle = handle
        else:
            num_tokens_list = torch.bincount(expert_indices.flatten(), minlength=self.num_experts).tolist()

        hidden_states, permuted_indices = moe_token_permute(
            hidden_states,
            routed_expert_indices=expert_indices,
            num_routed_tokens=num_tokens_list,
            num_experts_per_token=self.num_experts_per_token,
        )

        self.routing_scores = routing_scores
        self.permuted_indices = permuted_indices

        return hidden_states, num_tokens_list

    def _combine_deepep(self, hidden_states: torch.Tensor) -> torch.Tensor:
        outputs = moe_token_unpermute(
            hidden_states,
            probs=self.routing_scores,
            permuted_indices=self.permuted_indices,
            num_experts_per_token=self.num_experts_per_token,
        )

        if self.ep_world_size > 1:
            outputs = deepep_combine(
                outputs,
                self.handle,
                self.ep_group,
            )

        self.routing_scores = None
        self.permuted_indices = None

        return outputs

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        routing_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int]]:
        if self.backend == "all_to_all":
            return self._dispatch_a2a(
                hidden_states=hidden_states,
                expert_indices=expert_indices,
                routing_scores=routing_scores,
            )
        elif self.backend == "deep_ep":
            return self._dispatch_deepep(
                hidden_states=hidden_states,
                expert_indices=expert_indices,
                routing_scores=routing_scores,
            )
        else:
            raise ValueError

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.backend == "all_to_all":
            return self._combine_a2a(hidden_states=hidden_states)
        elif self.backend == "deep_ep":
            return self._combine_deepep(hidden_states=hidden_states)
        else:
            raise ValueError


def fully_shard_experts(
    experts: nn.Module,
    *,
    edp_mesh: DeviceMesh,
    ep_degree: int,
    mp_policy: MixedPrecisionPolicy,
    reshard_after_forward: bool = False,
):
    shard_placement_fn = None
    if edp_mesh.size() * ep_degree > experts.num_experts:
        shard_placement_fn = lambda param: Shard(1)  # noqa: E731

    fully_shard(
        experts,
        mesh=edp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
        shard_placement_fn=shard_placement_fn,
    )

    # Align experts' grad divisor with the non-expert FSDP modules: their mesh
    # is ``dp_mesh`` (= ``edp_mesh`` x ``ep``), so without this override
    # experts would be off by a factor of ``ep_degree``.
    experts.set_gradient_divide_factor(edp_mesh.size() * ep_degree)


def set_moe_fsdp_prefetch(
    layers: Sequence[nn.Module],
    experts_modules: Sequence[Optional[nn.Module]] = (),
    *,
    pre_module: Optional[nn.Module] = None,
    post_modules: Sequence[nn.Module] = (),
) -> None:
    layers = list(layers)
    if not layers:
        return

    if not experts_modules:
        experts_modules = [None] * len(layers)
    assert len(experts_modules) == len(layers), (
        f"experts_modules has {len(experts_modules)} entries but layers has {len(layers)}"
    )

    if pre_module is not None:
        pre_module.set_modules_to_forward_prefetch([layers[0]])

    for i, layer in enumerate(layers):
        if i + 1 < len(layers):
            modules = [layers[i + 1]]
            if experts_modules[i + 1] is not None:
                modules.append(experts_modules[i + 1])
            layer.set_modules_to_forward_prefetch(modules)
        elif post_modules:
            layer.set_modules_to_forward_prefetch(list(post_modules))

    if post_modules:
        post_modules[-1].set_modules_to_backward_prefetch([layers[-1]])

    for i in range(len(layers) - 1, -1, -1):
        layer = layers[i]
        if i - 1 >= 0:
            modules = [layers[i - 1]]
            if experts_modules[i - 1] is not None:
                modules.append(experts_modules[i - 1])
            layer.set_modules_to_backward_prefetch(modules)
        elif pre_module is not None:
            layer.set_modules_to_backward_prefetch([pre_module])
