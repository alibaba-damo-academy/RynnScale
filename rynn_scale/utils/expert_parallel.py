import inspect
import os
from typing import Any, List, Tuple

import torch
from deepspeed.moe.layer import MoE
from deepspeed.utils import groups

from .. import parallel_state as mpu
from ..ops import all_to_all, deepep_combine, deepep_dispatch, moe_token_permute, moe_token_unpermute

MOE_DISPATCH_BACKEND = os.environ.get("MOE_DISPATCH_BACKEND", "deep_ep")


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
        expert_indices = expert_indices.flatten()

        self.token_indices = torch.argsort(expert_indices, stable=True)
        hidden_states = hidden_states.index_select(0, self.token_indices // self.num_experts_per_token)

        num_tokens = torch.bincount(expert_indices, minlength=self.num_experts)

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
        hidden_size = hidden_states.size(-1)

        if self.ep_group is not None:
            hidden_states = torch.zeros_like(hidden_states).index_copy_(0, self.global_token_indices, hidden_states)
            hidden_states = all_to_all(
                hidden_states,
                self.input_split_sizes,
                self.output_split_sizes,
                group=self.ep_group,
            )

        outputs = hidden_states.new_zeros((self.token_indices.numel(), hidden_size))
        outputs.index_copy_(0, self.token_indices, hidden_states)
        outputs = outputs.view(-1, self.num_experts_per_token, hidden_size)

        outputs = outputs * self.routing_scores.unsqueeze(-1)
        outputs = outputs.sum(dim=1)

        self.token_indices = None
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


class BaseMoELayer(MoE):
    def __init__(self, num_experts: int):
        super(MoE, self).__init__()

        self.ep_group = mpu.get_expert_model_parallel_group()
        self.ep_world_size = mpu.get_expert_model_parallel_world_size()
        self.ep_rank = mpu.get_expert_model_parallel_rank()

        assert num_experts % self.ep_world_size == 0

        # To ensure compatibility with DeepSpeed
        self.num_experts = num_experts
        self.expert_group_name = f"ep_size_{self.ep_world_size}"
        self.enable_expert_tensor_parallelism = False

    def set_deepspeed_parallelism(self, use_data_before_expert_parallel_: bool = False) -> None:
        if use_data_before_expert_parallel_:
            raise NotImplementedError()

        # https://github.com/deepspeedai/DeepSpeed/blob/fffcf2f56925c00a34d74fec0be523a3aec479a8/deepspeed/moe/layer.py#L91
        if self.expert_group_name not in groups._get_expert_parallel_group_dict():
            print(f"No existing process group found, creating a new group named: {self.expert_group_name}")
            if (groups.mpu is None) or (not self.enable_expert_tensor_parallelism):
                # Condition 1 - no groups.mpu means no tensor parallelism
                # Condition 2 - disabling expert tensor parallelism on purpose
                groups._create_expert_and_data_parallel(
                    self.ep_world_size, use_data_before_expert_parallel_=use_data_before_expert_parallel_
                )
            else:
                # expert tensor parallelism is enabled
                groups._create_expert_data_and_model_parallel(
                    self.ep_world_size,
                    mpu=groups.mpu,
                    use_data_before_expert_parallel_=use_data_before_expert_parallel_,
                )

    def mark_moe_parameters(self):
        for p in self.parameters():
            p.allreduce = False
            p.group_name = self.expert_group_name

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        self.mark_moe_parameters()

    def _apply(self, *args, **kwargs):
        output = super()._apply(*args, **kwargs)
        self.mark_moe_parameters()
        return output


def gather_ep_params(
    model: torch.nn.Module,
    convert: bool = True,
):
    if "convert" in inspect.signature(model.state_dict).parameters:
        state_dict = model.state_dict(convert=convert)
    else:
        state_dict = model.state_dict()

    if mpu.get_expert_data_parallel_rank() != 0:
        torch.distributed.barrier()
        return state_dict

    ep_group = mpu.get_expert_model_parallel_group()
    ep_size = mpu.get_expert_model_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank()

    expert_param_names = set()
    for module_name, module in model.named_modules():
        if isinstance(module, BaseMoELayer):
            for key in state_dict:
                if key.startswith(module_name):
                    expert_param_names.add(key)

    for param_name in sorted(state_dict.keys()):
        if param_name not in expert_param_names:
            continue

        param = state_dict[param_name].cuda()
        if ep_rank == 0:
            outputs = [torch.empty_like(param) for _ in range(ep_size)]
        else:
            outputs = None

        torch.distributed.gather(
            param,
            outputs,
            group=ep_group,
            group_dst=0,
        )

        if ep_rank == 0:
            new_param = torch.cat(outputs, dim=0)
            state_dict[param_name] = new_param.cpu()

    torch.distributed.barrier()
    return state_dict
