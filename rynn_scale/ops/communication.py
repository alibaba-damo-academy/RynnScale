from typing import Optional, List

import torch


__all__ = ["all_to_all"]


class AllToAllFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        output_split_sizes: Optional[List[int]] = None,
        input_split_sizes: Optional[List[int]] = None,
        group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> torch.Tensor:
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group

        world_size = torch.distributed.get_world_size(group)
        if output_split_sizes is None:
            assert input_tensor.size(0) % world_size == 0
            output_split_sizes = [input_tensor.size(0) // world_size] * world_size
        if input_split_sizes is None:
            assert input_tensor.size(0) % world_size == 0
            input_split_sizes = [input_tensor.size(0) // world_size] * world_size

        output_tensor = input_tensor.new_zeros((sum(output_split_sizes), *input_tensor.shape[1:]))

        torch.distributed.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )

        return output_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = grad_output.new_zeros((sum(ctx.input_split_sizes), *grad_output.shape[1:]))

        torch.distributed.all_to_all_single(
            grad_input,
            grad_output,
            output_split_sizes=ctx.input_split_sizes,
            input_split_sizes=ctx.output_split_sizes,
            group=ctx.group,
        )

        return grad_input, None, None, None


def all_to_all(
    input_tensor: torch.Tensor,
    output_split_sizes: Optional[List[int]] = None,
    input_split_sizes: Optional[List[int]] = None,
    group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    return AllToAllFunction.apply(input_tensor, output_split_sizes, input_split_sizes, group)
