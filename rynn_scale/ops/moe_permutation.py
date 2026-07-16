from typing import List

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_token_permute_preprocess(
    sorted_indices,
    permuted_indices,
    n,
    max_routed_experts: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets_n < n

    src_indices = tl.load(sorted_indices + offsets_n, mask=mask)
    src_token_indices = src_indices // max_routed_experts
    src_topk_indices = src_indices % max_routed_experts

    o_ptrs = permuted_indices + src_token_indices * max_routed_experts + src_topk_indices
    tl.store(o_ptrs, offsets_n, mask=mask)


_configs = [
    triton.Config({"BLOCK_K": 64}),
    triton.Config({"BLOCK_K": 128}),
    triton.Config({"BLOCK_K": 256}),
    triton.Config({"BLOCK_K": 512}),
    triton.Config({"BLOCK_K": 1024}),
    triton.Config({"BLOCK_K": 2048}),
    triton.Config({"BLOCK_K": 4096}),
]


def _prune_func(configs, named_args, **kwargs):
    configs = [c for c in configs if c.kwargs["BLOCK_K"] <= named_args["hidden_size"] and named_args["hidden_size"] % c.kwargs["BLOCK_K"] == 0]
    assert len(configs), f"no configs found for hidden_size={named_args['hidden_size']}"
    return configs


@triton.autotune(
    configs=_configs,
    key=["hidden_size"],
    prune_configs_by={
        "early_config_prune": _prune_func,
    },
)
@triton.jit
def _moe_token_permute_fwd_kernel(
    inputs,
    outputs,
    permuted_indices,
    stride_xm,
    stride_xk,
    stride_on,
    stride_ok,
    hidden_size: tl.constexpr,
    max_routed_experts: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offsets_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    x_ptrs = inputs + pid_m * stride_xm + offsets_k * stride_xk
    x = tl.load(x_ptrs)

    for i in tl.range(max_routed_experts):
        permuted_idx = tl.load(permuted_indices + pid_m * max_routed_experts + i).to(tl.int64)
        if permuted_idx >= 0:
            o_ptrs = outputs + permuted_idx * stride_on + offsets_k * stride_ok
            tl.store(o_ptrs, x)


@triton.autotune(
    configs=_configs,
    key=["hidden_size"],
    prune_configs_by={
        "early_config_prune": _prune_func,
    },
)
@triton.jit
def _moe_token_permute_bwd_kernel(
    grad_outputs,
    grad_inputs,
    permuted_indices,
    stride_don,
    stride_dok,
    stride_xm,
    stride_xk,
    hidden_size: tl.constexpr,
    max_routed_experts: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offsets_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_K,), dtype=tl.float32)

    for i in tl.range(max_routed_experts):
        permuted_idx = tl.load(permuted_indices + pid_m * max_routed_experts + i)

        if permuted_idx >= 0:
            x_ptrs = grad_outputs + permuted_idx * stride_don + offsets_k * stride_dok
            x = tl.load(x_ptrs)
            accumulator += x

    accumulator = accumulator.to(grad_inputs.dtype.element_ty)
    o_ptrs = grad_inputs + pid_m * stride_xm + offsets_k * stride_xk
    tl.store(o_ptrs, accumulator)


@triton.autotune(
    configs=_configs,
    key=["hidden_size"],
    prune_configs_by={
        "early_config_prune": _prune_func,
    },
)
@triton.jit
def _moe_token_unpermute_fwd_kernel(
    inputs,
    probs,
    outputs,
    permuted_indices,
    stride_xn,
    stride_xk,
    stride_pm,
    stride_pe,
    stride_om,
    stride_ok,
    hidden_size: tl.constexpr,
    max_routed_experts: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offsets_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_K,), dtype=tl.float32)

    for i in tl.range(max_routed_experts):
        permuted_idx = tl.load(permuted_indices + pid_m * max_routed_experts + i)

        if permuted_idx >= 0:
            x_ptrs = inputs + permuted_idx * stride_xn + offsets_k * stride_xk
            x = tl.load(x_ptrs)

            prob = tl.load(probs + pid_m * stride_pm + i * stride_pe).to(tl.float32)

            x *= prob
            accumulator += x

    accumulator = accumulator.to(outputs.dtype.element_ty)
    o_ptrs = outputs + pid_m * stride_om + offsets_k * stride_ok
    tl.store(o_ptrs, accumulator)


@triton.autotune(
    configs=_configs,
    key=["hidden_size"],
    prune_configs_by={
        "early_config_prune": _prune_func,
    },
)
@triton.jit
def _moe_token_unpermute_bwd_dp_kernel(
    grad_outputs,
    grad_probs,
    inputs,
    permuted_indices,
    stride_dom,
    stride_dok,
    stride_dpm,
    stride_dpe,
    stride_xn,
    stride_xk,
    hidden_size: tl.constexpr,
    max_routed_experts: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offsets_k = tl.arange(0, BLOCK_K)
    do_ptrs_base = grad_outputs + pid_m * stride_dom + offsets_k * stride_dok

    for i in tl.range(max_routed_experts):
        permuted_idx = tl.load(permuted_indices + pid_m * max_routed_experts + i).to(tl.int64)

        if permuted_idx >= 0:
            x_ptrs_base = inputs + permuted_idx * stride_xn + offsets_k * stride_xk
            accumulator = 0.0

            for j in tl.range(tl.cdiv(hidden_size, BLOCK_K)):
                do_ptrs = do_ptrs_base + (j * BLOCK_K) * stride_dok
                x_ptrs = x_ptrs_base + (j * BLOCK_K) * stride_xk

                do = tl.load(do_ptrs).to(tl.float32)
                x = tl.load(x_ptrs).to(tl.float32)
                accumulator += tl.sum(do * x)

            dp_ptr = grad_probs + pid_m * stride_dpm + i * stride_dpe
            tl.store(dp_ptr, accumulator)


@triton.autotune(
    configs=_configs,
    key=["hidden_size"],
    prune_configs_by={
        "early_config_prune": _prune_func,
    },
)
@triton.jit
def _moe_token_unpermute_bwd_dx_kernel(
    grad_outputs,
    grad_inputs,
    probs,
    permuted_indices,
    stride_dom,
    stride_dok,
    stride_dxn,
    stride_dxk,
    stride_pm,
    stride_pe,
    hidden_size: tl.constexpr,
    max_routed_experts: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offsets_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    do_ptrs = grad_outputs + pid_m * stride_dom + offsets_k * stride_dok
    do = tl.load(do_ptrs)

    for i in tl.range(max_routed_experts):
        permuted_idx = tl.load(permuted_indices + pid_m * max_routed_experts + i).to(tl.int64)

        if permuted_idx >= 0:
            prob = tl.load(probs + pid_m * stride_pm + i * stride_pe).to(tl.float32)

            dx_ptrs = grad_inputs + permuted_idx * stride_dxn + offsets_k * stride_dxk
            tl.store(dx_ptrs, do * prob)


class MoeTokenPermuteFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.FloatTensor,
        routed_expert_indices: torch.LongTensor,
        num_routed_tokens: List[int],
        num_experts_per_token: int,
    ):
        num_tokens, hidden_size = hidden_states.size()

        num_permuted_tokens = sum(num_routed_tokens)
        sorted_indices = torch.argsort(routed_expert_indices.flatten(), stable=True)[-num_permuted_tokens:]

        permuted_indices = torch.empty((num_tokens, num_experts_per_token), dtype=torch.int32, device=hidden_states.device).fill_(-1)

        _moe_token_permute_preprocess[lambda META: (triton.cdiv(num_permuted_tokens, META["BLOCK_N"]),)](
            sorted_indices,
            permuted_indices,
            num_permuted_tokens,
            num_experts_per_token,
            BLOCK_N=64,
        )

        permuted_tokens = torch.empty(num_permuted_tokens, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device)

        _moe_token_permute_fwd_kernel[lambda META: (num_tokens, triton.cdiv(hidden_size, META["BLOCK_K"]))](
            hidden_states,
            permuted_tokens,
            permuted_indices,
            hidden_states.stride(0),
            hidden_states.stride(1),
            permuted_tokens.stride(0),
            permuted_tokens.stride(1),
            hidden_size,
            num_experts_per_token,
        )

        ctx.save_for_backward(permuted_indices)
        ctx.num_tokens = num_tokens
        ctx.hidden_size = hidden_size
        ctx.num_experts_per_token = num_experts_per_token

        return permuted_tokens, permuted_indices

    @staticmethod
    def backward(
        ctx,
        grad_outputs,
        grad_permuted_indices,
    ):
        permuted_indices, = ctx.saved_tensors
        grad_inputs = grad_outputs.new_empty((ctx.num_tokens, ctx.hidden_size))

        _moe_token_permute_bwd_kernel[lambda META: (ctx.num_tokens, triton.cdiv(ctx.hidden_size, META["BLOCK_K"]))](
            grad_outputs,
            grad_inputs,
            permuted_indices,
            grad_outputs.stride(0),
            grad_outputs.stride(1),
            grad_inputs.stride(0),
            grad_inputs.stride(1),
            ctx.hidden_size,
            ctx.num_experts_per_token,
        )

        return grad_inputs, None, None, None


class MoeTokenUnpermuteFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs: torch.FloatTensor,
        probs: torch.FloatTensor,
        permuted_indices: torch.LongTensor,
        num_experts_per_token: int,
    ):
        hidden_size = inputs.size(1)
        num_tokens = probs.size(0)

        outputs = inputs.new_empty((num_tokens, hidden_size))

        _moe_token_unpermute_fwd_kernel[lambda META: (num_tokens, triton.cdiv(hidden_size, META["BLOCK_K"]))](
            inputs,
            probs,
            outputs,
            permuted_indices,
            inputs.stride(0),
            inputs.stride(1),
            probs.stride(0),
            probs.stride(1),
            outputs.stride(0),
            outputs.stride(1),
            hidden_size,
            num_experts_per_token,
        )

        ctx.save_for_backward(inputs, probs, permuted_indices)
        ctx.num_tokens = num_tokens
        ctx.hidden_size = hidden_size
        ctx.num_experts_per_token = num_experts_per_token

        return outputs

    @staticmethod
    def backward(
        ctx,
        grad_outputs,
    ):
        inputs, probs, permuted_indices = ctx.saved_tensors
        grad_probs = torch.zeros_like(probs)

        _moe_token_unpermute_bwd_dp_kernel[lambda _: (ctx.num_tokens,)](
            grad_outputs,
            grad_probs,
            inputs,
            permuted_indices,
            grad_outputs.stride(0),
            grad_outputs.stride(1),
            grad_probs.stride(0),
            grad_probs.stride(1),
            inputs.stride(0),
            inputs.stride(1),
            ctx.hidden_size,
            ctx.num_experts_per_token,
        )

        shape, dtype = inputs.shape, inputs.dtype
        del inputs
        grad_inputs = torch.empty(shape, dtype=dtype, device=grad_probs.device)

        _moe_token_unpermute_bwd_dx_kernel[lambda META: (ctx.num_tokens, triton.cdiv(ctx.hidden_size, META["BLOCK_K"]))](
            grad_outputs,
            grad_inputs,
            probs,
            permuted_indices,
            grad_outputs.stride(0),
            grad_outputs.stride(1),
            grad_inputs.stride(0),
            grad_inputs.stride(1),
            probs.stride(0),
            probs.stride(1),
            ctx.hidden_size,
            ctx.num_experts_per_token,
        )

        return grad_inputs, grad_probs, None, None


def moe_token_permute(
    hidden_states: torch.Tensor,
    routed_expert_indices: torch.LongTensor,
    num_routed_tokens: List[int],
    num_experts_per_token: int,
    backend: str = "triton",
):
    if backend == "triton":
        return MoeTokenPermuteFunction.apply(
            hidden_states,
            routed_expert_indices,
            num_routed_tokens,
            num_experts_per_token,
        )
    elif backend == "torch":
        num_permuted_tokens = sum(num_routed_tokens)
        sorted_indices = torch.argsort(routed_expert_indices.flatten(), stable=True)[-num_permuted_tokens:]
        hidden_states = hidden_states.index_select(0, sorted_indices // num_experts_per_token)
        return hidden_states, sorted_indices
    else:
        raise ValueError(f"Unknown backend: {backend}")


def moe_token_unpermute(
    permuted_tokens: torch.Tensor,
    probs: torch.FloatTensor,
    permuted_indices: torch.LongTensor,
    num_experts_per_token: int,
    backend: str = "triton",
):
    if backend == "triton":
        return MoeTokenUnpermuteFunction.apply(
            permuted_tokens,
            probs,
            permuted_indices,
            num_experts_per_token,
        )
    elif backend == "torch":
        hidden_size = permuted_tokens.size(1)
        outputs = permuted_tokens.new_zeros((probs.numel(), hidden_size))
        outputs.index_copy_(0, permuted_indices, permuted_tokens)
        outputs = outputs.view(-1, num_experts_per_token, hidden_size)
        outputs = outputs * probs.unsqueeze(-1)
        return outputs.sum(dim=1)
    else:
        raise ValueError(f"Unknown backend: {backend}")
