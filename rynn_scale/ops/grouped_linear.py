import os
from typing import List

import torch
import triton
import triton.language as tl

MOE_GEMM_BACKEND = os.environ.get("MOE_GEMM_BACKEND", "cutlass")


try:
    from . import _C
except Exception:
    _C = None


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def num_sms():
    if is_cuda():
        return torch.cuda.get_device_properties("cuda").multi_processor_count
    return 148


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "NUM_SM": 84,
            }
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "NUM_SM": 128,
            }
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "NUM_SM": 84,
            }
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "NUM_SM": 128,
            }
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "NUM_SM": num_sms(),
            }
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "NUM_SM": num_sms(),
            }
        ),
    ],
    key=["num_groups"],
)
@triton.jit
def _grouped_gemm_packed_triton(
    A,
    B,
    D,
    M,
    N,
    K,
    A_split_dim: tl.constexpr,
    B_split_dim: tl.constexpr,
    D_split_dim: tl.constexpr,
    num_groups: tl.constexpr,
    # strides
    stride_Am: tl.constexpr,
    stride_Ak: tl.constexpr,
    stride_Bk: tl.constexpr,
    stride_Bn: tl.constexpr,
    stride_Dm: tl.constexpr,
    stride_Dn: tl.constexpr,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)

    last_problem_end = 0
    last_problem_end = last_problem_end.to(tl.int64)

    for g in range(num_groups):
        m = tl.load(M + g)
        n = tl.load(N + g)
        k = tl.load(K + g)

        if m > 0 and n > 0:
            num_m_tiles = tl.cdiv(m, BLOCK_M)
            num_n_tiles = tl.cdiv(n, BLOCK_N)
            num_tiles = num_m_tiles * num_n_tiles

            while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx = tile_idx_in_gemm // num_n_tiles
                tile_n_idx = tile_idx_in_gemm % num_n_tiles

                offs_m = tile_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = tile_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
                offs_k = tl.arange(0, BLOCK_K)

                A_ptrs = A + offs_m[:, None] * stride_Am + offs_k[None, :] * stride_Ak
                B_ptrs = B + offs_k[:, None] * stride_Bk + offs_n[None, :] * stride_Bn

                accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                for tile_k_idx in range(0, tl.cdiv(k, BLOCK_K)):
                    A_mask = (offs_m[:, None] < m) * (offs_k[None, :] < k)
                    B_mask = (offs_k[:, None] < k) * (offs_n[None, :] < n)

                    mat_A = tl.load(A_ptrs, mask=A_mask, other=0.0)
                    mat_B = tl.load(B_ptrs, mask=B_mask, other=0.0)
                    accumulator = tl.dot(mat_A, mat_B, accumulator)

                    A_ptrs += BLOCK_K * stride_Ak
                    B_ptrs += BLOCK_K * stride_Bk
                    offs_k += BLOCK_K

                D_ptrs = D + offs_m[:, None] * stride_Dm + offs_n[None, :] * stride_Dn
                D_mask = (offs_m[:, None] < m) * (offs_n[None, :] < n)
                tl.store(D_ptrs, accumulator.to(D.dtype.element_ty), mask=D_mask)

                tile_idx += NUM_SM

            last_problem_end += num_tiles

        A += m * stride_Am if A_split_dim == 0 else k * stride_Ak
        B += k * stride_Bk if B_split_dim == 0 else n * stride_Bn
        D += m * stride_Dm if D_split_dim == 0 else n * stride_Dn


def _grouped_linear_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    input_group_sizes: list[int],
):
    assert weight.size(0) % len(input_group_sizes) == 0, "Weight output features must be divisible by number of groups"
    group_out_features = weight.size(0) // len(input_group_sizes)

    assert input.size(0) == output.size(0), "Input and output batch size must match"
    assert input.size(1) == weight.size(1), "Input and weight feature size must match"
    assert group_out_features == output.size(1), "Weight and output feature size must match"

    assert input.is_contiguous(), "Input must be contiguous"
    assert weight.is_contiguous(), "Weight must be contiguous"
    assert output.is_contiguous(), "Output must be contiguous"

    num_groups = len(input_group_sizes)

    M = input.new_tensor(input_group_sizes, dtype=torch.int32)
    N = input.new_full((num_groups,), group_out_features, dtype=torch.int32)
    K = input.new_full((num_groups,), input.size(1), dtype=torch.int32)

    _grouped_gemm_packed_triton[lambda META: (META["NUM_SM"],)](
        input,
        weight,
        output,
        M,
        N,
        K,
        0,
        1,
        0,
        num_groups,
        input.stride(0),
        input.stride(1),
        weight.stride(1),
        weight.stride(0),
        output.stride(0),
        output.stride(1),
    )


def _grouped_linear_backward_dx(
    input: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    grad_input: torch.Tensor,
    input_group_sizes: list[int],
):
    assert weight.size(0) % len(input_group_sizes) == 0, "Weight output features must be divisible by number of groups"
    group_out_features = weight.size(0) // len(input_group_sizes)

    assert input.size(0) == grad_output.size(0), "Input and output batch size must match"
    assert input.size(1) == weight.size(1), "Input and weight feature size must match"
    assert group_out_features == grad_output.size(1), "Weight and output feature size must match"
    assert grad_input.size() == input.size(), "Grad input size must match input size"

    assert input.is_contiguous(), "Input must be contiguous"
    assert weight.is_contiguous(), "Weight must be contiguous"
    assert grad_output.is_contiguous(), "Output must be contiguous"
    assert grad_input.is_contiguous(), "Grad input must be contiguous"

    num_groups = len(input_group_sizes)

    M = input.new_tensor(input_group_sizes, dtype=torch.int32)
    N = input.new_full((num_groups,), group_out_features, dtype=torch.int32)
    K = input.new_full((num_groups,), input.size(1), dtype=torch.int32)

    _grouped_gemm_packed_triton[lambda META: (META["NUM_SM"],)](
        grad_output,
        weight,
        grad_input,
        M,
        K,
        N,
        0,
        0,
        0,
        num_groups,
        grad_output.stride(0),
        grad_output.stride(1),
        weight.stride(0),
        weight.stride(1),
        grad_input.stride(0),
        grad_input.stride(1),
    )


def _grouped_linear_backward_dw(
    input: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    grad_weight: torch.Tensor,
    input_group_sizes: list[int],
):
    assert weight.size(0) % len(input_group_sizes) == 0, "Weight output features must be divisible by number of groups"
    group_out_features = weight.size(0) // len(input_group_sizes)

    assert input.size(0) == grad_output.size(0), "Input and output batch size must match"
    assert input.size(1) == weight.size(1), "Input and weight feature size must match"
    assert group_out_features == grad_output.size(1), "Weight and output feature size must match"
    assert grad_weight.size() == weight.size(), "Grad weight size must match weight size"

    assert input.is_contiguous(), "Input must be contiguous"
    assert weight.is_contiguous(), "Weight must be contiguous"
    assert grad_output.is_contiguous(), "Output must be contiguous"
    assert grad_weight.is_contiguous(), "Grad weight must be contiguous"

    num_groups = len(input_group_sizes)

    M = input.new_tensor(input_group_sizes, dtype=torch.int32)
    N = input.new_full((num_groups,), group_out_features, dtype=torch.int32)
    K = input.new_full((num_groups,), input.size(1), dtype=torch.int32)

    _grouped_gemm_packed_triton[lambda META: (META["NUM_SM"],)](
        grad_output,
        input,
        grad_weight,
        N,
        K,
        M,
        1,
        0,
        0,
        num_groups,
        grad_output.stride(1),
        grad_output.stride(0),
        input.stride(0),
        input.stride(1),
        grad_weight.stride(0),
        grad_weight.stride(1),
    )


class GroupedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_group_sizes: List[int],
        backend: str = MOE_GEMM_BACKEND,
    ) -> torch.Tensor:
        output = input.new_empty((input.size(0), weight.size(0) // len(input_group_sizes)))

        if backend.lower() == "cutlass":
            assert _C is not None
            func = _C.grouped_linear_forward
        elif backend.lower() == "triton":
            func = _grouped_linear_forward
        else:
            raise ValueError(f"Unkwon backend: {backend}")

        func(input, weight, output, input_group_sizes)

        ctx.save_for_backward(input, weight)
        ctx.input_group_sizes = input_group_sizes
        ctx.backend = backend

        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        input, weight = ctx.saved_tensors
        input_group_sizes = ctx.input_group_sizes
        backend = ctx.backend

        if backend.lower() == "cutlass":
            assert _C is not None
            dx_kernel = _C.grouped_linear_backward_dx
            dw_kernel = _C.grouped_linear_backward_dw
        elif backend.lower() == "triton":
            dx_kernel = _grouped_linear_backward_dx
            dw_kernel = _grouped_linear_backward_dw
        else:
            raise ValueError(f"Unkwon backend: {backend}")

        if input.requires_grad:
            grad_input = torch.zeros_like(input)
            dx_kernel(input, weight, grad_output, grad_input, input_group_sizes)
        else:
            grad_input = None

        if weight.requires_grad:
            grad_weight = torch.zeros_like(weight)
            dw_kernel(input, weight, grad_output, grad_weight, input_group_sizes)
        else:
            grad_weight = None

        return grad_input, grad_weight, None, None


def grouped_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_group_sizes: List[int],
    backend: str = MOE_GEMM_BACKEND,
) -> torch.Tensor:
    if backend.lower() == "torch":
        inputs = torch.split(input, input_group_sizes, dim=0)
        weights = torch.chunk(weight, len(input_group_sizes), dim=0)
        outputs = [torch.nn.functional.linear(inp, w) for inp, w in zip(inputs, weights)]
        return torch.cat(outputs, dim=0)
    return GroupedLinearFunction.apply(input, weight, input_group_sizes, backend)
