import torch
import triton
import triton.language as tl

from ..constants import NORM_BACKEND


@triton.jit
def _rms_norm_fwd_kernel(
    X,
    W,
    Y,
    Rstd,
    stride_x,
    stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    eps,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(ACC_DTYPE)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    tl.store(Rstd + row, rstd)

    w = tl.load(W + cols, mask=mask, other=0.0).to(ACC_DTYPE)
    tl.store(Y + row * stride_y + cols, x * rstd * w, mask=mask)


@triton.jit
def _rms_norm_bwd_kernel(
    DY,
    DX,
    X,
    W,
    Rstd,
    PartialDW,
    stride_dy,
    stride_dx,
    stride_x,
    stride_pdw,
    M,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    # Grid of `num_programs` row-block workers. Each strides over its share of rows,
    # writes DX per row and accumulates a partial DW in registers -- so DY/X/Rstd are
    # read exactly once and DW needs no atomics (partials summed in a cheap final pass).
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    w = tl.load(W + cols, mask=mask, other=0.0).to(ACC_DTYPE)
    partial_dw = tl.zeros([BLOCK], dtype=ACC_DTYPE)

    for row in range(pid, M, num_programs):
        dy = tl.load(DY + row * stride_dy + cols, mask=mask, other=0.0).to(ACC_DTYPE)
        x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(ACC_DTYPE)
        rstd = tl.load(Rstd + row).to(ACC_DTYPE)

        x_hat = x * rstd
        wdy = w * dy
        c = tl.sum(x_hat * wdy, axis=0) / N  # mean over the row of (x_hat * w * dy)
        dx = (wdy - x_hat * c) * rstd
        tl.store(DX + row * stride_dx + cols, dx, mask=mask)
        partial_dw += dy * x_hat

    tl.store(PartialDW + pid * stride_pdw + cols, partial_dw, mask=mask)


class FusedRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.reshape(-1, N)
        M = x2.shape[0]

        acc = tl.float64 if x.dtype == torch.float64 else tl.float32
        rstd_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32

        y = torch.empty_like(x2)
        rstd = torch.empty(M, device=x.device, dtype=rstd_dtype)
        block = triton.next_power_of_2(N)

        _rms_norm_fwd_kernel[(M,)](
            x2,
            weight,
            y,
            rstd,
            x2.stride(0),
            y.stride(0),
            N=N,
            BLOCK=block,
            ACC_DTYPE=acc,
            eps=eps,
        )

        ctx.save_for_backward(x2, weight, rstd)
        ctx.orig_shape = orig_shape
        return y.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_out):
        x2, weight, rstd = ctx.saved_tensors
        N = x2.shape[-1]
        M = x2.shape[0]

        acc = tl.float64 if x2.dtype == torch.float64 else tl.float32
        pdw_dtype = torch.float64 if x2.dtype == torch.float64 else torch.float32
        dy2 = grad_out.reshape(-1, N).contiguous()
        dx = torch.empty_like(x2)
        block = triton.next_power_of_2(N)

        # One worker per row-block; DX written per row and DW accumulated as partials
        # (summed below). Cap workers so each does real work while filling the GPU.
        num_programs = min(M, 1024)
        partial_dw = torch.empty(num_programs, N, device=x2.device, dtype=pdw_dtype)

        _rms_norm_bwd_kernel[(num_programs,)](
            dy2,
            dx,
            x2,
            weight,
            rstd,
            partial_dw,
            dy2.stride(0),
            dx.stride(0),
            x2.stride(0),
            partial_dw.stride(0),
            M,
            N=N,
            BLOCK=block,
            ACC_DTYPE=acc,
        )

        grad_w = partial_dw.sum(dim=0).to(weight.dtype)
        return dx.reshape(ctx.orig_shape), grad_w, None


def _rms_norm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (out * weight.float()).to(dtype)


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    backend: str = NORM_BACKEND,
) -> torch.Tensor:
    """Fused RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight.

    Computes in fp32 (fp64 for double inputs) regardless of input dtype, matching
    the reference `F.rms_norm(x.float(), ...)` behaviour. ``weight`` is used as-is;
    callers using the zero-centered convention must pass ``weight + 1.0``.
    """
    if backend == "triton":
        return FusedRMSNormFunction.apply(x, weight, eps)
    elif backend == "torch":
        return _rms_norm_torch(x, weight, eps)
    else:
        raise ValueError(f"Unknown backend: {backend}")
