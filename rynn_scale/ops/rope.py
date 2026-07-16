import torch
import triton
import triton.language as tl


@triton.jit
def _apply_rope_kernel(
    q, k, q_out, k_out, cos, sin,
    stride_qb, stride_qn, stride_qh, stride_qd,
    stride_kb, stride_kn, stride_kh, stride_kd,
    stride_qob, stride_qon, stride_qoh, stride_qod,
    stride_kob, stride_kon, stride_koh, stride_kod,
    stride_cosb, stride_cosn, stride_cosd,
    stride_sinb, stride_sinn, stride_sind,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    INPLACE: tl.constexpr,
    IS_BACKWARD: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    offsets_k = tl.arange(0, HEAD_DIM // 2)

    cos_ptrs = cos + pid_b * stride_cosb + pid_n * stride_cosn + offsets_k[None, :] * stride_cosd
    cos = tl.load(cos_ptrs)
    sin_ptrs = sin + pid_b * stride_sinb + pid_n * stride_sinn + offsets_k[None, :] * stride_sind
    sin = tl.load(sin_ptrs)

    offsets_qh = tl.arange(0, NUM_Q_HEADS)
    q_first_ptrs = q + pid_b * stride_qb + pid_n * stride_qn + offsets_qh[:, None] * stride_qh + offsets_k[None, :] * stride_qd
    q_first = tl.load(q_first_ptrs).to(cos.dtype)
    q_second_ptrs = q_first_ptrs + HEAD_DIM // 2 * stride_qd
    q_second = tl.load(q_second_ptrs).to(cos.dtype)

    offsets_kh = tl.arange(0, NUM_KV_HEADS)
    k_first_ptrs = k + pid_b * stride_kb + pid_n * stride_kn + offsets_kh[:, None] * stride_kh + offsets_k[None, :] * stride_kd
    k_first = tl.load(k_first_ptrs).to(cos.dtype)
    k_second_ptrs = k_first_ptrs + HEAD_DIM // 2 * stride_kd
    k_second = tl.load(k_second_ptrs).to(cos.dtype)

    if INPLACE:
        qo_first_ptrs = q_first_ptrs
        qo_second_ptrs = q_second_ptrs
        ko_first_ptrs = k_first_ptrs
        ko_second_ptrs = k_second_ptrs
    else:
        qo_first_ptrs = q_out + pid_b * stride_qob + pid_n * stride_qon + offsets_qh[:, None] * stride_qoh + offsets_k[None, :] * stride_qod
        qo_second_ptrs = qo_first_ptrs + HEAD_DIM // 2 * stride_qod
        ko_first_ptrs = k_out + pid_b * stride_kob + pid_n * stride_kon + offsets_kh[:, None] * stride_koh + offsets_k[None, :] * stride_kod
        ko_second_ptrs = ko_first_ptrs + HEAD_DIM // 2 * stride_kod

    if IS_BACKWARD:
        tl.store(qo_first_ptrs, q_first * cos + q_second * sin)
        tl.store(qo_second_ptrs, q_second * cos - q_first * sin)
        tl.store(ko_first_ptrs, k_first * cos + k_second * sin)
        tl.store(ko_second_ptrs, k_second * cos - k_first * sin)
    else:
        tl.store(qo_first_ptrs, q_first * cos - q_second * sin)
        tl.store(qo_second_ptrs, q_second * cos + q_first * sin)
        tl.store(ko_first_ptrs, k_first * cos - k_second * sin)
        tl.store(ko_second_ptrs, k_second * cos + k_first * sin)


class ApplyRopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        inplace: bool = True,
    ):
        head_dim = q.size(-1)

        if inplace:
            q_out, k_out = q, k
            ctx.mark_dirty(q)
            ctx.mark_dirty(k)
        else:
            q_out, k_out = torch.empty_like(q), torch.empty_like(k)

        _apply_rope_kernel[lambda _: (q.shape[0], q.shape[1])](
            q, k, q_out, k_out, cos, sin,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            q_out.stride(0), q_out.stride(1), q_out.stride(2), q_out.stride(3),
            k_out.stride(0), k_out.stride(1), k_out.stride(2), k_out.stride(3),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            NUM_Q_HEADS=q.size(-2),
            NUM_KV_HEADS=k.size(-2),
            HEAD_DIM=head_dim,
            INPLACE=inplace,
            IS_BACKWARD=False,
        )

        ctx.save_for_backward(cos, sin)
        ctx.inplace = inplace

        return q_out, k_out

    @staticmethod
    def backward(ctx, grad_q_out, grad_k_out):
        cos, sin = ctx.saved_tensors
        head_dim = grad_q_out.size(-1)

        if ctx.inplace:
            grad_q, grad_k = grad_q_out, grad_k_out
        else:
            grad_q, grad_k = torch.empty_like(grad_q_out), torch.empty_like(grad_k_out)

        _apply_rope_kernel[lambda _: (grad_q_out.shape[0], grad_q_out.shape[1])](
            grad_q_out, grad_k_out, grad_q, grad_k, cos, sin,
            grad_q_out.stride(0), grad_q_out.stride(1), grad_q_out.stride(2), grad_q_out.stride(3),
            grad_k_out.stride(0), grad_k_out.stride(1), grad_k_out.stride(2), grad_k_out.stride(3),
            grad_q.stride(0), grad_q.stride(1), grad_q.stride(2), grad_q.stride(3),
            grad_k.stride(0), grad_k.stride(1), grad_k.stride(2), grad_k.stride(3),
            cos.stride(0), cos.stride(1), cos.stride(2),
            sin.stride(0), sin.stride(1), sin.stride(2),
            NUM_Q_HEADS=grad_q_out.size(-2),
            NUM_KV_HEADS=grad_k_out.size(-2),
            HEAD_DIM=head_dim,
            INPLACE=ctx.inplace,
            IS_BACKWARD=True,
        )

        return grad_q, grad_k, None, None, None


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inplace: bool = True,
    backend: str = "triton",
):
    if backend == "triton":
        q_embed, k_embed = ApplyRopeFunction.apply(
            q,
            k,
            cos,
            sin,
            inplace,
        )
        return q_embed, k_embed
    elif backend == "torch":
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
        q_embed = (q * cos) + (_rotate_half(q) * sin)
        k_embed = (k * cos) + (_rotate_half(k) * sin)
        return q_embed.to(q.dtype), k_embed.to(k.dtype)
    else:
        raise ValueError(f"Unknown backend: {backend}")
