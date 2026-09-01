import torch
import triton
import triton.language as tl

from ..constants import ROPE_BACKEND


@triton.jit
def _apply_rope_kernel(
    q,
    k,
    q_out,
    k_out,
    cos,
    sin,
    stride_qb,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_qob,
    stride_qon,
    stride_qoh,
    stride_qod,
    stride_kob,
    stride_kon,
    stride_koh,
    stride_kod,
    stride_cosb,
    stride_cosn,
    stride_cosd,
    stride_sinb,
    stride_sinn,
    stride_sind,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    BLOCK_QH: tl.constexpr,
    BLOCK_KH: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    IS_BACKWARD: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    # BLOCK_D pads HALF_DIM up to a power of 2 (tl.arange requirement); mask_d
    # drops the padding lanes so odd rotary dims (e.g. head_dim 72 -> half 36) work.
    offsets_d = tl.arange(0, BLOCK_D)
    mask_d = offsets_d < HALF_DIM

    # Compute the rotation in ACC_DTYPE (fp32 for bf16/fp16/fp32 inputs, fp64 for fp64):
    # the op is memory-bound so higher-precision math is free, it improves accuracy over
    # bf16, and it lets callers pass bf16 tensors directly instead of an explicit .float()
    # round-trip. tl.store casts the result back to the (possibly bf16) output dtype.
    cos_ptrs = cos + pid_b * stride_cosb + pid_n * stride_cosn + offsets_d[None, :] * stride_cosd
    cos_val = tl.load(cos_ptrs, mask=mask_d[None, :], other=0.0).to(ACC_DTYPE)
    sin_ptrs = sin + pid_b * stride_sinb + pid_n * stride_sinn + offsets_d[None, :] * stride_sind
    sin_val = tl.load(sin_ptrs, mask=mask_d[None, :], other=0.0).to(ACC_DTYPE)

    # --- Q rotation ---
    offsets_qh = tl.arange(0, BLOCK_QH)
    mask_qh = offsets_qh < NUM_Q_HEADS
    mask_q = mask_qh[:, None] & mask_d[None, :]
    q_base = q + pid_b * stride_qb + pid_n * stride_qn + offsets_qh[:, None] * stride_qh
    q_first_ptrs = q_base + offsets_d[None, :] * stride_qd
    q_first = tl.load(q_first_ptrs, mask=mask_q, other=0.0).to(ACC_DTYPE)
    q_second_ptrs = q_base + (offsets_d[None, :] + HALF_DIM) * stride_qd
    q_second = tl.load(q_second_ptrs, mask=mask_q, other=0.0).to(ACC_DTYPE)

    # --- K rotation ---
    offsets_kh = tl.arange(0, BLOCK_KH)
    mask_kh = offsets_kh < NUM_KV_HEADS
    mask_k = mask_kh[:, None] & mask_d[None, :]
    k_base = k + pid_b * stride_kb + pid_n * stride_kn + offsets_kh[:, None] * stride_kh
    k_first_ptrs = k_base + offsets_d[None, :] * stride_kd
    k_first = tl.load(k_first_ptrs, mask=mask_k, other=0.0).to(ACC_DTYPE)
    k_second_ptrs = k_base + (offsets_d[None, :] + HALF_DIM) * stride_kd
    k_second = tl.load(k_second_ptrs, mask=mask_k, other=0.0).to(ACC_DTYPE)

    # --- Output pointers ---
    qo_base = q_out + pid_b * stride_qob + pid_n * stride_qon + offsets_qh[:, None] * stride_qoh
    qo_first_ptrs = qo_base + offsets_d[None, :] * stride_qod
    qo_second_ptrs = qo_base + (offsets_d[None, :] + HALF_DIM) * stride_qod
    ko_base = k_out + pid_b * stride_kob + pid_n * stride_kon + offsets_kh[:, None] * stride_koh
    ko_first_ptrs = ko_base + offsets_d[None, :] * stride_kod
    ko_second_ptrs = ko_base + (offsets_d[None, :] + HALF_DIM) * stride_kod

    if IS_BACKWARD:
        tl.store(qo_first_ptrs, q_first * cos_val + q_second * sin_val, mask=mask_q)
        tl.store(qo_second_ptrs, q_second * cos_val - q_first * sin_val, mask=mask_q)
        tl.store(ko_first_ptrs, k_first * cos_val + k_second * sin_val, mask=mask_k)
        tl.store(ko_second_ptrs, k_second * cos_val - k_first * sin_val, mask=mask_k)
    else:
        tl.store(qo_first_ptrs, q_first * cos_val - q_second * sin_val, mask=mask_q)
        tl.store(qo_second_ptrs, q_second * cos_val + q_first * sin_val, mask=mask_q)
        tl.store(ko_first_ptrs, k_first * cos_val - k_second * sin_val, mask=mask_k)
        tl.store(ko_second_ptrs, k_second * cos_val + k_first * sin_val, mask=mask_k)


def _launch_rope_kernel(q, k, q_out, k_out, cos, sin, rotary_dim, is_backward):
    # Triton treats distinct pointer args as non-aliasing; an aliased in/out buffer
    # is only safe for the forward rotation (loads all lanes before storing). The
    # backward rotation must use fresh output buffers -- guard against a regression.
    if is_backward:
        assert q_out.data_ptr() != q.data_ptr() and k_out.data_ptr() != k.data_ptr(), (
            "rope backward output must not alias the incoming gradient buffer"
        )

    seq_len, num_q_heads, head_dim = q.shape[-3:]
    num_kv_heads = k.shape[-2]
    batch_size = q.numel() // (seq_len * num_q_heads * head_dim)

    q_4d = q.view(batch_size, seq_len, num_q_heads, head_dim)
    k_4d = k.view(batch_size, seq_len, num_kv_heads, head_dim)
    qo_4d = q_out.view(batch_size, seq_len, num_q_heads, head_dim)
    ko_4d = k_out.view(batch_size, seq_len, num_kv_heads, head_dim)
    cos_3d = cos.reshape(batch_size, seq_len, -1)
    sin_3d = sin.reshape(batch_size, seq_len, -1)

    q_rot = q_4d[..., :rotary_dim] if rotary_dim < head_dim else q_4d
    k_rot = k_4d[..., :rotary_dim] if rotary_dim < head_dim else k_4d
    qo_rot = qo_4d[..., :rotary_dim] if rotary_dim < head_dim else qo_4d
    ko_rot = ko_4d[..., :rotary_dim] if rotary_dim < head_dim else ko_4d

    # Accumulate in fp32 for low-precision inputs, fp64 for fp64 (keeps gradcheck exact).
    acc_dtype = tl.float64 if q.dtype == torch.float64 else tl.float32

    _apply_rope_kernel[lambda _: (batch_size, seq_len)](
        q_rot,
        k_rot,
        qo_rot,
        ko_rot,
        cos_3d,
        sin_3d,
        q_rot.stride(0),
        q_rot.stride(1),
        q_rot.stride(2),
        q_rot.stride(3),
        k_rot.stride(0),
        k_rot.stride(1),
        k_rot.stride(2),
        k_rot.stride(3),
        qo_rot.stride(0),
        qo_rot.stride(1),
        qo_rot.stride(2),
        qo_rot.stride(3),
        ko_rot.stride(0),
        ko_rot.stride(1),
        ko_rot.stride(2),
        ko_rot.stride(3),
        cos_3d.stride(0),
        cos_3d.stride(1),
        cos_3d.stride(2),
        sin_3d.stride(0),
        sin_3d.stride(1),
        sin_3d.stride(2),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        BLOCK_QH=triton.next_power_of_2(num_q_heads),
        BLOCK_KH=triton.next_power_of_2(num_kv_heads),
        HALF_DIM=rotary_dim // 2,
        BLOCK_D=triton.next_power_of_2(rotary_dim // 2),
        ACC_DTYPE=acc_dtype,
        IS_BACKWARD=is_backward,
    )


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
        head_dim = q.shape[-1]
        rotary_dim = cos.shape[-1] * 2

        if inplace:
            q_out, k_out = q, k
            ctx.mark_dirty(q)
            ctx.mark_dirty(k)
        else:
            q_out, k_out = torch.empty_like(q), torch.empty_like(k)
            if rotary_dim < head_dim:
                q_out[..., rotary_dim:] = q[..., rotary_dim:]
                k_out[..., rotary_dim:] = k[..., rotary_dim:]

        _launch_rope_kernel(q, k, q_out, k_out, cos, sin, rotary_dim, is_backward=False)

        ctx.save_for_backward(cos, sin)
        ctx.inplace = inplace
        ctx.rotary_dim = rotary_dim

        return q_out, k_out

    @staticmethod
    def backward(ctx, grad_q_out, grad_k_out):
        cos, sin = ctx.saved_tensors
        head_dim = grad_q_out.shape[-1]
        rotary_dim = ctx.rotary_dim

        # Always write to fresh buffers: the kernel would otherwise read and write
        # the same memory (input aliased to output), which Triton does not treat as
        # alias-safe and silently corrupts the gradient. Do NOT reuse grad_*_out here.
        grad_q, grad_k = torch.empty_like(grad_q_out), torch.empty_like(grad_k_out)
        if rotary_dim < head_dim:
            grad_q[..., rotary_dim:] = grad_q_out[..., rotary_dim:]
            grad_k[..., rotary_dim:] = grad_k_out[..., rotary_dim:]

        # autograd may pass expanded grads (e.g. stride-0 from sum()); contiguous() materializes them
        _launch_rope_kernel(
            grad_q_out.contiguous(),
            grad_k_out.contiguous(),
            grad_q,
            grad_k,
            cos,
            sin,
            rotary_dim,
            is_backward=True,
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
    backend: str = ROPE_BACKEND,
):
    head_dim = q.shape[-1]
    rotary_dim = cos.shape[-1] * 2

    if backend == "triton":
        q_embed, k_embed = ApplyRopeFunction.apply(q, k, cos, sin, inplace)
        return q_embed, k_embed
    elif backend == "torch":
        # Match the triton kernel: compute in fp32 regardless of input dtype.
        q_rot = q[..., :rotary_dim].float()
        k_rot = k[..., :rotary_dim].float()
        cos_expanded = cos.float().unsqueeze(-2)
        sin_expanded = sin.float().unsqueeze(-2)
        cos_full = torch.cat([cos_expanded, cos_expanded], dim=-1)
        sin_full = torch.cat([sin_expanded, sin_expanded], dim=-1)
        q_embed = (q_rot * cos_full) + (_rotate_half(q_rot) * sin_full)
        k_embed = (k_rot * cos_full) + (_rotate_half(k_rot) * sin_full)
        if rotary_dim < head_dim:
            q_embed = torch.cat([q_embed, q[..., rotary_dim:].float()], dim=-1)
            k_embed = torch.cat([k_embed, k[..., rotary_dim:].float()], dim=-1)
        return q_embed.to(q.dtype), k_embed.to(k.dtype)
    else:
        raise ValueError(f"Unknown backend: {backend}")
