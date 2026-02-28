"""Fused Rotary Position Embedding (RoPE) implemented with CuTe DSL.

Applies RoPE to query and key tensors in a single fused CUDA kernel launch:

    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin

where ``rotate_half(x) = concat(-x[..., head_dim/2:], x[..., :head_dim/2])``.

Input tensors
-------------
q   : Tensor of shape ``[total_tokens, num_q_heads, head_dim]``, dtype fp16/bf16.
k   : Tensor of shape ``[total_tokens, num_k_heads, head_dim]``, dtype fp16/bf16.
cos : Tensor of shape ``[total_tokens, head_dim/2]``, dtype float32.
sin : Tensor of shape ``[total_tokens, head_dim/2]``, dtype float32.

The ``cos`` and ``sin`` tensors are expected to be pre-computed (e.g. via
``RotaryEmbedding``) and already sliced to ``head_dim / 2`` (the rotation is
applied identically to both halves of the head dimension).
"""

import torch

try:
    from . import _C
except Exception:
    _C = None


def _fused_rope_forward_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference implementation (used when the CUDA extension is
    unavailable or for debugging / correctness verification)."""
    # cos/sin: [total_tokens, half_dim] -> [total_tokens, 1, head_dim]
    half_dim = cos.size(-1)
    cos_full = torch.cat([cos, cos], dim=-1).unsqueeze(1)  # [T, 1, D]
    sin_full = torch.cat([sin, sin], dim=-1).unsqueeze(1)  # [T, 1, D]

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat([-x2, x1], dim=-1)

    q_out = (q.float() * cos_full + rotate_half(q.float()) * sin_full).to(q.dtype)
    k_out = (k.float() * cos_full + rotate_half(k.float()) * sin_full).to(k.dtype)
    return q_out, k_out


def fused_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embeddings (RoPE) to *q* and *k* in a single
    fused CUDA kernel.

    Parameters
    ----------
    q   : ``[total_tokens, num_q_heads, head_dim]``  fp16 or bf16.
    k   : ``[total_tokens, num_k_heads, head_dim]``  fp16 or bf16.
    cos : ``[total_tokens, head_dim / 2]``           float32.
    sin : ``[total_tokens, head_dim / 2]``           float32.

    Returns
    -------
    q_out : same shape/dtype as *q*.
    k_out : same shape/dtype as *k*.
    """
    assert q.is_cuda and k.is_cuda, "q and k must be on a CUDA device"
    assert cos.is_cuda and sin.is_cuda, "cos and sin must be on a CUDA device"
    assert q.is_contiguous(), "q must be contiguous"
    assert k.is_contiguous(), "k must be contiguous"
    assert cos.is_contiguous(), "cos must be contiguous"
    assert sin.is_contiguous(), "sin must be contiguous"

    if _C is not None:
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        _C.fused_rope_forward(q, k, cos, sin, q_out, k_out)
    else:
        q_out, k_out = _fused_rope_forward_ref(q, k, cos, sin)

    return q_out, k_out
