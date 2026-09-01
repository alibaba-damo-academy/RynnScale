import pytest
import torch

from rynn_scale.ops.rope import apply_rope


def _precompute_rope_cache(seq_len, head_dim, device="cuda", dtype=torch.float32):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq).unsqueeze(0)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _generate_rope_data(
    batch_size, seq_len, num_q_heads, num_kv_heads, head_dim, rotary_dim=None, dtype=torch.bfloat16
):
    if rotary_dim is None:
        rotary_dim = head_dim
    q = torch.randn(batch_size, seq_len, num_q_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    cos, sin = _precompute_rope_cache(seq_len, rotary_dim, dtype=dtype)
    cos = cos.expand(batch_size, -1, -1)
    sin = sin.expand(batch_size, -1, -1)
    return q, k, cos, sin


# power-of-2 + non-power-of-2 head counts
@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(32, 4), (16, 16), (28, 4), (12, 3)])
@pytest.mark.parametrize("head_dim", [64, 72, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rope_forward_correctness(num_q_heads, num_kv_heads, head_dim, dtype):
    q, k, cos, sin = _generate_rope_data(2, 128, num_q_heads, num_kv_heads, head_dim, dtype=dtype)

    q_ref, k_ref = apply_rope(q.clone(), k.clone(), cos, sin, inplace=False, backend="torch")
    q_tri, k_tri = apply_rope(q.clone(), k.clone(), cos, sin, inplace=False, backend="triton")

    atol = 1e-2 if dtype == torch.bfloat16 else 5e-3
    assert torch.allclose(q_tri, q_ref, atol=atol, rtol=1e-2), f"Q max diff={torch.abs(q_tri - q_ref).max():.6f}"
    assert torch.allclose(k_tri, k_ref, atol=atol, rtol=1e-2), f"K max diff={torch.abs(k_tri - k_ref).max():.6f}"


@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(32, 4), (12, 3)])
@pytest.mark.parametrize("head_dim", [64, 72, 128])
def test_rope_backward_correctness(num_q_heads, num_kv_heads, head_dim):
    q, k, cos, sin = _generate_rope_data(2, 64, num_q_heads, num_kv_heads, head_dim, dtype=torch.float32)

    q_ref = q.clone().requires_grad_(True)
    k_ref = k.clone().requires_grad_(True)
    loss_ref = sum(x.sum() for x in apply_rope(q_ref, k_ref, cos, sin, inplace=False, backend="torch"))
    loss_ref.backward()

    q_tri = q.clone().requires_grad_(True)
    k_tri = k.clone().requires_grad_(True)
    loss_tri = sum(x.sum() for x in apply_rope(q_tri, k_tri, cos, sin, inplace=False, backend="triton"))
    loss_tri.backward()

    assert torch.allclose(q_tri.grad, q_ref.grad, atol=1e-5, rtol=1e-4), (
        f"Q grad max diff={torch.abs(q_tri.grad - q_ref.grad).max():.8f}"
    )
    assert torch.allclose(k_tri.grad, k_ref.grad, atol=1e-5, rtol=1e-4), (
        f"K grad max diff={torch.abs(k_tri.grad - k_ref.grad).max():.8f}"
    )


@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(32, 4), (12, 3)])
@pytest.mark.parametrize("head_dim", [64, 72, 128])
def test_rope_backward_inplace_correctness(num_q_heads, num_kv_heads, head_dim):
    # inplace=True is what the models use; its backward must match the reference.
    # The tensor fed to apply_rope must be a non-leaf so it can be mutated in place.
    # Use a DATA-DEPENDENT upstream grad (fixed random weights) as the incoming
    # gradient: a uniform grad (plain .sum()) hides index/stride/aliasing bugs.
    q, k, cos, sin = _generate_rope_data(2, 64, num_q_heads, num_kv_heads, head_dim, dtype=torch.float32)
    wq = torch.randn_like(q)
    wk = torch.randn_like(k)

    def loss(q_in, k_in, inplace, backend):
        qo, ko = apply_rope(q_in * 1.0, k_in * 1.0, cos, sin, inplace=inplace, backend=backend)
        return (qo * wq).sum() + (ko * wk).sum()

    q_ref = q.clone().requires_grad_(True)
    k_ref = k.clone().requires_grad_(True)
    loss(q_ref, k_ref, inplace=False, backend="torch").backward()

    q_tri = q.clone().requires_grad_(True)
    k_tri = k.clone().requires_grad_(True)
    loss(q_tri, k_tri, inplace=True, backend="triton").backward()

    assert torch.allclose(q_tri.grad, q_ref.grad, atol=1e-5, rtol=1e-4), (
        f"Q grad max diff={torch.abs(q_tri.grad - q_ref.grad).max():.8f}"
    )
    assert torch.allclose(k_tri.grad, k_ref.grad, atol=1e-5, rtol=1e-4), (
        f"K grad max diff={torch.abs(k_tri.grad - k_ref.grad).max():.8f}"
    )


@pytest.mark.parametrize("head_dim", [64, 72, 128])
def test_rope_gradcheck(head_dim):
    # gradcheck (fp64) validates the backward against numerical gradients -- the
    # gold-standard guard for a custom autograd Function's backward formula/kernel.
    q, k, cos, sin = _generate_rope_data(1, 8, 8, 2, head_dim, dtype=torch.float64)
    q = q.requires_grad_(True)
    k = k.requires_grad_(True)

    def fn(q_, k_):
        # inplace=False so gradcheck can re-run forward on the same inputs
        return apply_rope(q_, k_, cos, sin, inplace=False, backend="triton")

    assert torch.autograd.gradcheck(fn, (q, k), atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("head_dim", [64, 128])
def test_rope_inplace(head_dim):
    q, k, cos, sin = _generate_rope_data(2, 128, 16, 4, head_dim)

    q_oop, k_oop = apply_rope(q.clone(), k.clone(), cos, sin, inplace=False, backend="triton")

    q_inp = q.clone()
    k_inp = k.clone()
    q_inp_out, k_inp_out = apply_rope(q_inp, k_inp, cos, sin, inplace=True, backend="triton")

    assert torch.equal(q_inp_out, q_oop)
    assert torch.equal(k_inp_out, k_oop)
    assert q_inp_out.data_ptr() == q_inp.data_ptr()
    assert k_inp_out.data_ptr() == k_inp.data_ptr()


@pytest.mark.parametrize("rotary_dim", [32, 64])
def test_rope_partial_rotary(rotary_dim):
    q, k, cos, sin = _generate_rope_data(2, 128, 16, 4, 128, rotary_dim=rotary_dim)

    q_ref, k_ref = apply_rope(q.clone(), k.clone(), cos, sin, inplace=False, backend="torch")
    q_tri, k_tri = apply_rope(q.clone(), k.clone(), cos, sin, inplace=False, backend="triton")

    assert torch.allclose(q_tri, q_ref, atol=1e-2, rtol=1e-2)
    assert torch.allclose(k_tri, k_ref, atol=1e-2, rtol=1e-2)
    assert torch.equal(q_tri[..., rotary_dim:], q[..., rotary_dim:])
    assert torch.equal(k_tri[..., rotary_dim:], k[..., rotary_dim:])


@pytest.mark.parametrize("leading_dims", [(), (3,), (2, 3)])
def test_rope_arbitrary_leading_dims(leading_dims):
    seq_len, num_q_heads, num_kv_heads, head_dim = 64, 16, 4, 128
    q = torch.randn(*leading_dims, seq_len, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(*leading_dims, seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    cos, sin = _precompute_rope_cache(seq_len, head_dim, dtype=torch.bfloat16)
    cos = cos.squeeze(0).expand(*leading_dims, -1, -1)
    sin = sin.squeeze(0).expand(*leading_dims, -1, -1)

    q_ref, k_ref = apply_rope(q.clone(), k.clone(), cos, sin, inplace=False, backend="torch")
    q_tri, k_tri = apply_rope(q.clone(), k.clone(), cos, sin, inplace=False, backend="triton")

    assert torch.allclose(q_tri, q_ref, atol=1e-2, rtol=1e-2)
    assert torch.allclose(k_tri, k_ref, atol=1e-2, rtol=1e-2)
