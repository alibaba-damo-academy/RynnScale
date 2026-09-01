import pytest
import torch

from rynn_scale import parallel_state as mpu
from rynn_scale.utils.context_parallel import (
    EncoderContextDispatcher,
    ulysses_postprocess,
    ulysses_preprocess,
    ulysses_preprocess_single,
)

# ---------------------------------------------------------------------------
# Non-distributed tests for EncoderContextDispatcher._minimax_sum_split
# ---------------------------------------------------------------------------


def test_minimax_sum_split_equal_tokens():
    """Equal tokens per frame should be split one frame per rank."""
    num_tokens = [100, 100, 100, 100]

    for cp_rank in range(4):
        # Matches real usage: src_group_ids = [self.cp_rank] * len(num_tokens)
        src_group_ids = [cp_rank] * 4
        frame_split, input_split, output_split, cu_seqlens = EncoderContextDispatcher._minimax_sum_split(
            num_tokens,
            src_group_ids,
            cp_size=4,
            cp_rank=cp_rank,
        )
        assert frame_split == [1, 1, 1, 1]
        assert input_split == [100, 100, 100, 100]
        assert cu_seqlens[0] == 0
        assert cu_seqlens[-1] == 100


def test_minimax_sum_split_unequal_tokens():
    """A large frame should get its own rank; small frames are grouped."""
    num_tokens = [300, 100, 100, 100]
    src_group_ids = [0, 0, 0, 0]

    frame_split, input_split, _, _ = EncoderContextDispatcher._minimax_sum_split(
        num_tokens,
        src_group_ids,
        cp_size=2,
        cp_rank=0,
    )
    # rank 0: [300], rank 1: [100, 100, 100]
    assert frame_split[0] == 1
    assert frame_split[1] == 3
    assert input_split[0] == 300
    assert input_split[1] == 300


def test_minimax_sum_split_cu_seqlens_monotonic():
    """cu_seqlens should be strictly increasing."""
    num_tokens = [10, 20, 30, 40, 50, 60]
    src_group_ids = [0, 0, 0, 0, 0, 0]

    for cp_rank in range(3):
        _, _, _, cu_seqlens = EncoderContextDispatcher._minimax_sum_split(
            num_tokens,
            src_group_ids,
            cp_size=3,
            cp_rank=cp_rank,
        )
        assert cu_seqlens[0] == 0
        for i in range(1, len(cu_seqlens)):
            assert cu_seqlens[i] > cu_seqlens[i - 1]


def test_minimax_sum_split_multiple_src_groups():
    """input_split should only count tokens belonging to the queried rank."""
    num_tokens = [100, 200, 150, 50]
    src_group_ids = [0, 0, 1, 1]

    _, input_split_0, _, _ = EncoderContextDispatcher._minimax_sum_split(
        num_tokens,
        src_group_ids,
        cp_size=2,
        cp_rank=0,
    )
    _, input_split_1, _, _ = EncoderContextDispatcher._minimax_sum_split(
        num_tokens,
        src_group_ids,
        cp_size=2,
        cp_rank=1,
    )
    assert sum(input_split_0) == 300  # src 0 tokens: 100 + 200
    assert sum(input_split_1) == 200  # src 1 tokens: 150 + 50


def test_minimax_sum_split_frames_equals_ranks():
    """When #frames == cp_size, each rank gets exactly one frame."""
    num_tokens = [50, 60, 70, 80]
    src_group_ids = [0, 0, 0, 0]

    frame_split, _, _, _ = EncoderContextDispatcher._minimax_sum_split(
        num_tokens,
        src_group_ids,
        cp_size=4,
        cp_rank=0,
    )
    assert all(f == 1 for f in frame_split)


def test_minimax_sum_split_output_coverage():
    """Sum of output_split across all ranks equals total tokens."""
    num_tokens = [15, 25, 35, 45, 55]
    src_group_ids = [0, 0, 0, 0, 0]

    for cp_size in [2, 3, 5]:
        output_totals = []
        for cp_rank in range(cp_size):
            _, _, output_split, _ = EncoderContextDispatcher._minimax_sum_split(
                num_tokens,
                src_group_ids,
                cp_size=cp_size,
                cp_rank=cp_rank,
            )
            output_totals.append(sum(output_split))
        assert sum(output_totals) == sum(num_tokens)


# ---------------------------------------------------------------------------
# Non-distributed tests: cp_size=1 should be a no-op
# ---------------------------------------------------------------------------


def test_ulysses_preprocess_noop():
    q = torch.randn(1, 8, 4, 16)
    k = torch.randn(1, 8, 2, 16)
    v = torch.randn(1, 8, 2, 16)
    q_out, k_out, v_out = ulysses_preprocess(q, k, v)
    assert q_out is q
    assert k_out is k
    assert v_out is v


def test_ulysses_preprocess_single_noop():
    x = torch.randn(1, 8, 4, 16)
    x_out = ulysses_preprocess_single(x)
    assert x_out is x


def test_ulysses_postprocess_noop():
    x = torch.randn(1, 8, 4, 16)
    x_out = ulysses_postprocess(x)
    assert x_out is x


# ---------------------------------------------------------------------------
# Distributed tests
# ---------------------------------------------------------------------------


@pytest.mark.distributed(world_size=4)
def test_encoder_context_dispatcher(
    hidden_size: int = 16,
    dtype: torch.dtype = torch.bfloat16,
):
    """Dispatch + combine with merge_size=1 should be a round-trip identity."""
    mpu.initialize_model_parallel(
        context_parallel_size=torch.distributed.get_world_size(),
        encoder_context_parallel_size=torch.distributed.get_world_size(),
    )

    cp_rank = mpu.get_context_parallel_rank()

    # 4 frames of 4x4=16 tokens each, total 64 tokens
    grid_thw = torch.tensor([[4, 4, 4]], device="cuda")
    dispatcher = EncoderContextDispatcher(grid_thw, merge_size=1)

    total_tokens = 64
    global_hidden = (
        torch.arange(
            total_tokens,
            dtype=dtype,
            device="cuda",
        )[:, None]
        .expand(total_tokens, hidden_size)
        .contiguous()
    )

    # Dispatch: each rank gets its assigned tokens
    local_hidden = dispatcher.dispatch(global_hidden.clone())
    assert local_hidden.shape[0] == dispatcher.cp_input_split_sizes[cp_rank]

    # Combine: all_gather back to full tensor
    recovered = dispatcher.combine(local_hidden)
    assert recovered.shape == global_hidden.shape
    assert torch.all(recovered == global_hidden)

    # cu_seqlens sanity check
    assert dispatcher.cu_seqlens is not None
    assert dispatcher.cu_seqlens[0] == 0
    assert len(dispatcher.cu_seqlens) > 1


@pytest.mark.distributed(world_size=4)
def test_encoder_context_dispatcher_unequal_frames(
    hidden_size: int = 16,
    dtype: torch.dtype = torch.bfloat16,
):
    """Dispatch + combine with frames of different spatial sizes."""
    mpu.initialize_model_parallel(
        context_parallel_size=torch.distributed.get_world_size(),
        encoder_context_parallel_size=torch.distributed.get_world_size(),
    )

    cp_rank = mpu.get_context_parallel_rank()

    # 2 videos: first has 2 frames of 4x4=16, second has 2 frames of 2x4=8
    # num_tokens = [16, 16, 8, 8], total = 48
    grid_thw = torch.tensor([[2, 4, 4], [2, 2, 4]], device="cuda")
    dispatcher = EncoderContextDispatcher(grid_thw, merge_size=1)

    total_tokens = 48
    global_hidden = (
        torch.arange(
            total_tokens,
            dtype=dtype,
            device="cuda",
        )[:, None]
        .expand(total_tokens, hidden_size)
        .contiguous()
    )

    local_hidden = dispatcher.dispatch(global_hidden.clone())
    recovered = dispatcher.combine(local_hidden)
    assert recovered.shape == global_hidden.shape
    assert torch.all(recovered == global_hidden)


@pytest.mark.distributed(world_size=4)
def test_ulysses_preprocess(
    batch_size: int = 1,
    sequence_length: int = 16,
    num_q_heads: int = 8,
    num_kv_heads: int = 4,
    head_dim: int = 8,
    dtype: torch.dtype = torch.bfloat16,
):
    """preprocess converts (B, S_local, H, D) -> (B, S, H/cp, D)."""
    mpu.initialize_model_parallel(
        context_parallel_size=torch.distributed.get_world_size(),
        encoder_context_parallel_size=torch.distributed.get_world_size(),
    )

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    local_seq_len = sequence_length // cp_size
    num_q_local = num_q_heads // cp_size
    num_kv_local = num_kv_heads // cp_size

    # Build global tensors: value = seq_pos * 100 + head_idx
    seq_idx = torch.arange(sequence_length, dtype=dtype, device="cuda")
    q_head_idx = torch.arange(num_q_heads, dtype=dtype, device="cuda")
    kv_head_idx = torch.arange(num_kv_heads, dtype=dtype, device="cuda")

    q_global = (
        (seq_idx[None, :, None, None] * 100 + q_head_idx[None, None, :, None])
        .expand(batch_size, sequence_length, num_q_heads, head_dim)
        .contiguous()
    )

    k_global = (
        (seq_idx[None, :, None, None] * 100 + kv_head_idx[None, None, :, None])
        .expand(batch_size, sequence_length, num_kv_heads, head_dim)
        .contiguous()
    )

    v_global = (
        (seq_idx[None, :, None, None] * 100 + kv_head_idx[None, None, :, None] + 1000)
        .expand(batch_size, sequence_length, num_kv_heads, head_dim)
        .contiguous()
    )

    # Each rank holds its sequence shard
    q_local = q_global[:, cp_rank * local_seq_len : (cp_rank + 1) * local_seq_len].clone()
    k_local = k_global[:, cp_rank * local_seq_len : (cp_rank + 1) * local_seq_len].clone()
    v_local = v_global[:, cp_rank * local_seq_len : (cp_rank + 1) * local_seq_len].clone()

    q_out, k_out, v_out = ulysses_preprocess(q_local, k_local, v_local)

    # Shape: full sequence, local heads
    assert q_out.shape == (batch_size, sequence_length, num_q_local, head_dim)
    assert k_out.shape == (batch_size, sequence_length, num_kv_local, head_dim)
    assert v_out.shape == (batch_size, sequence_length, num_kv_local, head_dim)

    # Content: rank r gets heads [r*local : (r+1)*local] over the full sequence
    expected_q = q_global[:, :, cp_rank * num_q_local : (cp_rank + 1) * num_q_local]
    expected_k = k_global[:, :, cp_rank * num_kv_local : (cp_rank + 1) * num_kv_local]
    expected_v = v_global[:, :, cp_rank * num_kv_local : (cp_rank + 1) * num_kv_local]

    assert torch.all(q_out == expected_q)
    assert torch.all(k_out == expected_k)
    assert torch.all(v_out == expected_v)


@pytest.mark.distributed(world_size=4)
def test_ulysses_preprocess_single(
    batch_size: int = 1,
    sequence_length: int = 16,
    num_heads: int = 8,
    head_dim: int = 8,
    dtype: torch.dtype = torch.bfloat16,
):
    """preprocess_single converts (B, S_local, H, D) -> (B, S, H/cp, D)."""
    mpu.initialize_model_parallel(
        context_parallel_size=torch.distributed.get_world_size(),
        encoder_context_parallel_size=torch.distributed.get_world_size(),
    )

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    local_seq_len = sequence_length // cp_size
    num_local_heads = num_heads // cp_size

    seq_idx = torch.arange(sequence_length, dtype=dtype, device="cuda")
    head_idx = torch.arange(num_heads, dtype=dtype, device="cuda")

    x_global = (
        (seq_idx[None, :, None, None] * 100 + head_idx[None, None, :, None])
        .expand(batch_size, sequence_length, num_heads, head_dim)
        .contiguous()
    )

    x_local = x_global[:, cp_rank * local_seq_len : (cp_rank + 1) * local_seq_len].clone()

    x_out = ulysses_preprocess_single(x_local)

    assert x_out.shape == (batch_size, sequence_length, num_local_heads, head_dim)

    expected = x_global[:, :, cp_rank * num_local_heads : (cp_rank + 1) * num_local_heads]
    assert torch.all(x_out == expected)


@pytest.mark.distributed(world_size=4)
def test_ulysses_postprocess(
    batch_size: int = 1,
    sequence_length: int = 16,
    num_heads: int = 8,
    head_dim: int = 8,
    dtype: torch.dtype = torch.bfloat16,
):
    """postprocess converts (B, S, H/cp, D) -> (B, S_local, H, D)."""
    mpu.initialize_model_parallel(
        context_parallel_size=torch.distributed.get_world_size(),
        encoder_context_parallel_size=torch.distributed.get_world_size(),
    )

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    local_seq_len = sequence_length // cp_size
    num_local_heads = num_heads // cp_size

    seq_idx = torch.arange(sequence_length, dtype=dtype, device="cuda")
    head_idx = torch.arange(num_heads, dtype=dtype, device="cuda")

    x_global = (
        (seq_idx[None, :, None, None] * 100 + head_idx[None, None, :, None])
        .expand(batch_size, sequence_length, num_heads, head_dim)
        .contiguous()
    )

    # Each rank starts with full sequence, local heads (head-parallel layout)
    x_local = x_global[:, :, cp_rank * num_local_heads : (cp_rank + 1) * num_local_heads].contiguous()

    x_out = ulysses_postprocess(x_local)

    assert x_out.shape == (batch_size, local_seq_len, num_heads, head_dim)

    expected = x_global[:, cp_rank * local_seq_len : (cp_rank + 1) * local_seq_len]
    assert torch.all(x_out == expected)


@pytest.mark.distributed(world_size=4)
def test_ulysses_roundtrip(
    batch_size: int = 1,
    sequence_length: int = 16,
    num_heads: int = 8,
    head_dim: int = 8,
    dtype: torch.dtype = torch.bfloat16,
):
    """preprocess_single followed by postprocess should be an identity."""
    mpu.initialize_model_parallel(
        context_parallel_size=torch.distributed.get_world_size(),
        encoder_context_parallel_size=torch.distributed.get_world_size(),
    )

    cp_size = mpu.get_context_parallel_world_size()
    local_seq_len = sequence_length // cp_size

    x = torch.randn(
        batch_size,
        local_seq_len,
        num_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
    )

    x_pre = ulysses_preprocess_single(x)
    x_post = ulysses_postprocess(x_pre)

    assert torch.all(x_post == x)
