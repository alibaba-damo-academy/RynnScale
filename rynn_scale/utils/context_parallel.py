from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)

from . import logging
from .. import parallel_state as mpu
from ..ops import all_to_all

logger = logging.get_logger(__name__)


class _GatherSequenceFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        sizes: Optional[List[int]] = None,
        dim: int = 1,
        group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> torch.Tensor:
        world_size = torch.distributed.get_world_size(group)
        rank = torch.distributed.get_rank(group)

        if sizes is None:
            outputs = [
                tensor if i == rank else torch.empty_like(tensor)
                for i in range(world_size)
            ]
        else:
            assert len(sizes) == world_size
            outputs = []
            for i, size in enumerate(sizes):
                if i == rank:
                    assert size == tensor.size(dim)
                    outputs.append(tensor)
                else:
                    shape = list(tensor.shape)
                    shape[dim] = size
                    outputs.append(tensor.new_empty(shape))

        torch.distributed.all_gather(
            tensor_list=outputs,
            tensor=tensor,
            group=group,
        )

        ctx.sizes = sizes
        ctx.dim = dim
        ctx.group = group
        ctx.world_size = world_size
        ctx.rank = rank

        return torch.cat(outputs, dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.sizes is None:
            return grad_output.chunk(ctx.world_size, dim=ctx.dim)[ctx.rank], None, None, None
        return grad_output.split(ctx.sizes, dim=ctx.dim)[ctx.rank], None, None, None


class EncoderContextDispatcher(object):
    def __init__(
        self,
        grid_thw: torch.Tensor,
        merge_size: int = 1,
    ):
        self.encoder_cp_group = mpu.get_encoder_context_parallel_group()
        self.encoder_cp_size = mpu.get_encoder_context_parallel_world_size()
        self.encoder_cp_rank = mpu.get_encoder_context_parallel_rank()

        self.cp_group = mpu.get_context_parallel_group()
        self.cp_size = mpu.get_context_parallel_world_size()
        self.cp_rank = mpu.get_context_parallel_rank()

        assert self.encoder_cp_size >= self.cp_size

        self.cp_input_split_sizes = None
        self.cp_final_output_split_sizes = None
        self.input_split_sizes = None
        self.output_split_sizes = None
        self.final_input_split_sizes = None
        self.final_output_split_sizes = None
        self.cu_seqlens = None
        merge_factor = merge_size**2

        if self.encoder_cp_size <= 1:
            return

        num_tokens = grid_thw[:, 1:].prod(dim=1).repeat_interleave(grid_thw[:, 0])

        if self.cp_size > 1 and len(num_tokens) < self.cp_size:
            logger.warning(f"Number of frames ({len(num_tokens)}) < context parallel size ({self.cp_size}). This will lead to redundant calculations and a decrease in speed.")

        if self.cp_size > 1 and len(num_tokens) >= self.cp_size:
            src_group_ids = [self.cp_rank] * len(num_tokens)
            frame_split_sizes, input_split_sizes, _, cu_seqlens = self._minimax_sum_split(
                num_tokens.tolist(),
                src_group_ids,
                cp_size=self.cp_size,
                cp_rank=self.cp_rank,
            )
            num_tokens = num_tokens.split(frame_split_sizes, dim=0)[self.cp_rank]
            self.cp_input_split_sizes = input_split_sizes
            self.cp_final_output_split_sizes = [x // merge_factor for x in input_split_sizes]
            self.cu_seqlens = cu_seqlens

        if self.encoder_cp_size == self.cp_size:
            return

        num_frames_ranks = [num_tokens.new_empty(1) for _ in range(self.encoder_cp_size)]
        num_frames = num_tokens.new_ones(1) * len(num_tokens)
        torch.distributed.all_gather(
            num_frames_ranks,
            num_frames,
            group=self.encoder_cp_group,
        )
        num_frames_ranks = [x.item() for x in num_frames_ranks]
        src_group_ids = [i for i, n in enumerate(num_frames_ranks) for _ in range(n)]

        if len(src_group_ids) <= self.encoder_cp_size:
            return

        num_tokens_ranks = [num_tokens.new_empty(n) for n in num_frames_ranks]
        torch.distributed.all_gather(num_tokens_ranks, num_tokens, group=self.encoder_cp_group)
        num_tokens_all = torch.cat(num_tokens_ranks).tolist()

        _, input_split_sizes, output_split_sizes, cu_seqlens = self._minimax_sum_split(
            num_tokens_all,
            src_group_ids,
            cp_size=self.encoder_cp_size,
            cp_rank=self.encoder_cp_rank,
        )

        self.input_split_sizes = input_split_sizes
        self.output_split_sizes = output_split_sizes
        self.final_input_split_sizes = [x // merge_factor for x in output_split_sizes]
        self.final_output_split_sizes = [x // merge_factor for x in input_split_sizes]
        self.cu_seqlens = cu_seqlens

    @staticmethod
    def _minimax_sum_split(
        num_tokens: List[int],
        src_group_ids: List[int],
        cp_size: int,
        cp_rank: int,
    ):
        assert cp_size <= len(num_tokens)

        def can_split(max_s):
            splits = 1
            current_sum = 0
            for num in num_tokens:
                if current_sum + num > max_s:
                    splits += 1
                    current_sum = num
                else:
                    current_sum += num
            return splits <= cp_size

        left = max(num_tokens)
        right = sum(num_tokens)

        while left < right:
            mid = (left + right) // 2
            if can_split(mid):
                right = mid
            else:
                left = mid + 1

        limit = left

        frame_split_sizes = [0] * cp_size
        input_split_sizes = [0] * cp_size
        output_split_sizes = [0] * cp_size
        cu_seqlens = [0]

        tgt_group_id = 0
        current_sum = 0

        for i, (num, src_group_id) in enumerate(zip(num_tokens, src_group_ids)):
            if current_sum + num > limit and tgt_group_id < cp_size - 1:
                tgt_group_id += 1
                current_sum = num
            else:
                current_sum += num

            if src_group_id == cp_rank:
                input_split_sizes[tgt_group_id] += num
                frame_split_sizes[tgt_group_id] += 1
            if tgt_group_id == cp_rank:
                output_split_sizes[src_group_id] += num
                cu_seqlens.append(cu_seqlens[-1] + num)

            remaining_items = len(num_tokens) - 1 - i
            remaining_groups = cp_size - 1 - tgt_group_id
            if remaining_items == remaining_groups and remaining_groups > 0:
                tgt_group_id += 1
                current_sum = 0

        return frame_split_sizes, input_split_sizes, output_split_sizes, cu_seqlens

    def dispatch(self, hidden_states: torch.Tensor):
        if self.cp_input_split_sizes is not None:
            hidden_states = hidden_states.split(self.cp_input_split_sizes, dim=0)[self.cp_rank]
        if self.input_split_sizes is not None:
            hidden_states = all_to_all(
                hidden_states,
                self.output_split_sizes,
                self.input_split_sizes,
                self.encoder_cp_group,
            )
        return hidden_states

    def combine(self, hidden_states: torch.Tensor):
        if self.input_split_sizes is not None:
            hidden_states = all_to_all(
                hidden_states,
                self.final_output_split_sizes,
                self.final_input_split_sizes,
                self.encoder_cp_group,
            )
        if self.cp_input_split_sizes is not None:
            hidden_states = _GatherSequenceFunction.apply(
                hidden_states,
                self.cp_final_output_split_sizes,
                0,
                self.cp_group,
            )
        return hidden_states


class _AllToAllFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs) -> torch.Tensor:
        group = mpu.get_context_parallel_group()
        inputs = [x.contiguous() for x in inputs]
        outputs = [torch.empty_like(x) for x in inputs]
        torch.distributed.all_to_all(
            output_tensor_list=outputs,
            input_tensor_list=inputs,
            group=group,
        )
        ctx.group = group
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_outputs = [x.contiguous() for x in grad_outputs]
        grad_inputs = [torch.empty_like(x) for x in grad_outputs]
        torch.distributed.all_to_all(
            output_tensor_list=grad_inputs,
            input_tensor_list=grad_outputs,
            group=ctx.group,
        )
        return tuple(grad_inputs)


def ulysses_preprocess(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size == 1:
        return q, k, v

    num_q_heads = q.size(2) // cp_size
    num_k_heads = k.size(2) // cp_size
    num_v_heads = v.size(2) // cp_size

    q = q.chunk(cp_size, dim=2)
    k = k.chunk(cp_size, dim=2)
    v = v.chunk(cp_size, dim=2)

    inputs = [torch.cat(x, dim=2) for x in zip(q, k, v)]
    outputs = _AllToAllFunction.apply(*inputs)

    output = torch.cat(outputs, dim=1)
    q, k, v = output.split([num_q_heads, num_k_heads, num_v_heads], dim=2)

    return q, k, v


def ulysses_preprocess_single(
    x: torch.Tensor,
):
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size == 1:
        return x
    inputs = x.chunk(cp_size, dim=2)
    outputs = _AllToAllFunction.apply(*inputs)
    return torch.cat(outputs, dim=1)


def ulysses_postprocess(attn_output: torch.Tensor):
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size == 1:
        return attn_output
    outputs = attn_output.chunk(cp_size, dim=1)
    outputs = _AllToAllFunction.apply(*outputs)
    return torch.cat(outputs, dim=2)


def _update_out_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    q_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_out = block_out.to(block_lse.dtype)
    block_lse = block_lse.transpose(0, 1).unsqueeze(dim=-1)
    if out is None and lse is None:
        return block_out, block_lse
    if q_indices is None:
        out = out - F.sigmoid(block_lse - lse) * (out - block_out)
        lse = lse - F.logsigmoid(lse - block_lse)
    else:
        tmp_out, tmp_lse = out[q_indices], lse[q_indices]
        out[q_indices] = tmp_out - F.sigmoid(block_lse - tmp_lse) * (tmp_out - block_out)
        lse[q_indices] = tmp_lse - F.logsigmoid(tmp_lse - block_lse)
    return out, lse


def _update_grad(
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    block_dq: torch.Tensor,
    block_dk: torch.Tensor,
    block_dv: torch.Tensor,
    q_indices: torch.Tensor,
    k_indices: torch.Tensor,
):
    if dq is None and dk is None and dv is None:
        return block_dq.float(), block_dk.float(), block_dv.float()

    if block_dq.size(0) == dq.size(0):
        dq += block_dq
    else:
        dq[q_indices] += block_dq

    if block_dk.size(0) == dk.size(0):
        dk += block_dk
        dv += block_dv
    else:
        dk[k_indices] += block_dk
        dv[k_indices] += block_dv

    return dq, dk, dv


def _batch_isend_irecv(
    k: torch.Tensor,
    v: torch.Tensor,
    k_recv: torch.Tensor,
    v_recv: torch.Tensor,
    send_rank: int,
    recv_rank: int,
    group: Optional[torch.distributed.ProcessGroup],
    works: List[torch.distributed.Work],
):
    assert len(works) == 0
    p2p_ops = [
        torch.distributed.P2POp(
            torch.distributed.isend,
            k,
            group=group,
            group_peer=send_rank,
        ),
        torch.distributed.P2POp(
            torch.distributed.isend,
            v,
            group=group,
            group_peer=send_rank,
        ),
        torch.distributed.P2POp(
            torch.distributed.irecv,
            k_recv,
            group=group,
            group_peer=recv_rank,
        ),
        torch.distributed.P2POp(
            torch.distributed.irecv,
            v_recv,
            group=group,
            group_peer=recv_rank,
        ),
    ]
    works.extend(torch.distributed.batch_isend_irecv(p2p_ops))


def _wait(
    k: torch.Tensor,
    v: torch.Tensor,
    k_recv: torch.Tensor,
    v_recv: torch.Tensor,
    works: List[torch.distributed.Work],
):
    while len(works) > 0:
        work = works.pop()
        work.wait()
    return k_recv, v_recv, k, v


class _RingAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        cu_seqlens_q_half: Optional[torch.Tensor],
        cu_seqlens_k_half: Optional[torch.Tensor],
        max_seqlen_q_half: Optional[int],
        max_seqlen_k_half: Optional[int],
        q_second_half_indices: Optional[torch.Tensor],
        k_first_half_indices: Optional[torch.Tensor],
        softmax_scale: Optional[float],
        causal: bool,
        group: Optional[torch.distributed.ProcessGroup],
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        if cu_seqlens_q_half is None:
            cu_seqlens_q_half = cu_seqlens_q // 2
        if cu_seqlens_k_half is None:
            cu_seqlens_k_half = cu_seqlens_k // 2
        if max_seqlen_q_half is None:
            max_seqlen_q_half = max_seqlen_q // 2
        if max_seqlen_k_half is None:
            max_seqlen_k_half = max_seqlen_k // 2
        if q_second_half_indices is None:
            q_second_half_indices = torch.cat([
                torch.arange(cu_seqlens_q[i], cu_seqlens_q[i + 1], device=q.device, dtype=torch.long)
                for i in range(len(cu_seqlens_q) - 1)
            ]) + cu_seqlens_q[:-1]
        if k_first_half_indices is None:
            k_first_half_indices = torch.cat([
                torch.arange(cu_seqlens_k[i], cu_seqlens_k[i + 1], device=k.device, dtype=torch.long)
                for i in range(len(cu_seqlens_k) - 1)
            ])

        world_size = torch.distributed.get_world_size(group)
        rank = torch.distributed.get_rank(group)
        send_rank = (rank + 1) % world_size
        recv_rank = (rank - 1) % world_size

        k_local, v_local = k.contiguous(), v.contiguous()
        k, v = torch.empty_like(k_local), torch.empty_like(v_local)
        k_recv, v_recv = k_local.clone(), v_local.clone()

        output, lse, q_second_half = None, None, None
        if rank < world_size - 1:
            q_second_half = q[q_second_half_indices]

        works = []

        for i in range(world_size):
            k, v, k_recv, v_recv = _wait(k, v, k_recv, v_recv, works)
            if i < world_size - 1:
                _batch_isend_irecv(k, v, k_recv, v_recv, send_rank, recv_rank, group, works)

            if i == 0:
                block_output, block_lse, _, _ = _flash_attn_varlen_forward(
                    q=q,
                    k=k,
                    v=v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    dropout_p=0.0,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
                q_indices = None
            elif i <= rank:
                block_output, block_lse, _, _ = _flash_attn_varlen_forward(
                    q=q,
                    k=k[k_first_half_indices],
                    v=v[k_first_half_indices],
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k_half,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k_half,
                    dropout_p=0.0,
                    softmax_scale=softmax_scale,
                    causal=False,
                )
                q_indices = None
            else:
                block_output, block_lse, _, _ = _flash_attn_varlen_forward(
                    q=q_second_half,
                    k=k,
                    v=v,
                    cu_seqlens_q=cu_seqlens_q_half,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q_half,
                    max_seqlen_k=max_seqlen_k,
                    dropout_p=0.0,
                    softmax_scale=softmax_scale,
                    causal=False,
                )
                q_indices = q_second_half_indices

            output, lse = _update_out_lse(output, lse, block_output, block_lse, q_indices)

        output = output.to(q.dtype)
        lse = lse.transpose(0, 1).squeeze(dim=-1)

        ctx.save_for_backward(q, k_local, v_local, output, lse, cu_seqlens_q, cu_seqlens_k, q_second_half_indices, k_first_half_indices, cu_seqlens_q_half, cu_seqlens_k_half)
        ctx.group = group
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.send_rank = send_rank
        ctx.recv_rank = recv_rank
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.max_seqlen_q_half = max_seqlen_q_half
        ctx.max_seqlen_k_half = max_seqlen_k_half

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):

        q, k_local, v_local, output, lse, cu_seqlens_q, cu_seqlens_k, q_second_half_indices, k_first_half_indices, cu_seqlens_q_half, cu_seqlens_k_half = ctx.saved_tensors

        k, v = torch.empty_like(k_local), torch.empty_like(v_local)
        k_recv, v_recv = k_local.clone(), v_local.clone()

        dq = None
        dq_tmp = torch.empty_like(q)

        dk, dv = torch.empty_like(k_local, dtype=torch.float32), torch.empty_like(v_local, dtype=torch.float32)
        dk_recv, dv_recv = None, None
        dk_tmp, dv_tmp = torch.empty_like(k_local), torch.empty_like(v_local)

        q_second_half = None
        if ctx.rank < ctx.world_size - 1:
            q_second_half = q[q_second_half_indices]

        works, grad_works = [], []

        for i in range(ctx.world_size):
            k, v, k_recv, v_recv = _wait(k, v, k_recv, v_recv, works)
            if i < ctx.world_size - 1:
                _batch_isend_irecv(k, v, k_recv, v_recv, ctx.send_rank, ctx.recv_rank, ctx.group, works)

            if i == 0:
                _flash_attn_varlen_backward(
                    dout=grad_output,
                    q=q,
                    k=k,
                    v=v,
                    out=output,
                    softmax_lse=lse,
                    dq=dq_tmp,
                    dk=dk_tmp,
                    dv=dv_tmp,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=ctx.max_seqlen_q,
                    max_seqlen_k=ctx.max_seqlen_k,
                    dropout_p=0.0,
                    softmax_scale=ctx.softmax_scale,
                    causal=ctx.causal,
                    window_size_left=-1,
                    window_size_right=-1,
                    softcap=0.0,
                    alibi_slopes=None,
                    deterministic=False,
                )
                block_dq, block_dk, block_dv = dq_tmp, dk_tmp, dv_tmp
            elif i <= ctx.rank:
                block_dq = torch.empty_like(q)
                block_dk = k.new_empty(k.size(0) // 2, *k.shape[1:])
                block_dv = v.new_empty(v.size(0) // 2, *v.shape[1:])
                _flash_attn_varlen_backward(
                    dout=grad_output,
                    q=q,
                    k=k[k_first_half_indices],
                    v=v[k_first_half_indices],
                    out=output,
                    softmax_lse=lse,
                    dq=dq_tmp,
                    dk=dk_tmp[:k.size(0) // 2],
                    dv=dv_tmp[:k.size(0) // 2],
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k_half,
                    max_seqlen_q=ctx.max_seqlen_q,
                    max_seqlen_k=ctx.max_seqlen_k_half,
                    dropout_p=0.0,
                    softmax_scale=ctx.softmax_scale,
                    causal=False,
                    window_size_left=-1,
                    window_size_right=-1,
                    softcap=0.0,
                    alibi_slopes=None,
                    deterministic=False,
                )
                block_dq = dq_tmp
                block_dk, block_dv = dk_tmp[:k.size(0) // 2], dv_tmp[:k.size(0) // 2]
            else:
                _flash_attn_varlen_backward(
                    dout=grad_output[q_second_half_indices],
                    q=q_second_half,
                    k=k,
                    v=v,
                    out=output[q_second_half_indices],
                    softmax_lse=lse[:, q_second_half_indices],
                    dq=dq_tmp[:q.size(0) // 2],
                    dk=dk_tmp,
                    dv=dv_tmp,
                    cu_seqlens_q=cu_seqlens_q_half,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=ctx.max_seqlen_q_half,
                    max_seqlen_k=ctx.max_seqlen_k,
                    dropout_p=0.0,
                    softmax_scale=ctx.softmax_scale,
                    causal=False,
                    window_size_left=-1,
                    window_size_right=-1,
                    softcap=0.0,
                    alibi_slopes=None,
                    deterministic=False,
                )
                block_dq = dq_tmp[:q.size(0) // 2]
                block_dk, block_dv = dk_tmp, dv_tmp

            dk, dv, dk_recv, dv_recv = _wait(dk, dv, dk_recv, dv_recv, grad_works)
            dq, dk, dv = _update_grad(dq, dk, dv, block_dq, block_dk, block_dv, q_second_half_indices, k_first_half_indices)
            _batch_isend_irecv(dk, dv, dk_recv, dv_recv, ctx.recv_rank, ctx.send_rank, ctx.group, grad_works)

        dk, dv, dk_recv, dv_recv = _wait(dk, dv, dk_recv, dv_recv, grad_works)

        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None


def ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    cu_seqlens_q_half: Optional[torch.Tensor] = None,
    cu_seqlens_k_half: Optional[torch.Tensor] = None,
    max_seqlen_q_half: Optional[int] = None,
    max_seqlen_k_half: Optional[int] = None,
    q_second_half_indices: Optional[torch.Tensor] = None,
    k_first_half_indices: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    attn_implementation: str = "flash_attention_2",
):
    assert attn_implementation == "flash_attention_2"

    output = _RingAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        cu_seqlens_q_half,
        cu_seqlens_k_half,
        max_seqlen_q_half,
        max_seqlen_k_half,
        q_second_half_indices,
        k_first_half_indices,
        softmax_scale,
        causal,
        mpu.get_context_parallel_group(),
    )

    return output


def pad_sequence(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    pad_token_id: int = 0,
):
    cp_size = mpu.get_context_parallel_world_size()
    seq_len = input_ids.size(1)

    if cp_size > 1 and seq_len % (cp_size * 2) != 0:
        padding = cp_size * 2 - seq_len % (cp_size * 2)
        input_ids = F.pad(input_ids, (0, padding), value=pad_token_id)

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = F.pad(attention_mask, (0, padding), value=0)
            elif attention_mask.ndim == 4:
                attention_mask = F.pad(attention_mask, (0, padding, 0, padding), value=0)
            else:
                raise ValueError

        if position_ids is not None:
            position_ids = F.pad(position_ids, (0, padding), value=1)

        if labels is not None:
            labels = F.pad(labels, (0, padding), value=-100)

    return input_ids, attention_mask, position_ids, labels


def get_sequence_splitter(
    sequence_length: int,
    attn_kwargs: Optional[Dict[str, Any]] = None,
    backend: str = "ulysses",
):
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    if cp_size <= 1:
        return slice(None), attn_kwargs

    assert sequence_length % (cp_size * 2) == 0
    attn_kwargs = attn_kwargs.copy()

    if backend == "ulysses" or "cu_seq_lens_q" not in attn_kwargs:
        local_seq_len = sequence_length // cp_size
        sequence_splitter = slice(cp_rank * local_seq_len, (cp_rank + 1) * local_seq_len)
    elif backend == "ring":
        raise NotImplementedError
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return sequence_splitter, attn_kwargs


def get_multimodal_splitter(
    multimodal_mask: torch.Tensor,
    sequence_splitter: Union[slice, torch.LongTensor],
):
    local_multimodal_mask = torch.zeros_like(multimodal_mask)
    local_multimodal_mask[:, sequence_splitter] = multimodal_mask[:, sequence_splitter]
    valid_mask = local_multimodal_mask[multimodal_mask]
    return valid_mask
