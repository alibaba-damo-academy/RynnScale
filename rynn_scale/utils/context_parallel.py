import torch
from typing import List

from ..ops import all_to_all
from .. import parallel_state as mpu


class EncoderContextDispatcher(object):
    def __init__(
        self,
        grid_thw: torch.Tensor,
        merge_size: int = 1,
    ):
        self.group = mpu.get_encoder_context_parallel_group()
        self.world_size = mpu.get_encoder_context_parallel_world_size()
        self.rank = mpu.get_encoder_context_parallel_rank()

        self.cu_seqlens = None

        self._activated = self.world_size > 1
        if not self._activated:
            return

        num_tokens = grid_thw[:, 1:].prod(dim=1).repeat_interleave(grid_thw[:, 0])
        num_frames_ranks = [num_tokens.new_empty(1) for _ in range(self.world_size)]
        num_frames = num_tokens.new_ones(1) * len(num_tokens)
        torch.distributed.all_gather(
            num_frames_ranks,
            num_frames,
            group=self.group,
        )
        num_frames_ranks = [x.item() for x in num_frames_ranks]
        src_group_ids = [i for i, n in enumerate(num_frames_ranks) for _ in range(n)]

        if len(src_group_ids) <= self.world_size:
            self._activated = False
            return

        num_tokens_ranks = [num_tokens.new_empty(n) for n in num_frames_ranks]
        torch.distributed.all_gather(num_tokens_ranks, num_tokens, group=self.group)
        num_tokens_all = torch.cat(num_tokens_ranks).tolist()

        input_split_sizes, output_split_sizes, cu_seqlens = self._minimax_sum_split(num_tokens_all, src_group_ids)

        merge_factor = merge_size**2
        self.input_split_sizes = input_split_sizes
        self.output_split_sizes = output_split_sizes
        self.final_input_split_sizes = [x // merge_factor for x in output_split_sizes]
        self.final_output_split_sizes = [x // merge_factor for x in input_split_sizes]
        self.cu_seqlens = cu_seqlens

    @property
    def activated(self):
        return self._activated

    def _minimax_sum_split(
        self,
        num_tokens: List[int],
        src_group_ids: List[int],
    ):
        assert self.world_size <= len(num_tokens)

        def can_split(max_s):
            splits = 1
            current_sum = 0
            for num in num_tokens:
                if current_sum + num > max_s:
                    splits += 1
                    current_sum = num
                else:
                    current_sum += num
            return splits <= self.world_size

        left = max(num_tokens)
        right = sum(num_tokens)

        while left < right:
            mid = (left + right) // 2
            if can_split(mid):
                right = mid
            else:
                left = mid + 1

        limit = left

        input_split_sizes = [0] * self.world_size
        output_split_sizes = [0] * self.world_size
        cu_seqlens = [0]

        tgt_group_id = 0
        current_sum = 0

        for i, (num, src_group_id) in enumerate(zip(num_tokens, src_group_ids)):
            if current_sum + num > limit and tgt_group_id < self.world_size - 1:
                tgt_group_id += 1
                current_sum = num
            else:
                current_sum += num

            if src_group_id == self.rank:
                input_split_sizes[tgt_group_id] += num
            if tgt_group_id == self.rank:
                output_split_sizes[src_group_id] += num
                cu_seqlens.append(cu_seqlens[-1] + num)

            remaining_items = len(num_tokens) - 1 - i
            remaining_groups = self.world_size - 1 - tgt_group_id
            if remaining_items == remaining_groups and remaining_groups > 0:
                tgt_group_id += 1
                current_sum = 0

        return input_split_sizes, output_split_sizes, cu_seqlens

    def dispatch(self, hidden_states: torch.Tensor):
        if not self._activated:
            return hidden_states
        hidden_states = all_to_all(
            hidden_states,
            self.output_split_sizes,
            self.input_split_sizes,
            self.group,
        )
        return hidden_states

    def combine(self, hidden_states: torch.Tensor):
        if not self._activated:
            return hidden_states
        hidden_states = all_to_all(
            hidden_states,
            self.final_output_split_sizes,
            self.final_input_split_sizes,
            self.group,
        )
        return hidden_states
