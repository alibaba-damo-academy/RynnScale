import math
from typing import List

import torch
from torch.utils.data import Dataset, DistributedSampler


class DistributedBatchSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        sequence_lengths: List[int],
        num_replicas: int,
        rank: int,
        micro_batch_size: int,
        gradient_accumulation_steps: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        decoder_load_balancing: bool = False,
        dynamic_batching: bool = False,
        dynamic_batching_window_size: int = 128,
        model_max_length: int = 16384,
    ):
        self.dataset = dataset
        self.sequence_lengths = sequence_lengths
        self.num_replicas = num_replicas
        self.rank = rank
        self.micro_batch_size = micro_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.decoder_load_balancing = decoder_load_balancing
        self.dynamic_batching = dynamic_batching

        self.epoch = 0
        self.num_skipped_batches = 0

        assert not (decoder_load_balancing or dynamic_batching) or sequence_lengths is not None

        if self.dynamic_batching:
            raise NotImplementedError

        else:
            local_batch_size = micro_batch_size * gradient_accumulation_steps
            global_batch_size = local_batch_size * self.num_replicas

            # If the dataset length is evenly divisible by # of replicas, then there
            # is no need to drop any data, since the dataset will be split equally.
            if self.drop_last and len(self.dataset) % global_batch_size != 0:  # type: ignore[arg-type]
                # Split to nearest available length that is evenly divisible.
                # This is to ensure each rank receives the same amount of data when
                # using this Sampler.
                self.num_batches = math.ceil(
                    (len(self.dataset) - global_batch_size) / global_batch_size  # type: ignore[arg-type]
                )
            else:
                self.num_batches = math.ceil(len(self.dataset) / global_batch_size)  # type: ignore[arg-type]

    def _get_global_batch_indices(self):
        local_batch_size = self.micro_batch_size * self.gradient_accumulation_steps
        global_batch_size = local_batch_size * self.num_replicas

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        total_size = self.num_batches * global_batch_size
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:total_size]
        assert len(indices) == total_size

        indices = [indices[i :: self.num_replicas] for i in range(self.num_replicas)]
        batch_indices = [
            [local_indices[i : i + local_batch_size] for i in range(0, len(local_indices), local_batch_size)]
            for local_indices in indices
        ]

        assert len(batch_indices) == self.num_replicas
        assert all(len(local_batch_indices) == self.num_batches for local_batch_indices in batch_indices)

        return batch_indices

    def _longest_first_partition(self, data_indices: List[int]):
        partitions = [[] for _ in range(self.num_replicas)]
        batch_seqlens = [0 for _ in range(self.num_replicas)]

        seqlen_list = [self.sequence_lengths[i] for i in data_indices]
        sorted_seqlen_list = sorted(
            [(seqlen, i) for i, seqlen in enumerate(seqlen_list)],
            key=lambda x: x[0],
            reverse=True,
        )

        for i, (seqlen, idx) in enumerate(sorted_seqlen_list):
            if i < self.num_replicas:
                partition_id = i
            else:
                partition_id = min(list(range(self.num_replicas)), key=lambda x: batch_seqlens[x])
            partitions[partition_id].append(idx)
            batch_seqlens[partition_id] += seqlen

        new_data_indices = [[data_indices[i] for i in batch] for batch in partitions]

        return new_data_indices

    def __iter__(self):
        global_batch_indices = self._get_global_batch_indices()
        for i in range(0, self.num_batches):
            for j in range(self.gradient_accumulation_steps):
                if i * self.gradient_accumulation_steps + j < self.num_skipped_batches:
                    continue
                if self.decoder_load_balancing:
                    all_sample_indices = sum(
                        [
                            global_batch_indices[k][i][j * self.micro_batch_size : (j + 1) * self.micro_batch_size]
                            for k in range(self.num_replicas)
                        ],
                        [],
                    )
                    batch_indices = self._longest_first_partition(all_sample_indices)
                    yield batch_indices[self.rank]
                else:
                    yield global_batch_indices[self.rank][i][j * self.micro_batch_size : (j + 1) * self.micro_batch_size]

    def __len__(self) -> int:
        return self.num_batches * self.gradient_accumulation_steps

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.num_skipped_batches = 0

    def skip_first_batches(self, num_batches: int):
        self.num_skipped_batches = num_batches


if __name__ == "__main__":
    sampler = DistributedBatchSampler()
