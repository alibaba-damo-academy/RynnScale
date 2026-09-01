import io
import random
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset

from .vla_datasets import BaseVLADataset

_STATS_KEYS = ("mean", "std", "min", "max", "count")


def _is_leaf(value):
    return isinstance(value, dict) and "dim" in value


def _merge_leaf(a, b):
    """Merge two finalized leaves, combining their stats and preserving the
    structural metadata shared between them."""
    n1, n2 = a["count"], b["count"]
    n = n1 + n2

    mean1 = torch.tensor(a["mean"], dtype=torch.float64)
    mean2 = torch.tensor(b["mean"], dtype=torch.float64)
    std1 = torch.tensor(a["std"], dtype=torch.float64)
    std2 = torch.tensor(b["std"], dtype=torch.float64)

    mean = (n1 * mean1 + n2 * mean2) / n
    var = (n1 * (std1**2 + (mean1 - mean) ** 2) + n2 * (std2**2 + (mean2 - mean) ** 2)) / n

    out = {k: v for k, v in a.items() if k not in _STATS_KEYS}
    out["mean"] = mean.tolist()
    out["std"] = var.sqrt().tolist()
    out["min"] = [min(x, y) for x, y in zip(a["min"], b["min"])]
    out["max"] = [max(x, y) for x, y in zip(a["max"], b["max"])]
    out["count"] = n
    return out


def _merge_schema_node(dst, src):
    """Recursively merge ``src`` schema node into ``dst`` in place."""
    for k, v in src.items():
        if k not in dst:
            dst[k] = v
        elif _is_leaf(v):
            assert _is_leaf(dst[k]), f"Type mismatch for '{k}'"
            dst[k] = _merge_leaf(dst[k], v)
        elif isinstance(v, dict):
            assert isinstance(dst[k], dict) and not _is_leaf(dst[k]), f"Type mismatch for '{k}'"
            _merge_schema_node(dst[k], v)
        else:
            assert dst[k] == v, f"Inconsistent schema value for '{k}': {dst[k]!r} vs {v!r}"


def _merge_schemas(schema_list):
    assert len(schema_list) > 0
    merged = schema_list[0]
    for other in schema_list[1:]:
        for robot_type, robot_schema in other.items():
            if robot_type not in merged:
                merged[robot_type] = robot_schema
            else:
                _merge_schema_node(merged[robot_type], robot_schema)
    return merged


class ConcatDataset(torch.utils.data.ConcatDataset):
    def get_schema(
        self,
        num_workers: int = 8,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        per_dataset = [
            dataset.get_schema(num_workers=num_workers, process_group=process_group)
            for dataset in self.datasets
            if hasattr(dataset, "get_schema")
        ]
        if not per_dataset:
            return {"action": {}, "state": {}}
        return {
            "action": _merge_schemas([s["action"] for s in per_dataset]),
            "state": _merge_schemas([s["state"] for s in per_dataset]),
        }

    def __repr__(self) -> str:
        parts = []
        for i, d in enumerate(self.datasets):
            sub = repr(d).replace("\n", "\n    ")
            parts.append(f"    ({i}): {sub}")
        inner = "\n".join(parts)
        return f"{self.__class__.__name__}(\n{inner}\n)"


class StreamingVLADataset(IterableDataset):
    """IterableDataset wrapper over one or more BaseVLADataset instances.

    Reads data via ``iter_episode`` (sequential IO), handles DDP/worker
    sharding, episode interleaving, and reservoir shuffle — all internally.
    All episodes across datasets are flattened into a single list and
    served from one shared episode buffer.
    """

    def __init__(
        self,
        datasets: List[BaseVLADataset],
    ):
        assert len(datasets) > 0

        self.datasets = datasets
        self.epoch = 0

        self._rank: int = 0
        self._world_size: int = 1
        self._shuffle: bool = False
        self._seed: int = 0
        self._episode_buffer_size: int = 1
        self._shuffle_buffer_size: int = 1

        self._all_episodes: List[tuple] = []
        for ds in datasets:
            for ep_idx in range(ds.num_episodes):
                self._all_episodes.append((ds, ep_idx))

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def shard(self, rank: int, world_size: int):
        self._rank = rank
        self._world_size = world_size

    def shuffle(self, seed: int, episode_buffer_size: int = 4, shuffle_buffer_size: int = 1024):
        self._shuffle = True
        self._seed = seed
        self._episode_buffer_size = episode_buffer_size
        self._shuffle_buffer_size = shuffle_buffer_size

    def __len__(self):
        total = sum(ds.episode_lengths[ep_idx] for ds, ep_idx in self._all_episodes)
        return total // self._world_size

    def get_schema(
        self,
        num_workers: int = 8,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        per_dataset = [ds.get_schema(num_workers=num_workers, process_group=process_group) for ds in self.datasets]
        if not per_dataset:
            return {"action": {}, "state": {}}
        return {
            "action": _merge_schemas([s["action"] for s in per_dataset]),
            "state": _merge_schemas([s["state"] for s in per_dataset]),
        }

    def _get_shard_info(self):
        info = torch.utils.data.get_worker_info()
        if info is not None:
            wid, nw = info.id, info.num_workers
        else:
            wid, nw = 0, 1

        shard_id = self._rank * nw + wid
        num_shards = self._world_size * nw
        return shard_id, num_shards

    def _shard_episodes(self, episode_indices: List[int], shard_id: int, num_shards: int) -> List[int]:
        return episode_indices[shard_id::num_shards]

    @staticmethod
    def _encode_images(step: Dict) -> Dict:
        images = step.get("images")
        if images is None:
            return step
        encoded = {}
        for k, img in images.items():
            if isinstance(img, (bytes, bytearray)):
                encoded[k] = bytes(img)
            else:
                buf = io.BytesIO()
                Image.fromarray(img.numpy()).save(buf, format="PNG")
                encoded[k] = buf.getvalue()
        return {**step, "images": encoded}

    @staticmethod
    def _decode_images(step: Dict) -> Dict:
        images = step.get("images")
        if images is None:
            return step
        decoded = {}
        for k, raw in images.items():
            decoded[k] = torch.from_numpy(np.asarray(Image.open(io.BytesIO(raw))))
        return {**step, "images": decoded}

    def __iter__(self) -> Iterator[Dict]:
        shard_id, num_shards = self._get_shard_info()

        if self._shuffle:
            g = torch.Generator().manual_seed(self._seed + self.epoch)
            perm = torch.randperm(len(self._all_episodes), generator=g).tolist()
        else:
            perm = list(range(len(self._all_episodes)))
        my_eps = self._shard_episodes(perm, shard_id, num_shards)

        if not my_eps:
            return

        if self._shuffle:
            rng = random.Random(self._seed + self.epoch + shard_id)
            yield from self._iter_shuffled(my_eps, rng)
        else:
            yield from self._iter_sequential(my_eps)

    def _iter_sequential(self, my_eps: List[int]) -> Iterator[Dict]:
        for ep_global_idx in my_eps:
            ds, ep_idx = self._all_episodes[ep_global_idx]
            for step in ds.iter_episode(ep_idx):
                yield step

    def _iter_shuffled(self, my_eps: List[int], rng: random.Random) -> Iterator[Dict]:
        pool: List[Optional[Iterator[Dict]]] = []
        ep_cursor = 0
        n_eps = len(my_eps)

        fill = min(self._episode_buffer_size, n_eps)
        for _ in range(fill):
            ds, ep_idx = self._all_episodes[my_eps[ep_cursor]]
            ep_cursor += 1
            pool.append(ds.iter_episode(ep_idx))

        buf: List[Dict] = []

        while pool:
            i = rng.randint(0, len(pool) - 1)
            try:
                step = next(pool[i])
            except StopIteration:
                if ep_cursor < n_eps:
                    ds, ep_idx = self._all_episodes[my_eps[ep_cursor]]
                    ep_cursor += 1
                    pool[i] = ds.iter_episode(ep_idx)
                    continue
                else:
                    pool.pop(i)
                    continue

            buf.append(self._encode_images(step))
            if len(buf) >= self._shuffle_buffer_size:
                j = rng.randint(0, len(buf) - 1)
                buf[j], buf[-1] = buf[-1], buf[j]
                yield self._decode_images(buf.pop(0))

        rng.shuffle(buf)
        for step in buf:
            yield self._decode_images(step)
