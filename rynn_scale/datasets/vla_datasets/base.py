import hashlib
import json
import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import ProcessorMixin

from ...constants import CACHE_DIR, RobotType, RotationRepresentation
from ...utils.logging import get_logger
from ...utils.robot import (
    Arm,
    Position,
    RobotAction,
    RobotState,
    Rotation,
)

logger = get_logger(__name__)


@dataclass
class EpisodeMetadata:
    """Per-episode metadata. Subclasses produce a list of these in `_load_metadata`.

    `length`/`fps` are in source-fps space. `extras` is a free-form dict
    subclasses use to stash their own per-episode locators (file index,
    demo index, shard offset, etc.).
    """

    length: int
    fps: float
    robot_type: RobotType
    extras: Dict[str, Any] = field(default_factory=dict)


class _EpisodeStatsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: "BaseVLADataset"):
        self.dataset = dataset

    def __len__(self) -> int:
        return self.dataset.num_episodes

    def __getitem__(self, episode_index: int):
        return self.dataset._get_episode_schemas(episode_index)


def _identity_collate(batch):
    return batch[0]


_LEAF_META_KEYS = ("type", "is_relative", "allow_relative", "representation", "dim")


def _new_accumulator(tensor: torch.Tensor) -> Dict:
    flat = tensor.reshape(-1, tensor.shape[-1]).float()
    return {
        "sum": flat.sum(0),
        "sum_sq": (flat**2).sum(0),
        "min": flat.amin(0),
        "max": flat.amax(0),
        "count": int(flat.size(0)),
    }


def _merge_accumulator(dst: Dict, src: Dict) -> None:
    dst["sum"] = dst["sum"] + src["sum"]
    dst["sum_sq"] = dst["sum_sq"] + src["sum_sq"]
    dst["min"] = torch.minimum(dst["min"], src["min"])
    dst["max"] = torch.maximum(dst["max"], src["max"])
    dst["count"] = dst["count"] + src["count"]


def _finalize_leaf(leaf: Dict) -> Dict:
    count = leaf["count"]
    mean = leaf["sum"] / count
    if count > 1:
        var = (leaf["sum_sq"] - leaf["sum"] ** 2 / count) / (count - 1)
    else:
        var = torch.zeros_like(mean)
    std = torch.sqrt(var.clamp(min=0))
    out = {mk: leaf[mk] for mk in _LEAF_META_KEYS}
    out["mean"] = mean.tolist()
    out["std"] = std.tolist()
    out["min"] = leaf["min"].tolist()
    out["max"] = leaf["max"].tolist()
    out["count"] = count
    return out


def _leaf_to_cpu(leaf: Dict) -> Dict:
    return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in leaf.items()}


def _is_leaf(value) -> bool:
    return isinstance(value, dict) and "dim" in value


def _atomic_leaf(value) -> Dict:
    """Build a single leaf dict with structure metadata and accumulator stats
    inlined at the same level."""
    if isinstance(value, Position):
        meta = {
            "type": type(value).__name__,
            "is_relative": bool(value.is_relative),
            "allow_relative": bool(value.allow_relative),
            "representation": None,
            "dim": int(value.data.size(-1)),
        }
    elif isinstance(value, Rotation):
        meta = {
            "type": type(value).__name__,
            "is_relative": bool(value.is_relative),
            "allow_relative": bool(value.allow_relative),
            "representation": value.representation.value,
            "dim": int(value.data.size(-1)),
        }
    else:
        raise TypeError(f"Expected Position or Rotation, got {type(value).__name__}")
    return {**meta, **_new_accumulator(value.data)}


def _field_schema(value) -> Dict:
    """Build schema for one top-level RobotAction field."""
    if isinstance(value, (Position, Rotation)):
        return _atomic_leaf(value)
    if isinstance(value, Arm):
        out: Dict = {"type": type(value).__name__}
        for f in fields(value):
            v = getattr(value, f.name)
            if v is None:
                continue
            out[f.name] = _atomic_leaf(v)
        return out
    raise TypeError(f"Unsupported field type: {type(value).__name__}")


def _robot_schema(robot_action: RobotAction) -> Dict:
    """Build a full schema tree for a RobotAction/RobotState."""
    out: Dict = {"type": type(robot_action).__name__}
    for f in fields(robot_action):
        v = getattr(robot_action, f.name)
        if v is None:
            continue
        out[f.name] = _field_schema(v)
    return out


def _merge_schema(dst: Dict, src: Dict, _path: str = "") -> None:
    """Recursively merge ``src`` schema into ``dst`` in place.

    Enforces structural consistency for the same robot type (proposal §5.3):
    field sets and leaf metadata must match across episodes; mismatches are
    raised with the offending path so divergent dataset schemas surface early.
    """
    if set(dst.keys()) != set(src.keys()):
        only_dst = sorted(set(dst.keys()) - set(src.keys()))
        only_src = sorted(set(src.keys()) - set(dst.keys()))
        raise ValueError(
            f"Schema field set mismatch at '{_path or '<root>'}': only_in_existing={only_dst}, only_in_new={only_src}"
        )
    for k in dst:
        d, s = dst[k], src[k]
        path = f"{_path}.{k}" if _path else k
        if k == "type":
            if d != s:
                raise ValueError(f"Schema type mismatch at '{path}': {d!r} != {s!r}")
            continue
        if _is_leaf(d):
            if not _is_leaf(s):
                raise ValueError(f"Schema mismatch at '{path}': existing is a leaf, new is a sub-dict")
            for mk in _LEAF_META_KEYS:
                if d[mk] != s[mk]:
                    raise ValueError(f"Schema metadata mismatch at '{path}.{mk}': {d[mk]!r} != {s[mk]!r}")
            _merge_accumulator(d, s)
        else:
            if _is_leaf(s):
                raise ValueError(f"Schema mismatch at '{path}': existing is a sub-dict, new is a leaf")
            _merge_schema(d, s, _path=path)


def _finalize_schema(schema: Dict) -> Dict:
    out: Dict = {}
    for k, v in schema.items():
        if k == "type":
            out[k] = v
            continue
        if _is_leaf(v):
            out[k] = _finalize_leaf(v)
        else:
            out[k] = _finalize_schema(v)
    return out


def _schema_to_cpu(schema: Dict) -> Dict:
    out: Dict = {}
    for k, v in schema.items():
        if k == "type":
            out[k] = v
            continue
        if _is_leaf(v):
            out[k] = _leaf_to_cpu(v)
        else:
            out[k] = _schema_to_cpu(v)
    return out


class BaseVLADataset(torch.utils.data.Dataset, metaclass=ABCMeta):
    def __init__(
        self,
        data_path: str,
        action_chunk_size: int,
        use_delta_action: bool,
        eef_rotation_repr: Optional[RotationRepresentation] = None,
        target_fps: Optional[float] = None,
        processor: Optional[ProcessorMixin] = None,
        **kwargs,
    ):
        assert action_chunk_size > 0
        assert eef_rotation_repr is None or isinstance(eef_rotation_repr, RotationRepresentation)
        assert target_fps is None or target_fps > 0

        self.data_path = data_path
        self.action_chunk_size = action_chunk_size
        self.use_delta_action = use_delta_action
        self.eef_rotation_repr = eef_rotation_repr
        self.target_fps = target_fps
        self.processor = processor

        self._metadata: List[EpisodeMetadata] = self._load_metadata()

        src_lengths = np.asarray([m.length for m in self._metadata], dtype=np.int64)
        if target_fps is None:
            self._target_lengths = src_lengths
        else:
            target_lengths = np.empty_like(src_lengths)
            for i, n in enumerate(src_lengths.tolist()):
                src_fps = float(self.get_fps(i))
                if abs(src_fps - target_fps) < 1e-6:
                    target_lengths[i] = int(n)
                else:
                    target_lengths[i] = max(1, int(round(n * target_fps / src_fps)))
            self._target_lengths = target_lengths

        self._cum_lengths = np.cumsum(self._target_lengths)

    @abstractmethod
    def _load_metadata(self) -> List[EpisodeMetadata]:
        """Return per-episode metadata, one entry per episode in source-fps space."""
        ...

    @abstractmethod
    def _load_action(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotAction:
        """Load absolute action(s). No delta / no rotation convert / no pad."""
        ...

    @abstractmethod
    def _load_state(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotState:
        """Load absolute state(s). No rotation convert."""
        ...

    @abstractmethod
    def _load_images(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> Dict[str, torch.Tensor]: ...

    @abstractmethod
    def _load_instruction(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> List[str]: ...

    @abstractmethod
    def _iter_episode(
        self,
        episode_index: int,
        source_ranges: List[Tuple[int, int]],
        include_images: bool = True,
    ) -> Iterator[Dict]:
        """Stream raw, absolute step bundles for `episode_index`.

        Each entry in `source_ranges` is a (start, end) pair defining the
        source frame range for one step: state/instruction/images come from
        frame `start`, action spans frames `[start, end)`. The start values
        are guaranteed to be non-decreasing.

        Subclasses MUST:
          - open backing files fresh inside this generator (no shared cache
            with the random-access loaders) and close them on exit (try/finally
            or context manager) including the early-break / exception path
          - yield exactly len(source_ranges) items, in the given order
          - each yielded dict has keys {"state", "action", "instruction",
            "images"}; values are absolute (no rotation convert / no delta /
            no resample / no chunk pad). Base does all post-processing.
          - state is RobotState of length 1; action is RobotAction of length
            `end - start` (NOT padded)
          - when include_images=False, "images" may be None or absent.

        Memory: multiple generators may be alive concurrently (episode_buffer
        in streaming mode). Avoid bulk-loading images for the entire episode;
        prefer per-frame or small-window reads so that in-flight memory stays
        proportional to the number of live generators, not to episode length.
        """
        ...

    # ────────── metadata derived properties ──────────

    @property
    def metadata(self) -> Tuple[EpisodeMetadata, ...]:
        """Read-only per-episode metadata in source-fps space.

        Returned as a tuple so callers can index into it but can't grow or
        shrink the list. Note: per-entry ``extras`` dicts are still mutable —
        treat them as read-only by convention.
        """
        return tuple(self._metadata)

    @property
    def num_episodes(self) -> int:
        return len(self._metadata)

    @property
    def episode_lengths(self) -> List[int]:
        return self._target_lengths.tolist()

    def get_fps(self, episode_index: int) -> float:
        return self._metadata[episode_index].fps

    def get_robot_type(self, episode_index: int) -> RobotType:
        return self._metadata[episode_index].robot_type

    def __len__(self) -> int:
        return int(self._cum_lengths[-1]) if len(self._cum_lengths) > 0 else 0

    def __repr__(self) -> str:
        num_episodes = len(self._metadata)
        num_frames = sum(m.length for m in self._metadata)

        durations = []
        for m in self._metadata:
            durations.append(m.length / m.fps if m.fps > 0 else 0.0)
        total_secs = sum(durations)

        robot_types = sorted({m.robot_type.value for m in self._metadata})
        fps_list = [m.fps for m in self._metadata]

        parts = [
            f"{self.__class__.__name__}(",
            f"    data_path={self.data_path!r},",
            f"    num_episodes={num_episodes},",
            f"    num_frames={num_frames},",
            f"    duration={total_secs / 3600:.1f}h,",
            f"    robot_types={robot_types},",
            f"    mean_fps={sum(fps_list) / len(fps_list):.1f}",
            ")",
        ]
        return "\n".join(parts)

    def _resolve_index(self, index: int) -> Tuple[int, int, int]:
        """Map global index → (episode_index, target_frame_index, target_episode_length)."""
        cum = self._cum_lengths
        episode_index = int(np.searchsorted(cum, index, side="right"))
        episode_start = int(cum[episode_index - 1]) if episode_index > 0 else 0
        episode_end = int(cum[episode_index])
        return episode_index, index - episode_start, episode_end - episode_start

    def _target_to_source_index(self, episode_index: int, target_frame: int) -> int:
        """Nearest-neighbor mapping from target frame to source frame."""
        if self.target_fps is None:
            return target_frame
        src_fps = float(self.get_fps(episode_index))
        if abs(src_fps - self.target_fps) < 1e-6:
            return target_frame
        src_n = self._metadata[episode_index].length
        ratio = src_fps / self.target_fps
        return min(max(0, int(round(target_frame * ratio))), src_n - 1)

    def _action_chunk_source_indices(
        self,
        episode_index: int,
        target_start: int,
        chunk_size: int,
    ) -> Tuple[slice, Optional[Dict]]:
        """Determine source action indices for a target-fps action chunk.

        Given target frame `target_start` and `chunk_size` target action steps,
        computes the time window `chunk_size / target_fps` and collects all
        source frames within that window. Returns the source slice and resample
        kwargs for `action.resample()` to produce exactly `chunk_size` actions.

        Returns ``(src_slice, None)`` when no resampling is required.
        """
        if self.target_fps is None:
            src_n = self._metadata[episode_index].length
            src_end = min(target_start + chunk_size, src_n)
            return slice(target_start, src_end), None
        src_fps = float(self.get_fps(episode_index))
        if abs(src_fps - self.target_fps) < 1e-6:
            src_n = self._metadata[episode_index].length
            src_end = min(target_start + chunk_size, src_n)
            return slice(target_start, src_end), None

        src_n = self._metadata[episode_index].length
        # Time window covered by the action chunk in seconds
        chunk_duration = chunk_size / self.target_fps
        # Source frame corresponding to target_start (nearest)
        src_start = self._target_to_source_index(episode_index, target_start)
        # All source frames within the chunk time window
        src_end = min(src_n, src_start + max(1, int(round(chunk_duration * src_fps))))
        return slice(src_start, src_end), {
            "src_fps": src_fps,
            "tgt_fps": float(self.target_fps),
            "n_target": chunk_size,
        }

    def _postprocess_step(
        self,
        state: RobotState,
        action: RobotAction,
        resample_kwargs: Optional[Dict] = None,
    ) -> Tuple[RobotState, RobotAction]:
        """Single-source-of-truth post-processing for one step.

        Applied identically by ``__getitem__`` (random access) and
        ``iter_episode`` (streaming). Order: resample action to target fps
        → pad to action_chunk_size → rotation convert → delta-action.

        Inputs are absolute (subclass returns absolute action/state). Output
        action has length ``action_chunk_size``; state stays length 1.
        """
        if resample_kwargs is not None:
            action = action.resample(**resample_kwargs)

        if len(action) < self.action_chunk_size:
            action = action.pad_to(self.action_chunk_size)

        if self.eef_rotation_repr is not None:
            action = action.convert_rotation(self.eef_rotation_repr)
            state = state.convert_rotation(self.eef_rotation_repr)

        if self.use_delta_action:
            action = action - state

        return state, action

    def _to_index_list(self, episode_index: int, frame_index) -> List[int]:
        if isinstance(frame_index, int):
            return [frame_index]
        if isinstance(frame_index, slice):
            n = self._metadata[episode_index].length
            return list(range(*frame_index.indices(n)))
        return list(frame_index)

    def __getitem__(self, index: int):
        episode_index, target_frame, target_n = self._resolve_index(index)

        chunk_size = min(self.action_chunk_size, target_n - target_frame)
        src_slice, resample_kwargs = self._action_chunk_source_indices(
            episode_index,
            target_frame,
            chunk_size,
        )
        state_src_idx = self._target_to_source_index(episode_index, target_frame)

        action = self._load_action(episode_index, self._to_index_list(episode_index, src_slice))
        state = self._load_state(episode_index, [state_src_idx])

        state, action = self._postprocess_step(state, action, resample_kwargs)

        outputs = {
            "robot_type": self.get_robot_type(episode_index),
            "action": action,
            "state": state,
            "text": self._load_instruction(episode_index, [state_src_idx])[0],
            "images": {k: v[0] for k, v in self._load_images(episode_index, [state_src_idx]).items()},
        }

        if self.processor is None:
            return outputs

        return self.processor(**outputs, return_tensors="pt")

    def iter_episode(
        self,
        episode_index: int,
        start: int = 0,
        step: int = 1,
        include_images: bool = True,
    ) -> Iterator[Dict]:
        """Stream post-processed step dicts for `episode_index`.

        Iterates target-fps frames ``start, start+step, start+2*step, ...``
        up to (exclusive) the episode's target length.

        Internally translates the chosen target indices → source frame
        requests and invokes the subclass ``_iter_episode``. Output dict
        shape matches ``__getitem__`` (modulo the optional processor wrap,
        which iter_episode does not apply): {robot_type, state, action,
        text, images}.
        """
        target_n = int(self._target_lengths[episode_index])
        assert step >= 1, f"iter_episode: step must be >= 1, got {step}"
        assert 0 <= start < target_n, f"iter_episode: start {start} out of range [0, {target_n})"

        target_indices: List[int] = list(range(start, target_n, step))

        plans: List[Tuple[int, slice, Optional[Dict]]] = []
        for tk in target_indices:
            chunk_size = min(self.action_chunk_size, target_n - tk)
            src_slice, resample_kwargs = self._action_chunk_source_indices(
                episode_index,
                tk,
                chunk_size,
            )
            plans.append((src_slice, resample_kwargs))

        source_ranges = [(s.start, s.stop) for s, _ in plans]

        raw_iter = self._iter_episode(
            episode_index,
            source_ranges,
            include_images,
        )
        try:
            n_yielded = 0
            for (_, resample_kwargs), raw in zip(plans, raw_iter):
                state, action = self._postprocess_step(
                    raw["state"],
                    raw["action"],
                    resample_kwargs,
                )
                yield {
                    "robot_type": self.get_robot_type(episode_index),
                    "action": action,
                    "state": state,
                    "text": raw["instruction"],
                    # base normalises: subclass may return None or omit key
                    # when include_images=False.
                    "images": raw.get("images") if include_images else None,
                }
                n_yielded += 1
            assert n_yielded == len(plans), f"_iter_episode yielded {n_yielded} steps, expected {len(plans)}"
        finally:
            # Ensure the subclass generator's `finally` runs (closing files)
            # even when the caller breaks out of this generator early.
            raw_iter.close()

    def _get_episode_schemas(self, episode_index: int) -> Tuple[RobotType, Dict, Dict]:
        """Stream the episode through `iter_episode` and accumulate per-step
        chunk schemas. Each yielded step has post-processed state (length 1)
        and action (length K), so the accumulator just folds them in."""
        robot_type = self.get_robot_type(episode_index)

        state_schema: Optional[Dict] = None
        action_schema: Optional[Dict] = None

        for step in self.iter_episode(
            episode_index,
            include_images=False,
        ):
            if state_schema is None:
                state_schema = _robot_schema(step["state"])
                action_schema = _robot_schema(step["action"])
            else:
                _merge_schema(state_schema, _robot_schema(step["state"]))
                _merge_schema(action_schema, _robot_schema(step["action"]))

        assert state_schema is not None and action_schema is not None, f"episode {episode_index} produced no steps"
        return robot_type, action_schema, state_schema

    def _schema_cache_key(self) -> Dict:
        return {
            "data_path": os.path.abspath(self.data_path),
            "action_chunk_size": self.action_chunk_size,
            "use_delta_action": self.use_delta_action,
            "eef_rotation_repr": (self.eef_rotation_repr.value if self.eef_rotation_repr is not None else None),
            "target_fps": self.target_fps,
            "num_episodes": self.num_episodes,
            "num_frames": int(self._target_lengths.sum()),
        }

    def _schema_cache_path(self, key: Dict) -> str:
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()
        return os.path.join(CACHE_DIR, "schemas", f"{self.__class__.__name__}_{digest}.json")

    def _balanced_episode_assignment(self, world_size: int) -> List[List[int]]:
        lengths = self.episode_lengths
        sorted_eps = sorted(
            range(len(lengths)),
            key=lambda i: lengths[i],
            reverse=True,
        )
        loads = [0] * world_size
        assignments: List[List[int]] = [[] for _ in range(world_size)]
        for ep_idx in sorted_eps:
            min_w = min(range(world_size), key=lambda w: loads[w])
            assignments[min_w].append(ep_idx)
            loads[min_w] += lengths[ep_idx]
        return assignments

    def get_schema(
        self,
        num_workers: int = 8,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> Dict[str, Dict]:
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank(group=process_group)
            world_size = torch.distributed.get_world_size(group=process_group)
        else:
            rank = 0
            world_size = 1

        cache_key = self._schema_cache_key()
        cache_path = self._schema_cache_path(cache_key)

        cache_hit = False
        cached_result: Optional[Dict[str, Dict]] = None
        if rank == 0 and os.path.isfile(cache_path):
            logger.info(f"Loading cached schema from {cache_path}")
            with open(cache_path, "r") as f:
                cached = json.load(f)
            cached_result = {"action": cached["action"], "state": cached["state"]}
            cache_hit = True

        if world_size > 1:
            obj_list = [cache_hit]
            torch.distributed.broadcast_object_list(obj_list, group=process_group, group_src=0)
            cache_hit = obj_list[0]
            if cache_hit:
                obj_list = [cached_result]
                torch.distributed.broadcast_object_list(obj_list, group=process_group, group_src=0)
                cached_result = obj_list[0]

        if cache_hit:
            return cached_result

        if world_size > 1:
            assignments = self._balanced_episode_assignment(world_size)
            local_episodes = sorted(assignments[rank])
        else:
            local_episodes = list(range(self.num_episodes))

        subset = torch.utils.data.Subset(_EpisodeStatsDataset(self), local_episodes)
        dataloader = torch.utils.data.DataLoader(
            subset,
            batch_size=1,
            num_workers=num_workers,
            collate_fn=_identity_collate,
            shuffle=False,
        )

        action_local: Dict[str, Dict] = {}
        state_local: Dict[str, Dict] = {}
        for robot_type, ep_action, ep_state in tqdm(dataloader, desc="Computing schemas", disable=rank > 0):
            rt = robot_type.value
            if rt not in action_local:
                action_local[rt] = ep_action
            else:
                _merge_schema(action_local[rt], ep_action)
            if rt not in state_local:
                state_local[rt] = ep_state
            else:
                _merge_schema(state_local[rt], ep_state)

        if world_size > 1:
            payload = {
                "action": {rt: _schema_to_cpu(s) for rt, s in action_local.items()},
                "state": {rt: _schema_to_cpu(s) for rt, s in state_local.items()},
            }
            gathered: List[Optional[Dict]] = [None] * world_size
            torch.distributed.all_gather_object(gathered, payload, group=process_group)
            action_merged: Dict[str, Dict] = {}
            state_merged: Dict[str, Dict] = {}
            for partial in gathered:
                if not partial:
                    continue
                for rt, s in partial["action"].items():
                    if rt not in action_merged:
                        action_merged[rt] = s
                    else:
                        _merge_schema(action_merged[rt], s)
                for rt, s in partial["state"].items():
                    if rt not in state_merged:
                        state_merged[rt] = s
                    else:
                        _merge_schema(state_merged[rt], s)
            action_local, state_local = action_merged, state_merged

        result: Dict[str, Dict] = {
            "action": {rt: _finalize_schema(s) for rt, s in action_local.items()},
            "state": {rt: _finalize_schema(s) for rt, s in state_local.items()},
        }

        if rank == 0:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump({"metadata": cache_key, **result}, f, indent=4)

        if world_size > 1:
            torch.distributed.barrier(group=process_group)

        return result
