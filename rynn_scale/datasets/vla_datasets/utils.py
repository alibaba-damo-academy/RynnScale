import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Union

import torch
from tqdm import tqdm

from ...utils.misc import fork_safe_cache, suppress_hf_progress
from ...utils.video import (
    SequentialVideoReader,
    VideoReader,
    decode_video_frames_pyav_by_timestamps,
)

# Re-export so existing `from .utils import ...` lines keep working.
__all__ = [
    "fork_safe_cache",
    "suppress_hf_progress",
    "VideoReader",
    "SequentialVideoReader",
    "decode_video_frames_pyav_by_timestamps",
    "to_index_list",
    "frame_index_length",
    "gather_column",
    "mt_process",
    "mp_process",
]


# ═══════════════════════════════════════════════════════════════════
#  Common dataset helpers
# ═══════════════════════════════════════════════════════════════════


def to_index_list(frame_index: Union[int, List[int], slice]) -> List[int]:
    """Normalize *frame_index* to a plain ``list[int]``."""
    if isinstance(frame_index, int):
        return [frame_index]
    if isinstance(frame_index, slice):
        start = frame_index.start or 0
        stop = frame_index.stop or 0
        return list(range(start, stop))
    return list(frame_index)


def frame_index_length(frame_index: Union[int, List[int], slice]) -> int:
    """Return the number of frames *frame_index* represents."""
    if isinstance(frame_index, int):
        return 1
    if isinstance(frame_index, slice):
        return (frame_index.stop or 0) - (frame_index.start or 0)
    return len(frame_index)


def gather_column(ds, name: str, idx: List[int]) -> torch.Tensor:
    """Index into a HuggingFace dataset column and stack as float tensor."""
    out = torch.stack([ds[i][name] for i in idx], dim=0).float()
    if out.ndim == 1:
        out = out.unsqueeze(-1)
    return out


# ═══════════════════════════════════════════════════════════════════
#  Multi-threaded / multi-process helpers
# ═══════════════════════════════════════════════════════════════════


def mt_process(func, tasks, max_workers=None, desc="Processing", **kwargs):
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) * 4)

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(func, task, **kwargs): task for task in tasks}

        with tqdm(total=len(tasks), desc=desc, dynamic_ncols=True) as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    tqdm.write(f"\n[Error] Task {task} failed: {e}")
                finally:
                    pbar.update(1)

    return results


def mp_process(func, tasks, max_workers=None, desc="Processing", **kwargs):
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) * 4)

    results = [None] * len(tasks)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(func, task, **kwargs): i for i, task in enumerate(tasks)}

        with tqdm(total=len(tasks), desc=desc, dynamic_ncols=True) as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                except Exception as e:
                    tqdm.write(f"\n[Error] Task at index {idx} failed: {e}")
                finally:
                    pbar.update(1)

    return results
