import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from scipy.spatial.transform import Rotation as ScipyRotation
from tqdm import tqdm

from ...constants import RobotType, RotationRepresentation
from ...registry import DATASET_REGISTRY
from ...utils.robot import Arm, Position, RobotAction, RobotState, Rotation, _to_scipy_rotation
from .base import BaseVLADataset, EpisodeMetadata
from .utils import SequentialVideoReader, VideoReader, fork_safe_cache, suppress_hf_progress

_CAMERAS: Sequence[Tuple[str, str]] = (
    ("main", "observation.images.robot0_agentview_left"),
    ("main_right", "observation.images.robot0_agentview_right"),
    ("wrist", "observation.images.robot0_eye_in_hand"),
)

# observation.state (16-D) layout from RoboCasa meta/modality.json
_STATE_EEF_POS = slice(7, 10)
_STATE_EEF_ROT_QUAT_XYZW = slice(10, 14)
_STATE_GRIPPER_QPOS = slice(14, 16)

# action (12-D) layout
_ACTION_EEF_POS = slice(5, 8)
_ACTION_EEF_ROTVEC = slice(8, 11)
_ACTION_GRIPPER = slice(11, 12)


def _find_task_dirs(root: str) -> List[str]:
    """Walk *root* and return every directory that contains ``meta/info.json``.

    Mirrors the recipe in ``interndata_a1._find_task_dirs``: stop descending
    once we hit a LeRobot task dir so we don't walk into ``data/videos/meta``.
    """
    results: List[str] = []
    for dirpath, dirnames, _ in os.walk(root):
        if os.path.isfile(os.path.join(dirpath, "meta", "info.json")):
            results.append(dirpath)
            dirnames.clear()
        else:
            dirnames[:] = [d for d in dirnames if d not in ("data", "videos", "meta")]
    return sorted(results)


@DATASET_REGISTRY.register()
class RoboCasaDataset(BaseVLADataset):
    """RoboCasa LeRobot-format dataset.

    ``data_path`` should point at a directory whose tree matches the upstream
    layout, e.g.::

        <data_path>/v1.0/{pretrain,target}/{atomic,composite}/<Task>/<Date>[/mg/demo/<TS>]/lerobot/

    Either the ``v1.0`` root or any subtree containing ``meta/info.json`` works
    — discovery walks for ``meta/info.json`` markers.
    """

    _SUPPORTED_SPLITS = ("pretrain", "target")
    _SUPPORTED_CATEGORIES = ("atomic", "composite")
    _SUPPORTED_SOURCES = ("human", "mg")

    def __init__(
        self,
        *args,
        splits: Sequence[str] = ("pretrain", "target"),
        categories: Sequence[str] = ("atomic", "composite"),
        sources: Sequence[str] = ("human", "mg"),
        tasks: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        for s in splits:
            assert s in self._SUPPORTED_SPLITS, f"Unknown split {s!r}"
        for c in categories:
            assert c in self._SUPPORTED_CATEGORIES, f"Unknown category {c!r}"
        for s in sources:
            assert s in self._SUPPORTED_SOURCES, f"Unknown source {s!r}"

        self.splits = tuple(splits)
        self.categories = tuple(categories)
        self.sources = tuple(sources)
        self.task_filter = tuple(tasks) if tasks is not None else None
        super().__init__(*args, **kwargs)

    # ── discovery ──────────────────────────────────────────────

    def _candidate_roots(self) -> List[str]:
        """Roots to walk for task dirs.

        Honors ``splits`` / ``categories`` filters when the canonical
        ``v1.0/<split>/<category>`` layout is present. Falls back to the
        whole ``data_path`` so callers can also point at a single task.
        """
        roots: List[str] = []
        v1 = os.path.join(self.data_path, "v1.0")
        if os.path.isdir(v1):
            for split in self.splits:
                for cat in self.categories:
                    sub = os.path.join(v1, split, cat)
                    if os.path.isdir(sub):
                        roots.append(sub)
        if not roots:
            roots.append(self.data_path)
        return roots

    def _is_source_allowed(self, task_dir: str) -> bool:
        # `task_dir` is a path ending in ".../lerobot". The presence of
        # `mg/demo` between the date and `lerobot` distinguishes mimicgen
        # generations from human teleop demos.
        is_mg = f"{os.sep}mg{os.sep}demo{os.sep}" in task_dir + os.sep
        if is_mg and "mg" not in self.sources:
            return False
        if not is_mg and "human" not in self.sources:
            return False
        return True

    def _matches_task_filter(self, task_dir: str) -> bool:
        if self.task_filter is None:
            return True
        return any(t in task_dir for t in self.task_filter)

    def _collect_task_dirs(self) -> List[str]:
        out: List[str] = []
        for root in self._candidate_roots():
            for d in _find_task_dirs(root):
                if not d.endswith(os.sep + "lerobot") and os.path.basename(d) != "lerobot":
                    # The discovery anchor is `meta/info.json`, which lives
                    # under `<task>/<date>[/mg/demo/<TS>]/lerobot`. Skip
                    # anything that doesn't follow that convention.
                    continue
                if self._is_source_allowed(d) and self._matches_task_filter(d):
                    out.append(d)
        return sorted(set(out))

    def _scan_one(self, task_dir: str) -> List[EpisodeMetadata]:
        with open(os.path.join(task_dir, "meta", "info.json"), "r") as f:
            info = json.load(f)
        if int(info.get("total_episodes") or 0) == 0:
            return []

        chunks_size = int(info["chunks_size"])
        data_path_tpl = info["data_path"]
        video_path_tpl = info["video_path"]
        fps = float(info.get("fps", 20.0))

        tasks_map: Dict[int, str] = {}
        with open(os.path.join(task_dir, "meta", "tasks.jsonl"), "r") as f:
            for line in f:
                entry = json.loads(line)
                tasks_map[int(entry["task_index"])] = entry["task"]
        n_tasks = max(tasks_map) + 1 if tasks_map else 0
        task_texts = [tasks_map.get(i, "") for i in range(n_tasks)]

        out: List[EpisodeMetadata] = []
        with open(os.path.join(task_dir, "meta", "episodes.jsonl"), "r") as f:
            for line in f:
                entry = json.loads(line)
                n = int(entry["length"])
                if n <= 0:
                    continue
                ep_idx = int(entry["episode_index"])
                episode_chunk = ep_idx // chunks_size
                parquet_path = os.path.join(
                    task_dir,
                    data_path_tpl.format(
                        episode_chunk=episode_chunk,
                        episode_index=ep_idx,
                    ),
                )
                out.append(
                    EpisodeMetadata(
                        length=n,
                        fps=fps,
                        robot_type=RobotType.FRANKA_OMRON,
                        extras={
                            "task_dir": task_dir,
                            "parquet_path": parquet_path,
                            "video_tpl": video_path_tpl,
                            "episode_chunk": episode_chunk,
                            "parquet_ep_index": ep_idx,
                            "task_texts": task_texts,
                        },
                    )
                )
        return out

    def _load_metadata(self) -> List[EpisodeMetadata]:
        task_dirs = self._collect_task_dirs()
        assert task_dirs, f"No RoboCasa lerobot tasks found under {self.data_path}"

        out: List[EpisodeMetadata] = []
        with ThreadPoolExecutor(max_workers=16) as pool:
            for episodes in tqdm(
                pool.map(self._scan_one, task_dirs),
                total=len(task_dirs),
                desc="Scanning RoboCasa",
            ):
                out.extend(episodes)
        return out

    # ── per-frame loaders ──────────────────────────────────────

    def _video_path(self, extras: Dict, video_key: str) -> str:
        return os.path.join(
            extras["task_dir"],
            extras["video_tpl"].format(
                episode_chunk=extras["episode_chunk"],
                video_key=video_key,
                episode_index=extras["parquet_ep_index"],
            ),
        )

    @fork_safe_cache
    def _get_parquet(self, parquet_path: str):
        with suppress_hf_progress():
            return load_dataset(
                "parquet",
                data_files=parquet_path,
                split="train",
            ).with_format("torch")

    @fork_safe_cache
    def _get_video_reader(self, video_path: str):
        if not os.path.exists(video_path):
            return None
        return VideoReader(video_path)

    @staticmethod
    def _bulk_column(ds, name: str) -> torch.Tensor:
        col = ds[:][name]
        if isinstance(col, list):
            col = torch.tensor(col)
        return col.float()

    _GRIPPER_MAX_WIDTH = 0.04

    @staticmethod
    def _make_state(state_col: torch.Tensor, idx: List[int]) -> RobotState:
        s = state_col[idx]
        gripper_open = (s[:, 14:15] / RoboCasaDataset._GRIPPER_MAX_WIDTH).clamp(0.0, 1.0)
        return RobotState(
            left_arm=Arm(
                eef_position=Position(s[:, _STATE_EEF_POS]),
                eef_rotation=Rotation(
                    s[:, _STATE_EEF_ROT_QUAT_XYZW],
                    representation=RotationRepresentation.QUAT_XYZW,
                ),
            ),
            left_gripper=Position(gripper_open, allow_relative=False),
        )

    OSC_POS_SCALE = 0.05
    OSC_ROT_SCALE = 0.5

    @staticmethod
    def _make_action(action_col: torch.Tensor, state_col: torch.Tensor, idx: List[int]) -> RobotAction:
        a = action_col[idx]
        s = state_col[idx]

        pos_delta = a[:, _ACTION_EEF_POS] * RoboCasaDataset.OSC_POS_SCALE
        abs_pos = s[:, _STATE_EEF_POS] + pos_delta

        rot_delta = ScipyRotation.from_rotvec((a[:, _ACTION_EEF_ROTVEC] * RoboCasaDataset.OSC_ROT_SCALE).numpy())
        cur_rot = _to_scipy_rotation(s[:, _STATE_EEF_ROT_QUAT_XYZW], RotationRepresentation.QUAT_XYZW)
        abs_rot = torch.from_numpy((rot_delta * cur_rot).as_rotvec()).float()

        gripper = (1.0 - a[:, _ACTION_GRIPPER]) * 0.5
        return RobotAction(
            left_arm=Arm(
                eef_position=Position(abs_pos),
                eef_rotation=Rotation(abs_rot, representation=RotationRepresentation.ROT_VEC),
            ),
            left_gripper=Position(gripper, allow_relative=False),
        )

    def _load_action(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotAction:
        extras = self._metadata[episode_index].extras
        ds = self._get_parquet(extras["parquet_path"])
        action_col = self._bulk_column(ds, "action")
        state_col = self._bulk_column(ds, "observation.state")
        return self._make_action(action_col, state_col, frame_index)

    def _load_state(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotState:
        extras = self._metadata[episode_index].extras
        ds = self._get_parquet(extras["parquet_path"])
        state_col = self._bulk_column(ds, "observation.state")
        return self._make_state(state_col, frame_index)

    def _load_images(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> Dict[str, torch.Tensor]:
        extras = self._metadata[episode_index].extras
        images: Dict[str, torch.Tensor] = {}
        for out_name, video_key in _CAMERAS:
            reader = self._get_video_reader(self._video_path(extras, video_key))
            if reader is None:
                continue
            images[out_name] = reader.read(frame_index)
        return images

    def _load_instruction(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> List[str]:
        extras = self._metadata[episode_index].extras
        ds = self._get_parquet(extras["parquet_path"])
        task_texts = extras["task_texts"]
        return [task_texts[int(ds[i]["task_index"])] for i in frame_index]

    # ── _iter_episode ──────────────────────────────────────────

    def _iter_episode(
        self,
        episode_index: int,
        source_ranges: List[tuple],
        include_images: bool = True,
    ) -> Iterator[Dict]:
        meta = self._metadata[episode_index]
        extras = meta.extras
        n_total = meta.length
        task_texts = extras["task_texts"]

        with suppress_hf_progress():
            ds = load_dataset(
                "parquet",
                data_files=extras["parquet_path"],
                split="train",
            ).with_format("torch")

        all_idx = list(range(n_total))
        state_col = self._bulk_column(ds, "observation.state")
        action_col = self._bulk_column(ds, "action")
        full_state = self._make_state(state_col, all_idx)
        full_action = self._make_action(action_col, state_col, all_idx)

        ti_col = ds[:]["task_index"]
        all_task_indices = ti_col.tolist() if not isinstance(ti_col, list) else ti_col

        readers: Dict[str, SequentialVideoReader] = {}
        if include_images:
            for out_name, video_key in _CAMERAS:
                path = self._video_path(extras, video_key)
                if os.path.exists(path):
                    readers[out_name] = SequentialVideoReader(path)

        try:
            for start, end in source_ranges:
                images = None
                if include_images:
                    images = {k: r.read(start) for k, r in readers.items()}
                yield {
                    "state": full_state[start : start + 1],
                    "action": full_action[start:end],
                    "instruction": task_texts[int(all_task_indices[start])],
                    "images": images,
                }
        finally:
            for r in readers.values():
                r.close()
