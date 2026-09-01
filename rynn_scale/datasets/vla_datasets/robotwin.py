import json
import os
import random
from typing import Dict, Iterator, List

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as _ScipyRotation

from ...constants import RobotType, RotationRepresentation
from ...registry import DATASET_REGISTRY
from ...utils.processing import decode_image_bytes
from ...utils.robot import Arm, Position, RobotAction, RobotState, Rotation
from .base import BaseVLADataset, EpisodeMetadata
from .utils import fork_safe_cache, mp_process, mt_process


def _quat_wxyz_to_mat(q_wxyz: np.ndarray) -> np.ndarray:
    """(..., 4) WXYZ quaternions → (..., 3, 3) rotation matrices."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    xyzw = np.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], axis=-1)
    mats = _ScipyRotation.from_quat(xyzw.reshape(-1, 4)).as_matrix()
    return mats.reshape(*q.shape[:-1], 3, 3)


def _mat_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrices → (..., 4) WXYZ quaternions."""
    xyzw = _ScipyRotation.from_matrix(mat.reshape(-1, 3, 3)).as_quat()
    wxyz = np.stack([xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]], axis=-1)
    return wxyz.reshape(*mat.shape[:-2], 4)


VARIANTS = (
    "aloha-agilex_clean_50",
    "aloha-agilex_randomized_500",
    "arx-x5_clean_50",
    "arx-x5_randomized_500",
    "ur5_clean_50",
    "ur5_randomized_500",
    "franka_clean_50",
    "franka_randomized_500",
    "piper_clean_50",
    "piper_randomized_500",
)
VARIANT_TO_ROBOT = {
    "franka": RobotType.DUAL_FRANKA,
    "ur5": RobotType.DUAL_UR_5,
    "aloha-agilex": RobotType.AGILEX_COBOT_MAGIC_1,
    "arx-x5": RobotType.DUAL_ARX_X5,
    "piper": RobotType.DUAL_AGILEX_PIPER,
}

CAMERAS = ("front", "head", "left", "right")
INDEX_NAME = "episodes.jsonl"


def _scan_dir(job):
    data_path, task, variant = job
    data_dir = os.path.join(data_path, task, variant, "data")
    inst_dir = os.path.join(data_path, task, variant, "instructions")
    if not (os.path.isdir(data_dir) and os.path.isdir(inst_dir)):
        return []
    try:
        inst_files = {f for f in os.listdir(inst_dir) if f.endswith(".json")}
    except OSError:
        return []
    return [
        (
            os.path.join(data_dir, fname),
            os.path.join(inst_dir, fname[:-5] + ".json"),
            variant,
            os.path.join(data_path, task, variant),
        )
        for fname in os.listdir(data_dir)
        if fname.endswith(".hdf5") and fname[:-5] + ".json" in inst_files
    ]


def _scan_episode(job):
    hdf5_path, inst_path, variant, _ = job
    try:
        with h5py.File(hdf5_path, "r", libver="latest", swmr=True, locking=False, rdcc_nbytes=0) as f:
            n = int(f["endpose/left_endpose"].shape[0])
        with open(inst_path) as fp:
            ins = json.load(fp).get("seen", [])
        rt = VARIANT_TO_ROBOT[variant.rsplit("_", 2)[0]].value
        return {"path": hdf5_path, "length": n, "instructions": ins, "robot_type": rt} if n > 0 else None
    except Exception:
        return None


@DATASET_REGISTRY.register()
class RoboTwinDataset(BaseVLADataset):
    _SOURCE_FPS = 25.0

    def _load_metadata(self) -> List[EpisodeMetadata]:
        idx_path = os.path.join(self.data_path, INDEX_NAME)
        assert os.path.isfile(idx_path), (
            f"No {INDEX_NAME} found under {self.data_path}. Run scan first: python -m {__name__} --root {self.data_path}"
        )

        out: List[EpisodeMetadata] = []
        with open(idx_path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                ep = json.loads(line)
                path = ep["path"]
                if not os.path.isabs(path):
                    path = os.path.join(self.data_path, path)
                out.append(
                    EpisodeMetadata(
                        length=ep["length"],
                        fps=self._SOURCE_FPS,
                        robot_type=RobotType(ep["robot_type"]),
                        extras={
                            "path": path,
                            "instructions": ep.get("instructions", []),
                        },
                    )
                )

        assert out, f"No valid episodes in {idx_path}"
        return out

    @fork_safe_cache
    def _open_hdf5(self, path: str):
        return h5py.File(path, "r", libver="latest", swmr=True, locking=False, rdcc_nbytes=0)

    @staticmethod
    def _build_proprio(f, idxs, cls, robot_type=None):
        # ``endpose`` lives in a single world frame shared by both arms, while
        # the renderer's EEF+IK path expects each arm's pose in ITS OWN base
        # frame. For calibrated embodiments we remap world → base here; the
        # recorded joint angles are frame-independent and passed through as-is.
        base = RoboTwinDataset._ARM_BASE_IN_WORLD.get(robot_type)
        eef_off = RoboTwinDataset._EEF_FRAME_OFFSET.get(robot_type)
        arms, grippers = [], []
        for side in ("left", "right"):
            ep = np.asarray(f[f"endpose/{side}_endpose"][idxs], dtype=np.float64)
            gp = f[f"endpose/{side}_gripper"][idxs].reshape(-1, 1)
            ja = f[f"joint_action/{side}_arm"][idxs]
            if base is not None:
                eo = eef_off[side] if eef_off is not None else None
                eef_pos, eef_quat = RoboTwinDataset._world_to_base(ep, *base[side], eef_offset=eo)
            else:
                eef_pos, eef_quat = ep[:, :3], ep[:, 3:7]
            arms.append(
                Arm(
                    joint_position=Position(torch.from_numpy(np.ascontiguousarray(ja)).float()),
                    eef_position=Position(torch.from_numpy(np.ascontiguousarray(eef_pos)).float()),
                    eef_rotation=Rotation(
                        torch.from_numpy(np.ascontiguousarray(eef_quat)).float(),
                        representation=RotationRepresentation.QUAT_WXYZ,
                    ),
                )
            )
            grippers.append(Position(torch.from_numpy(gp).float(), allow_relative=False))
        return cls(
            left_arm=arms[0],
            right_arm=arms[1],
            left_gripper=grippers[0],
            right_gripper=grippers[1],
        )

    # Per-arm base_link pose T_{W←B} = (pos, quat_wxyz) in the dataset's shared
    # world frame (where ``endpose`` lives), calibrated by AX=ZB hand-eye against
    # renderer FK over ~1k motion-rich frames across tasks. All five RoboTwin
    # embodiments share RoboTwin's joint-axis kinematics (verified: frame-to-frame
    # rotation-increment corr ≥ 0.985), so every one can be aligned:
    #   - agilex / arx-x5: renderer EEF frame already matches RoboTwin's → base
    #     transform alone suffices (residual ~4 mm).
    #   - ur5 / franka / piper: renderer tracks a different EEF link (e.g. Franka
    #     ``hand_tcp`` vs RoboTwin ``panda_hand``); the fixed EEF offset in
    #     ``_EEF_FRAME_OFFSET`` re-expresses the pose in the renderer's EEF frame
    #     (residual ~10–30 mm / 1–4°). Calibrated base pos/rot match each
    #     embodiment's official config.yml ``robot_pose``.
    _ARM_BASE_IN_WORLD = {
        RobotType.AGILEX_COBOT_MAGIC_1: {
            "left": (np.array([-0.2947, -0.4129, 0.7846]), np.array([0.6985, -0.00071, 0.00027, 0.71561])),
            "right": (np.array([0.3037, -0.4125, 0.7840]), np.array([0.7023, 0.00015, 0.00033, 0.71188])),
        },
        RobotType.DUAL_ARX_X5: {
            "left": (np.array([-0.3014, -0.3514, 0.7799]), np.array([0.70713, 0.0001, -0.00066, 0.70708])),
            "right": (np.array([0.3017, -0.3517, 0.7802]), np.array([0.70724, 0.00025, 0.00133, 0.70698])),
        },
        RobotType.DUAL_UR_5: {
            "left": (np.array([-0.4063, -0.6482, 0.6497]), np.array([0.99998, 0.00381, 0.00312, -0.00312])),
            "right": (np.array([0.3761, -0.6548, 0.7434]), np.array([0.99995, 0.00538, -0.00182, -0.00842])),
        },
        RobotType.DUAL_FRANKA: {
            "left": (np.array([-0.4014, -0.6446, 0.7508]), np.array([0.70744, 0.00099, -0.00111, 0.70677])),
            "right": (np.array([0.3977, -0.6486, 0.7513]), np.array([0.7073, 0.00394, 0.00072, 0.7069])),
        },
        RobotType.DUAL_AGILEX_PIPER: {
            "left": (np.array([-0.34, -0.428, 0.7182]), np.array([0.75344, 0.01471, -0.0246, 0.65689])),
            "right": (np.array([0.2476, -0.4635, 0.7832]), np.array([0.75887, -0.01534, 0.0096, 0.65099])),
        },
    }

    # Fixed renderer-EEF → RoboTwin-EEF offset T_{E_r←E_rt} = (d, quat_wxyz), for
    # embodiments whose renderer tracks a different EEF link than RoboTwin's
    # endpose reference. Applied as a right-multiply in _world_to_base. Absent
    # embodiments (agilex / arx-x5) need no EEF re-expression (M=I, d=0).
    _EEF_FRAME_OFFSET = {
        RobotType.DUAL_UR_5: {
            "left": (np.array([0.0017, -0.0039, -0.0085]), np.array([0.50221, 0.5022, -0.49919, 0.49638])),
            "right": (np.array([0.0074, 0.0147, -0.0056]), np.array([0.50216, 0.48382, -0.49943, 0.51412])),
        },
        RobotType.DUAL_FRANKA: {
            "left": (np.array([-0.0004, 0.0002, -0.1474]), np.array([0.00137, 0.70886, -0.00492, 0.70533])),
            "right": (np.array([-0.0013, 0.0032, -0.1457]), np.array([0.00165, 0.70699, 0.0033, 0.70721])),
        },
        RobotType.DUAL_AGILEX_PIPER: {
            "left": (np.array([-0.0304, -0.0042, 0.0041]), np.array([0.72364, -0.07294, -0.68627, -0.00771])),
            "right": (np.array([0.0337, -0.0106, 0.0061]), np.array([-0.69356, 0.08181, 0.7156, 0.01422])),
        },
    }

    @staticmethod
    def _world_to_base(ep, base_pos, base_quat_wxyz, eef_offset=None):
        """Map (N,7) world-frame [xyz, quat_wxyz] EEF poses into an arm's base frame.

        Given the arm base pose T_{W←B} = (base_pos, base_quat), the world-frame
        endpose (RoboTwin's EEF link E_rt) is first expressed in the base frame:
        ``p_B = R_Bᵀ (p_W − t_B)``, ``R_eef_B = R_Bᵀ R_W``.

        ``eef_offset = (d, M_wxyz)`` optionally re-expresses the pose in the
        *renderer's* EEF link E_r (which may differ from RoboTwin's, e.g. Franka
        ``hand_tcp`` vs ``panda_hand``). With T_{E_r←E_rt} = (M, d):
        ``R_out = R_eef_B · Mᵀ``, ``p_out = p_B − R_out · d``.
        """
        R_B = _quat_wxyz_to_mat(base_quat_wxyz)  # (3, 3)
        R_W = _quat_wxyz_to_mat(ep[:, 3:7])  # (N, 3, 3)
        p_B = (ep[:, :3] - base_pos) @ R_B  # row-vector form of R_Bᵀ(p_W − t_B)
        R_eef_B = np.einsum("ji,njk->nik", R_B, R_W)  # R_Bᵀ R_W
        if eef_offset is not None:
            d, m_wxyz = eef_offset
            M = _quat_wxyz_to_mat(m_wxyz)  # renderer-EEF → RoboTwin-EEF rot
            R_eef_B = np.einsum("nij,kj->nik", R_eef_B, M)  # R_eef_B · Mᵀ
            p_B = p_B - np.einsum("nij,j->ni", R_eef_B, d)  # p_B − R_out · d
        return p_B.astype(np.float32), _mat_to_quat_wxyz(R_eef_B).astype(np.float32)

    def _load_action(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotAction:
        path = self._metadata[episode_index].extras["path"]
        f = self._open_hdf5(path)
        return self._build_proprio(f, frame_index, RobotAction, self._metadata[episode_index].robot_type)

    def _load_state(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotState:
        path = self._metadata[episode_index].extras["path"]
        f = self._open_hdf5(path)
        return self._build_proprio(f, frame_index, RobotState, self._metadata[episode_index].robot_type)

    def _load_images(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> Dict[str, torch.Tensor]:
        path = self._metadata[episode_index].extras["path"]
        f = self._open_hdf5(path)
        out = {}
        for cam in CAMERAS:
            key = f"observation/{cam}_camera/rgb"
            if key in f:
                out[cam] = torch.from_numpy(np.stack([decode_image_bytes(bytes(f[key][i])) for i in frame_index]))
        return out

    def _load_instruction(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> List[str]:
        ins = self._metadata[episode_index].extras["instructions"]
        return [random.choice(ins) if ins else ""] * len(frame_index)

    def _iter_episode(
        self,
        episode_index: int,
        source_ranges: List[tuple],
        include_images: bool = True,
    ) -> Iterator[Dict]:
        meta = self._metadata[episode_index]
        n_total = meta.length
        ins_list = meta.extras["instructions"]
        instruction = random.choice(ins_list) if ins_list else ""

        with h5py.File(meta.extras["path"], "r", libver="latest", swmr=True, locking=False, rdcc_nbytes=0) as f:
            full_state = self._build_proprio(f, list(range(n_total)), RobotState, meta.robot_type)
            full_action = self._build_proprio(f, list(range(n_total)), RobotAction, meta.robot_type)

            for start, end in source_ranges:
                images = None
                if include_images:
                    images = {}
                    for cam in CAMERAS:
                        key = f"observation/{cam}_camera/rgb"
                        if key in f:
                            raw = f[key][start]
                            images[cam] = torch.from_numpy(decode_image_bytes(bytes(raw)))
                yield {
                    "state": full_state[start : start + 1],
                    "action": full_action[start:end],
                    "instruction": instruction,
                    "images": images,
                }


# ── CLI: scan and build episodes.jsonl at root ───────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build RoboTwin episodes.jsonl index.")
    ap.add_argument("--root", required=True, help="dataset root")
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    tasks = sorted(x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x)) and not x.startswith("."))
    print(f"{len(tasks)} tasks found under {root}")

    jobs = [(root, t, v) for t in tasks for v in VARIANTS]
    nested = mt_process(_scan_dir, jobs, max_workers=32, desc="Listing RoboTwin")
    scans = [p for pairs in nested if pairs for p in pairs]
    print(f"{len(scans)} hdf5+json pairs found")

    records = [r for r in mp_process(_scan_episode, scans, max_workers=args.workers, desc="Scanning RoboTwin") if r]

    # Extract task name from path: root/task/variant/data/xxx.hdf5
    def _task_from_path(hdf5_path: str) -> str:
        rel = os.path.relpath(hdf5_path, root)
        return rel.split(os.sep)[0]

    output = os.path.join(root, INDEX_NAME)
    with open(output, "w") as fout:
        for idx, rec in enumerate(records):
            fout.write(
                json.dumps(
                    {
                        "episode_index": idx,
                        "path": os.path.relpath(rec["path"], root),
                        "task": _task_from_path(rec["path"]),
                        "length": rec["length"],
                        "instructions": rec["instructions"],
                        "robot_type": rec["robot_type"],
                    }
                )
                + "\n"
            )

    print(f"Wrote {len(records)} episodes to {output}")
