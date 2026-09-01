import json
import os
from typing import Dict, Iterator, List, Tuple

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyRotation

from ...constants import RobotType, RotationRepresentation
from ...registry import DATASET_REGISTRY
from ...utils.processing import decode_image_bytes
from ...utils.robot import Arm, Position, RobotAction, RobotState, Rotation
from .base import BaseVLADataset, EpisodeMetadata
from .utils import fork_safe_cache, mp_process

INDEX_NAME = "episodes.jsonl"

# Data-collection stations (``metadata/collector``) that recorded the two arms
# with left/right reversed: the whole left↔right chain (arm joints, end-effector
# pose, gripper, and the wrist cameras camera_left/camera_right) is swapped
# consistently for every episode from these collectors. Verified on the
# single-arm agilex_mobile tasks (front-camera anchor) where a "use the right
# arm" instruction drives the left channel; the swap is per-collector (each
# collector is uniformly swapped or not). Because the "swapped for the whole
# collector" assumption is not day-by-day verified, episodes from these
# collectors are DROPPED at index-load time rather than corrected in place.
_SWAPPED_COLLECTORS = frozenset(
    {
        "open_1fbc5a11f4ea41d499b5669981d96",  # move_magnetic_blocks_..._right_arm
        "open_49441f1aed50316a5e3718843168d2c",  # place_blue_tray_..._right_arm
        "open_e87016ecc7c9265631df68164b106b8e",  # place_green_plate_..._right_arm (+static)
        "open_6f3eeceaf47f8ca08d468abe60e5da3c",  # transfer_vegetables_... (+static)
    }
)

_Z_AXIS, _Y_AXIS = 2, 1


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


# ── Franka 7-DOF chain ──────────────────────────────────────────────────────
# Parameters from official franka_description kinematics.yaml (identical across
# fer / fr3 / fr3v2 / fr3v2_1 / fp3).

_FRANKA_CHAIN: Tuple[Tuple[np.ndarray, np.ndarray], ...] = (
    (np.array([0.0, 0.0, 0.333], dtype=np.float64), np.eye(3, dtype=np.float64)),
    (np.array([0.0, 0.0, 0.0], dtype=np.float64), _rx(-np.pi / 2)),
    (np.array([0.0, -0.316, 0.0], dtype=np.float64), _rx(np.pi / 2)),
    (np.array([0.0825, 0.0, 0.0], dtype=np.float64), _rx(np.pi / 2)),
    (np.array([-0.0825, 0.384, 0.0], dtype=np.float64), _rx(-np.pi / 2)),
    (np.array([0.0, 0.0, 0.0], dtype=np.float64), _rx(np.pi / 2)),
    (np.array([0.088, 0.0, 0.0], dtype=np.float64), _rx(np.pi / 2)),
)
_FRANKA_FLANGE_OFFSET = np.array([0.0, 0.0, 0.107], dtype=np.float64)

# ── UR5e 6-DOF chain ────────────────────────────────────────────────────────
# Parameters from assets/ur_5e/ur5e.xml (MuJoCo UR5e model).
# Joint axes follow the MuJoCo model: Z-Y-Y-Y-Z-Y.
# Flange = attachment_site at pos=(0, 0.1, 0) quat=(-1,1,0,0) ≡ Rx(-π/2).

_UR5E_CHAIN: Tuple[Tuple[np.ndarray, np.ndarray], ...] = (
    (np.array([0.0, 0.0, 0.163], dtype=np.float64), np.eye(3, dtype=np.float64)),
    (np.array([0.0, 0.138, 0.0], dtype=np.float64), _ry(np.pi / 2)),
    (np.array([0.0, -0.131, 0.425], dtype=np.float64), np.eye(3, dtype=np.float64)),
    (np.array([0.0, 0.0, 0.392], dtype=np.float64), _ry(np.pi / 2)),
    (np.array([0.0, 0.127, 0.0], dtype=np.float64), np.eye(3, dtype=np.float64)),
    (np.array([0.0, 0.0, 0.1], dtype=np.float64), np.eye(3, dtype=np.float64)),
)
_UR5E_JOINT_AXES = (_Z_AXIS, _Y_AXIS, _Y_AXIS, _Y_AXIS, _Z_AXIS, _Y_AXIS)
_UR5E_FLANGE_OFFSET = np.array([0.0, 0.1, 0.0], dtype=np.float64)
_UR5E_FLANGE_ROTATION = _rx(-np.pi / 2)

_EMBODIMENTS = {
    "agilex_cobot_magic": dict(
        robot_type=RobotType.AGILEX_COBOT_MAGIC_2,
        n_joints=6,
        gripper_range={"puppet": (0.0, 0.44), "master": (0.0, 8.0)},
    ),
    "ark": dict(robot_type=RobotType.ARX_LIFT2, n_joints=6),
    "franka": dict(
        robot_type=RobotType.FRANKA, n_joints=7, gripper_range={"puppet": (1.0, 0.0), "master": (1.0, 0.0)}
    ),
    # v2 RoboMIND2.0 "franka" is a dual-arm rig (both arm channels recorded,
    # loaded via _load_proprio_v2). v1 single-arm franka (h5_franka_1rgb/3rgb)
    # keeps the "franka" key above.
    "dual_franka": dict(robot_type=RobotType.DUAL_FRANKA, n_joints=7, gripper_invert=True),
    "franka_fr3_dual": dict(
        robot_type=RobotType.DUAL_FRANKA, n_joints=7, gripper_range={"puppet": (1.0, 0.0), "master": (1.0, 0.0)}
    ),
    "ur5": dict(robot_type=RobotType.UR_5E, n_joints=6, gripper_range={"puppet": (1.0, 0.0), "master": (1.0, 0.0)}),
    "ur5_dual": dict(robot_type=RobotType.DUAL_UR_5E, n_joints=6),
    "ur5_dex": dict(robot_type=RobotType.DUAL_UR_5E_DEX, n_joints=6),
    "tianyi": dict(robot_type=RobotType.TIANYI, n_joints=7),
    "tienkung": dict(robot_type=RobotType.TIENKUNG_2, n_joints=7),
    "tienkung_gello": dict(robot_type=RobotType.TIENKUNG_1, n_joints=7),
    "tienkung_xsens": dict(robot_type=RobotType.TIENKUNG_1, n_joints=7),
    "sim_franka": dict(robot_type=RobotType.FRANKA, n_joints=7),
    "sim_tienkung": dict(robot_type=RobotType.TIENKUNG_1, n_joints=7),
}


def _serial_fk(
    joints: np.ndarray,
    chain: Tuple[Tuple[np.ndarray, np.ndarray], ...],
    flange_offset: np.ndarray | None = None,
    flange_rotation: np.ndarray | None = None,
    joint_axes: Tuple[int, ...] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """FK for an N-DOF serial chain, vectorized over batch.

    Args:
        joints: (N, n_dof) joint angles in radians, float64.
        chain: per-joint (translation, fixed_rotation) tuples.
        flange_offset: optional fixed translation after the last joint.
        flange_rotation: optional fixed rotation after the last joint.
        joint_axes: per-joint rotation axis (_Z_AXIS or _Y_AXIS);
                    defaults to all-Z when *None*.
    Returns:
        pos: (N, 3) end-effector position.
        rot: (N, 3, 3) end-effector orientation.
    """
    n = joints.shape[0]
    pos = np.zeros((n, 3), dtype=np.float64)
    rot = np.broadcast_to(np.eye(3, dtype=np.float64), (n, 3, 3)).copy()

    for i, (offset, R_fixed) in enumerate(chain):
        pos += np.einsum("nij,j->ni", rot, offset)
        rot = rot @ R_fixed
        c = np.cos(joints[:, i])
        s = np.sin(joints[:, i])
        Rj = np.zeros((n, 3, 3), dtype=np.float64)
        ax = _Z_AXIS if joint_axes is None else joint_axes[i]
        if ax == _Z_AXIS:
            Rj[:, 0, 0] = c
            Rj[:, 0, 1] = -s
            Rj[:, 1, 0] = s
            Rj[:, 1, 1] = c
            Rj[:, 2, 2] = 1.0
        else:  # _Y_AXIS
            Rj[:, 0, 0] = c
            Rj[:, 0, 2] = s
            Rj[:, 1, 1] = 1.0
            Rj[:, 2, 0] = -s
            Rj[:, 2, 2] = c
        rot = rot @ Rj

    if flange_offset is not None:
        pos += np.einsum("nij,j->ni", rot, flange_offset)
    if flange_rotation is not None:
        rot = rot @ flange_rotation
    return pos, rot


def _normalize_gripper(raw: np.ndarray, spec: dict, who: str) -> np.ndarray:
    """Normalize gripper to [0, 1] (0 = closed, 1 = open)."""
    gripper_range = spec.get("gripper_range", {}).get(who)
    if gripper_range is not None:
        closed, opened = gripper_range
        raw = (raw - closed) / (opened - closed)
    return np.clip(raw, 0.0, 1.0).astype(np.float32)


@DATASET_REGISTRY.register()
class RoboMINDDataset(BaseVLADataset):
    """Unified RoboMIND dataset supporting both v1 and v2 HDF5 schemas.

    Schema version and subset are recorded per-episode in ``episodes.jsonl``
    (``version`` = 1 or 2, ``subset`` = directory name). Dispatches to the
    appropriate HDF5 reading logic based on these fields.

    v2 schema:
      images:      camera_observations/color_images/{cam}
      joints:      {who}/arm_{side}_position_align/data
      gripper:     {who}/end_effector_{side}_position_align/data
      eef:         {who}/end_effector_{side}_pose_align/data  (pos + quat_wxyz)
      instruction: metadata.attrs["language_instruction"]

    v1 schema:
      images:      observations/rgb_images/{cam}
      joints:      {who}/joint_position[_{side}]
      gripper:     packed in joint_position tail
      eef:         {who}/end_effector[_{side}]  (pos + euler_xyz)
      instruction: language_raw dataset
    """

    def _load_metadata(self) -> List[EpisodeMetadata]:
        data_dir = os.path.join(self.data_path, "data")
        search_root = data_dir if os.path.isdir(data_dir) else self.data_path

        # Per-subset indexes live inside each subset dir; top-level index is
        # the v1 fallback where paths are relative to the dataset root.
        index_sources = []
        for subset in sorted(os.listdir(search_root)):
            subset_root = os.path.join(search_root, subset)
            idx_path = os.path.join(subset_root, INDEX_NAME)
            if os.path.isfile(idx_path):
                index_sources.append((idx_path, subset_root))

        if not index_sources:
            top_idx = os.path.join(self.data_path, INDEX_NAME)
            if os.path.isfile(top_idx):
                index_sources.append((top_idx, self.data_path))

        out: List[EpisodeMetadata] = []
        for idx_path, resolve_base in index_sources:
            with open(idx_path) as fh:
                for line_no, line in enumerate(fh, 1):
                    if not line.strip():
                        continue
                    e = json.loads(line)
                    rt = e.get("robot_type")
                    if rt not in _EMBODIMENTS:
                        continue
                    if not e.get("aligned", True):
                        continue
                    if e.get("collector") in _SWAPPED_COLLECTORS:
                        continue  # left/right arms reversed for this collector
                    for key in ("version", "subset"):
                        if key not in e:
                            raise KeyError(
                                f"{idx_path}:{line_no} missing required field '{key}'. "
                                f"Regenerate with: python -m {__name__} --root ..."
                            )
                    path = e["path"]
                    if not os.path.isabs(path):
                        path = os.path.join(resolve_base, path)
                    version = int(e["version"])
                    out.append(
                        EpisodeMetadata(
                            length=int(e["length"]),
                            fps=float(e["fps"]) if e.get("fps") else 30.0,
                            robot_type=_EMBODIMENTS[rt]["robot_type"],
                            extras={
                                "path": path,
                                "version": version,
                                "subset": e["subset"],
                                "robot_type_key": rt,
                            },
                        )
                    )

        assert out, f"No usable episodes under {self.data_path}"
        return out

    def _spec(self, episode_index: int) -> Dict:
        return _EMBODIMENTS[self._metadata[episode_index].extras["robot_type_key"]]

    @fork_safe_cache
    def _open_hdf5(self, path: str):
        return h5py.File(path, "r", libver="latest", swmr=True, locking=False, rdcc_nbytes=0)

    def _load_state(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotState:
        version = self._metadata[episode_index].extras["version"]
        rtk = self._metadata[episode_index].extras["robot_type_key"]
        if version == 2:
            return self._load_proprio_v2("puppet", episode_index, frame_index, RobotState)
        if rtk == "tienkung_gello":
            return self._load_tienkung_gello("puppet", episode_index, frame_index, RobotState)
        if rtk == "tienkung_xsens":
            return self._load_tienkung_xsens("puppet", episode_index, frame_index, RobotState)
        if rtk == "franka_fr3_dual":
            return self._load_franka_fr3_dual("puppet", episode_index, frame_index, RobotState)
        if rtk == "sim_franka":
            return self._load_sim_franka("puppet", episode_index, frame_index, RobotState)
        if rtk == "sim_tienkung":
            return self._load_sim_tienkung("puppet", episode_index, frame_index, RobotState)
        return self._load_proprio_v1("puppet", episode_index, frame_index, RobotState)

    def _load_action(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> RobotAction:
        version = self._metadata[episode_index].extras["version"]
        rtk = self._metadata[episode_index].extras["robot_type_key"]
        if version == 2:
            return self._load_proprio_v2("master", episode_index, frame_index, RobotAction)
        if rtk == "tienkung_gello":
            return self._load_tienkung_gello("master", episode_index, frame_index, RobotAction)
        if rtk == "tienkung_xsens":
            return self._load_tienkung_xsens("master", episode_index, frame_index, RobotAction)
        if rtk == "franka_fr3_dual":
            return self._load_franka_fr3_dual("master", episode_index, frame_index, RobotAction)
        if rtk == "sim_franka":
            return self._load_sim_franka("master", episode_index, frame_index, RobotAction)
        if rtk == "sim_tienkung":
            return self._load_sim_tienkung("master", episode_index, frame_index, RobotAction)
        return self._load_proprio_v1("master", episode_index, frame_index, RobotAction)

    def _load_proprio_v2(self, who, episode_index, frame_index, cls):
        f = self._open_hdf5(self._metadata[episode_index].extras["path"])
        spec = self._spec(episode_index)
        n = spec["n_joints"]
        arms, grippers = [], []
        for side in ("left", "right"):
            ja = f[f"{who}/arm_{side}_position_align/data"][frame_index][:, :n]
            gp = f[f"{who}/end_effector_{side}_position_align/data"][frame_index]
            gp = np.asarray(gp, dtype=np.float32).reshape(len(ja), -1)
            if spec.get("gripper_invert"):
                # raw franka gripper is 0=open, ~1=closed; flip to 1=open, 0=closed
                gp = 1.0 - np.clip(gp, 0.0, 1.0)
            kw = dict(joint_position=Position(torch.from_numpy(ja).float()))
            eef_key = f"{who}/end_effector_{side}_pose_align/data"
            if eef_key in f and f[eef_key].shape[-1] >= 7:
                ee = f[eef_key][frame_index]
                kw["eef_position"] = Position(torch.from_numpy(np.asarray(ee[:, :3], dtype=np.float32)))
                kw["eef_rotation"] = Rotation(
                    torch.from_numpy(np.asarray(ee[:, 3:7], dtype=np.float32)),
                    representation=RotationRepresentation.QUAT_WXYZ,
                )
            arms.append(Arm(**kw))
            grippers.append(Position(torch.from_numpy(gp), allow_relative=False))
        return cls(left_arm=arms[0], right_arm=arms[1], left_gripper=grippers[0], right_gripper=grippers[1])

    def _load_proprio_v1(self, who, episode_index, frame_index, cls):
        f = self._open_hdf5(self._metadata[episode_index].extras["path"])
        spec = self._spec(episode_index)
        n = spec["n_joints"]
        dual = "puppet/joint_position_left" in f
        arms, grippers = [], []
        for suf in ("_left", "_right") if dual else ("",):
            jp = f[f"{who}/joint_position{suf}"][frame_index]
            kw = dict(joint_position=Position(torch.from_numpy(jp[:, :n]).float()))
            ee_key = f"{who}/end_effector{suf}"
            if ee_key in f:
                ee = f[ee_key][frame_index]
                kw["eef_position"] = Position(torch.from_numpy(ee[:, :3].astype(np.float32)))
                kw["eef_rotation"] = Rotation(
                    torch.from_numpy(ee[:, 3:6].astype(np.float32)),
                    representation=RotationRepresentation.EULER_XYZ,
                )
            elif who == "master":
                eef_pos, eef_rpy = self._fk_to_recorded(episode_index, jp[:, :n])
                kw["eef_position"] = Position(torch.from_numpy(eef_pos))
                kw["eef_rotation"] = Rotation(
                    torch.from_numpy(eef_rpy),
                    representation=RotationRepresentation.EULER_XYZ,
                )
            arms.append(Arm(**kw))
            gp = _normalize_gripper(jp[:, n : n + 1], spec, who)
            grippers.append(Position(torch.from_numpy(gp), allow_relative=False))
        if dual:
            return cls(left_arm=arms[0], right_arm=arms[1], left_gripper=grippers[0], right_gripper=grippers[1])
        return cls(left_arm=arms[0], left_gripper=grippers[0])

    def _load_images(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> Dict[str, torch.Tensor]:
        f = self._open_hdf5(self._metadata[episode_index].extras["path"])
        version = self._metadata[episode_index].extras["version"]
        if version == 2:
            grp = f.get("camera_observations/color_images")
        else:
            grp = f.get("observations/rgb_images")
        out = {}
        if grp is None:
            return out
        path = self._metadata[episode_index].extras["path"]
        for cam in grp:
            ds = grp[cam]
            frames = []
            for i in frame_index:
                buf = np.asarray(ds[i], np.uint8)
                if buf.size == 0:
                    continue
                try:
                    frames.append(decode_image_bytes(buf.tobytes()))
                except Exception as exc:
                    raise ValueError(f"image decode failed on {path} cam={cam} frame={i}") from exc
            if frames:
                out[cam] = torch.from_numpy(np.stack(frames))
        return out

    def _load_instruction(
        self,
        episode_index: int,
        frame_index: List[int],
    ) -> List[str]:
        f = self._open_hdf5(self._metadata[episode_index].extras["path"])
        version = self._metadata[episode_index].extras["version"]
        if version == 2:
            ins = f["metadata"].attrs.get("language_instruction", "")
        else:
            ins = f["language_raw"][0] if "language_raw" in f else ""
        if isinstance(ins, bytes):
            ins = ins.decode()
        return [str(ins)] * len(frame_index)

    def _iter_episode(
        self,
        episode_index: int,
        source_ranges: List[tuple],
        include_images: bool = True,
    ) -> Iterator[Dict]:
        meta = self._metadata[episode_index]
        path = meta.extras["path"]
        version = meta.extras["version"]
        n_total = meta.length

        with h5py.File(path, "r", libver="latest", swmr=True, locking=False, rdcc_nbytes=0) as f:
            full_state = self._load_proprio_from_handle(
                f,
                "puppet",
                episode_index,
                slice(0, n_total),
                RobotState,
                version,
            )
            full_action = self._load_proprio_from_handle(
                f,
                "master",
                episode_index,
                slice(0, n_total),
                RobotAction,
                version,
            )

            if version == 2:
                ins = f["metadata"].attrs.get("language_instruction", "")
            else:
                ins = f["language_raw"][0] if "language_raw" in f else ""
            if isinstance(ins, bytes):
                ins = ins.decode()
            instruction = str(ins)

            if version == 2:
                img_grp = f.get("camera_observations/color_images") if include_images else None
            else:
                img_grp = f.get("observations/rgb_images") if include_images else None

            for start, end in source_ranges:
                images = None
                if include_images and img_grp is not None:
                    images = {}
                    for cam in img_grp:
                        buf = np.asarray(img_grp[cam][start], np.uint8)
                        if buf.size == 0:
                            continue
                        try:
                            images[cam] = torch.from_numpy(decode_image_bytes(buf.tobytes()))
                        except Exception as exc:
                            raise ValueError(f"image decode failed on {path} cam={cam} frame={start}") from exc
                yield {
                    "state": full_state[start : start + 1],
                    "action": full_action[start:end],
                    "instruction": instruction,
                    "images": images,
                }

    def _load_proprio_from_handle(self, f, who, episode_index, frame_index, cls, version):
        rtk = self._metadata[episode_index].extras["robot_type_key"]
        if version == 2:
            return self._load_proprio_v2_handle(f, who, episode_index, frame_index, cls)
        if rtk == "tienkung_gello":
            return self._load_tienkung_gello_handle(f, who, episode_index, frame_index, cls)
        if rtk == "tienkung_xsens":
            return self._load_tienkung_xsens_handle(f, who, episode_index, frame_index, cls)
        if rtk == "franka_fr3_dual":
            return self._load_franka_fr3_dual_handle(f, who, episode_index, frame_index, cls)
        if rtk == "sim_franka":
            return self._load_sim_franka_handle(f, who, episode_index, frame_index, cls)
        if rtk == "sim_tienkung":
            return self._load_sim_tienkung_handle(f, who, episode_index, frame_index, cls)
        return self._load_proprio_v1_handle(f, who, episode_index, frame_index, cls)

    def _load_proprio_v2_handle(self, f, who, episode_index, frame_index, cls):
        spec = self._spec(episode_index)
        n = spec["n_joints"]
        arms, grippers = [], []
        for side in ("left", "right"):
            ja = f[f"{who}/arm_{side}_position_align/data"][frame_index][:, :n]
            gp = f[f"{who}/end_effector_{side}_position_align/data"][frame_index]
            gp = np.asarray(gp, dtype=np.float32).reshape(len(ja), -1)
            if spec.get("gripper_invert"):
                # raw franka gripper is 0=open, ~1=closed; flip to 1=open, 0=closed
                gp = 1.0 - np.clip(gp, 0.0, 1.0)
            kw = dict(joint_position=Position(torch.from_numpy(ja).float()))
            eef_key = f"{who}/end_effector_{side}_pose_align/data"
            if eef_key in f and f[eef_key].shape[-1] >= 7:
                ee = f[eef_key][frame_index]
                kw["eef_position"] = Position(torch.from_numpy(np.asarray(ee[:, :3], dtype=np.float32)))
                kw["eef_rotation"] = Rotation(
                    torch.from_numpy(np.asarray(ee[:, 3:7], dtype=np.float32)),
                    representation=RotationRepresentation.QUAT_WXYZ,
                )
            arms.append(Arm(**kw))
            grippers.append(Position(torch.from_numpy(gp), allow_relative=False))
        return cls(left_arm=arms[0], right_arm=arms[1], left_gripper=grippers[0], right_gripper=grippers[1])

    def _load_proprio_v1_handle(self, f, who, episode_index, frame_index, cls):
        spec = self._spec(episode_index)
        n = spec["n_joints"]
        dual = "puppet/joint_position_left" in f
        arms, grippers = [], []
        for suf in ("_left", "_right") if dual else ("",):
            jp = f[f"{who}/joint_position{suf}"][frame_index]
            kw = dict(joint_position=Position(torch.from_numpy(jp[:, :n]).float()))
            ee_key = f"{who}/end_effector{suf}"
            if ee_key in f:
                ee = f[ee_key][frame_index]
                kw["eef_position"] = Position(torch.from_numpy(ee[:, :3].astype(np.float32)))
                kw["eef_rotation"] = Rotation(
                    torch.from_numpy(ee[:, 3:6].astype(np.float32)),
                    representation=RotationRepresentation.EULER_XYZ,
                )
            elif who == "master":
                eef_pos, eef_rpy = self._fk_to_recorded_from_handle(f, episode_index, jp[:, :n])
                kw["eef_position"] = Position(torch.from_numpy(eef_pos))
                kw["eef_rotation"] = Rotation(
                    torch.from_numpy(eef_rpy),
                    representation=RotationRepresentation.EULER_XYZ,
                )
            arms.append(Arm(**kw))
            gp = _normalize_gripper(jp[:, n : n + 1], spec, who)
            grippers.append(Position(torch.from_numpy(gp), allow_relative=False))
        if dual:
            return cls(left_arm=arms[0], right_arm=arms[1], left_gripper=grippers[0], right_gripper=grippers[1])
        return cls(left_arm=arms[0], left_gripper=grippers[0])

    # ── Tienkung GELLO (16d packed: L-arm7 + L-hand1 + R-arm7 + R-hand1) ────

    def _load_tienkung_gello(self, who, episode_index, frame_index, cls):
        f = self._open_hdf5(self._metadata[episode_index].extras["path"])
        return self._load_tienkung_gello_handle(f, who, episode_index, frame_index, cls)

    def _load_tienkung_gello_handle(self, f, who, episode_index, frame_index, cls):
        jp = f[f"{who}/joint_position"][frame_index]
        left_arm = Arm(joint_position=Position(torch.from_numpy(jp[:, :7]).float()))
        right_arm = Arm(joint_position=Position(torch.from_numpy(jp[:, 8:15]).float()))
        left_hand = Position(torch.from_numpy(jp[:, 7:8].astype(np.float32)), allow_relative=False)
        right_hand = Position(torch.from_numpy(jp[:, 15:16].astype(np.float32)), allow_relative=False)
        return cls(left_arm=left_arm, right_arm=right_arm, left_hand=left_hand, right_hand=right_hand)

    # ── Tienkung XSens (14d joint + 12d hand; puppet==master, frame-shift) ───

    def _load_tienkung_xsens(self, who, episode_index, frame_index, cls):
        f = self._open_hdf5(self._metadata[episode_index].extras["path"])
        actual_idx = frame_index
        if who == "master":
            n = self._metadata[episode_index].length
            actual_idx = [min(i + 1, n - 1) for i in frame_index]
        return self._build_tienkung_xsens(f, actual_idx, cls)

    def _load_tienkung_xsens_handle(self, f, who, episode_index, frame_index, cls):
        jp = f["puppet/joint_position"][frame_index]
        ee = f["puppet/end_effector"][frame_index]
        if who == "master":
            jp = np.concatenate([jp[1:], jp[-1:]], axis=0)
            ee = np.concatenate([ee[1:], ee[-1:]], axis=0)
        return self._build_tienkung_xsens_from_arrays(jp, ee, cls)

    def _build_tienkung_xsens(self, f, frame_index, cls):
        jp = f["puppet/joint_position"][frame_index]
        ee = f["puppet/end_effector"][frame_index]
        return self._build_tienkung_xsens_from_arrays(jp, ee, cls)

    @staticmethod
    def _build_tienkung_xsens_from_arrays(jp, ee, cls):
        left_arm = Arm(joint_position=Position(torch.from_numpy(jp[:, :7]).float()))
        right_arm = Arm(joint_position=Position(torch.from_numpy(jp[:, 7:14]).float()))
        left_hand = Position(torch.from_numpy(ee[:, :6].astype(np.float32)), allow_relative=False)
        right_hand = Position(torch.from_numpy(ee[:, 6:12].astype(np.float32)), allow_relative=False)
        return cls(left_arm=left_arm, right_arm=right_arm, left_hand=left_hand, right_hand=right_hand)

    # ── Franka FR3 dual (16d packed: L-arm7 + L-grip1 + R-arm7 + R-grip1) ──

    def _load_franka_fr3_dual(self, who, episode_index, frame_index, cls):
        f = self._open_hdf5(self._metadata[episode_index].extras["path"])
        return self._load_franka_fr3_dual_handle(f, who, episode_index, frame_index, cls)

    def _load_franka_fr3_dual_handle(self, f, who, episode_index, frame_index, cls):
        spec = self._spec(episode_index)
        jp = f[f"{who}/joint_position"][frame_index]
        arms, grippers = [], []
        for arm_idx, (j_start, g_col) in enumerate([(0, 7), (8, 15)]):
            joints = jp[:, j_start : j_start + 7]
            kw = dict(joint_position=Position(torch.from_numpy(joints).float()))
            ee_key = f"{who}/end_effector"
            if ee_key in f:
                ee = f[ee_key][frame_index]
                ee_arm = ee[:, arm_idx * 6 : (arm_idx + 1) * 6]
                kw["eef_position"] = Position(torch.from_numpy(ee_arm[:, :3].astype(np.float32)))
                kw["eef_rotation"] = Rotation(
                    torch.from_numpy(ee_arm[:, 3:6].astype(np.float32)),
                    representation=RotationRepresentation.EULER_XYZ,
                )
            elif who == "master":
                eef_pos, eef_rpy = self._fk_fr3_dual_arm(f, episode_index, joints, arm_idx)
                kw["eef_position"] = Position(torch.from_numpy(eef_pos))
                kw["eef_rotation"] = Rotation(
                    torch.from_numpy(eef_rpy),
                    representation=RotationRepresentation.EULER_XYZ,
                )
            arms.append(Arm(**kw))
            gp = _normalize_gripper(jp[:, g_col : g_col + 1], spec, who)
            grippers.append(Position(torch.from_numpy(gp), allow_relative=False))
        return cls(left_arm=arms[0], right_arm=arms[1], left_gripper=grippers[0], right_gripper=grippers[1])

    def _fk_fr3_dual_arm(self, f, episode_index, master_joints, arm_idx):
        N = self._metadata[episode_index].length
        cidx = np.unique(np.linspace(0, N - 1, min(N, 16)).astype(int))
        j_start = arm_idx * 8
        ee_start = arm_idx * 6
        pj = f["puppet/joint_position"][cidx][:, j_start : j_start + 7]
        ee = f["puppet/end_effector"][cidx][:, ee_start : ee_start + 6]
        fp, fm = self._fk(RobotType.FRANKA, pj)
        rp = ee[:, :3]
        rm = ScipyRotation.from_euler("xyz", ee[:, 3:6]).as_matrix()
        R_tool = ScipyRotation.from_matrix(np.matmul(rm.transpose(0, 2, 1), fm)).mean().as_matrix()
        A = np.zeros((3 * len(cidx), 6))
        A[:, :3] = np.tile(np.eye(3), (len(cidx), 1))
        A[:, 3:] = rm.reshape(-1, 3)
        sol, *_ = np.linalg.lstsq(A, (fp - rp).reshape(-1), rcond=None)
        cal = {"t_base": sol[:3], "c": sol[3:], "R_tool": R_tool}
        fp, fm = self._fk(RobotType.FRANKA, master_joints)
        R_rec = fm @ cal["R_tool"].T
        p_rec = fp - cal["t_base"] - np.einsum("nij,j->ni", R_rec, cal["c"])
        rpy = ScipyRotation.from_matrix(R_rec).as_euler("xyz")
        return p_rec.astype(np.float32), rpy.astype(np.float32)

    # ── Sim helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _sim_resample_idx(n_img, n_prop):
        if n_prop == n_img:
            return np.arange(n_img)
        ratio = (n_prop - 1) / max(n_img - 1, 1)
        return np.clip(np.round(np.arange(n_img) * ratio).astype(int), 0, n_prop - 1)

    # ── Sim Franka (single-arm, franka/ schema, no puppet/master) ───────────

    def _load_sim_franka(self, who, episode_index, frame_index, cls):
        f = self._open_hdf5(self._metadata[episode_index].extras["path"])
        n_img = self._metadata[episode_index].length
        n_prop = f["franka/joint_position"].shape[0]
        resamp = self._sim_resample_idx(n_img, n_prop)
        prop_idx = resamp[frame_index]
        if who == "master":
            n = self._metadata[episode_index].length
            next_img = [min(i + 1, n - 1) for i in frame_index]
            prop_idx_next = resamp[next_img]
            return self._build_sim_franka(f, prop_idx_next, cls)
        return self._build_sim_franka(f, prop_idx, cls)

    def _load_sim_franka_handle(self, f, who, episode_index, frame_index, cls):
        n_img = self._metadata[episode_index].length
        n_prop = f["franka/joint_position"].shape[0]
        resamp = self._sim_resample_idx(n_img, n_prop)
        jp_all = f["franka/joint_position"][:][resamp]
        ee_all = f["franka/end_effector"][:][resamp] if "franka/end_effector" in f else None
        if who == "master":
            jp_all = np.concatenate([jp_all[1:], jp_all[-1:]], axis=0)
            if ee_all is not None:
                ee_all = np.concatenate([ee_all[1:], ee_all[-1:]], axis=0)
        jp = jp_all[frame_index]
        ee = ee_all[frame_index] if ee_all is not None else None
        return self._build_sim_franka_from_arrays(jp, ee, cls)

    def _build_sim_franka(self, f, prop_idx, cls):
        jp = f["franka/joint_position"][prop_idx]
        ee = f["franka/end_effector"][prop_idx] if "franka/end_effector" in f else None
        return self._build_sim_franka_from_arrays(jp, ee, cls)

    @staticmethod
    def _build_sim_franka_from_arrays(jp, ee, cls):
        kw = dict(joint_position=Position(torch.from_numpy(jp[:, :7]).float()))
        if ee is not None:
            kw["eef_position"] = Position(torch.from_numpy(ee[:, :3].astype(np.float32)))
            quat_xyzw = ee[:, 3:7].astype(np.float32)
            quat_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=1)
            kw["eef_rotation"] = Rotation(
                torch.from_numpy(quat_wxyz),
                representation=RotationRepresentation.QUAT_WXYZ,
            )
        arm = Arm(**kw)
        # Sim gripper uses inverted convention: 0 ≈ open, ~0.8 ≈ closed.
        gp = 1.0 - np.clip(jp[:, 7:8].astype(np.float32), 0.0, 1.0)
        gripper = Position(torch.from_numpy(gp), allow_relative=False)
        return cls(left_arm=arm, left_gripper=gripper)

    # ── Sim Tienkung (dual-arm, tiangong/ schema, no puppet/master) ────────

    def _load_sim_tienkung(self, who, episode_index, frame_index, cls):
        f = self._open_hdf5(self._metadata[episode_index].extras["path"])
        n_img = self._metadata[episode_index].length
        channels = [
            f["tiangong/left_arm_joint_pos_seq"],
            f["tiangong/right_arm_joint_pos_seq"],
            f["tiangong/left_hand_joint_pos_seq"],
            f["tiangong/right_hand_joint_pos_seq"],
            f["tiangong/left_end_effector_waist"],
            f["tiangong/right_end_effector_waist"],
        ]
        n_prop = min(ch.shape[0] for ch in channels)
        resamp = self._sim_resample_idx(n_img, n_prop)
        prop_idx = resamp[frame_index]
        if who == "master":
            n = self._metadata[episode_index].length
            next_img = [min(i + 1, n - 1) for i in frame_index]
            prop_idx_next = resamp[next_img]
            return self._build_sim_tienkung(channels, n_prop, prop_idx_next, cls)
        return self._build_sim_tienkung(channels, n_prop, prop_idx, cls)

    def _load_sim_tienkung_handle(self, f, who, episode_index, frame_index, cls):
        n_img = self._metadata[episode_index].length
        channels = [
            f["tiangong/left_arm_joint_pos_seq"],
            f["tiangong/right_arm_joint_pos_seq"],
            f["tiangong/left_hand_joint_pos_seq"],
            f["tiangong/right_hand_joint_pos_seq"],
            f["tiangong/left_end_effector_waist"],
            f["tiangong/right_end_effector_waist"],
        ]
        n_prop = min(ch.shape[0] for ch in channels)
        resamp = self._sim_resample_idx(n_img, n_prop)
        arrays = [ch[:n_prop][resamp] for ch in channels]
        if who == "master":
            arrays = [np.concatenate([a[1:], a[-1:]], axis=0) for a in arrays]
        la, ra, lh, rh, lee, ree = [a[frame_index] for a in arrays]
        return self._build_sim_tienkung_from_arrays(la, ra, lh, rh, lee, ree, cls)

    @staticmethod
    def _build_sim_tienkung(channels, n_prop, prop_idx, cls):
        la = channels[0][prop_idx][:, :7]
        ra = channels[1][prop_idx][:, :7]
        lh = channels[2][prop_idx]
        rh = channels[3][prop_idx]
        lee = channels[4][prop_idx]
        ree = channels[5][prop_idx]
        return RoboMINDDataset._build_sim_tienkung_from_arrays(la, ra, lh, rh, lee, ree, cls)

    @staticmethod
    def _build_sim_tienkung_from_arrays(la, ra, lh, rh, lee, ree, cls):
        def _make_arm(joints, ee):
            kw = dict(joint_position=Position(torch.from_numpy(joints.astype(np.float32))))
            if ee is not None and ee.shape[-1] >= 7:
                kw["eef_position"] = Position(torch.from_numpy(ee[:, :3].astype(np.float32)))
                quat_xyzw = ee[:, 3:7].astype(np.float32)
                quat_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=1)
                kw["eef_rotation"] = Rotation(
                    torch.from_numpy(quat_wxyz),
                    representation=RotationRepresentation.QUAT_WXYZ,
                )
            return Arm(**kw)

        left_arm = _make_arm(la, lee)
        right_arm = _make_arm(ra, ree)
        left_hand = Position(torch.from_numpy(lh.astype(np.float32)), allow_relative=False)
        right_hand = Position(torch.from_numpy(rh.astype(np.float32)), allow_relative=False)
        return cls(left_arm=left_arm, right_arm=right_arm, left_hand=left_hand, right_hand=right_hand)

    # ── V1 FK calibration (franka / ur5 master eef recovery) ─────────────────

    def _fk(self, robot_type: RobotType, joints: np.ndarray):
        if robot_type == RobotType.FRANKA:
            return _serial_fk(joints.astype(np.float64), _FRANKA_CHAIN, _FRANKA_FLANGE_OFFSET)
        if robot_type == RobotType.UR_5E:
            return _serial_fk(
                joints.astype(np.float64), _UR5E_CHAIN, _UR5E_FLANGE_OFFSET, _UR5E_FLANGE_ROTATION, _UR5E_JOINT_AXES
            )
        raise ValueError(f"unsupported robot_type for FK: {robot_type}")

    @fork_safe_cache
    def _calib(self, episode_index: int) -> Dict:
        spec = self._spec(episode_index)
        f = self._open_hdf5(self._metadata[episode_index].extras["path"])
        n = spec["n_joints"]
        N = self._metadata[episode_index].length
        cidx = np.unique(np.linspace(0, N - 1, min(N, 16)).astype(int))
        pj = f["puppet/joint_position"][cidx][:, :n]
        ee = f["puppet/end_effector"][cidx]
        fp, fm = self._fk(spec["robot_type"], pj)
        rp = ee[:, :3]
        rm = ScipyRotation.from_euler("xyz", ee[:, 3:6]).as_matrix()

        R_tool = ScipyRotation.from_matrix(np.matmul(rm.transpose(0, 2, 1), fm)).mean().as_matrix()
        A = np.zeros((3 * len(cidx), 6))
        A[:, :3] = np.tile(np.eye(3), (len(cidx), 1))
        A[:, 3:] = rm.reshape(-1, 3)
        sol, *_ = np.linalg.lstsq(A, (fp - rp).reshape(-1), rcond=None)
        return {"t_base": sol[:3], "c": sol[3:], "R_tool": R_tool}

    def _fk_to_recorded(self, episode_index: int, joints: np.ndarray):
        spec = self._spec(episode_index)
        cal = self._calib(episode_index)
        fp, fm = self._fk(spec["robot_type"], joints)
        R_rec = fm @ cal["R_tool"].T
        p_rec = fp - cal["t_base"] - np.einsum("nij,j->ni", R_rec, cal["c"])
        rpy = ScipyRotation.from_matrix(R_rec).as_euler("xyz")
        return p_rec.astype(np.float32), rpy.astype(np.float32)

    def _fk_to_recorded_from_handle(self, f, episode_index: int, joints: np.ndarray):
        spec = self._spec(episode_index)
        n = spec["n_joints"]
        N = self._metadata[episode_index].length
        cidx = np.unique(np.linspace(0, N - 1, min(N, 16)).astype(int))
        pj = f["puppet/joint_position"][cidx][:, :n]
        ee = f["puppet/end_effector"][cidx]
        fp_c, fm_c = self._fk(spec["robot_type"], pj)
        rp = ee[:, :3]
        rm = ScipyRotation.from_euler("xyz", ee[:, 3:6]).as_matrix()
        R_tool = ScipyRotation.from_matrix(np.matmul(rm.transpose(0, 2, 1), fm_c)).mean().as_matrix()
        A = np.zeros((3 * len(cidx), 6))
        A[:, :3] = np.tile(np.eye(3), (len(cidx), 1))
        A[:, 3:] = rm.reshape(-1, 3)
        sol, *_ = np.linalg.lstsq(A, (fp_c - rp).reshape(-1), rcond=None)
        cal = {"t_base": sol[:3], "c": sol[3:], "R_tool": R_tool}

        fp, fm = self._fk(spec["robot_type"], joints)
        R_rec = fm @ cal["R_tool"].T
        p_rec = fp - cal["t_base"] - np.einsum("nij,j->ni", R_rec, cal["c"])
        rpy = ScipyRotation.from_matrix(R_rec).as_euler("xyz")
        return p_rec.astype(np.float32), rpy.astype(np.float32)


RoboMINDV1Dataset = RoboMINDDataset
RoboMINDV2Dataset = RoboMINDDataset


if __name__ == "__main__":
    import argparse

    # ── Robot type mapping ──────────────────────────────────────────────────

    _DIR_TO_ROBOT = {
        # v2 dirs
        "agilex": "agilex_cobot_magic",
        "agilex_mobile": "agilex_cobot_magic",
        "ark": "ark",
        "ark_mobile": "ark",
        "franka": "dual_franka",  # v2 dual-arm rig (v1 single: h5_franka_1rgb/3rgb)
        "franka_sim": "franka",
        "tienkung": "tienkung",
        "tienkung_sim": "tienkung",
        "tienyi": "tianyi",
        "tienyi_mobile": "tianyi",
        "ur": "ur5_dual",
        "ur_dex": "ur5_dex",
        # v1 dirs
        "h5_agilex_3rgb": "agilex_cobot_magic",
        "h5_franka_1rgb": "franka",
        "h5_franka_3rgb": "franka",
        "h5_franka_fr3_dual": "franka_fr3_dual",
        "h5_sim_franka_3rgb": "sim_franka",
        "h5_sim_tienkung_1rgb": "sim_tienkung",
        "h5_simulation": "sim_franka",
        "h5_tienkung_gello_1rgb": "tienkung_gello",
        "h5_tienkung_prod1_gello_1rgb": "tienkung_gello",
        "h5_tienkung_xsens_1rgb": "tienkung_xsens",
        "h5_ur_1rgb": "ur5",
    }

    def _infer_robot_type(subset_name: str) -> str:
        low = subset_name.lower()
        if low in _DIR_TO_ROBOT:
            return _DIR_TO_ROBOT[low]
        if "franka" in low:
            base = "franka"
        elif "ur" in low:
            base = "ur5"
        elif "agilex" in low:
            base = "agilex_cobot_magic"
        elif "ark" in low:
            base = "ark"
        elif "tianyi" in low or "tienyi" in low:
            base = "tianyi"
        elif "tienkung" in low:
            base = "tienkung"
        else:
            raise ValueError(f"cannot infer robot_type from subset: {subset_name}")
        if "dex" in low:
            return base + "_dex"
        return base

    # ── Common helpers ───────────────────────────────────────────────────────

    def _first_dim(group):
        for name in sorted(group):
            node = group[name]
            if isinstance(node, h5py.Dataset) and node.ndim >= 1:
                return int(node.shape[0])
        return None

    def _episode_fps(ts_ds):
        if ts_ds is None or ts_ds.ndim < 1 or ts_ds.shape[0] < 2:
            return None
        ts = ts_ds[:].astype(np.float64)
        span = ts[-1] - ts[0]
        if span <= 0:
            return None
        unit_s = 1.0 if ts[0] > 1e9 else 1e-3
        return round(float((ts.shape[0] - 1) / (span * unit_s)), 3)

    def _proprio_len_v1(f):
        hits = []

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim >= 1:
                low = name.lower()
                if "joint_position" in low:
                    hits.append((0, obj.shape[0]))
                elif "end_effector" in low:
                    hits.append((1, obj.shape[0]))

        f.visititems(visit)
        if not hits:
            return None
        hits.sort()
        return int(hits[0][1])

    def _proprio_len_v2(f):
        hits = []

        def visit(name, obj):
            if not (isinstance(obj, h5py.Dataset) and obj.ndim >= 1):
                return
            low = name.lower()
            if not (low.endswith("/data") and "position_align" in low):
                return
            if "end_effector" in low:
                return
            if "puppet/arm" in low:
                hits.append((0, obj.shape[0]))
            elif "arm" in low:
                hits.append((1, obj.shape[0]))
            else:
                hits.append((2, obj.shape[0]))

        f.visititems(visit)
        if not hits:
            return None
        hits.sort()
        return int(hits[0][1])

    def _scan_one(path: str, default_fps: float = 30.0):
        try:
            with h5py.File(path, "r", libver="latest", swmr=True, locking=False, rdcc_nbytes=0) as f:
                coll = f["metadata"].attrs.get("collector", "") if "metadata" in f else ""
                if isinstance(coll, bytes):
                    coll = coll.decode(errors="replace")
                coll = str(coll)
                # v2 schema
                rgb = f.get("camera_observations/color_images")
                if rgb is not None:
                    length = _first_dim(rgb)
                    if length is None:
                        return None
                    fps = _episode_fps(f.get("camera_observations/timestamp"))
                    proprio = _proprio_len_v2(f)
                    return {
                        "path": path,
                        "version": 2,
                        "length": int(length),
                        "fps": fps or default_fps,
                        "aligned": proprio is not None and int(proprio) == int(length),
                        "collector": coll,
                    }
                # v1 schema
                rgb = f.get("observations/rgb_images")
                if rgb is not None:
                    length = _first_dim(rgb)
                    if length is None:
                        return None
                    is_sim = "puppet" not in f and ("franka" in f or "tiangong" in f)
                    if is_sim:
                        fps = default_fps
                        ts = f.get("sim_time_seq")
                        if ts is not None and ts.ndim >= 1 and ts.shape[0] >= 2:
                            t = ts[:].astype(np.float64)
                            span = t[-1] - t[0]
                            if span > 0:
                                fps = round(float((length - 1) / span), 3)
                        return {
                            "path": path,
                            "version": 1,
                            "length": int(length),
                            "fps": fps,
                            "aligned": True,
                            "collector": coll,
                        }
                    proprio = _proprio_len_v1(f)
                    return {
                        "path": path,
                        "version": 1,
                        "length": int(length),
                        "fps": default_fps,
                        "aligned": proprio is not None and int(proprio) == int(length),
                        "collector": coll,
                    }
                return None
        except Exception:
            return None

    # ── Discovery & scanning ─────────────────────────────────────────────────

    def _discover_subsets(root: str, only) -> List[str]:
        data_dir = os.path.join(root, "data")
        search_root = data_dir if os.path.isdir(data_dir) else root
        subs: List[str] = []
        for name in sorted(os.listdir(search_root)):
            if only and name not in only:
                continue
            p = os.path.join(search_root, name)
            if os.path.isdir(p):
                subs.append(p)
        return subs

    def _find_hdf5(subset_root: str) -> List[str]:
        out: List[str] = []
        for dirpath, _dirs, files in os.walk(subset_root):
            if os.sep + "success_episodes" + os.sep not in dirpath + os.sep:
                continue
            for fn in files:
                if fn.endswith(".hdf5"):
                    out.append(os.path.join(dirpath, fn))
        out.sort()
        return out

    def _scan(root: str, workers: int, absolute: bool, only, default_fps: float):
        from functools import partial

        root = os.path.abspath(root)
        subsets = _discover_subsets(root, set(only) if only else None)
        assert subsets, f"No subset dirs found under {root}"
        print(f"Scanning {len(subsets)} subset(s) under {root}")
        grand = 0
        scan_fn = partial(_scan_one, default_fps=default_fps)
        for sub in subsets:
            paths = _find_hdf5(sub)
            subset_name = os.path.basename(sub)
            print(f"  {subset_name}: {len(paths)} hdf5 files")
            if not paths:
                continue
            records = [r for r in mp_process(scan_fn, paths, max_workers=workers, desc=f"Scanning {subset_name}") if r]
            robot_type = _infer_robot_type(subset_name)
            output = os.path.join(sub, INDEX_NAME)
            with open(output, "w") as fout:
                for idx, rec in enumerate(records):
                    stored = rec["path"] if absolute else os.path.relpath(rec["path"], sub)
                    fout.write(
                        json.dumps(
                            {
                                "episode_index": idx,
                                "path": stored,
                                "subset": subset_name,
                                "version": rec["version"],
                                "robot_type": robot_type,
                                "length": rec["length"],
                                "fps": rec["fps"],
                                "aligned": rec["aligned"],
                                "collector": rec.get("collector", ""),
                            }
                        )
                        + "\n"
                    )
            misaligned = sum(1 for r in records if not r["aligned"])
            print(
                f"  wrote {len(records)} -> {output}  "
                f"(robot_type={robot_type}, {misaligned} misaligned, "
                f"{len(paths) - len(records)} skipped)"
            )
            grand += len(records)
        print(f"\nTotal: {grand} episodes across {len(subsets)} subset(s)")

    # ── CLI entry ────────────────────────────────────────────────────────────

    ap = argparse.ArgumentParser(
        description="Build RoboMIND episodes.jsonl indexes (per-subset).",
    )
    ap.add_argument("--root", required=True, help="dataset root (auto-detects v1/v2 HDF5 schema)")
    ap.add_argument("--fps", type=float, default=30.0, help="default fps when timestamps unavailable (default: 30)")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--absolute", action="store_true", help="store absolute paths instead of relative")
    ap.add_argument("--subset", nargs="+", default=None, help="subset dir names to scan; default: all")
    args = ap.parse_args()

    _scan(args.root, args.workers, args.absolute, args.subset, args.fps)
