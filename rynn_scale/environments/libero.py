import os
from contextlib import contextmanager
from typing import Any, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyRotation

from ..constants import RobotType, RotationRepresentation
from ..datasets.vla_datasets.libero import (
    BASE_POS,
    GRIPPER_OPEN_WIDTH,
    OSC_POS_SCALE,
    OSC_ROT_SCALE,
)
from ..registry import ENVIRONMENT_REGISTRY
from ..utils.robot import (
    Arm,
    Position,
    RobotState,
    Rotation,
)
from .robot import SimRobotEnvironment

# The robosuite / LIBERO conventions this env is driven by are imported from the
# dataset that decodes recordings of it, so the two cannot drift apart.
_BASE_POS = np.array(BASE_POS, dtype=np.float32)


@contextmanager
def numpy_safe_globals():
    """Let ``torch.load(weights_only=True)`` rebuild pickles of plain numpy arrays."""
    try:
        from numpy._core.multiarray import _reconstruct  # numpy >= 2
    except ImportError:
        from numpy.core.multiarray import _reconstruct

    allowlist = [
        # ``safe_globals`` matches on the path recorded in the pickle, which is
        # ``numpy.core`` for files written by numpy 1.x and ``numpy._core`` for 2.x --
        # neither necessarily the installed module's own path, so pin both.
        (_reconstruct, "numpy.core.multiarray._reconstruct"),
        (_reconstruct, "numpy._core.multiarray._reconstruct"),
        np.ndarray,
        np.dtype,
        # A pickled array carries its concrete dtype class, not just ``np.dtype``.
        *(getattr(np.dtypes, name) for name in np.dtypes.__all__),
    ]
    with torch.serialization.safe_globals(allowlist):
        yield


def _mute_cameras(env) -> list:
    inner = env.env
    cameras = [n for n in inner.observation_names if n.endswith("_image")]
    for name in cameras:
        inner.modify_observable(name, "enabled", False)
    return cameras


@ENVIRONMENT_REGISTRY.register()
class Libero(SimRobotEnvironment):
    robot_type = RobotType.FRANKA.value
    image_key_map = {"main": "main", "wrist": "wrist"}
    num_warmup_steps: int = 10

    @property
    def action_layout(self):
        return [
            {
                "path": ["left_arm", "eef_position"],
                "type": "Position",
                "dim": 3,
                "labels": ["eef_x", "eef_y", "eef_z"],
            },
            {
                "path": ["left_arm", "eef_rotation"],
                "type": "Rotation",
                "dim": 3,
                "representation": RotationRepresentation.ROT_VEC.value,
                "labels": ["eef_rx", "eef_ry", "eef_rz"],
            },
            {"path": ["left_gripper"], "type": "Position", "dim": 1, "labels": ["gripper"], "allow_relative": False},
        ]

    @property
    def state_layout(self):
        return self.action_layout

    @property
    def fps(self) -> int:
        return 20

    def __init__(
        self,
        bddl_file_name: Optional[str] = None,
        suite: Optional[str] = None,
        task_id: int = 0,
        height: int = 256,
        width: int = 256,
        data_reader: Any = None,
    ):
        self._suite, self._task_id = suite, task_id
        self._data_reader = data_reader

        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        if bddl_file_name is None:
            assert suite is not None, (
                "Libero needs either 'bddl_file_name' or a 'suite' to resolve it from the benchmark task suite."
            )

            task = self._task_suite(suite).get_task(task_id)
            bddl_file_name = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

        os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl_file_name,
            camera_heights=height,
            camera_widths=width,
        )
        self.env.seed(0)
        self.env.reset()
        self._cameras = _mute_cameras(self.env)
        self._state_cache: Optional[tuple] = None

    def _reset(
        self,
        init_state: Optional[np.ndarray] = None,
        init_index: int = 0,
        **kwargs,
    ):
        if init_state is None:
            assert self._suite is not None, (
                "Libero reset needs either an 'init_state' or a ctor 'suite' to resolve an 'init_index' against."
            )
            # LIBERO's ``get_task_init_states`` torch.load()s a pickled numpy array,
            # which ``weights_only=True`` (torch>=2.6 default) refuses without an
            # allowlist for the numpy globals it rebuilds.
            with numpy_safe_globals():
                init_states = self._task_suite(self._suite).get_task_init_states(self._task_id)

            init_state = init_states[init_index % init_states.shape[0]]
        self.env.reset()
        # robosuite rebuilds its observables in ``reset``, so the ctor's mute has
        # lapsed and has to be re-applied for this episode.
        self._cameras = _mute_cameras(self.env)

        self.env.set_init_state(init_state)
        # Warm up with zero-delta OSC commands (no movement). Nothing reads a frame
        # from these, and the cameras are muted, so they cost physics only.
        dummy_action = np.zeros(7, dtype=np.float32)
        dummy_action[6] = 1.0  # gripper close (+1 = close in robosuite)
        for _ in range(self.num_warmup_steps):
            self.env.step(dummy_action)

        # ``reset`` restarts the sim clock, so an entry from the previous episode would
        # sit on a timestamp this episode reaches again -- and be served as current.
        self._state_cache = None

    def _step(self, action: np.ndarray):
        # One flat ``[eef_pos(3, base_link), eef_rotvec(3), gripper(1)]`` vector.
        # Policy actions are canonicalized to ROT_VEC and flattened by
        # :meth:`~BaseRobotEnvironment.flatten_action` (over ``action_layout``);
        # manual MOVE/REPLAY chunks arrive in the same layout.
        flat = np.asarray(action, dtype=np.float32)
        target_pos_base = flat[0:3]  # base_link frame
        target_rot = ScipyRotation.from_rotvec(flat[3:6])
        # Gripper: model outputs [0, 1] (0=closed, 1=open); robosuite wants
        # [-1, 1] (+1=close, -1=open).
        grip = np.clip(-(flat[6:7] * 2.0 - 1.0), -1.0, 1.0)

        cur_pos, cur_rot, _ = self._read_proprio()  # world frame, as of now

        target_pos_world = target_pos_base + _BASE_POS
        pos_delta = np.clip((target_pos_world - cur_pos) / OSC_POS_SCALE, -1.0, 1.0)
        rot_delta_rv = np.clip((target_rot * cur_rot.inv()).as_rotvec() / OSC_ROT_SCALE, -1.0, 1.0)
        action_7d = np.concatenate([pos_delta, rot_delta_rv, grip])

        try:
            # The obs is dropped: the cameras are muted so it carries no frame, and its
            # proprio entries lag the step's end (robosuite samples them inside the
            # substep loop) -- the pose is read off mujoco where anyone asks for it.
            _, _, done, _ = self.env.step(action_7d)
        except Exception as e:  # noqa: BLE001 - surface sim errors as episode error
            self.env.close()
            return False, str(e)

        return bool(done), None

    def get_images(self):
        inner = self.env.env
        # The render is what an *enabled* camera observable costs, so ask for it by
        # enabling them across this one forced update and muting them again straight
        # after -- stepping stays render-free either side of it. Going through the
        # observables rather than ``sim.render`` directly keeps robosuite's own camera
        # setup and image conventions, which the flip below is matched to.
        for name in self._cameras:
            inner.modify_observable(name, "enabled", True)
        try:
            obs = inner._get_observations(force_update=True)
        finally:
            for name in self._cameras:
                inner.modify_observable(name, "enabled", False)

        return {
            "main": obs["agentview_image"][::-1],
            "wrist": obs["robot0_eye_in_hand_image"][::-1],
        }

    def get_state(self) -> RobotState:
        now = float(self.env.env.sim.data.time)
        if self._state_cache is not None and self._state_cache[0] == now:
            return self._state_cache[1]

        eef_pos_world, eef_rot, gripper_width = self._read_proprio()
        eef_pos_base = (eef_pos_world - _BASE_POS).astype(np.float32)
        eef_rotvec = eef_rot.as_rotvec().astype(np.float32)
        gripper_norm = np.clip(np.array([gripper_width / GRIPPER_OPEN_WIDTH], dtype=np.float32), 0.0, 1.0)

        state = RobotState(
            left_arm=Arm(
                eef_position=Position(torch.from_numpy(eef_pos_base).unsqueeze(0).float()),
                eef_rotation=Rotation(
                    torch.from_numpy(eef_rotvec).unsqueeze(0).float(),
                    representation=RotationRepresentation.ROT_VEC,
                ),
            ),
            left_gripper=Position(
                torch.from_numpy(gripper_norm).unsqueeze(0).float(),
                allow_relative=False,
            ),
        )
        self._state_cache = (now, state)
        return state

    def _read_proprio(self):
        robot = self.env.env.robots[0]
        data = robot.sim.data
        eef_pos = np.array(data.site_xpos[robot.eef_site_id], dtype=np.float64)
        # ``get_body_xquat`` is wxyz; robosuite's observable converts it to xyzw.
        eef_rot = ScipyRotation.from_quat(data.get_body_xquat(robot.robot_model.eef_name), scalar_first=True)
        gripper_width = float(data.qpos[robot._ref_gripper_joint_pos_indexes[0]])
        return eef_pos, eef_rot, gripper_width

    def close(self):
        if self.env is not None:
            self.env.close()
        self.env = None

    @staticmethod
    def _task_suite(suite: str):
        from libero.libero import benchmark

        return benchmark.get_benchmark_dict()[suite]()
