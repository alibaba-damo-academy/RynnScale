import os

import numpy as np

from ..constants import RobotType
from ..registry import RENDERER_REGISTRY
from ..utils.robot import Arm, RobotAction, _to_scipy_rotation
from ._lazy import mujoco
from .base import BaseRenderer

_LEFT_ARM_JOINTS = [
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "elbow_yaw_l_joint",
    "wrist_pitch_l_joint",
    "wrist_roll_l_joint",
]
_RIGHT_ARM_JOINTS = [
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "elbow_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joint",
]


@RENDERER_REGISTRY.register(name=RobotType.TIENKUNG_2.value)
class Tienkung2Renderer(BaseRenderer):
    """TienKung 2.0 Pro humanoid dual-arm renderer (joint playback).

    7-DOF arms, no dexterous hands. Full-body humanoid with legs, torso, head.
    """

    HOME_QPOS_ARM = np.array(
        [0.0, 0.3, 0.0, -0.8, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )

    N_ARM = 7

    def __init__(self, height: int = 480, width: int = 480, **_unused):
        super().__init__(height=height, width=width)

        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        self._arm_l_qadr = np.array(
            [self._joint_qadr(n) for n in _LEFT_ARM_JOINTS],
            dtype=np.int64,
        )
        self._arm_r_qadr = np.array(
            [self._joint_qadr(n) for n in _RIGHT_ARM_JOINTS],
            dtype=np.int64,
        )
        self._arm_l_jnt_range = self._jnt_ranges(_LEFT_ARM_JOINTS)
        self._arm_r_jnt_range = self._jnt_ranges(_RIGHT_ARM_JOINTS)

        self._arm_l_dofadr = np.array(
            [self._joint_dofadr(n) for n in _LEFT_ARM_JOINTS],
            dtype=np.int64,
        )
        self._arm_r_dofadr = np.array(
            [self._joint_dofadr(n) for n in _RIGHT_ARM_JOINTS],
            dtype=np.int64,
        )

        self._eef_l_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "wrist_roll_l_link",
        )
        self._eef_r_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "wrist_roll_r_link",
        )
        assert self._eef_l_body_id >= 0 and self._eef_r_body_id >= 0

        self._reset_home()
        mujoco.mj_forward(self.model, self.data)

        self.prev_qpos_l = self.HOME_QPOS_ARM.copy()
        self.prev_qpos_r = self.HOME_QPOS_ARM.copy()

        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0.0, 0.0, 0.5]
        self.camera.distance = 2.5
        self.camera.azimuth = 150
        self.camera.elevation = -15

        self.scene_option = mujoco.MjvOption()

    def _build_model(self):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(module_dir, "..", ".."))
        urdf_path = os.path.join(project_root, "assets", "tienkung_2", "tiangong2pro.urdf")
        assert os.path.exists(urdf_path), f"TienKung 2 Pro model not found: {urdf_path}"
        return mujoco.MjModel.from_xml_path(urdf_path)

    def _reset_home(self):
        self.data.qpos[self._arm_l_qadr] = self.HOME_QPOS_ARM
        self.data.qpos[self._arm_r_qadr] = self.HOME_QPOS_ARM

    def _apply_arm(self, arm: Arm, qadr, dofadr, jrange, body_id, prev_qpos):
        if arm.joint_position is not None:
            qpos = arm.joint_position.data[0].detach().cpu().numpy().astype(np.float64)
            assert qpos.shape == (self.N_ARM,), f"Tienkung2Renderer expects 7-DOF per arm, got shape {qpos.shape}"
            self.data.qpos[qadr] = np.clip(qpos, jrange[:, 0], jrange[:, 1])
            return self.data.qpos[qadr].copy()

        assert arm.eef_position is not None and arm.eef_rotation is not None, (
            "Tienkung2Renderer requires joint_position or full EEF pose"
        )
        eef_pos = arm.eef_position.data[0].detach().cpu().numpy().astype(np.float64)
        rot = _to_scipy_rotation(
            arm.eef_rotation.data.detach().cpu(),
            arm.eef_rotation.representation,
        )
        eef_rot_mat = rot.as_matrix()[0].astype(np.float64)
        self.solve_ik(
            target_pos=eef_pos,
            target_rot_mat=eef_rot_mat,
            qadr=qadr,
            dofadr=dofadr,
            jnt_range=jrange,
            body_id=body_id,
            seed_qpos=prev_qpos,
            null_space_gain=0.05,
            damping=1e-2,
        )
        return self.data.qpos[qadr].copy()

    def render(self, action: RobotAction) -> np.ndarray:
        assert len(action) == 1, f"render() expects single-step action, got chunk={len(action)}"
        for arm in (action.left_arm, action.right_arm):
            if arm is None:
                continue
            ref = arm.joint_position if arm.joint_position is not None else arm.eef_position
            assert ref is not None, "Tienkung2Renderer requires either joint_position or eef pose"
            assert not ref.is_relative, "render() expects an absolute action; convert delta actions first"

        if action.left_arm is not None:
            self.prev_qpos_l = self._apply_arm(
                action.left_arm,
                self._arm_l_qadr,
                self._arm_l_dofadr,
                self._arm_l_jnt_range,
                self._eef_l_body_id,
                self.prev_qpos_l,
            )
        else:
            self.data.qpos[self._arm_l_qadr] = self.HOME_QPOS_ARM

        if action.right_arm is not None:
            self.prev_qpos_r = self._apply_arm(
                action.right_arm,
                self._arm_r_qadr,
                self._arm_r_dofadr,
                self._arm_r_jnt_range,
                self._eef_r_body_id,
                self.prev_qpos_r,
            )
        else:
            self.data.qpos[self._arm_r_qadr] = self.HOME_QPOS_ARM

        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(
            self.data,
            camera=self.camera,
            scene_option=self.scene_option,
        )
        return self.renderer.render()

    def close(self):
        self.renderer.close()
