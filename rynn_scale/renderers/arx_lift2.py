"""ARX LIFT2 dual-arm renderer (two ARX R5a 6-DOF arms on a lift platform).

The dual-arm model is assembled at load time with ``mujoco.MjSpec``: the
single-arm ``arx_lift2.xml`` is attached twice under ``left/`` and ``right/``
prefixes. Arms are mounted upright on the platform.
"""

import os

import numpy as np

from ..constants import RobotType
from ..registry import RENDERER_REGISTRY
from ..utils.robot import Arm, Position, RobotAction, _to_scipy_rotation
from ._lazy import mujoco
from .base import BaseRenderer

# Mount frames for the LIFT2 platform.
# Arms are mounted upright on the platform (no rotation), matching the
# official LIFT2 URDF (joint11 origin rpy="0 0 0").
# quat [w,x,y,z] identity: [1, 0, 0, 0]
LEFT_MOUNT = ([0.0, 0.25, 0.0], [1.0, 0.0, 0.0, 0.0])
RIGHT_MOUNT = ([0.0, -0.25, 0.0], [1.0, 0.0, 0.0, 0.0])


@RENDERER_REGISTRY.register(name=RobotType.ARX_LIFT2.value)
class ArxLift2Renderer(BaseRenderer):
    """Dual ARX R5a renderer for the LIFT2 platform."""

    HOME_QPOS_ARM = np.array([0.0, 1.57, -1.35, 0.0, 0.0, 0.0], dtype=np.float64)
    N_ARM = 6

    FINGER_OPEN = 0.044

    # Wrist body used as the IK reference frame (6-DOF arm → no site defined).
    EEF_BODY = "link6"
    # Frame the dataset reports per-arm EEF poses in.
    BASE_BODY = "base_link"

    def __init__(self, height: int = 480, width: int = 480, action_source: str = "joint"):
        super().__init__(height=height, width=width)
        self.action_source = self._resolve_action_source(action_source)

        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        self._arm_l_qadr = np.array(
            [self._joint_qadr(f"left/joint{i}") for i in range(1, 7)],
            dtype=np.int64,
        )
        self._arm_r_qadr = np.array(
            [self._joint_qadr(f"right/joint{i}") for i in range(1, 7)],
            dtype=np.int64,
        )
        self._arm_l_jnt_range = np.stack(
            [
                self.model.jnt_range[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"left/joint{i}")]
                for i in range(1, 7)
            ]
        )
        self._arm_r_jnt_range = np.stack(
            [
                self.model.jnt_range[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"right/joint{i}")]
                for i in range(1, 7)
            ]
        )

        self._arm_l_dofadr = np.array(
            [self._joint_dofadr(f"left/joint{i}") for i in range(1, 7)],
            dtype=np.int64,
        )
        self._arm_r_dofadr = np.array(
            [self._joint_dofadr(f"right/joint{i}") for i in range(1, 7)],
            dtype=np.int64,
        )

        self._eef_l_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"left/{self.EEF_BODY}",
        )
        self._eef_r_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"right/{self.EEF_BODY}",
        )
        assert self._eef_l_id >= 0 and self._eef_r_id >= 0

        self._finger_l_qadr = np.array(
            [self._joint_qadr("left/joint7"), self._joint_qadr("left/joint8")],
            dtype=np.int64,
        )
        self._finger_r_qadr = np.array(
            [self._joint_qadr("right/joint7"), self._joint_qadr("right/joint8")],
            dtype=np.int64,
        )

        self.data.qpos[self._arm_l_qadr] = self.HOME_QPOS_ARM
        self.data.qpos[self._arm_r_qadr] = self.HOME_QPOS_ARM
        mujoco.mj_forward(self.model, self.data)

        # Each arm base is fixed to the (static) platform, so its world-frame
        # transform is constant and can be cached once. EEF targets arrive in
        # the arm's base frame and are lifted to world for IK.
        self._base_l_pos, self._base_l_mat = self._base_world_pose(f"left/{self.BASE_BODY}")
        self._base_r_pos, self._base_r_mat = self._base_world_pose(f"right/{self.BASE_BODY}")

        # Per-arm warm-start seeds; reset to HOME on convergence failure.
        self.prev_qpos_l = self.HOME_QPOS_ARM.copy()
        self.prev_qpos_r = self.HOME_QPOS_ARM.copy()

        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0.0, 0.0, 0.15]
        self.camera.distance = 1.6
        self.camera.azimuth = 150
        self.camera.elevation = -20

        self.scene_option = mujoco.MjvOption()

    def _build_model(self):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(module_dir, "..", ".."))
        r5a_path = os.path.join(project_root, "assets", "arx_lift2", "arx_lift2.xml")
        assert os.path.exists(r5a_path), f"R5a model not found: {r5a_path}"

        parent = mujoco.MjSpec()
        light = parent.worldbody.add_light()
        light.pos = [0.0, 0.0, 2.0]
        light.dir = [0.0, 0.0, -1.0]

        for prefix, (pos, quat) in (("left/", LEFT_MOUNT), ("right/", RIGHT_MOUNT)):
            child = mujoco.MjSpec.from_file(r5a_path)
            for key in list(child.keys):
                child.delete(key)
            for act in list(child.actuators):
                child.delete(act)
            frame = parent.worldbody.add_frame()
            frame.pos = pos
            frame.quat = quat
            frame.attach_body(child.body("base_link"), prefix, "")

        return parent.compile()

    def _base_world_pose(self, body_name: str):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        assert bid >= 0, f"body {body_name} not found"
        return (
            self.data.xpos[bid].copy(),
            self.data.xmat[bid].reshape(3, 3).copy(),
        )

    def _apply_arm(
        self,
        arm: Arm,
        gripper: Position,
        qadr: np.ndarray,
        jrange: np.ndarray,
        dofadr: np.ndarray,
        eef_body_id: int,
        base_pos: np.ndarray,
        base_mat: np.ndarray,
        finger_qadr: np.ndarray,
        prev_qpos: np.ndarray,
    ) -> np.ndarray:
        # Drive from recorded joints or IK-solved EEF pose per ``action_source``
        # (falling back to whichever source the action actually carries).
        if self._use_joint(arm):
            qpos = arm.joint_position.data[0].detach().cpu().numpy().astype(np.float64)
            assert qpos.shape == (self.N_ARM,), f"expected {self.N_ARM}-DOF arm joints, got shape {qpos.shape}"
            self.data.qpos[qadr] = np.clip(qpos, jrange[:, 0], jrange[:, 1])
        else:
            assert arm.eef_position is not None and arm.eef_rotation is not None, (
                "ArxLift2Renderer requires joint_position or full EEF pose"
            )
            eef_pos_local = arm.eef_position.data[0].detach().cpu().numpy().astype(np.float64)
            rot_local = _to_scipy_rotation(
                arm.eef_rotation.data.detach().cpu(),
                arm.eef_rotation.representation,
            )
            eef_rot_local = rot_local.as_matrix()[0].astype(np.float64)

            target_pos = base_mat @ eef_pos_local + base_pos
            target_rot = base_mat @ eef_rot_local

            residual = self.solve_ik(
                target_pos=target_pos,
                target_rot_mat=target_rot,
                qadr=qadr,
                dofadr=dofadr,
                jnt_range=jrange,
                body_id=eef_body_id,
                seed_qpos=prev_qpos,
                damping=1e-4,
            )
            if residual > 1e-2:
                self.solve_ik(
                    target_pos=target_pos,
                    target_rot_mat=target_rot,
                    qadr=qadr,
                    dofadr=dofadr,
                    jnt_range=jrange,
                    body_id=eef_body_id,
                    seed_qpos=self.HOME_QPOS_ARM,
                    damping=1e-4,
                )

        if gripper is not None:
            scale = float(np.clip(gripper.data[0, 0].item(), 0.0, 1.0))
            driver = scale * self.FINGER_OPEN
            self.data.qpos[finger_qadr[0]] = driver
            self.data.qpos[finger_qadr[1]] = driver

        return self.data.qpos[qadr].copy()

    def render(self, action: RobotAction) -> np.ndarray:
        assert len(action) == 1, f"render() expects a single-step action, got chunk={len(action)}"
        for arm in (action.left_arm, action.right_arm):
            if arm is None:
                continue
            ref = arm.joint_position if arm.joint_position is not None else arm.eef_position
            assert ref is not None
            assert not ref.is_relative, "render() expects an absolute action; convert delta actions first"

        if action.left_arm is not None:
            self.prev_qpos_l = self._apply_arm(
                action.left_arm,
                action.left_gripper,
                self._arm_l_qadr,
                self._arm_l_jnt_range,
                self._arm_l_dofadr,
                self._eef_l_id,
                self._base_l_pos,
                self._base_l_mat,
                self._finger_l_qadr,
                self.prev_qpos_l,
            )
        else:
            self.data.qpos[self._arm_l_qadr] = self.HOME_QPOS_ARM
            self.data.qpos[self._finger_l_qadr] = 0.0

        if action.right_arm is not None:
            self.prev_qpos_r = self._apply_arm(
                action.right_arm,
                action.right_gripper,
                self._arm_r_qadr,
                self._arm_r_jnt_range,
                self._arm_r_dofadr,
                self._eef_r_id,
                self._base_r_pos,
                self._base_r_mat,
                self._finger_r_qadr,
                self.prev_qpos_r,
            )
        else:
            self.data.qpos[self._arm_r_qadr] = self.HOME_QPOS_ARM
            self.data.qpos[self._finger_r_qadr] = 0.0

        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.scene_option)
        return self.renderer.render()

    def close(self):
        self.renderer.close()
