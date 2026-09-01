import os

import numpy as np

from ..constants import RobotType
from ..registry import RENDERER_REGISTRY
from ..utils.robot import Arm, Position, RobotAction, _to_scipy_rotation
from ._lazy import mujoco
from .base import BaseRenderer


@RENDERER_REGISTRY.register(name=RobotType.AGIBOT_G2.value)
class AgibotG2Renderer(BaseRenderer):
    """Renders an AgiBot G2 dual-arm robot via MuJoCo, solving IK from EEF poses.

    Right arm holds HOME when its ``Arm`` field is missing on the input
    ``RobotAction``.
    """

    HOME_QPOS_ARM = np.array(
        [0.0, -0.6, 0.0, -1.6, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    # Default bowed-forward operating posture for the 5-DOF waist. Mirrors the
    # baked euler posture the previous joint-less XML used so existing single-
    # arm visualizations still look natural when the dataset omits waist.
    HOME_QPOS_WAIST = np.array(
        [-0.6, 1.23, -0.6, 0.0, 0.0],
        dtype=np.float64,
    )
    HOME_QPOS_HEAD = np.array(
        [0.0, 0.0, 0.0],
        dtype=np.float64,
    )

    def __init__(self, height: int = 480, width: int = 480, action_source: str = "joint"):
        super().__init__(height=height, width=width)

        # ``action_source`` selects how each arm is driven when the action
        # carries both joints and an EEF pose: "joint" (default) plays back the
        # recorded joint angles, "eef" IK-solves the EEF pose. Whichever source
        # is absent triggers a fallback to the other.
        self.action_source = self._resolve_action_source(action_source)
        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        self._arm_l_qadr = np.array(
            [self._joint_qadr(f"arm_l_joint{i}") for i in range(1, 8)],
            dtype=np.int64,
        )
        self._arm_r_qadr = np.array(
            [self._joint_qadr(f"arm_r_joint{i}") for i in range(1, 8)],
            dtype=np.int64,
        )
        self._arm_l_dofadr = np.array(
            [self._joint_dofadr(f"arm_l_joint{i}") for i in range(1, 8)],
            dtype=np.int64,
        )
        self._arm_r_dofadr = np.array(
            [self._joint_dofadr(f"arm_r_joint{i}") for i in range(1, 8)],
            dtype=np.int64,
        )
        self._arm_l_jnt_range = np.stack(
            [
                self.model.jnt_range[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"arm_l_joint{i}")]
                for i in range(1, 8)
            ]
        )
        self._arm_r_jnt_range = np.stack(
            [
                self.model.jnt_range[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"arm_r_joint{i}")]
                for i in range(1, 8)
            ]
        )
        self._finger_l_qadr = np.array(
            [self._joint_qadr("arm_l_finger_l_joint"), self._joint_qadr("arm_l_finger_r_joint")],
            dtype=np.int64,
        )
        self._finger_r_qadr = np.array(
            [self._joint_qadr("arm_r_finger_l_joint"), self._joint_qadr("arm_r_finger_r_joint")],
            dtype=np.int64,
        )

        self._waist_qadr = np.array(
            [self._joint_qadr(f"idx0{i}_body_joint{i}") for i in range(1, 6)],
            dtype=np.int64,
        )
        self._waist_jnt_range = np.stack(
            [
                self.model.jnt_range[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"idx0{i}_body_joint{i}")
                ]
                for i in range(1, 6)
            ]
        )
        self._head_qadr = np.array(
            [self._joint_qadr(f"idx1{i}_head_joint{i}") for i in range(1, 4)],
            dtype=np.int64,
        )
        self._head_jnt_range = np.stack(
            [
                self.model.jnt_range[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"idx1{i}_head_joint{i}")
                ]
                for i in range(1, 4)
            ]
        )

        self.site_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "arm_l_grip_site")
        self.site_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "arm_r_grip_site")
        assert self.site_l_id >= 0 and self.site_r_id >= 0

        # arm_base_link is the frame the dataset reports EEF poses in. With
        # the waist now articulated, its world-frame transform changes every
        # frame and must be refreshed after the body settles, not cached once.
        self.arm_base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "arm_base_link")
        assert self.arm_base_body_id >= 0

        self.data.qpos[self._waist_qadr] = self.HOME_QPOS_WAIST
        self.data.qpos[self._head_qadr] = self.HOME_QPOS_HEAD
        self.data.qpos[self._arm_l_qadr] = self.HOME_QPOS_ARM
        self.data.qpos[self._arm_r_qadr] = self.HOME_QPOS_ARM
        mujoco.mj_forward(self.model, self.data)

        self._arm_base_world_pos = self.data.xpos[self.arm_base_body_id].copy()
        self._arm_base_world_mat = self.data.xmat[self.arm_base_body_id].reshape(3, 3).copy()

        # Per-arm warm-start seeds. Reset to HOME on convergence failure.
        self.prev_qpos_l = self.HOME_QPOS_ARM.copy()
        self.prev_qpos_r = self.HOME_QPOS_ARM.copy()

        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0.3, 0.0, 1.4]
        self.camera.distance = 3.0
        self.camera.azimuth = 150
        self.camera.elevation = -10

        self.scene_option = mujoco.MjvOption()

    def _build_model(self):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(module_dir, "..", ".."))
        xml_path = os.path.join(project_root, "assets", "agibot_g2", "g2.xml")
        assert os.path.exists(xml_path), f"AgiBot G2 model not found: {xml_path}"
        return mujoco.MjModel.from_xml_path(xml_path)

    def _apply_arm_ik(
        self,
        arm: Arm,
        site_id: int,
        qadr: np.ndarray,
        dofadr: np.ndarray,
        jrange: np.ndarray,
        prev_qpos: np.ndarray,
    ) -> np.ndarray:
        eef_pos_local = arm.eef_position.data[0].detach().cpu().numpy().astype(np.float64)
        rot_local = _to_scipy_rotation(arm.eef_rotation.data.detach().cpu(), arm.eef_rotation.representation)
        eef_rot_mat_local = rot_local.as_matrix()[0].astype(np.float64)

        target_pos_world = self._arm_base_world_mat @ eef_pos_local + self._arm_base_world_pos
        target_rot_mat_world = self._arm_base_world_mat @ eef_rot_mat_local

        # Warm-start only: no HOME-reset fallback. Restarting from HOME on
        # large residual lands in a different elbow configuration than the
        # neighboring frames, which manifests as visible "flying" between
        # frames where the warm-started solve was near a singularity.
        #
        # Base damping 1e-2 (vs Franka's 1e-4): some AgiBot trajectories drive
        # the arm near wrist singularities where low-damping DLS produces multi-
        # joint flips. The solver's manipulability gate ramps λ higher still as
        # those singularities are approached. The 7-DOF redundancy is resolved
        # least-motion (null target defaults to the warm-start seed).
        self.solve_ik(
            target_pos=target_pos_world,
            target_rot_mat=target_rot_mat_world,
            qadr=qadr,
            dofadr=dofadr,
            jnt_range=jrange,
            site_id=site_id,
            seed_qpos=prev_qpos,
            null_space_gain=0.05,
            damping=1e-2,
        )
        return self.data.qpos[qadr].copy()

    def _apply_arm_joint(
        self,
        arm: Arm,
        qadr: np.ndarray,
        jrange: np.ndarray,
    ) -> np.ndarray:
        qpos = arm.joint_position.data[0].detach().cpu().numpy().astype(np.float64)
        assert qpos.shape == (qadr.size,), f"expected {qadr.size}-dim arm joint action, got {qpos.shape}"
        self.data.qpos[qadr] = np.clip(qpos, jrange[:, 0], jrange[:, 1])
        return self.data.qpos[qadr].copy()

    def _apply_arm(
        self,
        arm: Arm,
        gripper: Position,
        site_id: int,
        qadr: np.ndarray,
        dofadr: np.ndarray,
        jrange: np.ndarray,
        finger_qadr: np.ndarray,
        prev_qpos: np.ndarray,
    ) -> np.ndarray:
        """Drive one arm in-place; returns the new prev_qpos seed.

        Picks joint playback vs. IK per ``action_source``, falling back to
        whichever source the arm actually carries.
        """
        if self._use_joint(arm):
            new_qpos = self._apply_arm_joint(arm, qadr, jrange)
        else:
            assert arm.eef_position is not None and arm.eef_rotation is not None, (
                "AgibotG2Renderer needs either joint_position or full EEF pose"
            )
            new_qpos = self._apply_arm_ik(
                arm,
                site_id,
                qadr,
                dofadr,
                jrange,
                prev_qpos,
            )

        # Gripper input is normalized: 0 = fully closed, 1 = fully open.
        gripper_norm = float(np.clip(gripper.data[0, 0].item(), 0.0, 1.0))
        self.data.qpos[finger_qadr] = gripper_norm * 0.04

        return new_qpos

    def _apply_joint_group(
        self,
        position: Position,
        qadr: np.ndarray,
        jrange: np.ndarray,
        home: np.ndarray,
    ) -> None:
        if position is None:
            self.data.qpos[qadr] = home
            return
        qpos = position.data[0].detach().cpu().numpy().astype(np.float64)
        assert qpos.shape == (qadr.size,), f"expected {qadr.size}-dim joint action, got {qpos.shape}"
        self.data.qpos[qadr] = np.clip(qpos, jrange[:, 0], jrange[:, 1])

    def render(self, action: RobotAction) -> np.ndarray:
        assert len(action) == 1, f"render() expects a single-step action, got chunk={len(action)}"
        for arm in (action.left_arm, action.right_arm):
            if arm is None:
                continue
            ref = arm.joint_position if arm.joint_position is not None else arm.eef_position
            assert ref is not None, "AgibotG2Renderer requires either joint_position or eef pose"
            assert not ref.is_relative, "render() expects an absolute action; convert delta actions first"

        # Apply torso + head first so arm_base_link's world pose reflects the
        # current torso configuration before we transform EEF targets into
        # world frame for IK.
        self._apply_joint_group(
            action.torso,
            self._waist_qadr,
            self._waist_jnt_range,
            self.HOME_QPOS_WAIST,
        )
        self._apply_joint_group(
            action.head,
            self._head_qadr,
            self._head_jnt_range,
            self.HOME_QPOS_HEAD,
        )
        mujoco.mj_forward(self.model, self.data)
        self._arm_base_world_pos = self.data.xpos[self.arm_base_body_id].copy()
        self._arm_base_world_mat = self.data.xmat[self.arm_base_body_id].reshape(3, 3).copy()

        if action.left_arm is not None:
            assert action.left_gripper is not None, "left_arm present requires left_gripper"
            self.prev_qpos_l = self._apply_arm(
                action.left_arm,
                action.left_gripper,
                self.site_l_id,
                self._arm_l_qadr,
                self._arm_l_dofadr,
                self._arm_l_jnt_range,
                self._finger_l_qadr,
                self.prev_qpos_l,
            )
        else:
            self.data.qpos[self._arm_l_qadr] = self.HOME_QPOS_ARM
            self.data.qpos[self._finger_l_qadr] = 0.0

        if action.right_arm is not None:
            assert action.right_gripper is not None, "right_arm present requires right_gripper"
            self.prev_qpos_r = self._apply_arm(
                action.right_arm,
                action.right_gripper,
                self.site_r_id,
                self._arm_r_qadr,
                self._arm_r_dofadr,
                self._arm_r_jnt_range,
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
