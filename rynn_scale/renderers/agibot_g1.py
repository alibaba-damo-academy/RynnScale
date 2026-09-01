import os

import numpy as np

from ..constants import RobotType
from ..registry import RENDERER_REGISTRY
from ..utils.robot import Arm, RobotAction
from ._lazy import mujoco
from .agibot_g2 import AgibotG2Renderer
from .base import BaseRenderer


@RENDERER_REGISTRY.register(name=RobotType.AGIBOT_G1.value)
class AgibotG1Renderer(AgibotG2Renderer):
    """Renders an AgiBot G1 dual-arm robot via MuJoCo, solving IK from EEF poses.

    Real G1 kinematics (assets/agibot_g1/g1.xml): a 2-DOF waist (prismatic lift
    + forward pitch), a 2-DOF head, mirrored 7-DOF arms, and rigid grippers.
    Reuses the G2 IK machinery (``_solve_ik_arm`` / ``_apply_arm_ik``) with
    G1-specific torso/head wiring.

    The dataset reports EEF poses in the fixed ``base_link`` frame, so IK targets
    resolve there directly. The waist stream is ``[pitch, lift_height]``; the
    lift maps to the prismatic joint by subtracting ``SLIDE_OFFSET``. With this
    mapping, FK of the recorded joints reproduces the dataset EEF to <4mm.
    """

    # Per-arm operating postures (the arms are mirror images). Used as IK
    # warm-start seeds and the null-space bias target — never read from the
    # dataset at render time.
    HOME_QPOS_ARM_L = np.array(
        [-1.21, 0.82, 0.57, -0.71, 0.72, 1.68, 0.13],
        dtype=np.float64,
    )
    HOME_QPOS_ARM_R = np.array(
        [1.34, -0.96, -1.28, 0.48, -0.47, -1.14, -0.44],
        dtype=np.float64,
    )
    # Waist qpos order is [idx01 slide, idx02 pitch].
    HOME_QPOS_WAIST = np.array([0.09, 0.55], dtype=np.float64)
    # Head qpos order is [idx11, idx12].
    HOME_QPOS_HEAD = np.array([-0.05, 0.51], dtype=np.float64)

    # Dataset waist[:, 1] is the lift height in metres; the MJCF prismatic joint
    # is zeroed at the USD rest pose, which sits this far above the lift origin.
    SLIDE_OFFSET = 0.30

    def __init__(self, height: int = 480, width: int = 480, action_source: str = "joint"):
        BaseRenderer.__init__(self, height=height, width=width)

        self.action_source = self._resolve_action_source(action_source)
        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        arm_l = [f"arm_l_joint{i}" for i in range(1, 8)]
        arm_r = [f"arm_r_joint{i}" for i in range(1, 8)]
        self._arm_l_qadr = np.array([self._joint_qadr(n) for n in arm_l], dtype=np.int64)
        self._arm_r_qadr = np.array([self._joint_qadr(n) for n in arm_r], dtype=np.int64)
        self._arm_l_dofadr = np.array([self._joint_dofadr(n) for n in arm_l], dtype=np.int64)
        self._arm_r_dofadr = np.array([self._joint_dofadr(n) for n in arm_r], dtype=np.int64)
        self._arm_l_jnt_range = self._jnt_ranges(arm_l)
        self._arm_r_jnt_range = self._jnt_ranges(arm_r)

        # 2-DOF waist: idx01 prismatic lift, idx02 forward pitch.
        self._slide_qadr = self._joint_qadr("idx01_body_joint1")
        self._pitch_qadr = self._joint_qadr("idx02_body_joint2")
        self._slide_range = self._jnt_ranges(["idx01_body_joint1"])[0]
        self._pitch_range = self._jnt_ranges(["idx02_body_joint2"])[0]

        # 2-DOF head.
        head = [f"idx1{i}_head_joint{i}" for i in range(1, 3)]
        self._head_qadr = np.array([self._joint_qadr(n) for n in head], dtype=np.int64)
        self._head_jnt_range = self._jnt_ranges(head)

        self.site_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "arm_l_grip_site")
        self.site_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "arm_r_grip_site")
        assert self.site_l_id >= 0 and self.site_r_id >= 0

        # The dataset reports EEF in the fixed base_link frame. base_link is the
        # root body (origin, identity) and is unaffected by the waist, so the IK
        # target transform is effectively identity — but we still resolve it
        # through the body pose so the inherited _apply_arm_ik works unchanged.
        self.arm_base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        assert self.arm_base_body_id >= 0

        self.data.qpos[self._slide_qadr] = self.HOME_QPOS_WAIST[0]
        self.data.qpos[self._pitch_qadr] = self.HOME_QPOS_WAIST[1]
        self.data.qpos[self._head_qadr] = self.HOME_QPOS_HEAD
        self.data.qpos[self._arm_l_qadr] = self.HOME_QPOS_ARM_L
        self.data.qpos[self._arm_r_qadr] = self.HOME_QPOS_ARM_R
        mujoco.mj_forward(self.model, self.data)

        self._arm_base_world_pos = self.data.xpos[self.arm_base_body_id].copy()
        self._arm_base_world_mat = self.data.xmat[self.arm_base_body_id].reshape(3, 3).copy()

        self.prev_qpos_l = self.HOME_QPOS_ARM_L.copy()
        self.prev_qpos_r = self.HOME_QPOS_ARM_R.copy()

        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0.4, 0.0, 1.0]
        self.camera.distance = 2.4
        self.camera.azimuth = 150
        self.camera.elevation = -10

        self.scene_option = mujoco.MjvOption()

    def _build_model(self):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(module_dir, "..", ".."))
        xml_path = os.path.join(project_root, "assets", "agibot_g1", "g1.xml")
        assert os.path.exists(xml_path), f"AgiBot G1 model not found: {xml_path}"
        return mujoco.MjModel.from_xml_path(xml_path)

    def _apply_arm_g1(
        self,
        arm: Arm,
        site_id: int,
        qadr: np.ndarray,
        dofadr: np.ndarray,
        jrange: np.ndarray,
        prev_qpos: np.ndarray,
    ) -> np.ndarray:
        """Drive one G1 arm in-place; returns the new warm-start seed.

        Joint playback vs. EEF IK is chosen per ``action_source`` (falling back
        to whichever source the arm carries). The gripper is rigid (no finger
        joints), so the gripper signal is not applied. The 7-DOF redundancy is
        resolved least-motion (the shared solver biases the null space toward
        the warm-start seed), so the mirrored per-arm HOME is only needed as the
        startup seed, not here.
        """
        if self._use_joint(arm):
            return self._apply_arm_joint(arm, qadr, jrange)
        assert arm.eef_position is not None and arm.eef_rotation is not None, (
            "AgibotG1Renderer needs a full EEF pose unless action_source='joint'"
        )
        return self._apply_arm_ik(arm, site_id, qadr, dofadr, jrange, prev_qpos)

    def render(self, action: RobotAction) -> np.ndarray:
        assert len(action) == 1, f"render() expects a single-step action, got chunk={len(action)}"
        for arm in (action.left_arm, action.right_arm):
            if arm is None:
                continue
            ref = arm.joint_position if arm.joint_position is not None else arm.eef_position
            assert ref is not None, "AgibotG1Renderer requires either joint_position or an EEF pose"
            assert not ref.is_relative, "render() expects an absolute action; convert delta actions first"

        # Torso: the waist stream is [pitch, lift_height]. Apply before the arms
        # so base_link's (fixed) world pose is settled for the IK target.
        if action.torso is not None:
            torso = action.torso.data[0].detach().cpu().numpy().astype(np.float64)
            pitch = torso[0]
            slide = torso[1] - self.SLIDE_OFFSET
        else:
            slide, pitch = self.HOME_QPOS_WAIST
        self.data.qpos[self._slide_qadr] = float(np.clip(slide, *self._slide_range))
        self.data.qpos[self._pitch_qadr] = float(np.clip(pitch, *self._pitch_range))

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
            self.prev_qpos_l = self._apply_arm_g1(
                action.left_arm,
                self.site_l_id,
                self._arm_l_qadr,
                self._arm_l_dofadr,
                self._arm_l_jnt_range,
                self.prev_qpos_l,
            )
        else:
            self.data.qpos[self._arm_l_qadr] = self.HOME_QPOS_ARM_L

        if action.right_arm is not None:
            self.prev_qpos_r = self._apply_arm_g1(
                action.right_arm,
                self.site_r_id,
                self._arm_r_qadr,
                self._arm_r_dofadr,
                self._arm_r_jnt_range,
                self.prev_qpos_r,
            )
        else:
            self.data.qpos[self._arm_r_qadr] = self.HOME_QPOS_ARM_R

        mujoco.mj_forward(self.model, self.data)

        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.scene_option)
        return self.renderer.render()
