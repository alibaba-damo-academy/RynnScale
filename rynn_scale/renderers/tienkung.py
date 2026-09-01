import os

import numpy as np

from ..constants import RobotType
from ..registry import RENDERER_REGISTRY
from ..utils.robot import Arm, Position, RobotAction, _to_scipy_rotation
from ._lazy import mujoco
from .base import BaseRenderer


@RENDERER_REGISTRY.register(name=RobotType.TIENKUNG_1.value)
class TienkungRenderer(BaseRenderer):
    """TienKung Pro humanoid dual-arm renderer (joint playback).

    Accepts ``RobotAction`` with ``left_arm`` / ``right_arm`` carrying
    7-DOF joint positions and optional ``left_hand`` / ``right_hand``
    carrying hand joint data (1d, 6d, or 12d per hand).
    """

    # Neutral arm pose — arms slightly bent forward, visually plausible.
    HOME_QPOS_ARM = np.array(
        [0.0, 0.3, 0.0, -0.8, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )

    N_ARM = 7

    # Left arm joint names in kinematic order (7 DOF).
    _LEFT_ARM_JOINTS = [
        "left_joint1",
        "shoulder_roll_l_joint",
        "left_joint3",
        "elbow_l_joint",
        "left_joint5",
        "left_joint6",
        "left_joint7",
    ]
    _RIGHT_ARM_JOINTS = [
        "right_joint1",
        "shoulder_roll_r_joint",
        "right_joint3",
        "elbow_r_joint",
        "right_joint5",
        "right_joint6",
        "right_joint7",
    ]

    # Hand actuated joints (6 per side): thumb_yaw, thumb_pitch, index, middle, ring, pinky.
    _LEFT_HAND_ACTUATED = [
        "L_thumb_proximal_yaw_joint",
        "L_thumb_proximal_pitch_joint",
        "L_index_proximal_joint",
        "L_middle_proximal_joint",
        "L_ring_proximal_joint",
        "L_pinky_proximal_joint",
    ]
    _RIGHT_HAND_ACTUATED = [
        "R_thumb_proximal_yaw_joint",
        "R_thumb_proximal_pitch_joint",
        "R_index_proximal_joint",
        "R_middle_proximal_joint",
        "R_ring_proximal_joint",
        "R_pinky_proximal_joint",
    ]

    # Hand mimic joints — driven by the corresponding actuated joint
    # via a fixed multiplier. Order matches the actuated list.
    _LEFT_HAND_MIMIC = [
        ("L_thumb_intermediate_joint", "L_thumb_proximal_pitch_joint", 1.6),
        ("L_thumb_distal_joint", "L_thumb_proximal_pitch_joint", 2.4),
        ("L_index_intermediate_joint", "L_index_proximal_joint", 1.0),
        ("L_middle_intermediate_joint", "L_middle_proximal_joint", 1.0),
        ("L_ring_intermediate_joint", "L_ring_proximal_joint", 1.0),
        ("L_pinky_intermediate_joint", "L_pinky_proximal_joint", 1.0),
    ]
    _RIGHT_HAND_MIMIC = [
        ("R_thumb_intermediate_joint", "R_thumb_proximal_pitch_joint", 1.6),
        ("R_thumb_distal_joint", "R_thumb_proximal_pitch_joint", 2.4),
        ("R_index_intermediate_joint", "R_index_proximal_joint", 1.0),
        ("R_middle_intermediate_joint", "R_middle_proximal_joint", 1.0),
        ("R_ring_intermediate_joint", "R_ring_proximal_joint", 1.0),
        ("R_pinky_intermediate_joint", "R_pinky_proximal_joint", 1.0),
    ]

    # All 12 hand joints in qpos order (actuated + mimic interleaved).
    _LEFT_HAND_ALL = [
        "L_thumb_proximal_yaw_joint",
        "L_thumb_proximal_pitch_joint",
        "L_thumb_intermediate_joint",
        "L_thumb_distal_joint",
        "L_index_proximal_joint",
        "L_index_intermediate_joint",
        "L_middle_proximal_joint",
        "L_middle_intermediate_joint",
        "L_ring_proximal_joint",
        "L_ring_intermediate_joint",
        "L_pinky_proximal_joint",
        "L_pinky_intermediate_joint",
    ]
    _RIGHT_HAND_ALL = [
        "R_thumb_proximal_yaw_joint",
        "R_thumb_proximal_pitch_joint",
        "R_thumb_intermediate_joint",
        "R_thumb_distal_joint",
        "R_index_proximal_joint",
        "R_index_intermediate_joint",
        "R_middle_proximal_joint",
        "R_middle_intermediate_joint",
        "R_ring_proximal_joint",
        "R_ring_intermediate_joint",
        "R_pinky_proximal_joint",
        "R_pinky_intermediate_joint",
    ]

    def __init__(self, height: int = 480, width: int = 480, **_unused):
        super().__init__(height=height, width=width)

        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        self._arm_l_qadr = np.array(
            [self._joint_qadr(n) for n in self._LEFT_ARM_JOINTS],
            dtype=np.int64,
        )
        self._arm_r_qadr = np.array(
            [self._joint_qadr(n) for n in self._RIGHT_ARM_JOINTS],
            dtype=np.int64,
        )
        self._arm_l_jnt_range = self._jnt_ranges(self._LEFT_ARM_JOINTS)
        self._arm_r_jnt_range = self._jnt_ranges(self._RIGHT_ARM_JOINTS)

        self._hand_l_act_qadr = np.array(
            [self._joint_qadr(n) for n in self._LEFT_HAND_ACTUATED],
            dtype=np.int64,
        )
        self._hand_r_act_qadr = np.array(
            [self._joint_qadr(n) for n in self._RIGHT_HAND_ACTUATED],
            dtype=np.int64,
        )
        self._hand_l_act_range = self._jnt_ranges(self._LEFT_HAND_ACTUATED)
        self._hand_r_act_range = self._jnt_ranges(self._RIGHT_HAND_ACTUATED)

        self._hand_l_all_qadr = np.array(
            [self._joint_qadr(n) for n in self._LEFT_HAND_ALL],
            dtype=np.int64,
        )
        self._hand_r_all_qadr = np.array(
            [self._joint_qadr(n) for n in self._RIGHT_HAND_ALL],
            dtype=np.int64,
        )
        self._hand_l_all_range = self._jnt_ranges(self._LEFT_HAND_ALL)
        self._hand_r_all_range = self._jnt_ranges(self._RIGHT_HAND_ALL)

        # Build mimic lookup: for each side, list of (mimic_qadr, parent_qadr, multiplier).
        self._mimic_l = self._build_mimic_table(self._LEFT_HAND_MIMIC)
        self._mimic_r = self._build_mimic_table(self._RIGHT_HAND_MIMIC)

        # IK support: site/body IDs for EEF tracking.
        self._arm_l_dofadr = np.array(
            [self._joint_dofadr(n) for n in self._LEFT_ARM_JOINTS],
            dtype=np.int64,
        )
        self._arm_r_dofadr = np.array(
            [self._joint_dofadr(n) for n in self._RIGHT_ARM_JOINTS],
            dtype=np.int64,
        )
        self._eef_l_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "left_link7",
        )
        self._eef_r_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "right_link7",
        )

        self._reset_home()
        mujoco.mj_forward(self.model, self.data)

        self.prev_qpos_l = self.HOME_QPOS_ARM.copy()
        self.prev_qpos_r = self.HOME_QPOS_ARM.copy()

        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0.0, 0.0, 0.35]
        self.camera.distance = 2.0
        self.camera.azimuth = 150
        self.camera.elevation = -15

        self.scene_option = mujoco.MjvOption()

    def _build_model(self):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(module_dir, "..", ".."))
        xml_path = os.path.join(project_root, "assets", "tienkung", "scene.xml")
        assert os.path.exists(xml_path), f"TienKung Pro model not found: {xml_path}"
        return mujoco.MjModel.from_xml_path(xml_path)

    def _build_mimic_table(self, mimic_spec):
        table = []
        for mimic_name, parent_name, mult in mimic_spec:
            m_qadr = self._joint_qadr(mimic_name)
            p_qadr = self._joint_qadr(parent_name)
            m_range = self.model.jnt_range[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, mimic_name)].copy()
            table.append((m_qadr, p_qadr, mult, m_range))
        return table

    def _reset_home(self):
        self.data.qpos[self._arm_l_qadr] = self.HOME_QPOS_ARM
        self.data.qpos[self._arm_r_qadr] = self.HOME_QPOS_ARM

    def _apply_mimic(self, mimic_table):
        for m_qadr, p_qadr, mult, m_range in mimic_table:
            val = self.data.qpos[p_qadr] * mult
            self.data.qpos[m_qadr] = np.clip(val, m_range[0], m_range[1])

    def _apply_hand(self, hand: Position, act_qadr, act_range, all_qadr, all_range, mimic_table):
        vals = hand.data[0].detach().cpu().numpy().astype(np.float64)
        d = vals.shape[0]

        if d == 1:
            # GELLO 1d: single value spread to all 6 actuated joints.
            self.data.qpos[act_qadr] = np.clip(
                np.full(6, vals[0]),
                act_range[:, 0],
                act_range[:, 1],
            )
            self._apply_mimic(mimic_table)
        elif d == 6:
            # XSens 6d: maps directly to the 6 actuated hand joints.
            self.data.qpos[act_qadr] = np.clip(vals, act_range[:, 0], act_range[:, 1])
            self._apply_mimic(mimic_table)
        elif d == 12:
            # Sim 12d: all hand joints in order.
            self.data.qpos[all_qadr] = np.clip(vals, all_range[:, 0], all_range[:, 1])
        else:
            # Best-effort: if dim matches actuated count, use it; otherwise ignore.
            if d == len(act_qadr):
                self.data.qpos[act_qadr] = np.clip(vals, act_range[:, 0], act_range[:, 1])
                self._apply_mimic(mimic_table)

    def _apply_arm(self, arm: Arm, qadr, dofadr, jrange, body_id, prev_qpos):
        if arm.joint_position is not None:
            qpos = arm.joint_position.data[0].detach().cpu().numpy().astype(np.float64)
            assert qpos.shape == (self.N_ARM,), f"TienkungRenderer expects 7-DOF per arm, got shape {qpos.shape}"
            self.data.qpos[qadr] = np.clip(qpos, jrange[:, 0], jrange[:, 1])
            return self.data.qpos[qadr].copy()

        assert arm.eef_position is not None and arm.eef_rotation is not None, (
            "TienkungRenderer requires joint_position or full EEF pose"
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
            assert ref is not None, "TienkungRenderer requires either joint_position or eef pose"
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

        if action.left_hand is not None:
            self._apply_hand(
                action.left_hand,
                self._hand_l_act_qadr,
                self._hand_l_act_range,
                self._hand_l_all_qadr,
                self._hand_l_all_range,
                self._mimic_l,
            )

        if action.right_hand is not None:
            self._apply_hand(
                action.right_hand,
                self._hand_r_act_qadr,
                self._hand_r_act_range,
                self._hand_r_all_qadr,
                self._hand_r_all_range,
                self._mimic_r,
            )

        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(
            self.data,
            camera=self.camera,
            scene_option=self.scene_option,
        )
        return self.renderer.render()

    def close(self):
        self.renderer.close()
