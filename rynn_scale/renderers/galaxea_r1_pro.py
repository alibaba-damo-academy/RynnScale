import os

import numpy as np

from ..constants import RobotType
from ..registry import RENDERER_REGISTRY
from ..utils.robot import Arm, Position, RobotAction, _to_scipy_rotation
from ._lazy import mujoco
from .base import BaseRenderer


@RENDERER_REGISTRY.register(name=RobotType.GALAXEA_R1_PRO.value)
class GalaxeaR1ProRenderer(BaseRenderer):
    """Renders a Galaxea R1 Pro dual-arm robot via MuJoCo.

    Compared to R1 Lite, R1 Pro has 7-DOF arms (vs 6) and a 4-DOF torso
    (vs 3). The extra arm DOF makes the arm redundant for 6-DOF EEF tasks,
    so IK uses null-space bias toward the previous solution.
    """

    HOME_QPOS_ARM = np.array(
        [0.0, 0.6, 0.0, -1.2, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    HOME_QPOS_TORSO = np.array(
        [0.0, 0.0],
        dtype=np.float64,
    )
    HOME_QPOS_HEAD = np.array(
        [0.0, 0.0],
        dtype=np.float64,
    )

    EEF_REFERENCE_BODY = "torso_link4"

    def __init__(
        self,
        height: int = 480,
        width: int = 480,
        action_source: str = "joint",
        eef_reference_body: str = None,
    ):
        super().__init__(height=height, width=width)

        self.action_source = self._resolve_action_source(action_source)
        self.eef_reference_body = eef_reference_body or self.EEF_REFERENCE_BODY

        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        self._arm_l_qadr = np.array(
            [self._joint_qadr(f"left_arm_joint{i}") for i in range(1, 8)],
            dtype=np.int64,
        )
        self._arm_r_qadr = np.array(
            [self._joint_qadr(f"right_arm_joint{i}") for i in range(1, 8)],
            dtype=np.int64,
        )
        self._arm_l_dofadr = np.array(
            [self._joint_dofadr(f"left_arm_joint{i}") for i in range(1, 8)],
            dtype=np.int64,
        )
        self._arm_r_dofadr = np.array(
            [self._joint_dofadr(f"right_arm_joint{i}") for i in range(1, 8)],
            dtype=np.int64,
        )
        self._arm_l_jnt_range = self._jnt_ranges([f"left_arm_joint{i}" for i in range(1, 8)])
        self._arm_r_jnt_range = self._jnt_ranges([f"right_arm_joint{i}" for i in range(1, 8)])

        self._gripper_l_qadr = np.array(
            [
                self._joint_qadr("left_gripper_finger_joint1"),
                self._joint_qadr("left_gripper_finger_joint2"),
            ],
            dtype=np.int64,
        )
        self._gripper_r_qadr = np.array(
            [
                self._joint_qadr("right_gripper_finger_joint1"),
                self._joint_qadr("right_gripper_finger_joint2"),
            ],
            dtype=np.int64,
        )

        self._torso_qadr = np.array(
            [self._joint_qadr(f"torso_joint{i}") for i in range(1, 3)],
            dtype=np.int64,
        )
        self._head_qadr = np.array(
            [self._joint_qadr(f"torso_joint{i}") for i in range(3, 5)],
            dtype=np.int64,
        )
        self._torso_jnt_range = self._jnt_ranges([f"torso_joint{i}" for i in range(1, 3)])
        self._head_jnt_range = self._jnt_ranges([f"torso_joint{i}" for i in range(3, 5)])

        self.site_l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_eef_site")
        self.site_r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_eef_site")
        assert self.site_l_id >= 0 and self.site_r_id >= 0

        self.eef_ref_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.eef_reference_body)
        assert self.eef_ref_body_id >= 0, f"EEF reference body '{self.eef_reference_body}' not found in model"

        self.data.qpos[self._torso_qadr] = self.HOME_QPOS_TORSO
        self.data.qpos[self._head_qadr] = self.HOME_QPOS_HEAD
        self.data.qpos[self._arm_l_qadr] = self.HOME_QPOS_ARM
        self.data.qpos[self._arm_r_qadr] = self.HOME_QPOS_ARM
        mujoco.mj_forward(self.model, self.data)

        self._eef_ref_world_pos = self.data.xpos[self.eef_ref_body_id].copy()
        self._eef_ref_world_mat = self.data.xmat[self.eef_ref_body_id].reshape(3, 3).copy()

        self.prev_qpos_l = self.HOME_QPOS_ARM.copy()
        self.prev_qpos_r = self.HOME_QPOS_ARM.copy()

        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0.0, 0.0, 1.2]
        self.camera.distance = 2.8
        self.camera.azimuth = 150
        self.camera.elevation = -10

        self.scene_option = mujoco.MjvOption()

    def _build_model(self):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(module_dir, "..", ".."))
        xml_path = os.path.join(project_root, "assets", "galaxea_r1_pro", "r1_pro.xml")
        assert os.path.exists(xml_path), f"R1 Pro model not found: {xml_path}"
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

        target_pos_world = self._eef_ref_world_mat @ eef_pos_local + self._eef_ref_world_pos
        target_rot_mat_world = self._eef_ref_world_mat @ eef_rot_mat_local

        # 7-DOF arm > 6-DOF task → enable null-space bias toward seed.
        self.solve_ik(
            target_pos=target_pos_world,
            target_rot_mat=target_rot_mat_world,
            qadr=qadr,
            dofadr=dofadr,
            jnt_range=jrange,
            site_id=site_id,
            seed_qpos=prev_qpos,
            damping=1e-3,
            max_iter=30,
            null_space_gain=0.5,
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
        self.data.qpos[qadr] = qpos
        return self.data.qpos[qadr].copy()

    def _apply_arm(
        self,
        arm: Arm,
        gripper: Position,
        site_id: int,
        qadr: np.ndarray,
        dofadr: np.ndarray,
        jrange: np.ndarray,
        gripper_qadr: np.ndarray,
        prev_qpos: np.ndarray,
    ) -> np.ndarray:
        if self._use_joint(arm):
            new_qpos = self._apply_arm_joint(arm, qadr, jrange)
        else:
            assert arm.eef_position is not None and arm.eef_rotation is not None, (
                "GalaxeaR1ProRenderer needs either joint_position or full EEF pose"
            )
            new_qpos = self._apply_arm_ik(
                arm,
                site_id,
                qadr,
                dofadr,
                jrange,
                prev_qpos,
            )

        gripper_norm = float(np.clip(gripper.data[0, 0].item(), 0.0, 1.0))
        driver = gripper_norm * 0.05
        self.data.qpos[gripper_qadr[0]] = driver
        self.data.qpos[gripper_qadr[1]] = -driver

        return new_qpos

    def _apply_joint_group(
        self,
        position: Position,
        qadr: np.ndarray,
        home: np.ndarray,
    ) -> None:
        if position is None:
            self.data.qpos[qadr] = home
            return
        qpos = position.data[0].detach().cpu().numpy().astype(np.float64)
        assert qpos.shape == (qadr.size,), f"expected {qadr.size}-dim joint action, got {qpos.shape}"
        self.data.qpos[qadr] = qpos

    def render(self, action: RobotAction) -> np.ndarray:
        assert len(action) == 1, f"render() expects a single-step action, got chunk={len(action)}"
        for arm in (action.left_arm, action.right_arm):
            if arm is None:
                continue
            ref = arm.joint_position if arm.joint_position is not None else arm.eef_position
            assert ref is not None, "GalaxeaR1ProRenderer requires either joint_position or eef pose"
            assert not ref.is_relative, "render() expects an absolute action; convert delta actions first"

        self._apply_joint_group(
            action.torso,
            self._torso_qadr,
            self.HOME_QPOS_TORSO,
        )
        self._apply_joint_group(
            action.head,
            self._head_qadr,
            self.HOME_QPOS_HEAD,
        )
        mujoco.mj_forward(self.model, self.data)
        self._eef_ref_world_pos = self.data.xpos[self.eef_ref_body_id].copy()
        self._eef_ref_world_mat = self.data.xmat[self.eef_ref_body_id].reshape(3, 3).copy()

        if action.left_arm is not None:
            assert action.left_gripper is not None, "left_arm present requires left_gripper"
            self.prev_qpos_l = self._apply_arm(
                action.left_arm,
                action.left_gripper,
                self.site_l_id,
                self._arm_l_qadr,
                self._arm_l_dofadr,
                self._arm_l_jnt_range,
                self._gripper_l_qadr,
                self.prev_qpos_l,
            )
        else:
            self.data.qpos[self._arm_l_qadr] = self.HOME_QPOS_ARM
            self.data.qpos[self._gripper_l_qadr] = 0.0

        if action.right_arm is not None:
            assert action.right_gripper is not None, "right_arm present requires right_gripper"
            self.prev_qpos_r = self._apply_arm(
                action.right_arm,
                action.right_gripper,
                self.site_r_id,
                self._arm_r_qadr,
                self._arm_r_dofadr,
                self._arm_r_jnt_range,
                self._gripper_r_qadr,
                self.prev_qpos_r,
            )
        else:
            self.data.qpos[self._arm_r_qadr] = self.HOME_QPOS_ARM
            self.data.qpos[self._gripper_r_qadr] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.scene_option)
        return self.renderer.render()

    def close(self):
        self.renderer.close()
