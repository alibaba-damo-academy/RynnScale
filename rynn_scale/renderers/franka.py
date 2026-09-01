import os

import numpy as np

from ..constants import RobotType
from ..registry import RENDERER_REGISTRY
from ..utils.robot import RobotAction, _to_scipy_rotation
from ._lazy import mujoco
from .base import BaseRenderer
from .dual_arm_base import BaseDualArmRenderer


@RENDERER_REGISTRY.register(name=RobotType.FRANKA.value)
@RENDERER_REGISTRY.register(name=RobotType.FRANKA_OMRON.value)
class FrankaRenderer(BaseRenderer):
    """Renders a Franka FR3 arm + hand via MuJoCo, solving IK from EEF poses.

    Model built from official franka_description (frankarobotics/franka_description)
    FR3 parameters: kinematics.yaml, joint_limits.yaml, inertials.yaml, dynamics.yaml.
    Visual meshes converted from official FR3 DAE files.
    """

    # Franka "ready" pose (within FR3 joint limits).
    HOME_QPOS = np.array(
        [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398],
        dtype=np.float64,
    )

    # Per-finger slide travel; finger_joint range is [0, 0.04] (open at 0.04).
    GRIPPER_OPEN_WIDTH = 0.04

    def __init__(self, height: int = 480, width: int = 480, **_unused):
        super().__init__(height=height, width=width)

        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        self.data.qpos[:7] = self.HOME_QPOS
        mujoco.mj_forward(self.model, self.data)

        # Warm-start seed for IK: holds the previous frame's solution so
        # nearby EEF targets converge to nearby joint configurations,
        # eliminating the iteration-noise jitter of solving from HOME_QPOS
        # every frame. Resets to HOME_QPOS on convergence failure.
        self.prev_qpos = self.HOME_QPOS.copy()

        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0.0, 0.0, 0.25]
        self.camera.distance = 1.8
        self.camera.azimuth = 180
        self.camera.elevation = -15

        self.scene_option = mujoco.MjvOption()

        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "hand_tcp")
        assert self.site_id >= 0, "hand_tcp not found in model"

        self._arm_qadr = np.arange(7, dtype=np.int64)
        self._arm_dofadr = np.arange(7, dtype=np.int64)
        self._arm_jnt_range = self.model.jnt_range[:7].copy()

    def _build_model(self):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(module_dir, "..", ".."))
        xml_path = os.path.join(project_root, "assets", "franka", "fr3.xml")
        assert os.path.exists(xml_path), f"FR3 model not found: {xml_path}"
        return mujoco.MjModel.from_xml_path(xml_path)

    def render(self, action: RobotAction) -> np.ndarray:
        arm = action.left_arm
        gripper = action.left_gripper
        assert arm is not None and gripper is not None, (
            "FrankaRenderer requires left_arm and left_gripper to be populated"
        )
        assert len(action) == 1, f"render() expects a single-step action, got chunk={len(action)}"

        # Gripper input is normalized: 0 = fully closed, 1 = fully open.
        gripper_norm = float(np.clip(gripper.data[0, 0].item(), 0.0, 1.0))
        gripper_pos = gripper_norm * self.GRIPPER_OPEN_WIDTH

        if arm.joint_position is not None:
            assert not arm.joint_position.is_relative, (
                "render() expects an absolute action; convert delta actions first"
            )
            joint_pos = arm.joint_position.data[0].detach().cpu().numpy().astype(np.float64)
            assert joint_pos.shape == (7,), f"FrankaRenderer expects 7-DOF joint position, got shape {joint_pos.shape}"
            self.data.qpos[:7] = np.clip(
                joint_pos,
                self.model.jnt_range[:7, 0],
                self.model.jnt_range[:7, 1],
            )
            self.prev_qpos = self.data.qpos[:7].copy()
        else:
            assert arm.eef_position is not None, "FrankaRenderer requires EEF pose or joint position"
            assert arm.eef_rotation is not None, "FrankaRenderer requires EEF rotation or joint position"
            assert not arm.eef_position.is_relative, "render() expects an absolute action; convert delta actions first"

            eef_pos = arm.eef_position.data[0].detach().cpu().numpy().astype(np.float64)
            rot = _to_scipy_rotation(arm.eef_rotation.data.detach().cpu(), arm.eef_rotation.representation)
            eef_rot_mat = rot.as_matrix()[0].astype(np.float64)

            residual = self.solve_ik(
                target_pos=eef_pos,
                target_rot_mat=eef_rot_mat,
                qadr=self._arm_qadr,
                dofadr=self._arm_dofadr,
                jnt_range=self._arm_jnt_range,
                site_id=self.site_id,
                seed_qpos=self.prev_qpos,
                null_space_gain=0.05,
                damping=1e-4,
            )
            # Retry from HOME_QPOS when warm-start lands far from the target
            # (large EEF jump, joint-limit trap, etc.), so a bad seed can't
            # poison subsequent frames.
            if residual > 1e-2:
                self.solve_ik(
                    target_pos=eef_pos,
                    target_rot_mat=eef_rot_mat,
                    qadr=self._arm_qadr,
                    dofadr=self._arm_dofadr,
                    jnt_range=self._arm_jnt_range,
                    site_id=self.site_id,
                    seed_qpos=self.HOME_QPOS,
                    null_space_gain=0.05,
                    damping=1e-4,
                )
            self.prev_qpos = self.data.qpos[:7].copy()

        self.data.qpos[7] = gripper_pos
        self.data.qpos[8] = gripper_pos
        mujoco.mj_forward(self.model, self.data)

        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.scene_option)
        return self.renderer.render()

    def close(self):
        self.renderer.close()


_FRANKA_ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]


@RENDERER_REGISTRY.register(name=RobotType.DUAL_FRANKA.value)
class DualFrankaRenderer(BaseDualArmRenderer):
    """Dual Franka FR3 renderer — two arms mounted side by side."""

    MODEL_PATH = ("franka", "fr3.xml")
    ROOT_BODY = "link0"
    ARM_JOINTS = _FRANKA_ARM_JOINTS
    N_ARM = 7
    HOME_QPOS_ARM = np.array([0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398], dtype=np.float64)
    EEF_SITE = "hand_tcp"
    LEFT_MOUNT = ([0.0, 0.35, 0.0], [1.0, 0.0, 0.0, 0.0])
    RIGHT_MOUNT = ([0.0, -0.35, 0.0], [1.0, 0.0, 0.0, 0.0])
    CAM_LOOKAT = [0.0, 0.0, 0.25]
    CAM_DISTANCE = 2.2
    CAM_AZIMUTH = 180.0
    CAM_ELEVATION = -20.0
    IK_NULL_SPACE_GAIN = 0.05
    GRIPPER_OPEN_WIDTH = 0.04

    _FINGER_JOINTS = ("finger_joint1", "finger_joint2")

    def _setup_grippers(self):
        self._finger = {}
        for side, prefix in (("left", "left/"), ("right", "right/")):
            self._finger[side] = np.array(
                [self._joint_qadr(f"{prefix}{j}") for j in self._FINGER_JOINTS], dtype=np.int64
            )

    def _apply_gripper(self, gripper_norm: float, side: str):
        # Parallel jaw: both prismatic fingers open the same width.
        g = float(np.clip(gripper_norm, 0.0, 1.0)) * self.GRIPPER_OPEN_WIDTH
        fq = self._finger[side]
        self.data.qpos[fq[0]] = g
        self.data.qpos[fq[1]] = g
