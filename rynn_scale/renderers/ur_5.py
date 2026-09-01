import os

import numpy as np

from ..constants import RobotType
from ..registry import RENDERER_REGISTRY
from ..utils.robot import RobotAction, _to_scipy_rotation
from ._lazy import mujoco
from .base import BaseRenderer
from .dual_arm_base import BaseDualArmRenderer


@RENDERER_REGISTRY.register(name=RobotType.UR_5.value)
class Ur5Renderer(BaseRenderer):
    """Renders a UR5 arm + Robotiq 2F-85 gripper via MuJoCo.

    Uses UR5 kinematics (different link lengths from UR5e).
    """

    HOME_QPOS = np.array(
        [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
        dtype=np.float64,
    )

    N_ARM = 6
    FINGER_JOINT_MAX = 0.8

    _MIMIC_JOINTS = [
        "left_inner_finger_joint",
        "left_inner_knuckle_joint",
        "right_outer_knuckle_joint",
        "right_inner_finger_joint",
        "right_inner_knuckle_joint",
    ]
    _MIMIC_MULTIPLIERS = np.array([-1.0, 1.0, 1.0, -1.0, 1.0], dtype=np.float64)

    def __init__(self, height: int = 480, width: int = 480, **_unused):
        super().__init__(height=height, width=width)

        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        self._finger_qadr = self._joint_qadr("finger_joint")
        self._mimic_qadr = np.array(
            [self._joint_qadr(n) for n in self._MIMIC_JOINTS],
            dtype=np.int64,
        )

        self.data.qpos[: self.N_ARM] = self.HOME_QPOS
        mujoco.mj_forward(self.model, self.data)

        self.prev_qpos = self.HOME_QPOS.copy()

        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0.0, 0.0, 0.4]
        self.camera.distance = 1.8
        self.camera.azimuth = 135
        self.camera.elevation = -25

        self.scene_option = mujoco.MjvOption()

        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        assert self.site_id >= 0, "attachment_site not found in UR5 model"

        self._arm_qadr = np.arange(self.N_ARM, dtype=np.int64)
        self._arm_dofadr = np.arange(self.N_ARM, dtype=np.int64)
        self._arm_jnt_range = self.model.jnt_range[: self.N_ARM].copy()

    def _build_model(self):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(module_dir, "..", ".."))
        xml_path = os.path.join(project_root, "assets", "ur_5", "scene.xml")
        assert os.path.exists(xml_path), f"UR5 model not found: {xml_path}"
        return mujoco.MjModel.from_xml_path(xml_path)

    def _apply_gripper(self, gripper_norm: float):
        finger_val = (1.0 - gripper_norm) * self.FINGER_JOINT_MAX
        self.data.qpos[self._finger_qadr] = finger_val
        self.data.qpos[self._mimic_qadr] = self._MIMIC_MULTIPLIERS * finger_val

    def render(self, action: RobotAction) -> np.ndarray:
        assert len(action) == 1, f"render() expects a single-step action, got chunk={len(action)}"
        arm = action.left_arm
        assert arm is not None, "Ur5Renderer requires left_arm to be populated"
        ref = arm.joint_position if arm.joint_position is not None else arm.eef_position
        assert ref is not None, "Ur5Renderer requires either joint_position or eef pose"
        assert not ref.is_relative, "render() expects an absolute action; convert delta actions first"

        if arm.joint_position is not None:
            joint_pos = arm.joint_position.data[0].detach().cpu().numpy().astype(np.float64)
            assert joint_pos.shape == (self.N_ARM,), (
                f"Ur5Renderer expects 6-DOF joint position, got shape {joint_pos.shape}"
            )
            self.data.qpos[: self.N_ARM] = np.clip(
                joint_pos,
                self.model.jnt_range[: self.N_ARM, 0],
                self.model.jnt_range[: self.N_ARM, 1],
            )
            self.prev_qpos = self.data.qpos[: self.N_ARM].copy()
        else:
            assert arm.eef_position is not None, "Ur5Renderer requires EEF pose or joint position"
            assert arm.eef_rotation is not None, "Ur5Renderer requires EEF rotation or joint position"

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
                damping=1e-4,
            )
            if residual > 1e-2:
                self.solve_ik(
                    target_pos=eef_pos,
                    target_rot_mat=eef_rot_mat,
                    qadr=self._arm_qadr,
                    dofadr=self._arm_dofadr,
                    jnt_range=self._arm_jnt_range,
                    site_id=self.site_id,
                    seed_qpos=self.HOME_QPOS,
                    damping=1e-4,
                )
            self.prev_qpos = self.data.qpos[: self.N_ARM].copy()

        if action.left_gripper is not None:
            gripper_norm = float(
                np.clip(
                    action.left_gripper.data[0, 0].item(),
                    0.0,
                    1.0,
                )
            )
            self._apply_gripper(gripper_norm)

        mujoco.mj_forward(self.model, self.data)

        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.scene_option)
        return self.renderer.render()

    def close(self):
        self.renderer.close()


_UR5_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
_UR5_MIMIC_JOINTS = [
    "left_inner_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_inner_finger_joint",
    "right_inner_knuckle_joint",
]
_UR5_MIMIC_MULTIPLIERS = np.array([-1.0, 1.0, 1.0, -1.0, 1.0], dtype=np.float64)


@RENDERER_REGISTRY.register(name=RobotType.DUAL_UR_5.value)
class DualUr5Renderer(BaseDualArmRenderer):
    """Dual UR5 renderer — two arms with Robotiq 2F-85 grippers."""

    MODEL_PATH = ("ur_5", "ur5.xml")
    ROOT_BODY = "base"
    ARM_JOINTS = _UR5_ARM_JOINTS
    N_ARM = 6
    HOME_QPOS_ARM = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=np.float64)
    EEF_SITE = "attachment_site"
    # HOME_QPOS_ARM reaches out along +y (horiz dir ≈ [-0.22, 0.98]). Mount the
    # two bases along x (perpendicular to the reach) so the arms sit side by side
    # and their end-effectors stay flush, instead of strung out front-to-back.
    LEFT_MOUNT = ([-0.30, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    RIGHT_MOUNT = ([0.30, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    CAM_LOOKAT = [0.0, 0.0, 0.4]
    CAM_DISTANCE = 2.2
    CAM_AZIMUTH = 135.0
    CAM_ELEVATION = -25.0
    FINGER_JOINT_MAX = 0.8

    def _setup_grippers(self):
        self._gr = {}
        for side, prefix in (("left", "left/"), ("right", "right/")):
            self._gr[side] = dict(
                finger=self._joint_qadr(f"{prefix}finger_joint"),
                mimic=np.array([self._joint_qadr(f"{prefix}{n}") for n in _UR5_MIMIC_JOINTS], dtype=np.int64),
            )

    def _apply_gripper(self, gripper_norm: float, side: str):
        g = self._gr[side]
        finger_val = (1.0 - gripper_norm) * self.FINGER_JOINT_MAX
        self.data.qpos[g["finger"]] = finger_val
        self.data.qpos[g["mimic"]] = _UR5_MIMIC_MULTIPLIERS * finger_val
