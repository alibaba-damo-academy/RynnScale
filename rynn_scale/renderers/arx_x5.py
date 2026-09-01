"""ARX-X5 renderer — 6-DOF ARX X5A arm + 2-finger parallel jaw.

The MJCF at `assets/arx_x5/x5a.xml` is built from the official ARXroboticsX
X5A URDF (kinematics, meshes, inertias, colors). Realistic joint limits, the
home keyframe, position actuators and the symmetric-finger coupling are
retained from the previous ARX-L5 model — the stock URDF only ships
placeholder +/-10 rad limits and no actuators.
"""

import os

import numpy as np

from ..constants import RobotType
from ..registry import RENDERER_REGISTRY
from ..utils.robot import RobotAction, _to_scipy_rotation
from ._lazy import mujoco
from .base import BaseRenderer
from .dual_arm_base import BaseDualArmRenderer


@RENDERER_REGISTRY.register(name=RobotType.ARX_X5.value)
class ArxX5Renderer(BaseRenderer):
    """6-DOF ARX arm + 2-finger parallel jaw."""

    # "home" keyframe from x5a.xml. Fingers fully open (±0.044 sym).
    HOME_QPOS = np.array(
        [0.0, 0.251, 0.314, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    HOME_FINGER = np.array([0.044, -0.044], dtype=np.float64)

    N_ARM = 6
    N_FINGER = 2
    # joint7 ∈ [0, 0.044], joint8 ∈ [-0.044, 0]; both move symmetrically with width.
    FINGER_MAX = 0.044

    # No site is defined in x5a.xml; we use the last arm body
    # (link6 → the wrist) as the IK reference frame.
    EEF_BODY = "link6"

    def __init__(self, height: int = 480, width: int = 480, **_unused):
        super().__init__(height=height, width=width)

        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        self.data.qpos[: self.N_ARM] = self.HOME_QPOS
        self.data.qpos[self.N_ARM : self.N_ARM + self.N_FINGER] = self.HOME_FINGER
        mujoco.mj_forward(self.model, self.data)

        self.prev_qpos = self.HOME_QPOS.copy()

        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0.0, 0.0, 0.3]
        self.camera.distance = 1.2
        self.camera.azimuth = 135
        self.camera.elevation = -25

        self.scene_option = mujoco.MjvOption()

        self.eef_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.EEF_BODY,
        )
        assert self.eef_body_id >= 0, f"{self.EEF_BODY} body not found in ARX-X5A model"

        self._arm_qadr = np.arange(self.N_ARM, dtype=np.int64)
        self._arm_dofadr = np.arange(self.N_ARM, dtype=np.int64)
        self._arm_jnt_range = self.model.jnt_range[: self.N_ARM].copy()

    def _build_model(self):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(module_dir, "..", ".."))
        xml_path = os.path.join(project_root, "assets", "arx_x5", "scene.xml")
        assert os.path.exists(xml_path), f"ARX-X5A model not found: {xml_path}"
        return mujoco.MjModel.from_xml_path(xml_path)

    def _apply_gripper(self, gripper_norm: float):
        """Map normalized gripper [0,1] → both fingers (joint7=+w, joint8=-w)."""
        # Input is normalized: 0 = fully closed, 1 = fully open. Open width per
        # finger is FINGER_MAX (joint7=+w, joint8=-w move symmetrically).
        w = float(np.clip(gripper_norm, 0.0, 1.0)) * self.FINGER_MAX
        self.data.qpos[self.N_ARM] = w
        self.data.qpos[self.N_ARM + 1] = -w

    def render(self, action: RobotAction) -> np.ndarray:
        assert len(action) == 1, f"render() expects a single-step action, got chunk={len(action)}"
        arm = action.left_arm
        assert arm is not None, "ArxX5Renderer requires left_arm to be populated"
        ref = arm.joint_position if arm.joint_position is not None else arm.eef_position
        assert ref is not None, "ArxX5Renderer requires either joint_position or eef pose"
        assert not ref.is_relative, "render() expects an absolute action; convert delta actions first"

        if arm.joint_position is not None:
            joint_pos = arm.joint_position.data[0].detach().cpu().numpy().astype(np.float64)
            assert joint_pos.shape == (self.N_ARM,), (
                f"ArxX5Renderer expects 6-DOF joint position, got shape {joint_pos.shape}"
            )
            self.data.qpos[: self.N_ARM] = np.clip(
                joint_pos,
                self.model.jnt_range[: self.N_ARM, 0],
                self.model.jnt_range[: self.N_ARM, 1],
            )
            self.prev_qpos = self.data.qpos[: self.N_ARM].copy()
        else:
            assert arm.eef_position is not None, "ArxX5Renderer requires EEF pose or joint position"
            assert arm.eef_rotation is not None, "ArxX5Renderer requires EEF rotation or joint position"

            eef_pos = arm.eef_position.data[0].detach().cpu().numpy().astype(np.float64)
            rot = _to_scipy_rotation(arm.eef_rotation.data.detach().cpu(), arm.eef_rotation.representation)
            eef_rot_mat = rot.as_matrix()[0].astype(np.float64)

            # No site in x5a.xml; track the wrist body (link6).
            # 6-DOF arm → no redundancy → null-space disabled.
            residual = self.solve_ik(
                target_pos=eef_pos,
                target_rot_mat=eef_rot_mat,
                qadr=self._arm_qadr,
                dofadr=self._arm_dofadr,
                jnt_range=self._arm_jnt_range,
                body_id=self.eef_body_id,
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
                    body_id=self.eef_body_id,
                    seed_qpos=self.HOME_QPOS,
                    damping=1e-4,
                )
            self.prev_qpos = self.data.qpos[: self.N_ARM].copy()

        if action.left_gripper is not None:
            self._apply_gripper(float(action.left_gripper.data[0, 0].item()))

        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.scene_option)
        return self.renderer.render()

    def close(self):
        self.renderer.close()


_ARX_ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]


@RENDERER_REGISTRY.register(name=RobotType.DUAL_ARX_X5.value)
class DualArxX5Renderer(BaseDualArmRenderer):
    """Dual ARX X5 renderer — two 6-DOF arms with 2-finger parallel jaw."""

    MODEL_PATH = ("arx_x5", "x5a.xml")
    ROOT_BODY = "base_link"
    ARM_JOINTS = _ARX_ARM_JOINTS
    N_ARM = 6
    HOME_QPOS_ARM = np.array([0.0, 0.251, 0.314, 0.0, 0.0, 0.0], dtype=np.float64)
    EEF_BODY = "link6"
    LEFT_MOUNT = ([0.0, 0.20, 0.0], [1.0, 0.0, 0.0, 0.0])
    RIGHT_MOUNT = ([0.0, -0.20, 0.0], [1.0, 0.0, 0.0, 0.0])
    CAM_LOOKAT = [0.0, 0.0, 0.2]
    CAM_DISTANCE = 1.5
    CAM_AZIMUTH = 135.0
    CAM_ELEVATION = -25.0
    FINGER_MAX = 0.044

    _FINGER_JOINTS = ("joint7", "joint8")

    def _add_lights(self, spec):
        # Match the single-arm scene.xml: bright ambient headlight keeps the dark
        # meshes visible, plus two point lights and a directional fill.
        spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
        spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
        spec.visual.headlight.specular = [0.0, 0.0, 0.0]
        l1 = spec.worldbody.add_light()
        l1.pos = [0.0, 0.0, 2.0]
        l1.dir = [0.0, 0.0, -1.0]
        l1.diffuse = [0.6, 0.6, 0.6]
        l2 = spec.worldbody.add_light()
        l2.pos = [0.5, 0.5, 1.5]
        l2.dir = [-0.3, -0.3, -1.0]
        l2.diffuse = [0.4, 0.4, 0.4]
        l3 = spec.worldbody.add_light()
        l3.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        l3.pos = [0.0, 0.0, 1.5]
        l3.dir = [0.0, 0.0, -1.0]
        l3.diffuse = [0.5, 0.5, 0.5]

    def _setup_grippers(self):
        self._finger = {}
        for side, prefix in (("left", "left/"), ("right", "right/")):
            fq = np.array([self._joint_qadr(f"{prefix}{j}") for j in self._FINGER_JOINTS], dtype=np.int64)
            self._finger[side] = fq
            self.data.qpos[fq[0]] = self.FINGER_MAX
            self.data.qpos[fq[1]] = -self.FINGER_MAX

    def _apply_gripper(self, gripper_norm: float, side: str):
        self._symmetric_gripper(self._finger[side], gripper_norm, self.FINGER_MAX)
