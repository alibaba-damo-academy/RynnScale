"""WidowX 250s renderer.

MuJoCo model hand-translated from the Interbotix ``wx250s.urdf.xacro`` in
``interbotix_ros_xsarms/interbotix_xsarm_descriptions`` (visual STL meshes and
link/joint frames taken verbatim; inertials are nominal since this renderer
only runs forward kinematics). The 6-DOF arm is IK-solved from the recorded
EEF pose to the ``tcp`` site placed at the URDF ``ee_gripper_link`` grasp
frame — BridgeData V2 reports EEF poses in the arm base frame at that point.
"""

import os

import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

from ..constants import RobotType
from ..registry import RENDERER_REGISTRY
from ..utils.robot import RobotAction, _to_scipy_rotation
from ._lazy import mujoco
from .base import BaseRenderer


@RENDERER_REGISTRY.register(name=RobotType.WIDOWX_250S.value)
class Widowx250sRenderer(BaseRenderer):
    """6-DOF WidowX 250s arm + 2-finger parallel jaw, IK-solved from EEF pose."""

    # Bent "ready" pose that seats the gripper in the low, forward BridgeData V2
    # workspace — a good IK warm-start seed and retry fallback.
    HOME_QPOS = np.array([0.0, -0.6, 0.9, 0.0, 0.7, 0.0], dtype=np.float64)

    N_ARM = 6
    # left_finger ∈ [0.015, 0.037], right_finger = -left_finger (mimic). Width
    # per finger runs from FINGER_CLOSED (jaw shut) to FINGER_OPEN (jaw open).
    FINGER_CLOSED = 0.015
    FINGER_OPEN = 0.037

    # BridgeData V2 reports the EEF orientation in a frame whose identity points
    # the gripper straight *down* (the top-down tabletop teleop convention),
    # whereas the model's ``tcp`` site is aligned gripper-*forward* (+x at zero
    # joints). Post-multiplying the IK target rotation by +90° about the local
    # y-axis bridges the two conventions so the rendered gripper matches the
    # recorded pose (validated frame-by-frame against the observation cameras).
    _EEF_ROT_OFFSET = ScipyRotation.from_euler("y", 90, degrees=True).as_matrix()

    def __init__(self, height: int = 480, width: int = 480, **_unused):
        super().__init__(height=height, width=width)

        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        self.data.qpos[: self.N_ARM] = self.HOME_QPOS
        self._apply_gripper(1.0)  # start with the jaw open
        mujoco.mj_forward(self.model, self.data)

        self.prev_qpos = self.HOME_QPOS.copy()

        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0.25, 0.0, 0.15]
        self.camera.distance = 1.1
        self.camera.azimuth = 150
        self.camera.elevation = -25

        self.scene_option = mujoco.MjvOption()

        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        assert self.site_id >= 0, "tcp site not found in wx250s model"

        self._arm_qadr = np.arange(self.N_ARM, dtype=np.int64)
        self._arm_dofadr = np.arange(self.N_ARM, dtype=np.int64)
        self._arm_jnt_range = self.model.jnt_range[: self.N_ARM].copy()

        self._finger_qadr = np.array(
            [
                self._joint_qadr("left_finger"),
                self._joint_qadr("right_finger"),
            ],
            dtype=np.int64,
        )

    def _build_model(self):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(module_dir, "..", ".."))
        xml_path = os.path.join(project_root, "assets", "widowx_250s", "scene.xml")
        assert os.path.exists(xml_path), f"WidowX 250s model not found: {xml_path}"
        return mujoco.MjModel.from_xml_path(xml_path)

    def _apply_gripper(self, gripper_norm: float):
        """Map normalized gripper [0,1] (0=closed, 1=open) → finger slides."""
        norm = float(np.clip(gripper_norm, 0.0, 1.0))
        w = self.FINGER_CLOSED + norm * (self.FINGER_OPEN - self.FINGER_CLOSED)
        qadr = getattr(self, "_finger_qadr", None)
        if qadr is None:  # during __init__ before qadr is cached: fixed layout
            self.data.qpos[self.N_ARM] = w
            self.data.qpos[self.N_ARM + 1] = -w
        else:
            self.data.qpos[qadr[0]] = w
            self.data.qpos[qadr[1]] = -w

    def render(self, action: RobotAction) -> np.ndarray:
        assert len(action) == 1, f"render() expects a single-step action, got chunk={len(action)}"
        arm = action.left_arm
        assert arm is not None, "Widowx250sRenderer requires left_arm to be populated"
        ref = arm.joint_position if arm.joint_position is not None else arm.eef_position
        assert ref is not None, "Widowx250sRenderer requires either joint_position or eef pose"
        assert not ref.is_relative, "render() expects an absolute action; convert delta actions first"

        if arm.joint_position is not None:
            joint_pos = arm.joint_position.data[0].detach().cpu().numpy().astype(np.float64)
            assert joint_pos.shape == (self.N_ARM,), (
                f"Widowx250sRenderer expects 6-DOF joint position, got shape {joint_pos.shape}"
            )
            self.data.qpos[: self.N_ARM] = np.clip(
                joint_pos,
                self.model.jnt_range[: self.N_ARM, 0],
                self.model.jnt_range[: self.N_ARM, 1],
            )
            self.prev_qpos = self.data.qpos[: self.N_ARM].copy()
        else:
            assert arm.eef_position is not None, "Widowx250sRenderer requires EEF pose or joint position"
            assert arm.eef_rotation is not None, "Widowx250sRenderer requires EEF rotation or joint position"

            eef_pos = arm.eef_position.data[0].detach().cpu().numpy().astype(np.float64)
            rot = _to_scipy_rotation(arm.eef_rotation.data.detach().cpu(), arm.eef_rotation.representation)
            eef_rot_mat = rot.as_matrix()[0].astype(np.float64) @ self._EEF_ROT_OFFSET

            # 6-DOF arm → no redundancy → null-space disabled. Track the tcp site
            # (ee_gripper_link grasp frame). Retry from HOME on a bad warm-start.
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
            self._apply_gripper(float(action.left_gripper.data[0, 0].item()))

        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.scene_option)
        return self.renderer.render()

    def close(self):
        # mujoco.Renderer only grew a close() in newer releases; guard so the
        # renderer works across the versions pinned in different environments.
        if hasattr(self.renderer, "close"):
            self.renderer.close()
