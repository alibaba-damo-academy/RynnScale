import os

import numpy as np

from ..constants import RobotType
from ..registry import RENDERER_REGISTRY
from ..utils.robot import RobotAction, _to_scipy_rotation
from ._lazy import mujoco
from .base import BaseRenderer


@RENDERER_REGISTRY.register(name=RobotType.UR_5E.value)
class Ur5eRenderer(BaseRenderer):
    """Renders a UR5e arm + Robotiq 2F-85 gripper via MuJoCo.

    The Robotiq 2F-85 is a parallel-jaw gripper with a four-bar linkage
    mechanism. Only ``finger_joint`` is the driver; five mimic joints are
    set manually (``mj_forward`` does not enforce equality constraints).
    Gripper input is normalized: 0 = fully closed, 1 = fully open.
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
        assert self.site_id >= 0, "attachment_site not found in UR5e model"

        self._arm_qadr = np.arange(self.N_ARM, dtype=np.int64)
        self._arm_dofadr = np.arange(self.N_ARM, dtype=np.int64)
        self._arm_jnt_range = self.model.jnt_range[: self.N_ARM].copy()

    def _build_model(self):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(module_dir, "..", ".."))
        xml_path = os.path.join(project_root, "assets", "ur_5e", "scene.xml")
        assert os.path.exists(xml_path), f"UR5e model not found: {xml_path}"
        return mujoco.MjModel.from_xml_path(xml_path)

    def _apply_gripper(self, gripper_norm: float):
        finger_val = (1.0 - gripper_norm) * self.FINGER_JOINT_MAX
        self.data.qpos[self._finger_qadr] = finger_val
        self.data.qpos[self._mimic_qadr] = self._MIMIC_MULTIPLIERS * finger_val

    def render(self, action: RobotAction) -> np.ndarray:
        assert len(action) == 1, f"render() expects a single-step action, got chunk={len(action)}"
        arm = action.left_arm
        assert arm is not None, "Ur5eRenderer requires left_arm to be populated"
        ref = arm.joint_position if arm.joint_position is not None else arm.eef_position
        assert ref is not None, "Ur5eRenderer requires either joint_position or eef pose"
        assert not ref.is_relative, "render() expects an absolute action; convert delta actions first"

        if arm.joint_position is not None:
            joint_pos = arm.joint_position.data[0].detach().cpu().numpy().astype(np.float64)
            assert joint_pos.shape == (self.N_ARM,), (
                f"Ur5eRenderer expects 6-DOF joint position, got shape {joint_pos.shape}"
            )
            self.data.qpos[: self.N_ARM] = np.clip(
                joint_pos,
                self.model.jnt_range[: self.N_ARM, 0],
                self.model.jnt_range[: self.N_ARM, 1],
            )
            self.prev_qpos = self.data.qpos[: self.N_ARM].copy()
        else:
            assert arm.eef_position is not None, "Ur5eRenderer requires EEF pose or joint position"
            assert arm.eef_rotation is not None, "Ur5eRenderer requires EEF rotation or joint position"

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


_UR5E_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
_UR5E_MIMIC_JOINTS = [
    "left_inner_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_inner_finger_joint",
    "right_inner_knuckle_joint",
]
_UR5E_MIMIC_MULTIPLIERS = np.array([-1.0, 1.0, 1.0, -1.0, 1.0], dtype=np.float64)


@RENDERER_REGISTRY.register(name=RobotType.DUAL_UR_5E.value)
@RENDERER_REGISTRY.register(name=RobotType.DUAL_UR_5E_DEX.value)
class DualUr5eRenderer(BaseRenderer):
    """Dual UR5e renderer — two arms with Robotiq 2F-85 grippers."""

    HOME_QPOS_ARM = np.array(
        [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
        dtype=np.float64,
    )
    N_ARM = 6
    FINGER_JOINT_MAX = 0.8

    def __init__(self, height: int = 480, width: int = 480, **_unused):
        super().__init__(height=height, width=width)

        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        self._arm_l_qadr = np.array(
            [self._joint_qadr(f"left/{n}") for n in _UR5E_ARM_JOINTS],
            dtype=np.int64,
        )
        self._arm_r_qadr = np.array(
            [self._joint_qadr(f"right/{n}") for n in _UR5E_ARM_JOINTS],
            dtype=np.int64,
        )
        self._arm_l_jnt_range = self._jnt_ranges([f"left/{n}" for n in _UR5E_ARM_JOINTS])
        self._arm_r_jnt_range = self._jnt_ranges([f"right/{n}" for n in _UR5E_ARM_JOINTS])

        self._arm_l_dofadr = np.array(
            [self._joint_dofadr(f"left/{n}") for n in _UR5E_ARM_JOINTS],
            dtype=np.int64,
        )
        self._arm_r_dofadr = np.array(
            [self._joint_dofadr(f"right/{n}") for n in _UR5E_ARM_JOINTS],
            dtype=np.int64,
        )

        self._finger_l_qadr = self._joint_qadr("left/finger_joint")
        self._finger_r_qadr = self._joint_qadr("right/finger_joint")
        self._mimic_l_qadr = np.array(
            [self._joint_qadr(f"left/{n}") for n in _UR5E_MIMIC_JOINTS],
            dtype=np.int64,
        )
        self._mimic_r_qadr = np.array(
            [self._joint_qadr(f"right/{n}") for n in _UR5E_MIMIC_JOINTS],
            dtype=np.int64,
        )

        self._site_l = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "left/attachment_site",
        )
        self._site_r = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "right/attachment_site",
        )
        assert self._site_l >= 0 and self._site_r >= 0

        self.data.qpos[self._arm_l_qadr] = self.HOME_QPOS_ARM
        self.data.qpos[self._arm_r_qadr] = self.HOME_QPOS_ARM
        mujoco.mj_forward(self.model, self.data)

        self.prev_qpos_l = self.HOME_QPOS_ARM.copy()
        self.prev_qpos_r = self.HOME_QPOS_ARM.copy()

        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0.0, 0.0, 0.4]
        self.camera.distance = 2.2
        self.camera.azimuth = 135
        self.camera.elevation = -25

        self.scene_option = mujoco.MjvOption()

    def _build_model(self):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(module_dir, "..", ".."))
        xml_path = os.path.join(project_root, "assets", "ur_5e", "ur5e.xml")
        assert os.path.exists(xml_path), f"UR5e model not found: {xml_path}"

        parent = mujoco.MjSpec()
        light = parent.worldbody.add_light()
        light.pos = [0.0, 0.0, 2.0]
        light.dir = [0.0, 0.0, -1.0]
        light.diffuse = [0.6, 0.6, 0.6]
        light2 = parent.worldbody.add_light()
        light2.pos = [0.5, 0.5, 1.5]
        light2.dir = [-0.3, -0.3, -1.0]
        light2.diffuse = [0.4, 0.4, 0.4]

        # HOME_QPOS_ARM reaches out along +y, so mount the two bases along x
        # (perpendicular to the reach) — arms sit side by side with flush
        # end-effectors instead of strung out front-to-back.
        for prefix, x in (("left/", -0.30), ("right/", 0.30)):
            child = mujoco.MjSpec.from_file(xml_path)
            for key in list(child.keys):
                child.delete(key)
            for act in list(child.actuators):
                child.delete(act)
            frame = parent.worldbody.add_frame()
            frame.pos = [x, 0.0, 0.0]
            frame.attach_body(child.body("base"), prefix, "")

        return parent.compile()

    def _apply_gripper(self, gripper_norm: float, finger_qadr, mimic_qadr):
        finger_val = (1.0 - gripper_norm) * self.FINGER_JOINT_MAX
        self.data.qpos[finger_qadr] = finger_val
        self.data.qpos[mimic_qadr] = _UR5E_MIMIC_MULTIPLIERS * finger_val

    def _apply_arm(self, arm, gripper, qadr, jrange, dofadr, site_id, finger_qadr, mimic_qadr, prev_qpos):
        if arm.joint_position is not None:
            qpos = arm.joint_position.data[0].detach().cpu().numpy().astype(np.float64)
            assert qpos.shape == (self.N_ARM,), f"DualUr5eRenderer expects 6-DOF per arm, got shape {qpos.shape}"
            self.data.qpos[qadr] = np.clip(qpos, jrange[:, 0], jrange[:, 1])
            new_prev = self.data.qpos[qadr].copy()
        else:
            assert arm.eef_position is not None and arm.eef_rotation is not None
            eef_pos = arm.eef_position.data[0].detach().cpu().numpy().astype(np.float64)
            rot = _to_scipy_rotation(
                arm.eef_rotation.data.detach().cpu(),
                arm.eef_rotation.representation,
            )
            eef_rot_mat = rot.as_matrix()[0].astype(np.float64)
            residual = self.solve_ik(
                target_pos=eef_pos,
                target_rot_mat=eef_rot_mat,
                qadr=qadr,
                dofadr=dofadr,
                jnt_range=jrange,
                site_id=site_id,
                seed_qpos=prev_qpos,
                damping=1e-4,
            )
            if residual > 1e-2:
                self.solve_ik(
                    target_pos=eef_pos,
                    target_rot_mat=eef_rot_mat,
                    qadr=qadr,
                    dofadr=dofadr,
                    jnt_range=jrange,
                    site_id=site_id,
                    seed_qpos=self.HOME_QPOS_ARM,
                    damping=1e-4,
                )
            new_prev = self.data.qpos[qadr].copy()

        if gripper is not None:
            g = float(np.clip(gripper.data[0, 0].item(), 0.0, 1.0))
            self._apply_gripper(g, finger_qadr, mimic_qadr)

        return new_prev

    def render(self, action: RobotAction) -> np.ndarray:
        assert len(action) == 1, f"render() expects single-step action, got chunk={len(action)}"
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
                self._site_l,
                self._finger_l_qadr,
                self._mimic_l_qadr,
                self.prev_qpos_l,
            )
        else:
            self.data.qpos[self._arm_l_qadr] = self.HOME_QPOS_ARM

        if action.right_arm is not None:
            self.prev_qpos_r = self._apply_arm(
                action.right_arm,
                action.right_gripper,
                self._arm_r_qadr,
                self._arm_r_jnt_range,
                self._arm_r_dofadr,
                self._site_r,
                self._finger_r_qadr,
                self._mimic_r_qadr,
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
