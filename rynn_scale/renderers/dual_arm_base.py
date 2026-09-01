"""Shared base class for two-arm renderers.

Both arms are the same single-arm model attached side by side at configurable
mounts. Each arm's EEF target is expressed in ITS OWN ``base_link`` frame; the
renderer lifts it to world through the (fixed) mount transform before IK. This
per-arm-base convention keeps the EEF pose independent of how the two arms are
mounted — swapping the ``LEFT_MOUNT`` / ``RIGHT_MOUNT`` constants adapts the
renderer to a different dual-arm layout without touching the URDF or IK.

Subclasses declare a handful of class constants (asset, joints, DOF, EEF frame,
mounts, camera) and may override the hooks ``_add_lights``, ``_postprocess_model``,
``_setup_grippers`` and ``_apply_gripper``.
"""

import os

import numpy as np

from ..utils.robot import RobotAction, _to_scipy_rotation
from ._lazy import mujoco
from .base import BaseRenderer

_SIDES = (("left", "left/"), ("right", "right/"))


class BaseDualArmRenderer(BaseRenderer):
    # --- subclass-defined geometry / policy (class constants) ----------------
    MODEL_PATH: tuple = ()  # path segments under assets/, e.g. ("arx_x5", "x5a.xml")
    ROOT_BODY: str = "base_link"  # body attached at each mount frame
    ARM_JOINTS: tuple = ()  # per-arm joint names (without left/right prefix)
    N_ARM: int = 6
    HOME_QPOS_ARM: np.ndarray = None
    EEF_BODY: str = None  # IK target body … or
    EEF_SITE: str = None  # … IK target site (exactly one is set)
    # (pos, quat_wxyz) of each arm base in the renderer world frame.
    LEFT_MOUNT = ([0.0, 0.30, 0.0], [1.0, 0.0, 0.0, 0.0])
    RIGHT_MOUNT = ([0.0, -0.30, 0.0], [1.0, 0.0, 0.0, 0.0])
    CAM_LOOKAT = [0.0, 0.0, 0.2]
    CAM_DISTANCE = 1.5
    CAM_AZIMUTH = 135.0
    CAM_ELEVATION = -25.0
    IK_NULL_SPACE_GAIN = 0.0
    IK_RETRY_THRESH = 1e-2

    def __init__(self, height: int = 480, width: int = 480, action_source: str = "joint"):
        super().__init__(height=height, width=width)
        self.action_source = self._resolve_action_source(action_source)

        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        self._arm = {}
        for side, prefix in _SIDES:
            names = [f"{prefix}{n}" for n in self.ARM_JOINTS]
            qadr = np.array([self._joint_qadr(n) for n in names], dtype=np.int64)
            dofadr = np.array([self._joint_dofadr(n) for n in names], dtype=np.int64)
            jrange = self._jnt_ranges(names)
            body_id, site_id = -1, -1
            if self.EEF_BODY is not None:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{self.EEF_BODY}")
                assert body_id >= 0, f"EEF body {prefix}{self.EEF_BODY} not found"
            else:
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}{self.EEF_SITE}")
                assert site_id >= 0, f"EEF site {prefix}{self.EEF_SITE} not found"
            self._arm[side] = dict(
                prefix=prefix,
                qadr=qadr,
                dofadr=dofadr,
                jrange=jrange,
                body_id=body_id,
                site_id=site_id,
                prev=self.HOME_QPOS_ARM.copy(),
            )
            self.data.qpos[qadr] = self.HOME_QPOS_ARM

        # Grippers (subclass caches finger addresses and may set an initial pose).
        self._setup_grippers()
        mujoco.mj_forward(self.model, self.data)

        # Each arm base is fixed to the (static) platform → cache its world pose.
        for side, prefix in _SIDES:
            bp, bm = self._base_world_pose(f"{prefix}{self.ROOT_BODY}")
            self._arm[side]["base_pos"] = bp
            self._arm[side]["base_mat"] = bm

        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = self.CAM_LOOKAT
        self.camera.distance = self.CAM_DISTANCE
        self.camera.azimuth = self.CAM_AZIMUTH
        self.camera.elevation = self.CAM_ELEVATION
        self.scene_option = mujoco.MjvOption()

    # --- model assembly ------------------------------------------------------

    def _build_model(self):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(module_dir, "..", ".."))
        model_path = os.path.join(project_root, "assets", *self.MODEL_PATH)
        assert os.path.exists(model_path), f"model not found: {model_path}"

        parent = mujoco.MjSpec()
        self._add_lights(parent)
        for prefix, (pos, quat) in (("left/", self.LEFT_MOUNT), ("right/", self.RIGHT_MOUNT)):
            child = mujoco.MjSpec.from_file(model_path)
            for key in list(child.keys):
                child.delete(key)
            for act in list(child.actuators):
                child.delete(act)
            frame = parent.worldbody.add_frame()
            frame.pos = list(pos)
            frame.quat = list(quat)
            frame.attach_body(child.body(self.ROOT_BODY), prefix, "")

        model = parent.compile()
        self._postprocess_model(model)
        return model

    # --- hooks (override in subclasses) --------------------------------------

    def _add_lights(self, spec):
        """Default two-point lighting (matches UR5 / Franka)."""
        l1 = spec.worldbody.add_light()
        l1.pos = [0.0, 0.0, 2.0]
        l1.dir = [0.0, 0.0, -1.0]
        l1.diffuse = [0.6, 0.6, 0.6]
        l2 = spec.worldbody.add_light()
        l2.pos = [0.5, 0.5, 1.5]
        l2.dir = [-0.3, -0.3, -1.0]
        l2.diffuse = [0.4, 0.4, 0.4]

    def _postprocess_model(self, model):
        """No-op by default; override to recolor geoms etc."""

    def _setup_grippers(self):
        """Cache per-arm finger addresses (and optional initial pose)."""
        raise NotImplementedError

    def _apply_gripper(self, gripper_norm: float, side: str):
        """Map a normalized [0,1] open fraction to finger qpos for ``side``."""
        raise NotImplementedError

    def _symmetric_gripper(self, finger_qadr, gripper_norm: float, max_w: float):
        """Common parallel-jaw driver: qpos = (+w, -w) with w = norm·max_w."""
        w = float(np.clip(gripper_norm, 0.0, 1.0)) * max_w
        self.data.qpos[finger_qadr[0]] = w
        self.data.qpos[finger_qadr[1]] = -w

    # --- per-arm drive -------------------------------------------------------

    def _apply_arm(self, side: str, arm, gripper):
        info = self._arm[side]
        if self._use_joint(arm):
            qpos = arm.joint_position.data[0].detach().cpu().numpy().astype(np.float64)
            assert qpos.shape == (self.N_ARM,), (
                f"{type(self).__name__} expects {self.N_ARM}-DOF per arm, got shape {qpos.shape}"
            )
            self.data.qpos[info["qadr"]] = np.clip(qpos, info["jrange"][:, 0], info["jrange"][:, 1])
        else:
            assert arm.eef_position is not None and arm.eef_rotation is not None, (
                f"{type(self).__name__} requires joint_position or full EEF pose"
            )
            eef_pos = arm.eef_position.data[0].detach().cpu().numpy().astype(np.float64)
            rot = _to_scipy_rotation(arm.eef_rotation.data.detach().cpu(), arm.eef_rotation.representation)
            eef_rot = rot.as_matrix()[0].astype(np.float64)
            # EEF pose is in this arm's base frame → lift to world for IK.
            target_pos = info["base_mat"] @ eef_pos + info["base_pos"]
            target_rot = info["base_mat"] @ eef_rot
            residual = self.solve_ik(
                target_pos=target_pos,
                target_rot_mat=target_rot,
                qadr=info["qadr"],
                dofadr=info["dofadr"],
                jnt_range=info["jrange"],
                body_id=info["body_id"],
                site_id=info["site_id"],
                seed_qpos=info["prev"],
                null_space_gain=self.IK_NULL_SPACE_GAIN,
                damping=1e-4,
            )
            if residual > self.IK_RETRY_THRESH:
                self.solve_ik(
                    target_pos=target_pos,
                    target_rot_mat=target_rot,
                    qadr=info["qadr"],
                    dofadr=info["dofadr"],
                    jnt_range=info["jrange"],
                    body_id=info["body_id"],
                    site_id=info["site_id"],
                    seed_qpos=self.HOME_QPOS_ARM,
                    null_space_gain=self.IK_NULL_SPACE_GAIN,
                    damping=1e-4,
                )
        info["prev"] = self.data.qpos[info["qadr"]].copy()

        if gripper is not None:
            self._apply_gripper(float(np.clip(gripper.data[0, 0].item(), 0.0, 1.0)), side)

    def render(self, action: RobotAction) -> np.ndarray:
        assert len(action) == 1, f"render() expects a single-step action, got chunk={len(action)}"
        pairs = (
            ("left", action.left_arm, action.left_gripper),
            ("right", action.right_arm, action.right_gripper),
        )
        for side, arm, grip in pairs:
            if arm is None:
                self.data.qpos[self._arm[side]["qadr"]] = self.HOME_QPOS_ARM
                self._apply_gripper(0.0, side)
                continue
            ref = arm.joint_position if arm.joint_position is not None else arm.eef_position
            assert ref is not None
            assert not ref.is_relative, "render() expects an absolute action; convert delta actions first"
            self._apply_arm(side, arm, grip)

        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.scene_option)
        return self.renderer.render()

    def close(self):
        self.renderer.close()
