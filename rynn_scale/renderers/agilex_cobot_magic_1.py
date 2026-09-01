"""AgileX Cobot Magic 1 renderer (two ARX5p2 6-DOF arms, official URDF).

Uses the official AgileX description from
github.com/agilexrobotics/mobile_aloha_sim (branch master).
"""

import numpy as np

from ..constants import RobotType
from ..registry import RENDERER_REGISTRY
from ._lazy import mujoco
from .dual_arm_base import BaseDualArmRenderer


@RENDERER_REGISTRY.register(name=RobotType.AGILEX_COBOT_MAGIC_1.value)
class AgilexCobotMagic1Renderer(BaseDualArmRenderer):
    """Dual ARX5p2 renderer (per-arm base frame)."""

    MODEL_PATH = ("agilex_cobot_magic_1", "arx5p2.urdf")
    ROOT_BODY = "base_link"
    ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
    N_ARM = 6
    HOME_QPOS_ARM = np.array([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0], dtype=np.float64)
    # 6-DOF arm → no site defined; track the wrist body (link6) for IK.
    EEF_BODY = "link6"
    LEFT_MOUNT = ([0.0, 0.30, 0.0], [1.0, 0.0, 0.0, 0.0])
    RIGHT_MOUNT = ([0.0, -0.30, 0.0], [1.0, 0.0, 0.0, 0.0])
    CAM_LOOKAT = [0.10, 0.0, 0.10]
    CAM_DISTANCE = 1.40
    CAM_AZIMUTH = 0.0
    CAM_ELEVATION = -20.0
    FINGER_MAX = 0.04

    _FINGER_JOINTS = ("joint7", "joint8")
    _GEOM_RGBA = np.array([0.79, 0.82, 0.93, 1.0], dtype=np.float32)

    def _add_lights(self, spec):
        l1 = spec.worldbody.add_light()
        l1.pos = [0.0, 0.0, 2.0]
        l1.dir = [0.0, 0.0, -1.0]
        l1.diffuse = [0.9, 0.9, 0.9]
        l1.specular = [0.5, 0.5, 0.5]
        l2 = spec.worldbody.add_light()
        l2.pos = [0.5, 0.5, 1.5]
        l2.dir = [-0.3, -0.3, -1.0]
        l2.diffuse = [0.5, 0.5, 0.5]

    def _postprocess_model(self, model):
        for i in range(model.ngeom):
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
                model.geom_rgba[i] = self._GEOM_RGBA

    def _setup_grippers(self):
        self._finger = {}
        for side, prefix in (("left", "left/"), ("right", "right/")):
            self._finger[side] = np.array(
                [self._joint_qadr(f"{prefix}{j}") for j in self._FINGER_JOINTS], dtype=np.int64
            )

    def _apply_gripper(self, gripper_norm: float, side: str):
        self._symmetric_gripper(self._finger[side], gripper_norm, self.FINGER_MAX)
