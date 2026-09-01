import os
from abc import ABC, abstractmethod

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

from ..utils.robot import RobotAction
from ._lazy import mujoco


class BaseRenderer(ABC):
    """Abstract base class for robot renderers.

    A renderer takes an absolute single-step ``RobotAction`` and produces
    an RGB image of the robot at that pose. Single-arm renderers ignore
    fields belonging to the other arm.

    Subclasses share one damped-least-squares IK solver (:meth:`solve_ik`)
    and the joint-address helpers below. Each renderer keeps only its own
    policy: which frame the dataset reports EEF poses in, the asset/camera,
    gripper mapping, and whether to retry from HOME on convergence failure.
    """

    # Valid values for the ``action_source`` policy shared by renderers that
    # can drive an arm from either recorded joints or IK-solved EEF poses.
    ACTION_SOURCES = ("eef", "joint")

    def __init__(self, height: int = 480, width: int = 480):
        self.height = height
        self.width = width

    @abstractmethod
    def render(self, action: RobotAction) -> np.ndarray: ...

    def close(self):
        pass

    # --- action-source policy ------------------------------------------------

    def _resolve_action_source(self, action_source: str) -> str:
        assert action_source in self.ACTION_SOURCES, (
            f"action_source must be one of {self.ACTION_SOURCES}, got {action_source!r}"
        )
        return action_source

    def _use_joint(self, arm) -> bool:
        """Whether to drive ``arm`` from recorded joints (vs. EEF + IK).

        Honors ``self.action_source``, falling back to the other source when
        the preferred one is absent (an ``Arm`` always carries at least one of
        joint_position / eef pose).
        """
        if self.action_source == "joint":
            return arm.joint_position is not None
        return arm.eef_position is None

    # --- model introspection helpers (require self.model) --------------------

    def _joint_qadr(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, f"joint {name} not found"
        return int(self.model.jnt_qposadr[jid])

    def _joint_dofadr(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, f"joint {name} not found"
        return int(self.model.jnt_dofadr[jid])

    def _jnt_ranges(self, names) -> np.ndarray:
        return np.stack(
            [self.model.jnt_range[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
        )

    def _base_world_pose(self, body_name: str):
        """World-frame (pos, 3x3 rot) of a body — requires a prior mj_forward.

        Used to cache a fixed arm-base transform so EEF targets reported in the
        arm's base frame can be lifted to world for IK.
        """
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        assert bid >= 0, f"body {body_name} not found"
        return (
            self.data.xpos[bid].copy(),
            self.data.xmat[bid].reshape(3, 3).copy(),
        )

    # --- shared IK solver ----------------------------------------------------

    def solve_ik(
        self,
        *,
        target_pos: np.ndarray,
        target_rot_mat: np.ndarray,
        qadr: np.ndarray,
        dofadr: np.ndarray,
        jnt_range: np.ndarray,
        seed_qpos: np.ndarray,
        site_id: int = -1,
        body_id: int = -1,
        rest_qpos: np.ndarray = None,
        null_space_gain: float = 0.0,
        damping: float = 1e-4,
        adaptive_damping: bool = True,
        manip_ratio: float = 1.0,
        max_damping: float = 1e-1,
        max_step: float = 0.5,
        max_iter: int = 200,
        tol: float = 1e-4,
    ) -> float:
        """Damped-least-squares IK that drives ``qpos[qadr]`` in place.

        Tracks either a site (``site_id``) or a body (``body_id``) frame to the
        given world-frame target and returns the final 6-D pose error norm.
        Callers transform dataset-frame poses into world frame and apply any
        retry policy themselves.

        Four anti-jitter techniques are folded in so consecutive frames yield
        nearby joint configurations:

        1. **Warm-start** — the solve begins from ``seed_qpos`` (typically the
           previous frame's solution), so nearby targets converge nearby.
        2. **Least-motion null bias** — for redundant arms, the spare DOFs are
           pulled toward ``rest_qpos`` (default ``seed_qpos``) through a *true*
           null projector ``I - J⁺J`` (tiny λ for numerical safety only). The
           damped projector ``I - Jᵀ(JJᵀ+λI)⁻¹J`` leaks the secondary task into
           task space; the true projector does not.
        3. **Manipulability-gated damping** — λ grows from ``damping`` toward
           ``max_damping`` as the Yoshikawa manipulability ``w=√det(JJᵀ)``
           collapses near a singularity. The threshold is captured at the seed
           (``w₀ = manip_ratio·w_seed``) so the ramp self-scales per robot and
           only fires when the pose grows *less* manipulable than where it
           started. Set ``adaptive_damping=False`` for fixed λ.
        4. **Step clamp** — each iteration's joint step is capped at
           ``max_step`` (rad) so a near-singular DLS spike can't fling the arm.
        """
        assert (site_id >= 0) != (body_id >= 0), "solve_ik requires exactly one of site_id / body_id"
        n = qadr.size
        eye6 = np.eye(6)
        eye_n = np.eye(n)
        rest = seed_qpos if rest_qpos is None else rest_qpos

        # Technique 1: warm-start from the seed pose.
        self.data.qpos[qadr] = seed_qpos

        w0 = None  # manipulability gauge captured at the seed (technique 3)
        err_norm = float("inf")
        for _ in range(max_iter):
            mujoco.mj_forward(self.model, self.data)

            if site_id >= 0:
                cur_pos = self.data.site_xpos[site_id].copy()
                cur_mat = self.data.site_xmat[site_id].reshape(3, 3).copy()
            else:
                cur_pos = self.data.xpos[body_id].copy()
                cur_mat = self.data.xmat[body_id].reshape(3, 3).copy()

            pos_err = target_pos - cur_pos
            rot_err = ScipyRotation.from_matrix(target_rot_mat @ cur_mat.T).as_rotvec()
            err = np.concatenate([pos_err, rot_err])

            err_norm = np.linalg.norm(err)
            if err_norm < tol:
                break

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            if site_id >= 0:
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
            else:
                mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)

            J = np.vstack([jacp[:, dofadr], jacr[:, dofadr]])
            JJt = J @ J.T

            # Technique 3: manipulability-gated damping.
            lam = damping
            if adaptive_damping:
                w = float(np.sqrt(max(np.linalg.det(JJt), 0.0)))
                if w0 is None:
                    w0 = manip_ratio * w
                if w0 > 0.0 and w < w0:
                    ratio = 1.0 - w / w0
                    lam = damping + (max_damping - damping) * ratio * ratio

            dq = J.T @ np.linalg.solve(JJt + lam * eye6, err)

            # Technique 2: least-motion null-space bias (true projector).
            if null_space_gain > 0.0:
                J_pinv = J.T @ np.linalg.solve(JJt + 1e-8 * eye6, eye6)
                null_proj = eye_n - J_pinv @ J
                dq = dq + null_space_gain * null_proj @ (rest - self.data.qpos[qadr])

            # Technique 4: per-iteration step clamp.
            if max_step is not None:
                step = float(np.linalg.norm(dq))
                if step > max_step:
                    dq = dq * (max_step / step)

            new_qpos = self.data.qpos[qadr] + dq
            self.data.qpos[qadr] = np.clip(new_qpos, jnt_range[:, 0], jnt_range[:, 1])

        return err_norm
