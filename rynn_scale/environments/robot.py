import multiprocessing as mp
import os
import struct
import time
import traceback
from abc import ABC, abstractmethod
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..constants import RotationRepresentation
from ..utils.processing import decode_image_bytes
from ..utils.robot import RobotAction, RobotState
from .base import BaseEnvironment


class BaseRobotEnvironment(BaseEnvironment):
    realtime: bool = False
    robot_type: Optional[str] = None
    image_key_map: Dict[str, str] = {}

    @property
    @abstractmethod
    def action_layout(self) -> List[Dict[str, Any]]:
        """Declare how this robot's fixed flat vector -- the array the sim
        ``_step`` consumes and the real control shm exchanges -- maps onto the
        structured standard schema (``RobotAction`` / ``RobotState``). An ordered
        list of leaf descriptors; the flat vector is those leaves' per-component
        values concatenated in list order::

            {"path": [...], "type": "Position" | "Rotation", "dim": int,
             "labels": [...], "representation": <RotationRepresentation value>?,
             "allow_relative": bool?}

        ``path`` addresses a leaf in the schema tree: a 1-element path is a
        top-level field (``["left_gripper"]``); a 2-element path descends through
        an ``Arm`` (``["left_arm", "eef_position"]``). ``representation`` (a
        ``RotationRepresentation`` value string) is required for ``Rotation``
        leaves -- the rotation encoding the flat layout expects. From this the
        base derives every structured<->flat converter below (and the GUI MOVE
        target), so a robot declares its layout once instead of hand-writing each
        conversion.

        An ordinary instance ``property``, so a robot whose layout depends on its ctor
        arguments can build one: nothing reads a layout before the env exists."""

    @property
    @abstractmethod
    def state_layout(self) -> List[Dict[str, Any]]:
        """Flat layout for ``RobotState`` -- same descriptor shape as
        :attr:`action_layout`, driving :meth:`flatten_state` /
        :meth:`unflatten_state`. Abstract separately even though one robot usually uses
        one layout for both: a robot implements it as ``return self.action_layout`` and
        has said so in one line, where inheriting it as a default let an env whose state
        and action layouts actually *differ* silently flatten its state against the
        action layout."""

    @staticmethod
    def _layout_width(layout: List[Dict[str, Any]]) -> int:
        """Columns a layout's flat vector has: its leaves' dims, concatenated."""
        return sum(int(leaf["dim"]) for leaf in layout)

    @property
    def action_dim(self) -> int:
        """Scalars per single action -- the width of the array the sim ``_step``
        consumes and the real control shm exchanges.

        Derived, not declared: ``_flatten`` concatenates one ``dim``-wide block per
        :attr:`action_layout` leaf, so this *is* the width, and a robot that declared
        it separately could disagree with its own layout. It used to, and the damage
        was silent -- the real path sizes its shm segments off this
        (:func:`bytes_shm_size` below), so a stale number truncates every command.
        """
        return self._layout_width(self.action_layout)

    @property
    def state_dim(self) -> int:
        """Same for :attr:`state_layout`, which a robot may declare differently from
        its action layout."""
        return self._layout_width(self.state_layout)

    @classmethod
    def _flatten(cls, obj: Any, layout: List[Dict[str, Any]]) -> np.ndarray:
        """Structured composite -> a ``(T, D)`` float32 flat array, concatenating
        each layout leaf's components in list order. ``Rotation`` leaves are first
        canonicalized to the representation the layout declares."""
        if not layout:
            raise ValueError(f"{cls.__name__} declares no layout for flattening.")
        parts = []
        for leaf in layout:
            # Descend the leaf's path (e.g. ``["left_arm", "eef_position"]``) to
            # the addressed ``Position``/``Rotation`` leaf.
            node = obj
            for key in leaf["path"]:
                node = getattr(node, key, None)
                if node is None:
                    raise ValueError(f"Structured value is missing a field required by the layout: {leaf['path']}")
            if leaf["type"] == "Rotation":
                node = node.convert_rotation(RotationRepresentation(leaf["representation"]))
            parts.append(node.to_flat())
        return np.concatenate(parts, axis=1).astype(np.float32)

    def flatten_action(self, action: RobotAction) -> np.ndarray:
        """Structured ``RobotAction`` -> the robot's fixed ``(T, action_dim)`` flat
        command array, per ``action_layout``. This is the structured->flat boundary
        both control paths cross before handing an action to the sim/hardware."""
        return self._flatten(action, self.action_layout)

    def flatten_state(self, state: RobotState) -> np.ndarray:
        """Structured ``RobotState`` -> the robot's fixed flat state array, per
        ``state_layout``."""
        return self._flatten(state, self.state_layout)

    def unflatten_state(self, flat: Any) -> RobotState:
        layout = self.state_layout
        if not layout:
            raise ValueError(f"{type(self).__name__} declares no layout for unflattening.")
        arr = np.asarray(flat, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None]
        tree: Dict[str, Any] = {}
        off = 0
        for leaf in layout:
            dim = leaf["dim"]
            cols = arr[:, off : off + dim].tolist()
            off += dim
            if leaf["type"] == "Rotation":
                node = {"type": "Rotation", "data": cols, "representation": leaf["representation"]}
            else:
                node = {"type": "Position", "data": cols, "allow_relative": leaf.get("allow_relative", True)}
            path = leaf["path"]
            if len(path) == 1:
                tree[path[0]] = node
            else:
                tree.setdefault(path[0], {"type": "Arm"})[path[1]] = node
        return RobotState.from_dict(tree)

    @abstractmethod
    def get_state(self) -> RobotState:
        """Return the current robot state as a structured ``RobotState``.

        Always a state, never ``None``: a robot that cannot report one yet is a
        caller error, not a value (sim raises before its first ``reset``; real
        always has the control loop's latest flat vector).

        The cheap half of the observation: it is answerable at any time, which is what
        lets a high-frequency state-only caller (the loop's snapshot publish while
        parked, a manual approach trajectory) have it without paying for a camera read.
        :meth:`get_images` is the costly other half, and :meth:`get_observation` is the
        two together -- so a caller that needs only one of them never pays for both.
        Structured, not a ``to_dict()``: every env-level interface that hands out state
        or action deals in ``RobotState`` / ``RobotAction``, and the ``to_dict()`` wire
        form is produced only where a payload actually leaves the process (the
        observation the agent turns into an inference request).
        """

    @abstractmethod
    def get_images(self) -> Dict[str, np.ndarray]:
        """This body's camera frames right now, keyed by wire name.

        The costly half of the observation, and the counterpart to :meth:`get_state`:
        a sim renders, a real one reads the image segments its control child publishes
        to -- a different job from that child's own
        :meth:`RealRobotEnvironment._read_images`, which is the camera read itself, on
        the other side of the shm. A camera that has nothing to show is left out rather
        than reported empty.
        """

    def reset(self, **kwargs) -> Dict[str, Any]:
        self._reset(**kwargs)
        return {"success": False, "error": None, "done": False}

    def step(self, action: RobotAction, *, manual: bool = False) -> Dict[str, Any]:
        flat = self.flatten_action(action)
        assert flat.shape[0] == 1, (
            f"{type(self).__name__}.step() takes one action, got {flat.shape[0]}. "
            "Iterate the chunk in the caller -- the env paces a single action per "
            "call, which is what lets the caller decide anything between two of "
            "them (see BaseEnvironment)."
        )
        done, error = self._step(flat[0])
        if error is not None:
            return {"success": False, "error": error, "done": False}
        return {"success": True, "error": None, "done": bool(done)}

    def get_observation(self) -> Dict[str, Any]:
        """The current frame as ``{state, images, robot_type}`` -- the inference
        observation, the recorded frame, and the GUI snapshot.

        The base's frame dict, with this family's keys named: ``state`` is the
        ``RobotState.to_dict()`` wire form the request, the serving leaf and the GUI
        snapshot all consume, ``images`` are the camera frames, ``robot_type`` says
        whose body they came from. One wire form for all three, so nothing here
        maintains a second shape of the same state.

        The costly half of a pass on either body is :meth:`get_images` -- a render on
        sim, a camera read plus a decode per camera on real. It goes first: on a sim the
        render is also what refreshes the low-dimensional read :meth:`get_state` answers
        from, so taking the state before it would date the two halves apart.
        """
        images = self.get_images()
        state = self.get_state()
        return {"state": state.to_dict(), "images": images, "robot_type": self.robot_type}

    @abstractmethod
    def _reset(self, **kwargs) -> None:
        """Put the body in a starting condition. Returns nothing -- the frame is
        :meth:`get_observation`'s answer, so a reset that produced one would be paying
        for a frame no caller has asked for.

        The one hook that may block for a long time: real waits until its control child
        is up and the arm is at rest. That wait is bounded here rather than by the
        caller, which has no way to interrupt it.
        """

    @abstractmethod
    def _step(self, action: np.ndarray):
        """Apply one flat ``(action_dim,)`` action and return ``(done, error)``.

        Every submitted action -- policy or manual -- is flattened to this single
        ``action_dim`` layout before it reaches here (see :meth:`step`), so each body
        has one execution path and no structured-action branch. No
        observation: what the body looks like afterwards is :meth:`get_observation`'s
        answer, and returning a frame here would mean rendering one every command.

        ``done`` is the world's task-completion signal (sim benchmarks fire it; a real
        robot has none, so it is always ``False``). ``error`` is a message when the
        action could not be applied at all.
        """


class SimRobotEnvironment(BaseRobotEnvironment):
    """A sim body: same interface as a real one, driven by a *logical* clock.

    The whole interface -- ``reset`` / ``step`` / ``get_observation`` / ``close`` and
    the hooks behind them -- is :class:`BaseRobotEnvironment`'s, and identical to what
    :class:`RealRobotEnvironment` gets. This class exists for what is genuinely
    different: time.

    A sim advances only when it is stepped, so one ``step`` *is* one tick and nothing
    happens between two of them. Nothing here has to be finished within a deadline,
    and an inference that takes longer than a command period costs wall-clock rather
    than a missed command -- which is why :attr:`realtime` is ``False`` and the agent
    does not have to hide its policy's latency (see
    :class:`~rynn_scale.agents.robot.RobotAgent`). On real, wall time passes whether a
    command arrives or not, so the same slow inference means the arm holds its last
    target instead.
    """

    # Declared rather than inherited from :class:`~.base.BaseEnvironment`'s default:
    # the clock is the reason this class exists, so it says so.
    realtime: bool = False


# ===========================================================================
# Real-robot control-process plumbing
# ===========================================================================
#
# What :class:`RealRobotEnvironment` is built from: the shared-memory segments gluing
# the main process to the forked control child, the interpolators that upsample its
# command-rate targets to the control rate, and the timer that paces it. Kept in this
# module rather than a utility one because their contracts *are* the real env's -- a
# leaf overriding :meth:`~RealRobotEnvironment._create_interpolator` reads them
# together with the hook it is implementing. The segment primitives have since grown a
# second consumer, ``api/control.py``'s GUI snapshot channel, which runs its own prefix
# and wire format over them.

# The env's own segment namespace. A constant rather than a ctor argument: the names
# carry the creating pid (see ``__init__``), so two controllers on one host cannot
# collide, and ``sweep_stale`` unlinks only a *name* -- a live run whose names another
# run's sweep removes keeps working through the mappings it and its fork already hold.
_SHM_PREFIX = "rynn_ctl"


def _seq_read(buf) -> int:
    return struct.unpack_from("<I", buf, 0)[0]


def _seq_bump(buf) -> None:
    """Advance the sequence counter; odd means a write is in progress.

    Both ends of a write are this same bump -- the first makes the counter odd so
    readers retry, the second makes it even again and publishes. The call sites say
    which half they are, since that is the only thing that differs.
    """
    struct.pack_into("<I", buf, 0, _seq_read(buf) + 1)


# ── Image shm: [seq:u32][ts:f64][size:u32][fmt:u32][w:u32][h:u32][data…] ──

_IMG_DATA_OFFSET = 28


def image_shm_size(buf_size: int) -> int:
    return _IMG_DATA_OFFSET + buf_size


def write_image_shm(shm, data: bytes, ts: float, is_jpeg: bool = False, width: int = 0, height: int = 0):
    buf = shm.buf
    _seq_bump(buf)  # odd: a write is in progress
    struct.pack_into("<dIIII", buf, 4, ts, len(data), 1 if is_jpeg else 0, width, height)
    buf[_IMG_DATA_OFFSET : _IMG_DATA_OFFSET + len(data)] = data
    _seq_bump(buf)  # even again: readers may take it


def read_image_shm(shm):
    """Returns (data_bytes, ts, is_jpeg, width, height) or (None, …)."""
    buf = shm.buf
    for _ in range(10):
        s1 = _seq_read(buf)
        if s1 == 0 or s1 & 1:
            continue
        ts, size, fmt, w, h = struct.unpack_from("<dIIII", buf, 4)
        if size == 0:
            if _seq_read(buf) == s1:
                return None, ts, False, 0, 0
            continue
        d = bytes(buf[_IMG_DATA_OFFSET : _IMG_DATA_OFFSET + size])
        if _seq_read(buf) == s1:
            return d, ts, bool(fmt), w, h
    return None, 0.0, False, 0, 0


# ── Bytes shm: [seq:u32][ts:f64][len:u32][data…] ──
#
# Carries a variable-but-bounded opaque blob (the exec<->control state/target
# payloads: flat ``float32`` vectors). Seqlock, single-writer/multi-reader.


def bytes_shm_size(max_nbytes: int) -> int:
    return 4 + 8 + 4 + max_nbytes


def write_bytes_shm(shm, data: bytes, ts: float):
    buf = shm.buf
    _seq_bump(buf)  # odd: a write is in progress
    struct.pack_into("<dI", buf, 4, ts, len(data))
    buf[16 : 16 + len(data)] = data
    _seq_bump(buf)  # even again: readers may take it


def read_bytes_shm(shm):
    """Returns (data_bytes, ts) or (None, 0.0) if nothing has been written."""
    buf = shm.buf
    for _ in range(10):
        s1 = _seq_read(buf)
        if s1 == 0 or s1 & 1:
            continue
        ts, n = struct.unpack_from("<dI", buf, 4)
        d = bytes(buf[16 : 16 + n])
        if _seq_read(buf) == s1:
            return d, ts
    return None, 0.0


def write_flat_shm(shm, flat: np.ndarray) -> None:
    write_bytes_shm(shm, np.asarray(flat, dtype=np.float32).tobytes(), time.time())


def read_flat_shm(shm) -> Optional[np.ndarray]:
    data, _ = read_bytes_shm(shm)
    return None if data is None else np.frombuffer(data, dtype=np.float32).copy()


def _detach_resource_tracker_from_shm() -> None:
    """Stop the multiprocessing resource_tracker from tracking shared_memory.

    Our segments are named and shared across processes (creator, forked control
    child, separately-launched GUI). The tracker's default behaviour -- every
    process that opens a segment unlinks it on exit (CPython bpo-38119) -- would
    let a reader or child destroy a segment the creator still uses, and its
    double-unregister spews ``KeyError`` noise. We manage lifecycle explicitly
    instead: the creator unlinks on ``close``, and ``create_shm`` clears any stale
    segment on the next run. Idempotent; applied once at import in every process
    that imports this module.
    """
    from multiprocessing import resource_tracker

    if getattr(resource_tracker, "_rynn_shm_detached", False):
        return
    # Capture the originals; only the module-level functions are patched so
    # non-shm resources (semaphores, etc.) still track normally.
    orig_register = resource_tracker.register
    orig_unregister = resource_tracker.unregister

    def register(name, rtype):
        if rtype == "shared_memory":
            return
        return orig_register(name, rtype)

    def unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return orig_unregister(name, rtype)

    resource_tracker.register = register
    resource_tracker.unregister = unregister
    resource_tracker._CLEANUP_FUNCS.pop("shared_memory", None)
    resource_tracker._rynn_shm_detached = True


_detach_resource_tracker_from_shm()


def sweep_stale(prefix: str) -> None:
    """Unlink any ``<prefix>_*`` segments left by a crashed prior run.

    Since the tracker no longer auto-cleans shared_memory, a controller that was
    hard-killed would leak its segments. The design allows one live controller
    per prefix (the manifest name is fixed per prefix), so reclaiming every
    ``<prefix>_*`` segment at startup is safe and self-healing. Linux-only
    (``/dev/shm``); a no-op elsewhere.
    """
    import glob

    shm_dir = "/dev/shm"
    if not os.path.isdir(shm_dir):
        return
    for path in glob.glob(os.path.join(shm_dir, f"{prefix}_*")):
        try:
            stale = shared_memory.SharedMemory(name=os.path.basename(path))
            stale.close()
            stale.unlink()
        except Exception:  # noqa: BLE001
            pass


def create_shm(name: str, size: int) -> shared_memory.SharedMemory:
    """Create a named segment, clearing any stale one with the same name."""
    try:
        stale = shared_memory.SharedMemory(name=name)
        stale.close()
        stale.unlink()
    except FileNotFoundError:
        pass
    return shared_memory.SharedMemory(name=name, create=True, size=size)


def unlink_shm(shm: Optional[shared_memory.SharedMemory]) -> None:
    """Close *and* unlink a segment. Idempotent / best-effort.

    Only the creating parent ever calls this. The forked control child inherits the
    handles and shares the parent's resource_tracker, so it must not unlink -- it
    simply lets them go when it exits -- and the GUI never touches the segments at
    all, so there is no close-without-unlink caller to parameterize for.
    """
    if shm is None:
        return
    try:
        shm.close()
    except Exception:  # noqa: BLE001
        pass
    try:
        shm.unlink()
    except Exception:  # noqa: BLE001
        pass


# ─── Interpolators (command-rate targets -> control-rate commands) ──────────


class BaseInterpolator(ABC):
    """Contract for the ``_create_interpolator`` hook: turn the command-rate target
    stream into a control-rate command stream.

    Three implementations live below: :class:`RuckigInterpolator` (jerk-limited, the
    default, and what an arm wants), :class:`LinearInterpolator` (a plain ramp, for a
    joint whose hardware smooths for itself or whose limits are not worth stating), and
    :class:`CompositeInterpolator` (one flat vector, a different interpolator per DOF
    subset -- an arm and a dexterous hand on one body). This stays the seam even so,
    because the hook is a leaf override point: a robot whose hardware wants something
    else again implements these methods and the control loop is none the wiser.
    """

    @abstractmethod
    def set_target(self, target: np.ndarray, dt: Optional[float] = None):
        """Called when a new target arrives from the command loop.

        ``dt`` is the measured seconds since the previous target (from the shm write
        timestamps). An interpolator that differentiates targets for a velocity
        feed-forward should use it rather than assume a fixed rate, since the agent's
        pacing is not exact; ``None`` (the first target, or a rate-agnostic
        interpolator) means fall back to the nominal input rate.
        """

    @abstractmethod
    def step(self) -> Optional[np.ndarray]:
        """Called every control tick. Returns interpolated command, or None."""

    def seed(self, state: np.ndarray) -> None:
        """Place the interpolator's internal plan at the robot's measured pose.

        Called once by the control loop before its first tick. An interpolator that
        plans *from* its own last output -- which is every jerk-limited one -- starts
        from whatever it was constructed with, so an unseeded plan would answer the
        first target with a trajectory from that construction value (zero) instead of
        from where the arm actually is. Default no-op: an interpolator that only
        reads targets has no plan to place.
        """

    def is_settled(self) -> bool:
        """Whether the last target handed to :meth:`set_target` has been reached.

        The control loop republishes this to the parent as the env's ``_settled``
        event, which is what :meth:`RealRobotEnvironment._wait_settled` -- and
        through it ``reset`` -- waits on: an interpolator lags the target stream
        by design, so sampling the first observation the instant the last target was
        *written* would catch the robot mid-motion.

        Defaults to ``True``, which is right for an interpolator that reaches its
        target within the tick and is the safe-to-inherit answer for one that does
        not only in that it makes ``reset`` return early rather than hang. Anything
        that lags must override it.
        """
        return True


class RuckigInterpolator(BaseInterpolator):
    """Jerk-limited trajectory interpolator using the ruckig library.

    Receives sparse position targets (at command_freq) via ``set_target``,
    produces smooth position commands (at control_freq) via ``step``,
    respecting velocity / acceleration / jerk limits.

    The limits are constructor arguments and nothing else can change them: they
    describe the *robot*, so they are fixed for an interpolator's life, and a
    build-then-configure API let a leaf that only did the first half run on defaults
    which silently permit almost anything. Each accepts a scalar (same limit for
    every DOF) or a length-``dof`` sequence -- the latter being how a robot with
    mixed units gets sane bounds, e.g. radians for the arm joints next to metres for
    a gripper. The defaults are permissive on purpose (``pi`` rad/s and derived
    acceleration/jerk): they smooth the 30 Hz staircase without constraining motion,
    which is the right default for a robot whose real limits are unknown here, and
    the wrong one for any robot that has them.
    """

    def __init__(
        self,
        dof: int,
        input_freq: float,
        output_freq: float,
        *,
        max_velocity: Any = None,
        max_acceleration: Any = None,
        max_jerk: Any = None,
        position_limits: Any = None,
    ):
        # Imported here, not at module scope: this module is on the import path of
        # every env, sim included, and only a real robot's control process ever
        # builds one of these.
        import ruckig

        # Resolved here rather than looked up per control tick: :meth:`is_settled`
        # gates ``reset``, so if the binding ever renames this the failure should be
        # one clear error where the interpolator is built -- not a traceback every
        # 5 ms from inside the forked control child, where nothing raises.
        self._finished_code = ruckig.Result.Finished

        self._ndof = dof
        self._input_freq = input_freq

        self._q_plan = np.zeros(dof)
        self._qd_plan = np.zeros(dof)
        self._qdd_plan = np.zeros(dof)
        self._q_plan_last = np.zeros(dof)
        self._qd_plan_last = np.zeros(dof)
        self._qdd_plan_last = np.zeros(dof)

        self._q_target = np.zeros(dof)
        self._qd_target = np.zeros(dof)
        self._qdd_target = np.zeros(dof)
        self._q_target_last = np.zeros(dof)
        self._qd_target_last = np.zeros(dof)

        self._otg = ruckig.Ruckig(dof, 1.0 / output_freq)
        self._inp = ruckig.InputParameter(dof)
        self._out = ruckig.OutputParameter(dof)

        # Acceleration and jerk default to the ratios of the velocity limit they
        # usually stand in for, so declaring the velocity alone is the common case --
        # and it stays the common case whether the velocity was given or defaulted.
        vel = self._per_dof(np.pi if max_velocity is None else max_velocity, dof, "max_velocity")
        acc = self._per_dof(vel * 10 if max_acceleration is None else max_acceleration, dof, "max_acceleration")
        lower, upper = (-np.pi, np.pi) if position_limits is None else position_limits
        self._pos_lower = self._per_dof(lower, dof, "position_limits[0]")
        self._pos_upper = self._per_dof(upper, dof, "position_limits[1]")
        self._vel_upper, self._vel_lower = vel, -vel
        self._acc_upper, self._acc_lower = acc, -acc
        jerk = self._per_dof(vel * 1000 if max_jerk is None else max_jerk, dof, "max_jerk")
        # Only the upper bounds reach ruckig, which treats its limits as symmetric;
        # the lower ones are ours, for clipping the incoming target in
        # :meth:`_clip_targets`. The jerk limit is not kept at all -- nothing clips
        # against it, so ruckig is its only reader.
        self._inp.max_velocity = self._vel_upper
        self._inp.max_acceleration = self._acc_upper
        self._inp.max_jerk = jerk

        self._has_target = False
        # Whether the OTG has run the current target's plan to completion; see
        # :meth:`is_settled`. Starts ``True``: no target, nothing to reach.
        self._finished = True

    @staticmethod
    def _per_dof(value: Any, dof: int, what: str) -> np.ndarray:
        """A ``(dof,)`` limit array from a scalar (broadcast) or a sequence.

        The length is *checked* rather than broadcast: handing a one-arm limit vector
        to a two-arm robot is the mistake worth catching here, and numpy would
        otherwise either broadcast it into a silently wrong bound or raise somewhere
        far downstream inside the control child.
        """
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(dof, float(arr))
        assert arr.shape == (dof,), (
            f"{what} has shape {arr.shape}, expected a scalar or ({dof},) to match the robot's action_dim."
        )
        return arr.copy()

    def seed(self, state: np.ndarray) -> None:
        # Park the plan on the measured pose, at rest, with itself as the target.
        # Without this the OTG's ``current_position`` is the constructor's zeros, so
        # the first real target would be planned from the origin and the arm would
        # lurch there -- a jerk-limited plan makes that a smooth motion to the wrong
        # place, which is worse than an obvious jump. The velocities and accelerations
        # start at zero because a seeded arm is a *stopped* one, and seeding a nonzero
        # one would have the OTG plan its first trajectory as if it were already moving.
        self._q_plan = np.asarray(state, dtype=np.float64)
        self._q_target = np.asarray(state, dtype=np.float64)
        self._qd_plan = np.zeros(self._ndof)
        self._qd_target = np.zeros(self._ndof)
        self._qdd_plan = np.zeros(self._ndof)
        self._qdd_target = np.zeros(self._ndof)
        self._save_state()

    def set_target(self, target: np.ndarray, dt: Optional[float] = None):
        # The velocity/acceleration feed-forward differentiates successive targets, so
        # it needs the rate they actually arrived at. The control loop measures that
        # from the shm write timestamps and passes it as ``dt``; differentiating
        # against ``1/dt`` makes the feed-forward correct regardless of the agent's
        # pacing jitter (asyncio.sleep will not land exactly on ``1/command_freq``).
        # ``dt`` is ``None`` only for the very first target -- no previous timestamp to
        # difference against -- where we fall back to the nominal ``_input_freq``; the
        # first target's feed-forward barely matters since the plan starts at rest.
        freq = self._input_freq if not dt or dt <= 0.0 else 1.0 / dt
        self._q_target = np.asarray(target, dtype=np.float64)
        self._qd_target = (self._q_target - self._q_target_last) * freq
        self._qdd_target = (self._qd_target - self._qd_target_last) * freq
        self._has_target = True
        self._finished = False

    def step(self) -> Optional[np.ndarray]:
        if not self._has_target:
            return None
        self._clip_targets()
        self._step_ruckig()
        self._save_state()
        return np.array(self._q_plan)

    def is_settled(self) -> bool:
        # Ruckig's own verdict: ``update`` reports ``Finished`` once the plan has
        # run out, which for a jerk-limited OTG is the only honest answer -- how
        # long a target takes to reach depends on the limits and on how far the
        # plan was from it, so no tick count predicts it.
        return self._finished

    def _save_state(self):
        self._q_target_last = self._q_target.copy()
        self._qd_target_last = self._qd_target.copy()
        self._q_plan_last = self._q_plan.copy()
        self._qd_plan_last = self._qd_plan.copy()
        self._qdd_plan_last = self._qdd_plan.copy()

    def _clip_targets(self):
        self._q_target = np.clip(self._q_target, self._pos_lower, self._pos_upper)
        self._qd_target = np.clip(self._qd_target, self._vel_lower, self._vel_upper)
        self._qdd_target = np.clip(self._qdd_target, self._acc_lower, self._acc_upper)

    def _step_ruckig(self):
        self._inp.current_position = self._q_plan_last
        self._inp.current_velocity = self._qd_plan_last
        self._inp.current_acceleration = self._qdd_plan_last
        self._inp.target_position = self._q_target
        self._inp.target_velocity = self._qd_target
        self._inp.target_acceleration = self._qdd_target

        result = self._otg.update(self._inp, self._out)
        if result >= 0:  # Working or Finished
            self._finished = result == self._finished_code
            self._q_plan = np.array(self._out.new_position)
            self._qd_plan = np.array(self._out.new_velocity) * 0.99
            self._qdd_plan = np.array(self._out.new_acceleration) * 0.99
        else:
            # A failed update freezes the plan, so nothing is going to move any
            # more: report settled rather than making ``reset`` wait out a target
            # that can never be reached.
            self._finished = True
            self._q_plan = self._q_plan_last.copy()
            self._qd_plan = self._qd_plan_last.copy()
            self._qdd_plan = self._qdd_plan_last.copy()


class LinearInterpolator(BaseInterpolator):
    """Straight ramp from the last target to the new one, over one input period.

    What to reach for when a jerk-limited plan is the wrong tool rather than a luxury:
    a DOF whose own hardware already filters what it is handed (Astribot's ``filter``
    control mode), or one whose real limits are not stated anywhere and whose travel
    per command is small enough that a ramp is the honest answer (a dexterous hand's
    finger joints). It needs no limits, so unlike :class:`RuckigInterpolator` there is
    nothing here to get wrong by omission.

    The ramp spans ``output_freq / input_freq`` ticks -- the nominal number between two
    targets -- and then holds, so a target that arrives late is held at rather than
    extrapolated past.
    """

    def __init__(self, dof: int, input_freq: float, output_freq: float):
        # ``dof`` is taken and not stored: nothing here is sized per-DOF (numpy shapes
        # the ramp off the targets), and the signature matches
        # :class:`RuckigInterpolator`'s so the two are interchangeable in a
        # :class:`CompositeInterpolator` group.
        # Ticks per ramp. Floored at 1 so a control rate at (or below) the command
        # rate degenerates to a passthrough instead of dividing by zero.
        self._ratio = max(1, round(output_freq / input_freq))
        # Ramp origin: the seeded pose, then each target the previous ramp aimed at.
        self._from: Optional[np.ndarray] = None
        self._to: Optional[np.ndarray] = None
        self._tick = 0

    def seed(self, state: np.ndarray) -> None:
        # Only the origin, deliberately: ``_to`` stays ``None`` so :meth:`step` keeps
        # returning ``None`` until a real target arrives, and the control loop actuates
        # nothing before the first command. Without this the first ramp would have no
        # origin but the target itself, i.e. an instant jump to it.
        self._from = np.asarray(state, dtype=np.float64)

    def set_target(self, target: np.ndarray, dt: Optional[float] = None):
        # ``dt`` is ignored: a ramp has no feed-forward to differentiate, so the rate
        # targets actually arrived at does not change what it outputs.
        target = np.asarray(target, dtype=np.float64)
        if self._to is not None:
            self._from = self._to
        elif self._from is None:
            self._from = target
        self._to = target
        self._tick = 0

    def step(self) -> Optional[np.ndarray]:
        if self._to is None:
            return None
        self._tick += 1
        alpha = min(1.0, self._tick / self._ratio)
        return self._from + alpha * (self._to - self._from)

    def is_settled(self) -> bool:
        return self._to is None or self._tick >= self._ratio


class CompositeInterpolator(BaseInterpolator):
    """One flat vector, a different interpolator per DOF subset.

    For a body whose flat layout spans hardware that wants different smoothing: an arm
    with real jerk limits next to a dexterous hand that has none stated
    (:class:`~rynn_scale.environments.marvin_wuji.MarvinWuji` is Ruckig over its 14 arm
    joints composed with a ramp over its 40 finger joints). The control loop sees one
    interpolator over ``action_dim`` columns and knows nothing of the split.

    ``groups`` is ``[(indices, interpolator), ...]``, each interpolator sized for its
    own subset -- so a leaf states the split once, as indices into its own layout.
    """

    def __init__(self, dof: int, groups: Sequence[Tuple[Sequence[int], BaseInterpolator]]):
        self._dof = dof
        self._groups = [(np.asarray(idx, dtype=np.intp), interp) for idx, interp in groups]
        # A partition, not a cover: a column no group owns would be commanded as
        # whatever :meth:`step` left in the output buffer, and one that two groups own
        # would be written twice, with the later group silently winning.
        owned = sorted(int(i) for idx, _ in self._groups for i in idx)
        assert owned == list(range(dof)), (
            f"the groups must partition all {dof} columns of the flat vector exactly, "
            f"got {len(owned)} indices covering {sorted(set(owned))[:8]}... -- every "
            "column of the layout belongs to exactly one interpolator."
        )

    def seed(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float64)
        for idx, interp in self._groups:
            interp.seed(state[idx])

    def set_target(self, target: np.ndarray, dt: Optional[float] = None):
        target = np.asarray(target, dtype=np.float64)
        for idx, interp in self._groups:
            interp.set_target(target[idx], dt=dt)

    def step(self) -> Optional[np.ndarray]:
        out = np.zeros(self._dof, dtype=np.float64)
        for idx, interp in self._groups:
            sub = interp.step()
            # All or nothing: a group that has nothing to say leaves its columns at the
            # buffer's zeros, and publishing that would command those joints *to zero*
            # rather than leave them alone. Every group is handed every target, so they
            # start producing on the same tick and this only ever withholds the ticks
            # before the first target -- which is exactly what a single interpolator
            # withholds too.
            if sub is None:
                return None
            out[idx] = sub
        return out

    def is_settled(self) -> bool:
        return all(interp.is_settled() for _, interp in self._groups)


# ─── Precise periodic timer ─────────────────────────────────────────────────


class PrecisePeriodicTimer:
    """Periodic timer using Linux timerfd for microsecond-precision scheduling.

    Falls back to monotonic-clock absolute-time sleep if timerfd is unavailable.
    """

    def __init__(self, freq_hz: float):
        self._dt = 1.0 / freq_hz
        self._fd = None
        self._tick_start = time.monotonic()
        try:
            self._init_timerfd(freq_hz)
        except (OSError, AttributeError, TypeError):
            self._fd = None

    def _init_timerfd(self, freq_hz: float):
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        CLOCK_MONOTONIC = 1
        TFD_CLOEXEC = 0o2000000

        fd = libc.timerfd_create(CLOCK_MONOTONIC, TFD_CLOEXEC)
        if fd < 0:
            raise OSError(ctypes.get_errno(), "timerfd_create failed")

        class _timespec(ctypes.Structure):
            _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

        class _itimerspec(ctypes.Structure):
            _fields_ = [("it_interval", _timespec), ("it_value", _timespec)]

        interval_ns = int(1e9 / freq_hz)
        sec = interval_ns // 1_000_000_000
        nsec = interval_ns % 1_000_000_000
        new_value = _itimerspec(
            it_interval=_timespec(sec, nsec),
            it_value=_timespec(sec, nsec),
        )
        ret = libc.timerfd_settime(fd, 0, ctypes.byref(new_value), None)
        if ret < 0:
            os.close(fd)
            raise OSError(ctypes.get_errno(), "timerfd_settime failed")
        self._fd = fd

    def wait(self):
        """Block until the next timer tick."""
        if self._fd is not None:
            os.read(self._fd, 8)
        else:
            elapsed = time.monotonic() - self._tick_start
            remaining = self._dt - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def tick(self):
        """Mark the start of a new cycle (for software fallback)."""
        self._tick_start = time.monotonic()

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


# ===========================================================================
# Real-robot deploy environment
# ===========================================================================
#
# Wall-clock sibling of ``SimRobotEnvironment``: the high-frequency control loop
# runs in a separate process, glued to the main process by the shared-memory
# segments above. Same ``reset``/``step`` contract, so the agent drives real and sim
# identically. Hardware IO happens only in the control process.


class RealRobotEnvironment(BaseRobotEnvironment):
    robot_type: str
    image_buffer_size: int = 8 * 1024 * 1024

    # A wall clock: the agent paces ``step`` at ``command_freq``, so its Real-Time
    # Chunking has latency to hide here (unlike sim).
    realtime: bool = True

    @property
    def fps(self) -> int:
        """A real robot's frame rate *is* the rate it is commanded at.

        Rollout videos and the agent's manual-approach trajectories both ask the env
        for ``fps``, so answering with anything other than the rate ``step`` actually
        issues commands at would play every video at the wrong speed.
        """
        return self.command_freq

    def __init__(
        self,
        command_freq: int = 30,
        control_freq: Optional[int] = None,
        data_reader: Any = None,
    ):
        # The rate ``step`` issues commands at, and therefore this env's ``fps``
        # (see above).
        self.command_freq = command_freq
        # The rate the control process actuates at, i.e. the interpolator's output
        # rate. Resolved once, here, rather than kept as the requested value: it
        # defaults to ``command_freq`` and is floored at it, so a leaf building its own
        # interpolator reads the effective number instead of ``None``.
        self.control_freq = max(control_freq, command_freq) if control_freq is not None else command_freq
        # REPLAY trajectory source: a live, absolute VLA dataset, or None (see
        # ``_data_reader``).
        self._data_reader = data_reader
        self._ready = False
        self._closed = False

        self._mp = mp.get_context("fork")
        self._stop_event = self._mp.Event()
        # Mirrors ``interpolator.is_settled()`` from the control child, so ``reset``
        # can wait for the arm to come to rest. Starts set: no target has been
        # written, so there is nothing to settle onto.
        self._settled = self._mp.Event()
        self._settled.set()
        self._parent_pid = os.getpid()

        # ---- shared memory: the parent <-> forked control child IPC, and nothing else.
        # The child inherits the handles by fork rather than attaching by name, so no
        # third process can reach these segments; they are named only so ``sweep_stale``
        # can reclaim ones a hard-killed run leaked. The state/target segments carry raw
        # flat ``float32`` vectors, not serialized structured payloads: both ends produce
        # and consume flat vectors, so structuring is deferred to the ``get_state``
        # boundary. A little headroom rides on the length-prefixed ``bytes`` segment, and
        # the two segments share one width so that a robot whose state layout is wider
        # than its action layout still fits.
        #
        # Each is a **single-writer** seqlock (writers bump a sequence number, readers
        # retry on an odd one -- see :func:`write_bytes_shm` above), so a second
        # concurrent writer would hand the control process torn targets. Nothing here
        # enforces that: every call into this env comes from the agent's ``control``
        # concurrency group, which is one thread, and that same thread is the one that
        # calls ``close`` from its own ``finally`` -- so the image segments' only reader
        # is never in flight when they are unlinked either.
        sweep_stale(_SHM_PREFIX)
        prefix = f"{_SHM_PREFIX}_{self._parent_pid}"
        flat_nbytes = max(self.action_dim, self.state_dim) * 4 + 256
        self._shms: Dict[str, Any] = {
            "state": create_shm(f"{prefix}_state", bytes_shm_size(flat_nbytes)),
            "target": create_shm(f"{prefix}_target", bytes_shm_size(flat_nbytes)),
        }
        for cam in self.image_key_map:
            self._shms[f"image_{cam}"] = create_shm(f"{prefix}_image_{cam}", image_shm_size(self.image_buffer_size))
        self._state_shm = self._shms["state"]
        self._target_shm = self._shms["target"]
        self._image_shms = {cam: self._shms[f"image_{cam}"] for cam in self.image_key_map}

        # Fork the control process, as the last thing this constructor does: it runs
        # for the env's whole life and calls straight back into the hooks below
        # (``_initialize`` / ``_read_state`` / ``_create_interpolator``), which read
        # attributes set above. A leaf whose control-side hooks need its *own*
        # constructor state has the same constraint one level up: set that up before
        # calling ``super().__init__``, which is what forks.
        self._control_proc = self._mp.Process(target=self._control_process_entry, daemon=True)
        self._control_proc.start()

    # ---- env-specific IO; the control process calls these ----

    def _initialize(self):
        """Set up cameras / SDK. Called once in the control process (optional)."""

    @abstractmethod
    def _read_state(self) -> np.ndarray:
        """Return robot state as float32 ndarray (action_dim,)."""

    @abstractmethod
    def _read_images(self) -> Dict[str, np.ndarray]:
        """Return {internal cam name -> HxWx3 uint8} for the current step."""

    @abstractmethod
    def send_action(self, position: np.ndarray):
        """Send a flat position vector (action_dim,) to the hardware."""

    def _create_interpolator(self) -> BaseInterpolator:
        return RuckigInterpolator(self.action_dim, self.command_freq, self.control_freq)

    def _shutdown(self):
        """Release resources. Called when the control process exits."""

    @staticmethod
    def _decode_image(data: bytes, is_jpeg: bool, width: int, height: int) -> Optional[np.ndarray]:
        if is_jpeg:
            try:
                return decode_image_bytes(data)
            except Exception:
                return None
        if width > 0 and height > 0:
            return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3).copy()
        return None

    def get_state(self) -> RobotState:
        flat = read_flat_shm(self._state_shm)
        if flat is None:
            flat = np.zeros(self.state_dim, dtype=np.float32)
        return self.unflatten_state(flat)

    def get_images(self) -> Dict[str, np.ndarray]:
        """The latest frame per camera off the control child's image segments: a memcpy
        per segment plus a decode per camera. A camera that has published nothing yet is
        left out rather than reported empty.

        The reading end of :meth:`_read_images`, which runs in the control child and is
        what put the frames there.
        """
        images: Dict[str, np.ndarray] = {}
        for cam, wire in self.image_key_map.items():
            d, _, is_jpeg, w, h = read_image_shm(self._image_shms[cam])
            if d is None:
                continue
            img = self._decode_image(d, is_jpeg, w, h)
            if img is not None:
                images[wire] = img
        return images

    def _wait_ready(self, timeout: float = 30.0) -> bool:
        if self._ready:
            return True
        deadline = time.monotonic() + timeout
        while not self._stop_event.is_set():
            if read_flat_shm(self._state_shm) is not None:
                self._ready = True
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return False

    def _wait_settled(self, timeout: float = 5.0) -> bool:
        # Blocking, so the deadline and ``_stop_event`` are the only two ways out --
        # there is no cancellation to fall back on (same for :meth:`_wait_ready`).
        deadline = time.monotonic() + timeout
        while not self._settled.is_set():
            if self._stop_event.is_set() or time.monotonic() >= deadline:
                return False
            time.sleep(0.02)
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        self._control_proc.join(timeout=5)
        if self._control_proc.is_alive():
            self._control_proc.terminate()
            self._control_proc.join(timeout=2)
        for shm in self._shms.values():
            unlink_shm(shm)

    def _should_stop(self) -> bool:
        if self._stop_event.is_set():
            return True
        if os.getppid() != self._parent_pid:  # parent died -> orphan, exit
            return True
        return False

    def _control_process_entry(self):
        self._parent_pid = os.getppid()
        try:
            self._control_loop()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        finally:
            self._shutdown()

    def _control_loop(self):
        self._initialize()
        interpolator = self._create_interpolator()
        # Seed the plan on the real pose before anything can be commanded: a
        # jerk-limited interpolator plans from its own state, and an unseeded one
        # would answer the first target by travelling there from the origin.
        interpolator.seed(np.asarray(self._read_state(), dtype=np.float32))

        timer = PrecisePeriodicTimer(self.control_freq)
        last_target_ts = 0.0
        img_interval = 1.0 / self.command_freq
        last_img_read = 0.0
        # Local mirror of the ``_settled`` event, so the (locking) cross-process
        # flag is only touched on a transition rather than every control tick.
        was_settled = True

        while not self._should_stop():
            timer.tick()
            t0 = time.time()

            # Actuate: interpolate the latest target up to the control rate.
            state = np.asarray(self._read_state(), dtype=np.float32)
            data, ts = read_bytes_shm(self._target_shm)
            if data is not None and ts > last_target_ts:
                # Measure the actual spacing between targets from the shm write
                # timestamps and hand it to the interpolator, so its feed-forward is
                # differentiated against the rate they *really* arrived at rather than
                # a nominal constant -- the agent paces with asyncio.sleep now, which
                # will not hit exact ``1/command_freq``. ``None`` on the first target
                # (no previous timestamp) falls back to the nominal input_freq.
                dt = ts - last_target_ts if last_target_ts > 0.0 else None
                last_target_ts = ts
                interpolator.set_target(np.frombuffer(data, dtype=np.float32).copy(), dt=dt)
            cmd = interpolator.step()
            if cmd is not None:
                self.send_action(cmd)

            # Republish "the arm has reached the last target" for ``reset``
            # (:meth:`_wait_settled`). Read *after* stepping, so a target that arrived
            # this tick already counts as unsettled.
            settled = interpolator.is_settled()
            if settled != was_settled:
                was_settled = settled
                (self._settled.set if settled else self._settled.clear)()

            # Publish state + (throttled) images for the request / GUI.
            write_flat_shm(self._state_shm, state)
            if t0 - last_img_read >= img_interval:
                last_img_read = t0
                for cam, img in self._read_images().items():
                    write_image_shm(
                        self._image_shms[cam],
                        img.tobytes(),
                        t0,
                        is_jpeg=False,
                        width=img.shape[1],
                        height=img.shape[0],
                    )

            timer.wait()

        timer.close()

    def _reset(self, **kwargs) -> None:
        """Wait until the arm is up and at rest. Nothing is commanded: a real body is
        already wherever it is, and the world it works in is not ours to place.

        The two waits block the caller, each with its own timeout: a control child that
        never publishes fails the reset after 30 s rather than hanging it, and an arm that
        never settles costs 5 s and is accepted as-is.
        """
        if not self._wait_ready():
            raise RuntimeError(
                f"{type(self).__name__}.reset(): the control process published no "
                "state. Its _initialize (SDK / cameras) most likely failed -- the "
                "traceback is printed by the control child, not raised here."
            )
        self._wait_settled()

    def _step(self, action: np.ndarray):
        """Publish the target and return at once -- the control child actuates it.

        Always ``(False, None)``: the write cannot fail here (once closed, or once the
        control child is stopping, there is nothing left to actuate the target and it is
        dropped), and a real robot has no task-completion signal, so nothing but the
        caller ends an episode.
        """
        if not (self._closed or self._stop_event.is_set()):
            write_flat_shm(self._target_shm, action)
        return False, None
