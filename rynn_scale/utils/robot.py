import warnings
from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Dict, Optional, Union

import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyRotation
from scipy.spatial.transform import Slerp

from ..constants import RotationRepresentation

# ---------------------------------------------------------------------------
# Rotation helpers (unchanged logic, just reorganized)
# ---------------------------------------------------------------------------


def _rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    x_raw, y_raw = d6[:, 0:6:2], d6[:, 1:7:2]

    def normalize(v, eps=1e-9):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)

    x = normalize(x_raw)
    z = normalize(np.cross(x, y_raw, axis=-1))
    y = np.cross(z, x, axis=-1)
    return np.stack([x, y, z], axis=-1).reshape(-1, 3, 3)


def _matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return matrix[:, :, :2].reshape(-1, 6)


def _to_scipy_rotation(value: torch.Tensor, representation: RotationRepresentation) -> ScipyRotation:
    v = value.numpy()
    converters = {
        RotationRepresentation.EULER_XYZ: lambda: ScipyRotation.from_euler("xyz", v),
        RotationRepresentation.EULER_ZYX: lambda: ScipyRotation.from_euler("zyx", v),
        RotationRepresentation.QUAT_XYZW: lambda: ScipyRotation.from_quat(v, scalar_first=False),
        RotationRepresentation.QUAT_WXYZ: lambda: ScipyRotation.from_quat(v, scalar_first=True),
        RotationRepresentation.ROT_6D: lambda: ScipyRotation.from_matrix(_rotation_6d_to_matrix(v)),
        RotationRepresentation.ROT_VEC: lambda: ScipyRotation.from_rotvec(v),
    }
    if representation not in converters:
        raise ValueError(f"Conversion from {representation} is not implemented")
    return converters[representation]()


def _convert_rotation(
    value: Optional[Union[torch.Tensor, ScipyRotation]],
    src_repr: Optional[RotationRepresentation],
    tgt_repr: RotationRepresentation,
):
    if value is None:
        return value
    if isinstance(value, torch.Tensor):
        if src_repr == tgt_repr:
            return value
        rotation = _to_scipy_rotation(value, src_repr)
    elif isinstance(value, ScipyRotation):
        rotation = value
    else:
        raise ValueError(f"Unsupported value type: {type(value)}")

    exporters = {
        RotationRepresentation.EULER_XYZ: lambda r: r.as_euler("xyz"),
        RotationRepresentation.EULER_ZYX: lambda r: r.as_euler("zyx"),
        RotationRepresentation.QUAT_XYZW: lambda r: r.as_quat(scalar_first=False),
        RotationRepresentation.QUAT_WXYZ: lambda r: r.as_quat(scalar_first=True),
        RotationRepresentation.ROT_6D: lambda r: _matrix_to_rotation_6d(r.as_matrix()),
        RotationRepresentation.ROT_VEC: lambda r: r.as_rotvec(),
    }
    if tgt_repr not in exporters:
        raise ValueError(f"Conversion to {tgt_repr} is not implemented")
    return torch.from_numpy(np.ascontiguousarray(exporters[tgt_repr](rotation)))


def _compose_rotations(r1_data, r1_repr, r2_data, r2_repr, tgt_repr, inverse_r1=False):
    """Compose two rotations: R1 @ R2 (or R1^T @ R2 if inverse_r1)."""
    R1 = _to_scipy_rotation(r1_data, r1_repr).as_matrix()
    R2 = _to_scipy_rotation(r2_data, r2_repr).as_matrix()
    if inverse_r1:
        R1 = R1.transpose(0, 2, 1)
    result = ScipyRotation.from_matrix(R1 @ R2)
    out = _convert_rotation(result, src_repr=None, tgt_repr=tgt_repr)
    return torch.from_numpy(_unwrap_along_chunk(out.numpy().copy(), tgt_repr))


# ---------------------------------------------------------------------------
# Unwrap helpers
# ---------------------------------------------------------------------------


def _quat_unwrap_(quats: np.ndarray) -> np.ndarray:
    for i in range(1, len(quats)):
        if float(np.dot(quats[i], quats[i - 1])) < 0.0:
            quats[i] = -quats[i]
    return quats


def _rotvec_unwrap_(rvs: np.ndarray) -> np.ndarray:
    eps, two_pi = 1e-8, 2.0 * np.pi
    for i in range(1, len(rvs)):
        r = rvs[i]
        n = float(np.linalg.norm(r))
        if n < eps:
            continue
        r_alt = -(two_pi - n) * r / n
        if np.linalg.norm(r - rvs[i - 1]) > np.linalg.norm(r_alt - rvs[i - 1]):
            rvs[i] = r_alt
    return rvs


def _unwrap_along_chunk(value: np.ndarray, tgt_repr: RotationRepresentation) -> np.ndarray:
    if len(value) < 2:
        return value
    if tgt_repr in (RotationRepresentation.QUAT_XYZW, RotationRepresentation.QUAT_WXYZ):
        return _quat_unwrap_(value)
    if tgt_repr == RotationRepresentation.ROT_VEC:
        return _rotvec_unwrap_(value)
    if tgt_repr in (RotationRepresentation.EULER_XYZ, RotationRepresentation.EULER_ZYX):
        return np.unwrap(value, axis=0)
    return value


# ---------------------------------------------------------------------------
# Resample helpers
# ---------------------------------------------------------------------------


def _resample_positions(n_src, src_fps, tgt_fps, n_target, src_offset):
    ratio = float(src_fps) / float(tgt_fps)
    positions = src_offset + np.arange(n_target, dtype=np.float64) * ratio
    return np.clip(positions, 0.0, float(n_src - 1))


def _interpolate_positions(n_src, chunk_size):
    """Even source positions for re-timing ``n_src`` keyframes to ``chunk_size``
    frames: ``linspace(0, n_src - 1, chunk_size)``.

    Both endpoints are hit exactly, so the first and last keyframes survive
    verbatim and only the frames between them are interpolated. A single source
    frame degenerates to holding it (all positions 0)."""
    assert n_src > 0, "Cannot interpolate an empty chunk"
    assert chunk_size > 0, f"chunk_size must be positive, got {chunk_size}"
    return np.linspace(0.0, float(n_src - 1), chunk_size, dtype=np.float64)


def _resample_tensor(value, positions, mode):
    n_src = value.size(0)
    if mode == "nearest":
        idx = np.clip(np.round(positions).astype(np.int64), 0, n_src - 1)
        return value[torch.from_numpy(idx)]
    if mode == "linear":
        floor_idx = np.clip(np.floor(positions).astype(np.int64), 0, n_src - 1)
        ceil_idx = np.minimum(floor_idx + 1, n_src - 1)
        frac = positions - floor_idx
        lo = value[torch.from_numpy(floor_idx)]
        hi = value[torch.from_numpy(ceil_idx)]
        w = torch.from_numpy(frac).to(lo.dtype).reshape(-1, *([1] * (lo.ndim - 1)))
        return lo * (1 - w) + hi * w
    raise ValueError(f"Unsupported resample mode: {mode}")


def _resample_rotation(value, representation, positions):
    n_src = value.size(0)
    if n_src == 1:
        return value[torch.zeros(len(positions), dtype=torch.long)]
    src_rot = _to_scipy_rotation(value, representation)
    slerp = Slerp(np.arange(n_src, dtype=np.float64), src_rot)
    return _convert_rotation(slerp(positions), src_repr=None, tgt_repr=representation)


# ---------------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------------


def _norm_coeffs(stats, norm_type, dtype, device):
    """Return (offset, scale) such that normalize = (x - offset) / scale."""

    def t(k):
        return torch.tensor(stats[k], dtype=dtype, device=device)

    if norm_type == "mean_std":
        return t("mean"), t("std") + 1e-8
    if norm_type == "min_max":
        vmin, vmax = t("min"), t("max")
        return vmin, (vmax - vmin + 1e-8) / 2.0
    if norm_type == "q01_q99":
        q01, q99 = t("q01"), t("q99")
        return q01, (q99 - q01 + 1e-8) / 2.0
    raise ValueError(f"Unsupported normalization type: {norm_type}")


# ---------------------------------------------------------------------------
# Type registry (auto-populated by __init_subclass__)
# ---------------------------------------------------------------------------

_TYPE_REGISTRY: Dict[str, type] = {}


# ---------------------------------------------------------------------------
# _TensorChunk — leaf node holding a single (T, D) tensor
# ---------------------------------------------------------------------------


class _TensorChunk:
    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        _TYPE_REGISTRY[cls.__name__] = cls

    def __post_init__(self):
        assert isinstance(self.data, torch.Tensor)
        if self.data.ndim == 1:
            self.data = self.data.unsqueeze(0)
        assert self.data.ndim == 2, f"{type(self).__name__}.data must be 2D (T, D), got shape {tuple(self.data.shape)}"

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, index):
        if isinstance(index, int):
            n = len(self)
            if index < -n or index >= n:
                raise IndexError(f"Index {index} out of range for length {n}")
            if index < 0:
                index += n
            index = slice(index, index + 1)
        elif not isinstance(index, slice):
            raise TypeError(f"Unsupported index type: {type(index)}")
        return replace(self, data=self.data[index])

    def __radd__(self, other):
        return self.__add__(other)

    def pad_to(self, chunk_size: int):
        cur = len(self)
        assert cur > 0, f"Cannot pad an empty {type(self).__name__}"
        assert chunk_size >= cur
        if chunk_size == cur:
            return self
        pad = self.data[-1:].expand(chunk_size - cur, *self.data.shape[1:])
        return replace(self, data=torch.cat([self.data, pad], dim=0))

    def unpack(self):
        return [self[i] for i in range(len(self))]

    @classmethod
    def cat(cls, items):
        """Concatenate same-type chunks along the chunk (time) dim into one.

        The inverse of :meth:`unpack`; non-``data`` fields (e.g. rotation
        representation) are taken from the first item."""
        return replace(items[0], data=torch.cat([it.data for it in items], dim=0))

    def normalize(self, stats=None, norm_type: str = "mean_std"):
        return replace(self)

    def denormalize(self, stats=None, norm_type: str = "mean_std"):
        return replace(self)

    def convert_rotation(self, target_representation: RotationRepresentation):
        return self

    def resample(self, src_fps, tgt_fps, n_target=None, src_offset=0.0, mode="linear"):
        n_src = len(self)
        assert n_src > 0
        if n_target is None:
            n_target = max(1, int(round(n_src * tgt_fps / src_fps)))
        if abs(src_fps - tgt_fps) < 1e-6 and abs(src_offset) < 1e-9 and n_target == n_src:
            return self
        positions = _resample_positions(n_src, src_fps, tgt_fps, n_target, src_offset)
        return replace(self, data=_resample_tensor(self.data, positions, mode))

    def resample_at(self, positions, mode="linear"):
        """Resample at explicit fractional source positions (0..len-1).

        Unlike :meth:`resample` (which derives positions from an fps ratio), this
        takes the positions directly, so a caller can interpolate between
        keyframes at weights of its own choosing -- e.g. the env's approach chunk,
        which ``cat``s ``[current state, target]`` into two keyframes and samples
        this at even weights in ``[0, 1]``. ``Rotation`` overrides this to
        SLERP."""
        positions = np.asarray(positions, dtype=np.float64)
        return replace(self, data=_resample_tensor(self.data, positions, mode))

    def interpolate(self, chunk_size: int, mode: str = "linear"):
        """Re-time this chunk to exactly ``chunk_size`` frames, interpolating
        evenly between its frames as keyframes.

        The frame-count counterpart of :meth:`resample`: same interpolation, but
        the caller states the length it wants instead of an fps pair, for chunks
        whose index is progress rather than time (the env's two-keyframe approach
        chunk). Positions are ``linspace(0, len - 1, chunk_size)``, so the first
        and last frames come through verbatim; ``chunk_size == len(self)`` is a
        no-op, a larger one interpolates and a smaller one drops frames.
        Dispatches per leaf like :meth:`resample_at` (positions lerp, rotations
        SLERP), so it is unlike :meth:`pad_to`, which reaches ``chunk_size`` by
        repeating the last frame instead of stretching the trajectory."""
        if chunk_size == len(self):
            return self
        return self.resample_at(_interpolate_positions(len(self), chunk_size), mode)

    def to_dict(self) -> dict:
        out = {"type": type(self).__name__}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, torch.Tensor):
                out[f.name] = v.tolist()
            elif isinstance(v, Enum):
                out[f.name] = v.value
            elif v is not None:
                out[f.name] = v
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "_TensorChunk":
        kw = {}
        for f in fields(cls):
            if f.name not in d:
                continue
            v = d[f.name]
            if f.type is torch.Tensor or f.type == torch.Tensor:
                kw[f.name] = torch.tensor(v)
            elif f.type is RotationRepresentation or f.type == RotationRepresentation:
                kw[f.name] = RotationRepresentation(v)
            else:
                kw[f.name] = v
        return cls(**kw)

    # ---- flat conversion ----

    def to_flat(self) -> np.ndarray:
        """This leaf's tensor as a ``(T, D)`` float32 numpy array.

        The torch->numpy primitive the env's layout-driven flatten
        (:meth:`~rynn_scale.environments.robot.BaseRobotEnvironment._flatten`)
        concatenates per layout leaf. Composites have no ``to_flat``: flattening a
        whole ``RobotState``/``RobotAction`` is only meaningful against a declared
        ``action_layout`` (which fixes the component order and the rotation
        representation), so it lives on the env, not the schema."""
        return self.data.detach().to(torch.float32).cpu().contiguous().numpy()


# ---------------------------------------------------------------------------
# _TensorChunkComposite — composite node holding Optional children
# ---------------------------------------------------------------------------


class _TensorChunkComposite:
    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        _TYPE_REGISTRY[cls.__name__] = cls

    def __post_init__(self):
        lengths = {len(v) for _, v in self._fields()}
        assert len(lengths) > 0, f"{type(self).__name__} must have at least one populated field"
        assert len(lengths) == 1, f"Inconsistent lengths across {type(self).__name__} fields: {lengths}"

    def _fields(self):
        """Yield (name, value) for non-None dataclass fields."""
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None:
                yield f.name, v

    def __len__(self) -> int:
        for _, v in self._fields():
            return len(v)
        return 0

    def _apply(self, fn, other=None):
        """Apply fn to each populated field. Unary: fn(name, v). Binary: fn(name, v, ov)."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if v is None:
                out[f.name] = None
            elif other is not None:
                ov = getattr(other, f.name)
                out[f.name] = v if ov is None else fn(f.name, v, ov)
            else:
                out[f.name] = fn(f.name, v)
        return self.__class__(**out)

    def __getitem__(self, index):
        return self._apply(lambda _, v: v[index])

    def pad_to(self, chunk_size):
        return self._apply(lambda _, v: v.pad_to(chunk_size))

    def normalize(self, stats: Dict, norm_type: str = "mean_std"):
        return self._apply(lambda name, v: v.normalize(stats[name], norm_type) if name in stats else v)

    def denormalize(self, stats: Dict, norm_type: str = "mean_std"):
        return self._apply(lambda name, v: v.denormalize(stats[name], norm_type) if name in stats else v)

    def convert_rotation(self, target_representation: RotationRepresentation):
        return self._apply(lambda _, v: v.convert_rotation(target_representation))

    def resample(self, src_fps, tgt_fps, n_target=None, src_offset=0.0, mode="linear"):
        n_src = len(self)
        if n_target is None:
            n_target = max(1, int(round(n_src * tgt_fps / src_fps)))
        if abs(src_fps - tgt_fps) < 1e-6 and abs(src_offset) < 1e-9 and n_target == n_src:
            return self
        return self._apply(lambda _, v: v.resample(src_fps, tgt_fps, n_target, src_offset, mode))

    def resample_at(self, positions, mode="linear"):
        """Resample each populated field at explicit fractional positions;
        dispatches per field (``Rotation`` SLERPs, others lerp), ``Arm`` recurses."""
        return self._apply(lambda _, v: v.resample_at(positions, mode))

    def interpolate(self, chunk_size: int, mode: str = "linear"):
        """Re-time to exactly ``chunk_size`` frames, interpolating evenly between
        this chunk's frames as keyframes (see :meth:`_TensorChunk.interpolate`).
        The positions are derived once from the composite's own length -- every
        field shares it -- then applied through :meth:`resample_at`."""
        if chunk_size == len(self):
            return self
        return self.resample_at(_interpolate_positions(len(self), chunk_size), mode)

    def __sub__(self, other):
        return self._apply(lambda _, a, b: a - b, other)

    def __add__(self, other):
        return self._apply(lambda _, a, b: a + b, other)

    def __radd__(self, other):
        return self.__add__(other)

    def unpack(self):
        return [self[i] for i in range(len(self))]

    @classmethod
    def cat(cls, items):
        """Concatenate same-type composites along the chunk dim, field by field.

        The inverse of :meth:`unpack`; a child populated in *every* item is
        concatenated via its own :meth:`cat`, and one that any item leaves ``None``
        is dropped -- it has no value over part of the concatenated span, so the
        result carries the intersection of the inputs' populated fields (an empty
        intersection trips the composite's own "at least one populated field"
        assert). That is what lets frames with unequal field sets be joined at all
        -- e.g. a dataset trajectory poorer than the current state -- and a dropped
        field the env's ``action_layout`` needs then fails loudly at flatten time
        rather than silently carrying a stale value. The result's class and every
        non-``data`` attribute (rotation representation, ``allow_relative``, ...)
        come from ``items[0]``."""
        ref = items[0]
        out = {}
        for f in fields(ref):
            vals = [getattr(it, f.name) for it in items]
            out[f.name] = type(vals[0]).cat(vals) if all(v is not None for v in vals) else None
        return ref.__class__(**out)

    def to_dict(self) -> dict:
        out = {"type": type(self).__name__}
        for name, v in self._fields():
            out[name] = v.to_dict()
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "_TensorChunkComposite":
        kw = {}
        for name, value in d.items():
            if name == "type" or name not in cls.__dataclass_fields__:
                continue
            child_cls = _TYPE_REGISTRY[value["type"]]
            kw[name] = child_cls.from_dict(value)
        return cls(**kw)


@dataclass
class Position(_TensorChunk):
    data: torch.Tensor
    is_relative: bool = False
    allow_relative: bool = True

    def normalize(self, stats, norm_type="mean_std"):
        off, scale = _norm_coeffs(stats, norm_type, self.data.dtype, self.data.device)
        return replace(self, data=(self.data - off[None]) / scale[None])

    def denormalize(self, stats, norm_type="mean_std"):
        off, scale = _norm_coeffs(stats, norm_type, self.data.dtype, self.data.device)
        return replace(self, data=self.data * scale[None] + off[None])

    def __sub__(self, other) -> "Position":
        if not self.allow_relative:
            return self
        assert not self.is_relative, "Cannot subtract from a relative Position"
        return Position(data=self.data - other.data, is_relative=True, allow_relative=self.allow_relative)

    def __add__(self, other) -> "Position":
        if not self.allow_relative:
            return self
        assert self.is_relative and not other.is_relative, (
            "Position.__add__ requires self.is_relative=True and other.is_relative=False"
        )
        return Position(data=self.data + other.data, is_relative=False, allow_relative=self.allow_relative)


@dataclass
class Rotation(_TensorChunk):
    data: torch.Tensor
    representation: RotationRepresentation
    is_relative: bool = False
    allow_relative: bool = True

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.representation, str):
            self.representation = RotationRepresentation(self.representation)
        assert isinstance(self.representation, RotationRepresentation)
        assert self.data.size(1) == self.representation.dim, (
            f"Rotation data dim {self.data.size(1)} != representation.dim {self.representation.dim}"
        )

    def normalize(self, stats=None, norm_type: str = "mean_std"):
        if stats is None:
            return replace(self)
        off, scale = _norm_coeffs(stats, norm_type, self.data.dtype, self.data.device)
        return replace(self, data=(self.data - off[None]) / scale[None])

    def denormalize(self, stats=None, norm_type: str = "mean_std"):
        if stats is None:
            return replace(self)
        off, scale = _norm_coeffs(stats, norm_type, self.data.dtype, self.data.device)
        return replace(self, data=self.data * scale[None] + off[None])

    def convert_rotation(self, target_representation: RotationRepresentation) -> "Rotation":
        if target_representation == self.representation:
            return replace(self)
        new_data = _convert_rotation(self.data, self.representation, target_representation)
        return Rotation(
            data=new_data,
            representation=target_representation,
            is_relative=self.is_relative,
            allow_relative=self.allow_relative,
        )

    def resample(self, src_fps, tgt_fps, n_target=None, src_offset=0.0, mode="linear"):
        n_src = len(self)
        assert n_src > 0
        if n_target is None:
            n_target = max(1, int(round(n_src * tgt_fps / src_fps)))
        if abs(src_fps - tgt_fps) < 1e-6 and abs(src_offset) < 1e-9 and n_target == n_src:
            return self
        positions = _resample_positions(n_src, src_fps, tgt_fps, n_target, src_offset)
        return replace(self, data=_resample_rotation(self.data, self.representation, positions))

    def resample_at(self, positions, mode="linear"):
        """SLERP at explicit fractional positions (overrides the leaf lerp)."""
        positions = np.asarray(positions, dtype=np.float64)
        return replace(self, data=_resample_rotation(self.data, self.representation, positions))

    @classmethod
    def cat(cls, items):
        """Concatenate rotations, canonicalizing each to ``items[0]``'s
        representation first.

        The leaf :meth:`~_TensorChunk.cat` keeps ``items[0]``'s non-``data`` fields,
        so stacking raw data across mixed representations would label e.g. euler
        rows as ``ROT_VEC`` -- silently, whenever the dims happen to agree. Joining
        differently-encoded frames is normal (a policy chunk, a dataset trajectory
        and the robot's own state need not share an encoding), so the conversion
        belongs here rather than in every caller."""
        ref = items[0]
        return super().cat([it.convert_rotation(ref.representation) for it in items])

    def __sub__(self, other) -> "Rotation":
        if not self.allow_relative:
            return self
        assert not self.is_relative, "Cannot subtract from a relative Rotation"
        delta = _compose_rotations(
            other.data,
            other.representation,
            self.data,
            self.representation,
            self.representation,
            inverse_r1=True,
        )
        return Rotation(
            data=delta, representation=self.representation, is_relative=True, allow_relative=self.allow_relative
        )

    def __add__(self, other) -> "Rotation":
        if not self.allow_relative:
            return self
        assert self.is_relative and not other.is_relative, (
            "Rotation.__add__ requires self.is_relative=True and other.is_relative=False"
        )
        composed = _compose_rotations(
            other.data,
            other.representation,
            self.data,
            self.representation,
            self.representation,
            inverse_r1=False,
        )
        return Rotation(
            data=composed, representation=self.representation, is_relative=False, allow_relative=self.allow_relative
        )


@dataclass
class Arm(_TensorChunkComposite):
    joint_position: Optional[Position] = None
    eef_position: Optional[Position] = None
    eef_rotation: Optional[Rotation] = None

    def __post_init__(self):
        has_joint = self.joint_position is not None
        has_eef_pos = self.eef_position is not None
        has_eef_rot = self.eef_rotation is not None
        assert has_eef_pos == has_eef_rot, "eef_position and eef_rotation must both be set or both be None"
        assert has_joint or has_eef_pos, "Arm must have at least joint_position or (eef_position, eef_rotation)"
        if has_joint:
            assert isinstance(self.joint_position, Position)
        if has_eef_pos:
            assert isinstance(self.eef_position, Position)
            assert self.eef_position.data.size(1) == 3, (
                f"eef_position must have D=3, got {self.eef_position.data.size(1)}"
            )
        if has_eef_rot:
            assert isinstance(self.eef_rotation, Rotation)
        super().__post_init__()


@dataclass
class RobotAction(_TensorChunkComposite):
    left_arm: Optional[Arm] = None
    right_arm: Optional[Arm] = None
    left_gripper: Optional[Position] = None
    right_gripper: Optional[Position] = None
    left_hand: Optional[Position] = None
    right_hand: Optional[Position] = None
    torso: Optional[Position] = None
    head: Optional[Position] = None

    def __post_init__(self):
        super().__post_init__()
        for grip_name in ("left_gripper", "right_gripper"):
            g = getattr(self, grip_name)
            if g is not None and g.allow_relative:
                warnings.warn(
                    f"{grip_name}.allow_relative is True; most grippers should "
                    f"not participate in delta. Pass Position(..., allow_relative=False) "
                    f"if this was unintentional.",
                    stacklevel=2,
                )


@dataclass
class RobotState(RobotAction):
    def __post_init__(self):
        super().__post_init__()
        for name, v in self._fields():
            if isinstance(v, (Position, Rotation)):
                assert not v.is_relative, f"RobotState.{name}.is_relative must be False"
            elif isinstance(v, Arm):
                for sub_name, sub_v in v._fields():
                    assert not sub_v.is_relative, f"RobotState.{name}.{sub_name}.is_relative must be False"

    def __sub__(self, other):
        return NotImplemented

    def __add__(self, other):
        return NotImplemented
