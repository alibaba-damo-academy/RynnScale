"""Environments: the worlds an agent loop drives.

:class:`~rynn_scale.environments.base.BaseEnvironment` is the contract and nothing
else -- reset, step, get_observation, close, plus the rate the caller paces itself
by. It says nothing about what is being driven, so a second kind of world (one whose
actions are not a robot's) is a new module here rather than a special case inside the
robot one.

:mod:`rynn_scale.environments.robot` is that first kind: envs whose actions and
states are the standard ``RobotAction`` / ``RobotState`` schema, flattened per a
declared layout onto the one vector a body consumes. Both clocks live there --
:class:`~rynn_scale.environments.robot.SimRobotEnvironment` (logical, stepping is
rendering) and :class:`~rynn_scale.environments.robot.RealRobotEnvironment` (wall,
with a forked control process and the shm / interpolator / timer plumbing it needs)
-- and :class:`~rynn_scale.environments.libero.Libero` is a leaf of the first.
"""

from .base import BaseEnvironment
from .libero import Libero
from .robot import (
    BaseRobotEnvironment,
    RealRobotEnvironment,
    SimRobotEnvironment,
)

__all__ = [
    "Libero",
    "BaseEnvironment",
    "BaseRobotEnvironment",
    "SimRobotEnvironment",
    "RealRobotEnvironment",
]
