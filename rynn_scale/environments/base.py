from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseEnvironment(ABC):
    @abstractmethod
    def reset(self, **kwargs) -> Dict[str, Any]:
        """Place the world in a starting condition and report the outcome.

        ``kwargs`` say which world and which starting condition (a bddl file, an init
        state) -- not what the robot is being asked to do in it (see the class note).
        Returns ``{"success", "error", "done"}`` only, never a frame: nothing has
        executed yet, so ``done`` is ``False`` and ``error`` is ``None``. The first
        inference observation is fetched separately with :meth:`get_observation` once
        this returns.

        **Slow whichever kind of env answers it, and it blocks for that long.** A sim
        builds or resets a world and renders it; a real machine waits until it is
        physically at rest, so that the first observation does not catch it mid-motion
        and have the policy plan from a pose it has already left. An env that waits
        bounds its own waiting with a timeout -- nothing outside can interrupt it, so
        the deadline has to be its own.
        """

    @abstractmethod
    def step(self, action: Any, *, manual: bool = False) -> Dict[str, Any]:
        """Submit exactly one action and report what happened during this call.

        One action, not a chunk: iterating is the caller's job, and it is what lets it
        decide anything -- launch an inference, take a new command, stop -- between two
        of them. ``manual=True`` marks an inference-free action (a hand-driven MOVE, a
        replayed trajectory); implementations play whatever they are handed verbatim,
        so it is informational only.

        **Does not pace.** The command rate is the caller's: a real env publishes the
        target to its control process and returns immediately, a sim advances its
        logical clock by one action. Neither blocks out a command tick -- the loop one
        layer up spends that time, which is what lets it gate the rate and overlap
        inference (see :class:`~rynn_scale.agents.robot.RobotAgent`).

        Returns ``{"success", "error", "done"}`` -- no frame (fetch it with
        :meth:`get_observation`). ``success`` is whether the action was *applied*
        (real: always, since a write cannot fail here; sim: the sim stepped without
        raising). ``error`` carries the message when it was not. ``done`` is the
        world's task-completion signal (sim benchmarks fire it; a real robot never
        has one). There is no step count -- one call is one action.
        """

    @abstractmethod
    def get_observation(self) -> Dict[str, Any]:
        """The current frame -- the inference observation and the recorded frame.

        Split from :meth:`step` so the caller fetches a frame exactly when it needs
        one (before an inference, and once per action it records) rather than on every
        action unconditionally. The keys are the env family's to name. This is the
        costly half of a pass on either kind of env -- a sensor read and a decode per
        camera on a real one, the render on a sim -- which is the whole point of the
        split: an action that nobody wants to look at costs neither.

        Costly and still sync: the work is a blocking render or a blocking read, and
        the caller has nothing to run while it happens anyway -- an inference in flight
        is another process's work, which proceeds whether or not this one yields.
        """

    @abstractmethod
    def close(self) -> None:
        """Release resources.

        Sync like the rest, and here it matters most: the caller closes from a
        ``finally``, where an ``await`` could be cancelled and leak whatever the env
        holds (a GL context, a forked control child and its shm segments).
        """
