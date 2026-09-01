import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence

import ray

from ..registry import AGENT_REGISTRY
from ..serving.client import InferenceClient
from ..serving.requests import VLAInferenceRequest
from ..utils.robot import RobotAction
from .base import BaseAgent

logger = logging.getLogger(__name__)


class CommandType(Enum):
    WAIT = 0
    RUN = 1
    STEP = 2
    MOVE = 3
    REPLAY = 4
    STOP = 5
    RESET = 6


@dataclass
class CommandMessage:
    type: CommandType
    extra_args: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.type in (CommandType.RUN, CommandType.STEP):
            assert self.extra_args.get("prompt") is not None, "RUN/STEP requires extra_args['prompt']."
            if "host" in self.extra_args:
                assert isinstance(self.extra_args["host"], str) and len(self.extra_args["host"]) > 0, (
                    "extra_args['host'] must be a non-empty string."
                )
            if "token" in self.extra_args:
                assert isinstance(self.extra_args["token"], str) and len(self.extra_args["token"]) > 0, (
                    "extra_args['token'] must be a non-empty string."
                )
            if "port" in self.extra_args:
                assert isinstance(self.extra_args["port"], int), "extra_args['port'] must be an int."
        elif self.type == CommandType.MOVE:
            assert isinstance(self.extra_args.get("target"), dict), (
                "MOVE extra_args must contain 'target' (a RobotAction.to_dict()-shaped dict)."
            )
        elif self.type == CommandType.REPLAY:
            assert isinstance(self.extra_args.get("episode_index"), int), (
                "REPLAY extra_args must contain 'episode_index' (int)."
            )
            assert self.extra_args.get("source") in ("action", "state"), (
                "REPLAY extra_args['source'] must be 'action' or 'state'."
            )
        elif self.type == CommandType.RESET:
            # RESET may carry the prompt to restart with (optional).
            pass
        else:
            assert "prompt" not in self.extra_args, "Only RUN/STEP/RESET carry a prompt."


@AGENT_REGISTRY.register("RobotAgent")
class RobotAgent(BaseAgent):
    dedicated_process = True
    ray_actor_options = {"concurrency_groups": {"control": 1, "gui": 4, "command": 1}}

    def __init__(
        self,
        model: InferenceClient,
        buffer: Any = None,
        episode_id: Optional[str] = None,
        *,
        env_type: str,
        env_config: Dict[str, Any] = {},
        reset_config: Dict[str, Any] = {},
        max_steps: Optional[int] = None,
        infer_interval: Optional[int] = None,
        latency_window: int = 10,
    ):
        super().__init__(model, buffer, episode_id)
        self.env_type = env_type
        self.env_config = env_config
        self.reset_config = reset_config
        self.max_steps = max_steps
        self.infer_interval = infer_interval
        assert self.infer_interval is None or self.infer_interval > 0, (
            f"infer_interval must be positive, got: {self.infer_interval}"
        )

        self._fixed_episode_id = episode_id
        self._latencies: Deque[float] = deque(maxlen=max(1, int(latency_window)))

        self.env: Optional[Any] = None
        self._command_queue: Optional[Deque[CommandMessage]] = None
        self._mode: Optional[int] = None
        self._fps: Optional[int] = None
        self._ready = False
        self._stopped = False

    @ray.method(concurrency_group="gui")
    async def wait_until_ready(self, timeout: float = 60.0) -> Dict[str, Any]:
        deadline = time.monotonic() + float(timeout)
        while not self._ready:
            if self._stopped:
                raise RuntimeError(
                    f"episode {self.episode_id}: the agent has been stopped, so no loop "
                    "will become ready on it -- whether or not one ever ran."
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"episode {self.episode_id}: env {self.env_type} was not ready "
                    f"within {timeout:g}s. Either building it takes longer than that "
                    "(raise the timeout), or ``loop`` was never called / already "
                    "failed -- check the loop's own result for the error."
                )
            await asyncio.sleep(0.01)

        action_layout = list(self.env.action_layout)
        state_layout = list(self.env.state_layout)
        for name, layout in (("action_layout", action_layout), ("state_layout", state_layout)):
            assert layout, (
                f"env {self.env_type} declares an empty {name}, so there is nothing to "
                "name its components with -- the GUI is built entirely from these."
            )
        return {
            "image_keys": self.env.image_key_map,
            "action_template": action_layout,
            "state_template": state_layout,
        }

    @ray.method(concurrency_group="command")
    def submit_command(self, msg: CommandMessage) -> None:
        assert self._ready, (
            "a command was submitted before the agent is ready. The env, the command queue and the "
            "mode live in ``loop`` -- built on its own ``control`` thread, dropped when "
            "it leaves -- so this has nothing to answer with until the loop is up, and "
            "nothing again after ``shutdown``. Await ``wait_until_ready`` first (and "
            "note that a command only means something while a loop is running)."
        )
        self._command_queue.append(msg)

    @ray.method(concurrency_group="command")
    def shutdown(self) -> None:
        # Withdrawn here rather than in the loop's ``finally``: the loop can be a whole
        # inference away from noticing, and for that window it is neither going to take
        # another command nor worth being asked about. The teardown itself stays in the
        # ``finally`` -- the ``control`` thread owns that state and is still using it.
        self._ready = False
        self._stopped = True

    def _latency_ms(self) -> float:
        seen = list(self._latencies)
        return 1000.0 * sum(seen) / len(seen) if seen else 0.0

    @ray.method(concurrency_group="control")
    async def loop(
        self,
        prompt: str = "",
        *,
        seed_commands: Sequence[CommandMessage] = (),
        exit_when_idle: bool = False,
        snapshot_callback: Optional[Callable[..., None]] = None,
    ) -> Dict[str, Any]:
        assert not self._ready, (
            f"episode {self.episode_id}: a loop is already running on this agent, and "
            "the env, the queue and the mode are its own -- a second one would build a "
            "second env over the first (two GL contexts, two robots commanded at once) "
            "and race it for the commands. One agent, one loop."
        )

        from ..environments.robot import BaseRobotEnvironment
        from ..registry import ENVIRONMENT_REGISTRY

        env_cls = ENVIRONMENT_REGISTRY[self.env_type]
        if not (isinstance(env_cls, type) and issubclass(env_cls, BaseRobotEnvironment)):
            raise TypeError(
                f"RobotAgent requires a BaseRobotEnvironment; env_type {self.env_type!r} is registered as {env_cls!r}."
            )

        env = self.env = env_cls(**self.env_config)
        self._command_queue = deque(seed_commands)
        self._mode = CommandType.WAIT.value
        self._fps = int(env.fps)

        self._latencies.clear()

        pace_hz: Optional[float] = float(self._fps) if env.realtime else None
        next_tick: Optional[float] = None

        max_steps = self.max_steps
        rk = dict(self.reset_config)
        mode = "STOP"

        alive = False
        pending: List[RobotAction] = []
        since_trigger = 0
        trigger_pending = 0
        gen: Optional[asyncio.Future] = None
        recording: Optional[str] = None
        outcome = {"success": False, "error": None, "steps": 0}

        last_action: Optional[RobotAction] = None

        def _fold(result: Dict[str, Any]) -> bool:
            if result["done"]:
                outcome["success"] = True
            if result["error"] is not None:
                outcome["error"] = result["error"]
            return bool(result["done"] or result["error"] is not None)

        async def _open_recording() -> None:
            nonlocal recording
            if self.buffer is None:
                return
            await _close_recording()
            recording = self._fixed_episode_id or uuid.uuid4().hex
            await self.buffer.create_episode.remote(recording, fps=self._fps)

        async def _close_recording() -> None:
            nonlocal recording
            if self.buffer is None or recording is None:
                return
            rec, recording = recording, None
            await self.buffer.finalize.remote(rec)

        async def _record(obs: Dict[str, Any], action: Optional[RobotAction]) -> None:
            if self.buffer is None or recording is None:
                return
            images = obs.get("images")
            await self.buffer.put.remote(
                recording,
                [images] if images else [],
                obs.get("state"),
                action=action.to_dict() if action is not None else None,
            )

        def _publish(obs: Optional[Dict[str, Any]] = None) -> None:
            if snapshot_callback is None:
                return
            snapshot_callback(
                command=int(self._mode),
                latency_ms=self._latency_ms(),
                state=env.get_state().to_dict() if obs is None else obs["state"],
                action=None if last_action is None else last_action.to_dict(),
                images=None if obs is None else obs["images"],
            )

        async def _harvest() -> None:
            nonlocal gen, pending
            g, gen = gen, None
            arrived = await g
            assert len(arrived) > trigger_pending, (
                f"episode {self.episode_id}: the policy returned {len(arrived)} "
                f"action(s) having been handed {trigger_pending} un-executed one(s), so "
                "none of it reaches past the plan it already had. A chunk is aligned to "
                "the trigger, so it has to be longer than prev_actions to contribute "
                "anything. Either this policy's horizon is shorter than infer_interval, "
                "or it is not honouring prev_actions/delay_steps at all."
            )
            pending = arrived.unpack()[since_trigger:]

        def _launch() -> "asyncio.Future":
            nonlocal since_trigger, trigger_pending
            delay = (
                self.infer_interval if (rtc and not env.realtime) else round(self._latency_ms() / 1000.0 * self._fps)
            )
            req = VLAInferenceRequest.from_observation(
                obs,
                prompt,
                prev_actions=(env.flatten_action(RobotAction.cat(pending)) if pending else None),
                delay_steps=min(len(pending), delay),
            )
            since_trigger, trigger_pending = 0, len(pending)
            started = time.perf_counter()
            response = self.model.generate_async(req)

            async def _await_chunk() -> RobotAction:
                chunk = await response
                self._latencies.append(time.perf_counter() - started)
                return RobotAction.from_dict(chunk) if isinstance(chunk, dict) else chunk

            return asyncio.ensure_future(_await_chunk())

        def _cancel_inflight() -> None:
            nonlocal gen, since_trigger, trigger_pending
            if gen is not None:
                gen.cancel()
                gen = None
            since_trigger = trigger_pending = 0

        def _budget_left() -> bool:
            return max_steps is None or outcome["steps"] < max_steps

        async def _pace() -> None:
            nonlocal next_tick
            if pace_hz is None:
                return
            dt = 1.0 / pace_hz
            now = time.monotonic()
            if next_tick is None or next_tick < now - dt:
                next_tick = now
            delay = next_tick - now
            if delay > 0:
                await asyncio.sleep(delay)
            next_tick += dt

        def _ensure_started():
            nonlocal alive, pending
            _cancel_inflight()
            pending = []
            if not alive:
                alive = not _fold(env.reset(**rk))
                if snapshot_callback is not None:
                    _publish(env.get_observation())

        def _move_to(target: RobotAction) -> RobotAction:
            assert len(target) == 1, (
                f"_move_to target must be a single frame, got {len(target)}: the "
                "interpolation runs between the current state and one pose."
            )
            n = max(1, self._fps)
            chunk = RobotAction.cat([env.get_state(), target])  # idx0 = now, idx1 = target
            return chunk.interpolate(n + 1)[1:]  # [1:] drops the current pose

        async def _play_manual(traj: RobotAction) -> None:
            nonlocal alive, last_action
            for action in traj.unpack():
                if self._stopped or self._command_queue or not _budget_left():
                    break
                obs = env.get_observation()
                _publish(obs)
                await _pace()
                over = _fold(env.step(action, manual=True))
                last_action = action
                outcome["steps"] += 1
                await _record(obs, action)
                if over:
                    alive = False
                    break

        try:
            self._ready = True

            while not self._stopped:
                # Only this thread pops, and an ``append`` never empties a non-empty
                # deque, so a truthy check cannot be followed by a failing ``popleft``.
                cmd = self._command_queue.popleft() if self._command_queue else None

                if cmd is not None:
                    # Realize it -- updating mode/alive/prompt -- and then start the pass
                    # over, so the loop only ever steps with an empty queue. That is what
                    # lets a command already waiting behind this one pre-empt the motion
                    # this one just authorized: a STOP sitting behind a RUN parks the
                    # episode without an action being played. It is also why the seeds can
                    # go through the queue at all -- realized anywhere but here, a RUN
                    # would step in the same breath and leave that STOP a step too late.
                    ea = cmd.extra_args or {}
                    if cmd.type in (CommandType.RUN, CommandType.STEP):
                        prompt = ea.get("prompt") or prompt
                        self._mode = cmd.type.value
                        _ensure_started()
                        mode = "RUN" if cmd.type == CommandType.RUN else "STEP_ONCE"
                    elif cmd.type == CommandType.MOVE:
                        self._mode = cmd.type.value
                        _ensure_started()
                        # The GUI names one pose to reach; ``_move_to`` interpolates from the
                        # current state to it. Playing it leaves the robot where the target
                        # was, so a following RUN infers on where it *is* now.
                        target = RobotAction.from_dict(ea["target"])
                        await _play_manual(_move_to(target))
                        mode = "STOP"
                    elif cmd.type == CommandType.REPLAY:
                        self._mode = cmd.type.value
                        _ensure_started()
                        # Approach the episode's first frame, then play the recorded
                        # trajectory -- one inference-free run. Each ``iter_episode``
                        # step yields a structured ``action`` (``RobotAction``) and
                        # ``state`` (``RobotState``); ``source`` picks which, and we take
                        # that step's immediate (first) frame. The dataset must be
                        # *absolute* (``use_delta_action=False``, enforced where the
                        # reader is built -- ``api/control.py``) -- a relative trajectory
                        # cannot be replayed against, or interpolated from, the absolute
                        # current state.
                        reader = env._data_reader
                        assert reader is not None, "REPLAY requires a data_reader."
                        frames = [
                            step[ea["source"]][0:1]
                            for step in reader.iter_episode(ea["episode_index"], include_images=False)
                        ]
                        if frames:
                            traj = RobotAction.cat(frames)  # absolute trajectory, len N
                            traj = RobotAction.cat([_move_to(traj[0:1]), traj])
                        else:
                            traj = env.get_state()[0:1]  # empty episode -> hold pose
                        await _play_manual(traj)
                        mode = "STOP"
                    elif cmd.type == CommandType.STOP:
                        # Nothing to interrupt: whatever was playing already returned, and
                        # dropping ``pending`` plus the in-flight inference is the whole of
                        # stopping. The robot holds its last commanded pose; the episode is
                        # considered ended, so a later RUN starts fresh.
                        _cancel_inflight()
                        self._mode = CommandType.STOP.value
                        mode, alive, pending = "STOP", False, []
                    elif cmd.type == CommandType.RESET:
                        # Force a fresh episode: dropping ``alive`` makes ``_ensure_started``
                        # reset, and it also drops the in-flight inference and whatever actions
                        # were in hand, so nothing of the old episode's motion survives.
                        prompt = ea.get("prompt") or prompt
                        alive = False
                        _ensure_started()
                        self._mode = CommandType.RUN.value
                        mode = "RUN"

                    if mode in ("RUN", "STEP_ONCE"):
                        if alive:
                            # A run is starting, so this is where its recording opens: the one
                            # place, for all three commands that start one, and late enough to
                            # know there is an episode to record -- a reset that raised never
                            # gets here, and one that came back already over takes the branch
                            # below instead of leaving an empty episode open in the buffer.
                            await _open_recording()
                        else:
                            # The episode this command asked to run was over before any
                            # inference was needed (a reset that came back ``done``). Park, so
                            # the idle branch below sleeps (or returns) instead of spinning,
                            # and the GUI stops reporting RUN over a dead episode.
                            self._mode = CommandType.STOP.value
                            mode = "STOP"
                    continue

                if mode in ("RUN", "STEP_ONCE") and alive:
                    rtc = self.infer_interval is not None and mode == "RUN"

                    obs = env.get_observation()
                    _publish(obs)

                    if gen is not None and (
                        gen.done() if env.realtime else rtc and since_trigger >= self.infer_interval
                    ):
                        await _harvest()

                    # Launch. Either there is nothing left to play, or ``interval``
                    # actions have gone by since the last trigger. Never more than one
                    # in flight, so a slow policy simply stretches the period.
                    if gen is None and (not pending or (rtc and since_trigger >= self.infer_interval)):
                        # The request leaves *here*, at the trigger, because ``_launch``
                        # is a plain call rather than a coroutine -- which is also where
                        # the RTC pair and the trigger counters are worked out.
                        gen = _launch()

                    # Out of actions, so there is nothing to overlap with: block for
                    # the answer with the robot holding pose. This is where the
                    # sequential path spends every one of its inferences. One harvest is
                    # all it takes -- ``_harvest`` cannot come back empty-handed, so there
                    # is nothing to retry.
                    if not pending:
                        if gen is None:
                            gen = _launch()
                        await _harvest()
                        # We just held pose for a whole inference; re-anchor the pace
                        # grid to now so the next action fires immediately rather than
                        # the pacer bursting to "catch up" the ticks it missed.
                        next_tick = None

                    # Pace, then submit. ``_pace`` is the command clock on a real robot
                    # (no-op on sim); ``step`` publishes the target / advances the sim
                    # and returns at once, so the tick we just spent in ``_pace`` -- and
                    # the yield inside it -- is what lets the in-flight inference land
                    # and be seen by the harvest at the top of the next pass.
                    action = pending.pop(0)
                    await _pace()
                    over = _fold(env.step(action))
                    last_action = action
                    outcome["steps"] += 1
                    await _record(obs, action)
                    since_trigger += 1

                    if over or not _budget_left() or (mode == "STEP_ONCE" and not pending):
                        # The world ended it, the budget ran out, or the single step
                        # finished its chunk. ``steps`` counts every action this call
                        # played (MOVE/REPLAY frames included) and is never reset, so
                        # ``max_steps`` bounds the whole ``loop``, not each run inside
                        # it: under a budget, a re-RUN or RESET continues the same
                        # count. Eval drives one run per loop, so there the two read
                        # the same; deploy sets no budget.
                        _cancel_inflight()
                        self._mode = CommandType.STOP.value
                        mode, alive, pending = "STOP", False, []
                else:
                    # Parked -- by a STOP, by the world saying done, by the budget, by
                    # a STEP finishing its chunk, or by a MOVE/REPLAY that has played
                    # out. Whichever it was, the run is over, so its recording ends
                    # here. Closing off the *state* rather than at each of those
                    # transitions is what keeps "a recording is one Start..Stop" true
                    # without every one of them having to remember to say so -- and it
                    # lands after a manual trajectory's frames, so those are still part
                    # of the run that was playing them.
                    await _close_recording()
                    if exit_when_idle and not self._command_queue:
                        break  # parked, and nobody else will command us
                    # Nothing was waiting (a command would have taken the pass above),
                    # so there is nothing to do but keep the GUI's view current. Status
                    # only: parked means no new frame exists on a sim, and the reader
                    # goes on showing the last one it was sent.
                    _publish()
                    await asyncio.sleep(0.005)
            return {**outcome, "extra": {}}
        finally:
            _cancel_inflight()

            self._ready = False
            self.env = None
            self._command_queue = None
            self._mode = None
            self._fps = None
            try:
                env.close()
            except Exception as e:  # noqa: BLE001
                logger.warning("closing env for episode %s failed: %s", self.episode_id, e)

            await _close_recording()

    @ray.method(concurrency_group="control")
    async def rollout(self, prompt: str = "") -> Dict[str, Any]:
        out = await self.loop(
            prompt,
            seed_commands=[CommandMessage(type=CommandType.RUN, extra_args={"prompt": prompt})],
            exit_when_idle=True,
        )
        return {
            "episode_id": self.episode_id,
            "success": bool(out["success"]),
            "error": out["error"],
            "steps": out["steps"],
            **(out.get("extra") or {}),
        }
