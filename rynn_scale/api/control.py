"""Deployment entrypoint for a RynnScale VLA policy: one robot, two actors.

Deploy runs **the same agent loop eval does**, just parameterized differently: where eval
seeds one RUN and returns when the episode ends, deploy leaves the loop running and feeds
it commands. It owns no GPU -- the policy is somebody else's process, reached over
``--server_url`` -- but it does hold a Ray runtime, over which it places two actors:

  * a :class:`~rynn_scale.agents.robot.RobotAgent` over a self-driving
    :class:`~rynn_scale.environments.robot.BaseRobotEnvironment`, which owns the command
    state machine (RUN/STEP/STOP/MOVE/REPLAY/WAIT) and initiates inference itself. Same
    actor form eval hosts, same :attr:`~..agents.robot.RobotAgent.ray_actor_options`;
  * a :class:`GradioGUI`, optional, in a **process of its own** because it is the
    expensive half -- a web server, worker threads, plot dataframes and decoded frames,
    none of which belong next to a control loop that must hit its clock.

Two channels, split by what they carry. **Control plane over Ray RPC**: one
``wait_until_ready`` at startup (it blocks until the env exists *and* answers with its
layouts), then a ``submit_command`` per click, plus the lifecycle calls -- low rate, and
each needs an answer or an ordering guarantee. **Observations over shared memory**: the
loop pushes each frame it already fetched (:func:`publish_snapshot`) and the GUI's widgets
sample those segments at their own rate, so a 20 Hz robot never pays for a redraw.

That second channel is why **both actors are pinned to the driver's node**
(:func:`_local_node_strategy`): shared memory is node-local, so a GUI placed elsewhere
would attach to nothing and show a blank panel forever while every RPC kept answering
normally -- a placement bug wearing a GUI bug's clothes.

Serve the policy:  ``python -m rynn_scale.api.serve --model_path ... --port 8000``
Headless:  ``python -m rynn_scale.api.control --server_url http://host:8000 --controller <robot>``
With GUI:  add ``--gui`` (requires ``gradio``).
"""

import asyncio
import functools
import json
import logging
import os
import signal
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from ..agents.robot import CommandMessage, CommandType

logger = logging.getLogger("control")

# Ray's ``num_cpus`` is accounting, not a limit: declared so a node already fully
# allocated to another job refuses these actors at startup, rather than running them
# alongside work promised the same cores -- which a control loop shows as missed ticks.
_ACTOR_CPUS = 1.0


# ─── Log formatting ─────────────────────────────────────────────────────────

_LEVEL_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[1;31m",
}
_RESET, _DIM = "\033[0m", "\033[2m"


class _ColorFormatter(logging.Formatter):
    def format(self, record):
        ts = self.formatTime(record, "%H:%M:%S")
        ms = int(record.created * 1000) % 1000
        color = _LEVEL_COLORS.get(record.levelname, "")
        level = f"{color}{record.levelname:<7s}{_RESET}"
        name = f"{_DIM}{record.name}{_RESET}"
        return f"{_DIM}{ts}.{ms:03d}{_RESET} {level} {name}  {record.getMessage()}"


def _setup_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(_ColorFormatter())
    logging.root.handlers.clear()
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)


# ─── Ray plumbing ───────────────────────────────────────────────────────────


def _local_node_strategy():
    """A *hard* affinity to the node this driver runs on -- see the module docstring for
    why both actors need it. ``soft=False`` so an unplaceable actor fails here; a soft
    affinity would fall back to another node, which is the silent-blank-panel outcome."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    return NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(), soft=False)


async def _await_ref(ref: Any) -> Any:
    """Await an ``ObjectRef`` through a coroutine, so ``asyncio.wait`` accepts it.

    A ref is awaitable but is not an ``asyncio.Future``, and :func:`_drive` *races* refs
    (the loop against the readiness barrier, then the loop against the GUI) rather than
    awaiting them in turn -- which is what surfaces a loop that died building its env as
    its own error instead of as somebody else's timeout.
    """
    return await ref


# ─── CLI inputs ─────────────────────────────────────────────────────────────


def _load_env_config(spec: Optional[str]) -> Dict[str, Any]:
    """Parse ``--env_config`` (a file path or inline JSON) into env ctor kwargs."""
    if not spec:
        return {}
    text = open(spec).read() if os.path.isfile(spec) else spec
    cfg = json.loads(text)
    assert isinstance(cfg, dict), "--env_config must be a JSON object"
    return cfg


def _build_replay_reader(args) -> Any:
    """The REPLAY dataset named by ``--data_path`` / ``--data_type``.

    Built here, in the CLI, and handed to the env as a live dataset: a recording is a
    deploy *input* like the model path, so an unregistered ``data_type`` or an unreadable
    path raises where the operator is standing -- built inside the env, which happens in
    the agent actor, the same mistake arrives as an actor that died in construction. It
    also keeps ``DataArguments`` and the heavy VLA data stack out of the env layer, and
    imported only when a recording was actually asked for.

    Two values are not the caller's to choose. ``use_delta_action`` is forced off: REPLAY
    plays the recording against -- and interpolates from -- the *absolute* current state.
    And the rotation encoding is left unset: the recording is read in whatever it was
    written in, and every consumer converts rather than relabels (``Rotation.cat`` before
    joining with the current state, then the env on the way to a flat command).
    """
    from ..arguments import DataArguments
    from ..datasets import build_dataset

    assert args.data_type, "--data_path needs a --data_type naming the dataset class that reads it."
    reader = build_dataset(
        DataArguments(
            data_type=args.data_type,
            data_path=args.data_path,
            use_delta_action=False,
            eef_rotation_repr=None,
            target_fps=args.target_fps,
        )
    )
    assert reader.use_delta_action is False, "REPLAY data_reader must be absolute."
    logger.info("Loaded %s replay episodes from %s", getattr(reader, "num_episodes", "?"), args.data_path)
    return reader


def _startup_commands(args) -> List[CommandMessage]:
    """The ``--preset_commands`` file, then an auto-RUN of ``--prompt``.

    These ride in on the ``loop`` call as its ``seed_commands`` rather than down the queue
    the GUI's buttons use, because that queue is the loop's own and does not exist before
    it. A preset is a *sequence* (MOVE somewhere, then RUN), so handing over the whole list
    in the call that starts the loop is both the order asked for and the earliest it can
    be given.
    """
    preset: List[Dict[str, Any]] = []
    if args.preset_commands:
        with open(args.preset_commands) as f:
            preset = json.load(f)
        assert isinstance(preset, list), "--preset_commands must be a JSON list"

    out = []
    for p in preset:
        # ``extra``, not ``extra or None``: a bare ``{"type": "STOP"}`` has no args, and
        # ``CommandMessage.__post_init__`` tests membership on the dict it is handed.
        extra = {
            k: p[k]
            for k in ("prompt", "host", "port", "token", "target", "episode_index", "source")
            if p.get(k) is not None
        }
        out.append(CommandMessage(type=CommandType[p["type"]], extra_args=extra))
    if args.prompt:
        out.append(CommandMessage(type=CommandType.RUN, extra_args={"prompt": args.prompt}))
    return out


# ─── GUI snapshot channel (shared memory) ────────────────────────────────────
#
# How the robot's live state reaches the GUI. The agent loop *pushes*: it is handed
# :func:`publish_snapshot` as its ``snapshot_callback`` and calls it with the frame it
# already fetched for inference, so the GUI's view costs no extra state read.
#
# Shared memory because the writer and the readers are in different processes: the loop
# writes from the agent actor, the GUI's server threads only read, and a segment addressed
# by name is all either needs to find the other -- no RPC on the observation path, so a
# frame costs no round trip and the control thread never waits on a reader. The seqlock is
# what makes that safe without a lock: a reader gets the newest complete frame or retries.
#
# Two kinds of segment, both single-writer seqlock (reusing ``environments/robot.py``'s
# primitives) and both carrying the write timestamp:
#
#   <prefix>_status        one JSON object, {command, latency_ms, state, action}
#                          -- ``action`` is null until one has been commanded.
#   <prefix>_img_<wire>    one per camera, raw RGB with its width/height.
#
# State and action travel as the structured ``to_dict()`` wire form -- the same shape the
# inference request and the recorded frame exchange -- and the GUI addresses a component
# by its leaf ``path``, so nothing here has to agree with a flat column order and a value
# the layout does not mention simply is not plotted.

_GUI_SHM: Dict[str, Any] = {}  # segment name -> SharedMemory, per process
_GUI_MAX_STATUS_BYTES = 64 * 1024  # the status segment's JSON payload cap


def publish_snapshot(
    prefix: str,
    *,
    command: int,
    latency_ms: float,
    state,
    action=None,
    images=None,
    max_image_bytes: int = 1280 * 720 * 3,
) -> None:
    """Write one GUI snapshot. This is the agent loop's ``snapshot_callback``.

    ``state`` and ``action`` are the structured wire dicts; ``action`` is ``None`` until
    one has been commanded. Single-writer: only the thread running the loop may call it,
    which is what the seqlocks assume. Segments are created on first use and sized from
    the caps, since a segment cannot be resized once a reader has attached.
    """
    import numpy as np

    from ..environments.robot import (
        bytes_shm_size,
        create_shm,
        image_shm_size,
        sweep_stale,
        write_bytes_shm,
        write_image_shm,
    )

    name = f"{prefix}_status"
    if name not in _GUI_SHM:
        # Reclaim a hard-killed prior run's segments before taking the prefix.
        sweep_stale(prefix)
        _GUI_SHM[name] = create_shm(name, bytes_shm_size(_GUI_MAX_STATUS_BYTES))
    payload = json.dumps(
        {"command": int(command), "latency_ms": float(latency_ms), "state": state, "action": action}
    ).encode()
    assert len(payload) <= _GUI_MAX_STATUS_BYTES, (
        f"a {len(payload)}B snapshot exceeds the status segment's {_GUI_MAX_STATUS_BYTES}B payload cap."
    )
    write_bytes_shm(_GUI_SHM[name], payload, time.time())

    for wire, img in (images or {}).items():
        name = f"{prefix}_img_{wire}"
        if name not in _GUI_SHM:
            _GUI_SHM[name] = create_shm(name, image_shm_size(max_image_bytes))
        data = np.ascontiguousarray(img, np.uint8).tobytes()
        assert len(data) <= max_image_bytes, (
            f"camera {wire} produced a {len(data)}B frame, over the "
            f"{max_image_bytes}B cap -- raise --gui_max_image_bytes."
        )
        write_image_shm(_GUI_SHM[name], data, time.time(), is_jpeg=False, width=img.shape[1], height=img.shape[0])


def _attach_shm(name: str):
    """Open an existing segment, caching the handle. ``None`` until it exists -- the
    GUI is up before the loop has published anything, and a reader must not create."""
    shm = _GUI_SHM.get(name)
    if shm is None:
        from multiprocessing import shared_memory

        from ..environments.robot import _detach_resource_tracker_from_shm

        _detach_resource_tracker_from_shm()
        try:
            shm = _GUI_SHM[name] = shared_memory.SharedMemory(name=name)
        except FileNotFoundError:
            return None
    return shm


def read_snapshot(prefix: str) -> Optional[Dict[str, Any]]:
    """The latest status as ``{command, latency_ms, state, action, ts}``, or ``None`` if
    nothing has been published yet."""
    from ..environments.robot import read_bytes_shm

    shm = _attach_shm(f"{prefix}_status")
    if shm is None:
        return None
    data, ts = read_bytes_shm(shm)
    if not data:
        return None
    snap = json.loads(data)
    snap["ts"] = ts
    return snap


def read_snapshot_image(prefix: str, wire: str):
    """The latest frame for one camera as ``HxWx3`` uint8, or ``None``."""
    import numpy as np

    from ..environments.robot import read_image_shm

    shm = _attach_shm(f"{prefix}_img_{wire}")
    if shm is None:
        return None
    data, _, _, w, h = read_image_shm(shm)
    if data is None or w <= 0 or h <= 0:
        return None
    return np.frombuffer(data, np.uint8).reshape(h, w, 3)


# ─── Control GUI (body-agnostic; commands by RPC, observations over shm) ──────


class GradioGUI:
    """Blocks control GUI, hosted as a Ray actor of its own (see :func:`_launch_gui`).

    Commands go out as ``submit_command`` RPCs on the ``RobotAgent`` handle it is given;
    the robot's live state comes back the other way entirely, through the shared-memory
    channel above. A command is rare and must not be lost; a frame is constant and must
    not cost a round trip.

    Everything drawn comes from the runtime ``meta`` (``image_keys`` /
    ``action_template`` / ``state_template``), so no robot-specific knowledge is baked
    in. The env's two layouts are the *single ordered source* for both the labels and the
    values read: a leaf declares its ``path`` / ``dim`` / ``labels`` once, and the plots,
    the MOVE inputs and the traces all derive from that. Both layouts are taken because
    an env may declare ``state_layout`` differently from ``action_layout`` -- a plot's two
    traces are the same joint by construction, because both are addressed by leaf ``path``
    plus offset within that leaf rather than by a shared column order. MOVE is an action,
    so its inputs come from ``action_template`` alone. Both templates are guaranteed
    non-empty by :meth:`~rynn_scale.agents.robot.RobotAgent.wait_until_ready`, so there is
    no layout-less GUI to degrade to.

    No poller and no cached payload: every widget reads the segments directly at whatever
    rate its own ``every=`` asks for, and a read is a seqlock memcpy. That is what
    decouples the display rate from the loop's publish rate.
    """

    def __init__(self, controller, meta: Dict[str, Any], shm_prefix: str, default_prompt: str = "", window: int = 100):
        import gradio as gr

        self.controller = controller
        self.shm_prefix = shm_prefix
        self.image_keys = list(meta.get("image_keys", []))
        self.action_template = list(meta.get("action_template", []))
        self.state_template = list(meta.get("state_template", []))
        # Index ``i`` names the ``i``-th published action value *and* the ``i``-th MOVE
        # input; the plot slots lead with this order too.
        self.action_labels = self._layout_labels(self.action_template)
        self._plots = self._plot_slots()
        self.default_prompt = default_prompt
        self._traces: Dict[int, deque] = {i: deque(maxlen=window) for i in range(len(self._plots))}

        with gr.Blocks() as self.interface:
            with gr.Tab("Visualization"):
                self._add_visualization_tab(gr)
            with gr.Tab("Control"):
                self._add_control_tab(gr)

            self.move_button.click(self._on_move, inputs=self.numbers, outputs=[])
            self.start_button.click(self._on_start, inputs=self.prompt_box, outputs=[])
            self.step_button.click(self._on_step, inputs=self.prompt_box, outputs=[])
            self.stop_button.click(self._on_stop, inputs=[], outputs=[])
            self.reset_button.click(self._on_reset, inputs=self.prompt_box, outputs=[])

    def _layout_labels(self, layout: List[Dict[str, Any]]) -> List[str]:
        """Flat component labels in ``layout`` order.

        Each leaf contributes *exactly* ``dim`` labels, so the list runs one-for-one with
        the layout's components -- which, for the action layout, is the order
        :meth:`_pack_target` consumes the MOVE inputs in. A leaf whose ``labels`` are
        absent or don't match ``dim`` is named from its path (``eef_position[0]``)."""
        out: List[str] = []
        for leaf in layout:
            dim = leaf["dim"]
            labs = list(leaf.get("labels") or [])
            if len(labs) != dim:
                labs = [f"{leaf['path'][-1]}[{i}]" for i in range(dim)]
            out.extend(labs)
        return out

    def _plot_slots(self) -> List[Tuple[str, Tuple[str, ...], int]]:
        """One ``(label, path, offset)`` per component to plot.

        A component is addressed the way the published payloads are shaped -- leaf
        ``path`` plus offset within that leaf -- so one slot reads both halves of a
        snapshot and the two traces are the same joint whatever order each layout lists
        its leaves in, and whether or not both carry the component at all. Action order
        first, matching the MOVE inputs; state-only components follow.
        """

        def keyed(layout, labels):
            out = {}
            for leaf in layout:
                for off in range(leaf["dim"]):
                    out[(tuple(leaf["path"]), off)] = labels[len(out)]
            return out

        action = keyed(self.action_template, self.action_labels)
        state = keyed(self.state_template, self._layout_labels(self.state_template))
        keys = list(action) + [k for k in state if k not in action]
        return [(action[key] if key in action else state[key], key[0], key[1]) for key in keys]

    @staticmethod
    def _component(payload, path: Tuple[str, ...], off: int) -> Optional[float]:
        """One component's current value out of a structured wire dict, or ``None``.

        The single reader for both halves of a snapshot, since a ``RobotState.to_dict()``
        and a ``RobotAction.to_dict()`` are the same shape. ``None`` covers every way a
        component can be absent -- nothing published yet, no action commanded yet, a
        payload that does not carry this leaf, or a rotation stored in a representation
        with fewer components than the layout names -- and the trace skips ``None``s. A
        snapshot is of an instant, so the leaf's leading (time) axis is indexed away.
        """
        node = payload
        for key in path:
            if not isinstance(node, dict) or node.get(key) is None:
                return None
            node = node[key]
        frame = (node.get("data") or [[]])[0]
        return float(frame[off]) if off < len(frame) else None

    def _add_visualization_tab(self, gr):
        cams = list(self.image_keys)
        with gr.Accordion(label="Camera", open=True):
            for i in range(0, len(cams), 3):
                with gr.Row():
                    for cam in cams[i : i + 3]:
                        gr.Image(
                            value=functools.partial(self._image, cam),
                            every=0.5,
                            label=cam,
                            type="numpy",
                            interactive=False,
                        )

        with gr.Accordion(label="State & Action", open=True):
            for i in range(0, len(self._plots), 3):
                with gr.Row():
                    for idx in range(i, min(i + 3, len(self._plots))):
                        gr.LinePlot(
                            value=functools.partial(self._update_joint, idx),
                            every=0.5,
                            label=self._plots[idx][0],
                            x="timestamp",
                            y="value",
                            color="label",
                            height=200,
                        )

    def _add_control_tab(self, gr):
        with gr.Row():
            gr.Textbox(value=self._update_controller_state, every=0.1, label="Controller State", interactive=False)
            gr.Textbox(value=self._update_latency, every=0.1, label="Infer Latency", interactive=False)
        with gr.Accordion(label="Move", open=False):
            self.numbers = []
            for i in range(0, len(self.action_labels), 7):
                with gr.Row():
                    for lab in self.action_labels[i : i + 7]:
                        self.numbers.append(gr.Number(value=0, label=lab, interactive=True))
            self.move_button = gr.Button("Move")
        with gr.Accordion(label="Policy", open=True):
            self.prompt_box = gr.Textbox(value=self.default_prompt, label="Prompt")
            with gr.Row():
                self.start_button = gr.Button("Start")
                self.step_button = gr.Button("Step")
                self.reset_button = gr.Button("Reset")
        self.stop_button = gr.Button("Stop")

    # ---- callbacks (GUI -> agent, over Ray RPC) ----

    def _submit(self, msg: CommandMessage) -> None:
        # Awaited, not fired and forgotten: ``RobotAgent`` is an async actor and Ray does
        # not order concurrent calls into one of those, so two quick clicks could reach
        # the loop inverted -- and Stop-then-Start inverted is a robot that runs when it
        # was told to park. ``submit_command`` only appends to the loop's deque, on the
        # ``command`` group's own thread, so this cannot block behind a ``step``.
        import ray

        ray.get(self.controller.submit_command.remote(msg))

    def _on_move(self, *values):
        self._submit(CommandMessage(type=CommandType.MOVE, extra_args={"target": self._pack_target(values)}))

    def _pack_target(self, values):
        """Pack the flat numeric inputs into a ``RobotAction.to_dict()`` following the
        env's ``action_template``: values consumed in template order, each leaf nested
        along its ``path``, intermediate arms becoming ``Arm`` nodes."""
        it = iter(values)
        composite_type = {"left_arm": "Arm", "right_arm": "Arm"}
        root = {"type": "RobotAction"}
        for leaf in self.action_template:
            data = [float(next(it)) for _ in range(leaf["dim"])]
            node = {"type": leaf["type"], "data": [data]}
            if leaf.get("representation"):
                node["representation"] = leaf["representation"]
            if leaf["type"] == "Position":
                # Carry the leaf's delta policy like the env's own unflatten does, so a
                # gripper declared ``allow_relative: False`` stays that way.
                node["allow_relative"] = leaf.get("allow_relative", True)
            d = root
            for p in leaf["path"][:-1]:
                d = d.setdefault(p, {"type": composite_type[p]})
            d[leaf["path"][-1]] = node
        return root

    def _on_start(self, prompt):
        self._submit(CommandMessage(type=CommandType.RUN, extra_args={"prompt": prompt or self.default_prompt}))

    def _on_step(self, prompt):
        self._submit(CommandMessage(type=CommandType.STEP, extra_args={"prompt": prompt or self.default_prompt}))

    def _on_stop(self):
        self._submit(CommandMessage(type=CommandType.STOP))

    def _on_reset(self, prompt):
        self._submit(CommandMessage(type=CommandType.RESET, extra_args={"prompt": prompt or self.default_prompt}))

    # ---- widget renderers (read the snapshot segments directly) ----

    def _image(self, cam: str):
        return read_snapshot_image(self.shm_prefix, cam)

    def _update_joint(self, idx: int):
        import pandas as pd

        # One read for both traces, so a point's state and action came from the same
        # publish rather than from two reads straddling one.
        snap = read_snapshot(self.shm_prefix) or {}
        _, path, off = self._plots[idx]
        self._traces[idx].append(
            (self._component(snap.get("state"), path, off), self._component(snap.get("action"), path, off))
        )
        ts, values, labels = [], [], []
        for t, (sv, av) in enumerate(self._traces[idx]):
            if sv is not None:
                ts.append(t)
                values.append(sv)
                labels.append("state")
            if av is not None:
                ts.append(t)
                values.append(av)
                labels.append("action")
        return pd.DataFrame({"timestamp": ts, "value": values, "label": labels})

    def _update_controller_state(self):
        snap = read_snapshot(self.shm_prefix)
        return CommandType(int(snap["command"])).name if snap else "UNKNOWN"

    def _update_latency(self):
        snap = read_snapshot(self.shm_prefix)
        ms = float(snap["latency_ms"]) if snap else 0.0
        return f"{ms:.0f} ms" if ms > 0 else "-"

    def launch(self, **kwargs):
        """Serve until the server stops. As an actor method this never returns while the
        GUI is up, which is what makes its ref usable as a liveness signal."""
        self.interface.launch(**kwargs)


# ─── Orchestration ──────────────────────────────────────────────────────────


def _launch_gui(args, agent, meta):
    """Place the GUI actor on this node and start it serving.

    ``meta`` is what the readiness barrier answered with -- the layouts of the env that is
    now up -- so there is no second call to make and no window in which the agent could
    have stopped between the two. :class:`GradioGUI` is wrapped here rather than at import
    because this module is imported to read its helpers, and that should not require Ray
    to have seen a class.

    Returns ``(handle, launch_ref)``. The handle has to be *held* by the caller: Ray
    destroys an actor once the last handle to it goes out of scope, and a pending task ref
    is not a handle -- dropping it would take the GUI down as this function returned.
    """
    import ray

    gui = (
        ray.remote(GradioGUI)
        .options(
            num_cpus=_ACTOR_CPUS,
            scheduling_strategy=_local_node_strategy(),
        )
        .remote(agent, meta, args.gui_shm_prefix, args.prompt)
    )

    # Probed before launching, because ``launch`` never returns: an actor that could not
    # be placed, or whose ctor raised on a bad layout or a missing ``gradio``, would
    # otherwise be indistinguishable from a GUI that came up fine and is simply not
    # being visited.
    ray.get(gui.__ray_ready__.remote(), timeout=args.startup_timeout)

    logger.info("Launching GUI at %s:%d ...", args.gradio_host, args.gradio_port)
    return gui, gui.launch.remote(server_name=args.gradio_host, server_port=args.gradio_port)


async def _drive(args, agent, startup, snapshot_callback) -> Dict[str, Any]:
    """Run the loop in the agent actor, and put the GUI actor on it once its env is up."""
    # The loop builds the env inside the actor, on the ``control`` group's single thread --
    # the only one allowed to touch it, which is what a GL context requires. No
    # ``exit_when_idle``: unlike eval it parks when an episode ends and waits for the next
    # command. No command rate either: the loop reads that off the env it just built
    # (``env.fps``), so the rate is configured in one place, the env ctor.
    loop_task = asyncio.ensure_future(
        _await_ref(agent.loop.remote(args.prompt, seed_commands=startup, snapshot_callback=snapshot_callback))
    )

    # Nothing may be asked of the agent until that env exists, and one call is both the
    # barrier and the answer -- it returns the layouts of the env it waited for, so there
    # is no second RPC and no gap in which a ``shutdown`` could land between them. Raced
    # against the loop rather than simply awaited, so a loop that died building the world
    # (a bad ``--env_config``, an EGL failure, a robot that will not connect) surfaces
    # *its* error here instead of as this barrier's timeout. It answers on the ``gui``
    # group while ``control`` is still building, which is what makes ``--startup_timeout``
    # a real deadline.
    logger.info("Waiting for env %s to come up ...", args.controller)
    ready = asyncio.ensure_future(_await_ref(agent.wait_until_ready.remote(timeout=args.startup_timeout)))
    done, _ = await asyncio.wait({loop_task, ready}, return_when=asyncio.FIRST_COMPLETED)
    if loop_task in done:
        ready.cancel()
        await loop_task
        raise RuntimeError("the agent loop exited before it was ever ready.")
    meta = await ready
    logger.info("Env is up; agent loop running.")

    if not args.gui:
        logger.info("Headless: agent loop running, Ctrl+C to stop.")
        return await loop_task

    # ``gui`` is bound and not used again on purpose -- see ``_launch_gui``: this frame is
    # what keeps the actor alive for the length of the run.
    gui, gui_ref = _launch_gui(args, agent, meta)  # noqa: F841
    gui_task = asyncio.ensure_future(_await_ref(gui_ref))

    # A GUI that stops -- a taken port, a dead server, a killed actor -- takes the
    # controller with it rather than leaving a robot driven by nobody. ``shutdown`` is
    # what the loop watches, so the run ends the way Ctrl+C ends it, same teardown.
    done, _ = await asyncio.wait({loop_task, gui_task}, return_when=asyncio.FIRST_COMPLETED)
    if gui_task in done:
        try:
            await gui_task
            logger.error("the GUI stopped; taking the controller down with it")
        except BaseException:  # noqa: BLE001
            logger.exception("the GUI stopped; taking the controller down with it")
        agent.shutdown.remote()
    else:
        gui_task.cancel()
    return await loop_task


def run(args):
    _setup_logging()

    import ray

    from ..agents import RobotAgent
    from ..environments.robot import sweep_stale
    from ..serving.client import HttpClient

    startup = _startup_commands(args)

    # This process is the driver and nothing else: no env, no GUI, no event loop the robot
    # depends on -- it holds the two handles and waits. ``ray.init`` joins an existing
    # cluster when ``RAY_ADDRESS`` says so and starts a local one otherwise; either way
    # the node the actors pin to is *this* node.
    ray.init()
    logger.info("Ray node: %s", ray.get_runtime_context().get_node_id())

    # ``--env_config`` reaches the env ctor verbatim and is the *only* channel to it, so
    # an env declares its own parameters and a typo is the ctor's own ``TypeError``.
    # ``--infer_interval`` / ``--latency_window`` are deliberately not among them: they
    # steer Real-Time Chunking, which lives in the agent, so they are its ctor args below.
    env_config = _load_env_config(args.env_config)
    if args.data_path:
        assert "data_reader" not in env_config, (
            "--data_path and --env_config both name a REPLAY reader. The CLI builds it "
            "now, so drop the env_config key."
        )
        env_config["data_reader"] = _build_replay_reader(args)

    # The agent gets a process of its own, exactly as eval gives it one (see
    # :func:`~rynn_scale.evaluation.evaluator.episode_agent`); the class is plain and is
    # wrapped here. That wrapping is what turns its ``ray.method`` groups into real
    # threads: ``control`` owns the env and the loop that blocks in ``step``, while
    # ``command`` and ``gui`` stay answerable, so a button press and the readiness barrier
    # are answered mid-step. ``max_restarts=0`` because a restarted controller would come
    # back with a robot in an unknown pose -- a failure here ends the run.
    agent = (
        ray.remote(**RobotAgent.ray_actor_options)(RobotAgent)
        .options(
            num_cpus=_ACTOR_CPUS,
            max_restarts=0,
            scheduling_strategy=_local_node_strategy(),
        )
        .remote(
            HttpClient(args.server_url, timeout=args.request_timeout),
            env_type=args.controller,
            env_config=env_config,
            infer_interval=args.infer_interval,
            latency_window=args.latency_window,
        )
    )
    logger.info("Policy: %s", args.server_url)

    # A partial over a module function, so it ships with the ``loop`` call and runs *in
    # the agent actor* -- this driver only picks the prefix and the cap.
    snapshot_callback = (
        functools.partial(publish_snapshot, args.gui_shm_prefix, max_image_bytes=args.gui_max_image_bytes)
        if args.gui
        else None
    )

    def _on_signal(sig, _frame):
        # Asked, not killed: the loop notices on its next pass, and its ``finally`` closes
        # the env -- a real robot's last commanded pose, its control process and its
        # segments all depend on that running. A second signal is the hard stop.
        # Fire-and-forget because this runs on the main thread, which is sitting in
        # ``asyncio.run`` and must not wait for a reply; ``shutdown`` is on the
        # ``command`` group, so it is taken while ``control`` is still in ``step``.
        logger.info("Shutting down ...")
        agent.shutdown.remote()
        signal.signal(sig, signal.SIG_DFL)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        asyncio.run(_drive(args, agent, startup, snapshot_callback))
    finally:
        # ``ray.shutdown`` takes both actors down (the GUI's only handle went out of scope
        # with ``_drive``'s frame). Then the segments, by name: the publisher was the agent
        # actor, so there is no handle here to close. Ordering is all that matters, and
        # reaching this point means the loop already ran its own ``finally`` -- a signal
        # asks rather than kills, and every path through ``_drive`` awaits the loop.
        ray.shutdown()
        sweep_stale(args.gui_shm_prefix)


def main():
    from transformers import HfArgumentParser

    from ..arguments import ControlArguments

    args = HfArgumentParser(ControlArguments).parse_args_into_dataclasses()[0]
    run(args)


if __name__ == "__main__":
    main()
