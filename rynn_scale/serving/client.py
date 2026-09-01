"""Unified policy entry point for the agent loop.

Two entry points, one per kind of host. :meth:`~InferenceClient.generate_async`
dispatches an inference and hands back an awaitable for its chunk -- and the dispatch is
an ordinary synchronous call, so the request leaves at the moment its caller decided to
ask rather than whenever that caller's event loop next happens to suspend. That is what
the agent loop (:mod:`rynn_scale.agents`) needs: it is async and transport-agnostic, many
episodes overlap their (remote) inference on one event loop, and each one plays actions
in the gap between asking and being answered. :meth:`~InferenceClient.generate` is the
blocking form, for a host that owns its thread and has nothing to interleave with
(on-board / embedded); it is **not** usable from an agent, which runs inside an event
loop -- a Serve handle's sync path raises there. The concrete transports
(proposal §4.4/§4.5):

  * ``RayServeClient`` -- in-cluster ``DeploymentHandle`` to the ``InferenceServer``
    (loopback gRPC + plasma, no HTTP); the ``DeploymentResponse`` (an ObjectRef)
    it returns is awaited later, so env stepping and remote inference overlap.
    Used by eval.
  * ``HttpClient``     -- ``POST /generate`` against a server someone else is already
    running (``api/serve.py``). For a caller with no Ray cluster of its own: the deploy
    controller (``api/control.py``) drives the robot from the machine it is wired to and
    reaches the policy by URL.

Heavy deps (ray / requests) are imported lazily inside each transport so this module
stays light.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from .requests import VLAInferenceRequest


class InferenceClient(ABC):
    @abstractmethod
    def generate(self, req: VLAInferenceRequest) -> Any:
        """Run one inference, blocking until its action chunk is there.

        For a host that owns its thread and has nothing to interleave with. Not usable
        from an agent: both in-tree loops (eval ``rollout``, deploy ``loop``) are
        coroutines, and a Serve handle's sync path refuses to run inside a running event
        loop (``RuntimeError: Sync methods should not be called from within an asyncio
        event loop``). Those want :meth:`generate_async`.
        """

    @abstractmethod
    def generate_async(self, req: VLAInferenceRequest) -> Any:
        """Dispatch one inference **now**; return an awaitable for its action chunk.

        ``_async`` names what this is *for* -- a host running an event loop -- not how it
        is written: it is a plain ``def``, and that is the whole contract. By the time it
        returns, the request has left this process (or the work is queued on the thread
        that will do it), so a caller that launches an inference and then goes off to do
        something else is not charged for the dispatch.

        The agent loop is that caller, and it needs the guarantee rather than merely the
        tendency: an RTC chunk is aligned to the step inference was *launched* at, so a
        dispatch that slid to wherever the loop next suspends would be latency charged to
        a policy that never saw it, silently (see
        :class:`~rynn_scale.agents.robot.RobotAgent`). No task-creation API can give
        that -- ``ensure_future`` and ``create_task`` only schedule a coroutine, and
        Python 3.12's eager-start task merely runs it to its first suspension, which is
        what ``await asyncio.sleep(0)`` already buys. The guarantee has to come from the
        dispatch being a call, which is why this must not become ``async def``.

        A policy's own failure arrives through the awaitable; only a transport that
        cannot dispatch at all raises here.
        """


class RayServeClient(InferenceClient):
    """In-cluster transport: a Ray Serve ``DeploymentHandle`` to the ``InferenceServer``.

    Loopback gRPC + plasma (no HTTP); ``await``ing the ``DeploymentResponse``
    (backed by an ObjectRef) lets other episodes run while this one's inference
    is in flight -- the source of eval concurrency.

    This process owns the only client->server hop there is, and that is deliberate:
    a *replica* that both takes Ray traffic and makes downstream Ray calls
    segfaults (ray#50802, see :mod:`rynn_scale.serving.server`), so the
    orchestration lives on the caller's side, where the server is a leaf.

    ``self._handle.generate`` below is the *deployment's* method, which happens
    to share a name with this class's -- not a recursive call.
    """

    def __init__(self, handle):
        self._handle = handle

    def generate(self, req: VLAInferenceRequest) -> Any:
        # ``DeploymentResponse.result()`` is the blocking twin of awaiting it.
        return self._handle.generate.remote(req).result()

    def generate_async(self, req: VLAInferenceRequest) -> Any:
        # ``.remote()`` *is* the synchronous dispatch
        # :meth:`InferenceClient.generate_async` promises: it hands the request to the
        # handle's router, which runs on an event loop of its own thread
        # (``RAY_SERVE_RUN_ROUTER_IN_SEPARATE_LOOP``, on by default), and returns the
        # ``DeploymentResponse`` to await. So the request travels without this process's
        # event loop yielding at all.
        return self._handle.generate.remote(req)


class HttpClient(InferenceClient):
    """Out-of-cluster transport: POST to a running ``rynn_scale.api.serve`` server.

    The transport for a host that has a policy *somewhere else* and no reason to join
    its Ray cluster -- the deploy controller (:mod:`rynn_scale.api.control`), which runs
    on the machine the robot is wired to, holds no Ray driver of its own and reaches the
    server by URL. Nothing about the server is different: this is the same
    ``POST /generate`` schema :mod:`rynn_scale.api.client` documents, so the same
    replicas answer eval's in-cluster handles and this.

    The price against :class:`RayServeClient` is the wire format -- each frame is JPEG
    encoded here and decoded there, and the payload is JSON rather than plasma -- which
    is why in-cluster callers should keep using the handle.
    """

    def __init__(self, url: str, *, timeout: float = 120.0, jpeg_quality: int = 95):
        import requests

        self._endpoint = url.rstrip("/") + "/generate"
        self._timeout = float(timeout)
        self._jpeg_quality = int(jpeg_quality)
        # One session, so the TCP (and TLS) handshake is paid once rather than per
        # inference. urllib3's pool behind it is thread-safe, which is what
        # ``generate_async``'s hand-off needs.
        self._session = requests.Session()

    def _payload(self, req: VLAInferenceRequest) -> dict:
        import base64
        import io

        import numpy as np
        from PIL import Image

        def encode(frame) -> str:
            buf = io.BytesIO()
            Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buf, format="JPEG", quality=self._jpeg_quality)
            return base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "text": req.text,
            # Already the ``RobotState.to_dict`` wire form the env produced, so there is
            # nothing to convert: it is JSON as it stands.
            "state": req.state,
            "images": {name: encode(img) for name, img in (req.images or {}).items()},
            "robot_type": req.robot_type,
            "prev_actions": (None if req.prev_actions is None else np.asarray(req.prev_actions).tolist()),
            "delay_steps": int(req.delay_steps),
            "num_steps": int(req.num_steps),
        }

    def generate(self, req: VLAInferenceRequest) -> Any:
        resp = self._session.post(self._endpoint, json=self._payload(req), timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    def generate_async(self, req: VLAInferenceRequest) -> Any:
        # Encoding and the POST both block, so they go to a thread -- and they go there
        # *now*: ``run_in_executor`` queues the call and hands back its future, which is
        # the dispatch guarantee :meth:`InferenceClient.generate_async` makes, whereas
        # ``asyncio.to_thread`` is itself a coroutine, so through it the hand-off would
        # not happen until the awaiting task was first scheduled.
        return asyncio.get_running_loop().run_in_executor(None, self.generate, req)
