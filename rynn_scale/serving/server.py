"""The inference server: one deployment, preprocessing and inference in one replica.

``generate`` takes one request or a list of them, fans a list out
per request, and routes each through ``process`` (in-process) to the
``@serve.batch``-ed ``infer`` -- so the batch a forward pass runs on is composed by
Serve's queue across all callers, not by whoever passed the longest list. ``sglang``
is the exception: it owns a continuous-batching scheduler, so it is submitted one
request at a time (``_generate_sglang`` says why). This used to be three deployments
-- ``Processor`` x N (CPU) and ``Model`` x GPUs behind a ``Gateway`` whose whole body
was ``await model.infer.remote(processor.process.remote(req))``, the canonical Serve
model-composition shape -- and that shape **segfaults**: a process that both
receives Ray tasks at a high rate *and* issues downstream Ray calls double-frees
a Python thread state inside the core worker's io thread
(`ray#50802 <https://github.com/ray-project/ray/issues/50802>`_, open; PC in
``tstate_delete_common`` / ``take_gil``, no Python frame, thread ``worker.io``).
Under eval load the Gateway died after 9-230 requests, which capped a
500-episode LIBERO run at 32 episodes and killed a 400-sample ERQA run at 78.
Only one process may hold this hop, and it has to be one that does not also
serve Ray traffic -- i.e. the client. Fusing is what leaves the *client*
contract untouched: callers still hold a single ``DeploymentHandle`` and call
``generate.remote(req)``.

It buys throughput and spends latency. Throughput, measured saturated (real
server, fake clients, 4 replicas, 180s):

===========================  =========  =========  ======
workload                     split      fused      delta
===========================  =========  =========  ======
vla                          2090 req   2160 req   +3.3%
hf + one 1920x360 image        980 req   1170 req   +19%
===========================  =========  =========  ======

because the split topology paid a cross-process transfer of the **preprocessed**
payload on every request (multi-MB ``model_inputs`` for vla, ``pixel_values`` for a
VLM), which costs more than the preprocessing it parallelized -- preprocessing is
~24ms of a ~350ms vla request, and the GPU is the bottleneck either way (the
Processor replicas ran at 7% of their capacity).

Latency is the other side of that: ``process`` shares the replica's event loop, so
a request waits behind the other in-flight requests' preprocessing, and there is no
``num_processor_workers`` any more to give preprocessing capacity of its own.
Measured on the full 500-episode LIBERO eval, ``generate`` p50 394ms against the
split topology's 24ms (process) + 302ms (infer). LIBERO is latency-bound, not
throughput-bound -- the GPUs idle at 0-17% while MuJoCo renders and each episode
waits on its own inference before it steps -- so it is the *latency* that shows:
25m56s against 23m34s, same score (93.6% both).

Handing ``process`` to a ``ThreadPoolExecutor`` is the obvious answer to that and it
**was tried and did not pay**: 96 episodes took 4m07s on 4 threads against 3m50s
inline, for a p50 of 364ms against 381ms. Preprocessing is only ~24ms of a ~350ms
request, and the threads then contend for the GIL with the decode loop's ten
sequential forward passes. It is worth revisiting only where preprocessing is *big*
-- a VLM sampling ``max_frames=180`` of video, where one ``process`` outweighs one
inference (untested). Re-splitting into two deployments is only safe once ray#50802
is fixed.
"""

import asyncio
import base64
import functools
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from fastapi import FastAPI
from ray import serve

from ..utils.logging import LogLevel, get_logger, set_verbosity
from ..utils.processing import decode_image_bytes
from .requests import VLAInferenceRequest, VLMInferenceRequest

logger = get_logger(__name__)

app = FastAPI(title="RynnScale Inference Server")


def _decode_image(value: Any) -> np.ndarray:
    """Decode one camera frame into an ``HxWx3`` RGB uint8 array.

    Accepts a base64-encoded JPEG/PNG (str) or an already-decoded nested pixel
    list / array. Base64 goes through PIL; lists are taken as-is.
    """
    if isinstance(value, str):
        try:
            return decode_image_bytes(base64.b64decode(value))
        except Exception as exc:
            raise ValueError("failed to decode base64 image") from exc
    return np.asarray(value, dtype=np.uint8)


@serve.ingress(app)
class InferenceServer:
    """One replica: build the wrapper once, preprocess, then batch-infer, and answer
    HTTP on the same replica.

    Not a ``@serve.deployment`` itself: its own :meth:`build` wraps it, so every
    replica option lives in one place instead of half here and half in
    ``.options()``. The HTTP routes are on *this* class rather than an ingress
    deployment in front of it, because an ingress forwarding
    ``await server.generate.remote(req)`` would be the hop described above. HTTP is
    opt-in from the deployer's side: ``api/serve.py`` runs it under a
    ``route_prefix``, while eval and ``api/control.py`` run it with
    ``route_prefix=None`` and reach ``generate`` through a ``DeploymentHandle``, which
    works regardless of the ASGI wrapper.

    Engines:

      * ``vla``             -- ``wrapper.process`` (tokenize + image preprocessing)
        plus the opaque state / RTC fields ``_infer_vla`` needs.
      * ``hf`` / ``sglang`` -- VLM preprocessing (chat template -> image/video
        load+process -> ``process_text``). ``sglang`` additionally lowers the
        processor output into the ``input_ids`` / ``image_data`` / ``video_data``
        shape ``sgl.Engine.async_generate`` expects.
    """

    @classmethod
    def build(
        cls,
        engine: str = "vla",
        model_type: Optional[str] = None,
        model_path: Optional[str] = None,
        dtype: str = "bfloat16",
        attn: str = "flash_attention_2",
        num_model_replicas: int = 1,
        max_running_requests: int = 8,
        batch_wait_timeout_s: float = 0.02,
        model_num_gpus: float = 1,
        max_replicas_per_node: Optional[int] = None,
        sampling_params: Optional[dict] = None,
        parallel_params: Optional[dict] = None,
        processing_params: Optional[dict] = None,
        log_level: Optional[str] = None,
    ):
        """Bind ``cls`` as a single Serve deployment, ready for ``serve.run``.

        ``num_model_replicas`` / ``model_num_gpus`` (= tp*pp) / ``max_replicas_per_node``
        typically come from
        :func:`rynn_scale.evaluation.placement.model_deployment_kwargs`.
        ``sampling_params`` / ``parallel_params`` steer the hf/sglang generate backends;
        ``processing_params`` steers VLM image/video preprocessing.

        A classmethod so the class being deployed is the one it is called on -- a
        subclass deploys itself. The HTTP route lands on the *same* replica that runs
        the model; an ingress deployment forwarding to this one would be the hop this
        module exists to avoid.

        One deployment, on purpose -- see the module docstring for why composing two
        behind a forwarding third segfaults (ray#50802).
        """
        assert model_type and model_path, "model_type/model_path required"

        # Lazy so importing this module does not pull in the model stack. The
        # partial is the picklable, zero-arg factory ``__init__`` calls per replica.
        from ..inference_wrappers import build_inference_wrapper

        wrapper_builder = functools.partial(
            build_inference_wrapper,
            model_type=model_type,
            model_path=model_path,
            dtype=dtype,
            attn_implementation=attn,
        )

        opts = dict(
            num_replicas=num_model_replicas,
            ray_actor_options={"num_gpus": model_num_gpus},
            # Serve admits at most ``max_ongoing_requests`` calls to a replica at once,
            # and its default is 5 -- below which ``@serve.batch`` can never fill a batch
            # of ``max_running_requests`` no matter what the flag says (Serve warns:
            # "max_batch_size (8) * max_concurrent_batches (1) is larger than
            # max_ongoing_requests (5)", and that line was in every eval log). Twice the
            # batch size so a second batch's worth of requests is already admitted and
            # queued when the current forward returns, and ``wait_for_batch`` can drain a
            # full batch without waiting out ``batch_wait_timeout_s`` again. That is all
            # the headroom buys for vla/hf: ``process`` and the forward are synchronous on
            # this replica's single event loop thread, so nothing is preprocessed while the
            # GPU runs -- only ``sglang``, whose generate is awaited, actually overlaps.
            # Never below Serve's own default, so ``max_running_requests=1`` (batching off,
            # one forward per request) still keeps requests queued behind the running one.
            max_ongoing_requests=max(5, 2 * max_running_requests),
        )
        if max_replicas_per_node is not None:
            opts["max_replicas_per_node"] = max_replicas_per_node

        return (
            serve.deployment(cls)
            .options(**opts)
            .bind(
                wrapper_builder,
                engine,
                sampling_params,
                parallel_params,
                processing_params,
                max_running_requests,
                batch_wait_timeout_s,
                log_level,
            )
        )

    def __init__(
        self,
        wrapper_builder: Callable[[], object],
        engine: str = "vla",
        sampling_params: Optional[dict] = None,
        parallel_params: Optional[dict] = None,
        processing_params: Optional[dict] = None,
        max_batch_size: int = 8,
        batch_wait_timeout_s: float = 0.02,
        log_level: Optional[str] = None,
    ):
        if log_level is not None:
            set_verbosity(LogLevel(log_level))

        # wrapper_builder is a picklable, zero-arg factory (built in ``build``).
        # One wrapper serves both halves -- the split topology built two per
        # replica pair, the preprocessing one only for its processor metadata.
        self.engine = engine
        self.wrapper = wrapper_builder()
        self.sampling_params = dict(sampling_params or {})
        self.processing_params = processing_params or {}

        if engine == "vla":
            _ = self.wrapper.model  # trigger GPU weight load
        elif engine == "hf":
            _ = self.wrapper.model  # trigger GPU weight load
            if self.sampling_params.get("temperature", None) == 0.0:
                self.sampling_params["do_sample"] = False
        elif engine == "sglang":
            # sglang owns the GPU / weights; the wrapper is used here for its
            # ``model_path`` and (in ``_process_vlm``) its processor metadata.
            import sglang as sgl

            self.sgl_engine = sgl.Engine(
                model_path=self.wrapper.model_path,
                mem_fraction_static=0.8,
                **(parallel_params or {}),
            )
        else:
            raise ValueError(f"Unknown engine: {engine!r} (expected vla/hf/sglang)")

        # Tune the @serve.batch knobs at runtime (decorator sets the defaults).
        self._infer.set_max_batch_size(max_batch_size)
        self._infer.set_batch_wait_timeout_s(batch_wait_timeout_s)

    async def generate(self, reqs):
        if isinstance(reqs, (list, tuple)):
            return list(await asyncio.gather(*(self.generate(r) for r in reqs)))
        if self.engine == "sglang":
            return await self._generate_sglang(reqs)
        return await self._infer(self.process(reqs))

    def _build_request(self, payload: Dict[str, Any]):
        """JSON -> request object. ``self.engine`` picks the schema, so the payload
        and the backend cannot disagree."""
        if self.engine == "vla":
            images = {k: _decode_image(v) for k, v in (payload.get("images") or {}).items()}
            prev = payload.get("prev_actions")
            return VLAInferenceRequest(
                text=payload["text"],
                state=payload["state"],
                images=images,
                robot_type=payload["robot_type"],
                prev_actions=None if prev is None else np.asarray(prev, dtype=np.float32),
                delay_steps=int(payload.get("delay_steps", 0)),
                num_steps=int(payload.get("num_steps", 10)),
            )
        return VLMInferenceRequest(
            conversation=payload["conversation"],
            enable_thinking=bool(payload.get("enable_thinking", False)),
        )

    @app.post("/generate")
    async def http_generate(self, payload: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Any:
        if isinstance(payload, list):
            return await self.generate([self._build_request(p) for p in payload])
        return await self.generate(self._build_request(payload))

    def process(self, req) -> dict:
        if self.engine == "vla":
            return {
                "model_inputs": self.wrapper.process(
                    text=req.text,
                    images=req.images,
                    state=req.state,
                    robot_type=req.robot_type,
                ),
                "state": req.state,
                "robot_type": req.robot_type,
                "prev_actions": req.prev_actions,
                "delay_steps": req.delay_steps,
                "num_steps": req.num_steps,
            }
        return self._process_vlm(req)

    def _process_vlm(self, req) -> dict:
        prompt = self.wrapper.apply_chat_template(req.conversation, enable_thinking=req.enable_thinking)

        images, videos = [], []
        for message in req.conversation:
            for content in message["content"]:
                if content["type"] == "image":
                    images.append(content["image"])
                elif content["type"] == "video":
                    videos.append(content["video"])

        image_inputs, video_inputs = {}, {}
        if len(images):
            images = self.wrapper.load_images(images, processing_params=self.processing_params)
            image_inputs = self.wrapper.process_images(images, processing_params=self.processing_params)
        if len(videos):
            videos = self.wrapper.load_videos(videos, processing_params=self.processing_params)
            video_inputs = self.wrapper.process_videos(videos, processing_params=self.processing_params)

        model_inputs = self.wrapper.process_text(
            text=prompt,
            image_inputs=image_inputs,
            video_inputs=video_inputs,
        )

        if self.engine == "sglang":
            image_data, video_data = {"format": "processor_output"}, {"format": "processor_output"}
            for name, value in model_inputs.items():
                if name in self.wrapper.processor.image_processor.model_input_names:
                    image_data[name] = value
                if name in self.wrapper.processor.video_processor.model_input_names:
                    video_data[name] = value
            model_inputs = {
                "input_ids": model_inputs["input_ids"][0].tolist(),
                "image_data": image_data if len(image_data) > 1 else None,
                "video_data": video_data if len(video_data) > 1 else None,
            }

        return {"model_inputs": model_inputs}

    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.02)
    async def _infer(self, items: List[dict]) -> List[dict]:
        logger.debug("infer batch_size=%d (max %d)", len(items), self._infer._get_max_batch_size())
        if self.engine == "vla":
            return self._infer_vla(items)
        if self.engine == "hf":
            return self._infer_hf(items)
        raise NotImplementedError(f"Serve engine={self.engine!r} is not batched")

    def _infer_vla(self, items: List[dict]) -> List[dict]:
        model_inputs = self.wrapper.collate([it["model_inputs"] for it in items])
        device = self.wrapper.model.device
        model_inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in model_inputs.items()}

        # Everything the request carried reaches ``decode``, per batch element wherever
        # it can differ between callers. Nothing is filtered against the wrapper's
        # signature: a wrapper with no use for a field declares it and ignores it, which
        # says so in the one place a reader would look, instead of this dispatcher
        # deciding on its behalf and dropping data without a word. Three such drops lived
        # here -- ``robot_type`` never forwarded at all (so ``get_action_mask`` was dead
        # and sampling ran unmasked against a model trained masked), and ``num_steps`` /
        # ``delay_steps`` taken from ``items[0]`` and served to the whole batch.
        #
        # ``prev_actions`` in particular must stay per element: it used to be passed only
        # when *every* element had one, so a single just-started caller silently stripped
        # Real-Time Chunking from everyone batched with it.
        num_steps = {it["num_steps"] for it in items}
        assert len(num_steps) == 1, (
            f"batched requests disagree on num_steps ({sorted(num_steps)}). It schedules "
            "the single shared forward pass, so unlike the other fields it cannot be "
            "honoured per element -- silently taking one of them is what this replaces."
        )
        decode_kwargs = {
            "num_steps": num_steps.pop(),
            "robot_type": [it["robot_type"] for it in items],
            "prev_actions": [it.get("prev_actions") for it in items],
            "delay_steps": [int(it.get("delay_steps", 0)) for it in items],
        }

        with torch.inference_mode():
            cache = self.wrapper.prefill(model_inputs)
            actions = self.wrapper.decode(model_inputs, cache, **decode_kwargs)

        actions = actions.cpu().float()
        return [
            self.wrapper.post_process(a, state=it["state"], robot_type=it["robot_type"])
            for a, it in zip(actions, items)
        ]

    def _infer_hf(self, items: List[dict]) -> List[dict]:
        device = self.wrapper.model.device
        outputs = []
        for it in items:
            model_inputs = it["model_inputs"].to(device)
            with torch.inference_mode():
                texts = self.wrapper.generate(
                    model_inputs=model_inputs,
                    sampling_params=self.sampling_params,
                )
            outputs.append({"text": texts[0]})
        return outputs

    async def _generate_sglang(self, req) -> dict:
        """One request, preprocessed and submitted the moment it is ready -- deliberately unbatched.

        sglang schedules continuously inside its own engine, so it wants requests as
        early and as independently as it can get them. Behind ``@serve.batch`` it got
        neither: the batched version fanned the list straight back out with
        ``asyncio.gather``, so there was never a shared forward pass to win, while the
        queue in front cost ``batch_wait_timeout_s`` before the first submission, made
        every request wait for the slowest one in its batch (Serve resolves a batch's
        futures only once the whole list returns) and -- with ``max_concurrent_batches``
        at 1 -- refused to submit the next batch until the current one had fully
        drained, holding sglang's scheduler to one lockstep group at a time.
        """
        result = await self.sgl_engine.async_generate(
            **self.process(req)["model_inputs"],
            sampling_params=self.sampling_params,
        )
        return {"text": result["text"]}
