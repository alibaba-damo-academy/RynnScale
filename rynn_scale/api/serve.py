"""HTTP serving entrypoint for a RynnScale VLM/VLA model.

Exposes the same server eval uses
(:class:`~rynn_scale.serving.server.InferenceServer` -- preprocessing, ``@serve.batch``ed
inference *and* the HTTP routes, all in one replica) to out-of-cluster clients. The
routes are on that class rather than on an ingress deployment in front of it: an
ingress that forwarded ``await server.generate.remote(req)`` would be a replica both
taking Ray traffic and issuing downstream Ray calls, which segfaults under concurrent
load (ray#50802 -- see ``serving/server.py``). So this module only parses args and
``serve.run``s it under a route prefix; eval runs the same class with
``route_prefix=None`` and reaches it by handle. ``engine`` selects both the backend and
the request schema, exactly as in eval:

  * ``vla``            -- POST ``/generate`` with ``{text, state, images,
    robot_type, prev_actions?, delay_steps?, num_steps?}``. ``state`` is a
    :meth:`RobotState.to_dict` mapping; ``images`` maps each camera name to a
    base64-encoded JPEG/PNG (or a nested pixel list). The response is the action
    chunk as :meth:`RobotAction.to_dict`.
  * ``hf`` / ``sglang`` -- POST ``/generate`` with ``{conversation,
    enable_thinking?}`` (an OpenAI-style chat list whose content items may be
    text/image/video; media are URLs/paths the server loads). The response is
    ``{"text": ...}``.

Serving knobs mirror :class:`~rynn_scale.arguments.EvaluationArguments`. The
number of Model replicas is set explicitly via ``--num_model_replicas``; each
replica takes ``tensor_parallel_size * pipeline_parallel_size`` GPUs.

    python -m rynn_scale.api.serve --model_path ... --engine vla --port 8000
"""

import logging

import ray
from ray import serve

from ..serving.server import InferenceServer

logger = logging.getLogger("serve")


def run(args):
    logging.basicConfig(level=logging.INFO)

    if not ray.is_initialized():
        ray.init()

    serve.start(http_options={"host": args.host, "port": args.port})
    server = InferenceServer.build(
        engine=args.engine,
        model_type=args.model_type,
        model_path=args.model_path,
        dtype=args.param_dtype,
        attn=args.attn_implementation,
        max_running_requests=args.max_running_requests,
        batch_wait_timeout_s=args.batch_wait_timeout_s,
        sampling_params=args.sampling_params,
        parallel_params=args.parallel_params,
        processing_params=args.processing_params,
        num_model_replicas=args.num_model_replicas,
        model_num_gpus=args.tensor_parallel_size * args.pipeline_parallel_size,
        log_level=args.log_level,
    )
    serve.run(server, route_prefix=args.route_prefix)

    logger.info(
        "Serving engine=%s at http://%s:%d%s (POST /generate) -- Ctrl+C to stop.",
        args.engine,
        args.host,
        args.port,
        args.route_prefix.rstrip("/"),
    )
    try:
        import time

        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Shutting down ...")
        serve.shutdown()
        ray.shutdown()


def main():
    from transformers import HfArgumentParser

    from ..arguments import ServeArguments

    args = HfArgumentParser(ServeArguments).parse_args_into_dataclasses()[0]
    run(args)


if __name__ == "__main__":
    main()
