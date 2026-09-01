import logging

import ray
from transformers import HfArgumentParser

from ..arguments import EvaluationArguments
from ..benchmarks import build_benchmark
from ..evaluation import Evaluator
from ..inference_wrappers import build_inference_wrapper

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(EvaluationArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Join an existing cluster when there is one (``address="auto"``), otherwise
    # start a local one. Eval plans placement cluster-wide (model replicas per
    # GPU, one reserved CPU slot per concurrent episode), so attaching to the real
    # cluster rather than a fresh single-node one is load-bearing.
    if not ray.is_initialized():
        try:
            ray.init(address="auto")
        except ConnectionError:
            ray.init()

    inference_wrapper = build_inference_wrapper(
        model_type=args.model_type,
        model_path=args.model_path,
        dtype=args.param_dtype,
        attn_implementation=args.attn_implementation,
    )

    benchmarks = [
        build_benchmark(
            benchmark,
            prompt_format=args.prompt_format,
            enable_thinking=args.enable_thinking,
            inference_wrapper=inference_wrapper,
        )
        for benchmark in args.benchmarks
    ]

    evaluator = Evaluator(
        args=args,
        inference_wrapper=inference_wrapper,
        benchmarks=benchmarks,
    )
    try:
        evaluator.eval()
    finally:
        # The app comes down with the run, not with the process. Eval attaches to
        # whatever cluster is already there (``address="auto"`` above) and Serve's
        # controller is a *detached* actor, so a deployment left behind outlives this
        # process and goes on holding every GPU the plan handed it -- which the next
        # run then cannot place. This is load-bearing on the failure path in
        # particular: a broken episode now stops the run, so leaving early is an
        # ordinary way to get here rather than an exceptional one.
        try:
            from ray import serve

            serve.shutdown()
        except Exception:  # noqa: BLE001
            # Swallowed rather than raised: from a ``finally`` this would replace
            # whichever error is on its way out, and that one is worth more.
            logger.warning("serve.shutdown() failed on the way out", exc_info=True)
        ray.shutdown()


if __name__ == "__main__":
    main()
