import ray
from transformers import HfArgumentParser

from ..arguments import EvaluationArguments
from ..benchmarks import build_benchmark
from ..evaluation import Evaluator
from ..inference_wrappers import build_inference_wrapper


def main():
    parser = HfArgumentParser(EvaluationArguments)
    args = parser.parse_args_into_dataclasses()[0]

    if ray.is_initialized():
        ray.init(address="auto")
    else:
        ray.init()

    inference_wrapper = build_inference_wrapper(
        model_type=args.model_type,
        model_path=args.model_path,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    benchmarks = [
        build_benchmark(
            benchmark,
            prompt_format=args.prompt_format,
            enable_thinking=args.enable_thinking,
        )
        for benchmark in args.benchmarks
    ]

    evaluator = Evaluator(
        args=args,
        inference_wrapper=inference_wrapper,
        benchmarks=benchmarks,
    )
    evaluator.eval()


if __name__ == "__main__":
    main()
