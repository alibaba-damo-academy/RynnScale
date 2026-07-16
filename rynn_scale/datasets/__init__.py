from transformers import PretrainedConfig, ProcessorMixin

from ..registry import DATASET_REGISTRY
from . import vlm_datasets


def build_dataset(args, model_config: PretrainedConfig, processor: ProcessorMixin):
    if args.data_mixture is None:
        return DATASET_REGISTRY[args.data_type](
            model_config=model_config,
            processor=processor,
            data_path=args.data_path,
            data_mixture=args.data_mixture,
            model_max_length=args.model_max_length,
            mm_max_length=args.mm_max_length,
            fps=args.fps,
            max_frames=args.max_frames,
            seed=args.seed,
        )
    raise NotImplementedError
