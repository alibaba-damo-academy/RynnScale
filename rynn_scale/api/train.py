import re
from copy import deepcopy
from functools import partial

from transformers import HfArgumentParser

from ..arguments import TrainingArguments
from ..datasets import build_dataset
from ..models import build_model, build_processor, init_weights
from ..ops import cross_entropy_loss
from ..projects import register_projects
from ..training import (
    DataCollator,
    Trainer,
)
from ..utils import logging, storage
from ..utils.determinism import set_seed

logger = logging.get_logger(__name__)


def train():
    register_projects()

    parser = HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed, full_determinism=args.full_determinism)

    contents = storage.listdir(args.output_dir) if storage.exists(args.output_dir) else []
    resume_from_checkpoint = any(x.startswith("checkpoint-") for x in contents)

    train_dataset = build_dataset(args)

    # The processor needs the dataset's schema, so it is built after the dataset
    # and handed to ``build_model`` rather than being rebuilt in there.
    processor_overrides = deepcopy(args.processor_overrides)
    get_schema = getattr(train_dataset, "get_schema", None)
    schema = get_schema() if get_schema is not None else None
    if schema is not None:
        processor_overrides["schema"] = schema

    processor = build_processor(
        model_type=args.model_type,
        model_path=args.model_path,
        processor_overrides=processor_overrides,
    )

    train_dataset.processor = processor

    config_overrides = processor.get_config_overrides()
    config_overrides.update(args.config_overrides)

    model, processor = build_model(
        model_type=args.model_type,
        model_path=args.model_path,
        param_dtype=args.param_dtype,
        attn_implementation=args.attn_implementation,
        config_overrides=config_overrides,
        vision_encoder_path=args.vision_encoder_path,
        reduced_layers_in_stage_zero=args.reduced_layers_in_stage_zero,
        reshard_after_forward=args.reshard_after_forward,
        master_param_dtype=args.master_param_dtype,
        reduce_dtype=args.reduce_dtype,
        processor=processor,
    )

    init_weights(
        model,
        pretrained_model_name_or_path=args.model_path if not resume_from_checkpoint else None,
    )

    model.loss_function = partial(
        cross_entropy_loss,
        loss_reduction_scope=args.loss_reduction_scope,
    )

    # Process Model
    if args.frozen_parameters is not None:
        for name, param in model.named_parameters():
            if any(re.match(pattern, name) for pattern in args.frozen_parameters):
                param.requires_grad_(False)
    frozen_params = [name for name, param in model.named_parameters() if not param.requires_grad]

    logger.info(
        f"Dataset: {train_dataset}\n\n"
        f"Model config: {model.config}\n\n"
        f"Processor: {processor}\n\n"
        f"Model: {model}\n\n"
        f"Frozen parameters: {frozen_params}\n\n"
    )

    data_collator = DataCollator(
        processor=processor,
        sequence_packing=args.sequence_packing,
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        processing_class=processor,
    )

    return trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    train()
