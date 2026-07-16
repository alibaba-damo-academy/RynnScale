import os
import re
from functools import partial

from transformers import HfArgumentParser
from transformers.trainer_utils import enable_full_determinism, set_seed

from ..arguments import TrainingArguments
from ..datasets import build_dataset
from ..models import build_model, init_weights
from ..ops import cross_entropy_loss
from ..projects import register_projects
from ..training import (
    DataCollator,
    Trainer,
)
from ..utils import logging, oss

logger = logging.get_logger(__name__)


def train():
    register_projects()

    parser = HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    enable_full_determinism(args.seed) if args.full_determinism else set_seed(args.seed)

    if args.output_dir.startswith("oss://"):
        contents = oss.listdir(args.output_dir)
    elif os.path.exists(args.output_dir):
        contents = os.listdir(args.output_dir)
    else:
        contents = []
    resume_from_checkpoint = any(x.startswith("checkpoint-") for x in contents)

    model, processor = build_model(
        model_type=args.model_type,
        model_path=args.model_path,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        vision_encoder_path=args.vision_encoder_path,
        reduced_layers_in_stage_zero=args.reduced_layers_in_stage_zero,
    )

    init_weights(
        model,
        pretrained_model_name_or_path=args.model_path if not resume_from_checkpoint else None,
    )

    model.loss_function = partial(
        cross_entropy_loss,
        loss_reduction_scope=args.loss_reduction_scope,
        loss_implementation=args.loss_implementation,
    )

    # Process Model
    if args.frozen_parameters is not None:
        for name, param in model.named_parameters():
            if any(re.match(pattern, name) for pattern in args.frozen_parameters):
                param.requires_grad_(False)
    frozen_params = [name for name, param in model.named_parameters() if not param.requires_grad]

    logger.info(
        f"Model config: {model.config}\n\n"
        f"Processor: {processor}\n\n"
        f"Model: {model}\n\n"
        f"Frozen parameters: {frozen_params}\n\n"
    )

    train_dataset = build_dataset(args, model_config=model.config, processor=processor)
    logger.info(f"Dataset: {train_dataset}\n\n")

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
