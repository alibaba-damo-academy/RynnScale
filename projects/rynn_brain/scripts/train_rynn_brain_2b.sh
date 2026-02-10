#!/bin/bash
if [[ -n $DEBUG && $DEBUG -eq 1 ]]; then
    WORLD_SIZE=1
    NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
    MASTER_ADDR="127.0.0.1"
    MASTER_PORT=16666
    RANK=0
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"


MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct
OUTPUT_DIR=work_dirs/rynn_brain/2b
DATA_PATH=/path/to/annotations.jsonl


DATA_ARGS=(
    --data_path $DATA_PATH
    --model_max_length 16384
    --mm_max_length 10240
    --fps 2
    --max_frames 512
    --micro_batch_size 8
    --gradient_accumulation_steps 1
    --num_train_epochs 1
)

OPTIMIZER_ARGS=(
    --learning_rate 1e-6
    --weight_decay 0.0
    --warmup_ratio 0.03
    --lr_scheduler_type "cosine"
)

TRAINING_ARGS=(
    --deepspeed scripts/zero1.json
    --gradient_checkpointing True
    --bf16 True
    --fp16 False
    --dataloader_num_workers 8
    --decoder_load_balancing True
    --loss_reduction_scope sequence
    --average_tokens_across_devices True
)

LOG_ARGS=(
    --output_dir $OUTPUT_DIR
    --logging_steps 1
    --report_to tensorboard
    --save_strategy "steps"
    --save_steps 1000
    --save_total_limit 2
)

set -x

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $RANK \
    --rdzv_conf="timeout=7200,join_timeout=7200" \
    -m rynn_scale.api.train \
    --model_path $MODEL_PATH
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOG_ARGS[@]}
