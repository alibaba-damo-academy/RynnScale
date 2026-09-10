#!/bin/bash
set -e

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

MODEL_PATH=""
OUTPUT_DIR=""
DATA_PATH=""

DATA_MIXTURE="[{\"data_type\": \"LiberoDataset\", \"data_path\": \"$DATA_PATH\", \"subsets\": [\"libero_spatial\"]}]"

MODEL_ARGS=(
    --model_path $MODEL_PATH
    --model_type rynn_brain_vla
    --attn_implementation flash_attention_2
)

DATA_ARGS=(
    --data_mixture "$DATA_MIXTURE"
    --action_chunk_size 32
    --use_delta_action True
    --eef_rotation_repr rot_6d
    --micro_batch_size 8
    --gradient_accumulation_steps 1
    --num_train_epochs 15
    --sequence_packing False
)

OPTIMIZER_ARGS=(
    --learning_rate 2e-5
    --lr_scheduler_type cosine
    --warmup_ratio 0.03
    --weight_decay 0.0
)

TRAINING_ARGS=(
    --gradient_checkpointing False
    --param_dtype bfloat16
    --dataloader_num_workers 8
)

LOG_ARGS=(
    --output_dir $OUTPUT_DIR
    --report_to tensorboard
    --logging_steps 1
    --save_strategy steps
    --save_steps 2000
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
    "${MODEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${LOG_ARGS[@]}"
