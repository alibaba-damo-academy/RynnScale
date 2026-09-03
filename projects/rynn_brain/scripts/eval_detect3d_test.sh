#!/bin/bash
# Pipeline test: only run on first 10 images
# export DETECT3D_MAX_IMAGES=30


MODEL_PATH="/path/to/model"

timestamp=$(date +"%Y%m%d_%H%M%S")
# export DETECT3D_VIS_DIR=${save_dir}/vis
save_dir=/path/to/${timestamp}
export DETECT3D_VIS_DIR=${save_dir}/vis

ARGS=(
    --model_type qwen3_5
    --model_path $MODEL_PATH
    --benchmarks Detect3D
    --save_dir "$save_dir"
    --engine hf
    --image_min_pixels $((16 * 32 * 32))
    --image_max_pixels $((16384 * 32 * 32))
    --max_new_tokens 512
    --temperature 0.0
    --attn_implementation sdpa
)

python -m rynn_scale.api.eval ${ARGS[@]}
