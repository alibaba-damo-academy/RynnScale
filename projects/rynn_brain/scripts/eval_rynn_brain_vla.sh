MODEL_PATH=""
SAVE_DIR=""

ARGS=(
    --model_path "$MODEL_PATH"
    --save_dir "$SAVE_DIR"
    --engine vla
    --attn_implementation flash_attention_2
    --benchmarks LiberoSpatial
    --max_running_requests 4
    --max_concurrent_episodes 64
    --save_rollout True
)

python -m rynn_scale.api.eval ${ARGS[@]}
