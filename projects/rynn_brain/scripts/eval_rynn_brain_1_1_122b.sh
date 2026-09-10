MODEL_PATH=""
SAVE_DIR=""

COMMON_ARGS=(
    --model_path "$MODEL_PATH"
    --save_dir "$SAVE_DIR"
    --prompt_format RynnBrain
    --engine sglang
    --tensor_parallel_size 8
    --fps 2
    --max_frames 512
)

QA_BENCH_ARGS=(
    --benchmarks VSIBench MMSI ERQA RoboSpatial EgoTaskQA EgoTextVQAIndoor OpenXVQA QAEgo4D MindCube
    --image_min_pixels $((16 * 32 * 32))
    --image_max_pixels $((16384 * 32 * 32))
    --video_max_pixels $((24576 * 32 * 32 * 2))
    --max_new_tokens 1024
    --temperature 0.0
)

RYNNBRAIN_BENCH_ARGS=(
    --benchmarks RynnBrainCog RynnBrainLoc
    --image_min_pixels $((16 * 32 * 32))
    --image_max_pixels $((16384 * 32 * 32))
    --video_max_pixels $((24576 * 32 * 32 * 2))
    --max_new_tokens 1024
    --temperature 0.0
)

POINTING_BENCH_ARGS=(
    --benchmarks RefSpatial ShareRobot
    --image_min_pixels $((1024 * 32 * 32))
    --image_max_pixels $((16384 * 32 * 32))
    --max_new_tokens 1024
    --temperature 0.0
)

GENERAL_BENCH_ARGS=(
    --benchmarks AI2D ChartQA DocVQA MVBench RealWorldQA InfoVQA VideoMME
    --image_min_pixels $((16 * 32 * 32))
    --image_max_pixels $((16384 * 32 * 32))
    --video_max_pixels $((24576 * 32 * 32 * 2))
    --max_new_tokens 1024
    --temperature 0.0
)

if [ $RANK -eq 0 ]; then
    ray start --head --port=$MASTER_PORT --include-dashboard=false
else
    ray start --address="$MASTER_ADDR:$MASTER_PORT"
fi

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port 16666 \
    --node_rank $RANK \
    /dev/null

if [ $RANK -eq 0 ]; then
    python -m rynn_scale.api.eval ${COMMON_ARGS[@]} ${QA_BENCH_ARGS[@]}
    python -m rynn_scale.api.eval ${COMMON_ARGS[@]} ${RYNNBRAIN_BENCH_ARGS[@]}
    python -m rynn_scale.api.eval ${COMMON_ARGS[@]} ${POINTING_BENCH_ARGS[@]}
    python -m rynn_scale.api.eval ${COMMON_ARGS[@]} ${GENERAL_BENCH_ARGS[@]}
fi

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port 16666 \
    --node_rank $RANK \
    --rdzv_conf="timeout=360000,join_timeout=360000" \
    /dev/null
