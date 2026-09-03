<p align="center">
<img src="https://github.com/alibaba-damo-academy/RynnBrain/blob/main/cookbooks/assets/images/logo.png" style="width: 40%; height: auto;">
</p>

We present RynnBrain, an embodied foundation model grounded in physical reality. The goal of RynnBrain is not just to “observe” the environment, but to anchor its understanding within the physical world through comprehensive egocentric cognition, precise spatiotemporal grounding and real task planning. This systematic upgrade pushes the boundaries of embodied brains, moving them from passive observation toward active, physics-aware reasoning and complex task execution.



## Data Preparation

All data for RynnBrain training generally follows the standard [RynnScale VLM data format](). Besides, for better representing coordinates in different frames, the video was extracted into frames at 2 fps, and the frame index (starts from 0) is added in text format before each frame. For example:

```json
{
    "conversation": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<frame 0>:"},
                {"type": "image", "text": "/path/to/frame_0.png"},
                {"type": "text", "text": "<frame 1>:"}
                {"type": "image", "text": "/path/to/frame_1.png"},
                {"type": "text", "text": "I need to heat food; which object should I use?"}
            ],
        }
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<object> <frame 1>; (100,100),(200,200) </object>"}
            ],
        }
    ]
}
```

For tasks with coordinate inputs (object referring, area referring) or with coordinate outputs (object, area, affordance, trajectory, and grasp pose prediction), the sequence of points are formatted as XML tags. Specifically, there are five types of point sequences in RynnBrain:

**Object** (the top left corner point and bottom right corner point of the bounding box)

`<object> <frame n>; (x1,y1),(x2,y2) </object>`

**Area** (several points in the target area)

`<area> <frame n>; (x1,y1),(x2,y2),...,(xn,yn) </area>`

**Affordance** (the target point for an operation on an object)

`<affordance> <frame n>; (x1,y1) </affordance>`

**Trajectory** (several points on the trajectory)

`<traj> <frame n>; (x1,y1),(x2,y2),...,(xn,yn) </traj>`

**Grasp pose** (the top left corner point and bottom right corner point of the bounding box)

`<grasp_pose> <frame n>; (x1,y1),(x2,y2) </grasp_pose>`

For object and grasp pose,  The x and y coordinates of the points are normalized to 0-1000. For image QA, the frame index part will be discarded.



## Training

We provide some example scripts in the [scripts](./scripts) directory. Please refer to the [documentation](../docs) for more details.

```shell
bash scripts/train_rynn_brain_2b.sh
```



## Evaluation

The following script can be used to reproduce all the results reported in the technical report.

```shell
MODEL_PATH=""
SAVE_DIR=""

export ENDPOINT_URL=""
export OPENAI_API_KEY=""

python -m rynn_scale.api.eval \
    --model_path $MODEL_PATH \
    --benchmarks VSIBench MMSI EgoSchema EgoTaskQA EgoTextVQAIndoor OpenXVQA QAEgo4D MindCube AI2D ChartQA DocVQA MVBench InfoVQA RealWorldQA VideoMME \
    --prompt_format RynnBrain \
    --save_dir $SAVE_DIR \
    --backend hf \
    --num_processor_workers 4 \
    --fps 2 \
    --max_frames 512 \
    --image_min_pixels $((16 * 32 * 32)) \
    --image_max_pixels $((16384 * 32 * 32)) \
    --video_max_pixels $((24576 * 32 * 32 * 2)) \
    --temperature 0.0

python -m rynn_scale.api.eval \
    --model_path $MODEL_PATH \
    --benchmarks RynnBrainCog RynnBrainLoc \
    --prompt_format RynnBrain \
    --save_dir $SAVE_DIR \
    --backend hf \
    --num_processor_workers 4 \
    --fps 2 \
    --max_frames 512 \
    --image_min_pixels $((16 * 32 * 32)) \
    --image_max_pixels $((16384 * 32 * 32)) \
    --video_max_pixels $((24576 * 32 * 32 * 2)) \
    --temperature 0.0

python -m rynn_scale.api.eval \
    --model_path $MODEL_PATH \
    --benchmarks ERQA RoboSpatial \
    --prompt_format RynnBrain \
    --save_dir $SAVE_DIR \
    --backend hf \
    --num_processor_workers 4 \
    --fps 2 \
    --max_frames 512 \
    --image_min_pixels $((16 * 32 * 32)) \
    --image_max_pixels $((16384 * 32 * 32)) \
    --temperature 0.2 \
    --top_p 0.95 \
    --top_k 50

python -m rynn_scale.api.eval \
    --model_path $MODEL_PATH \
    --benchmarks RefSpatial ShareRobot \
    --prompt_format RynnBrain \
    --save_dir $SAVE_DIR \
    --backend hf \
    --num_processor_workers 4 \
    --fps 2 \
    --max_frames 512 \
    --image_min_pixels $((1024 * 32 * 32)) \
    --image_max_pixels $((16384 * 32 * 32)) \
    --temperature 0.2 \
    --top_p 0.95 \
    --top_k 50
```



## RynnBrain-VLA

TBD...
