import json
import os

from ..registry import BENCHMARK_REGISTRY
from .base import BaseBenchmark


def make_prompt(question, label_answer, pred_answer):
    prompt = f"###Question: {question} ###Label Answer: {label_answer} ###Predicted Answer: {pred_answer}"
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Please judge whether the predicted answer is correct or not according to the given question and labeled answer. You need to consider the answer from two perspectives: accuracy and completeness. Output ###Judge: True only when the predicted answer is accurate and complete; otherwise, output ###Judge: False",
        },
        {
            "role": "user",
            "content": "###Question: I need you to clean the dust on the red-marked line in this area. How many tables and chairs should you move before cleaning? ###Label Answer: 0 tables and 1 chair. ###Predicted Answer: One chair",
        },
        {"role": "assistant", "content": "###Judge: False"},
        {
            "role": "user",
            "content": "###Question: If you are now standing outside the restroom stall door away from the sink. Which edge will the door rotate around if you move forward? ###Label Answer: Using the left side edge as the axis. ###Predicted Answer: The door will rotate around the left edge",
        },
        {"role": "assistant", "content": "###Judge: True"},
        {
            "role": "user",
            "content": "###Question: If you start from the position of the backpack next to the sofa and keep moving towards the direction of the bicycle. What would happen if you stop immediately after hitting the bicycle? What if you don't stop immediately but continue moving forward? ###Label Answer: If you stop immediately after hitting the bicycle, the bicycle will sway slightly but won't fall over. If you continue moving forward, the bicycle will tilt towards the window until it leans against it. ###Predicted Answer: If you stop immediately after hitting the bicycle, you would likely stabilize the bicycle and yourself. If you don't stop immediately but continue moving forward, you might knock over the bicycle and potentially cause damage or injury",
        },
        {"role": "assistant", "content": "###Judge: False"},
        {
            "role": "user",
            "content": "###Question: I took some documents from the drawer in this room earlier, please help me check if there are any drawers that haven't been pushed in completely. ###Label Answer: No. ###Predicted Answer: All drawers appear to be closed properly",
        },
        {"role": "assistant", "content": "###Judge: True"},
        {"role": "user", "content": prompt},
    ]
    return messages


def make_prompt_multi_granularity(question, label_answer, half_label_answer, pred_answer):
    prompt = f"###Question: {question} ###5 Score Answer: {label_answer} ###3 Score answer: {half_label_answer} ###Predicted Answer: {pred_answer}"

    question1 = "If you pick up the footstool between the sofa and the TV and flip it 180° upside down, which objects in the scene will be affected and how?"
    label_answer1 = "Three handles and one remote control will fall onto the carpet."
    half_label_answer1 = "The remote control will fall onto the carpet"
    pred_answer1 = "remote control will fall onto the carpet"
    score1 = "3"

    question2 = "If you pick up the footstool between the sofa and the TV and flip it 180° upside down, which objects in the scene will be affected and how?"
    label_answer2 = "Three handles and one remote control will fall onto the carpet."
    half_label_answer2 = "The remote control will fall onto the carpet"
    pred_answer2 = "handles"
    score2 = "2"

    question3 = "If you pick up the footstool between the sofa and the TV and flip it 180° upside down, which objects in the scene will be affected and how?"
    label_answer3 = "Three handles and one remote control will fall onto the carpet."
    half_label_answer3 = "The remote control will fall onto the carpet"
    pred_answer3 = "handles and control will fall "
    score3 = "4"

    question4 = "Which area in this room is better for studying? Why?"
    label_answer4 = "The U-shaped table area, which has matching chairs and good lighting"
    half_label_answer4 = "The U-shaped table area"
    pred_answer4 = "may be table area"
    score4 = "1"

    question5 = "Which area in this room is better for studying? Why?"
    label_answer5 = "The U-shaped table area, which has matching chairs and good lighting"
    half_label_answer5 = "The U-shaped table area"
    pred_answer5 = "Over there by the sofa."
    score5 = "0"

    question6 = "Which area in this room is better for studying? Why?"
    label_answer6 = "The U-shaped table area, which has matching chairs and good lighting"
    half_label_answer6 = "The U-shaped table area"
    pred_answer6 = "may be table area and chairs"
    score6 = "4"

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Please score the predicted answer according to the given question and huamn labeled the 5-score answer, and the 3-score answer. 0 score represents completely wrong, 5 scores represents completely correct, and 3 scores represents partially correct. Please refer to them to score the predicted answer: [0, 1, 2, 3, 4, 5]. You need to consider the answer from two perspectives: accuracy and completeness. Output ###Judge:",
        },
        {
            "role": "user",
            "content": f"###Question: {question1} ###5 Score Answer: {label_answer1}. ###3 Score answer: {half_label_answer1} ###Predicted Answer: {pred_answer1}",
        },
        {"role": "assistant", "content": f"###Judge: {score1}"},
        {
            "role": "user",
            "content": f"###Question: {question2} ###5 Score Answer: {label_answer2}. ###3 Score answer: {half_label_answer2} ###Predicted Answer: {pred_answer2}",
        },
        {"role": "assistant", "content": f"###Judge: {score2}"},
        {
            "role": "user",
            "content": f"###Question: {question3} ###5 Score Answer: {label_answer3}. ###3 Score answer: {half_label_answer3} ###Predicted Answer: {pred_answer3}",
        },
        {"role": "assistant", "content": f"###Judge: {score3}"},
        {
            "role": "user",
            "content": f"###Question: {question4} ###5 Score Answer: {label_answer4}. ###3 Score answer: {half_label_answer4} ###Predicted Answer: {pred_answer4}",
        },
        {"role": "assistant", "content": f"###Judge: {score4}"},
        {
            "role": "user",
            "content": f"###Question: {question5} ###5 Score Answer: {label_answer5}. ###3 Score answer: {half_label_answer5} ###Predicted Answer: {pred_answer5}",
        },
        {"role": "assistant", "content": f"###Judge: {score5}"},
        {
            "role": "user",
            "content": f"###Question: {question6} ###5 Score Answer: {label_answer6}. ###3 Score answer: {half_label_answer6} ###Predicted Answer: {pred_answer6}",
        },
        {"role": "assistant", "content": f"###Judge: {score6}"},
        {"role": "user", "content": prompt},
    ]

    return messages


@BENCHMARK_REGISTRY.register()
class ECBench(BaseBenchmark):
    def load_data(self, data_root):
        json_path = os.path.join(data_root, "ECbench_qa.json")
        with open(json_path, "r", encoding="utf-8") as file:
            data_list = json.load(file)

        video_folder = os.path.join(data_root, "rgb_video")

        data_dict = {}
        idx = 0
        for data in data_list:
            source_path = data["source_dataset"]
            if data["video_source"] == "scannet":
                source_path = data["video_source"]
            video_path = os.path.join(video_folder, source_path, f"{data['video_name']}.mp4")

            try:
                task_types = json.loads(data["dimensions"].replace("'", '"'))
            except json.decoder.JSONDecodeError:
                task_types = [data["dimensions"]]
            task_types = [task_type.replace("_", " ") for task_type in task_types]

            data_dict[idx] = {
                "videos": [video_path],
                "task_type": task_types,
                "scene_type": data["scene_type"],
                "question": data["question_en_v2.2"],
                "ground_truth": data["answer_en_v2.2"],
                "half_label_answer": data["Partially correct answer"],
            }
            idx += 1

        return data_dict

    def generate_instruction(self, data_id):
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        system_prompt = "You are moving in an indoor environment. \
                The image sequence is the scene you just saw. \
                You are now staying at the last frame of the video. \
                Please answer the question with one word or one sentence, as concise and accurate as possible.\n"

        contents = [
            {"type": "video", "video": meta_data["videos"][0]},
            {"type": "text", "text": system_prompt + f"The question: {question}"},
        ]
        instruction = [{"role": "user", "content": contents}]

        return instruction

    async def process_response(self, data_id, response):
        return response

    async def get_matching_score(self, data_id, prediction):
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        label_answer = meta_data["ground_truth"]
        half_label_answer = meta_data["half_label_answer"]

        if half_label_answer == 0:
            messages = make_prompt(question, label_answer, prediction)
        elif isinstance(half_label_answer, str):
            messages = make_prompt_multi_granularity(question, label_answer, half_label_answer, prediction)

        try:
            completion = await self.openai_client.chat.completions.create(
                model="gpt-4o-0806",
                messages=messages,
                max_completion_tokens=100,
            )
            response = completion.choices[0].message.content
            response = response.split("###Judge:")[1].strip()

            if half_label_answer == 0:
                if response == "True":
                    score = 1
                else:
                    score = 0
            elif isinstance(half_label_answer, str):
                score = float(int(response) / 5)

        except Exception:
            print(f"Failed to process item with question: {question}")
            print(f"Prompt messages sent to model: {messages}")
            score = 0

        return score * 100

    def compute_metrics(self, results):
        return self._summarize_scores(results, category_key="task_type")
