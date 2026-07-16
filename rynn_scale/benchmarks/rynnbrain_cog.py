import os
import re
import json
from collections import defaultdict
from typing import Any, Dict, List, Union

from .base import BaseBenchmark
from ..registry import BENCHMARK_REGISTRY

TASKS = {
    "object_conigtion": "rynnbrain_object_2000.jsonl",
    "spatial_conigtion": "rynnbrain_spatial_2000.jsonl",
    # "object_conigtion": "rynnbrain_counting_200.jsonl",  # for COT Model
}


@BENCHMARK_REGISTRY.register()
class RynnBrainCog(BaseBenchmark):

    # THINKING_MODE = True
    THINKING_MODE = False

    TYPE_LIST = {
        "Object Cognition": [
            "category",
            "color",
            "material",
            "shape",
            "state",
            "position",
            "function",
            "surface detail",
            "size",
            "counting",
        ],
        "Historical": ["camera_rotation", "camerall_distance"],
        "Present": [
            "relative_direction_to_camera",
            "distance_to_camera",
            "direction_to_camera",
            "direction_to_camera_3",
            "relative_direction_to_camera_2",
            "distance_to_camera_choice_3",
            "distance_to_camera_2",
            "relative_distance_to_object_choice_2",
        ],
        "Future": ["future_direction_to_camera", "future_direction_to_camera_rotate", "future_direction_object_3"],
        "Size": [
            "height_data",
            "tall_choice_3",
            "small_predicate_3",
            "tall_predicate",
            "tall_choice",
            "short_predicate",
            "short_choice",
            "big_predicate",
            "small_predicate",
        ],
        "Distance": ["height_from_ground", "distance_compare", "center_distance"],
        "Position": [
            "above_predicate",
            "above_choice",
            "below_predicate",
            "below_choice",
            "between",
            "directly_above",
        ],
    }
    SPATIAL_TYPES = ["Historical", "Present", "Future", "Size", "Distance", "Position"]

    def _load_jsonl_file(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"JSONL file not found: {path}")
        data_list = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
                data_list.append(obj)
        return data_list

    def load_data(self, data_root: str) -> Dict[Union[str, int], Any]:
        """
        Load data from jsonl files.
        """
        data_dict = {}
        
        for task_name, json_path in TASKS.items():
            data_path = os.path.join(data_root, json_path)
            data_folder = os.path.join(data_root, "data")
            
            # print('data_path', data_path)
            data_list = self._load_jsonl_file(data_path)
            # print('len(data_list)',len(data_list))
            for i, data in enumerate(data_list):
                # if i<4:
                #     continue
                # if i>0:
                #     break
                data_id = data.get("id", 0)
                data_id = f"{task_name}_{data_id}"
                task_type = data.get("task_type", "unknown")
                conversation = data.get("conversation", [])

                # resolve data
                for msg in conversation:
                    if msg["role"] == "assistant":
                        # print('msg["content"]', msg["content"])
                        assistant_content = msg.get('content', [])
                        answer = assistant_content[-1].get("text", "") if assistant_content else ""
                    elif msg["role"] == "user":
                        user_content = msg.get('content', [])
                        image_path = [os.path.join(data_folder, item['image']) for item in user_content if item.get('type') == 'image']
                        question = user_content[-1].get("text", "") if user_content else ""
                
                data_dict[data_id] = {
                    "images": image_path,
                    "ground_truth": answer,
                    "task_type": task_type,
                    "question": question,
                    # custom fields
                }
                # print('data_dict[data_id]',data_dict[data_id])

        print(f"Loaded {len(data_dict)} samples from {data_root}.")
        return data_dict

    def generate_instruction(self, data_id: Union[int, str]) -> List[Dict[str, Any]]:
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        image_path = meta_data["images"]
        task_type = meta_data["task_type"]

        # instruction = f'{question}'

        question = (
            f"{question} Your current position is at the last frame of the video."
            if "object_conigtion" in str(data_id)
            else f"{question}"
        )

        if self.THINKING_MODE and task_type in ['counting']:
            thinking_prompt = f"\nOutput format: `#### <answer><counting>N</counting></answer>` where N is the count."
            question = f'{question}{thinking_prompt}'
            self.SYSTEM_PROMPT = SYSTEM_PROMPT_COUNTING

        if isinstance(question, str):
            if not isinstance(image_path, list):
                image_path = [image_path]
            content = []
            for i, path in enumerate(image_path):
                content.append({"type": "text", "text": f"<frame {i}>: "})
                content.append({"type": "image", "image": path})  

            messages = []
            if self.THINKING_MODE:
                messages.append({"role": "system", "content": self.SYSTEM_PROMPT})
            messages.append({
                    "role": "user",
                    "content": content + [{"type": "text", "text": question}],
            })
            # print('messages', messages)

        return messages

    async def process_response(self, data_id: Union[int, str], response: str) -> Any:
        """Process the raw model response."""
        # Normalize the response similarly to the ground truth for fair comparison
        if self.THINKING_MODE:
            match = re.findall(r'<answer><counting>(.*?)</counting></answer>', response, re.DOTALL)
            response = match[0].strip() if match else response
        return response.strip()

    async def get_matching_score(self, data_id, prediction):
        """
        Compute the matching score between model prediction and ground truth.
        """
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        ground_truth = meta_data["ground_truth"]
        task_type = meta_data["task_type"]

        record = {
            "idx": data_id,
            "question": question,
            "answer": ground_truth,
            "pred": prediction,
            "type": task_type,
        }
        score = await calculate_score(record, self.openai_client)
        print(f'| dataid: {data_id} | answer: {ground_truth} | pred: {prediction} | score: {score} |')
        return score

    def compute_metrics(self, results):
        sample_scores = defaultdict(list)
        for data in results:
            data_id = data["data_id"]
            score = data["score"]
            task_type = self.data_dict[data_id]["task_type"]
            sample_scores[task_type].append(score)

        # sub task metrics
        # task_num = {f'{t}_num': len(scores) for t, scores in sample_scores.items()}
        task_metrics = {t: sum(scores) / len(scores) for t, scores in sample_scores.items()}
        # print("Numbers by task_type:", task_num)
        print("Average scores by task_type:", task_metrics)

        # category metrics
        category_map = {}
        for category, subtypes in self.TYPE_LIST.items():
            for subtype in subtypes:
                category_map[subtype] = category

        category_scores = defaultdict(list)
        for task_type, scores in sample_scores.items():
            if task_type in category_map:
                category = category_map[task_type]
                category_scores[category].extend(scores)
            else:
                print(f"Warning: task_type '{task_type}' not found in TYPE_LIST")

        # Compute average score per category
        category_num = {f"{cat}_num": len(scores) for cat, scores in category_scores.items()}
        category_metircs = {cat: sum(scores) / len(scores) for cat, scores in category_scores.items()}
        print("Numbers by category:", category_num)
        print("Average scores by category:", category_metircs)

        # Spatial cognition metrics
        spatial_scores = []
        spatial_metrics = {}
        for cat, scores in category_scores.items():
            if cat in self.SPATIAL_TYPES:
                spatial_scores.extend(scores)
        if spatial_scores:
            spatial_metrics = {"Spatial Cognition": sum(spatial_scores) / len(spatial_scores)}

        metrics = {**spatial_metrics, **category_metircs, **task_metrics}

        return metrics
        # return self._summarize_scores(results, category_key="task_type")

SYSTEM_PROMPT_COUNTING = [
    {
        "type": "text",
        "text": (
            "You are an embodied agent. You are given a video to solve a counting problem. "
            # "Put your final answer in the format of `#### <answer><counting>N</counting></answer>`."
        ),
    }
]   

SCORE_TYPE = {
    "camera_rotation": "numerical",
    "camerall_distance": "numerical",
    "relative_direction_to_camera": "gpt",
    "distance_to_camera": "numerical",
    "direction_to_camera": "gpt",
    "direction_to_camera_3": "gpt",
    "distance_to_camera_choice_3": "gpt",
    "relative_direction_to_camera_2": "gpt",
    "distance_to_camera_2": "numerical",
    "relative_distance_to_object_choice_2": "gpt",
    "future_direction_to_camera": "gpt",
    "future_direction_to_camera_rotate": "numerical",
    "future_direction_object_3": "gpt",
    "height_data": "numerical",
    "tall_choice_3": "gpt",
    "small_predicate_3": "gpt",
    "tall_predicate": "gpt",
    "tall_choice": "gpt",
    "short_predicate": "gpt",
    "short_choice": "gpt",
    "big_predicate": "gpt",
    "small_predicate": "gpt",
    "height_from_ground": "numerical",
    "distance_compare": "gpt",
    "center_distance": "numerical",
    "between": "gpt",
    "above_predicate": "gpt",
    "above_choice": "gpt",
    "below_predicate": "gpt",
    "below_choice": "gpt",
    "directly_above": "gpt",
    "universal": "gpt_multi_granularity",
    "counting": "gpt",
    "category": "gpt_multi_granularity",
    "color": "gpt_multi_granularity",
    "material": "gpt_multi_granularity",
    "shape": "gpt_multi_granularity",
    "state": "gpt_multi_granularity",
    "position": "gpt_multi_granularity",
    "function": "gpt_multi_granularity",
    "surface detail": "gpt_multi_granularity",
    "size": "gpt_multi_granularity",
}

UNIT_LIST = ["centimeters", "meters", "feet", "inches", "degrees", "o\'clock"]

UNIT_EXCHANGE = {
    "length": {"centimeters": 100.0, "meters": 1.0, "inches": 39.3701, "feet": 3.28084},
}


def flatten_structure(d):
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            flat_values = []
            for subkey, subvalue in value.items():
                flat_values.extend(subvalue)
                result[subkey] = subvalue
            result[key] = flat_values
        else:
            result[key] = value
    return result


def clean_text(text):
    text = text.replace("<video>", "")
    text = text.replace("<REGION>", "")

    text = re.sub(r"[\[\]]", "", text)
    text = text.strip()
    return text


def make_prompt(question, label_answer, pred_answer):
    # question = clean_text(question)
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


def make_prompt_multi_granularity(question, label_answer, pred_answer):
    # question = clean_text(question)
    prompt = f"###Question: {question} ###5 Score Answer: {label_answer} ###Predicted Answer: {pred_answer}"

    question1 = "If you pick up the footstool between the sofa and the TV and flip it 180° upside down, which objects in the scene will be affected and how?"
    label_answer1 = "Three handles and one remote control will fall onto the carpet."
    pred_answer1 = "remote control will fall onto the carpet"
    score1 = "3"

    question2 = "If you pick up the footstool between the sofa and the TV and flip it 180° upside down, which objects in the scene will be affected and how?"
    label_answer2 = "Three handles and one remote control will fall onto the carpet."
    pred_answer2 = "handles"
    score2 = "2"

    question3 = "If you pick up the footstool between the sofa and the TV and flip it 180° upside down, which objects in the scene will be affected and how?"
    label_answer3 = "Three handles and one remote control will fall onto the carpet."
    pred_answer3 = "handles and control will fall "
    score3 = "4"

    question4 = "Which area in this room is better for studying? Why?"
    label_answer4 = "The U-shaped table area, which has matching chairs and good lighting"
    pred_answer4 = "may be table area"
    score4 = "1"

    question5 = "Which area in this room is better for studying? Why?"
    label_answer5 = "The U-shaped table area, which has matching chairs and good lighting"
    pred_answer5 = "Over there by the sofa."
    score5 = "0"

    question6 = "Which area in this room is better for studying? Why?"
    label_answer6 = "The U-shaped table area, which has matching chairs and good lighting"
    pred_answer6 = "may be table area and chairs"
    score6 = "4"

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Please score the predicted answer according to the given question and huamn labeled the 5-score answer, and the 3-score answer. 0 score represents completely wrong, 5 scores represents completely correct, and 3 scores represents partially correct. Please refer to them to score the predicted answer: [0, 1, 2, 3, 4, 5]. You need to consider the answer from two perspectives: accuracy and completeness. Output ###Judge:",
        },
        {
            "role": "user",
            "content": f"###Question: {question1} ###5 Score Answer: {label_answer1}. ###Predicted Answer: {pred_answer1}",
        },
        {"role": "assistant", "content": f"###Judge: {score1}"},
        {
            "role": "user",
            "content": f"###Question: {question2} ###5 Score Answer: {label_answer2}. ###Predicted Answer: {pred_answer2}",
        },
        {"role": "assistant", "content": f"###Judge: {score2}"},
        {
            "role": "user",
            "content": f"###Question: {question3} ###5 Score Answer: {label_answer3}. ###Predicted Answer: {pred_answer3}",
        },
        {"role": "assistant", "content": f"###Judge: {score3}"},
        {
            "role": "user",
            "content": f"###Question: {question4} ###5 Score Answer: {label_answer4}. ###Predicted Answer: {pred_answer4}",
        },
        {"role": "assistant", "content": f"###Judge: {score4}"},
        {
            "role": "user",
            "content": f"###Question: {question5} ###5 Score Answer: {label_answer5}. ###Predicted Answer: {pred_answer5}",
        },
        {"role": "assistant", "content": f"###Judge: {score5}"},
        {
            "role": "user",
            "content": f"###Question: {question6} ###5 Score Answer: {label_answer6}. ###Predicted Answer: {pred_answer6}",
        },
        {"role": "assistant", "content": f"###Judge: {score6}"},
        {"role": "user", "content": prompt},
    ]

    return messages


async def score_by_gpt(item, client, score_type="binary"):
    prompt_maker = {"binary": make_prompt, "multi": make_prompt_multi_granularity}

    question = item["question"]
    label_answer = item["answer"]
    pred_answer = item["pred"]
    question = clean_text(question)
    messages = prompt_maker[score_type](question, label_answer, pred_answer)

    try:
        completion = await client.chat.completions.create(
            model="gpt-4o-0806",
            messages=messages,
            max_completion_tokens=100,
        )
        response_raw = completion.choices[0].message.content
        response = response_raw.split("###Judge:")[1].strip()

        try:
            if score_type == "binary":
                score = 1 if response == "True" else 0
            else:
                match = re.match(r"(\d+)(.*)", response)
                response = match.group(1)
                score = float(int(response) / 5)

            return score, True
        except Exception:
            print(f"Model output: {response_raw}\n")
            # print(f"Prompt messages sent to model: {messages}")
            return 0.0, False
    except Exception as e:
        print(f"Error: {e}")
        # print(f"Prompt messages sent to model: {messages}")
        return await score_by_gpt(item, client, score_type)


def extract_number_and_unit_first(text):
    units_regex = "|".join(UNIT_LIST)

    match = re.search(r"(\d+(\.\d+)?)\s*(" + units_regex + r")", text, re.IGNORECASE)
    if match:
        return float(match.group(1)), match.group(3)
    else:
        return None, None


def extract_number_and_unit(text):
    units_regex = "|".join(UNIT_LIST)

    # match = re.search(r'(\d+(\.\d+)?)\s*(' + units_regex + r')', text, re.IGNORECASE)
    matches = re.findall(r"(\d+(\.\d+)?)\s*(" + units_regex + r")", text, re.IGNORECASE)

    if matches:
        return float(matches[-1][0]), matches[-1][2]
    else:
        return None, None


def calculate_MRA(number_result, number_gt):
    seta_group = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    metric = 0
    for seta in seta_group:
        if number_gt != 0.0:
            if abs(number_result - number_gt) / number_gt < 1 - seta:
                metric += 0.1
    return metric


def calculate_MRA_degree(number_result, number_gt, ref=90):
    seta_group = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    metric = 0
    for seta in seta_group:
        if abs(number_result - number_gt) / ref < 1 - seta:
            metric += 0.1
    return metric


def numerical_score(item):
    label_answer = item["answer"]
    pred_answer = item["pred"]

    number_gt, unit_gt = extract_number_and_unit(label_answer)
    number_pred, unit_pred = extract_number_and_unit(pred_answer)

    if number_pred is None:
        return 0.0

    if unit_gt in UNIT_EXCHANGE["length"]:
        if unit_pred not in UNIT_EXCHANGE["length"]:
            score = 0.0
        else:
            number_gt = number_gt / UNIT_EXCHANGE["length"][unit_gt]
            number_pred = number_pred / UNIT_EXCHANGE["length"][unit_pred]
            score = calculate_MRA(number_pred, number_gt)

    elif "o'clock" in unit_gt:
        score = max(
            calculate_MRA_degree(number_pred, number_gt, ref=3),
            calculate_MRA_degree(number_pred - 12, number_gt, ref=3),
        )
    elif "degree" in unit_gt:
        score = calculate_MRA_degree(number_pred, number_gt, ref=90)

    else:
        score = calculate_MRA(number_pred, number_gt)

    return score


async def calculate_score(d, client):
    if isinstance(d["type"], list):
        type_ = d["type"][0]
    else:
        type_ = d["type"]
    # print('task_type', type_)

    score_type = SCORE_TYPE[type_.lower()]
    if score_type == "gpt":
        score, success = await score_by_gpt(d, client, "binary")
    elif score_type == "gpt_multi_granularity":
        score, success = await score_by_gpt(d, client, "multi")
    elif score_type == "numerical":
        score = numerical_score(d)

    return score
