import json
import os
import re

import numpy as np
from PIL import Image

from ..registry import BENCHMARK_REGISTRY
from .base import BaseBenchmark


def resample_points(points, target_length):
    if len(points) == 0:
        return np.array([])

    points = np.array(points)

    if len(points) == 1:
        return np.tile(points, (target_length, 1))

    if len(points) == target_length:
        return points

    original_indices = np.linspace(0, len(points) - 1, len(points))
    target_indices = np.linspace(0, len(points) - 1, target_length)

    resampled_x = np.interp(target_indices, original_indices, points[:, 0])
    resampled_y = np.interp(target_indices, original_indices, points[:, 1])

    return np.column_stack((resampled_x, resampled_y))


def calculate_min_distance(points, ground_truth):
    resampled_points = resample_points(points, len(ground_truth))
    ground_truth = np.array(ground_truth)

    distances = np.sqrt(np.sum((resampled_points - ground_truth) ** 2, axis=1)) / 1e3

    min_distance = np.min(distances) if len(distances) > 0 else 0

    return min_distance, distances.tolist()


def calculate_dfd(points, ground_truth):
    # if not seq1 or not seq2:
    #     return math.exp( - float('inf'))

    resampled_points = resample_points(points, len(ground_truth))

    seq1 = np.array(resampled_points)
    seq2 = np.array(ground_truth)
    m, n = len(seq1), len(seq2)

    def dist_func(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    C = np.zeros((m, n))
    C[0, 0] = dist_func(seq1[0], seq2[0])

    for i in range(1, m):
        C[i, 0] = max(C[i - 1, 0], dist_func(seq1[i], seq2[0]))

    for j in range(1, n):
        C[0, j] = max(C[0, j - 1], dist_func(seq1[0], seq2[j]))

    for i in range(1, m):
        for j in range(1, n):
            C[i, j] = max(min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1]), dist_func(seq1[i], seq2[j]))

    return C[m - 1, n - 1]


def create_predicted_box_original_size(predict_points, ground_truth):
    x, y = predict_points
    x1_gt, y1_gt, x2_gt, y2_gt = ground_truth

    gt_width = x2_gt - x1_gt
    gt_height = y2_gt - y1_gt

    half_width = gt_width / 2
    half_height = gt_height / 2

    return [x - half_width, y - half_height, x + half_width, y + half_height]


def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


@BENCHMARK_REGISTRY.register()
class ShareRobot(BaseBenchmark):
    def load_data(self, data_root):
        data_dict = {}

        for task_type in ["affordance", "trajectory"]:
            json_file = os.path.join(data_root, task_type, "test.json")

            with open(json_file, "r", encoding="utf-8") as f:
                data_list = json.load(f)

            for data in data_list:
                if isinstance(data["image"], list):
                    image_path = os.path.join(data_root, task_type, data["image"][0])
                else:
                    image_path = os.path.join(data_root, task_type, data["image"])

                assert os.path.exists(image_path), f"Cannot find the image file: {image_path}"
                W, H = Image.open(image_path).size

                if task_type == "affordance":
                    x1, y1, x2, y2 = json.loads(data["conversations"][1]["value"])
                    ground_truth = [int(x1 / W * 1000), int(y1 / H * 1000), int(x2 / W * 1000), int(y2 / H * 1000)]
                elif task_type == "trajectory":
                    points = json.loads(data["conversations"][1]["value"].replace("(", "[").replace(")", "]"))
                    ground_truth = [(int(p[0] / W * 1000), int(p[1] / H * 1000)) for p in points]

                question = data["conversations"][0]["value"].split("<image>")[1].lstrip()

                data_dict[f"{task_type}_{data['id']}"] = {
                    "images": [image_path],
                    "ground_truth": ground_truth,
                    "question": question,
                    "task_type": task_type,
                }

        return data_dict

    def generate_instruction(self, data_id):
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]

        if self.prompt_format == "RynnBrain":
            match = re.search(r'"([^"]*)"', question)
            if match:
                extracted_task = match.group(1)
            else:
                match_unquoted = re.search(r"The task is\s+([^.]*?)\s*[.\n]", question)
                if match_unquoted:
                    extracted_task = match_unquoted.group(1).strip()
                else:
                    raise ValueError

            if meta_data["task_type"] == "affordance":
                suffix = " \nGenerate coordinates for one affordance point. Constraints: x∈[0,1000], y∈[0,1000]. Response must be in the format: <affordance> (x, y) </affordance>"
            elif meta_data["task_type"] == "trajectory":
                suffix = "\nTask: Trajectory point prediction\n- Identify up to 10 key points for the trajectory\n- Normalize all coordinates to 0-1000 range\n- Output format: <trajectory> (x1, y1), (x2, y2), ... </trajectory>"
            else:
                raise ValueError
            question = extracted_task + suffix
            question = question.replace("points", "points of Robot Gripper")

        contents = [
            {"type": "image", "image": meta_data["images"][0]},
            {"type": "text", "text": question},
        ]
        instruction = [{"role": "user", "content": contents}]

        return instruction

    async def process_response(self, data_id, response):
        meta_data = self.data_dict[data_id]
        task_type = meta_data["task_type"]

        if self.prompt_format == "RynnBrain":
            if task_type == "affordance":
                pattern = r"<affordance> (.*?) </affordance>"
            else:
                pattern = r"<trajectory> (.*?) </trajectory>"

            match = re.search(pattern, response)
            if match:
                response = match.group(1)

        points_str = "[" + ", ".join(re.findall(r"\(\d+,\s*\d+\)", response)) + "]"
        points_str = points_str.replace("(", "[").replace(")", "]")

        try:
            points = json.loads(points_str)
        except Exception:
            points = []

        if self.prompt_format == "RynnBrain" and task_type == "affordance":
            points = create_predicted_box_original_size(points[0], meta_data["ground_truth"])

        return points

    async def get_matching_score(self, data_id, prediction):
        meta_data = self.data_dict[data_id]
        ground_truth = meta_data["ground_truth"]

        if len(prediction):
            try:
                if meta_data["task_type"] == "affordance":
                    score = calculate_iou(prediction, ground_truth) * 100
                else:
                    normed_points = [[p / 1000 for p in point] for point in prediction]
                    normed_ground_truth = [[p / 1000 for p in point] for point in ground_truth]
                    score = calculate_dfd(normed_points, normed_ground_truth)
            except Exception:
                import traceback

                traceback.print_exc()
                score = 0
        else:
            score = 0

        return score

    def compute_metrics(self, results):
        metrics = self._summarize_scores(results, category_key="task_type")
        metrics.pop("Overall")
        return metrics
