import json
import os
import re
import ast

import numpy as np
from PIL import Image

from .base import BaseBenchmark
from ..registry import BENCHMARK_REGISTRY


def point_in_polygon(x, y, poly):
    num = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, num + 1):
        p2x, p2y = poly[i % num]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                else:
                    xinters = p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y
    return inside


@BENCHMARK_REGISTRY.register()
class RoboSpatial(BaseBenchmark):
    def load_data(self, data_root):
        data_dict = {}

        json_file = os.path.join(data_root, "annotations_revised.json")
        with open(json_file, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        for data in data_list:
            question_id = data["id"]
            image_path = os.path.join(data_root, data["img"])
            assert os.path.exists(image_path), f"Cannot find the image file: {image_path}"

            data_dict[question_id] = {
                # required fields for data loading
                "images": [image_path],
                # required fields for evaluation
                "ground_truth": data["answer"],
                "mask_path": os.path.join(data_root, data["mask"]) if data["mask"] is not None else None,
                "is_binary": data["answer"].lower() in ["yes", "no"],
                "task_type": data["category"],
                # custom fields for instruction generation and post processing
                "question": data["question"],
            }

        return data_dict

    def generate_instruction(self, data_id):
        meta_data = self.data_dict[data_id]
        contents = [
            {"type": "image", "image": meta_data["images"][0]},
            {"type": "text", "text": meta_data["question"]},
        ]
        instruction = [{"role": "user", "content": contents}]
        return instruction

    async def process_response(self, data_id, response):
        meta_data = self.data_dict[data_id]
        if meta_data["is_binary"]:
            response = response.lower()
            if "yes" in response:
                prediction = "yes"
            elif "no" in response:
                prediction = "no"
            else:
                prediction = None
            return prediction

        matches = re.findall(r"[\[\(]\s*(-?\d+)\s*,\s*(-?\d+)\s*[\]\)]", response)
        if not matches:
            points = None
        else:
            points = [[int(x), int(y)] for x, y in matches]

        return points

    async def get_matching_score(self, data_id, prediction):
        meta_data = self.data_dict[data_id]
        ground_truth = meta_data["ground_truth"]

        if prediction is None:
            score = 0.0
        else:
            if meta_data["is_binary"]:
                score = int(prediction.lower() == ground_truth.lower())
            else:
                if meta_data["mask_path"] is not None:
                    mask = mask = np.array(Image.open(meta_data["mask_path"])) > 0
                    if mask.ndim == 3:
                        mask = mask[..., 0]

                    h, w = mask.shape
                    final_points = []
                    for point in prediction:
                        x, y = point
                        px = int(x / 1000 * (w - 1))
                        py = int(y / 1000 * (h - 1))
                        final_points.append([px, py])

                    final_points = np.array(final_points)
                    in_range = (
                        (final_points[:, 0] >= 0)
                        & (final_points[:, 0] < w)
                        & (final_points[:, 1] >= 0)
                        & (final_points[:, 1] < h)
                    )
                    score = np.concatenate(
                        [
                            mask[final_points[in_range, 1], final_points[in_range, 0]],
                            np.zeros(final_points.shape[0] - in_range.sum()),
                        ]
                    ).mean()
                else:
                    gt_polygon = ast.literal_eval(ground_truth)
                    match = [point_in_polygon(point[0], point[1], gt_polygon) for point in prediction]
                    score = sum(match) / len(match)

        return score * 100

    def compute_metrics(self, results):
        return self._summarize_scores(results, category_key="task_type")
