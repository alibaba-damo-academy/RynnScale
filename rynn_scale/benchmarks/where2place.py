import json
import os
import re

import numpy as np
from PIL import Image

from .base import BaseBenchmark
from ..registry import BENCHMARK_REGISTRY

TUPLE_PATTERN = re.compile(r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)")


def text2pts(text, width=640, height=480):
    """Parse coordinate tuples from model output text into pixel-space points.

    Model outputs coordinates in 0-1000 range. They are scaled to pixel space.

    Handles two formats:
      - 2-tuple (x, y): single point
      - 4-tuple (x0, y0, x1, y1): bounding box, expanded to all interior pixels
    """
    matches = TUPLE_PATTERN.findall(text)
    points = []
    for match in matches:
        vector = [float(num) for num in match.split(',')]
        if len(vector) == 2:
            x, y = vector
            x = int(x / 1000 * width)
            y = int(y / 1000 * height)
            points.append((x, y))
        elif len(vector) == 4:
            x0, y0, x1, y1 = vector
            x0 = int(x0 / 1000 * width)
            y0 = int(y0 / 1000 * height)
            x1 = int(x1 / 1000 * width)
            y1 = int(y1 / 1000 * height)
            mask = np.zeros((height, width), dtype=bool)
            mask[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
            y_coords, x_coords = np.where(mask)
            if len(x_coords) > 0:
                points.extend(list(np.stack([x_coords, y_coords], axis=1)))
    return np.array(points)


@BENCHMARK_REGISTRY.register()
class Where2Place(BaseBenchmark):
    def load_data(self, data_root):
        data_dict = {}
        json_file = os.path.join(data_root, "point_questions.jsonl")
        with open(json_file, "r") as f:
            for line in f:
                item = json.loads(line)
                qid = item["question_id"]
                question = item["text"]
                cut_marker = "The coordinates should be between 0 and 1"
                if cut_marker in question:
                    question = question[:question.index(cut_marker)].rstrip(". ")
                data_dict[qid] = {
                    "images": [os.path.join(data_root, "images", item["image"])],
                    "ground_truth": os.path.join(data_root, "masks", f"{qid:02d}.jpg"),
                    "task_type": item.get("category", "unknown"),
                    "question": question,
                }
        print(f"[Where2Place] Loaded {len(data_dict)} samples from {json_file}")
        return data_dict

    def generate_instruction(self, data_id):
        meta = self.data_dict[data_id]
        return [{"role": "user", "content": [
            {"type": "image", "image": meta["images"][0]},
            {"type": "text", "text": meta["question"]},
        ]}]

    async def process_response(self, data_id, response):
        prediction = response.strip()
        print(f"[Where2Place] qid={data_id} | prediction: {prediction}")
        return prediction

    async def get_matching_score(self, data_id, prediction):
        meta = self.data_dict[data_id]
        mask = np.array(Image.open(meta["ground_truth"]).convert("L")) / 255.0
        height, width = mask.shape

        try:
            points = text2pts(prediction, width=width, height=height)
        except Exception:
            print(f"[Where2Place] qid={data_id} | FAILED to parse, raw: {prediction[:200]}")
            return 0.0

        if len(points) == 0:
            print(f"[Where2Place] qid={data_id} | No points parsed, raw: {prediction[:200]}")
            return 0.0

        in_range = (
            (points[:, 0] >= 0) & (points[:, 0] < width)
            & (points[:, 1] >= 0) & (points[:, 1] < height)
        )
        acc = np.concatenate([
            mask[points[in_range, 1], points[in_range, 0]],
            np.zeros(points.shape[0] - in_range.sum())
        ]).mean()

        print(f"[Where2Place] qid={data_id} | points={len(points)} | in_range={in_range.sum()} | score={acc:.4f}")
        return float(acc) * 100

    def compute_metrics(self, results):
        metrics = self._summarize_scores(results, category_key="task_type")
        print(f"[Where2Place] Metrics: {metrics}")
        return metrics
