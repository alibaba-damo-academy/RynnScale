import io
import json
import os
import re
import traceback

import numpy as np
import pandas as pd
from PIL import Image

from ..registry import BENCHMARK_REGISTRY
from .base import BaseBenchmark


@BENCHMARK_REGISTRY.register()
class RefSpatial(BaseBenchmark):
    def load_data(self, data_root: str):
        data_dict = {}
        idx = 0

        for task_type in ["location", "placement", "unseen"]:
            ann_path = os.path.join(data_root, "data", f"{task_type}-00000-of-00001.parquet")
            df = pd.read_parquet(ann_path)

            for data in df.itertuples():
                data_dict[idx] = {
                    "images": [Image.open(io.BytesIO(data.image["bytes"]))],
                    "mask": Image.open(io.BytesIO(data.mask["bytes"])),
                    "task_type": task_type,
                    "object": data.object,
                    "question": data.prompt,
                    "suffix": data.suffix,
                }
                idx += 1

        return data_dict

    def generate_instruction(self, data_id):
        meta_data = self.data_dict[data_id]

        if self.prompt_format == "RynnBrain":
            question = meta_data["question"] + " " + meta_data["suffix"]
        elif self.prompt_format == "Qwen3-VL":
            question = f"Locate {meta_data['object']}. Output the point coordinates in JSON format."
        else:
            question = meta_data["question"] + " " + meta_data["suffix"]

        contents = [
            {"type": "image", "image": meta_data["images"][0]},
            {"type": "text", "text": question},
        ]
        instruction = [{"role": "user", "content": contents}]

        return instruction

    async def process_response(self, data_id, response):
        if self.prompt_format == "RynnBrain":
            pattern = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
            pairs = [[int(a), int(b)] for a, b in pattern.findall(response)]
        elif self.prompt_format == "Qwen3-VL":
            match = re.search(r"(\{.*\})", response, re.DOTALL)
            try:
                pairs = [json.loads(match[0])["point_2d"]]
            except Exception:
                pairs = []
        else:
            match = re.search(r'x1="(\d+)" y1="(\d+)"', response)
            if match:
                x1 = int(match.group(1))
                y1 = int(match.group(2))
                pairs = [[x1, y1]]
            else:
                pairs = []

        return pairs

    async def get_matching_score(self, data_id, prediction):
        mask = np.array(self.data_dict[data_id]["mask"]) / 255.0
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask > 0).astype(np.uint8)
        h, w = mask.shape

        if len(prediction) > 0:
            try:
                points = np.array(prediction)
                points[:, 0] = int(points[:, 0] / 1000 * (w - 1))
                points[:, 1] = int(points[:, 1] / 1000 * (h - 1))

                in_range = (
                    (points[:, 0] >= 0)
                    & (points[:, 0] < mask.shape[1])
                    & (points[:, 1] >= 0)
                    & (points[:, 1] < mask.shape[0])
                )
                score = (
                    np.concatenate(
                        [mask[points[in_range, 1], points[in_range, 0]], np.zeros(points.shape[0] - in_range.sum())]
                    )
                    .mean()
                    .item()
                )
            except Exception:
                traceback.print_exc()
                score = 0.0
        else:
            score = 0.0

        return score * 100

    def compute_metrics(self, results):
        return self._summarize_scores(results, category_key="task_type")
