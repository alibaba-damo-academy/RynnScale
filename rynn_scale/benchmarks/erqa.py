import os
import re
import json

from .base import BaseBenchmark
from ..registry import BENCHMARK_REGISTRY


@BENCHMARK_REGISTRY.register()
class ERQA(BaseBenchmark):
    def load_data(self, data_root):
        data_dict = {}
        data_path = os.path.join(data_root, "erqa_reformat_v4.json")
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data_dict = json.load(f)

        for data_id, data in raw_data_dict.items():
            images = [os.path.join(data_root, image) for image in data["image_path"]]
            question = data["question"]
            options = re.findall(r"[A-D]\.\s*(.*)", question)
            options = {chr(ord("A") + i): option.strip(".").strip() for i, option in enumerate(options)}
            data_dict[data_id] = {
                "images": images,
                "task_type": data["task_type"],
                "question": question,
                "options": options,
                "ground_truth": data["original_answer"],
            }

        return data_dict

    def generate_instruction(self, data_id):
        meta_data = self.data_dict[data_id]
        contents = [{"type": "image", "image": image} for image in meta_data["images"]]
        contents.append({"type": "text", "text": meta_data["question"]})
        instruction = [{"role": "user", "content": contents}]
        return instruction

    async def process_response(self, data_id, response):
        matches = re.findall(r"[\(\ ]*[A-D][\)\ ]*", response)
        if not matches:
            options = self.data_dict[data_id]["options"]
            for k, option in options.items():
                if option.lower() in response.lower():
                    prediction = k
                    break
            else:
                prediction = None
        else:
            prediction = matches[0]
        return prediction

    async def get_matching_score(self, data_id, prediction):
        ground_truth = self.data_dict[data_id]["ground_truth"]
        match = prediction == ground_truth
        return int(match) * 100

    def compute_metrics(self, results):
        return self._summarize_scores(results, category_key="task_type")
