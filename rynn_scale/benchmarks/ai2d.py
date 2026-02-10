import json
import os
import re

from .base import BaseBenchmark
from ..registry import BENCHMARK_REGISTRY


@BENCHMARK_REGISTRY.register()
class AI2D(BaseBenchmark):
    def load_data(self, data_root):
        data_dict = {}

        json_file = os.path.join(data_root, "ai2d_test_vlmevalkit.jsonl")
        data_list = [json.loads(item.strip()) for item in open(json_file).readlines()]

        for data in data_list:
            question_id = data["question_id"]
            image_path = os.path.join(data_root, data["image"])
            assert os.path.exists(image_path), f"Cannot find the image file: {image_path}"

            answer = ord(data["answer"]) - 65
            assert answer >= 0 and answer < 4, (
                f"Wrong ground truth for file: {image_path}. Ground Truth: {data['answer']}"
            )

            data_dict[question_id] = {
                # required fields for data loading
                "images": [image_path],
                # required fields for evaluation
                "ground_truth": answer,
                "task_type": "test",
                # custom fields for instruction generation and post processing
                "question": data["question"],
                "category": data["category"],
            }

        return data_dict

    def generate_instruction(self, data_id):
        meta_data = self.data_dict[data_id]
        contents = [{"type": "image", "image": image} for image in meta_data["images"]]
        contents.append({"type": "text", "text": meta_data["question"]})
        instruction = [{"role": "user", "content": contents}]
        return instruction

    async def process_response(self, data_id, response):
        letters = ["A", "B", "C", "D"]

        response = response.replace("answer", "")
        response = response.replace("Answer", "")
        pred_answer = re.findall(r"[\(\ ]*[A-D][\)\ ]*", response)

        if len(pred_answer) == 0:
            pred_idx = None
        else:
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip("()")
            pred_idx = letters.index(pred_answer)

        return pred_idx

    async def get_matching_score(self, data_id, prediction):
        ground_truth = self.data_dict[data_id]["ground_truth"]
        match = prediction == ground_truth
        return int(match) * 100

    def compute_metrics(self, results):
        return self._summarize_scores(results, category_key="category")
