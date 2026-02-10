import json
import os
import re

from .base import BaseBenchmark
from ..registry import BENCHMARK_REGISTRY


@BENCHMARK_REGISTRY.register()
class MindCube(BaseBenchmark):
    def load_data(self, data_root):
        data_dict = {}

        image_folder = os.path.join(data_root, "data")
        json_file = os.path.join(data_root, "data/raw/MindCube.jsonl")

        for line in open(json_file):
            data = json.loads(line)
            data_dict[data["id"]] = {
                "images": [os.path.join(image_folder, img_path) for img_path in data["images"]],
                "question": data["question"],
                "ground_truth": data["gt_answer"],
                "task_type": data["category"],
            }

        return data_dict

    def generate_instruction(self, data_id):
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        instruction = f"Select the best answer to the following multiple-choice question based on the image.\n{question}\nAnswer with the option's letter from the given choices directly and only give the best option. The best answer is: "

        contents = [{"type": "image", "image": image} for image in meta_data["images"]]
        contents.append({"type": "text", "text": meta_data["question"]})
        instruction = [{"role": "user", "content": contents}]

        return instruction

    async def process_response(self, data_id, response):
        letters = ["A", "B", "C", "D"]

        response = response.replace("answer", "")
        response = response.replace("Answer", "")
        pred_answer = re.findall(r"[\(\ ]*[A-D][\)\ ]*", response)

        if not pred_answer:
            print(f"Cannot find the answer in the options: {response}")
            return 0

        pred_answer = pred_answer[0].strip()
        pred_answer = pred_answer.strip("()")
        pred_idx = letters.index(pred_answer)
        return pred_idx

    async def get_matching_score(self, data_id, prediction):
        ground_truth = ord(self.data_dict[data_id]["ground_truth"]) - ord("A")
        return int(prediction == ground_truth) * 100

    def compute_metrics(self, results):
        return self._summarize_scores(results, category_key="task_type")
