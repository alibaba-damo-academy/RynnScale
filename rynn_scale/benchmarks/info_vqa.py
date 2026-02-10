import json
import os

from .base import BaseBenchmark
from ..registry import BENCHMARK_REGISTRY


@BENCHMARK_REGISTRY.register()
class InfoVQA(BaseBenchmark):
    def load_data(self, data_root):
        data_dict = {}

        with open(os.path.join(data_root, "infographicsVQA_val_v1.0_withQT.json"), "r") as f:
            data_list = json.load(f)["data"]

        for data in data_list:
            question_id = data["questionId"]
            image_path = os.path.join(data_root, "images", data["image_local_name"])
            assert os.path.exists(image_path), f"Cannot find the image file: {image_path}"

            data_dict[question_id] = {
                "images": [image_path],
                "ground_truth": data["answers"],
                "question": data["question"],
            }

        return data_dict

    def generate_instruction(self, data_id):
        meta_data = self.data_dict[data_id]
        contents = [{"type": "image", "image": image} for image in meta_data["images"]]
        contents.append(
            {"type": "text", "text": meta_data["question"] + "\nAnswer the question with a single word or phrase."}
        )
        instruction = [{"role": "user", "content": contents}]
        return instruction

    async def process_response(self, data_id, response):
        return response

    def levenshtein_distance(self, s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    async def get_matching_score(self, data_id, prediction):
        values = []
        for answer in self.data_dict[data_id]["ground_truth"]:
            gt_answer = " ".join(answer.strip().lower().split())
            det_answer = " ".join(prediction.strip().lower().split())
            dist = self.levenshtein_distance(gt_answer, det_answer)
            length = max(len(answer.upper()), len(prediction.upper()))
            values.append(0.0 if length == 0 else float(dist) / float(length))
        return (1 - min(values)) * 100

    def compute_metrics(self, results):
        return self._summarize_scores(results)
