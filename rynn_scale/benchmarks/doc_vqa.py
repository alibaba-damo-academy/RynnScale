import json
import os

from .base import BaseBenchmark
from ..registry import BENCHMARK_REGISTRY


@BENCHMARK_REGISTRY.register()
class DocVQA(BaseBenchmark):
    def load_data(self, data_root):
        data_dict = {}

        json_file = os.path.join(data_root, "val.jsonl")
        data_list = [json.loads(item.strip()) for item in open(json_file).readlines()]

        for data in data_list:
            question_id = data["question_id"]
            image_path = os.path.join(data_root, data["image"])
            assert os.path.exists(image_path), f"Cannot find the image file: {image_path}"

            data_dict[question_id] = {
                "images": [image_path],
                "ground_truth": data["answer"],
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
        return response.strip().lower()

    async def get_matching_score(self, data_id, prediction):
        ground_truth = self.data_dict[data_id]["ground_truth"]
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]

        scores = []
        for answer in ground_truth:
            gt_answer = " ".join(answer.strip().lower().split())
            det_answer = " ".join(prediction.split())

            dist = self.levenshtein_distance(gt_answer, det_answer)
            length = max(len(answer.upper()), len(prediction.upper()))
            scores.append(0.0 if length == 0 else float(dist) / float(length))

        score = 1 - min(scores)
        return score * 100

    def compute_metrics(self, results):
        return self._summarize_scores(results)

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
