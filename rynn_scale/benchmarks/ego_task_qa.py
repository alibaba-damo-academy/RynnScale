import json
import os
import re

from .base import BaseBenchmark
from ..registry import BENCHMARK_REGISTRY


@BENCHMARK_REGISTRY.register()
class EgoTaskQA(BaseBenchmark):
    def load_data(self, data_root):
        data_dict = {}

        video_folder = os.path.join(data_root, "videos")
        json_file = os.path.join(data_root, "egotaskqa.json")
        with open(json_file, "r") as f:
            data_list = json.load(f)

        for i, data in enumerate(data_list):
            data_dict[i] = {
                "videos": [os.path.join(video_folder, data["video_path"])],
                "question": data["q"],
                "options": data["option"],
                "ground_truth": data["a"],
            }

        return data_dict

    def generate_instruction(self, data_id):
        meta_data = self.data_dict[data_id]

        prompt = f"Select the best answer to the following multiple-choice question based on the video.\n{meta_data['question']}\nOptions:\n"
        for letter in sorted(meta_data["options"].keys()):
            prompt += f"({letter}) {meta_data['options'][letter]}\n"
        prompt += "\nAnswer with the option's letter from the given choices directly and only give the best option. The best answer is: "

        contents = [{"type": "video", "video": meta_data["videos"][0]}]
        contents.append({"type": "text", "text": prompt})
        instruction = [{"role": "user", "content": contents}]

        return instruction

    async def process_response(self, data_id, response):
        letters = sorted(self.data_dict[data_id]["options"].keys())
        options = [self.data_dict[data_id]["options"][x] for x in letters]

        response = response.replace("answer", "")
        response = response.replace("Answer", "")
        pred_answer = re.findall(r"[\(\ ]*[A-E][\)\ ]*", response)

        find_flag = False
        if len(pred_answer) == 0:
            for idx, opt in enumerate(options):
                opt = opt.strip()
                opt = opt.strip(".")
                if opt.lower() in response.lower():
                    pred_idx = idx
                    find_flag = True
                    break
        else:
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip("()")
            pred_idx = letters.index(pred_answer)
            find_flag = True

        assert find_flag, f"Cannot find the answer in the options: {response}"
        return chr(ord("A") + pred_idx)

    async def get_matching_score(self, data_id, prediction):
        ground_truth = self.data_dict[data_id]["ground_truth"]
        match = prediction == ground_truth
        return int(match) * 100

    def compute_metrics(self, results):
        return self._summarize_scores(results)
