import ast
import os
import re

from datasets import load_dataset

from ..registry import BENCHMARK_REGISTRY
from .base import BaseBenchmark


@BENCHMARK_REGISTRY.register()
class OpenXVQA(BaseBenchmark):
    def load_data(self, data_root):
        data_dict = {}
        dataset = load_dataset("parquet", data_files={"test": os.path.join(data_root, "data/test-*-of-*.parquet")})
        for data in dataset["test"]:
            data_dict[data["id"]] = {
                "images": [data["image"]],
                "question": data["question"],
                "options": ast.literal_eval(data["choices"]),
                "ground_truth": int(data["correct_answer"]),
            }
        return data_dict

    def generate_instruction(self, data_id):
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        options = meta_data["options"]
        letters = "ABCDEFG"

        opts_text = "\n".join(f"({letters[i]}) {opt}" for i, opt in enumerate(options))

        instruction = f"Select the best answer to the following multiple-choice question based on the image.\n{question}\nOptions:\n{opts_text}\nAnswer with the option's letter from the given choices directly and only give the best option. The best answer is: "
        contents = [{"type": "image", "image": image} for image in meta_data["images"]]
        contents.append({"type": "text", "text": instruction})
        instruction = [{"role": "user", "content": contents}]

        return instruction

    async def process_response(self, data_id, response):
        options = self.data_dict[data_id]["options"]
        letters = list("ABCDEFG")[: len(options)]
        last_letter = letters[-1]

        response = response.replace("answer", "")
        response = response.replace("Answer", "")
        pred_answer = re.findall(rf"[\(\ ]*[{letters[0]}-{last_letter}][\)\ ]*", response)

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

        if not find_flag:
            pred_idx = 0
            print(f"Cannot find the answer in the options: {response}")
        return pred_idx

    async def get_matching_score(self, data_id, prediction):
        ground_truth = self.data_dict[data_id]["ground_truth"]
        match = prediction == ground_truth
        return int(match) * 100

    def compute_metrics(self, results):
        return self._summarize_scores(results)
