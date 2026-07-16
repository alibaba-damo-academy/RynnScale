import json
import os
import re

from .base import BaseBenchmark
from ..registry import BENCHMARK_REGISTRY

SUBSETS = ["agibot", "bridgev2", "holoassist", "robofail", "robovqa"]


@BENCHMARK_REGISTRY.register()
class CosmosReason1(BaseBenchmark):
    def load_data(self, data_root):
        data_dict = {}
        idx = 0

        for subset in SUBSETS:
            json_file = os.path.join(data_root, subset, f"{subset}_benchmark_qa_pairs.json")
            with open(json_file, "r") as f:
                data_list = json.load(f)

            for item in data_list:
                qa = item["qa_pairs"]
                option_letters = sorted(qa["index2ans"].keys())
                options = [qa["index2ans"][k] for k in option_letters]
                answer_idx = option_letters.index(qa["answer"])

                data_dict[idx] = {
                    "videos": [os.path.join(data_root, subset, item["video"])],
                    "task_type": subset,
                    "ground_truth": answer_idx,
                    "question": qa["question"],
                    "options": options,
                    "option_letters": option_letters,
                }
                idx += 1

        return data_dict

    def _format_options(self, option_letters, options):
        return "".join(f"({letter}) {option}\n" for letter, option in zip(option_letters, options))

    def generate_instruction(self, data_id):
        meta_data = self.data_dict[data_id]
        option_string = self._format_options(meta_data["option_letters"], meta_data["options"])

        instruction_text = (
            f"Select the best answer to the following multiple-choice question based on the video.\n"
            f"Question: {meta_data['question']}\n"
            f"Options:\n{option_string}"
            f"Answer with the option's letter from the given choices directly and only give the best option."
            f"The best answer is:"
        )

        contents = [{"type": "video", "video": video} for video in meta_data["videos"]]
        contents.append({"type": "text", "text": instruction_text})
        return [{"role": "user", "content": contents}]

    def _extract_letter(self, response, option_letters):
        letters_pattern = "".join(option_letters)
        cleaned = re.sub(r"(?i)\bthe\s+answer\s+is\b", "", response)
        cleaned = re.sub(r"(?i)\banswer\s*:", "", cleaned)
        matches = re.findall(rf"[\(,\ ]*([{letters_pattern}])[\),\ ]*", cleaned)
        if not matches:
            return None
        return matches[-1].strip()

    def _match_option_text(self, response, options):
        response_lower = response.lower()
        for idx, opt in enumerate(options):
            if opt.strip().strip(".").lower() in response_lower:
                return idx
        return None

    async def process_response(self, data_id, response):
        meta_data = self.data_dict[data_id]
        options = meta_data["options"]
        option_letters = meta_data["option_letters"]
        gt_idx = meta_data["ground_truth"]

        letter = self._extract_letter(response, option_letters)
        if letter and letter in option_letters:
            pred_idx = option_letters.index(letter)
        else:
            pred_idx = self._match_option_text(response, options)

        gt_letter = option_letters[gt_idx]
        gt_text = options[gt_idx]
        pred_letter = option_letters[pred_idx] if pred_idx is not None else "N/A"
        pred_text = options[pred_idx] if pred_idx is not None else "N/A"
        is_correct = pred_idx == gt_idx if pred_idx is not None else False

        print(
            f"\n[CosmosReason1] data_id={data_id} | subset={meta_data['task_type']}\n"
            f"  Question : {meta_data['question']}\n"
            f"  Response : {response[:200]}{'...' if len(response) > 200 else ''}\n"
            f"  Parsed   : ({pred_letter}) {pred_text}\n"
            f"  GT       : ({gt_letter}) {gt_text}\n"
            f"  Correct  : {is_correct}"
        )

        return pred_idx

    async def get_matching_score(self, data_id, prediction):
        if prediction is None:
            return 0
        ground_truth = self.data_dict[data_id]["ground_truth"]
        return int(prediction == ground_truth) * 100

    def compute_metrics(self, results):
        return self._summarize_scores(results, category_key="task_type")
