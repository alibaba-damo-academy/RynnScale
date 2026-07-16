import base64
import json
import os
import re
from io import BytesIO

from PIL import Image

from .base import BaseBenchmark
from ..registry import BENCHMARK_REGISTRY


@BENCHMARK_REGISTRY.register()
class EmbSpatial(BaseBenchmark):
    def load_data(self, data_root):
        data_dict = {}
        json_file = os.path.join(data_root, "embspatial_bench.json")
        with open(json_file, "r") as f:
            data_list = json.load(f)

        for data in data_list:
            question_id = data["question_id"]
            image = Image.open(BytesIO(base64.b64decode(data["image"]))).convert("RGB")

            options = data["answer_options"]
            answer_idx = data["answer"]
            option_letters = [chr(ord('A') + i) for i in range(len(options))]

            data_dict[question_id] = {
                "images": [image],
                "ground_truth": answer_idx,
                "task_type": data["relation"],
                "question": data["question"],
                "options": options,
                "option_letters": option_letters,
            }

        return data_dict

    def generate_instruction(self, data_id):
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        options = meta_data["options"]
        option_letters = meta_data["option_letters"]

        option_string = ""
        for letter, option in zip(option_letters, options):
            option_string += f"({letter}) {option}\n"

        instruction = (
            f"Select the best answer to the following multiple-choice question based on the image.\n"
            f"Question: {question}\n"
            f"Options:\n{option_string}"
            f"Answer with the option's letter from the given choices directly and only give the best option."
            f"The best answer is:"
        )

        contents = [{"type": "image", "image": image} for image in meta_data["images"]]
        contents.append({"type": "text", "text": instruction})
        return [{"role": "user", "content": contents}]

    async def process_response(self, data_id, response):
        meta_data = self.data_dict[data_id]
        option_letters = meta_data["option_letters"]
        options = meta_data["options"]
        ground_truth = meta_data["ground_truth"]

        print(f"[EmbSpatial] {data_id} | Q: {meta_data['question']} | Response: {response}")

        cleaned = re.sub(r'\b[Aa]nswer\b', '', response)
        pred_answer = re.findall(r'(?:^|[\s\(])([A-D])(?:[\s\)\.\,\:]|$)', cleaned)

        if len(pred_answer) == 0:
            last_match_idx = None
            last_match_pos = -1
            for idx, opt in enumerate(options):
                pattern = r'\b' + re.escape(opt.lower().strip()) + r'\b'
                matches = list(re.finditer(pattern, response.lower()))
                if matches and matches[-1].start() > last_match_pos:
                    last_match_pos = matches[-1].start()
                    last_match_idx = idx
            if last_match_idx is not None:
                print(f"[EmbSpatial] {data_id} | Pred: {last_match_idx} (text: '{options[last_match_idx]}') | GT: {ground_truth} | {'✓' if last_match_idx == ground_truth else '✗'}")
                return last_match_idx
            print(f"[EmbSpatial] {data_id} | Pred: None | GT: {ground_truth}")
            return None
        else:
            pred_answer = pred_answer[-1].strip()
            pred_idx = option_letters.index(pred_answer)
            print(f"[EmbSpatial] {data_id} | Pred: {pred_idx} ({pred_answer}) | GT: {ground_truth} | {'✓' if pred_idx == ground_truth else '✗'}")
            return pred_idx

    async def get_matching_score(self, data_id, prediction):
        ground_truth = self.data_dict[data_id]["ground_truth"]
        if prediction is None:
            return 0
        return int(prediction == ground_truth) * 100

    def compute_metrics(self, results):
        return self._summarize_scores(results, category_key="task_type")
