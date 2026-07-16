import os
import re

import pyarrow.parquet as pq

from ..registry import BENCHMARK_REGISTRY
from .base import BaseBenchmark


@BENCHMARK_REGISTRY.register()
class VSIBench(BaseBenchmark):
    def load_data(self, data_root: str):
        parquet_file = os.path.join(data_root, "test-00000-of-00001.parquet")
        table = pq.read_table(parquet_file)
        df = table.to_pandas()

        data_dict = {}
        idx = 0

        for record in df.itertuples():
            video_id = record.scene_name
            video_dataset = record.dataset
            for video_format in ["mp4", "avi", "mov", "mkv"]:
                temp_path = os.path.join(data_root, video_dataset, f"{video_id}.{video_format}")
                if os.path.exists(temp_path):
                    video_path = temp_path
                    break
            assert os.path.exists(video_path), f"Cannot find the video file: {video_id}"

            if record.options is not None:
                options = list(record.options)
            else:
                options = None

            data_dict[record.id] = {
                # required fields for data loading
                "videos": [video_path],
                # required fields for evaluation
                "task_type": record.question_type,
                "ground_truth": record.ground_truth,
                # custom fields for instruction generation and post processing
                "question": record.question,
                "options": options,
            }

            idx += 1

        return data_dict

    def generate_instruction(self, data_id):
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        options = meta_data["options"]
        if options is not None:
            prompt = f"{question}\n"
            for option_idx, option in enumerate(options):
                prompt += f"{option}\n"
            prompt += "\nAnswer with the option's letter from the given choices directly."
        else:
            prompt = f"{question}\n\nAnswer the question with an exact number, which should be accurate to at most two decimal places."

        contents = [
            {"type": "video", "video": meta_data["videos"][0]},
            {"type": "text", "text": prompt},
        ]
        instruction = [{"role": "user", "content": contents}]

        return instruction

    async def process_response(self, data_id, response):
        if self.data_dict[data_id]["options"] is not None:
            options = [re.findall(r"[A-D]\. (.*).", x)[0] for x in self.data_dict[data_id]["options"]]
            letters = ["A", "B", "C", "D"]
            digit2word = {
                "1": "one",
                "2": "two",
                "3": "three",
                "4": "four",
                "5": "five",
                "6": "six",
                "7": "seven",
                "8": "eight",
                "9": "nine",
                "0": "zero",
            }

            response = response.replace("answer", "")
            response = response.replace("Answer", "")
            pred_answer = re.findall(r"[\(\ \[]*([A-Da-d])[\)\.\ \]]*", response)
            find_flag = False
            if len(pred_answer) == 0:
                for idx, opt in enumerate(options):
                    opt = opt.strip()
                    opt = opt.strip(".")
                    # Arabic numerals -> English words
                    opt2 = opt
                    if opt in digit2word:
                        opt2 = digit2word[opt]
                    if opt.lower() in response.lower() or opt2.lower() in response.lower():
                        pred_idx = idx
                        find_flag = True
                        break
            else:
                pred_answer = pred_answer[0].strip()
                pred_answer = pred_answer.strip("()").upper()
                pred_idx = letters.index(pred_answer)
                find_flag = True

            if find_flag:
                prediction = letters[pred_idx]
            else:
                prediction = None

        else:
            prediction = None
            word_to_num = {
                "zero": "0",
                "one": "1",
                "two": "2",
                "three": "3",
                "four": "4",
                "five": "5",
                "six": "6",
                "seven": "7",
                "eight": "8",
                "nine": "9",
                "ten": "10",
                "eleven": "11",
                "twelve": "12",
                "thirteen": "13",
                "fourteen": "14",
                "fifteen": "15",
                "sixteen": "16",
                "seventeen": "17",
                "eighteen": "18",
                "nineteen": "19",
                "twenty": "20",
                "thirty": "30",
                "forty": "40",
                "fifty": "50",
                "sixty": "60",
                "seventy": "70",
                "eighty": "80",
                "ninety": "90",
                "hundred": "100",
            }

            for word, num in word_to_num.items():
                response = re.sub(rf"\b{re.escape(word)}\b", num, response, flags=re.IGNORECASE)

            matches = re.findall(r"[-+]?[0-9]+\.?[0-9]*", response)

            for match in matches:
                if re.fullmatch(r"[-+]?(\d+(\.\d*)?|\.\d+)", match):
                    prediction = match
                    break
        return prediction

    async def get_matching_score(self, data_id, prediction):
        if prediction is None:
            return 0.0
        metadata = self.data_dict[data_id]
        ground_truth = metadata["ground_truth"]
        if metadata["options"] is not None:
            score = int(prediction == ground_truth)
        else:
            seta_group = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            prediction, ground_truth = float(prediction), float(ground_truth)
            score = 0
            for seta in seta_group:
                if abs(prediction - ground_truth) / ground_truth < 1 - seta:
                    score += 0.1
        return score * 100

    def compute_metrics(self, results):
        return self._summarize_scores(results, category_key="task_type")
