import os
import re
from copy import deepcopy

from .base import BaseBenchmark
from ..registry import BENCHMARK_REGISTRY


@BENCHMARK_REGISTRY.register()
class VideoMME(BaseBenchmark):
    def load_data(self, data_root):
        import pysubs2
        from pyarrow import parquet as pq

        parquet_file = os.path.join(data_root, "test-00000-of-00001.parquet")
        table = pq.read_table(parquet_file)
        df = table.to_pandas()

        video_folder = os.path.join(data_root, "videos")
        subtitle_folder = os.path.join(data_root, "subtitles")
        data_dict = {}

        for record in df.itertuples():
            video_id = record.videoID
            for video_format in ["mp4", "avi", "mov", "mkv"]:
                temp_path = os.path.join(video_folder, f"{video_id}.{video_format}")
                if os.path.exists(temp_path):
                    video_path = temp_path
                    break
            assert os.path.exists(video_path), f"Cannot find the video file: {video_id}"

            meta_data = {
                # required fields for data loading
                "videos": [video_path],
                # required fields for evaluation
                "task_type": record.task_type,
                "ground_truth": record.answer,
                # custom fields for instruction generation and post processing
                "question": record.question,
                "options": list(record.options),
                "question_id": record.question_id,
            }

            data_dict[record.question_id + "_wo_sub"] = meta_data

            subtitle_path = os.path.join(subtitle_folder, f"{video_id}.srt")
            if os.path.exists(subtitle_path):
                subtitles = pysubs2.load(subtitle_path, encoding="utf-8")
            else:
                subtitles = None

            meta_data = deepcopy(meta_data)
            meta_data["subtitles"] = subtitles
            data_dict[record.question_id + "_w_sub"] = meta_data

        return data_dict

    def generate_instruction(self, data_id):
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        options = meta_data["options"]

        prompt = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\n"
        prompt += f"{question}\n"
        for option_idx, option in enumerate(options):
            prompt += f"{option}\n"
        prompt += "Answer with the option's letter from the given choices directly and only give the best option. The best answer is:"

        if "subtitles" in meta_data and meta_data["subtitles"] is not None:
            # selected_subtitles = []
            # for timestamp in timestamps:
            #     sub_text = ""
            #     for subtitle in subtitles:
            #         if subtitle.start < timestamp * 1000 < subtitle.end:
            #             sub_text = subtitle.text.replace("\\N", " ")
            #             break
            #     if sub_text.strip():
            #         selected_subtitles.append(sub_text)

            selected_subtitles = []
            for subtitle in meta_data["subtitles"]:
                selected_subtitles.append(subtitle.text.replace("\\N", " "))
            subtitle_string = "\n".join(selected_subtitles)
            prompt = f"This video's subtitles are listed below:\n{subtitle_string}\n" + prompt

        contents = [{"type": "video", "video": video} for video in meta_data["videos"]]
        contents.append({"type": "text", "text": prompt})
        instruction = [{"role": "user", "content": contents}]

        return instruction

    async def process_response(self, data_id, response):
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
        pred_answer = re.findall(r"[\(\ \[]*([A-D])[\)\.\ \]]*", response)

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
            pred_answer = pred_answer.strip("()")
            pred_idx = letters.index(pred_answer)
            find_flag = True

        assert find_flag, f"Cannot find the answer in the options: {response}"
        prediction = letters[pred_idx]

        prediction = prediction.strip()
        answer_prefixes = [
            "The best answer is",
            "The correct answer is",
            "The answer is",
            "The answer",
            "The best option isThe correct option is",
            "Best answer:Best option:",
        ]
        for answer_prefix in answer_prefixes:
            prediction = prediction.replace(answer_prefix, "")

        if len(prediction.split()) > 10 and not re.search("[ABCD]", prediction):
            raise ValueError(f"Cannot find the answer in the options: {prediction}")
        matches = re.search(r"[ABCD]", prediction)
        if matches is None:
            raise ValueError(f"Cannot find the answer in the options: {prediction}")
        prediction = matches[0]

        return prediction

    async def get_matching_score(self, data_id, prediction):
        ground_truth = self.data_dict[data_id]["ground_truth"]
        match = prediction == ground_truth
        return int(match) * 100

    def compute_metrics(self, results):
        results_wo_sub, results_w_sub = [], []
        for data in results:
            if "subtitles" in self.data_dict[data["data_id"]]:
                results_w_sub.append(data)
            else:
                results_wo_sub.append(data)

        metrics = {}
        if len(results_wo_sub):
            metrics["without_subtitles"] = self._summarize_scores(results_wo_sub, category_key="task_type")
        if len(results_w_sub):
            metrics["with_subtitles"] = self._summarize_scores(results_w_sub, category_key="task_type")

        return metrics
