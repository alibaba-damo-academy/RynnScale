import json
import os
import re
import requests

from .base import BaseBenchmark
from ..registry import BENCHMARK_REGISTRY


@BENCHMARK_REGISTRY.register()
class EgoSchema(BaseBenchmark):
    def load_data(self, data_root):
        data_dict = {}

        video_folder = os.path.join(data_root, "good_clips_git")
        json_file = os.path.join(data_root, "questions.json")
        with open(json_file, "r") as f:
            data_list = json.load(f)

        for data in data_list:
            question_id = data["q_uid"]
            for video_format in ["mp4", "avi", "mov", "mkv"]:
                video_path = os.path.join(video_folder, f"{question_id}.{video_format}")
                if os.path.exists(video_path):
                    break
            assert os.path.exists(video_path), f"Cannot find the video file: {question_id}"

            data_dict[question_id] = {
                # required fields for data loading
                "videos": [video_path],
                # custom fields for instruction generation and post processing
                "question": data["question"],
                "options": [data[f"option {i}"] for i in range(5)],
            }

        return data_dict

    def generate_instruction(self, data_id):
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        options = meta_data["options"]
        prompt = "Select the best answer to the following multiple-choice question based on the video.\n"
        prompt += f"{question}\nOptions:\n(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}\n(D) {options[3]}\n(E) {options[4]}\n"
        prompt += "Answer with the option's letter from the given choices directly and only give the best option. The best answer is: "

        contents = [
            {"type": "video", "video": meta_data["videos"][0]},
            {"type": "text", "text": prompt},
        ]
        instruction = [{"role": "user", "content": contents}]

        return instruction

    async def process_response(self, data_id, response):
        options = self.data_dict[data_id]["options"]
        letters = ["A", "B", "C", "D", "E"]

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
        return pred_idx

    async def get_matching_score(self, data_id, prediction):
        return None

    def compute_metrics(self, results):
        url = "https://validation-server.onrender.com/api/upload/"
        headers = {"Content-Type": "application/json"}
        submission = {result["data_id"]: result["prediction"] for result in results}

        response = requests.post(url, headers=headers, json=submission)
        assert response.status_code == 200, f"Failed to send POST request. Status code: {response.status_code}"
        matches = re.findall(r"(\d+) correct, (\d+) wrong", response.text)
        assert len(matches) == 2, f"Failed to parse the response: {response.text}"

        total_correct, total_wrong = matches[0]
        total_correct, total_wrong = int(total_correct), int(total_wrong)
        subset_correct, subset_wrong = matches[1]
        subset_correct, subset_wrong = int(subset_correct), int(subset_wrong)
        metrics = {
            "Subset": subset_correct / (subset_correct + subset_wrong) * 100,
            "Total": total_correct / (total_correct + total_wrong) * 100,
        }

        return metrics
