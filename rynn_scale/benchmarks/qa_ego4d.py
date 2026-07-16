import json
import os

from ..registry import BENCHMARK_REGISTRY
from .base import BaseBenchmark


def make_prompt(question, ground_truth, pred_answer):
    prompt = f"##question: {question} ##groundtruth: {ground_truth} ##predicted answer: {pred_answer}."
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Please evaluate the predicted answer based on the given question. A score of 0 means the answer is completely incorrect; a score of 5 means it is completely correct; a score of 3 means it is partially correct. Output only a single score from the following set: [0, 1, 2, 3, 4, 5].",
        },
        {"role": "user", "content": prompt},
    ]
    return messages


@BENCHMARK_REGISTRY.register()
class QAEgo4D(BaseBenchmark):
    def load_data(self, data_root):
        data_dict = {}

        video_folder = os.path.join(data_root, "videos")
        json_file = os.path.join(data_root, "annotations.test_cut.json")
        with open(json_file, "r") as f:
            data_list = json.load(f)

        for data in data_list:
            video_path = os.path.join(video_folder, data["new_video"])
            if not os.path.exists(video_path):
                continue

            data_dict[data["sample_id"]] = {
                # required fields for data loading
                "videos": [video_path],
                # custom fields for instruction generation and post processing
                "question": data["question"],
                "ground_truth": data["answer"],
            }

        return data_dict

    def generate_instruction(self, data_id):
        meta_data = self.data_dict[data_id]
        contents = [{"type": "video", "video": video} for video in meta_data["videos"]]
        contents.append({"type": "text", "text": meta_data["question"]})
        instruction = [{"role": "user", "content": contents}]
        return instruction

    async def process_response(self, data_id, response):
        return response.strip()

    async def get_matching_score(self, data_id, prediction):
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        ground_truth = meta_data["ground_truth"]

        if prediction.lower() == ground_truth.lower():
            score = 1.0
        else:
            messages = make_prompt(question, ground_truth, prediction)
            try:
                completion = await self.openai_client.chat.completions.create(
                    model="gpt-4o-0806",
                    messages=messages,
                    max_completion_tokens=100,
                )
                response = completion.choices[0].message.content
                score = float(int(response) / 5)

            except Exception:
                print(f"Failed to process item with question: {question}")
                print(f"Prompt messages sent to model: {messages}")
                score = 0.0

        return score * 100

    def compute_metrics(self, results):
        return self._summarize_scores(results)
