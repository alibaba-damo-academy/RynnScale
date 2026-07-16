import json
import os
import time
from typing import Any, Dict, Optional

from ..registry import BENCHMARK_REGISTRY
from .base import BaseBenchmark


@BENCHMARK_REGISTRY.register()
class ChartQA(BaseBenchmark):
    def load_data(self, data_root: str) -> Dict[int, Any]:
        data_dict = {}

        json_paths = {"Augmented": "test_augmented_renamed.jsonl", "Human": "test_human_renamed.jsonl"}
        for task_type, json_path in json_paths.items():
            json_file = os.path.join(data_root, json_path)
            data_list = [json.loads(item.strip()) for item in open(json_file).readlines()]

            for data in data_list:
                image_path = os.path.join(data_root, data["image"])
                assert os.path.exists(image_path), f"Cannot find the image file: {image_path}"

                data_dict[data["question_id"]] = {
                    "images": [image_path],
                    "ground_truth": data["answer"],
                    "task_type": task_type,
                    "question": data["question"],
                }

        return data_dict

    def generate_instruction(self, data_id):
        meta_data = self.data_dict[data_id]
        contents = [{"type": "image", "image": image} for image in meta_data["images"]]
        contents.append(
            {"type": "text", "text": meta_data["question"] + "\nAnswer the question using a single word or phrase."}
        )
        instruction = [{"role": "user", "content": contents}]
        return instruction

    async def process_response(self, data_id, response):
        return response.strip()

    async def get_matching_score(self, data_id, prediction):
        question = self.data_dict[data_id]["question"]
        ground_truth = self.data_dict[data_id]["ground_truth"]
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        score = max([await self.relaxed_correctness(ann, prediction, question) for ann in ground_truth]) * 100.0
        return score

    def compute_metrics(self, results):
        return self._summarize_scores(results, category_key="task_type")

    async def relaxed_correctness(
        self,
        target: str,
        prediction: str,
        question: str,
        max_relative_change: float = 0.05,
    ) -> bool:
        def _to_float(text: str) -> Optional[float]:
            try:
                if text.endswith("%"):
                    # Convert percentages to floats.
                    return float(text.rstrip("%")) / 100.0
                else:
                    return float(text)
            except ValueError:
                return None

        prediction_float = _to_float(prediction)
        target_float = _to_float(target)
        if prediction_float is not None and target_float:
            relative_change = abs(prediction_float - target_float) / abs(target_float)
            relative_change1 = abs(prediction_float - target_float / 100) / abs(target_float / 100)
            relative_change2 = abs(prediction_float / 100 - target_float) / abs(target_float)
            return (
                relative_change <= max_relative_change
                or relative_change1 <= max_relative_change
                or relative_change2 <= max_relative_change
            )
        else:
            return prediction.lower() == target.lower() or await self.match(target, prediction, question)

    async def match(self, gt, prediction, question):
        prompt = self.build_prompt(gt, prediction, question)
        retry_limit = 10

        for retry in range(retry_limit):
            try:
                extraction = await self.get_chat_response(prompt, patience=10)
                return "true" in extraction
            except Exception:
                time.sleep(1)
        return False

    def build_prompt(self, gt, prediction, question):
        tmpl = (
            "You are an AI assistant. Help me determine whether my answer is correct compared to the ground truth."
            "Note that answers that are not exactly the same but have the same meaning or accurately answer the question should still be considered correct."
            "Please output according to the specified template."
            "Output template: True means correct, False means incorrect. Only output True or False.\n"
            "Example 1: \n"
            "Question: Which colored bar trumps all the bars?, Ground Truth: Dark Blue, Answer: Blue, Your output: False\n"
            "Example 2: \n"
            "Question: Which two values are same in the upper graph?, Ground Truth: [77, 77], Answer: 77, Your output: True\n"
            "Example 3: \n"
            "Question: What's the age strucutre in 2019 for 0-14 and 15-64?, Ground Truth: [42.47, 54.91], Answer: 42.47, Your output: False\n"
            "Example 4: \n"
            "Question: Which animal has the least cost of keeping?, Ground Truth: Rabbit**, Answer: Rabbit, Your output: True\n"
            "Example 5: \n"
            "Question: How many years has number of visitor below 1000?, Ground Truth: 3, Answer: 2, Your output: False\n"
            "Example 6: \n"
            "Question: Does the graph increase or decrease?, Ground Truth: increasing, Answer: Increase, Your output: True\n"
            "Question: {}, Ground Truth: {}, Answer: {}, Your output: "
        )
        return tmpl.format(question, gt, prediction)

    async def get_chat_response(self, promot, n=1, patience=10000000, sleep_time=0):
        messages = [
            {"role": "user", "content": promot},
        ]
        while patience > 0:
            patience -= 1
            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini-0718",
                    messages=messages,
                    temperature=0.7,
                    max_completion_tokens=800,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                )
                if n == 1:
                    prediction = response.choices[0].message.content.strip()
                    if prediction != "" and prediction is not None:
                        return prediction
                else:
                    prediction = [choice.message.content.strip() for choice in response.choices]
                    if prediction[0] != "" and prediction[0] is not None:
                        return prediction

            except Exception as e:
                if "Rate limit" not in str(e):
                    print(e)

                if "Please reduce the length of the messages" in str(e):
                    print("!!Reduce promot size")
                    # reduce input prompt and keep the tail
                    new_size = int(len(promot) * 0.9)
                    new_start = len(promot) - new_size
                    promot = promot[new_start:]
                    messages = [
                        {"role": "user", "content": promot},
                    ]

                if sleep_time > 0:
                    time.sleep(sleep_time)
        return ""
