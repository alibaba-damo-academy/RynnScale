import asyncio

from datasets import load_dataset

from .base import BaseBenchmark
from ..registry import BENCHMARK_REGISTRY


@BENCHMARK_REGISTRY.register()
class RealWorldQA(BaseBenchmark):
    def load_data(self, data_root):
        data_dict = {}

        data_list = load_dataset(data_root, split="test")

        for question_id, data in enumerate(data_list):
            data_dict[question_id] = {
                # required fields for data loading
                "images": [data["image"].convert("RGB")],
                # required fields for evaluation
                "ground_truth": data["answer"].strip().lower(),
                # custom fields for instruction generation and post processing
                "question": data["question"],
            }

        return data_dict

    def generate_instruction(self, data_id):
        meta_data = self.data_dict[data_id]
        contents = [{"type": "image", "image": image} for image in meta_data["images"]]
        contents.append({"type": "text", "text": meta_data["question"]})
        instruction = [{"role": "user", "content": contents}]
        return instruction

    async def process_response(self, data_id, response):
        response = response.strip().lower().rstrip(".")
        return response

    async def _get_chat_response(self, promot, n=1, patience=10000000, sleep_time=0):
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
                    max_tokens=800,
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
                    await asyncio.sleep(sleep_time)
        return ""

    def _build_prompt(self, gt, prediction):
        tmpl = (
            "You are an AI assistant. Help me determine whether my answer is correct compared to the ground truth."
            "Note that answers that are not exactly the same but have the same meaning should still be considered correct."
            "Please output according to the specified template."
            "Output template: True means correct, False means incorrect. Only output True or False.\n"
            "Example 1: \n"
            "Ground Truth: yes, Answer: no, Your output: False\n"
            "Example 2: \n"
            "Ground Truth: three, Answer: 3, Your output: True\n"
            "Example 3: \n"
            "Ground Truth: 2, Answer: two, Your output: True\n"
            "Ground Truth: {}, Answer: {}, Your output: "
        )
        return tmpl.format(gt, prediction)

    async def _judge(self, gt, prediction):
        prompt = self._build_prompt(gt, prediction)
        retry_limit = 10

        for retry in range(retry_limit):
            try:
                extraction = await self._get_chat_response(prompt, patience=10)
                return extraction
            except Exception:
                await asyncio.sleep(1)
        return "False"

    async def get_matching_score(self, data_id, prediction):
        ground_truth = self.data_dict[data_id]["ground_truth"]
        if prediction == ground_truth:
            correct = True
        else:
            judge = await self._judge(ground_truth, prediction)
            correct = "true" in judge.lower()
        return int(correct) * 100

    def compute_metrics(self, results):
        return self._summarize_scores(results)
