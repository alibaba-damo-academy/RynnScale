import io
import os
import re

import pandas as pd
from PIL import Image

from ..registry import BENCHMARK_REGISTRY
from .base import BaseBenchmark


@BENCHMARK_REGISTRY.register()
class MMSI(BaseBenchmark):
    def load_data(self, data_root):
        data_dict = {}
        data_path = os.path.join(data_root, "MMSI_Bench.parquet")
        df = pd.read_parquet(data_path)

        for data in df.itertuples():
            options = re.findall(r"([A-D]):\s*(.*?)(?=, [A-Z]:|$)", data.question)
            options = {letter: option.strip() for letter, option in options}
            data_dict[data.id] = {
                "images": [Image.open(io.BytesIO(image)) for image in data.images.tolist()],
                "task_type": data.question_type,
                "question": data.question,
                "options": options,
                "ground_truth": data.answer,
                "difficulty": data.difficulty,
            }

        return data_dict

    def generate_instruction(self, data_id):
        meta_data = self.data_dict[data_id]
        contents = [{"type": "image", "image": image} for image in meta_data["images"]]
        contents.append({"type": "text", "text": meta_data["question"]})
        instruction = [{"role": "user", "content": contents}]
        return instruction

    async def process_response(self, data_id, response):
        matches = re.findall(r"[\(\ ]*[A-D][\)\ ]*", response)
        if not matches:
            options = self.data_dict[data_id]["options"]
            for k, option in options.items():
                if option.lower() in response.lower():
                    prediction = k
                    break
            else:
                prediction = None
        else:
            prediction = matches[0]
        return prediction

    async def get_matching_score(self, data_id, prediction):
        ground_truth = self.data_dict[data_id]["ground_truth"]
        match = prediction == ground_truth
        return int(match) * 100

    def compute_metrics(self, results):
        return self._summarize_scores(results, category_key="task_type")
