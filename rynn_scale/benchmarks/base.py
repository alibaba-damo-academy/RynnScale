import json
import os
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Union

from openai import AsyncOpenAI
from torch.utils.data import Dataset


class BaseBenchmark(Dataset, metaclass=ABCMeta):
    def __init__(
        self,
        data_root: str,
        prompt_format: Optional[str] = None,
        enable_thinking: bool = False,
    ) -> None:
        self.prompt_format = prompt_format
        self.enable_thinking = enable_thinking

        if os.getenv("ENDPOINT_URL", None):
            self.openai_client = AsyncOpenAI(
                base_url=os.getenv("ENDPOINT_URL", None),
                api_key=os.getenv("OPENAI_API_KEY", None),
            )
        else:
            self.openai_client = None

        data_dict = self.load_data(data_root)
        for key, value in data_dict.items():
            data_dict[key] = MappingProxyType(value)
        data_dict = MappingProxyType(data_dict)
        self.data_dict = data_dict

        aggregated_data = dict()
        for data_id, meta_data in self.data_dict.items():
            mm_items = []
            for key in ["images", "videos"]:
                if key not in meta_data:
                    continue
                if isinstance(meta_data[key], (list, tuple)):
                    mm_items.extend(meta_data[key])
                else:
                    mm_items.append(meta_data[key])

            if len(mm_items) == 0:
                aggregated_id = data_id
            else:
                try:
                    aggregated_id = json.dumps(mm_items)
                except Exception:
                    aggregated_id = data_id

            if aggregated_id not in aggregated_data:
                aggregated_data[aggregated_id] = {
                    "data_ids": [data_id],
                    "images": meta_data.get("images", None),
                    "videos": meta_data.get("videos", None),
                }
            else:
                aggregated_data[aggregated_id]["data_ids"].append(data_id)

        self._aggregated_data = [x for _, x in aggregated_data.items()]

    @property
    def n_samples(self) -> int:
        return len(self.data_dict)

    def __len__(self) -> int:
        return len(self._aggregated_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        output = {
            "data_ids": self._aggregated_data[idx]["data_ids"],
            "enable_thinking": self.enable_thinking,
            "conversations": [
                self.generate_instruction(data_id) for data_id in self._aggregated_data[idx]["data_ids"]
            ],
        }
        return output

    @abstractmethod
    def load_data(self, data_root) -> Dict[Union[int, str], Any]:
        """
        Load the dataset meta data.

        Args:
            data_root (str): path to the dataset.

        Returns:
            data_dict (Dict[Union[int, str], Any]): dataset meta data, with data_id as key.
            example:
            {
                0: {
                    # required fields for data loading
                    "video_path": os.path.join(video_folder, data["video"]),
                    "start_time": data["start"] if task_info[3] else None,
                    "end_time": data["end"] if task_info[3] else None,
                    # required fields for evaluation
                    "task_type": task_name,
                    "ground_truth": answer_idx,
                    # custom fields for instruction generation and post processing
                    "question": data["question"],
                    "options": options,
                    "option_letters": option_letters,
                }
                ...
            }
        """
        pass

    @abstractmethod
    def generate_instruction(self, data_id: Union[int, str]) -> List[Dict[str, Any]]:
        """
        Generate instruction(s) for model inference.

        Args:
            data_id (Union[int, str]): identifier of the data.

        Returns:
            instruction (Union[str, Dict[str, str]]): instruction(s) for model inference.
        """
        pass

    @abstractmethod
    async def process_response(self, data_id: Union[int, str], response: str) -> Any:
        """
        Process the original model responses to desired format for evaluation and visualization.

        Args:
            data_id (Union[int, str]): identifier of the data.
            response (str): model response.

        Returns:
            result (Any): processed model response for evaluation.
        """

    @abstractmethod
    async def get_matching_score(self, data_id: Union[int, str], prediction: Any) -> Any:
        """
        Compute the matching score between model prediction and ground truth.

        Args:
            data_id (Union[int, str]): identifier of the data.
            prediction (Any): processed model response.

        Returns:
            score (Any): computed score.
        """
        pass

    @abstractmethod
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute overall metrics for the benchmark.

        Args:
            results (List[Dict[str, Any]]): list of individual results.
        Returns:
            metrics (Dict[str, Any]): computed overall metrics.
        """
        pass

    def _summarize_scores(
        self,
        results: List[Dict[str, Any]],
        category_key: Optional[str] = None,
    ) -> Dict[str, float]:
        scores = defaultdict(list)

        for data in results:
            data_id = data["data_id"]
            score = data["score"]
            if category_key is not None:
                categories = self.data_dict[data_id][category_key]
                if not isinstance(categories, (list, tuple)):
                    categories = [categories]
                for category in categories:
                    scores[category].append(score)
            else:
                scores[""].append(score)

        if category_key is not None:
            reduced_scores = {category: sum(score_list) / len(score_list) for category, score_list in scores.items()}
        else:
            reduced_scores = {}

        overall_scores = sum(scores.values(), [])
        reduced_scores["Overall"] = sum(overall_scores) / len(overall_scores)

        return reduced_scores
