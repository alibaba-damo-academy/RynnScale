from typing import Optional

from ..inference_wrappers import BaseInferenceWrapper
from .ai2d import AI2D
from .base import BaseBenchmark
from .chart_qa import ChartQA
from .configs import DATA_PATH
from .cosmos_reason1 import CosmosReason1
from .doc_vqa import DocVQA
from .ecbench import ECBench
from .ego_task_qa import EgoTaskQA
from .ego_text_vqa import EgoTextVQAIndoor
from .egoschema import EgoSchema
from .emb_spatial import EmbSpatial
from .erqa import ERQA
from .info_vqa import InfoVQA
from .libero import *
from .mind_cube import MindCube
from .mmsi import MMSI
from .mv_bench import MVBench
from .openx_vqa import OpenXVQA
from .qa_ego4d import QAEgo4D
from .real_world_qa import RealWorldQA
from .refspatial import RefSpatial
from .robospatial import RoboSpatial
from .rynnbrain_cog import RynnBrainCog
from .rynnbrain_loc import RynnBrainLoc
from .share_robot import ShareRobot
from .video_mme import VideoMME
from .vsibench import VSIBench
from .where2place import Where2Place


def build_benchmark(
    benchmark: str,
    inference_wrapper: BaseInferenceWrapper,
    prompt_format: Optional[str] = None,
    enable_thinking: bool = False,
) -> BaseBenchmark:
    from ..registry import BENCHMARK_REGISTRY

    return BENCHMARK_REGISTRY[benchmark](
        data_root=DATA_PATH.get(benchmark, None),
        inference_wrapper=inference_wrapper,
        prompt_format=prompt_format,
        enable_thinking=enable_thinking,
    )
