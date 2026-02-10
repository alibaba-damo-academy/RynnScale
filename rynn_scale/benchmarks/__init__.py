from typing import Optional

from .configs import DATA_PATH
from .ai2d import AI2D
from .base import BaseBenchmark
from .chart_qa import ChartQA
from .doc_vqa import DocVQA
from .ecbench import ECBench
from .egoschema import EgoSchema
from .ego_task_qa import EgoTaskQA
from .ego_text_vqa import EgoTextVQAIndoor
from .erqa import ERQA
from .info_vqa import InfoVQA
from .mind_cube import MindCube
from .mmsi import MMSI
from .mv_bench import MVBench
from .qa_ego4d import QAEgo4D
from .openx_vqa import OpenXVQA
from .refspatial import RefSpatial
from .robospatial import RoboSpatial
from .real_world_qa import RealWorldQA
from .rynnbrain_loc import RynnBrainLoc
from .rynnbrain_cog import RynnBrainCog
from .share_robot import ShareRobot
from .video_mme import VideoMME
from .vsibench import VSIBench
from ..registry import BENCHMARK_REGISTRY


def build_benchmark(
    benchmark: str,
    use_cot: bool,
    prompt_format: Optional[str] = None,
) -> BaseBenchmark:
    return BENCHMARK_REGISTRY[benchmark](
        data_root=DATA_PATH[benchmark],
        use_cot=use_cot,
        prompt_format=prompt_format,
    )
