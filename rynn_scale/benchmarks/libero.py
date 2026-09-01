import os
from contextlib import contextmanager

import numpy as np
import torch

from ..registry import BENCHMARK_REGISTRY
from .base import BaseBenchmark

__all__ = ["LiberoSpatial", "LiberoObject", "LiberoGoal", "Libero90", "Libero10", "Libero100"]


@contextmanager
def numpy_safe_globals():
    """Let ``torch.load(weights_only=True)`` rebuild pickles of plain numpy arrays."""
    try:
        from numpy._core.multiarray import _reconstruct  # numpy >= 2
    except ImportError:
        from numpy.core.multiarray import _reconstruct

    allowlist = [
        # ``safe_globals`` matches on the path recorded in the pickle, which is
        # ``numpy.core`` for files written by numpy 1.x and ``numpy._core`` for 2.x --
        # neither necessarily the installed module's own path, so pin both.
        (_reconstruct, "numpy.core.multiarray._reconstruct"),
        (_reconstruct, "numpy._core.multiarray._reconstruct"),
        np.ndarray,
        np.dtype,
        # A pickled array carries its concrete dtype class, not just ``np.dtype``.
        *(getattr(np.dtypes, name) for name in np.dtypes.__all__),
    ]
    with torch.serialization.safe_globals(allowlist):
        yield


class LiberoBase(BaseBenchmark):
    benchmark_name: str = ""

    def load_data(self, data_root):
        data_dict = {}

        from libero.libero import benchmark, get_libero_path

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.benchmark_name]()

        for i in range(task_suite.n_tasks):
            task = task_suite.get_task(i)
            bddl_file_name = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

            # LIBERO's ``get_task_init_states`` torch.load()s a pickled numpy array,
            # which ``weights_only=True`` (torch>=2.6 default) refuses without an
            # allowlist for the numpy globals it rebuilds.
            with numpy_safe_globals():
                init_states = task_suite.get_task_init_states(i)

            for j in range(50):
                data_dict[f"{i}_{j}"] = {
                    "task_type": task.name,
                    "instruction": task.language,
                    "bddl_file_name": bddl_file_name,
                    "init_state": init_states[j % init_states.shape[0]],
                }

        return data_dict

    def generate_instruction(self, data_id):
        return self.data_dict[data_id]["instruction"]

    def get_agent_config(self, data_id):
        # ``bddl_file_name`` fixes the MuJoCo model (objects, regions, success
        # predicate) -> env ctor; ``init_state`` only writes qpos/qvel inside it (the
        # 7 arm joints + the object placements) -> env reset. One bddl per task, 50
        # init states per bddl.
        return {
            "type": "RobotAgent",
            "env_type": "Libero",
            "env_config": {"bddl_file_name": self.data_dict[data_id]["bddl_file_name"]},
            "reset_config": {"init_state": self.data_dict[data_id]["init_state"]},
            # LIBERO's own protocol: 600 command steps per episode.
            "max_steps": 600,
        }

    async def process_response(self, data_id, response):
        return response

    async def get_matching_score(self, data_id, prediction):
        success = prediction["success"]
        return int(success) * 100

    def compute_metrics(self, results):
        return self._summarize_scores(results, category_key="task_type")


@BENCHMARK_REGISTRY.register()
class LiberoSpatial(LiberoBase):
    benchmark_name = "libero_spatial"


@BENCHMARK_REGISTRY.register()
class LiberoObject(LiberoBase):
    benchmark_name = "libero_object"


@BENCHMARK_REGISTRY.register()
class LiberoGoal(LiberoBase):
    benchmark_name = "libero_goal"


@BENCHMARK_REGISTRY.register()
class Libero90(LiberoBase):
    benchmark_name = "libero_90"


@BENCHMARK_REGISTRY.register()
class Libero10(LiberoBase):
    benchmark_name = "libero_10"


@BENCHMARK_REGISTRY.register()
class Libero100(LiberoBase):
    benchmark_name = "libero_100"
