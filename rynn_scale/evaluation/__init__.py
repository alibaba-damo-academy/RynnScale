"""Eval orchestration: the evaluator, its placement planning and its video buffer.

The policy clients are re-exported here for convenience; everything heavier is
loaded on first attribute access (PEP 562), for the same reason as in
``rynn_scale/__init__.py``: ``Evaluator`` pulls the whole model + benchmark stack
and ``EpisodeBuffer`` pulls ray, while ``placement`` is a pure planner that a test
(or a sizing script) should be able to import on its own.
"""

import importlib
from typing import Any, List

from ..serving.client import InferenceClient, RayServeClient

__all__ = [
    "Evaluator",
    "InferenceClient",
    "RayServeClient",
    "EpisodeBuffer",
    "Episode",
    "observation_to_frame",
]

# attribute -> submodule it lives in
_LAZY = {
    "Evaluator": ".evaluator",
    "EpisodeBuffer": ".episode_buffer",
    "Episode": ".episode_buffer",
    "observation_to_frame": ".episode_buffer",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_LAZY))
