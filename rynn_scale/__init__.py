import importlib
from typing import Any, List

_SUBMODULES = (
    "agents",
    "benchmarks",
    "datasets",
    "environments",
    "inference_wrappers",
    "models",
)


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module  # import once, then plain attribute access
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_SUBMODULES))
