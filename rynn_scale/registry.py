from typing import Optional, List


class Registry(object):
    def __init__(self):
        self._objects = dict()

    def _add(self, name: str, obj: object):
        if name in self._objects:
            raise ValueError(f"Object {name} is already registered.")
        self._objects[name] = obj

    def register(self, name: Optional[str] = None, obj: Optional[object] = None):
        if obj is not None:
            name = obj.__name__ if name is None else name
            self._add(name, obj)
            return

        def decorator(cls_or_fn: object):
            obj_name = cls_or_fn.__name__ if name is None else name
            self._add(obj_name, cls_or_fn)
            return cls_or_fn

        return decorator

    def keys(self) -> List[str]:
        return list(self._objects.keys())

    def __getitem__(self, name: str) -> object:
        if name not in self._objects:
            raise KeyError(f"Object {name} is not registered.")
        return self._objects[name]


DATASET_REGISTRY = Registry()
BENCHMARK_REGISTRY = Registry()
INFERENCE_WRAPPER_REGISTRY = Registry()
