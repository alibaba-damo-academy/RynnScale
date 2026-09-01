import functools
import os
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Callable, Optional
from weakref import WeakKeyDictionary


def fork_safe_cache(
    method: Optional[Callable] = None,
    *,
    maxsize: Optional[int] = None,
) -> Callable:
    """Method decorator: memoise by `(args, kwargs)`, per-instance, per-process.

    Why fork-safe: PyTorch DataLoader workers use `fork` (default on Linux),
    so an instance and any open file handles it owns are inherited by each
    worker. HDF5 / video readers / network sockets are typically NOT
    fork-safe — sharing one handle across processes causes races, corruption
    or crashes. This decorator detects "I'm running in a new process" via
    `os.getpid()` and rebuilds the cache, so each worker opens its own
    resources without subclasses needing to write per-pid bookkeeping.

    Cache state lives in a closure (one `WeakKeyDictionary` per decorated
    method); when the owning instance is GC'd, its entries are dropped
    automatically. All positional and keyword arguments must be hashable.

    Args:
        maxsize: optional max number of entries per instance. If set, oldest
            entries are evicted (true LRU). Default `None` = unbounded.

    Both forms are supported::

        @fork_safe_cache
        def _open(self, path): ...

        @fork_safe_cache(maxsize=128)
        def _decode(self, path, ts): ...
    """

    def decorator(method: Callable) -> Callable:
        cache_by_instance: "WeakKeyDictionary[Any, OrderedDict[Any, Any]]" = WeakKeyDictionary()
        pid_by_instance: "WeakKeyDictionary[Any, int]" = WeakKeyDictionary()

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            pid = os.getpid()
            if pid_by_instance.get(self) != pid:
                cache_by_instance[self] = OrderedDict()
                pid_by_instance[self] = pid
            cache = cache_by_instance[self]
            key = (args, tuple(sorted(kwargs.items())))
            if key in cache:
                if maxsize is not None:
                    cache.move_to_end(key)
                return cache[key]
            value = method(self, *args, **kwargs)
            cache[key] = value
            if maxsize is not None and len(cache) > maxsize:
                cache.popitem(last=False)
            return value

        return wrapper

    if method is not None:
        return decorator(method)
    return decorator


@contextmanager
def suppress_hf_progress():
    from datasets.utils.logging import (
        disable_progress_bar,
        enable_progress_bar,
        is_progress_bar_enabled,
    )

    was_enabled = is_progress_bar_enabled()
    disable_progress_bar()
    try:
        yield
    finally:
        if was_enabled:
            enable_progress_bar()
