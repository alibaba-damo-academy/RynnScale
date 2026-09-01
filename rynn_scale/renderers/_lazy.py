"""Deferred ``mujoco`` import for the renderers.

``renderers/__init__.py`` eagerly imports every renderer module, so a top-level
``import mujoco`` in each of them pulls the native library (and its GL bindings)
into any process that merely touches the renderer registry. Every renderer only
uses ``mujoco`` inside function bodies, so the import can be deferred to first
attribute access instead:

    from ._lazy import mujoco     # instead of: import mujoco

Call sites are unchanged -- ``mujoco.MjSpec``, ``mujoco.mj_forward``, ... all
still work, they just trigger the real import the first time one is read.
"""


class _LazyMujoco:
    """Proxy that imports ``mujoco`` on first attribute access."""

    _mod = None

    def __getattr__(self, name):
        mod = _LazyMujoco._mod
        if mod is None:
            import mujoco as mod  # noqa: PLC0415

            _LazyMujoco._mod = mod
        return getattr(mod, name)


mujoco = _LazyMujoco()
