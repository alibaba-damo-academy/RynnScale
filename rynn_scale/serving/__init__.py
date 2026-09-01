"""GPU-side inference serving layer (Ray Serve).

:class:`~rynn_scale.serving.server.InferenceServer` (``server``) is preprocessing and
batched inference in one replica, plus the ``InferenceServer.build`` classmethod that
binds it as a Serve deployment. It imports ray + torch + fastapi, so callers import it
from the submodule rather than from this package: re-exporting it here would make
``requests`` (a numpy-only leaf) and ``client`` (heavy deps lazy per transport) pay for
the server's stack, since importing a submodule runs this ``__init__`` first.

Callers reach the model via a ``DeploymentHandle`` (``RayServeClient``); the HTTP
transport is opt-in -- the routes are on the same class, and ``api/serve.py`` is the CLI
that runs it under a route prefix.
"""
