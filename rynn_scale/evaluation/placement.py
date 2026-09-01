"""Dynamic placement planning for the eval Serve Model deployment.

Used only by :class:`~rynn_scale.evaluation.evaluator.Evaluator` -- serve/control
set their replica count explicitly. Planned from current cluster resources:

  * **Model replicas** -- each takes ``tp_size * pp_size`` GPUs (Ray keeps one
    actor's GPUs on one node); replicas are spread so **every GPU is occupied**
    (``num_replicas = sum_node(⌊node_gpus / (tp*pp)⌋)``), which is what eval
    wants -- more replicas = more concurrent episodes served.

The plan feeds :func:`model_deployment_kwargs` -> ``InferenceServer.build``.

Agent (episode) placement is *not* planned here. An env-owning agent is one actor
per episode (:func:`~rynn_scale.evaluation.evaluator.episode_agent`), scheduled
wherever a CPU is free when it starts, and how many run at once is
``max_concurrent_episodes`` as given. ``agents_per_node`` / ``plan.agent_node_ids``
remain only so the plan shape is unchanged; the evaluator passes the minimum and
ignores them.
"""

from dataclasses import dataclass, field
from typing import List, Optional


def _short(node_id: str) -> str:
    return node_id[:8] if node_id else "?"


@dataclass
class NodePlan:
    node_id: str
    num_gpus: int
    num_cpus: int
    model_replicas: int
    agent_workers: int
    idle_gpus: int


@dataclass
class PlacementPlan:
    gpus_per_model: int
    num_model_replicas: int
    max_replicas_per_node: int
    agent_node_ids: List[str] = field(default_factory=list)
    nodes: List[NodePlan] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def num_agents(self) -> int:
        return len(self.agent_node_ids)


def plan_placement(
    tp_size: int = 1,
    pp_size: int = 1,
    agents_per_node: int = 1,
    *,
    agent_cpus: float = 1.0,
    model_cpus: float = 1.0,
    nodes: Optional[list] = None,
) -> PlacementPlan:
    """Plan model-replica + agent-worker placement over the cluster.

    ``agents_per_node`` is a manual knob: exactly that many agent workers are
    placed on **each** model-hosting node (uniform distribution). ``nodes``
    defaults to the live nodes from ``ray.nodes()``; pass a list of
    ``{"NodeID", "Resources": {"GPU", "CPU"}}`` dicts to plan offline/in tests.
    """
    gpus_per_model = tp_size * pp_size
    assert gpus_per_model >= 1, "tp_size * pp_size must be >= 1"
    assert agents_per_node >= 1, "agents_per_node must be >= 1"

    if nodes is None:
        import ray

        nodes = [n for n in ray.nodes() if n.get("Alive")]

    node_plans: List[NodePlan] = []
    agent_ids: List[str] = []
    warnings: List[str] = []

    for n in nodes:
        node_id = n.get("NodeID", "")
        res = n.get("Resources", {})
        g = int(res.get("GPU", 0))
        c = int(res.get("CPU", 0))

        if g < gpus_per_model:
            if g > 0:
                warnings.append(f"node {_short(node_id)}: {g} GPU < tp*pp={gpus_per_model}; unused for model")
            continue

        replicas = g // gpus_per_model
        idle = g - replicas * gpus_per_model
        if idle:
            warnings.append(
                f"node {_short(node_id)}: {idle} GPU idle (GPU={g} not divisible by tp*pp={gpus_per_model})"
            )

        # Manual, uniform: agents_per_node on every model-hosting node. Honor the
        # count; warn (don't cap) if CPU looks insufficient -- Ray will leave
        # over-subscribed actors pending rather than corrupt the plan.
        cpu_free = c - replicas * model_cpus
        if agents_per_node * agent_cpus > max(0.0, cpu_free):
            warnings.append(
                f"node {_short(node_id)}: {agents_per_node} agents need "
                f"{agents_per_node * agent_cpus:g} CPU but only {cpu_free:g} free "
                f"after model (CPU={c}); some agent actors may stay pending"
            )

        agent_ids.extend([node_id] * agents_per_node)
        node_plans.append(NodePlan(node_id, g, c, replicas, agents_per_node, idle))

    total = sum(p.model_replicas for p in node_plans)
    max_rpn = max((p.model_replicas for p in node_plans), default=0)

    distinct = {p.model_replicas for p in node_plans}
    if len(distinct) > 1:
        warnings.append(
            f"heterogeneous GPU nodes (replicas/node={sorted(distinct)}); a single "
            f"max_replicas_per_node={max_rpn} lets Serve fill big nodes but may idle "
            f"GPUs on smaller ones -- use per-node placement groups if that matters"
        )
    if total == 0:
        warnings.append("no node can host a model replica (check tp*pp vs per-node GPUs)")

    return PlacementPlan(
        gpus_per_model=gpus_per_model,
        num_model_replicas=total,
        max_replicas_per_node=max_rpn,
        agent_node_ids=agent_ids,
        nodes=node_plans,
        warnings=warnings,
    )


def model_deployment_kwargs(plan: PlacementPlan) -> dict:
    """kwargs for ``InferenceServer.build`` so its replicas fill every GPU evenly."""
    return {
        "num_model_replicas": plan.num_model_replicas,
        "model_num_gpus": plan.gpus_per_model,
        "max_replicas_per_node": plan.max_replicas_per_node,
    }
