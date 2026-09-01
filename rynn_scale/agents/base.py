import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseAgent(ABC):
    """One way of turning a benchmark sample into one scored result.

    ``model`` is the inference client the agent generates through; ``buffer`` is the
    optional rollout-video sink (an ``EpisodeBuffer`` handle); ``episode_id``
    identifies this run of one sample. All three come from the evaluator rather than
    the benchmark -- the id in particular is a plain counter over dispatch order,
    unique for the whole run, which is what lets the buffer key episodes on it
    alone; a caller with nothing to number (deploy, a test) leaves it ``None`` and
    gets a uuid. Subclasses add whatever else *their* kind of episode needs, and the
    benchmark fills those in through
    :meth:`~rynn_scale.benchmarks.base.BaseBenchmark.get_agent_config` -- that
    config is exactly this constructor's remaining kwargs, so an agent is fully
    specified before it runs and :meth:`rollout` only has to start it.

    Subclasses are hosted in one of two formats, and the class *says which* rather
    than being wrapped into it: :attr:`dedicated_process` asks for a process per
    episode (a simulator needs that), and left ``False`` the agent runs on the
    evaluator's own event loop. :func:`~rynn_scale.evaluation.evaluator.episode_agent`
    reads these two attributes and hosts accordingly, so the registry only ever holds
    plain classes and any of them can also be constructed directly (deploy does).
    """

    #: Does one episode of this agent need a process of its own? ``True`` makes the
    #: evaluator wrap the class in ``ray.remote`` for an actor per episode, which is
    #: what anything owning a blocking, unkillable resource requires -- a simulator's
    #: GL context, a wedged physics step, a robot connection. ``False`` (the default)
    #: runs it on the shared event loop, so such an agent must own nothing that
    #: blocks: there is no process to kill.
    dedicated_process: bool = False

    #: ``ray.remote`` kwargs for that wrapping, and only the ones that are the
    #: *class's* to state -- in practice ``concurrency_groups``, which name the
    #: threads its own ``ray.method`` annotations refer to. The episode's terms
    #: (``num_cpus``, ``max_concurrency``, ``max_restarts``) belong to whoever hosts
    #: the episode and are applied there. Ignored unless
    #: :attr:`dedicated_process`.
    ray_actor_options: Dict[str, Any] = {}

    def __init__(self, model: Any, buffer: Any = None, episode_id: Optional[str] = None):
        self.model = model
        self.buffer = buffer
        # ``None`` means nobody is numbering episodes -- deploy, a script, a test --
        # so make one up rather than leaving the id empty: it names this run's video
        # and every log line about it, and the buffer keys episodes on it.
        self.episode_id = episode_id or uuid.uuid4().hex

    @abstractmethod
    async def rollout(self, prompt: str = "") -> Any:
        """Run one sample; return the payload the benchmark scores.

        ``async def`` is required rather than stylistic: the evaluator awaits this,
        and for an actor agent Ray picks the actor's whole threading model from
        whether the class has any coroutine methods.
        """
