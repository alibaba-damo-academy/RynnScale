import asyncio
import inspect
import itertools
import json
import logging
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Union

from tqdm import tqdm

from ..arguments import EvaluationArguments
from ..benchmarks import BaseBenchmark
from ..inference_wrappers import BaseInferenceWrapper

logger = logging.getLogger(__name__)


def filter_metadata(data: Union[Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any]]:
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            if isinstance(data[key], (dict, list, tuple)):
                new_data[key] = filter_metadata(value)
            elif isinstance(data[key], (int, float, bool, str)):
                new_data[key] = value
        return new_data
    elif isinstance(data, (list, tuple)):
        new_data = []
        for item in data:
            if isinstance(item, (dict, list, tuple)):
                new_data.append(filter_metadata(item))
            elif isinstance(item, (int, float, bool, str)):
                new_data.append(item)
        return new_data
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


# --------------------------------------------------------------- agent hosting

# CPU one env-owning agent's process occupies while its episode runs. Ray's
# scheduler is what this feeds: an agent it cannot place queues in
# ``episode_agent`` until a CPU frees up.
AGENT_CPUS = 1.0

# How long an agent actor may sit unplaced before that is worth a log line. Not a
# deadline -- an episode waits for its CPU for as long as it takes -- purely a
# voice, so a cluster with nothing free does not read as a frozen eval. A worker
# comes up in ~0.15s, plus however long the previous episode's actor takes to
# exit, so anything past this is a wait on something else.
PENDING_WARN_S = 120.0


async def await_ref(ref: Any) -> Any:
    """Await a Ray ObjectRef through a coroutine, so ``asyncio.wait_for`` accepts it."""
    return await ref


@asynccontextmanager
async def episode_agent(
    agent_cls: Any, *args, cpus: float = AGENT_CPUS, pending_warn_s: float = PENDING_WARN_S, **kwargs
):
    """Build one episode's agent, yield it, then take it down again.

    ``agent_cls`` is always a plain class -- that is all the registry holds -- and it
    is the class's own
    :attr:`~rynn_scale.agents.base.BaseAgent.dedicated_process` that decides which of
    two formats it is hosted in, and therefore what comes back:

      * **Actor agent** (``dedicated_process = True``, e.g.
        :class:`~rynn_scale.agents.robot.RobotAgent`) -- wrapped in ``ray.remote``
        here, with the class's own
        :attr:`~rynn_scale.agents.base.BaseAgent.ray_actor_options`. The agent *is* a
        process of its own, env included, and is yielded as the **actor handle**, so
        the caller awaits ``rollout.remote(...)``. It is created for one episode and
        killed with it, which is what a simulator needs: a leaked EGL context, an
        env poisoned by a sim error, or a wedged MuJoCo step can never outlive the
        episode that caused it, because taking the agent down is a plain
        ``ray.kill`` -- the only recovery there is from a blocking sim step.
      * **Driver agent** (the default) -- constructed here, in the caller's process,
        and yielded as the **instance**, whose ``rollout`` the caller awaits
        directly. It gets no process, no CPU and no teardown, so it must own nothing
        that blocks: whatever it does runs on the event loop every other episode
        shares, where a wedged step cannot be killed. A simulator belongs in an
        actor agent.

    Telling the two apart at the call site is a plain
    ``isinstance(agent, ray.actor.ActorHandle)`` -- see ``_rollout`` below.

    ``args``/``kwargs`` are the agent's own constructor arguments, except ``cpus``
    and ``pending_warn_s``, which are this function's -- so an agent cannot take
    ctor kwargs by those two names.

    An actor agent that cannot be placed yet is **waited for, not written off**:
    this call does not return until the cluster has its CPU. See the readiness
    probe below for why that wait is the real admission control.
    """
    import ray

    _assert_async_rollout(agent_cls)
    if not agent_cls.dedicated_process:
        yield agent_cls(*args, **kwargs)
        return

    # The wrapping happens here, per episode, rather than once at import: the class
    # is what the registry holds and what deploy constructs, so nothing outside this
    # function needs Ray to have seen it. ``ray.remote`` refuses an empty call, hence
    # the two spellings.
    options = agent_cls.ray_actor_options
    actor_cls = ray.remote(**options)(agent_cls) if options else ray.remote(agent_cls)

    handle = actor_cls.options(
        # The episode's terms, not the class's -- which is why the class declares
        # only what it needs to be hostable at all (its concurrency groups):
        # ``num_cpus`` is what makes Ray's accounting agree with the episode
        # semaphore, ``max_concurrency=1`` keeps the actor to one episode, and
        # ``max_restarts=0`` stops a crashed agent from coming back with a stale
        # env -- the caller gets ``RayActorError`` and the run stops instead. Note
        # the env's GL thread affinity does *not* come from ``max_concurrency``
        # (that only sizes the default group, where ``__ray_ready__`` below lands)
        # but from the single-threaded ``control`` group ``RobotAgent`` declares.
        num_cpus=cpus,
        max_concurrency=1,
        max_restarts=0,
    ).remote(*args, **kwargs)
    try:
        # ``.remote()`` hands back a handle immediately and, with no CPU free, the
        # actor sits in ``PENDING_CREATION`` until one frees up. Waiting that out is
        # deliberate: it is the only admission control that reads *real* resources.
        # The evaluator's semaphore cannot -- it is a count, sized once from a CPU
        # reading that goes stale the moment anything else on the cluster moves, and
        # its permit is returned before the killed actor's process has actually let
        # go of the CPU. So an episode queues here for as long as it takes rather
        # than being written off with a fabricated score.
        #
        # ``__ray_ready__`` (Ray injects it into every actor class) is what resolves
        # once the actor is running. It *raises* rather than waits if the actor died
        # coming up, so a broken ``__init__`` still surfaces immediately instead of
        # queueing forever. And it reports scheduling and nothing else only because
        # ``RobotAgent`` builds its env inside the loop rather than in ``__init__``;
        # keep it that way, or a slow env build becomes an invisible wait.
        #
        # The one thing an unbounded wait must not be is silent -- that is
        # indistinguishable from a wedged eval -- so it says so on a period.
        ready = asyncio.ensure_future(await_ref(handle.__ray_ready__.remote()))
        try:
            waited = 0.0
            while True:
                done, _ = await asyncio.wait({ready}, timeout=pending_warn_s)
                if done:
                    break
                waited += pending_warn_s
                logger.warning(
                    "agent actor still unplaced %gs after creation (%g CPU "
                    "requested); the episode is queued until the cluster frees one. "
                    "If it never clears, nothing is going to release that CPU -- "
                    "check what else holds it.",
                    waited,
                    cpus,
                )
            await ready  # re-raise here if the actor died on the way up
        finally:
            # No-op once resolved; what it is for is the outer task being cancelled
            # mid-wait, which would otherwise strand this future with an exception
            # nobody retrieves.
            ready.cancel()
        yield handle
    finally:
        # Also the disposal path for an actor that never started: killing a
        # pending actor takes it out of the scheduling queue, so it cannot come up
        # later and take a CPU behind the run's back.
        try:
            ray.kill(handle)
        except Exception as e:  # noqa: BLE001 - already dead (e.g. a sim segfault)
            logger.debug("killing agent actor failed: %s", e)


def _assert_async_rollout(agent_cls: Any) -> None:
    """Both formats: ``rollout`` must be a coroutine function.

    For a driver agent it is what the caller awaits. For an actor agent it is
    subtler -- Ray decides whether an actor is asyncio-based or thread-based by
    scanning the class (whole MRO) for coroutine methods, and that single implicit
    switch changes the threading model the env's GL context depends on *and* flips
    which policy client API is legal (a Serve handle's sync path raises inside a
    running event loop).
    """
    rollout = getattr(agent_cls, "rollout", None)
    assert inspect.iscoroutinefunction(rollout), (
        f"{getattr(agent_cls, '__name__', agent_cls)}.rollout must be 'async def': "
        "the caller awaits it, and for an actor agent Ray also picks the actor's "
        "threading model from whether the class has any coroutine methods -- a "
        "thread-based actor would change both the env's GL thread affinity and "
        "which policy-client API is legal."
    )


class Evaluator(object):
    def __init__(
        self,
        args: EvaluationArguments,
        inference_wrapper: BaseInferenceWrapper,
        benchmarks: Union[BaseBenchmark, List[BaseBenchmark]],
    ):
        if not isinstance(benchmarks, (list, tuple)):
            benchmarks = [benchmarks]
        self.args = args
        self.benchmarks = benchmarks
        self.inference_wrapper = inference_wrapper

    async def _eval(self):
        run_id = datetime.now().strftime("%Y%m%d%H%M%S")

        if self.args.save_rollout:
            rollout_save_dir = os.path.join(self.args.save_dir, f"{run_id}_rollout")
            os.makedirs(rollout_save_dir, exist_ok=True)
        else:
            rollout_save_dir = None

        import ray
        from ray import serve

        # Importing the package is what populates ``AGENT_REGISTRY`` -- including
        # ``SingleTurnAgent``, the default for a benchmark that declares no config.
        from .. import agents  # noqa: F401  (registers the agent types)
        from ..registry import AGENT_REGISTRY
        from ..serving.client import RayServeClient
        from ..serving.server import InferenceServer
        from .episode_buffer import EpisodeBuffer
        from .placement import model_deployment_kwargs, plan_placement

        # All engines run through the same Serve orchestration. Plan: model
        # replicas fill every GPU (replica GPUs = tp*pp). Agent placement is not
        # planned here -- an env-owning agent is scheduled on whatever node has a
        # free CPU when its episode starts -- so pass the minimum and ignore
        # ``plan.agent_node_ids``.
        plan = plan_placement(
            tp_size=self.args.tensor_parallel_size,
            pp_size=self.args.pipeline_parallel_size,
            agents_per_node=1,
        )
        for warn in plan.warnings:
            logger.warning("[placement] %s", warn)

        server = InferenceServer.build(
            engine=self.args.engine,
            model_type=self.args.model_type,
            model_path=self.args.model_path,
            dtype=self.args.param_dtype,
            attn=self.args.attn_implementation,
            max_running_requests=self.args.max_running_requests,
            batch_wait_timeout_s=self.args.batch_wait_timeout_s,
            sampling_params=self.args.sampling_params,
            parallel_params=self.args.parallel_params,
            processing_params=self.args.processing_params,
            **model_deployment_kwargs(plan),
        )
        server_handle = serve.run(server, route_prefix=None)

        # One policy client for everything: VLM samples await it right here on the
        # driver's event loop, env agents get the handle passed into their actor.
        # Either way inference is awaited, so many samples overlap.
        client = RayServeClient(server_handle)

        capacity = self.args.max_concurrent_episodes
        episode_sem = asyncio.Semaphore(capacity)

        # Streaming episode buffer (Ray actor): each agent pushes its executed
        # frames here and the buffer owns rollout-video rendering. Created only
        # when saving rollouts.
        episode_buffer = None
        if rollout_save_dir is not None:
            # No frame rate here: a rollout video plays at the command rate of the
            # env that produced it, and each agent sends that when it opens the episode
            # (``create_episode``) -- one buffer serves episodes from any env, so
            # there is nothing for this end to assume. ``args.fps`` is *not* it: that
            # is the VLM frame-sampling knob (default 1), and writing a 20 Hz
            # trajectory at 1 fps is what made every rollout video 20x too slow.
            episode_buffer = EpisodeBuffer.remote(
                visualize_every=1,
                video_dir=rollout_save_dir,
            )

        benchmarks = {benchmark.__class__.__name__: benchmark for benchmark in self.benchmarks}

        pbar = tqdm(
            total=sum(len(benchmark) for benchmark in benchmarks.values()),
            desc="Rollout",
            position=0,
        )

        sem = asyncio.Semaphore(capacity * 2)
        results = defaultdict(list)

        async def _rollout(agent, *args, **kwargs):
            """Await one episode from either agent format.

            ``episode_agent`` yields an actor *handle* for an actor agent and the
            *instance* for a driver agent, so the call is either a remote one or a
            plain coroutine. This funnels both into one awaitable.
            """
            if isinstance(agent, ray.actor.ActorHandle):
                return await agent.rollout.remote(*args, **kwargs)
            return await agent.rollout(*args, **kwargs)

        def _agent_spec(agent_config, benchmark):
            """One benchmark sample's agent config -> ``(class, ctor kwargs)``.

            ``None`` -- what every env-less (VLM) benchmark returns -- is one
            generation per sample, so it defaults to ``SingleTurnAgent``, whose only
            knob is the benchmark's thinking flag. Otherwise the config *is* the
            constructor: ``type`` names the agent and everything else is passed
            through, so a key the agent does not accept fails loudly here rather
            than being dropped.
            """
            cfg = (
                dict(agent_config)
                if agent_config
                else {"type": "SingleTurnAgent", "enable_thinking": benchmark.enable_thinking}
            )
            return AGENT_REGISTRY[cfg.pop("type")], cfg

        async def _run_episode(agent_cls, episode_id, prompt, init_kwargs):
            """One sample in a fresh agent of its own (see ``episode_agent``).

            ``model``, ``buffer`` and ``episode_id`` are the evaluator's to supply --
            the client every agent generates through, the rollout-video sink, and this
            run's id -- and the benchmark's config is the rest of the constructor.
            """
            # Nothing is caught here, and that is the design: an episode is either
            # scored on what it actually did or it stops the run. A dead agent
            # process, an error out of ``rollout`` -- all of it propagates, because a
            # number fabricated for an episode that did not happen is worse than no
            # number, and it is worse in the way that is hardest to catch later: the
            # run still writes metrics. Two things that might look missing are not
            # failures at all -- an agent with no CPU to start in queues (see
            # ``episode_agent``), and a sample whose *content* the benchmark cannot
            # parse is that benchmark's business.
            #
            # There is no deadline: an episode is awaited for as long as it takes.
            # Leaving the block still reclaims the actor on every path above.
            async with episode_agent(
                agent_cls,
                client,
                episode_buffer,
                episode_id,
                **init_kwargs,
            ) as agent:
                return await _rollout(agent, prompt)

        async def _run_sample(prompt, agent_config, episode_id, benchmark):
            # Bound concurrent samples. For an actor agent this permit is also what
            # accounts for its CPU, so it is held until the actor is gone.
            async with episode_sem:
                agent_cls, init_kwargs = _agent_spec(agent_config, benchmark)
                return await _run_episode(agent_cls, episode_id, prompt, init_kwargs)

        async def _rollout_batch(data, benchmark_name, benchmark, episode_ids):
            try:
                # One sample per coroutine, dispatched concurrently.
                responses = await asyncio.gather(
                    *[
                        _run_sample(prompt, agent_config, episode_id, benchmark)
                        for prompt, agent_config, episode_id in zip(data["prompt"], data["agent_config"], episode_ids)
                    ]
                )

                for data_id, response in zip(data["data_id"], responses):
                    # A VLM sample's response is the generated text; an env (VLA)
                    # episode's is the whole result dict (success/steps/...), which
                    # is what those benchmarks score.
                    prediction = await benchmark.process_response(data_id, response)
                    score = await benchmark.get_matching_score(data_id, prediction)

                    result = {
                        "data_id": data_id,
                        "benchmark": benchmark_name,
                        "response": response,
                        "prediction": prediction,
                        "score": score,
                    }
                    results[benchmark_name].append(result)
            finally:
                # Release in ``finally``, and around the rollout as well as the
                # scoring: an episode that raises would otherwise strand the permit,
                # and the dispatch loop below would block on ``acquire`` forever
                # instead of reaching the error that is trying to stop the run.
                sem.release()
                pbar.update(1)

        # Episode ids: a plain counter over dispatch order, handed to the agent at
        # construction. Unique for the whole run, which is what lets the buffer key
        # an episode on the id alone -- no node, no benchmark name. Zero-padded so
        # the rollout videos list in the order they were dispatched.
        counter = itertools.count()

        # A ``TaskGroup`` rather than a task set and a closing ``gather``: the first
        # batch to raise cancels its siblings *and* the body of this ``async with``,
        # so a broken run stops here instead of dispatching every remaining batch
        # first and only reporting once they are all done. It is also what keeps a
        # dispatch loop parked on ``sem.acquire()`` from waiting forever on a permit
        # nobody is left to release. The error arrives as an ``ExceptionGroup``, so a
        # run that broke several ways at once reports all of them rather than
        # whichever landed first. Nothing is scored or saved on that path -- the
        # metrics below are only reached if every episode was.
        try:
            async with asyncio.TaskGroup() as tg:
                for name, benchmark in benchmarks.items():
                    for data in benchmark:
                        await sem.acquire()
                        episode_ids = [f"{next(counter):06d}" for _ in data["data_id"]]
                        tg.create_task(_rollout_batch(data, name, benchmark, episode_ids))
        finally:
            # Close the bar even on the abort path, or tqdm's leftover line lands in
            # the middle of the traceback.
            pbar.close()

        for name in benchmarks:
            benchmark = benchmarks[name]
            metrics = benchmark.compute_metrics(results[name])

            logger.info("%s Results on %s %s", "=" * 20, name, "=" * 20)
            logger.info(json.dumps(metrics, indent=4))

            save_path = os.path.join(self.args.save_dir, f"{run_id}_{name}.json")

            for result in results[name]:
                result["metadata"] = benchmark.data_dict[result["data_id"]]
                result["metadata"] = filter_metadata(dict(result["metadata"]))

            os.makedirs(self.args.save_dir, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(
                    {
                        "model_path": self.args.model_path,
                        "benchmark": name,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                        "prompt_format": self.args.prompt_format,
                        "enable_thinking": self.args.enable_thinking,
                        "sampling_params": self.args.sampling_params,
                        "processing_params": self.args.processing_params,
                        "metrics": metrics,
                        "metadata": results[name],
                    },
                    f,
                    indent=4,
                )

    def eval(self):
        asyncio.run(self._eval())
