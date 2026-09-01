import asyncio
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import ray

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """One finished trajectory: its id and its (lightweight) steps.

    Each step is ``{"state": <RobotState dict>, "action": <flat/action>}`` -- the
    bulky images are streamed straight to the video and dropped, so an episode
    kept in the queue holds only its metadata.
    """

    episode_id: str
    steps: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"episode_id": self.episode_id, "steps": self.steps}


def observation_to_frame(step: Any) -> Optional[np.ndarray]:
    """Render one RGB frame from a recorded step for visualization.

    Concatenates ``step["images"]`` (name-sorted) horizontally, padding shorter
    views to the tallest -- the same layout the sim env uses for rollout videos.
    Returns ``None`` when the step carries no images.
    """
    images = step.get("images") if isinstance(step, dict) else None
    if not images:
        return None
    views = [np.asarray(images[k]) for k in sorted(images)]
    height = max(v.shape[0] for v in views)
    views = [
        np.pad(v, ((0, height - v.shape[0]), (0, 0), (0, 0)), mode="constant") if v.shape[0] < height else v
        for v in views
    ]
    return np.concatenate(views, axis=1)


@ray.remote(num_cpus=0, max_concurrency=16)
class EpisodeBuffer:
    """Bounded, streaming-visualization episode buffer.

    One flat queue keyed by episode id, because the evaluator hands out those ids
    itself, in dispatch order, and they are unique for the whole run -- which is
    what makes the producing node irrelevant here.

    Args:
        max_episodes: max finished episodes kept; the oldest is dropped when the
            queue is full.
        visualize_every: record every ``visualize_every``-th episode to a video
            (``0`` / ``None`` disables visualization).
        video_dir: directory the visualization mp4s are written to.
        fps: fallback frame rate, for a ``create_episode`` that names none.
    """

    def __init__(
        self,
        max_episodes: int = 100,
        visualize_every: Optional[int] = None,
        video_dir: Optional[str] = None,
        fps: int = 20,
    ):
        assert max_episodes >= 1, "max_episodes must be >= 1"

        # Finished episodes, oldest first.
        self._queue: Deque[Episode] = deque(maxlen=max_episodes)
        # Episodes still being written, keyed by episode id.
        self._in_progress: Dict[str, Episode] = {}

        # Wakes blocked ``get`` callers when a new episode is enqueued.
        self._cond = asyncio.Condition()

        # ---- streaming visualization ----
        self._visualize_every = int(visualize_every) if visualize_every else 0
        self._video_dir = video_dir
        self._fps = int(fps)
        self._started_count = 0
        self._finished_count = 0
        # One open writer per *recorded* in-progress episode; an episode with no
        # entry here is one that ``create_episode`` decided not to record.
        self._writers: Dict[str, Any] = {}
        if self._visualize_every and self._video_dir:
            os.makedirs(self._video_dir, exist_ok=True)

    # --------------------------------------------------------- create_episode

    def create_episode(self, episode_id: str, fps: Optional[int] = None) -> None:
        """Open an episode. Must be called before its first :meth:`put`.

        Opening is its own call because both things it settles are known only at the
        start and only to the producer: whether this episode is one of the recorded
        ones (``visualize_every``), and ``fps`` -- the command rate of the env about
        to produce the frames, which sizes the video writer. ``None`` falls back to
        the ctor's ``fps``.

        Inferring all that from a first ``put`` instead is what this replaces: a put
        for an unknown id would have been indistinguishable from a new episode, so a
        lost or misrouted chunk silently started one.

        Re-opening an id **overwrites**: a producer that reuses one is recording over
        the same trajectory on purpose (an agent that was handed an episode id records
        every run under it, so its video sits at a path the caller can predict), so
        whatever was open under that id is discarded rather than merged into the new
        recording or refused. It is still worth a line in the log, because the
        discarded frames were paid for.
        """
        stale = self._in_progress.pop(episode_id, None)
        if stale is not None:
            writer = self._writers.pop(episode_id, None)
            if writer is not None:
                writer.close()
            logger.warning(
                "episode %s was still open with %d frames; discarding it and recording over the same id.",
                episode_id,
                len(stale.steps),
            )
        self._in_progress[episode_id] = Episode(episode_id=episode_id)
        self._started_count += 1
        record = bool(self._visualize_every and self._video_dir and self._started_count % self._visualize_every == 0)
        if record:
            path = os.path.join(self._video_dir, f"ep{episode_id}.mp4")
            self._writers[episode_id] = self._new_writer(path, int(fps) if fps else self._fps)

    # ------------------------------------------------------------------ put

    async def put(
        self,
        episode_id: str,
        images: List[Dict[str, Any]],
        state: Optional[Dict[str, Any]] = None,
        action: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one executed frame to an open episode.

        Called once per action by :meth:`~rynn_scale.agents.robot.RobotAgent`'s
        ``_record``, which hands over a single frame -- ``images`` is a list because
        writing them happens on a worker thread and batching stays available, not
        because the caller has more than one. ``state`` is that frame's
        ``RobotState.to_dict()`` and ``action`` the ``RobotAction.to_dict()`` the agent
        played from it, kept as the ``(state, action)`` pair the frame produced. Frames
        are rendered and streamed into the episode's video writer, then dropped -- only
        the lightweight ``state`` / ``action`` are retained, one entry per action.

        Appending only: closing is :meth:`finalize`'s job. The producer knows a run has
        ended, but not always *on* a frame -- a STOP command or a spent step budget
        ends one between two actions -- so a "this was the last one" flag here would be
        a flag the caller often had no frame left to hang it on, and would then fake
        with an empty put.

        The id must name an episode :meth:`create_episode` opened and :meth:`finalize`
        has not closed; anything else raises rather than being absorbed.
        """
        episode = self._in_progress.get(episode_id)
        assert episode is not None, (
            f"no open episode {episode_id!r}: call create_episode first (or this episode was already finalized)."
        )

        # Stream this chunk's frames to the video, then release them: encode off
        # the event loop so ffmpeg backpressure never stalls the actor.
        writer = self._writers.get(episode_id)
        if writer is not None and images:
            await asyncio.to_thread(self._write_frames, writer, images)

        # Retain only the lightweight (state, action) pair for consumers; images gone.
        if state is not None or action is not None:
            episode.steps.append({"state": state, "action": action})

    # ------------------------------------------------------------- finalize

    async def finalize(self, episode_id: str) -> None:
        """Close an open episode: finish its video and queue it for consumers.

        The counterpart of :meth:`create_episode`, and deliberately not a flag on the
        last :meth:`put`: a run ends when its producer says so, which need not coincide
        with a frame (see :meth:`put`).

        **A no-op for an id that is not open**, which is what lets the producer call it
        from every path a run can end on -- a STOP, the world reporting done, the step
        budget, a fresh run over the same id, a ``finally`` on the way out -- without
        any of them tracking whether another already did. An id that was never opened
        is the same case: nothing was recorded under it, so there is nothing to close.
        """
        episode = self._in_progress.pop(episode_id, None)
        if episode is None:
            return

        # Close the video (drains the writer thread) off the loop, then move the
        # episode into the bounded queue and wake anyone blocked in ``get``.
        writer = self._writers.pop(episode_id, None)
        if writer is not None:
            await asyncio.to_thread(writer.close)

        self._queue.append(episode)
        self._finished_count += 1
        async with self._cond:
            self._cond.notify_all()

    # ------------------------------------------------------------------ get

    async def get(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Pop and return the oldest finished trajectory (as a dict), blocking if
        none is queued.

        Args:
            timeout: max seconds to block; ``None`` waits indefinitely. Returns
                ``None`` if the timeout elapses with nothing available.

        Returns the trajectory ``{"episode_id", "steps"}`` or ``None`` on timeout.
        """
        loop = asyncio.get_event_loop()
        deadline = None if timeout is None else loop.time() + timeout

        async with self._cond:
            while True:
                if self._queue:
                    return self._queue.popleft().to_dict()

                if deadline is None:
                    await self._cond.wait()
                else:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        return None
                    try:
                        await asyncio.wait_for(self._cond.wait(), remaining)
                    except asyncio.TimeoutError:
                        return None

    # ------------------------------------------------------------------ stats

    def stats(self) -> Dict[str, Any]:
        """Queue depth plus in-flight / started / finished counters."""
        return {
            "queued": len(self._queue),
            "in_progress": len(self._in_progress),
            "started": self._started_count,
            "finished": self._finished_count,
        }

    # ------------------------------------------------------------- visualize

    def _new_writer(self, path: str, fps: int) -> Any:
        """Open a streaming video writer (ffmpeg opens lazily on first frame)."""
        from ..utils.video import AsyncVideoWriter

        return AsyncVideoWriter(path, fps=fps)

    def _write_frames(self, writer: Any, images: List[Dict[str, Any]]) -> None:
        """Render each frame's ``{cam: HxWx3}`` dict and hand it to the writer
        (worker thread).

        ``AsyncVideoWriter.write`` copies the frame into its bounded queue, so the
        images can be dropped as soon as this returns."""
        for frame_images in images:
            frame = observation_to_frame({"images": frame_images})
            if frame is not None:
                writer.write(frame)
