from typing import Any, Optional

from ..registry import AGENT_REGISTRY
from ..serving.requests import VLMInferenceRequest
from .base import BaseAgent


@AGENT_REGISTRY.register("SingleTurnAgent")
class SingleTurnAgent(BaseAgent):
    """One generation per sample: the VLM path, and the evaluator's default.

    There is no env and no episode here -- a query is one call and the generated
    text *is* the result the benchmark scores -- so this leaves
    :attr:`~rynn_scale.agents.base.BaseAgent.dedicated_process` at its default and is
    deliberately **not** a Ray actor: it runs on the evaluator's own event loop, where
    an actor would buy nothing and cost a scheduling round-trip per sample. Many of
    these overlap while their inference is in flight, which is the only concurrency
    this path needs.
    """

    def __init__(
        self, model: Any, buffer: Any = None, episode_id: Optional[str] = None, *, enable_thinking: bool = False
    ):
        super().__init__(model, buffer, episode_id)
        self.enable_thinking = enable_thinking

    async def rollout(self, prompt: str = "") -> str:
        out = await self.model.generate_async(
            VLMInferenceRequest.from_prompt(prompt, enable_thinking=self.enable_thinking)
        )
        # The Model deployment replies ``{"text": ...}`` for the hf/sglang engines.
        return out.get("text", "") if isinstance(out, dict) else str(out)
