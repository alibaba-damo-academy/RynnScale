"""Policy input contracts for the serving layer.

``VLAInferenceRequest`` / ``VLMInferenceRequest`` are what the policy (``InferenceClient`` ->
``InferenceServer``) consumes. They live here -- not in the environment layer -- because they
are a *serving* concern: the env only produces an opaque observation dict, and
the caller (the agent loop, an HTTP ingress) assembles the request from that
observation plus the task text / horizon it owns. This module
is a dependency-free leaf (numpy + stdlib only) so both ``environments`` and
``serving.client`` can import it without a cycle.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class VLAInferenceRequest:
    """Everything the policy needs for one inference.

    Carries an opaque observation (``state`` + ``images`` + ``robot_type``),
    the task text, and the RTC fields. The policy does not interpret the
    internal layout of these fields.
    """

    text: str
    state: Dict[str, Any]
    images: Dict[str, np.ndarray]
    robot_type: str
    # Real-Time Chunking: the not-yet-executed tail of the previous chunk and
    # the number of command steps between image capture and the first
    # to-be-executed action. Both default to the zero-regression values, which is
    # what a logical-clock sim (nothing in flight, no latency to hide) always sends.
    prev_actions: Optional[np.ndarray] = None
    delay_steps: int = 0
    num_steps: int = 10

    @classmethod
    def from_observation(
        cls,
        obs: Dict[str, Any],
        text: str = "",
        num_steps: int = 10,
        *,
        prev_actions: Optional[np.ndarray] = None,
        delay_steps: int = 0,
    ) -> "VLAInferenceRequest":
        """Assemble a request from an env observation dict plus the caller's own
        ``text`` / ``num_steps`` / RTC fields.

        The observation is one frame -- ``state`` / ``images`` / ``robot_type`` -- and
        that is the whole of what the env knows, since a ``step`` plays a single
        action. Everything else is the caller's: the task text, the action horizon,
        and the RTC pair, which describes *its* chunking policy (what it is still
        holding un-executed, and how much of that it expects to play before this
        chunk can land) rather than anything the env reported.
        """
        return cls(
            text=text,
            state=obs["state"],
            images=obs.get("images") or {},
            robot_type=obs["robot_type"],
            prev_actions=prev_actions,
            delay_steps=delay_steps,
            num_steps=num_steps,
        )


@dataclass
class VLMInferenceRequest:
    """One VLM/LLM generation request (hf/sglang engines).

    Carries a chat ``conversation`` (list of ``{"role", "content": [...]}``
    messages whose content items may be ``text``/``image``/``video``); the
    Processor applies the chat template and preprocesses any media. Sampling is
    fixed per Serve deployment, so it is not carried here.
    """

    conversation: List[Dict[str, Any]]
    enable_thinking: bool = False

    @classmethod
    def from_prompt(cls, prompt: Any, *, enable_thinking: bool = False) -> "VLMInferenceRequest":
        """Build a request from whatever the benchmark handed out for one sample.

        A conversation passes through; a bare string becomes a single user turn.
        The un-flattened form is what matters: the Processor locates the
        request's image/video items inside the message content, so collapsing the
        conversation to a string here would drop the media.
        """
        conversation = (
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}] if isinstance(prompt, str) else prompt
        )
        return cls(conversation=conversation, enable_thinking=enable_thinking)
