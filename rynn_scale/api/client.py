"""Client for the RynnScale inference server (``rynn_scale.api.serve``).

Two engines, two request shapes -- both POST to ``/generate`` (see the server's
module docstring):

  * ``vla``  -- ``{text, state, images, robot_type, prev_actions?,
    delay_steps?, num_steps?}`` -> action chunk (``RobotAction.to_dict``).
    ``images`` maps each camera name to a base64 JPEG; ``state`` is a
    ``RobotState.to_dict`` mapping.
  * ``vlm``  -- ``{conversation, enable_thinking?}`` -> ``{"text": ...}``. The
    conversation is an OpenAI-style chat list; image/video content items are
    URLs/paths the server loads itself (no base64 needed).

Usage
-----
VLM (text, optionally with images)::

    python -m rynn_scale.api.client vlm \\
        --url http://localhost:8000 \\
        --prompt "Describe the scene." \\
        --image /path/to/frame.jpg

VLA (send a full request from a JSON file, attaching camera images)::

    python -m rynn_scale.api.client vla \\
        --url http://localhost:8000 \\
        --payload request.json \\
        --image base_camera=/path/cam0.jpg --image wrist=/path/cam1.jpg

where ``request.json`` holds at least ``{"text": ..., "state": {...},
"robot_type": "franka"}`` (``state`` in ``RobotState.to_dict`` layout). Any
``--image cam=path`` pairs are base64-encoded and merged into ``images``.

Programmatic use: import :func:`generate_vla` / :func:`generate_vlm`.
"""

import argparse
import base64
import json
from typing import Any, Dict, List, Optional

import requests


def _encode_image(path: str) -> str:
    """Read an image file and return its base64 string (as the server expects)."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _post(url: str, payload: Dict[str, Any], timeout: float) -> Any:
    endpoint = url.rstrip("/") + "/generate"
    resp = requests.post(endpoint, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def generate_vla(
    url: str,
    text: str,
    state: Dict[str, Any],
    images: Dict[str, str],
    robot_type: str,
    prev_actions: Optional[List] = None,
    delay_steps: int = 0,
    num_steps: int = 10,
    timeout: float = 120.0,
) -> Any:
    """Call the ``vla`` engine. ``images`` maps camera name -> image file path."""
    payload = {
        "text": text,
        "state": state,
        "images": {name: _encode_image(p) for name, p in images.items()},
        "robot_type": robot_type,
        "delay_steps": delay_steps,
        "num_steps": num_steps,
    }
    if prev_actions is not None:
        payload["prev_actions"] = prev_actions
    return _post(url, payload, timeout)


def generate_vlm(
    url: str,
    prompt: str,
    images: Optional[List[str]] = None,
    videos: Optional[List[str]] = None,
    enable_thinking: bool = False,
    timeout: float = 120.0,
) -> Any:
    """Call the ``hf``/``sglang`` engine with a single user turn.

    ``images``/``videos`` are URLs or server-visible paths (the server loads
    them); they are added as ``image``/``video`` content items before the text.
    """
    content: List[Dict[str, Any]] = []
    for img in images or []:
        content.append({"type": "image", "image": img})
    for vid in videos or []:
        content.append({"type": "video", "video": vid})
    content.append({"type": "text", "text": prompt})
    payload = {
        "conversation": [{"role": "user", "content": content}],
        "enable_thinking": enable_thinking,
    }
    return _post(url, payload, timeout)


def _parse_image_pairs(items: List[str]) -> Dict[str, str]:
    """Parse ``--image cam=path`` (vla) into a {cam: path} mapping."""
    out: Dict[str, str] = {}
    for item in items:
        assert "=" in item, f"--image for vla must be 'cam=path', got {item!r}"
        cam, path = item.split("=", 1)
        out[cam] = path
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("engine", choices=["vla", "vlm"], help="Which request schema to send.")
    parser.add_argument("--url", default="http://localhost:8000", help="Server base URL.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="vla: 'cam=path' (repeatable, base64-encoded). vlm: image URL/path (repeatable).",
    )

    # VLM knobs
    parser.add_argument("--prompt", default="Describe what you see.", help="vlm: user text.")
    parser.add_argument("--video", action="append", default=[], help="vlm: video URL/path (repeatable).")
    parser.add_argument("--enable-thinking", action="store_true", help="vlm: enable thinking.")

    # VLA knobs
    parser.add_argument("--payload", help="vla: JSON file with {text, state, robot_type, ...}.")
    parser.add_argument("--text", help="vla: task text (overrides payload['text']).")
    parser.add_argument("--robot-type", help="vla: robot type (overrides payload['robot_type']).")
    parser.add_argument("--num-steps", type=int, default=10, help="vla: decode steps.")

    args = parser.parse_args()

    if args.engine == "vlm":
        result = generate_vlm(
            args.url,
            args.prompt,
            images=args.image,
            videos=args.video,
            enable_thinking=args.enable_thinking,
            timeout=args.timeout,
        )
    else:
        assert args.payload, "vla requires --payload pointing at a request JSON file."
        with open(args.payload) as f:
            body = json.load(f)
        text = args.text or body.get("text")
        robot_type = args.robot_type or body.get("robot_type")
        assert text and robot_type, "vla request needs 'text' and 'robot_type'."
        result = generate_vla(
            args.url,
            text=text,
            state=body["state"],
            images=_parse_image_pairs(args.image),
            robot_type=robot_type,
            prev_actions=body.get("prev_actions"),
            delay_steps=body.get("delay_steps", 0),
            num_steps=args.num_steps if args.num_steps is not None else body.get("num_steps", 10),
            timeout=args.timeout,
        )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
