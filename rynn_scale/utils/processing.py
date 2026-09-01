import io
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from transformers.image_utils import load_image as _load_image

from . import storage
from .video import load_video  # noqa: F401  (re-exported for backward compatibility)


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode encoded image bytes (JPEG/PNG/...) into an ``HxWx3`` RGB uint8 array.

    Raises ``PIL.UnidentifiedImageError`` / ``OSError`` on malformed input rather
    than returning a sentinel.
    """
    with Image.open(io.BytesIO(data)) as image:
        # np.asarray would alias PIL's immutable buffer, and torch.from_numpy
        # warns on non-writable arrays.
        return np.array(image.convert("RGB"), dtype=np.uint8)


def load_image(image: str | Image.Image):
    if isinstance(image, str) and storage.is_oss(image):
        with storage.open_file(image) as result:
            buffer = io.BytesIO(result.read())
        image = Image.open(buffer)
        image.load()
        buffer.close()
        return image
    return _load_image(image)


def load_multimodal_data(
    conversation: List[Dict[str, Any]],
    fps: int = 1,
    max_frames: Optional[int] = None,
):
    images, videos, video_metadatas = [], [], []
    for message in conversation:
        for content in message["content"]:
            if content["type"] == "image":
                images.append(load_image(content["image"]))
            elif content["type"] == "video":
                video, video_metadata = load_video(content["video"], fps=fps, max_frames=max_frames)
                videos.append(video)
                video_metadatas.append(video_metadata)
            elif content["type"] == "text":
                pass
            else:
                raise ValueError(f"Unsupported content type: {content['type']}")
    return (
        images if len(images) > 0 else None,
        videos if len(videos) > 0 else None,
        video_metadatas if len(video_metadatas) > 0 else None,
    )
