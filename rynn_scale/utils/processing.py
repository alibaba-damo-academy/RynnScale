import io
import os
from typing import Any, Dict, List, Optional, Union

import ffmpeg
import numpy as np
from PIL import Image
from transformers.image_utils import load_image as _load_image
from transformers.video_utils import VideoMetadata

from . import oss


def read_video_ffmpeg(
    video: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    fps: Optional[float] = None,
    max_frames: Optional[float] = None,
    precise_time: bool = False,
    verbose: bool = False,
):
    if video.startswith("oss://"):
        video = oss.sign_url(video)

    probe = ffmpeg.probe(video)
    duration = float(probe["format"]["duration"])
    video_stream = next((stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)
    w, h = int(video_stream["width"]), int(video_stream["height"])
    video_fps = video_stream["avg_frame_rate"]
    if "/" in video_fps:
        numerator, denominator = map(int, video_fps.split("/"))
        if denominator == 0:
            video_fps = 0.0
        else:
            video_fps = numerator / denominator
    else:
        video_fps = float(video_fps)
    total_num_frames = round(video_fps * duration)

    kwargs, input_kwargs, output_kwargs = {}, {}, {}
    do_trim = start_time is not None or end_time is not None
    if start_time is not None:
        new_start_time = max(float(video_stream["start_time"]), start_time)
        duration -= new_start_time - start_time
        start_time = new_start_time
    else:
        start_time = float(video_stream["start_time"])
    if end_time is not None:
        duration = min(duration, end_time - start_time)
    else:
        duration = duration
    if do_trim:
        kwargs = {"ss": start_time, "t": duration}
    if precise_time:
        output_kwargs.update(kwargs)
    else:
        input_kwargs.update(kwargs)

    stream = ffmpeg.input(video, **input_kwargs)
    if fps is not None:
        stream = ffmpeg.filter(stream, "fps", fps=fps, round="near")
    stream = ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt="rgb24", **output_kwargs)
    out, _ = ffmpeg.run(stream, capture_stdout=True, quiet=not verbose)

    frames = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3]).transpose([0, 3, 1, 2]).copy()

    if fps is not None:
        timestamps = np.arange(start_time, start_time + duration + 1 / fps, 1 / fps)[: len(frames)]
        frames_indices = np.round(timestamps * video_fps)
    else:
        total_num_frames = len(frames)
        frames_indices = np.arange(total_num_frames)

    if max_frames is not None and len(frames) > max_frames:
        indices = np.round(np.linspace(0, len(frames) - 1, max_frames)).astype(np.int32)
        frames = frames[indices]
        frames_indices = frames_indices[indices]

    metadata = VideoMetadata(
        total_num_frames=total_num_frames,
        fps=video_fps,
        frames_indices=frames_indices,
    )
    return frames, metadata


def read_video_frames(
    video: Union[str, List[str]],
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    fps: Optional[float] = None,
    max_frames: Optional[float] = None,
    **kwargs,
):
    if isinstance(video, str):
        if video.startswith("oss://"):
            frames = sorted([os.path.join(video, x) for x in oss.listdir(video) if x.endswith((".jpg", ".jpeg", ".png"))])
        else:
            frames = sorted([os.path.join(video, x) for x in os.listdir(video) if x.endswith((".jpg", ".jpeg", ".png"))])
    else:
        frames = video

    total_num_frames = len(frames)
    # if "shareVideoGPTV" in video:
    #     video_fps = 2
    # else:
    #     raise ValueError(f"Unkown video data source: {video}")
    video_fps = 2
    timestamps = [i / video_fps for i in range(total_num_frames)]
    frames_indices = list(range(total_num_frames))

    if start_time is not None:
        assert start_time >= 0, f"start_time {start_time} must be non-negative"
        start_index = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - start_time))
    else:
        start_index = 0

    if end_time is not None:
        assert end_time >= 0, f"end_time {end_time} must be non-negative"
        end_index = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - end_time))
        frames = frames[: end_index + 1]
        timestamps = timestamps[: end_index + 1]
    else:
        end_index = total_num_frames - 1

    frames_indices = frames_indices[start_index : end_index + 1]

    if fps is not None:
        assert fps <= video_fps, f"Cannot sample {fps} from {video_fps}"
        sample_rate = int(video_fps / fps)
        frames_indices = frames_indices[::sample_rate]

    if max_frames is not None and len(frames_indices) > max_frames:
        frames_indices = [frames_indices[round(i)] for i in np.linspace(0, len(frames_indices) - 1, max_frames)]

    frames = [load_image(frames[i]).convert("RGB") for i in frames_indices]
    metadata = VideoMetadata(
        total_num_frames=total_num_frames,
        fps=video_fps,
        frames_indices=frames_indices,
    )
    return frames, metadata


def load_video(
    video: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    fps: Optional[float] = None,
    max_frames: Optional[float] = None,
    precise_time: bool = False,
    verbose: bool = False,
):
    if isinstance(video, (list, tuple)):
        is_frames = True
    elif video.startswith("oss://"):
        is_frames = oss.isdir(video)
    else:
        is_frames = os.path.isdir(video)

    if is_frames:
        return read_video_frames(
            video=video,
            start_time=start_time,
            end_time=end_time,
            fps=fps,
            max_frames=max_frames,
        )

    return read_video_ffmpeg(
        video=video,
        start_time=start_time,
        end_time=end_time,
        fps=fps,
        max_frames=max_frames,
        precise_time=precise_time,
        verbose=verbose,
    )


def load_image(image: str | Image.Image):
    if isinstance(image, str) and image.startswith("oss://"):
        with oss.get_object(image) as result:
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
