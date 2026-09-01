import json
import os
import queue
import subprocess
import threading
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers.video_utils import VideoMetadata

from ..constants import VIDEO_DECODE_BACKEND
from . import storage

_SUPPORTED_BACKENDS = ("ffmpeg", "pyav", "torchcodec")


def _run_ffmpeg(args: Sequence[str]) -> bytes:
    """Run ``ffmpeg`` with ``args`` and return raw stdout bytes."""
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", *args],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({' '.join(args)}): {proc.stderr.decode(errors='replace')}")
    return proc.stdout


class VideoReader:
    def __init__(self, video_path, backend: Optional[str] = None):
        self._backend = backend or VIDEO_DECODE_BACKEND
        if self._backend not in _SUPPORTED_BACKENDS:
            raise ValueError(f"Unknown video backend: {self._backend!r} (expected one of {_SUPPORTED_BACKENDS})")
        self._meta: Optional[Dict[str, float]] = None
        # An image-frame directory / explicit list is handled without a decoder.
        self._is_frames = isinstance(video_path, (list, tuple)) or (
            isinstance(video_path, str) and storage.isdir(video_path)
        )
        if self._is_frames:
            self._frame_source = video_path
            self._path = None
            return

        # Keep the original path. pyav/torchcodec read OSS through a seekable
        # file object (byte-range GETs, no URL expiry); ffmpeg's CLI can't take a
        # Python file object, so it uses a presigned URL instead.
        self._path = video_path
        self._fileobj = None
        if self._backend == "torchcodec":
            from torchcodec.decoders import VideoDecoder

            source = self._open_source()
            self._fileobj = None if isinstance(source, str) else source
            self._dec = VideoDecoder(source)
        elif self._backend == "pyav":
            self._dec = None
        else:  # ffmpeg
            self._signed = storage.sign_url(video_path)

    def _open_source(self):
        """Decoder source: the path for local files, a seekable file object for OSS."""
        if storage.is_oss(self._path):
            return storage.open_file(self._path, stream=True)
        return self._path

    @contextmanager
    def _pyav_container(self):
        """Open a pyav container over the source, closing it (and any OSS file
        object) on exit."""
        import av

        source = self._open_source()
        container = av.open(source)
        try:
            yield container
        finally:
            container.close()
            if not isinstance(source, str):
                source.close()

    @property
    def metadata(self) -> Dict[str, float]:
        """Container metadata (width/height/fps/duration/start_time/total_num_frames).

        Probed lazily and cached, so every backend exposes the same information
        through a single accessor.
        """
        if self._meta is None:
            self._meta = self._probe_ffmpeg() if self._backend == "ffmpeg" else self._probe_decoder()
        return self._meta

    def _probe_ffmpeg(self) -> Dict[str, float]:
        """Probe container metadata via ffprobe."""
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                self._signed,
            ],
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffprobe failed for {self._path!r}: {proc.stderr.decode(errors='replace')}")
        probe = json.loads(proc.stdout)
        video_stream = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
        if video_stream is None:
            raise ValueError(f"No video stream found in {self._path!r}")
        width, height = int(video_stream["width"]), int(video_stream["height"])

        raw_fps = video_stream["avg_frame_rate"]
        if "/" in raw_fps:
            num, den = map(int, raw_fps.split("/"))
            video_fps = 0.0 if den == 0 else num / den
        else:
            video_fps = float(raw_fps)

        # Prefer the video stream's own duration/frame count over the container-level
        # values: the format duration can overshoot (padding, audio tail), which
        # would otherwise request non-existent trailing frame indices.
        duration = float(video_stream.get("duration") or probe["format"]["duration"])
        nb_frames = video_stream.get("nb_frames")
        total_num_frames = int(nb_frames) if nb_frames else round(video_fps * duration)

        return {
            "width": width,
            "height": height,
            "fps": video_fps,
            "duration": duration,
            "start_time": float(video_stream.get("start_time", 0.0)),
            "total_num_frames": total_num_frames,
        }

    def _probe_decoder(self) -> Dict[str, float]:
        """Probe container metadata via the pyav/torchcodec decoders."""
        if self._backend == "torchcodec":
            meta = self._dec.metadata
            fps = float(getattr(meta, "average_fps", None) or 0.0)
            duration = float(getattr(meta, "duration_seconds", None) or 0.0)
            num_frames = getattr(meta, "num_frames", None)
            return {
                "width": int(meta.width),
                "height": int(meta.height),
                "fps": fps,
                "duration": duration,
                "start_time": float(getattr(meta, "begin_stream_seconds", None) or 0.0),
                "total_num_frames": int(num_frames if num_frames else round(fps * duration)),
            }

        import av

        with self._pyav_container() as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            if stream.duration is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = float(container.duration) / av.time_base
            else:
                duration = 0.0
            start_time = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0.0
            total = stream.frames or round(fps * duration)
            return {
                "width": int(stream.codec_context.width),
                "height": int(stream.codec_context.height),
                "fps": fps,
                "duration": duration,
                "start_time": start_time,
                "total_num_frames": int(total),
            }

    def read(self, indices: List[int]) -> torch.Tensor:
        """Read frames by indices. Returns (T, H, W, C) uint8 tensor.

        Random-access entry point used by the VLA datasets.
        """
        return self._sample([int(i) for i in indices])

    def _sample(self, indices: List[int]) -> torch.Tensor:
        """Decode frames by indices. Returns (T, H, W, C) uint8 tensor."""
        if self._backend == "torchcodec":
            frames = self._dec.get_frames_at(indices=indices).data
            return frames.permute(0, 2, 3, 1).contiguous()
        elif self._backend == "pyav":
            return self._sample_pyav(indices)
        else:
            return self._sample_ffmpeg(indices)

    def _sample_pyav(self, indices: List[int]) -> torch.Tensor:
        sorted_indices = sorted(set(indices))
        with self._pyav_container() as container:
            stream = container.streams.video[0]
            stream.thread_type = "FRAME"
            rate = float(stream.average_rate)
            time_base = float(stream.time_base)

            first = sorted_indices[0]
            last = sorted_indices[-1]
            pts = int(round(first / rate / time_base))
            container.seek(pts, stream=stream, any_frame=False, backward=True)

            wanted = set(sorted_indices)
            decoded: Dict[int, np.ndarray] = {}
            pos = -1
            for frame in container.decode(stream):
                if pos < 0:
                    pos = int(round(frame.pts * time_base * rate)) if frame.pts is not None else 0
                else:
                    pos += 1
                if pos < first:
                    continue
                if pos in wanted:
                    decoded[pos] = frame.to_ndarray(format="rgb24")
                    if len(decoded) == len(wanted):
                        break
                if pos > last:
                    break

            if len(decoded) != len(wanted):
                missing = sorted(wanted - decoded.keys())
                raise IndexError(f"Could not decode frames {missing} from {self._path}")
            arr = np.stack([decoded[i] for i in indices], axis=0)
            return torch.from_numpy(arr)

    def _sample_ffmpeg(self, indices: List[int]) -> torch.Tensor:
        sorted_indices = sorted(set(indices))
        # Frame-accurate selection: keep only the requested source frames and
        # drop pts pacing so exactly len(sorted_indices) frames are emitted. The
        # commas inside eq() are escaped for the filtergraph parser.
        select_expr = "+".join(f"eq(n\\,{i})" for i in sorted_indices)
        out = _run_ffmpeg(
            [
                "-i",
                self._signed,
                "-vf",
                "select=" + select_expr,
                "-vsync",
                "0",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:",
            ]
        )

        h, w = self.metadata["height"], self.metadata["width"]
        arr = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
        if arr.shape[0] != len(sorted_indices):
            raise IndexError(f"Requested {len(sorted_indices)} frames but decoded {arr.shape[0]} from {self._path}")
        lut = {idx: arr[k] for k, idx in enumerate(sorted_indices)}
        result = np.stack([lut[i] for i in indices], axis=0)
        return torch.from_numpy(result.copy())

    @property
    def num_frames(self) -> int:
        if self._backend == "torchcodec":
            return len(self._dec)
        return int(self.metadata["total_num_frames"])

    def sample_frames(
        self,
        indices: Optional[Sequence[int]] = None,
        fps: Optional[float] = None,
        max_frames: Optional[float] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Tuple[np.ndarray, VideoMetadata]:
        """Sample frames and return ``(T, C, H, W)`` uint8 plus :class:`VideoMetadata`.

        Supports three, mutually-exclusive sampling modes, all restricted to an
        optional ``[start_time, end_time]`` window:

        * **by index** — pass ``indices`` to read exactly those frames (``fps`` and
          ``max_frames`` must be unset).
        * **uniform by count** — pass only ``max_frames`` to keep that many evenly
          spaced (``linspace``) frames.
        * **by fps (+ cap)** — pass ``fps`` to sample at ``1/fps`` steps; with
          ``max_frames`` also set, fall back to uniform sampling when the fps count
          would exceed the cap, otherwise fps-sample and trim any overshoot.

        With none of the three given, every frame in the window is returned.

        For an image-frame source the frames are loaded from disk instead of
        decoded, and a list of PIL images is returned in place of the array.
        """
        if self._is_frames:
            return self._sample_image_frames(indices, fps, max_frames, start_time, end_time)

        info = self.metadata
        video_fps = info["fps"]
        full_duration = info["duration"]
        native_total = info["total_num_frames"]
        stream_start_time = info["start_time"]

        # --- by index: read exactly the requested frames ---
        if indices is not None:
            if fps is not None or max_frames is not None:
                raise ValueError("`indices` is mutually exclusive with `fps` / `max_frames`")
            frames_indices = [int(i) for i in indices]
            frames = self._sample(frames_indices).permute(0, 3, 1, 2).contiguous().numpy()
            return frames, VideoMetadata(
                total_num_frames=len(frames),
                fps=video_fps,
                frames_indices=frames_indices,
            )

        # --- time-window trimming -> [start_time, start_time + duration] ---
        duration = full_duration
        if start_time is not None:
            new_start_time = max(stream_start_time, start_time)
            duration -= new_start_time - start_time
            start_time = new_start_time
        else:
            start_time = stream_start_time
        if end_time is not None:
            duration = min(duration, end_time - start_time)

        start_idx = max(int(round(start_time * video_fps)), 0)
        end_idx = int(round((start_time + duration) * video_fps))
        if native_total > 0:
            end_idx = min(end_idx, native_total)
        window = list(range(start_idx, end_idx))

        # Full-clip frame count, computed before any [start, end] trimming.
        total_full = round(video_fps * full_duration)

        report_full_total = False

        if fps is None and max_frames is None:
            # every frame in the window
            frames_indices = window
        elif fps is None:
            # uniform sampling to max_frames
            frames_indices = self._uniform(window, max_frames)
        else:
            # fps sampling: round(ts * native_fps) -> frame indices. Kept
            # backend-agnostic (no ffmpeg fps filter) so all backends agree.
            timestamps = np.arange(start_time, start_time + duration + 1 / fps, 1 / fps)
            fi = np.round(timestamps * video_fps).astype(np.int64)
            if native_total > 0:
                fi = np.clip(fi, 0, native_total - 1)
            if max_frames is not None and len(fi) > max_frames:
                # fps would exceed the cap -> uniform sampling instead
                frames_indices = self._uniform(window, max_frames)
            else:
                report_full_total = True
                frames_indices = fi.tolist()

        frames = self._sample(frames_indices).permute(0, 3, 1, 2).contiguous().numpy()

        total_num_frames = total_full if report_full_total else len(frames)
        metadata = VideoMetadata(
            total_num_frames=total_num_frames,
            fps=video_fps,
            frames_indices=frames_indices,
        )
        return frames, metadata

    @staticmethod
    def _uniform(window: List[int], max_frames: float) -> List[int]:
        """Pick ``max_frames`` evenly spaced entries from ``window`` (or all)."""
        if len(window) <= max_frames:
            return window
        sel = np.round(np.linspace(0, len(window) - 1, int(max_frames))).astype(np.int64)
        return [window[i] for i in sel]

    def _sample_image_frames(
        self,
        indices: Optional[Sequence[int]] = None,
        fps: Optional[float] = None,
        max_frames: Optional[float] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Tuple[list, VideoMetadata]:
        """Sample from a directory / list of image frames. Returns a list of PIL images.

        Frames are assumed to be extracted at a fixed 2 fps; ``fps`` may only
        downsample. Sizes can vary per frame, so a list (not a stacked array) is
        returned.
        """
        from .processing import load_image  # local import to avoid an import cycle

        source = self._frame_source
        if isinstance(source, str):
            frames = sorted(
                os.path.join(source, x) for x in storage.listdir(source) if x.endswith((".jpg", ".jpeg", ".png"))
            )
        else:
            frames = list(source)
        total_num_frames = len(frames)
        video_fps = 2

        # by explicit index
        if indices is not None:
            if fps is not None or max_frames is not None:
                raise ValueError("`indices` is mutually exclusive with `fps` / `max_frames`")
            frames_indices = [int(i) for i in indices]
        else:
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
            else:
                end_index = total_num_frames - 1
            frames_indices = frames_indices[start_index : end_index + 1]

            if fps is not None:
                assert fps <= video_fps, f"Cannot sample {fps} from {video_fps}"
                sample_rate = int(video_fps / fps)
                frames_indices = frames_indices[::sample_rate]

            if max_frames is not None and len(frames_indices) > max_frames:
                frames_indices = [
                    frames_indices[round(i)] for i in np.linspace(0, len(frames_indices) - 1, int(max_frames))
                ]

        images = [load_image(frames[i]).convert("RGB") for i in frames_indices]
        metadata = VideoMetadata(
            total_num_frames=total_num_frames,
            fps=video_fps,
            frames_indices=frames_indices,
        )
        return images, metadata

    def iter_frames(self, indices: Sequence[int]) -> Iterator[torch.Tensor]:
        """Yield ``(H, W, C)`` uint8 frames for ``indices``, one at a time, in order.

        Optimized for non-decreasing ``indices``: the pyav backend opens the
        stream once and decodes forward (a backward jump triggers a seek),
        matching the episode-iteration access pattern. torchcodec and ffmpeg
        fetch each index independently. Not supported for image-frame sources.
        """
        if self._is_frames:
            raise ValueError("iter_frames is not supported for image-frame sources; use sample_frames")
        if self._backend == "torchcodec":
            for i in indices:
                yield self._dec.get_frames_at(indices=[int(i)]).data[0].permute(1, 2, 0).contiguous()
        elif self._backend == "pyav":
            yield from self._iter_pyav(indices)
        else:
            yield from self._iter_ffmpeg(indices)

    def _iter_pyav(self, indices: Sequence[int]) -> Iterator[torch.Tensor]:
        with self._pyav_container() as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            rate = float(stream.average_rate)
            time_base = float(stream.time_base)

            # Hold a one-frame lookahead: `cur` is the frame at position `cur_pos`.
            frame_iter = None
            cur = None
            cur_pos = -1

            def restart_from(index: int):
                nonlocal frame_iter, cur, cur_pos
                pts = int(round(index / rate / time_base))
                container.seek(pts, stream=stream, any_frame=False, backward=True)
                frame_iter = container.decode(stream)
                cur = next(frame_iter)
                cur_pos = int(round(cur.pts * time_base * rate)) if cur.pts is not None else 0

            restart_from(0)  # prime the stream at the first frame

            for raw in indices:
                index = int(raw)
                if index < cur_pos:  # backward jump -> seek to a keyframe behind it
                    restart_from(index)
                while cur_pos < index:
                    try:
                        cur = next(frame_iter)
                    except StopIteration:
                        raise IndexError(f"Frame {index} not reachable in {self._path} (stopped at pos {cur_pos})")
                    cur_pos += 1
                if cur_pos != index:
                    raise IndexError(f"Frame {index} not found in {self._path}")
                yield torch.from_numpy(cur.to_ndarray(format="rgb24"))

    def _iter_ffmpeg(self, indices: Sequence[int]) -> Iterator[torch.Tensor]:
        info = self.metadata
        w, h, video_fps = info["width"], info["height"], info["fps"]
        for raw in indices:
            index = int(raw)
            # Fast input seek by timestamp, then grab a single frame. Approximate
            # for VFR sources; prefer torchcodec/pyav when exactness matters.
            ss = index / video_fps if video_fps else 0.0
            out = _run_ffmpeg(
                [
                    "-ss",
                    repr(ss),
                    "-i",
                    self._signed,
                    "-frames:v",
                    "1",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "pipe:",
                ]
            )
            arr = np.frombuffer(out, np.uint8).reshape([h, w, 3])
            yield torch.from_numpy(arr.copy())


def load_video(
    video,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    fps: Optional[float] = None,
    max_frames: Optional[float] = None,
    backend: Optional[str] = None,
):
    """Load and sample frames from a video file or an image-frame directory/list.

    Thin wrapper over :meth:`VideoReader.sample_frames`. :class:`VideoReader`
    detects the source: a directory / list of image frames is loaded from disk
    (returning a list of PIL images); a video file is decoded with the selected
    backend (``ffmpeg`` / ``pyav`` / ``torchcodec``, defaulting to
    :data:`VIDEO_DECODE_BACKEND`) and returns ``(T, C, H, W)`` uint8 frames.
    Both return a :class:`VideoMetadata`.
    """
    return VideoReader(video, backend=backend).sample_frames(
        start_time=start_time,
        end_time=end_time,
        fps=fps,
        max_frames=max_frames,
    )


class SequentialVideoReader:
    """Sequential frame reader optimized for _iter_episode access pattern.

    Opens a video once and reads frames in non-decreasing index order.
    Supports torchcodec and pyav backends.
    """

    def __init__(self, video_path: str, backend: Optional[str] = None):
        self._path = video_path
        self._backend = backend or VIDEO_DECODE_BACKEND

        if self._backend == "torchcodec":
            from torchcodec.decoders import VideoDecoder

            self._dec = VideoDecoder(video_path)
        elif self._backend == "pyav":
            import av

            self._container = av.open(video_path)
            self._stream = self._container.streams.video[0]
            self._stream.thread_type = "AUTO"
            self._pos = 0
            self._iter = self._container.decode(self._stream)
            self._prefetched = None
        else:
            raise ValueError(f"Unknown video backend: {self._backend!r}")

    def read(self, index: int) -> torch.Tensor:
        """Read a single frame by index. Returns (H, W, C) uint8 tensor."""
        if self._backend == "torchcodec":
            frame = self._dec.get_frames_at(indices=[index]).data
            return frame[0].permute(1, 2, 0).contiguous()
        else:
            return self._read_pyav(index)

    def read_batch(self, indices: List[int]) -> torch.Tensor:
        """Read multiple frames. Returns (T, H, W, C) uint8 tensor."""
        if self._backend == "torchcodec":
            frames = self._dec.get_frames_at(indices=indices).data
            return frames.permute(0, 2, 3, 1).contiguous()
        else:
            return torch.stack([self._read_pyav(i) for i in indices])

    def _seek_pyav(self, index: int):
        rate = float(self._stream.average_rate)
        time_base = float(self._stream.time_base)
        pts = int(round(index / rate / time_base))
        self._container.seek(pts, stream=self._stream, any_frame=False, backward=True)
        self._iter = self._container.decode(self._stream)
        frame = next(self._iter)
        self._pos = int(round(frame.pts * time_base * rate)) if frame.pts is not None else 0
        if self._pos == index:
            self._prefetched = frame
        else:
            self._prefetched = None

    def _read_pyav(self, index: int) -> torch.Tensor:
        if index < self._pos:
            self._seek_pyav(index)
            if self._prefetched is not None:
                arr = self._prefetched.to_ndarray(format="rgb24")
                self._prefetched = None
                self._pos += 1
                return torch.from_numpy(arr)
        while self._pos <= index:
            try:
                frame = next(self._iter)
            except StopIteration:
                raise IndexError(f"Frame {index} not reachable in {self._path} (stopped at pos {self._pos})")
            if self._pos == index:
                arr = frame.to_ndarray(format="rgb24")
                self._pos += 1
                return torch.from_numpy(arr)
            self._pos += 1
        raise IndexError(f"Frame {index} not found in {self._path}")

    def close(self):
        if self._backend == "torchcodec":
            self._dec = None
        else:
            self._container.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def decode_video_frames_pyav_by_timestamps(
    video_path: str,
    timestamps: List[float],
    tolerance_s: float = 0.05,
) -> torch.Tensor:
    """Decode frames by timestamp using PyAV.

    Seeks to the nearest keyframe before the first timestamp and decodes
    forward, then selects the closest frame to each query timestamp.

    Returns (T, H, W, C) uint8 tensor.
    """
    import av

    first_ts = min(timestamps)
    last_ts = max(timestamps)

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "FRAME"

        target_pts = int(first_ts / float(stream.time_base))
        container.seek(target_pts, stream=stream, backward=True)

        loaded_frames: List[torch.Tensor] = []
        loaded_ts: List[float] = []

        for frame in container.decode(stream):
            current_ts = float(frame.pts * stream.time_base)
            arr = frame.to_ndarray(format="rgb24")
            loaded_frames.append(torch.from_numpy(arr))
            loaded_ts.append(current_ts)
            if current_ts >= last_ts + tolerance_s:
                break

        if not loaded_frames:
            raise RuntimeError(f"No frames decoded from {video_path} for timestamps {timestamps}")

        loaded_ts_t = torch.tensor(loaded_ts, dtype=torch.float64)
        query_ts_t = torch.tensor(timestamps, dtype=torch.float64)
        dists = torch.abs(query_ts_t.unsqueeze(1) - loaded_ts_t.unsqueeze(0))
        closest = dists.argmin(dim=1)

        result = []
        for i, qi in enumerate(closest):
            dist = dists[i, qi].item()
            if dist > tolerance_s:
                raise ValueError(
                    f"Frame at ts={timestamps[i]:.4f}s: closest decoded "
                    f"frame is {dist:.4f}s away (tolerance={tolerance_s}s) "
                    f"in {video_path}"
                )
            result.append(loaded_frames[qi])

        return torch.stack(result)
    finally:
        container.close()


class VideoWriter:
    """Encode RGB frames to an mp4 through a piped ``ffmpeg`` subprocess.

    The ffmpeg process is opened lazily on the first :meth:`write` so the frame
    size (``W x H``) is learned from the data -- callers just push uint8
    ``(H, W, 3)`` frames and :meth:`close`. ``close`` is idempotent and safe to
    call from teardown; a broken pipe (ffmpeg died) is swallowed rather than
    propagated so recording never breaks the caller.

    Usage::

        with VideoWriter(path, fps=20) as w:
            for frame in frames:
                w.write(frame)
    """

    def __init__(
        self,
        path: str,
        fps: int = 20,
        *,
        codec: str = "libx264",
        pix_fmt: str = "yuv420p",
        input_pix_fmt: str = "rgb24",
    ):
        self._path = path
        self._fps = fps
        self._codec = codec
        self._pix_fmt = pix_fmt
        self._input_pix_fmt = input_pix_fmt
        self._proc: Optional[subprocess.Popen] = None

    def _open(self, width: int, height: int) -> None:
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{width}x{height}",
                "-pix_fmt",
                self._input_pix_fmt,
                "-r",
                str(self._fps),
                "-i",
                "-",
                "-c:v",
                self._codec,
                "-pix_fmt",
                self._pix_fmt,
                "-loglevel",
                "error",
                self._path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, frame: np.ndarray) -> None:
        """Write one ``(H, W, 3)`` uint8 frame. Opens ffmpeg on first call."""
        if frame is None:
            return
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if self._proc is None:
            height, width = frame.shape[:2]
            self._open(width, height)
        try:
            self._proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, ValueError):
            pass  # ffmpeg died / stdin closed -- don't let recording break the caller

    def close(self) -> None:
        """Flush and close the writer. Idempotent."""
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001 - never let teardown raise
            proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class AsyncVideoWriter(VideoWriter):
    """:class:`VideoWriter` that encodes on a background thread.

    :meth:`write` copies the frame into a bounded queue and returns immediately;
    a worker thread drains the queue and does the (potentially blocking) pipe
    write, so the caller (e.g. a rollout step) never stalls on ffmpeg
    backpressure. The queue is bounded, so a persistently slow encoder applies
    backpressure rather than growing memory without bound. :meth:`close` drains
    the queue, then finalizes the underlying ffmpeg process.
    """

    def __init__(self, path: str, fps: int = 20, *, max_queue: int = 256, **kwargs):
        super().__init__(path, fps, **kwargs)
        self._queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=max_queue)
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="AsyncVideoWriter", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            frame = self._queue.get()
            if frame is None:  # sentinel
                break
            VideoWriter.write(self, frame)

    def write(self, frame: np.ndarray) -> None:
        if frame is None or self._closed:
            return
        # Copy: the caller may reuse/mutate the buffer (sim obs / shm view)
        # before the worker thread gets to encode it.
        self._queue.put(np.array(frame, dtype=np.uint8))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)  # sentinel: drain remaining frames, then stop
        self._thread.join(timeout=60)
        super().close()
