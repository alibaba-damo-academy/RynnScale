import os
import queue
import threading
from dataclasses import fields
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import HfArgumentParser

from ..arguments import ReplayArguments
from ..datasets import build_dataset
from ..inference_wrappers import build_inference_wrapper
from ..projects import register_projects
from ..registry import RENDERER_REGISTRY
from ..renderers import build_renderer
from ..utils.logging import get_logger
from ..utils.robot import Arm, Position, RobotAction, RobotState
from ..utils.video import AsyncVideoWriter

logger = get_logger("rynn_scale.api.replay")

MAX_PLOT_STEPS = 200
_PLOT_TITLE_FONTSIZE = 10
_PLOT_DPI = 100
VIDEO_HEIGHT = 1080


def _collect_action_dims(action: RobotAction, side: str) -> Dict[str, np.ndarray]:
    """Flatten one arm + its gripper into a dim-name → ndarray dict.

    Returns the populated subset of {joint_position, eef_position, eef_rotation,
    gripper_position}; arm fields come from the side's ``Arm`` and the gripper
    comes from the matching top-level ``Position``.
    """
    dims: Dict[str, np.ndarray] = {}
    arm: Optional[Arm] = getattr(action, f"{side}_arm")
    if arm is not None:
        for f in fields(arm):
            v = getattr(arm, f.name)
            if v is not None:
                dims[f.name] = v.data.detach().cpu().numpy()
    gripper: Optional[Position] = getattr(action, f"{side}_gripper")
    if gripper is not None:
        dims["gripper_position"] = gripper.data.detach().cpu().numpy()
    return dims


def _make_dim_label(dim_name: str, d: int, num_dims: int) -> str:
    if num_dims <= 1:
        return dim_name
    if dim_name == "eef_position":
        suffix = ["x", "y", "z"][d] if d < 3 else f"d{d}"
    elif dim_name == "eef_rotation":
        suffix = f"r{d}"
    elif dim_name == "joint_position":
        suffix = f"j{d}"
    else:
        suffix = f"d{d}"
    return f"{dim_name}_{suffix}"


def _init_streaming_figure(
    gt_dims_left: Dict[str, np.ndarray],
    gt_dims_right: Dict[str, np.ndarray],
    has_right: bool,
    total_steps: int,
    chunk_size: int,
    plot_pred: bool,
    plot_state: bool = False,
    target_height: Optional[int] = None,
):
    """Pre-allocate the figure with one Line2D per (arm, dim, d) backed by a NaN buffer."""

    def count_dims(d):
        return sum(arr.shape[-1] for arr in d.values())

    num_left = count_dims(gt_dims_left)
    num_right = count_dims(gt_dims_right) if has_right else 0
    total_plots = num_left + num_right

    max_rows_per_col = 8
    n_rows = min(max_rows_per_col, max(1, total_plots))
    n_cols = max(1, (total_plots + max_rows_per_col - 1) // max_rows_per_col)

    col_width = MAX_PLOT_STEPS * 0.035
    max_fig_width = col_width * 3
    fig_width = col_width * n_cols
    if fig_width > max_fig_width:
        col_width = max_fig_width / n_cols
        fig_width = max_fig_width
    fig_height = 2.5 * n_rows

    dpi = _PLOT_DPI
    if target_height is not None and fig_height > 0:
        dpi = max(_PLOT_DPI, target_height / fig_height)

    fig, axes_arr = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        dpi=dpi,
    )
    # Column-major flatten so plots fill down each column before moving right.
    axes = axes_arr.T.flatten()

    axis_meta: List[Tuple[str, str, int]] = []
    gt_lines: Dict[Tuple[str, str, int], "plt.Line2D"] = {}
    pred_lines: Dict[Tuple[str, str, int], "plt.Line2D"] = {}
    state_lines: Dict[Tuple[str, str, int], "plt.Line2D"] = {}
    gt_buffers: Dict[Tuple[str, str, int], np.ndarray] = {}
    pred_buffers: Dict[Tuple[str, str, int], np.ndarray] = {}
    state_buffers: Dict[Tuple[str, str, int], np.ndarray] = {}
    scatter_artists: Dict[Tuple[str, str, int], "plt.PathCollection"] = {}
    # Insert a NaN gap slot after each chunk to break line connections between chunks
    num_chunks = (total_steps + chunk_size - 1) // chunk_size
    buf_len = total_steps + num_chunks  # data slots + NaN gap slots
    xs = np.arange(buf_len)
    idx = 0

    def configure(dims, arm_key, arm_label):
        nonlocal idx
        for dim_name, arr in dims.items():
            num_dims = arr.shape[-1]
            for d in range(num_dims):
                ax = axes[idx]
                title = f"{arm_label} / {_make_dim_label(dim_name, d, num_dims)}"
                ax.set_title(title, fontsize=_PLOT_TITLE_FONTSIZE, fontweight="medium")
                ax.tick_params(labelsize=9)
                ax.grid(True, alpha=0.3, linewidth=0.5)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_xlim(0, max(1, buf_len - 1))

                key = (arm_key, dim_name, d)
                gt_buffers[key] = np.full(buf_len, np.nan)
                (gt_line,) = ax.plot(
                    xs,
                    gt_buffers[key],
                    color="#2196F3",
                    linewidth=1.5,
                    alpha=0.9,
                    label="Ground Truth" if idx == 0 else None,
                )
                gt_lines[key] = gt_line
                if plot_pred:
                    pred_buffers[key] = np.full(buf_len, np.nan)
                    (pred_line,) = ax.plot(
                        xs,
                        pred_buffers[key],
                        color="#FF5722",
                        linewidth=1.5,
                        linestyle="--",
                        alpha=0.9,
                        label="Predicted" if idx == 0 else None,
                    )
                    pred_lines[key] = pred_line
                if plot_state:
                    state_buffers[key] = np.full(buf_len, np.nan)
                    (state_line,) = ax.plot(
                        xs,
                        state_buffers[key],
                        color="#4CAF50",
                        linewidth=1.5,
                        alpha=0.9,
                        label="State" if idx == 0 else None,
                    )
                    state_lines[key] = state_line

                scat = ax.scatter([], [], color="#2196F3", s=20, zorder=5)
                scatter_artists[key] = scat

                axis_meta.append(key)
                idx += 1

    configure(gt_dims_left, "left", "Left Arm")
    if has_right:
        configure(gt_dims_right, "right", "Right Arm")

    if total_plots > 0:
        axes[0].legend(fontsize=7, loc="upper right", framealpha=0.8)

    for k in range(total_plots, len(axes)):
        axes[k].set_visible(False)

    plt.tight_layout()
    return (
        fig,
        axes[:total_plots],
        axis_meta,
        gt_lines,
        pred_lines,
        state_lines,
        gt_buffers,
        pred_buffers,
        state_buffers,
        scatter_artists,
    )


def _concat_dim_dicts(chunks: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not chunks or not chunks[0]:
        return {}
    result = {}
    for dim_name in chunks[0]:
        result[dim_name] = np.concatenate([c[dim_name] for c in chunks])
    return result


def _prefill_plot_data(
    plot_axes,
    axis_meta: List[Tuple[str, str, int]],
    gt_lines: Dict[Tuple[str, str, int], "plt.Line2D"],
    pred_lines: Dict[Tuple[str, str, int], "plt.Line2D"],
    state_lines: Dict[Tuple[str, str, int], "plt.Line2D"],
    gt_buffers: Dict[Tuple[str, str, int], np.ndarray],
    pred_buffers: Dict[Tuple[str, str, int], np.ndarray],
    state_buffers: Dict[Tuple[str, str, int], np.ndarray],
    scatter_artists: Dict[Tuple[str, str, int], "plt.PathCollection"],
    all_gt_left: Dict[str, np.ndarray],
    all_gt_right: Dict[str, np.ndarray],
    all_pred_left: Dict[str, np.ndarray],
    all_pred_right: Dict[str, np.ndarray],
    all_state_left: Dict[str, np.ndarray],
    all_state_right: Dict[str, np.ndarray],
    chunk_size: int,
    seg_length: int,
):
    """Fill all plot buffers from pre-collected data and set final y-limits."""
    if seg_length == 0:
        return
    indices = np.arange(seg_length)
    chunk_indices = indices // chunk_size
    buf_indices = indices + chunk_indices
    num_chunks = int(chunk_indices[-1]) + 1

    for ax, key in zip(plot_axes, axis_meta):
        arm_key, dim_name, d = key
        gt_dims = all_gt_left if arm_key == "left" else all_gt_right
        pred_dims = all_pred_left if arm_key == "left" else all_pred_right
        state_dims = all_state_left if arm_key == "left" else all_state_right
        y_vals = []

        if dim_name in gt_dims:
            vals = gt_dims[dim_name][:, d].astype(np.float64)
            gt_buffers[key][buf_indices] = vals
            for ci in range(1, num_chunks):
                gap_buf = ci * chunk_size + ci
                gt_buffers[key][gap_buf - 1] = gt_buffers[key][gap_buf - 2]
            gt_lines[key].set_ydata(gt_buffers[key])
            y_vals.append(vals)
            chunk_start_steps = np.arange(0, seg_length, chunk_size)
            chunk_start_buf = chunk_start_steps + chunk_start_steps // chunk_size
            scatter_artists[key].set_offsets(np.column_stack([chunk_start_buf, vals[chunk_start_steps]]))

        if key in pred_lines and dim_name in pred_dims:
            vals = pred_dims[dim_name][:, d].astype(np.float64)
            pred_buffers[key][buf_indices] = vals
            pred_lines[key].set_ydata(pred_buffers[key])
            y_vals.append(vals)

        if key in state_lines and dim_name in state_dims:
            vals = state_dims[dim_name][:, d].astype(np.float64)
            state_buffers[key][buf_indices] = vals
            for ci in range(1, num_chunks):
                gap_buf = ci * chunk_size + ci
                state_buffers[key][gap_buf - 1] = state_buffers[key][gap_buf - 2]
            state_lines[key].set_ydata(state_buffers[key])
            y_vals.append(vals)

        if y_vals:
            combined = np.concatenate(y_vals)
            ymin, ymax = float(combined.min()), float(combined.max())
            margin = max(0.01, (ymax - ymin) * 0.05)
            ax.set_ylim(ymin - margin, ymax + margin)


def _render_vline_only(
    canvas: FigureCanvasAgg,
    fig,
    plot_axes,
    vlines: List,
    backgrounds: List,
) -> np.ndarray:
    """Redraw only the animated vlines over pre-cached static backgrounds."""
    for i, ax in enumerate(plot_axes):
        fig.canvas.restore_region(backgrounds[i])
        ax.draw_artist(vlines[i])
        fig.canvas.blit(ax.bbox)
    return np.ascontiguousarray(np.asarray(canvas.buffer_rgba())[..., :3])


def _fast_resize(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if target_h == h and target_w == w:
        return img
    pil_img = Image.fromarray(img)
    resample = Image.BOX if (target_h <= h and target_w <= w) else Image.BILINEAR
    pil_img = pil_img.resize((target_w, target_h), resample)
    return np.array(pil_img)


def _pil_resize(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    return _fast_resize(img, target_w, target_h)


def _resize_keep_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * target_h / h)))
    return _fast_resize(img, new_w, target_h)


def _resize_fit(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    return _fast_resize(img, target_w, target_h)


def _load_font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


_TEXT_FONT = _load_font(18)


_caption_cache: Dict[Tuple[str, int, int], np.ndarray] = {}


def _put_caption(img: np.ndarray, text: str, fontsize: int = 16) -> np.ndarray:
    w = img.shape[1]
    cache_key = (text, w, fontsize)
    bar = _caption_cache.get(cache_key)
    if bar is None:
        font = _load_font(fontsize)
        bar_h = fontsize + 10
        pil_bar = Image.new("RGB", (w, bar_h), (40, 40, 40))
        draw = ImageDraw.Draw(pil_bar)
        draw.text((8, (bar_h - fontsize) // 2), text, fill=(255, 255, 255), font=font)
        bar = np.array(pil_bar)
        _caption_cache[cache_key] = bar
    img[: bar.shape[0], :, :] = bar
    return img


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, font) -> List[str]:
    if not text:
        return [""]
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if (bbox[2] - bbox[0]) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _render_text_bar(text: str, width: int, max_lines: int = 2) -> np.ndarray:
    line_h = 24
    pad_y = 6
    bar_h = pad_y * 2 + line_h * max_lines
    if bar_h % 2:
        bar_h += 1
    pil_img = Image.new("RGB", (width, bar_h), (30, 30, 30))
    draw = ImageDraw.Draw(pil_img)
    safe_text = "" if text is None else str(text)
    lines = _wrap_text(draw, safe_text, max(1, width - 16), _TEXT_FONT)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            lines[-1] = lines[-1].rstrip() + "..."
    y = pad_y
    for line in lines:
        draw.text((8, y), line, fill=(240, 240, 240), font=_TEXT_FONT)
        y += line_h
    return np.array(pil_img)


def _video_col_height(panels: List[Tuple[str, np.ndarray]], panel_w: int) -> int:
    """Compute total video column height from panel aspect ratios (without captions)."""
    total = 0
    for _, img in panels:
        if img is None:
            total += panel_w * 3 // 4
        else:
            h, w = img.shape[:2]
            total += max(1, int(round(h * panel_w / w)))
    return max(1, total)


def _stack_panels_keep_ratio(
    panels: List[Tuple[str, np.ndarray]],
    panel_w: int,
    caption_fontsize: int = 16,
) -> np.ndarray:
    """Stack panels vertically, each resized to panel_w while keeping its aspect ratio."""
    if not panels:
        return np.zeros((1, panel_w, 3), dtype=np.uint8)

    stacked = []
    for label, img in panels:
        if img is None:
            img = np.zeros((panel_w * 3 // 4, panel_w, 3), dtype=np.uint8)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        h, w = img.shape[:2]
        if w != panel_w:
            new_h = max(1, int(round(h * panel_w / w)))
            img = _pil_resize(img, panel_w, new_h)
        img = _put_caption(img, label, fontsize=caption_fontsize)
        stacked.append(img)
    return np.vstack(stacked)


def _compose_frame(
    plot_img: np.ndarray,
    video_panels: List[Tuple[str, np.ndarray]],
    video_width: int,
    text: Optional[str] = None,
) -> np.ndarray:
    # Compute the scale factor so caption font matches matplotlib title in final frame
    col_h = _video_col_height(video_panels, video_width)
    plot_h = plot_img.shape[0]
    scale = col_h / plot_h
    # matplotlib title: _PLOT_TITLE_FONTSIZE pt at _PLOT_DPI → pixel size, then scaled
    caption_fontsize = int(round(_PLOT_TITLE_FONTSIZE / 72 * _PLOT_DPI * scale))
    caption_fontsize = max(10, min(caption_fontsize, 48))

    video_col = _stack_panels_keep_ratio(video_panels, video_width, caption_fontsize)
    total_h = video_col.shape[0]
    plot_panel = _resize_keep_height(plot_img, total_h)
    columns = np.hstack([plot_panel, video_col])
    if text is None:
        return columns
    text_bar = _render_text_bar(text, columns.shape[1])
    return np.vstack([text_bar, columns])


_PREFETCH_CAPACITY = 16


class _ImagePrefetcher:
    """Background-thread image loader: reads and resizes frames ahead of the
    rendering loop so that I/O overlaps with matplotlib / MuJoCo work."""

    def __init__(self, dataset, ep_idx, seg_start, seg_length, video_width):
        self._length = seg_length
        self._queue: queue.Queue = queue.Queue(maxsize=_PREFETCH_CAPACITY)
        self._thread = threading.Thread(
            target=self._worker,
            args=(dataset, ep_idx, seg_start, seg_length, video_width),
            daemon=True,
        )
        self._thread.start()

    def _worker(self, dataset, ep_idx, seg_start, seg_length, video_width):
        seg_iter = dataset.iter_episode(
            episode_index=ep_idx,
            start=seg_start,
            step=1,
            include_images=True,
        )
        try:
            for _ in range(seg_length):
                data = next(seg_iter)
                resized = {}
                for k, v in data["images"].items():
                    if v.ndim == 4:
                        v = v[0]
                    h, w = v.shape[0], v.shape[1]
                    target_h = max(1, int(round(h * video_width / w)))
                    t = v.permute(2, 0, 1).unsqueeze(0).float()
                    t = torch.nn.functional.interpolate(
                        t,
                        size=(target_h, video_width),
                        mode="area",
                    )
                    resized[k] = t.squeeze(0).permute(1, 2, 0).to(torch.uint8).numpy()
                self._queue.put(resized)
        except Exception as e:
            self._queue.put(e)
        finally:
            seg_iter.close()

    def __iter__(self):
        for _ in range(self._length):
            item = self._queue.get()
            if isinstance(item, Exception):
                raise item
            yield item

    def __len__(self):
        return self._length

    def close(self):
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=5)


def _is_relative_chunk(chunk: RobotAction) -> bool:
    """A populated atom anywhere in ``chunk`` carries the relative flag."""
    for _, v in chunk._fields():
        if isinstance(v, Arm):
            for _, sub in v._fields():
                return bool(sub.is_relative)
        else:
            return bool(v.is_relative)
    return False


def _to_absolute_chunk(chunk: RobotAction, state: RobotState) -> RobotAction:
    if _is_relative_chunk(chunk):
        return chunk + state
    return chunk


def main():
    register_projects()

    parser = HfArgumentParser(ReplayArguments)
    args = parser.parse_args_into_dataclasses()[0]

    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        args.save_dir = f"replay_{timestamp}"

    os.makedirs(args.save_dir, exist_ok=True)

    use_model = args.model_path is not None

    if use_model:
        inference_wrapper = build_inference_wrapper(
            model_type=args.model_type,
            model_path=args.model_path,
            dtype=args.param_dtype,
            attn_implementation=args.attn_implementation,
        )
    else:
        inference_wrapper = None

    dataset = build_dataset(args)
    chunk_size = args.action_chunk_size

    metadata = dataset.metadata
    total_episodes = len(metadata)

    if args.robot_types:
        robot_type_set = set(args.robot_types)
        eligible_indices = [i for i in range(total_episodes) if metadata[i].robot_type.value in robot_type_set]
        if not eligible_indices:
            raise ValueError(
                f"No episodes match robot_types={args.robot_types}. "
                f"Available: {sorted({m.robot_type.value for m in metadata})}"
            )
        logger.info(
            f"Filtered to {len(eligible_indices)}/{total_episodes} episodes matching robot_types={args.robot_types}"
        )
    else:
        eligible_indices = list(range(total_episodes))

    rng = np.random.default_rng(args.seed)

    def _episode_fps(ep_idx: int) -> float:
        fps = metadata[ep_idx].fps
        if not fps:
            raise ValueError(f"Episode {ep_idx} has no fps in metadata; cannot determine the video framerate.")
        return float(fps)

    def _sample_segment():
        """Draw a random episode, then a random segment within it."""
        ep_idx = eligible_indices[int(rng.integers(0, len(eligible_indices)))]
        ep_length = int(metadata[ep_idx].length)
        n_seg = (ep_length + MAX_PLOT_STEPS - 1) // MAX_PLOT_STEPS
        if args.max_segments_per_episode > 0:
            n_seg = min(n_seg, args.max_segments_per_episode)
        seg_idx = int(rng.integers(0, n_seg))
        return ep_idx, ep_length, seg_idx

    robot_type = None
    renderer: Optional[object] = None
    _prev_rt_key: Optional[str] = None
    video_width: Optional[int] = None

    def _run_inference(data, state):
        model_inputs = inference_wrapper.process(
            text=data["text"],
            images=data["images"],
            state=state.to_dict(),
            robot_type=data["robot_type"].value,
        )
        for key, value in model_inputs.items():
            if torch.is_tensor(value):
                model_inputs[key] = value.to(inference_wrapper.model.device)
        with torch.inference_mode():
            cache = inference_wrapper.prefill(model_inputs)
            pred_actions = inference_wrapper.decode(
                model_inputs,
                cache,
                num_steps=args.num_inference_steps,
                # One sample per call, so the per-element fields are one-element lists.
                # Replay steps a recorded episode open-loop and holds nothing in flight,
                # so there is no Real-Time Chunking state to report.
                robot_type=[data["robot_type"].value],
                prev_actions=[None],
                delay_steps=[0],
            )
        action_dict = inference_wrapper.post_process(
            pred_actions[0].cpu(),
            state=state.to_dict(),
            robot_type=data["robot_type"].value,
        )
        return RobotAction.from_dict(action_dict)

    try:
        for replay_idx in range(args.num_segments):
            ep_idx, ep_length, seg_idx = _sample_segment()
            seg_start = seg_idx * MAX_PLOT_STEPS
            seg_length = min(MAX_PLOT_STEPS, ep_length - seg_start)
            num_chunks = (seg_length + chunk_size - 1) // chunk_size

            fps_native = _episode_fps(ep_idx)
            t_start = seg_start / fps_native
            t_end = (seg_start + seg_length) / fps_native

            seg_tag = f"Episode {ep_idx} [{t_start:.1f}-{t_end:.1f}s]"
            save_path = os.path.join(
                args.save_dir,
                f"episode_{ep_idx:06d}_{t_start:.1f}-{t_end:.1f}s.mp4",
            )

            # ---- Peek first frame for video_width and panel sizing ----
            peek_iter = dataset.iter_episode(
                episode_index=ep_idx,
                start=seg_start,
                step=1,
                include_images=True,
            )
            peek_data = next(peek_iter)
            peek_iter.close()

            if video_width is None:
                aspect_sum = 0.0
                for k, v in peek_data["images"].items():
                    img = v[0] if v.ndim == 4 else v
                    aspect_sum += img.shape[0] / img.shape[1]
                rt_key_tmp = (
                    peek_data["robot_type"].value
                    if hasattr(peek_data["robot_type"], "value")
                    else peek_data["robot_type"]
                )
                if rt_key_tmp in RENDERER_REGISTRY:
                    aspect_sum += 1.0
                video_width = max(1, int(round(VIDEO_HEIGHT / aspect_sum)))

            peek_panels: List[Tuple[str, np.ndarray]] = []
            for k, v in sorted(peek_data["images"].items()):
                img = v[0] if v.ndim == 4 else v
                h, w = img.shape[0], img.shape[1]
                target_h = max(1, int(round(h * video_width / w)))
                peek_panels.append((k, np.zeros((target_h, video_width, 3), dtype=np.uint8)))

            # ---- Pass 1: Collect metadata (skip images unless model needs them) ----
            all_texts: List[str] = []
            all_states: List[RobotState] = []
            all_gt_chunks: List[RobotAction] = []
            all_pred_chunks: List[RobotAction] = []
            gt_dims_left_chunks: List[Dict[str, np.ndarray]] = []
            gt_dims_right_chunks: List[Dict[str, np.ndarray]] = []
            pred_dims_left_chunks: List[Dict[str, np.ndarray]] = []
            pred_dims_right_chunks: List[Dict[str, np.ndarray]] = []
            has_right = False

            seg_iter = dataset.iter_episode(
                episode_index=ep_idx,
                start=seg_start,
                step=1,
                include_images=use_model,
            )
            try:
                for c in tqdm(range(num_chunks), desc=f"{seg_tag} (loading)"):
                    chunk_start = c * chunk_size
                    K_actual = min(chunk_size, seg_length - chunk_start)
                    gt_chunk_abs: Optional[RobotAction] = None
                    pred_chunk_abs: Optional[RobotAction] = None

                    for j in range(K_actual):
                        data = next(seg_iter)
                        text_val = data.get("text", "")
                        if isinstance(text_val, (list, tuple)):
                            text_val = text_val[0] if text_val else ""
                        text_str = "" if text_val is None else str(text_val)
                        all_texts.append(f"{seg_tag} {text_str}")
                        all_states.append(data["state"])
                        if j == 0:
                            robot_type = data["robot_type"]
                            gt_chunk_abs = _to_absolute_chunk(data["action"], data["state"])
                            if use_model:
                                pred_chunk_abs = _run_inference(data, data["state"])

                    has_right = gt_chunk_abs.right_arm is not None
                    all_gt_chunks.append(gt_chunk_abs)
                    if use_model:
                        all_pred_chunks.append(pred_chunk_abs)

                    gt_dl = {k: v[:K_actual] for k, v in _collect_action_dims(gt_chunk_abs, "left").items()}
                    gt_dr = {
                        k: v[:K_actual]
                        for k, v in (_collect_action_dims(gt_chunk_abs, "right") if has_right else {}).items()
                    }
                    gt_dims_left_chunks.append(gt_dl)
                    gt_dims_right_chunks.append(gt_dr)

                    if use_model:
                        pd_l = {k: v[:K_actual] for k, v in _collect_action_dims(pred_chunk_abs, "left").items()}
                        pd_r = {
                            k: v[:K_actual]
                            for k, v in (_collect_action_dims(pred_chunk_abs, "right") if has_right else {}).items()
                        }
                        pred_dims_left_chunks.append(pd_l)
                        pred_dims_right_chunks.append(pd_r)
            finally:
                seg_iter.close()

            # Concatenate action dims across chunks
            all_gt_left = _concat_dim_dicts(gt_dims_left_chunks)
            all_gt_right = _concat_dim_dicts(gt_dims_right_chunks)
            all_pred_left = _concat_dim_dicts(pred_dims_left_chunks) if use_model else {}
            all_pred_right = _concat_dim_dicts(pred_dims_right_chunks) if use_model else {}

            # Stack per-step state dims
            state_left_steps = [_collect_action_dims(s, "left") for s in all_states]
            all_state_left: Dict[str, np.ndarray] = {}
            if state_left_steps and state_left_steps[0]:
                for dim_name in state_left_steps[0]:
                    all_state_left[dim_name] = np.stack([sd[dim_name].squeeze(0) for sd in state_left_steps])
            all_state_right: Dict[str, np.ndarray] = {}
            if has_right:
                state_right_steps = [_collect_action_dims(s, "right") for s in all_states]
                if state_right_steps and state_right_steps[0]:
                    for dim_name in state_right_steps[0]:
                        all_state_right[dim_name] = np.stack([sd[dim_name].squeeze(0) for sd in state_right_steps])

            # ---- Initialize renderer ----
            rt_key = robot_type.value if hasattr(robot_type, "value") else robot_type
            if rt_key != _prev_rt_key:
                if renderer is not None:
                    renderer.close()
                    renderer = None
                _prev_rt_key = rt_key
                if rt_key in RENDERER_REGISTRY:
                    renderer = build_renderer(
                        robot_type,
                        height=args.render_size,
                        width=args.render_size,
                        action_source=args.action_source,
                    )
                else:
                    logger.warning(f"No renderer registered for robot_type={rt_key}; skipping 3D render panel")

            # ---- Initialize figure and pre-fill all data ----
            est_panels = list(peek_panels)
            if rt_key in RENDERER_REGISTRY:
                est_panels.append(("mj", np.zeros((args.render_size, args.render_size, 3), dtype=np.uint8)))
            target_col_h = _video_col_height(est_panels, video_width)

            (
                fig,
                plot_axes,
                axis_meta,
                gt_lines,
                pred_lines,
                state_lines,
                gt_buffers,
                pred_buffers,
                state_buffers,
                scatter_artists,
            ) = _init_streaming_figure(
                all_gt_left,
                all_gt_right,
                has_right,
                total_steps=seg_length,
                chunk_size=chunk_size,
                plot_pred=use_model,
                plot_state=bool(all_state_left),
                target_height=target_col_h,
            )
            canvas = FigureCanvasAgg(fig)
            vlines = [ax.axvline(0, color="red", linewidth=1.2, alpha=0.85) for ax in plot_axes]
            for vl in vlines:
                vl.set_animated(True)

            _prefill_plot_data(
                plot_axes,
                axis_meta,
                gt_lines,
                pred_lines,
                state_lines,
                gt_buffers,
                pred_buffers,
                state_buffers,
                scatter_artists,
                all_gt_left,
                all_gt_right,
                all_pred_left,
                all_pred_right,
                all_state_left,
                all_state_right,
                chunk_size,
                seg_length,
            )

            # Draw static background once (lines + scatter baked in)
            canvas.draw()
            backgrounds = [fig.canvas.copy_from_bbox(ax.bbox) for ax in plot_axes]

            # ---- Pass 2: Render frames (prefetch images in background thread) ----
            prefetcher = _ImagePrefetcher(
                dataset,
                ep_idx,
                seg_start,
                seg_length,
                video_width,
            )
            video_writer = None
            try:
                for i, obs in enumerate(
                    tqdm(
                        prefetcher,
                        total=seg_length,
                        desc=f"{seg_tag} (rendering)",
                    )
                ):
                    chunk_idx = i // chunk_size
                    j_in_chunk = i % chunk_size
                    buf_idx = i + chunk_idx

                    for vl in vlines:
                        vl.set_xdata([buf_idx, buf_idx])

                    plot_img = _render_vline_only(canvas, fig, plot_axes, vlines, backgrounds)

                    video_panels: List[Tuple[str, np.ndarray]] = [(k, obs[k]) for k in sorted(obs.keys())]

                    if renderer is not None:
                        video_panels.append(("GT Action", renderer.render(all_gt_chunks[chunk_idx][j_in_chunk])))
                        if use_model:
                            video_panels.append(
                                ("Pred Action", renderer.render(all_pred_chunks[chunk_idx][j_in_chunk]))
                            )

                    frame = _compose_frame(
                        plot_img,
                        video_panels,
                        video_width=video_width,
                        text=all_texts[i],
                    )
                    h, w = frame.shape[:2]
                    if h % 2:
                        frame = frame[:-1]
                    if w % 2:
                        frame = frame[:, :-1]

                    if video_writer is None:
                        video_writer = AsyncVideoWriter(save_path, fps=round(fps_native))
                    video_writer.write(frame)
            finally:
                prefetcher.close()
                if video_writer is not None:
                    video_writer.close()
                if fig is not None:
                    plt.close(fig)

            logger.info(f"{seg_tag} replay video saved to: {save_path}")
    finally:
        if renderer is not None:
            renderer.close()


if __name__ == "__main__":
    main()
