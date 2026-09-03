import copy
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm

from ..inference_wrappers import BaseInferenceWrapper
from ..registry import BENCHMARK_REGISTRY
from .base import BaseBenchmark


# ---------------------------------------------------------------------------
# 3D geometry utilities
# ---------------------------------------------------------------------------

def _polygon_area(vertices: np.ndarray) -> float:
    n = len(vertices)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]
    return abs(area) / 2.0


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    from scipy.spatial import ConvexHull

    if len(points) < 3:
        return points
    try:
        hull = ConvexHull(points)
        return points[hull.vertices]
    except Exception:
        return points


def _polygon_intersection_area(poly1: np.ndarray, poly2: np.ndarray) -> float:
    from scipy.spatial import ConvexHull

    def _point_in_convex_polygon(point, polygon):
        n = len(polygon)
        for i in range(n):
            j = (i + 1) % n
            cross = (
                (polygon[j, 0] - polygon[i, 0]) * (point[1] - polygon[i, 1])
                - (polygon[j, 1] - polygon[i, 1]) * (point[0] - polygon[i, 0])
            )
            if cross < -1e-10:
                return False
        return True

    def _line_segment_intersection(p1, p2, p3, p4):
        d1 = p2 - p1
        d2 = p4 - p3
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-10:
            return None
        t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
        u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross
        if 0 <= t <= 1 and 0 <= u <= 1:
            return p1 + t * d1
        return None

    hull1 = _convex_hull_2d(poly1)
    hull2 = _convex_hull_2d(poly2)

    intersection_points = []
    for p in hull1:
        if _point_in_convex_polygon(p, hull2):
            intersection_points.append(p)
    for p in hull2:
        if _point_in_convex_polygon(p, hull1):
            intersection_points.append(p)

    for i in range(len(hull1)):
        j = (i + 1) % len(hull1)
        for k in range(len(hull2)):
            l_idx = (k + 1) % len(hull2)
            pt = _line_segment_intersection(hull1[i], hull1[j], hull2[k], hull2[l_idx])
            if pt is not None:
                intersection_points.append(pt)

    if len(intersection_points) < 3:
        return 0.0
    intersection_points = np.array(intersection_points)
    try:
        hull = ConvexHull(intersection_points)
        return hull.volume
    except Exception:
        return 0.0


def _box3d_corners(center: np.ndarray, dims_whl: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Build 8 corners from center, dims=[W,H,L], and rotation matrix."""
    w, h, l = dims_whl[0], dims_whl[1], dims_whl[2]
    x = np.array([w / 2, 0, 0])
    y = np.array([0, h / 2, 0])
    z = np.array([0, 0, l / 2])
    corners = np.array([
        -x - y - z, -x - y + z, -x + y - z, -x + y + z,
        +x - y - z, +x - y + z, +x + y - z, +x + y + z,
    ])
    corners = (R @ corners.T).T + center
    return corners


def _box3d_iou(corners1: np.ndarray, corners2: np.ndarray) -> float:
    bev1 = corners1[:, [0, 2]]
    bev2 = corners2[:, [0, 2]]
    inter_2d = _polygon_intersection_area(bev1, bev2)
    if inter_2d < 1e-10:
        return 0.0
    y1_min, y1_max = corners1[:, 1].min(), corners1[:, 1].max()
    y2_min, y2_max = corners2[:, 1].min(), corners2[:, 1].max()
    y_inter = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    if y_inter < 1e-10:
        return 0.0
    inter_3d = inter_2d * y_inter
    area1 = _polygon_area(_convex_hull_2d(bev1))
    area2 = _polygon_area(_convex_hull_2d(bev2))
    vol1 = area1 * (y1_max - y1_min)
    vol2 = area2 * (y2_max - y2_min)
    union_3d = vol1 + vol2 - inter_3d
    if union_3d < 1e-10:
        return 0.0
    return inter_3d / union_3d


def _euler_to_rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Euler angles (pitch=Rx, yaw=Ry, roll=Rz) to rotation matrix R = Rz @ Ry @ Rx."""
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cr, sr = np.cos(roll), np.sin(roll)

    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])#pitch
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])#yaw
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])#Roll
    return Rz @ Ry @ Rx


def _canonicalize_rotation(R_cam: np.ndarray, dims_whl: np.ndarray) -> np.ndarray:
    R_out = np.array(R_cam, dtype=np.float64).copy()
    w, h, l = float(dims_whl[0]), float(dims_whl[1]), float(dims_whl[2])
    if w > l:
        col0 = R_out[:, 0].copy()
        R_out[:, 0] = -R_out[:, 2]
        R_out[:, 2] = col0
    yaw = np.arctan2(-R_out[2, 0], R_out[0, 0])
    if yaw < 0 or yaw > np.pi - 1e-4:
        R_out[:, 0] = -R_out[:, 0]
        R_out[:, 2] = -R_out[:, 2]
    return R_out


def _rotation_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    R_rel = R1 @ R2.T
    trace = np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0)
    return np.arccos(trace)


def _scale_iou(dims1: np.ndarray, dims2: np.ndarray) -> float:
    min_whl = np.minimum(dims1, dims2)
    vol1 = np.prod(dims1)
    vol2 = np.prod(dims2)
    inter = np.prod(min_whl)
    union = vol1 + vol2 - inter
    if union < 1e-10:
        return 0.0
    return inter / union


# ---------------------------------------------------------------------------
# VOC-style AP calculation (adapted from 3DETR eval_det.py)
# ---------------------------------------------------------------------------


def _voc_ap(rec, prec):
    """Compute VOC AP given precision and recall arrays."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _eval_det_cls(pred, gt, ovthresh=0.25):
    """Compute precision/recall/AP for a single class.

    pred: {img_id: [(corners_8x3, score), ...]}
    gt:   {img_id: [corners_8x3, ...]}
    """
    class_recs = {}
    npos = 0
    for img_id in gt:
        bbox = gt[img_id]
        det = [False] * len(bbox)
        npos += len(bbox)
        class_recs[img_id] = {"bbox": bbox, "det": det}
    for img_id in pred:
        if img_id not in class_recs:
            class_recs[img_id] = {"bbox": [], "det": []}

    image_ids = []
    confidence = []
    BB = []
    for img_id in pred:
        for box, score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            BB.append(box)

    if len(BB) == 0:
        return np.array([]), np.array([]), 0.0

    confidence = np.array(confidence)
    sorted_ind = np.argsort(-confidence)
    BB = [BB[i] for i in sorted_ind]
    image_ids = [image_ids[i] for i in sorted_ind]

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d]
        ovmax = -np.inf
        BBGT = R["bbox"]
        jmax = -1

        for j, bbgt in enumerate(BBGT):
            iou = _box3d_iou(bb, bbgt)
            if iou > ovmax:
                ovmax = iou
                jmax = j

        if ovmax > ovthresh:
            if not R["det"][jmax]:
                tp[d] = 1.0
                R["det"][jmax] = True
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos) if npos > 0 else np.zeros_like(tp)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = _voc_ap(rec, prec)
    return rec, prec, ap


# ---------------------------------------------------------------------------
# Spatial attribute generation for disambiguation
# ---------------------------------------------------------------------------

def _generate_attr(ann: Dict, img_width: int, img_height: int) -> str:
    bbox = ann["bbox2D_proj"]
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2

    if cx < img_width / 3:
        h_pos = "on the left"
    elif cx > img_width * 2 / 3:
        h_pos = "on the right"
    else:
        h_pos = "in the center"

    if cy < img_height / 3:
        v_pos = "at the top"
    elif cy > img_height * 2 / 3:
        v_pos = "at the bottom"
    else:
        v_pos = "in the middle"

    return f"{h_pos} and {v_pos} of the image"


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

# DETECT3D_PROMPT_TEMPLATE = """\
# Find the {cat}.

# The camera intrinsics matrix is:
# {K}

# Predict one 3D bounding box in the camera coordinate system, where:
# - x points to the right
# - y points downward
# - z points forward

# Return:
# <3D Grounding> cx, cy, cz, w, l, h, pitch, yaw, roll </3D Grounding>

# Definitions:
# - cx, cy, cz: 3D coordinates of the box center in the camera coordinate system, in meters
# - w, l, h: box dimensions in the box local coordinate system, in meters
#   - w: extent along the local z-axis
#   - l: extent along the local x-axis
#   - h: extent along the local y-axis
# - pitch: rotation around the camera x-axis, in radians
# - yaw: rotation around the camera y-axis, in radians
# - roll: rotation around the camera z-axis, in radians

# Constraints:
# - w <= l
# - Use meters for cx, cy, cz, w, l, h
# - Use radians for pitch, yaw, roll
# - Output exactly one 3D bounding box"""



# DETECT3D_PROMPT_TEMPLATE = """Locate the antenna that is {cat} on the right side

# Camera intrinsics:
# {K}
# Predict the 3D grounding box in meters with pitch, yaw, roll in radians. Respond with the result in JSON format.
# """
# DETECT3D_IMAGE_ROOTS = {}

# _BOX_EDGES = [
#     (0, 1), (2, 3), (4, 5), (6, 7),
#     (0, 2), (1, 3), (4, 6), (5, 7),
#     (0, 4), (1, 5), (2, 6), (3, 7),
# ]

# DETECT3D_PROMPT_TEMPLATE = """Locate the antenna that is {cat} on the right side

# Camera intrinsics:
# {K}
# Predict the 3D grounding box in meters with pitch, yaw, roll in radians. Respond with the result in JSON format.
# """


DETECT3D_PROMPT_TEMPLATE = """Find the {cat}.

The camera intrinsics matrix is:
{K}

Predict one 3D bounding box in the camera coordinate system, where:
- x points to the right
- y points downward
- z points forward

Return:
<3D Grounding> cx, cy, cz, x_size, y_size, z_size, pitch, yaw, roll </3D Grounding>

Definitions:
- cx, cy, cz: 3D coordinates of the box center in the camera coordinate system, in meters
- x_size, y_size, z_size: box dimensions in the box local coordinate system, in meters
  - x_size: extent along the local x-axis
  - y_size: extent along the local y-axis
  - z_size: extent along the local z-axis
- pitch: rotation around the camera x-axis, in radians (range: -1 to 1, corresponding to -180° to 180°)
- yaw: rotation around the camera y-axis, in radians (range: -1 to 1, corresponding to -180° to 180°)
- roll: rotation around the camera z-axis, in radians (range: -1 to 1, corresponding to -180° to 180°)

Constraints:
- x_size >= z_size
- Use meters for cx, cy, cz, x_size, y_size, z_size
- Use radians for pitch, yaw, roll, with each value in the range [-1, 1]
- Output exactly one 3D bounding box
<think>\n\n</think>\n\n
"""

DETECT3D_IMAGE_ROOTS = {}

_BOX_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _project_corners_to_2d(corners: np.ndarray, K: np.ndarray) -> List:
    """Project 8×3 corners to 2D pixel coordinates using intrinsics K (3×3)."""
    pts_2d = []
    for pt in corners:
        if pt[2] <= 0 or not np.isfinite(pt).all():
            pts_2d.append(None)
            continue
        px = K[0, 0] * (pt[0] / pt[2]) + K[0, 2]
        py = K[1, 1] * (pt[1] / pt[2]) + K[1, 2]
        if not np.isfinite(px) or not np.isfinite(py):
            pts_2d.append(None)
            continue
        pts_2d.append((int(round(px)), int(round(py))))
    return pts_2d


def _draw_box_edges(img: np.ndarray, pts_2d: List, color: tuple,
                    thickness: int = 2, edges: List = None):
    import cv2
    for i, j in (edges or _BOX_EDGES):
        if pts_2d[i] is not None and pts_2d[j] is not None:
            cv2.line(img, pts_2d[i], pts_2d[j], color, thickness)

# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------

@BENCHMARK_REGISTRY.register()
class Detect3D(BaseBenchmark):
    def __init__(
        self,
        data_root: str,
        inference_wrapper: BaseInferenceWrapper,
        prompt_format: Optional[str] = None,
        enable_thinking: bool = False,
        iou_thresholds: Optional[List[float]] = None,
        parse_json: bool = False,
    ) -> None:
        self.iou_thresholds = iou_thresholds or [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        self.parse_json = parse_json
        self._gt_by_ann_id: Dict[int, Dict] = {}
        self._categories: Dict[int, str] = {}
        self._cat_name_to_id: Dict[str, int] = {}
        self._prompt_cache: Dict[Union[int, str], Dict[str, Any]] = {}
        super().__init__(
            data_root=data_root,
            inference_wrapper=inference_wrapper,
            prompt_format=prompt_format,
            enable_thinking=enable_thinking,
        )

    @staticmethod
    def _resolve_image_path(data_root: str, file_path: str) -> str:
        for prefix, image_root in DETECT3D_IMAGE_ROOTS.items():
            if file_path.startswith(prefix + "/"):
                return os.path.join(image_root, os.path.basename(file_path))
            if file_path.startswith(prefix):
                return os.path.join(image_root, file_path[len(prefix):].lstrip("/"))
        return os.path.join(data_root, file_path)

    def _format_K(self, K: List) -> str:
        rows = []
        for row in K:
            rows.append("[" + ", ".join(f"{v:.2f}" for v in row) + "]")
        return "[" + ", ".join(rows) + "]"

    # Toggle phrase-augmented prompt: uncomment to enable, comment to disable
    USE_PHRASE = True

    def load_data(self, data_root: str) -> Dict[Union[int, str], Any]:
        ann_path = os.path.join(data_root, "annotations", "InTheWild_v3_val_with_phrase.json")
        if not os.path.exists(ann_path):
            ann_path = os.path.join(data_root, "annotations", "InTheWild_v3_val.json")
        if not os.path.exists(ann_path):
            ann_path = os.path.join(data_root, "annotations.json")
        with open(ann_path, "r") as f:
            ann_data = json.load(f)

        for cat in ann_data["categories"]:
            self._categories[cat["id"]] = cat["name"]
            self._cat_name_to_id[cat["name"]] = cat["id"]

        max_images = int(os.environ.get("DETECT3D_MAX_IMAGES", 0))
        images_list = ann_data["images"]
        if max_images > 0:
            images_list = images_list[:max_images]

        images_by_id = {}
        for img in images_list:
            images_by_id[img["id"]] = img

        data_dict = {}
        n_skipped_missing = 0
        for ann in ann_data["annotations"]:
            if not ann.get("valid3D", True):
                continue
            if ann["image_id"] not in images_by_id:
                continue

            img_info = images_by_id[ann["image_id"]]
            file_path = self._resolve_image_path(data_root, img_info["file_path"])
            if not os.path.exists(file_path):
                if n_skipped_missing < 10:
                    print(f"[Detect3D] Skipping missing image: {file_path}")
                n_skipped_missing += 1
                continue

            center = np.array(ann["center_cam"], dtype=np.float64)
            dims_raw = np.array(ann["dimensions"], dtype=np.float64)
            # annotation stores [L, H, W] (z, y, x); convert to [W, H, L] (x, y, z)
            dims = np.array([dims_raw[2], dims_raw[1], dims_raw[0]], dtype=np.float64)
            R = np.array(ann["R_cam"], dtype=np.float64)

            if "bbox3D_cam" in ann and ann["bbox3D_cam"] is not None:
                corners = np.array(ann["bbox3D_cam"], dtype=np.float64)
            else:
                corners = _box3d_corners(center, dims, R)

            ann_id = ann["id"]
            cat_name = ann.get("category_name", self._categories.get(ann["category_id"], ""))
            attr = _generate_attr(ann, img_info["width"], img_info["height"])

            gt_entry = {
                "ann_id": ann_id,
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "category_name": cat_name,
                "center_cam": ann["center_cam"],
                "dimensions": dims.tolist(),  # [W, H, L] (x, y, z)
                "R_cam": ann["R_cam"],
                "corners": corners,
                "depth": center[2],
            }
            self._gt_by_ann_id[ann_id] = gt_entry

            data_dict[ann_id] = {
                "images": [file_path],
                "image_id": ann["image_id"],
                "ann_id": ann_id,
                "width": img_info["width"],
                "height": img_info["height"],
                "K": img_info.get("K", None),
                "category_name": cat_name,
                "attributive_phrase": ann.get("attributive_phrase", ""),
                "attr": attr,
                "ground_truth": gt_entry,
                "task_type": "detect3d",
            }

        print(f"[Detect3D] Loaded {len(data_dict)} samples "
              f"({len(images_by_id)} images, {len(self._categories)} categories)"
              f"{f', skipped {n_skipped_missing} (image not found)' if n_skipped_missing else ''}")
        return data_dict

    def _build_cat_description(self, meta: Dict) -> str:
        cat = meta["category_name"]
        phrase = meta.get("attributive_phrase", "")
        if self.USE_PHRASE and phrase:
            return f"{cat} which is {phrase}"
        return cat

    def generate_instruction(self, data_id: Union[int, str]) -> List[Dict[str, Any]]:
        meta = self.data_dict[data_id]

        K_str = self._format_K(meta["K"]) if meta["K"] else "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"
        cat_desc = self._build_cat_description(meta)
        prompt = DETECT3D_PROMPT_TEMPLATE.format(
            cat=cat_desc,
            K=K_str,
        )

        if self.enable_thinking:
            prompt = "Think step by step about the object's position and orientation in 3D space.\n" + prompt

        self._prompt_cache[data_id] = {
            "prompt": prompt,
            "use_phrase": self.USE_PHRASE,
            "attributive_phrase": meta.get("attributive_phrase", ""),
        }

        contents = [
            {"type": "image", "image": meta["images"][0]},
            {"type": "text", "text": prompt},
        ]
        return [{"role": "user", "content": contents}]

    async def process_response(self, data_id: Union[int, str], response: str) -> Any:
        if self.enable_thinking and "</think>" in response:
            response = response.split("</think>")[-1]

        print(f"response is {response}")

        parsed = self._parse_3d_grounding(response)

        # print(f"parsed is {parsed}")

        if parsed is not None:
            center = np.array(parsed["center_cam"], dtype=np.float64)
            dims_wlh = np.array(parsed["dims_wlh"], dtype=np.float64)
            R = np.array(parsed["R_cam"], dtype=np.float64)
            # model output [w, l, h] matches annotation [L, H, W] convention:
            #   w -> L (z-axis), l -> W (x-axis), h -> H (y-axis)
            # convert to [W, H, L] (x, y, z) for _box3d_corners
            # dims_whl = np.array([dims_wlh[1], dims_wlh[2], dims_wlh[0]])
            dims_whl = np.array([dims_wlh[0], dims_wlh[1], dims_wlh[2]])
            corners = _box3d_corners(center, dims_whl, R)
            parsed["corners"] = corners.tolist()
            parsed["dimensions"] = dims_whl.tolist()  # [W, H, L] (x, y, z)

        prompt_info = self._prompt_cache.get(data_id, {})
        return json.dumps({
            "ann_id": data_id,
            "prompt": prompt_info.get("prompt", ""),
            "use_phrase": prompt_info.get("use_phrase", False),
            "attributive_phrase": prompt_info.get("attributive_phrase", ""),
            "parsed": parsed,
        })

    def _parse_3d_grounding(self, response: str) -> Optional[Dict]:
        box_3d = None
        label = ""

        # Try <3D Grounding> cx, cy, cz, x_size, y_size, z_size, pitch, yaw, roll </3D Grounding>
        grounding_match = re.search(r"<3D Grounding>\s*(.*?)\s*</3D Grounding>", response)
        if grounding_match:
            try:
                values = [float(v.strip()) for v in grounding_match.group(1).split(",")]
                if len(values) == 9:
                    box_3d = values
            except ValueError:
                pass

        # Try <<<<values> fallback
        if box_3d is None:
            angle_pat = (
                r"<+\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,"
                r"\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,"
                r"\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*>"
            )
            m = re.search(angle_pat, response)
            if m:
                box_3d = [float(m.group(i)) for i in range(1, 10)]

        # Try JSON format
        if box_3d is None:
            match = re.search(r"```(?:json)?\s*(.*?)```", response, re.DOTALL)
            json_str = match.group(1).strip() if match else response.strip()

            data = None
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                json_match = re.search(r"\[.*\]", response, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass

            if data is not None:
                if isinstance(data, list):
                    if len(data) == 0:
                        data = None
                    elif isinstance(data[0], dict):
                        data = data[0]
                    elif len(data) == 9 and all(isinstance(x, (int, float)) for x in data):
                        box_3d = [float(x) for x in data]
                        data = None
                    else:
                        data = None

            if data is not None and isinstance(data, dict):
                box_3d = data.get("box_3d") or data.get("bbox_3d")
                label = data.get("label", "")
                if box_3d is not None and len(box_3d) != 9:
                    box_3d = None

        if box_3d is None or len(box_3d) != 9:
            return None

        cx, cy, cz = float(box_3d[0]), float(box_3d[1]), float(box_3d[2])
        x_size, y_size, z_size = float(box_3d[3]), float(box_3d[4]), float(box_3d[5])
        pitch, yaw, roll = float(box_3d[6]), float(box_3d[7]), float(box_3d[8])
        R = _euler_to_rotation_matrix(pitch, yaw, roll)

        return {
            "center_cam": [cx, cy, cz],
            "dims_wlh": [x_size, y_size, z_size],
            "euler": [pitch, yaw, roll],
            "R_cam": R.tolist(),
            "label": label,
        }

    async def get_matching_score(self, data_id: Union[int, str], prediction: Any) -> Any:
        pred_data = json.loads(prediction)
        parsed = pred_data["parsed"]
        if parsed is None:
            return 0.0

        gt = self._gt_by_ann_id[data_id]
        corners_pred = np.array(parsed["corners"])
        corners_gt = np.array(gt["corners"])
        iou = _box3d_iou(corners_pred, corners_gt)
        return iou

    @staticmethod
    def _corners_for_vis(center, dims_whl, R_or_euler, angles_normalized=False):
        """Compute 8 corners using dims=[x_size, y_size, z_size] and rotation.

        Args:
            center: [cx, cy, cz] in camera coords
            dims_whl: [x_size, y_size, z_size]
            R_or_euler: either a 3x3 rotation matrix or [pitch, yaw, roll] euler angles
            angles_normalized: if True, euler angles are in [-1, 1] range (model output);
                             if False, they are in radians (GT annotations)
        """
        import math
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        x_size, y_size, z_size = float(dims_whl[0]), float(dims_whl[1]), float(dims_whl[2])
        hx, hy, hz = x_size / 2, y_size / 2, z_size / 2

        local_corners = [
            [-hx, -hy, -hz], [+hx, -hy, -hz],
            [+hx, -hy, +hz], [-hx, -hy, +hz],
            [-hx, +hy, -hz], [+hx, +hy, -hz],
            [+hx, +hy, +hz], [-hx, +hy, +hz],
        ]

        is_euler = (isinstance(R_or_euler, (list, tuple)) and len(R_or_euler) == 3
                    and not isinstance(R_or_euler[0], (list, tuple, np.ndarray)))
        if is_euler:
            pitch, yaw, roll = float(R_or_euler[0]), float(R_or_euler[1]), float(R_or_euler[2])
            if angles_normalized:
                p = np.deg2rad(pitch * 180)
                y_ = np.deg2rad(yaw * 180)
                r = np.deg2rad(roll * 180)
            else:
                p, y_, r = pitch, yaw, roll
        else:
            # R_or_euler is a 3x3 rotation matrix — apply directly
            R_mat = np.array(R_or_euler, dtype=np.float64)
            local_arr = np.array(local_corners)
            corners = (R_mat @ local_arr.T).T + np.array([cx, cy, cz])
            return corners

        def rotate_xyz(pt, _p, _y, _r):
            x0, y0, z0 = pt
            x1 = x0
            y1 = y0 * math.cos(_p) - z0 * math.sin(_p)
            z1 = y0 * math.sin(_p) + z0 * math.cos(_p)
            x2 = x1 * math.cos(_y) + z1 * math.sin(_y)
            y2 = y1
            z2 = -x1 * math.sin(_y) + z1 * math.cos(_y)
            x3 = x2 * math.cos(_r) - y2 * math.sin(_r)
            y3 = x2 * math.sin(_r) + y2 * math.cos(_r)
            z3 = z2
            return [x3, y3, z3]

        corners = []
        for corner in local_corners:
            rotated = rotate_xyz(corner, p, y_, r)
            corners.append([rotated[0] + cx, rotated[1] + cy, rotated[2] + cz])

        return np.array(corners)

    def _visualize_result(self, ann_id, parsed, gt, iou, vis_dir):
        """Combined visualization: image projection + 3D bbox + BEV + front + side.

        Layout (2 rows x 3 cols):
          Row 1: Image with projected boxes | 3D perspective | BEV (filled polygons)
          Row 2: BEV (outline) | Front (XY) | Side (ZY)
        """
        import cv2
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.gridspec import GridSpec
        from scipy.spatial import ConvexHull

        meta = self.data_dict[ann_id]
        img_path = meta["images"][0]
        img = cv2.imread(img_path)
        if img is None:
            return

        K_raw = meta.get("K")
        if K_raw is None:
            return
        K = np.array(K_raw, dtype=np.float64)

        cat_name = gt["category_name"]

        # GT corners
        corners_gt = self._corners_for_vis(gt["center_cam"], gt["dimensions"], gt["R_cam"])
        pts_gt = _project_corners_to_2d(corners_gt, K)

        # Prediction corners
        pred_corners = None
        pts_pred = None
        if parsed is not None:
            euler = parsed.get("euler", None)
            if euler is not None:
                pred_corners = self._corners_for_vis(
                    parsed["center_cam"], parsed["dimensions"],
                    euler, angles_normalized=True)
            else:
                pred_corners = self._corners_for_vis(
                    parsed["center_cam"], parsed["dimensions"], parsed["R_cam"])
            pts_pred = _project_corners_to_2d(pred_corners, K)

        vis_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        # --- Draw boxes on image ---
        _draw_box_edges(img, pts_gt, color=(0, 255, 0), thickness=2, edges=vis_edges)
        if pts_pred is not None:
            _draw_box_edges(img, pts_pred, color=(0, 0, 255), thickness=2, edges=vis_edges)

        label = f"{cat_name} IoU={iou:.3f}"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        cv2.putText(img, "green=GT  blue=Pred", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        gt_corners_list = [corners_gt]
        pred_corners_list = [pred_corners] if pred_corners is not None else []
        n_pred = len(pred_corners_list)

        # --- Create 2x3 figure ---
        fig = plt.figure(figsize=(20, 12), dpi=120)
        gs = GridSpec(2, 3, figure=fig)
        fig.suptitle(
            f"{ann_id}  {cat_name}  IoU={iou:.3f}",
            fontsize=12,
        )

        # Helper for drawing convex hull polygons
        def _draw_box_polygon_2d(ax, corners_8x3, axis0, axis1, color, linewidth=1.5,
                                 label=None):
            pts = corners_8x3[:, [axis0, axis1]]
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])
                ax.plot(hull_pts[:, 0], hull_pts[:, 1], color=color,
                        linewidth=linewidth, label=label)
            except Exception:
                pass

        # (0,0): Image with projected boxes
        ax_img = fig.add_subplot(gs[0, 0])
        ax_img.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax_img.set_title("Image Projection", fontsize=9)
        ax_img.axis("off")

        # (0,1): 3D perspective
        ax3d = fig.add_subplot(gs[0, 1], projection="3d")
        for corners in gt_corners_list:
            for i, j in vis_edges:
                ax3d.plot3D(
                    [corners[i, 0], corners[j, 0]],
                    [corners[i, 2], corners[j, 2]],
                    [corners[i, 1], corners[j, 1]],
                    color="green", linewidth=1.2,
                )
        for corners in pred_corners_list:
            for i, j in vis_edges:
                ax3d.plot3D(
                    [corners[i, 0], corners[j, 0]],
                    [corners[i, 2], corners[j, 2]],
                    [corners[i, 1], corners[j, 1]],
                    color="red", linewidth=1.0,
                )
        ax3d.set_xlabel("X", fontsize=7)
        ax3d.set_ylabel("Z", fontsize=7)
        ax3d.set_zlabel("Y", fontsize=7)
        ax3d.tick_params(labelsize=5)
        ax3d.set_title("3D BBox", fontsize=9)

        # (0,2): BEV with filled polygons (all 8 corners projected to XZ)
        ax_bev_fill = fig.add_subplot(gs[0, 2])
        for corners in gt_corners_list:
            pts_xz = corners[:, [0, 2]]
            try:
                hull = ConvexHull(pts_xz)
                pts = pts_xz[hull.vertices]
                poly = MplPolygon(pts, closed=True, edgecolor="green",
                                  facecolor=(0, 1, 0, 0.15), linewidth=1.5)
                ax_bev_fill.add_patch(poly)
            except Exception:
                pass
        for corners in pred_corners_list:
            pts_xz = corners[:, [0, 2]]
            try:
                hull = ConvexHull(pts_xz)
                pts = pts_xz[hull.vertices]
                poly = MplPolygon(pts, closed=True, edgecolor="red",
                                  facecolor=(1, 0, 0, 0.15), linewidth=1.5)
                ax_bev_fill.add_patch(poly)
            except Exception:
                pass
        ax_bev_fill.set_xlabel("X [m]", fontsize=7)
        ax_bev_fill.set_ylabel("Z [m]", fontsize=7)
        ax_bev_fill.set_title("BEV (filled)", fontsize=9)
        ax_bev_fill.set_aspect("equal")
        ax_bev_fill.grid(True, alpha=0.3)
        ax_bev_fill.autoscale()
        ax_bev_fill.tick_params(labelsize=6)

        # (1,0): BEV outline
        ax_bev = fig.add_subplot(gs[1, 0])
        for i, corners in enumerate(gt_corners_list):
            _draw_box_polygon_2d(ax_bev, corners, 0, 2, color="green",
                                 label="GT" if i == 0 else None)
        for i, corners in enumerate(pred_corners_list):
            _draw_box_polygon_2d(ax_bev, corners, 0, 2, color="red",
                                 label="Pred" if i == 0 else None)
        ax_bev.set_xlabel("X [m]", fontsize=7)
        ax_bev.set_ylabel("Z [m]", fontsize=7)
        ax_bev.set_title("BEV (X-Z)", fontsize=9)
        ax_bev.set_aspect("equal")
        ax_bev.grid(True, alpha=0.3)
        ax_bev.legend(fontsize=7)

        # (1,1): Front view (X-Y)
        ax_front = fig.add_subplot(gs[1, 1])
        for i, corners in enumerate(gt_corners_list):
            _draw_box_polygon_2d(ax_front, corners, 0, 1, color="green",
                                 label="GT" if i == 0 else None)
        for i, corners in enumerate(pred_corners_list):
            _draw_box_polygon_2d(ax_front, corners, 0, 1, color="red",
                                 label="Pred" if i == 0 else None)
        ax_front.set_xlabel("X [m]", fontsize=7)
        ax_front.set_ylabel("Y [m]", fontsize=7)
        ax_front.set_title("Front (X-Y)", fontsize=9)
        ax_front.set_aspect("equal")
        ax_front.invert_yaxis()
        ax_front.grid(True, alpha=0.3)
        ax_front.legend(fontsize=7)

        # (1,2): Side view (Z-Y)
        ax_side = fig.add_subplot(gs[1, 2])
        for i, corners in enumerate(gt_corners_list):
            _draw_box_polygon_2d(ax_side, corners, 2, 1, color="green",
                                 label="GT" if i == 0 else None)
        for i, corners in enumerate(pred_corners_list):
            _draw_box_polygon_2d(ax_side, corners, 2, 1, color="red",
                                 label="Pred" if i == 0 else None)
        ax_side.set_xlabel("Z [m]", fontsize=7)
        ax_side.set_ylabel("Y [m]", fontsize=7)
        ax_side.set_title("Side (Z-Y)", fontsize=9)
        ax_side.set_aspect("equal")
        ax_side.invert_yaxis()
        ax_side.grid(True, alpha=0.3)
        ax_side.legend(fontsize=7)

        plt.tight_layout()
        os.makedirs(vis_dir, exist_ok=True)
        safe_cat_name = cat_name.replace(' ', '_').replace('/', '_')
        save_path = os.path.join(vis_dir, f"{ann_id}_{safe_cat_name}.jpg")
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        vis_dir = os.environ.get("DETECT3D_VIS_DIR", "")
        iou_thresholds = np.array(self.iou_thresholds)
        n_total = len(results)
        n_parsed = 0

        ious = []
        ate_list = []
        aoe_list = []
        aoe_sym_list = []
        aoe_canonical_list = []
        ase_list = []

        per_cat_ious: Dict[str, List[float]] = defaultdict(list)

        # For AP computation: pred_all[image_id] = [(class_id, corners, score), ...]
        #                      gt_all[image_id] = [(class_id, corners), ...]
        pred_all: Dict[int, List] = defaultdict(list)
        gt_all: Dict[int, List] = defaultdict(list)

        for result in results:
            ann_id = result["data_id"]
            pred_data = json.loads(result["prediction"])
            parsed = pred_data["parsed"]
            gt = self._gt_by_ann_id[ann_id]
            cat_name = gt["category_name"]
            image_id = gt["image_id"]
            category_id = gt["category_id"]

            # Collect GT boxes (grouped by image_id)
            gt_all[image_id].append((category_id, np.array(gt["corners"])))

            if parsed is None:
                ious.append(0.0)
                per_cat_ious[cat_name].append(0.0)
                continue

            n_parsed += 1
            corners_pred = np.array(parsed["corners"])
            corners_gt = np.array(gt["corners"])
            iou = _box3d_iou(corners_pred, corners_gt)
            ious.append(iou)
            per_cat_ious[cat_name].append(iou)

            # Assign confidence score based on IoU for AP ranking
            # Use a high score for correct predictions, low for wrong ones
            score = 1.0 - (1.0 - iou) * 0.5  # maps IoU 0→0.5, IoU 1→1.0
            pred_all[image_id].append((category_id, corners_pred, score))

            pred_center = np.array(parsed["center_cam"])
            gt_center = np.array(gt["center_cam"])
            ate = np.linalg.norm(pred_center - gt_center)
            gt_radius = np.linalg.norm(np.array(gt["dimensions"])) / 2
            ate_list.append(ate / (gt_radius + 1e-6))

            try:
                R_pred = np.array(parsed["R_cam"])
                R_gt = np.array(gt["R_cam"])
                angle = _rotation_angle(R_pred, R_gt)
                aoe_list.append(angle / np.pi)
                aoe_sym_list.append(min(angle, np.pi - angle) / (np.pi / 2))
                R_pred_c = _canonicalize_rotation(R_pred, np.array(parsed["dimensions"]))
                R_gt_c = _canonicalize_rotation(R_gt, np.array(gt["dimensions"]))
                angle_c = _rotation_angle(R_pred_c, R_gt_c)
                aoe_canonical_list.append(angle_c / np.pi)
            except Exception:
                aoe_list.append(1.0)
                aoe_sym_list.append(1.0)
                aoe_canonical_list.append(1.0)

            ase = 1.0 - _scale_iou(np.array(parsed["dimensions"]), np.array(gt["dimensions"]))
            ase_list.append(ase)

        ious_arr = np.array(ious)
        metrics = {}

        # Acc@IoU thresholds
        for t in iou_thresholds:
            t_str = str(int(t * 100))
            metrics[f"Acc@{t_str}"] = float(np.mean(ious_arr >= t)) if n_total > 0 else 0.0

        metrics["mIoU"] = float(np.mean(ious_arr)) if n_total > 0 else 0.0
        metrics["parse_rate"] = n_parsed / n_total if n_total > 0 else 0.0

        metrics["ATE"] = float(np.mean(ate_list)) if ate_list else float("nan")
        metrics["AOE"] = float(np.mean(aoe_list)) if aoe_list else float("nan")
        metrics["AOE_Sym"] = float(np.mean(aoe_sym_list)) if aoe_sym_list else float("nan")
        metrics["AOE_Canonical"] = float(np.mean(aoe_canonical_list)) if aoe_canonical_list else float("nan")
        metrics["ASE"] = float(np.mean(ase_list)) if ase_list else float("nan")

        # Per-class Acc@25
        per_class = {}
        for cat_name, cat_ious in sorted(per_cat_ious.items()):
            cat_arr = np.array(cat_ious)
            per_class[cat_name] = float(np.mean(cat_arr >= 0.25))
        metrics["per_class_Acc@25"] = per_class

        # AP computation (VOC-style detection evaluation)
        ap_thresholds = [0.15, 0.25, 0.5]
        # Build category name mapping
        cat_id_to_name: Dict[int, str] = {v: k for k, v in self._cat_name_to_id.items()}

        for ovthresh in ap_thresholds:
            t_str = str(int(ovthresh * 100))

            # Split by class
            pred_cls: Dict[int, Dict[int, List]] = defaultdict(lambda: defaultdict(list))
            gt_cls: Dict[int, Dict[int, List]] = defaultdict(lambda: defaultdict(list))

            for img_id, preds in pred_all.items():
                for cls_id, corners, score in preds:
                    pred_cls[cls_id][img_id].append((corners, score))

            for img_id, gts in gt_all.items():
                for cls_id, corners in gts:
                    gt_cls[cls_id][img_id].append(corners)

            ap_dict = {}
            rec_dict = {}
            for cls_id in sorted(gt_cls.keys()):
                cls_pred = dict(pred_cls.get(cls_id, {}))
                cls_gt = dict(gt_cls[cls_id])
                rec, prec, ap = _eval_det_cls(cls_pred, cls_gt, ovthresh)
                cls_name = cat_id_to_name.get(cls_id, str(cls_id))
                ap_dict[cls_name] = ap
                if len(rec) > 0:
                    rec_dict[cls_name] = rec[-1]
                else:
                    rec_dict[cls_name] = 0.0

            ap_values = np.array(list(ap_dict.values()), dtype=np.float32)
            ap_values[np.isnan(ap_values)] = 0
            mAP = float(ap_values.mean()) if len(ap_values) > 0 else 0.0
            mAR = float(np.mean(list(rec_dict.values()))) if rec_dict else 0.0

            metrics[f"mAP@{t_str}"] = mAP
            metrics[f"AR@{t_str}"] = mAR
            for cls_name, ap_val in ap_dict.items():
                metrics[f"AP@{t_str}_{cls_name}"] = float(ap_val)
            for cls_name, rec_val in rec_dict.items():
                metrics[f"Rec@{t_str}_{cls_name}"] = float(rec_val)

        # Log
        log_lines = [
            f"\n===== Detect3D Evaluation Results =====",
            f"Samples: {n_total}, Parsed: {n_parsed} ({metrics['parse_rate']:.1%})",
            f"mIoU: {metrics['mIoU']:.4f}",
        ]
        for t in iou_thresholds:
            t_str = str(int(t * 100))
            log_lines.append(f"  Acc@{t_str}: {metrics[f'Acc@{t_str}']:.4f}")
        log_lines.append(f"ATE: {metrics['ATE']:.4f}")
        log_lines.append(f"AOE: {metrics['AOE']:.4f}")
        log_lines.append(f"AOE_Sym: {metrics['AOE_Sym']:.4f}")
        log_lines.append(f"AOE_Canonical: {metrics['AOE_Canonical']:.4f}")
        log_lines.append(f"ASE: {metrics['ASE']:.4f}")

        for ovthresh in ap_thresholds:
            t_str = str(int(ovthresh * 100))
            log_lines.append(f"\n--- AP @ IoU Threshold: {ovthresh} ---")
            log_lines.append(f"  mAP@{t_str}: {metrics[f'mAP@{t_str}']:.4f}")
            log_lines.append(f"  AR@{t_str}:  {metrics[f'AR@{t_str}']:.4f}")
            # Print per-class AP
            all_cats = set()
            for key in metrics:
                if key.startswith(f"AP@{t_str}_"):
                    all_cats.add(key.split("_", 1)[1])
            for cls_name in sorted(all_cats):
                ap_key = f"AP@{t_str}_{cls_name}"
                rec_key = f"Rec@{t_str}_{cls_name}"
                if ap_key in metrics:
                    log_lines.append(
                        f"    {cls_name:20s}  AP={metrics[ap_key]:.4f}  "
                        f"Rec={metrics[rec_key]:.4f}"
                    )
        print("\n".join(log_lines))

        if vis_dir:
            for result in tqdm(results, desc="Visualizing"):
                ann_id = result["data_id"]
                pred_data = json.loads(result["prediction"])
                parsed = pred_data["parsed"]
                gt = self._gt_by_ann_id[ann_id]
                iou = float(result.get("score", 0.0))
                self._visualize_result(ann_id, parsed, gt, iou, vis_dir)

        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Detect3DQwen — Qwen3.5 native 3D grounding format
# ---------------------------------------------------------------------------

DETECT3D_QWEN_PROMPT_TEMPLATE = (
    'Find all {cat} in this image. For each {cat}, provide its 3D bounding box. '
    'The camera intrinsics matrix is: {K}. '
    'The output format required is JSON: '
    '`[{{"bbox_3d":[x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw],"label":"category"}}]`.'
)


@BENCHMARK_REGISTRY.register()
class Detect3DQwen(Detect3D):
    """Detect3D variant using Qwen3.5 native bbox_3d prompt/output format.

    Output: bbox_3d = [cx, cy, cz, x_size, y_size, z_size, roll, pitch, yaw]
    Dims are [W, H, L] (x, y, z). Angles order is roll, pitch, yaw.
    """

    def generate_instruction(self, data_id: Union[int, str]) -> List[Dict[str, Any]]:
        meta = self.data_dict[data_id]

        K_str = self._format_K(meta["K"]) if meta["K"] else "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"
        prompt = DETECT3D_QWEN_PROMPT_TEMPLATE.format(
            cat=meta["category_name"],
            K=K_str,
        )

        if self.enable_thinking:
            prompt = "Think step by step about the object's position and orientation in 3D space.\n" + prompt

        contents = [
            {"type": "image", "image": meta["images"][0]},
            {"type": "text", "text": prompt},
        ]
        return [{"role": "user", "content": contents}]

    def _parse_3d_grounding(self, response: str) -> Optional[Dict]:
        box_3d = None
        label = ""

        match = re.search(r"```(?:json)?\s*(.*?)```", response, re.DOTALL)
        json_str = match.group(1).strip() if match else response.strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    data = None
            else:
                data = None

        if data is not None:
            if isinstance(data, list):
                if len(data) == 0:
                    data = None
                elif isinstance(data[0], dict):
                    data = data[0]
                elif len(data) == 9 and all(isinstance(x, (int, float)) for x in data):
                    box_3d = [float(x) for x in data]
                    data = None
                else:
                    data = None

        if data is not None and isinstance(data, dict):
            box_3d = data.get("bbox_3d") or data.get("box_3d")
            label = data.get("label", "")
            if box_3d is not None and len(box_3d) != 9:
                box_3d = None

        if box_3d is None:
            angle_pat = (
                r"<+\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,"
                r"\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,"
                r"\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*>"
            )
            m = re.search(angle_pat, response)
            if m:
                box_3d = [float(m.group(i)) for i in range(1, 10)]

        if box_3d is None or len(box_3d) != 9:
            return None

        cx, cy, cz = float(box_3d[0]), float(box_3d[1]), float(box_3d[2])
        x_size, y_size, z_size = float(box_3d[3]), float(box_3d[4]), float(box_3d[5])
        pitch, yaw, roll = float(box_3d[6]), float(box_3d[7]), float(box_3d[8])

        R = _euler_to_rotation_matrix(pitch, yaw, roll)

        return {
            "center_cam": [cx, cy, cz],
            "dims_wlh": [x_size, y_size, z_size],
            "euler": [pitch, yaw, roll],
            "R_cam": R.tolist(),
            "label": label,
        }

    async def process_response(self, data_id: Union[int, str], response: str) -> Any:
        if self.enable_thinking and "</think>" in response:
            response = response.split("</think>")[-1]

        parsed = self._parse_3d_grounding(response)

        if parsed is not None:
            center = np.array(parsed["center_cam"], dtype=np.float64)
            # dims_wlh = [x_size, y_size, z_size] = [W, H, L] already in correct order
            dims_whl = np.array(parsed["dims_wlh"], dtype=np.float64)
            R = np.array(parsed["R_cam"], dtype=np.float64)
            corners = _box3d_corners(center, dims_whl, R)
            parsed["corners"] = corners.tolist()
            parsed["dimensions"] = dims_whl.tolist()

        meta = self.data_dict[data_id]
        return json.dumps({
            "ann_id": data_id,
            "parsed": parsed,
        })


# ---------------------------------------------------------------------------
# Detect3DTrain — same as Detect3DQwen but using training data
# ---------------------------------------------------------------------------

_DETECT3D_TRAIN_ANN_NAME = "InTheWild_v3_train_human.json"


@BENCHMARK_REGISTRY.register()
class Detect3DTrain(Detect3DQwen):
    """Detect3DQwen variant using WildDet3D training split (human annotations).

    Expected data_root structure (can be built with symlinks):
        data_root/
            annotations/InTheWild_v3_train_human.json
            images/   -> symlink to actual image directory
    """

    def load_data(self, data_root: str) -> Dict[Union[int, str], Any]:
        ann_path = os.path.join(data_root, "annotations", _DETECT3D_TRAIN_ANN_NAME)
        # print("annotations file path is")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(
                f"Annotation file not found: {ann_path}\n"
                f"Please create symlinks so that data_root={data_root} contains:\n"
                f"  annotations/{_DETECT3D_TRAIN_ANN_NAME}\n"
                f"  images/ -> (actual image directory)"
            )
        with open(ann_path, "r") as f:
            ann_data = json.load(f)

        for cat in ann_data["categories"]:
            self._categories[cat["id"]] = cat["name"]
            self._cat_name_to_id[cat["name"]] = cat["id"]

        max_images = int(os.environ.get("DETECT3D_MAX_IMAGES", 0))
        images_list = ann_data["images"]
        if max_images > 0:
            images_list = images_list[:max_images]

        images_by_id = {}
        for img in images_list:
            images_by_id[img["id"]] = img

        image_dir = os.path.join(data_root, "images")

        data_dict = {}
        n_skipped_missing = 0
        for ann in ann_data["annotations"]:
            if not ann.get("valid3D", True):
                continue
            if ann["image_id"] not in images_by_id:
                continue

            img_info = images_by_id[ann["image_id"]]
            file_name = img_info.get("file_name", os.path.basename(img_info.get("file_path", "")))
            file_path = os.path.join(image_dir, file_name)
            if not os.path.exists(file_path):
                if n_skipped_missing < 10:
                    print(f"[Detect3DTrain] Skipping missing image: {file_path}")
                n_skipped_missing += 1
                continue

            center = np.array(ann["center_cam"], dtype=np.float64)
            dims_raw = np.array(ann["dimensions"], dtype=np.float64)
            dims = np.array([dims_raw[2], dims_raw[1], dims_raw[0]], dtype=np.float64)
            R = np.array(ann["R_cam"], dtype=np.float64)

            if "bbox3D_cam" in ann and ann["bbox3D_cam"] is not None:
                corners = np.array(ann["bbox3D_cam"], dtype=np.float64)
            else:
                corners = _box3d_corners(center, dims, R)

            ann_id = ann["id"]
            cat_name = ann.get("category_name", self._categories.get(ann["category_id"], ""))
            attr = _generate_attr(ann, img_info["width"], img_info["height"])

            gt_entry = {
                "ann_id": ann_id,
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "category_name": cat_name,
                "center_cam": ann["center_cam"],
                "dimensions": dims.tolist(),
                "R_cam": ann["R_cam"],
                "corners": corners,
                "depth": center[2],
            }
            self._gt_by_ann_id[ann_id] = gt_entry

            data_dict[ann_id] = {
                "images": [file_path],
                "image_id": ann["image_id"],
                "ann_id": ann_id,
                "width": img_info["width"],
                "height": img_info["height"],
                "K": img_info.get("K", None),
                "category_name": cat_name,
                "attr": attr,
                "ground_truth": gt_entry,
                "task_type": "detect3d",
            }

        print(f"[Detect3DTrain] Loaded {len(data_dict)} samples "
              f"({len(images_by_id)} images, {len(self._categories)} categories)"
              f"{f', skipped {n_skipped_missing} (image not found)' if n_skipped_missing else ''}")
        return data_dict
