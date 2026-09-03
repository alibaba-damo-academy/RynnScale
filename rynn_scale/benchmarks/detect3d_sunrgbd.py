import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm

import numpy as np

from ..inference_wrappers import BaseInferenceWrapper
from ..registry import BENCHMARK_REGISTRY
from .base import BaseBenchmark

# ---------------------------------------------------------------------------
# SUN RGB-D constants
# ---------------------------------------------------------------------------

SUNRGBD_TYPE2CLASS = {
    "bed": 0,
    "table": 1,
    "sofa": 2,
    "chair": 3,
    "toilet": 4,
    "desk": 5,
    "dresser": 6,
    "night_stand": 7,
    "bookshelf": 8,
    "bathtub": 9,
}
SUNRGBD_CLASS2TYPE = {v: k for k, v in SUNRGBD_TYPE2CLASS.items()}

# ---------------------------------------------------------------------------
# 3D geometry utilities
# ---------------------------------------------------------------------------


def _roty(t: float) -> np.ndarray:
    """Rotation about the Y-axis (camera coords)."""
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rotz(t: float) -> np.ndarray:
    """Rotation about the Z-axis (upright depth coords)."""
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _flip_axis_to_camera(pc: np.ndarray) -> np.ndarray:
    """Upright depth (X-right, Y-forward, Z-up) -> Camera (X-right, Y-down, Z-forward)."""
    out = pc.copy()
    out[..., [0, 1, 2]] = out[..., [0, 2, 1]]
    out[..., 1] *= -1
    return out


def _get_3d_box(box_size_lwh: np.ndarray, heading_angle: float,
                center: np.ndarray) -> np.ndarray:
    """Compute 8 corners of a 3D box in camera coordinates.

    This matches 3DETR's get_3d_box: Y-axis is vertical (camera convention).
    box_size = (l, w, h) as half-sizes: l=X-extent, w=Z-extent, h=Y-extent.
    Returns (8, 3) corners.
    """
    R = _roty(heading_angle)
    l, w, h = box_size_lwh
    x_corners = [l, l, -l, -l, l, l, -l, -l]
    y_corners = [h, h, h, h, -h, -h, -h, -h]
    z_corners = [w, -w, -w, w, w, -w, -w, w]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]
    return corners_3d.T


def _get_3d_box_upright_depth(size_lwh: np.ndarray, R: np.ndarray,
                              center: np.ndarray) -> np.ndarray:
    """Compute 8 corners in upright depth coords (X-right, Y-forward, Z-up).

    size_lwh = (l, w, h) as half-sizes.
    R = 3x3 rotation matrix (rotz).
    Returns (8, 3) corners matching README ordering.
    """
    l, w, h = size_lwh
    x = np.array([-l, l, l, -l, -l, l, l, -l])
    y = np.array([w, w, -w, -w, w, w, -w, -w])
    z = np.array([h, h, h, h, -h, -h, -h, -h])
    local_corners = np.vstack([x, y, z])  # (3, 8)
    world_corners = R @ local_corners + center.reshape(3, 1)
    return world_corners.T  # (8, 3)



def _ud_corners_to_camera(corners_ud: np.ndarray, Rtilt: np.ndarray) -> np.ndarray:
    """Transform 8x3 corners from upright depth to actual camera coords.

    Pipeline: Rtilt.T (undo tilt) -> axis flip (to camera).
    """
    pc = (Rtilt.T @ corners_ud.T).T
    cam = pc.copy()
    cam[:, [0, 1, 2]] = cam[:, [0, 2, 1]]
    cam[:, 1] *= -1
    return cam

def _project_corners_ud_to_2d(corners_ud: np.ndarray, K: np.ndarray,
                               Rtilt: np.ndarray) -> List:
    """Project 8x3 corners from upright depth coords to 2D pixels.

    Pipeline: undo Rtilt -> axis flip to camera -> project with K.
    """
    pc = (Rtilt.T @ corners_ud.T).T
    cam = pc.copy()
    cam[:, [0, 1, 2]] = cam[:, [0, 2, 1]]
    cam[:, 1] *= -1
    pts_2d = []
    for pt in cam:
        if pt[2] <= 0:
            pts_2d.append(None)
            continue
        px = K[0, 0] * (pt[0] / pt[2]) + K[0, 2]
        py = K[1, 1] * (pt[1] / pt[2]) + K[1, 2]
        pts_2d.append((int(round(px)), int(round(py))))
    return pts_2d


def _box3d_iou(corners1: np.ndarray, corners2: np.ndarray) -> float:
    """Compute 3D IoU between two sets of 8 corners (camera convention: Y-down).

    Uses BEV (X-Z plane) polygon intersection × Y overlap.
    Handles slightly tilted boxes by using convex hull BEV and robust Y bounds.
    """
    from scipy.spatial import ConvexHull

    def _polygon_clip(subjectPolygon, clipPolygon):
        def inside(p):
            return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) >= (cp2[1] - cp1[1]) * (p[0] - cp1[0])
        def computeIntersection():
            dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
            dp = [s[0] - e[0], s[1] - e[1]]
            n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
            n2 = s[0] * e[1] - s[1] * e[0]
            n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
            return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

        outputList = list(subjectPolygon)
        cp1 = clipPolygon[-1]
        for clipVertex in clipPolygon:
            cp2 = clipVertex
            inputList = outputList
            outputList = []
            if len(inputList) == 0:
                return None
            s = inputList[-1]
            for subjectVertex in inputList:
                e = subjectVertex
                if inside(e):
                    if not inside(s):
                        outputList.append(computeIntersection())
                    outputList.append(e)
                elif inside(s):
                    outputList.append(computeIntersection())
                s = e
            cp1 = cp2
            if len(outputList) == 0:
                return None
        return outputList

    # BEV: project all 8 corners to X-Z plane, use ConvexHull for proper polygon
    pts1_xz = corners1[:, [0, 2]]
    pts2_xz = corners2[:, [0, 2]]
    try:
        hull1 = ConvexHull(pts1_xz)
        rect1 = [tuple(pts1_xz[i]) for i in hull1.vertices]
        area1 = hull1.volume
    except Exception:
        return 0.0
    try:
        hull2 = ConvexHull(pts2_xz)
        rect2 = [tuple(pts2_xz[i]) for i in hull2.vertices]
        area2 = hull2.volume
    except Exception:
        return 0.0

    inter_p = _polygon_clip(rect1, rect2)
    if inter_p is not None and len(inter_p) >= 3:
        try:
            inter_area = ConvexHull(inter_p).volume
        except Exception:
            inter_area = 0.0
    else:
        inter_area = 0.0

    # Y overlap: use max/min of all 8 corners (no face ordering dependency)
    y1_max = corners1[:, 1].max()
    y1_min = corners1[:, 1].min()
    y2_max = corners2[:, 1].max()
    y2_min = corners2[:, 1].min()
    ymax = min(y1_max, y2_max)
    ymin = max(y1_min, y2_min)
    inter_vol = inter_area * max(0.0, ymax - ymin)

    vol1 = area1 * (y1_max - y1_min)
    vol2 = area2 * (y2_max - y2_min)
    union = vol1 + vol2 - inter_vol
    if union < 1e-10:
        return 0.0
    return inter_vol / union


def _euler_to_rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Euler angles (normalized [-1,1]) → rotation matrix R = Rz @ Ry @ Rx.

    pitch, yaw, roll are in normalized range [-1, 1] mapping to [-180°, 180°].
    """
    p = np.deg2rad(pitch * 180)
    y = np.deg2rad(yaw * 180)
    r = np.deg2rad(roll * 180)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    cr, sr = np.cos(r), np.sin(r)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotation_matrix_to_euler(R: np.ndarray) -> tuple:
    """Rotation matrix → normalized euler angles (pitch, yaw, roll) in [-1, 1].

    Decomposition: R = Rz(roll) @ Ry(yaw) @ Rx(pitch).
    Returns (pitch, yaw, roll) each in [-1, 1] mapping to [-180°, 180°].
    """
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    yaw = np.arcsin(sy)

    if np.abs(sy) < 0.99999:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        roll = 0.0

    return (pitch / np.pi, yaw / np.pi, roll / np.pi)


def _box3d_corners_from_params(center, dims_xyz, R):
    """Build 8 corners from center, dims=[x_size, y_size, z_size], and rotation matrix.

    Returns 8 corners in construction order (no sorting).
    """
    w, h, l = dims_xyz[0], dims_xyz[1], dims_xyz[2]
    x = np.array([w / 2, 0, 0])
    y = np.array([0, h / 2, 0])
    z = np.array([0, 0, l / 2])
    corners = np.array([
        -x - y - z, -x - y + z, -x + y - z, -x + y + z,
        +x - y - z, +x - y + z, +x + y - z, +x + y + z,
    ])
    corners = (R @ corners.T).T + np.array(center)
    return corners


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
# Prompt template (multi-object variant)
# ---------------------------------------------------------------------------

SUNRGBD_PROMPT_TEMPLATE = """Find all {cat} in this image.

The camera intrinsics matrix is:
{K}

Predict 3D bounding boxes in the camera coordinate system, where:
- x points to the right
- y points downward
- z points forward

For each object, return:
<3D Grounding> cx, cy, cz, x_size, y_size, z_size, pitch, yaw, roll </3D Grounding>

Definitions:
- cx, cy, cz: 3D coordinates of the box center in the camera coordinate system, in meters
- x_size, y_size, z_size: box dimensions in the box local coordinate system, in meters
  - x_size: extent along the local x-axis
  - y_size: extent along the local y-axis
  - z_size: extent along the local z-axis
- pitch: rotation around the camera x-axis, in radians (range: -1 to 1, corresponding to -180 to 180 degrees)
- yaw: rotation around the camera y-axis, in radians (range: -1 to 1, corresponding to -180 to 180 degrees)
- roll: rotation around the camera z-axis, in radians (range: -1 to 1, corresponding to -180 to 180 degrees)

Constraints:
- x_size >= z_size
- Use meters for cx, cy, cz, x_size, y_size, z_size
- Use radians for pitch, yaw, roll, with each value in the range [-1, 1]
<think>\n\n</think>\n\n
"""


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

# 3DETR corner order edges (for _get_3d_box output in camera coords)
_3DETR_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

# Upright depth corner order edges (from _get_3d_box_upright_depth / README)
# Corner order: 0-3 top face (z=+h), 4-7 bottom face (z=-h)
#   1----0 (top)     5----4 (bottom)
#   |    |           |    |
#   2----3           6----7
_UD_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

# _box3d_corners_from_params corner order edges (for predictions)
_PRED_BOX_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _project_corners_to_2d(corners: np.ndarray, K: np.ndarray) -> List:
    """Project 8x3 corners to 2D pixel coordinates using intrinsics K (3x3)."""
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
    for i, j in (edges or _3DETR_BOX_EDGES):
        if i < len(pts_2d) and j < len(pts_2d):
            if pts_2d[i] is not None and pts_2d[j] is not None:
                cv2.line(img, pts_2d[i], pts_2d[j], color, thickness)


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------


@BENCHMARK_REGISTRY.register()
class Detect3DSUNRGBD(BaseBenchmark):
    def __init__(
        self,
        data_root: str,
        inference_wrapper: BaseInferenceWrapper,
        prompt_format: Optional[str] = None,
        enable_thinking: bool = False,
        parse_json: bool = False,
    ) -> None:
        self._gt_by_data_id: Dict[str, Dict] = {}
        self.parse_json = parse_json
        super().__init__(
            data_root=data_root,
            inference_wrapper=inference_wrapper,
            prompt_format=prompt_format,
            enable_thinking=enable_thinking,
        )

    def load_data(self, data_root: str) -> Dict[Union[int, str], Any]:
        import scipy.io as sio

        meta_path = os.path.join(data_root, "SUNRGBDMeta3DBB_v2.mat")
        meta_data = sio.loadmat(meta_path)
        meta = meta_data["SUNRGBDMeta"][0]

        split_path = os.path.join(
            data_root, "SUNRGBDtoolbox", "traintestSUNRGBD", "allsplit.mat"
        )
        split_data = sio.loadmat(split_path)

        test_paths = set()
        for i in range(split_data["alltest"].shape[1]):
            p = split_data["alltest"][0, i][0]
            rel = p.replace("/n/fs/sun3d/data/", "").rstrip("/")
            test_paths.add(rel)

        max_images = int(os.environ.get("DETECT3D_MAX_IMAGES", 0))
        image_count = 0

        # Group GT by (image_idx, category) -> list of boxes
        # Each unique (image, category) pair = one data entry
        data_dict = {}

        for i in range(len(meta)):
            sample = meta[i]
            seq_name = str(sample["sequenceName"][0]).rstrip("/")
            if seq_name not in test_paths:
                continue

            img_dir = os.path.join(data_root, seq_name, "image")
            if not os.path.exists(img_dir):
                continue
            img_files = sorted(os.listdir(img_dir))
            if not img_files:
                continue
            image_path = os.path.join(img_dir, img_files[0])

            if max_images > 0 and image_count >= max_images:
                break
            image_count += 1

            K = sample["K"].astype(np.float64)
            Rtilt = sample["Rtilt"].astype(np.float64)
            gt_bbs = sample["groundtruth3DBB"]

            if gt_bbs.size == 0:
                continue

            # Group boxes by category for this image
            cat_boxes: Dict[str, List[np.ndarray]] = defaultdict(list)
            cat_boxes_ud: Dict[str, List[np.ndarray]] = defaultdict(list)
            cat_gt_groundings: Dict[str, List[str]] = defaultdict(list)

            # Transform matrix: UD -> actual camera
            # AxisFlip @ Rtilt.T where AxisFlip = [[1,0,0],[0,0,-1],[0,1,0]]
            axis_flip = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
            T_ud_to_cam = axis_flip @ Rtilt.T

            for j in range(gt_bbs.shape[1]):
                bb = gt_bbs[0, j]
                classname = str(bb["classname"][0])
                if classname not in SUNRGBD_TYPE2CLASS:
                    continue

                centroid = bb["centroid"].flatten().astype(np.float64)
                coeffs = bb["coeffs"].flatten().astype(np.float64)
                orientation = bb["orientation"].flatten().astype(np.float64)

                # Correct heading and rotation (from preprocess_sunrgbd_rotmat_full.py)
                heading_angle = -1.0 * np.arctan2(orientation[1], orientation[0])
                R_ud = _rotz(-1.0 * heading_angle)

                # Fixed size mapping: l=coeffs[1], w=coeffs[0], h=coeffs[2]
                l_half = coeffs[1]
                w_half = coeffs[0]
                h_half = coeffs[2]

                # Corners in upright depth (for visualization projection)
                corners_ud = _get_3d_box_upright_depth(
                    np.array([l_half, w_half, h_half]), R_ud, centroid
                )
                cat_boxes_ud[classname].append(corners_ud)

                # Corners in actual camera coords (for IoU with model predictions)
                corners_cam = _ud_corners_to_camera(corners_ud, Rtilt)
                cat_boxes[classname].append(corners_cam)

                # GT in <3D Grounding> format (camera coords)
                center_cam = T_ud_to_cam @ centroid
                R_cam = T_ud_to_cam @ R_ud
                x_size, y_size, z_size = 2 * l_half, 2 * w_half, 2 * h_half
                pitch, yaw, roll = _rotation_matrix_to_euler(R_cam)
                gt_str = (
                    f"<3D Grounding> {center_cam[0]:.4f}, {center_cam[1]:.4f}, "
                    f"{center_cam[2]:.4f}, {x_size:.4f}, {y_size:.4f}, "
                    f"{z_size:.4f}, {pitch:.4f}, {yaw:.4f}, {roll:.4f} "
                    f"</3D Grounding>"
                )
                cat_gt_groundings[classname].append(gt_str)

            for cat_name, boxes_list in cat_boxes.items():
                data_id = f"{i}_{cat_name}"
                class_id = SUNRGBD_TYPE2CLASS[cat_name]

                gt_entry = {
                    "image_id": i,
                    "category_name": cat_name,
                    "class_id": class_id,
                    "gt_corners": [b.tolist() for b in boxes_list],
                    "gt_corners_ud": [b.tolist() for b in cat_boxes_ud[cat_name]],
                    "gt_groundings": cat_gt_groundings[cat_name],
                    "n_objects": len(boxes_list),
                }
                self._gt_by_data_id[data_id] = gt_entry

                data_dict[data_id] = {
                    "images": [image_path],
                    "image_id": i,
                    "K": K.tolist(),
                    "Rtilt": Rtilt.tolist(),
                    "category_name": cat_name,
                    "class_id": class_id,
                    "task_type": "detect3d_sunrgbd",
                }

        n_images = len(set(d["image_id"] for d in data_dict.values()))
        n_queries = len(data_dict)
        n_gt_boxes = sum(gt["n_objects"] for gt in self._gt_by_data_id.values())
        print(
            f"[Detect3DSUNRGBD] Loaded {n_queries} queries "
            f"({n_images} images, {n_gt_boxes} GT boxes, "
            f"{len(SUNRGBD_TYPE2CLASS)} categories)"
        )
        return data_dict

    @staticmethod
    def _format_K(K: List) -> str:
        rows = []
        for row in K:
            rows.append("[" + ", ".join(f"{v:.2f}" for v in row) + "]")
        return "[" + ", ".join(rows) + "]"

    def generate_instruction(self, data_id: Union[int, str]) -> List[Dict[str, Any]]:
        meta = self.data_dict[data_id]
        K_str = self._format_K(meta["K"])
        prompt = SUNRGBD_PROMPT_TEMPLATE.format(
            cat=meta["category_name"],
            K=K_str,
        )

        contents = [
            {"type": "image", "image": meta["images"][0]},
            {"type": "text", "text": prompt},
        ]
        return [{"role": "user", "content": contents}]

    def _parse_all_3d_groundings(self, response: str) -> List[Dict]:
        """Parse ALL 3D bounding boxes from model response."""
        results = []

        # Parse all <3D Grounding> tags
        for match in re.findall(
            r"<3D Grounding>\s*(.*?)\s*</3D Grounding>", response
        ):
            try:
                values = [float(v.strip()) for v in match.split(",")]
                if len(values) == 9:
                    results.append(values)
            except ValueError:
                continue

        # Fallback: <<<<values> pattern
        if not results:
            for m in re.finditer(
                r"<+\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,"
                r"\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,"
                r"\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*>",
                response,
            ):
                values = [float(m.group(i)) for i in range(1, 10)]
                results.append(values)

        # JSON fallback when parse_json=True
        if not results and self.parse_json:
            json_match = re.search(r"```(?:json)?\s*(.*?)```", response, re.DOTALL)
            json_str = json_match.group(1).strip() if json_match else response.strip()

            data = None
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                arr_match = re.search(r"\[.*\]", response, re.DOTALL)
                if arr_match:
                    try:
                        data = json.loads(arr_match.group(0))
                    except json.JSONDecodeError:
                        pass

            if data is not None and isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        box_3d = item.get("bbox_3d") or item.get("box_3d")
                        if box_3d is not None and len(box_3d) == 9:
                            results.append([float(x) for x in box_3d])

        parsed_boxes = []
        for idx, box_9 in enumerate(results):
            cx, cy, cz = box_9[0], box_9[1], box_9[2]
            x_size, y_size, z_size = box_9[3], box_9[4], box_9[5]
            pitch, yaw, roll = box_9[6], box_9[7], box_9[8]
            R = _euler_to_rotation_matrix(pitch, yaw, roll)
            center = np.array([cx, cy, cz])
            dims = np.array([x_size, y_size, z_size])
            corners = _box3d_corners_from_params(center, dims, R)
            # Assign descending confidence scores so first box = highest confidence
            score = 1.0 - idx * 0.01
            parsed_boxes.append({
                "corners": corners.tolist(),
                "center": [cx, cy, cz],
                "dims": [x_size, y_size, z_size],
                "euler": [pitch, yaw, roll],
                "score": max(score, 0.01),
            })

        return parsed_boxes

    async def process_response(self, data_id: Union[int, str], response: str) -> Any:
        if self.enable_thinking and "</think>" in response:
            response = response.split("</think>")[-1]

        parsed_boxes = self._parse_all_3d_groundings(response)

        gt = self._gt_by_data_id[data_id]
        return json.dumps({
            "data_id": data_id,
            "parsed_boxes": parsed_boxes,
            "n_parsed": len(parsed_boxes),
            "gt_groundings": gt["gt_groundings"],
        })

    async def get_matching_score(self, data_id: Union[int, str], prediction: Any) -> Any:
        pred_data = json.loads(prediction)
        parsed_boxes = pred_data["parsed_boxes"]
        if not parsed_boxes:
            return 0.0

        gt = self._gt_by_data_id[data_id]
        gt_corners_list = [np.array(c) for c in gt["gt_corners"]]

        best_iou = 0.0
        for pb in parsed_boxes:
            pred_corners = np.array(pb["corners"])
            for gt_corners in gt_corners_list:
                iou = _box3d_iou(pred_corners, gt_corners)
                best_iou = max(best_iou, iou)
        return best_iou

    @staticmethod
    def _draw_box_polygon_2d(ax, corners_8x3, axis0, axis1, color, linewidth=1.5,
                             label=None):
        """Draw a 3D box projected to a 2D plane as a convex hull polygon."""
        from scipy.spatial import ConvexHull
        pts = corners_8x3[:, [axis0, axis1]]
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])
            ax.plot(hull_pts[:, 0], hull_pts[:, 1], color=color,
                    linewidth=linewidth, label=label)
        except Exception:
            pass

    def _visualize_result(self, data_id, parsed_boxes, gt, best_iou, vis_dir):
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

        meta = self.data_dict[data_id]
        img_path = meta["images"][0]
        img = cv2.imread(img_path)
        if img is None:
            return

        K = np.array(meta["K"], dtype=np.float64)
        Rtilt = np.array(meta["Rtilt"], dtype=np.float64)
        cat_name = meta["category_name"]

        gt_corners_list = [np.array(c) for c in gt["gt_corners"]]
        pred_corners_list = [np.array(pb["corners"]) for pb in parsed_boxes]

        # --- Draw boxes on image ---
        for gt_corners_ud in gt["gt_corners_ud"]:
            corners_ud = np.array(gt_corners_ud)
            pts = _project_corners_ud_to_2d(corners_ud, K, Rtilt)
            _draw_box_edges(img, pts, color=(0, 255, 0), thickness=2,
                            edges=_UD_BOX_EDGES)
        for corners in pred_corners_list:
            pts = _project_corners_to_2d(corners, K)
            _draw_box_edges(img, pts, color=(0, 0, 255), thickness=2,
                            edges=_PRED_BOX_EDGES)

        label = f"{cat_name} IoU={best_iou:.3f} GT={gt['n_objects']} Pred={len(parsed_boxes)}"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        cv2.putText(img, "green=GT  red=Pred", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # --- Create 2x3 figure ---
        fig = plt.figure(figsize=(20, 12), dpi=120)
        gs = GridSpec(2, 3, figure=fig)
        fig.suptitle(
            f"{data_id}  {cat_name}  IoU={best_iou:.3f}  "
            f"GT={gt['n_objects']}  Pred={len(parsed_boxes)}",
            fontsize=12,
        )

        # (0,0): Image with projected boxes
        ax_img = fig.add_subplot(gs[0, 0])
        ax_img.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax_img.set_title("Image Projection", fontsize=9)
        ax_img.axis("off")

        # (0,1): 3D perspective
        ax3d = fig.add_subplot(gs[0, 1], projection="3d")
        for corners in gt_corners_list:
            for i, j in _UD_BOX_EDGES:
                ax3d.plot3D(
                    [corners[i, 0], corners[j, 0]],
                    [corners[i, 2], corners[j, 2]],
                    [corners[i, 1], corners[j, 1]],
                    color="green", linewidth=1.2,
                )
        for corners in pred_corners_list:
            for i, j in _PRED_BOX_EDGES:
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
            self._draw_box_polygon_2d(
                ax_bev, corners, 0, 2, color="green",
                label="GT" if i == 0 else None)
        for i, corners in enumerate(pred_corners_list):
            self._draw_box_polygon_2d(
                ax_bev, corners, 0, 2, color="red",
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
            self._draw_box_polygon_2d(
                ax_front, corners, 0, 1, color="green",
                label="GT" if i == 0 else None)
        for i, corners in enumerate(pred_corners_list):
            self._draw_box_polygon_2d(
                ax_front, corners, 0, 1, color="red",
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
            self._draw_box_polygon_2d(
                ax_side, corners, 2, 1, color="green",
                label="GT" if i == 0 else None)
        for i, corners in enumerate(pred_corners_list):
            self._draw_box_polygon_2d(
                ax_side, corners, 2, 1, color="red",
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
        fig.savefig(os.path.join(vis_dir, f"{data_id}.jpg"), dpi=120,
                    bbox_inches="tight")
        plt.close(fig)

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        vis_dir = os.environ.get("DETECT3D_VIS_DIR", "")

        # Collect predictions and GT per image
        # pred_all[image_id] = [(class_id, corners_8x3, score), ...]
        # gt_all[image_id] = [(class_id, corners_8x3), ...]
        pred_all: Dict[int, List] = defaultdict(list)
        gt_all: Dict[int, List] = defaultdict(list)
        gt_collected_images = set()

        n_total = len(results)
        n_parsed = 0
        n_total_pred_boxes = 0

        for result in results:
            data_id = result["data_id"]
            pred_data = json.loads(result["prediction"])
            gt = self._gt_by_data_id[data_id]
            image_id = gt["image_id"]
            class_id = gt["class_id"]

            # Add GT boxes (only once per data_id to avoid duplicates)
            gt_key = (image_id, class_id)
            if gt_key not in gt_collected_images:
                gt_collected_images.add(gt_key)
                for corners in gt["gt_corners"]:
                    gt_all[image_id].append((class_id, np.array(corners)))

            # Add predictions
            parsed_boxes = pred_data["parsed_boxes"]
            if parsed_boxes:
                n_parsed += 1
                for pb in parsed_boxes:
                    corners = np.array(pb["corners"])
                    score = pb["score"]
                    pred_all[image_id].append((class_id, corners, score))
                    n_total_pred_boxes += 1

            # # Visualize
            # if vis_dir:
            #     best_iou = float(result.get("score", 0.0))
            #     self._visualize_result(
            #         data_id, parsed_boxes, gt, best_iou, vis_dir
            #     )

        # Compute AP at different thresholds
        ap_thresholds = [0.15, 0.25, 0.5]
        metrics = {}
        metrics["parse_rate"] = n_parsed / n_total if n_total > 0 else 0.0
        metrics["n_queries"] = n_total
        metrics["n_parsed"] = n_parsed
        metrics["n_pred_boxes"] = n_total_pred_boxes
        metrics["n_gt_boxes"] = sum(len(v) for v in gt_all.values())

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
                cls_name = SUNRGBD_CLASS2TYPE.get(cls_id, str(cls_id))
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

        # Print summary
        log_lines = [
            "\n===== Detect3D SUN RGB-D Evaluation Results =====",
            f"Queries: {n_total}, Parsed: {n_parsed} ({metrics['parse_rate']:.1%})",
            f"Predicted boxes: {n_total_pred_boxes}, GT boxes: {metrics['n_gt_boxes']}",
        ]
        for ovthresh in ap_thresholds:
            t_str = str(int(ovthresh * 100))
            log_lines.append(f"\n--- IoU Threshold: {ovthresh} ---")
            log_lines.append(f"  mAP@{t_str}: {metrics[f'mAP@{t_str}']:.4f}")
            log_lines.append(f"  AR@{t_str}:  {metrics[f'AR@{t_str}']:.4f}")
            for cls_name in sorted(SUNRGBD_TYPE2CLASS.keys()):
                ap_key = f"AP@{t_str}_{cls_name}"
                rec_key = f"Rec@{t_str}_{cls_name}"
                if ap_key in metrics:
                    log_lines.append(
                        f"    {cls_name:12s}  AP={metrics[ap_key]:.4f}  "
                        f"Rec={metrics[rec_key]:.4f}"
                    )
        print("\n".join(log_lines))

        for result in tqdm(results):
            data_id = result["data_id"]
            pred_data = json.loads(result["prediction"])
            gt = self._gt_by_data_id[data_id]

            # Add predictions
            parsed_boxes = pred_data["parsed_boxes"]
            # Visualize
            if vis_dir:
                best_iou = float(result.get("score", 0.0))
                self._visualize_result(
                    data_id, parsed_boxes, gt, best_iou, vis_dir
                )

        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()}
