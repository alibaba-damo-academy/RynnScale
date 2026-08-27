"""Recompute AP using center-distance / GT-radius matching instead of volume IoU.

Usage:
    python recompute_ap_center_dist.py <results_json>

Matching criterion: distance(pred_center, gt_center) / gt_radius < threshold
    thresholds = [0.5, 0.55, 0.60, ..., 1.0]
Reports per-threshold AP and the mean AP across all thresholds.
Uses COCO-style 101-point interpolation and excludes empty classes.
"""

import json
import sys
from collections import defaultdict

import numpy as np

REC_THRS = np.linspace(0.0, 1.0, 101)


def eval_det_cls_center_dist(pred, gt, dist_thresh):
    """Compute COCO-style precision at 101 recall points for a single class.

    pred: {ann_id: [(pred_center, gt_radius, score), ...]}
    gt:   {ann_id: [(gt_center, gt_radius), ...]}

    Returns precision array of shape (101,), or None if no GT exists.
    """
    class_recs = {}
    npos = 0
    for ann_id in gt:
        boxes = gt[ann_id]
        det = [False] * len(boxes)
        npos += len(boxes)
        class_recs[ann_id] = {"boxes": boxes, "det": det}
    for ann_id in pred:
        if ann_id not in class_recs:
            class_recs[ann_id] = {"boxes": [], "det": []}

    if npos == 0:
        return None

    image_ids = []
    confidence = []
    pred_centers = []
    for ann_id in pred:
        for pred_center, gt_radius, score in pred[ann_id]:
            image_ids.append(ann_id)
            confidence.append(score)
            pred_centers.append(pred_center)

    if len(pred_centers) == 0:
        return np.zeros(101)

    confidence = np.array(confidence)
    sorted_ind = np.argsort(-confidence)
    pred_centers = [pred_centers[i] for i in sorted_ind]
    image_ids = [image_ids[i] for i in sorted_ind]

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        pc = pred_centers[d]
        best_ratio = np.inf
        jmax = -1

        for j, (gc, gr) in enumerate(R["boxes"]):
            dist = np.linalg.norm(pc - gc)
            ratio = dist / (gr + 1e-6)
            if ratio < best_ratio:
                best_ratio = ratio
                jmax = j

        if best_ratio < dist_thresh:
            if not R["det"][jmax]:
                tp[d] = 1.0
                R["det"][jmax] = True
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / (tp + fp)

    # COCO-style: sample precision at 101 recall points with monotone envelope
    for i in range(nd - 1, 0, -1):
        if prec[i] > prec[i - 1]:
            prec[i - 1] = prec[i]

    q = np.zeros(101)
    inds = np.searchsorted(rec, REC_THRS, side="left")
    for ri, pi in enumerate(inds):
        if pi < nd:
            q[ri] = prec[pi]
    return q


def main():
    results_path = sys.argv[1]
    with open(results_path) as f:
        data = json.load(f)

    items = data["metadata"]
    print(f"Loaded {len(items)} samples from {results_path}")

    thresholds = np.arange(0.5, 1.0 + 0.05, 0.05)

    # Each annotation is a separate query (1-to-1), so use ann_id as the
    # grouping key to avoid cross-matching different objects in the same image.
    pred_all = defaultdict(lambda: defaultdict(list))
    gt_all = defaultdict(lambda: defaultdict(list))

    n_parsed = 0
    for item in items:
        meta = item["metadata"]
        gt_info = meta["ground_truth"]
        cat_id = gt_info["category_id"]
        ann_id = item["data_id"]

        gt_center = np.array(gt_info["center_cam"], dtype=np.float64)
        gt_dims = np.array(gt_info["dimensions"], dtype=np.float64)
        gt_radius = np.linalg.norm(gt_dims) / 2.0

        gt_all[cat_id][ann_id].append((gt_center, gt_radius))

        pred_data = json.loads(item["prediction"])
        parsed = pred_data["parsed"]
        if parsed is None:
            continue

        n_parsed += 1
        pred_center = np.array(parsed["center_cam"], dtype=np.float64)
        dist = np.linalg.norm(pred_center - gt_center)
        dist_ratio = dist / (gt_radius + 1e-6)
        score = max(0.0, 1.0 - dist_ratio)

        pred_all[cat_id][ann_id].append((pred_center, gt_radius, score))

    print(f"Parsed: {n_parsed}/{len(items)} ({n_parsed/len(items):.1%})")

    # Build cat_id -> cat_name mapping
    cat_id_to_name = {}
    for item in items:
        gt_info = item["metadata"]["ground_truth"]
        cat_id_to_name[gt_info["category_id"]] = gt_info["category_name"]

    all_cat_ids = sorted(gt_all.keys())

    # precision tensor: (T, R, K) — COCO style
    T = len(thresholds)
    K = len(all_cat_ids)
    precision = -np.ones((T, 101, K))

    for ki, cat_id in enumerate(all_cat_ids):
        cls_pred = dict(pred_all.get(cat_id, {}))
        cls_gt = dict(gt_all[cat_id])
        for ti, thresh in enumerate(thresholds):
            q = eval_det_cls_center_dist(cls_pred, cls_gt, thresh)
            if q is not None:
                precision[ti, :, ki] = q

    # Print results
    print(f"\n{'='*60}")
    print(f"AP (center-distance / GT-radius, COCO-style)")
    print(f"{'='*60}")
    for ti, thresh in enumerate(thresholds):
        s = precision[ti, :, :]
        s = s[s > -1]
        mAP = float(np.mean(s)) if s.size else 0.0
        print(f"  Threshold {thresh:.2f}:  mAP={mAP:.4f}")

    # Overall mAP: flat mean over all (T, R, K) where valid
    s_all = precision[precision > -1]
    mean_ap = float(np.mean(s_all)) if s_all.size else 0.0
    print(f"\n  >>> mAP (COCO-style, all thresholds): {mean_ap:.4f} <<<")

    # Per-class AP (averaged over all thresholds), show top/bottom 10
    cat_map = {}
    for ki, cat_id in enumerate(all_cat_ids):
        s = precision[:, :, ki]
        s = s[s > -1]
        if s.size == 0:
            continue
        cat_map[cat_id_to_name.get(cat_id, str(cat_id))] = float(np.mean(s))

    sorted_cats = sorted(cat_map.items(), key=lambda x: x[1], reverse=True)
    print(f"\n--- Per-class mAP (top 10) ---")
    for name, val in sorted_cats[:10]:
        print(f"    {name:30s}  mAP={val:.4f}")
    print(f"\n--- Per-class mAP (bottom 10) ---")
    for name, val in sorted_cats[-10:]:
        print(f"    {name:30s}  mAP={val:.4f}")


if __name__ == "__main__":
    main()
