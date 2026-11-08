"""
evaluate.py - counting and image-quality metrics against ground truth.

Counting protocol
-----------------
* Countable fruit: centre inside the evaluation ROI of the reference view and
  at least `min_vis` of its disc visible in the FOLIAGE-FREE reference view
  (fruit hidden behind other fruit cannot be counted by any front-on method).
* A detection matches a ground-truth fruit when their reference-image centres
  are closer than `tol` times the fruit's pixel radius; matching is one-to-one
  and greedy by distance (shortest first).
* Reported: TP, FP, FN, precision, recall, F1, the count n_hat and the
  counting error (n_hat - n) / n.  Only detections inside the ROI count.

Image protocol
--------------
PSNR and SSIM (Wang et al. 2004, 11x11 Gaussian window, sigma 1.5) of a
leaf-free image against the foliage-free photo of the same view, inside the ROI.

Dependencies: numpy, opencv-python.
"""

import numpy as np
import cv2


def countable_fruit(scene, min_vis=0.25):
    """Ground-truth fruit that lie in the ROI and are visible without foliage."""
    x0, y0, x1, y1 = scene["roi_ref"]
    out = []
    for fr in scene["fruit"]:
        u, v = fr["uv_ref"]
        if x0 <= u < x1 and y0 <= v < y1 and fr["visible_frac_ref_clean"] >= min_vis:
            out.append(fr)
    return out


def in_roi(dets, roi):
    x0, y0, x1, y1 = roi
    return [d for d in dets if x0 <= d["x"] < x1 and y0 <= d["y"] < y1]


def match_counts(dets, gts, tol=0.8):
    """
    One-to-one greedy matching of detections to ground truth in the reference image.
    Returns dict of counts and rates, plus the matched index pairs.
    """
    pairs = []
    for i, d in enumerate(dets):
        for j, g in enumerate(gts):
            dist = np.hypot(d["x"] - g["uv_ref"][0], d["y"] - g["uv_ref"][1])
            if dist <= tol * max(g["r_px_ref"], 1.0):
                pairs.append((dist, i, j))
    pairs.sort()
    used_d, used_g, matched = set(), set(), []
    for dist, i, j in pairs:
        if i in used_d or j in used_g:
            continue
        used_d.add(i)
        used_g.add(j)
        matched.append((i, j))
    tp = len(matched)
    fp = len(dets) - tp
    fn = len(gts) - tp
    n = len(gts)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, n_gt=n, n_pred=len(dets), precision=prec, recall=rec, f1=f1,
                count_error=(len(dets) - n) / n if n else 0.0, matched=matched)


def pr_sweep(dets, gts, thresholds, tol=0.8):
    """Precision/recall/F1 for each score threshold (detections with score >= t)."""
    rows = []
    for t in thresholds:
        sel = [d for d in dets if d["score"] >= t]
        m = match_counts(sel, gts, tol)
        rows.append(dict(t=float(t), precision=m["precision"], recall=m["recall"], f1=m["f1"],
                         n_pred=m["n_pred"], n_gt=m["n_gt"], count_error=m["count_error"]))
    return rows


def psnr(a, b, roi):
    x0, y0, x1, y1 = roi
    m = np.mean((a[y0:y1, x0:x1].astype(np.float64) - b[y0:y1, x0:x1]) ** 2)
    return float(10 * np.log10(1.0 / max(m, 1e-12)))


def ssim(a, b, roi):
    """Mean SSIM over the ROI of the luminance images (inputs float BGR in [0, 1])."""
    x0, y0, x1, y1 = roi
    ga = cv2.cvtColor(np.clip(a, 0, 1).astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.float64)
    gb = cv2.cvtColor(np.clip(b, 0, 1).astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.float64)
    C1, C2 = (0.01) ** 2, (0.03) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = blur(ga), blur(gb)
    s_aa = blur(ga * ga) - mu_a ** 2
    s_bb = blur(gb * gb) - mu_b ** 2
    s_ab = blur(ga * gb) - mu_a * mu_b
    smap = ((2 * mu_a * mu_b + C1) * (2 * s_ab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (s_aa + s_bb + C2))
    return float(smap[y0:y1, x0:x1].mean())


def mask_scores(pred, gt):
    """Recall, precision and IoU of a binary mask against ground truth."""
    p, g = pred > 0.5, gt > 0.5
    inter = np.sum(p & g)
    return dict(recall=float(inter / max(g.sum(), 1)), precision=float(inter / max(p.sum(), 1)),
                iou=float(inter / max(np.sum(p | g), 1)))
