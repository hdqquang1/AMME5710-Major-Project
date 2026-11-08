"""
baselines.py - reference methods the occlusion-aware pipeline is compared with.

1. Colour pipeline of the team's first prototype (finns-branch, commit 71c8398,
   scripts cherry_sai_refocus.py and cherry_count_compare.py), re-implemented
   with the same thresholds so it can run on the synthetic benchmark:
     * leaf mask: HSV green interval H 28-102, S >= 45, V >= 30 (OpenCV units),
       dilated 3x3;
     * synthetic aperture image: mean of the views refocused on ONE fruit
       depth, ignoring leaf-masked samples (at least 2 contributors);
     * detector: Lab a >= 140, R > 1.18 G + 1, R > 1.12 B + 1, R >= 26,
       3x3 open/close, contours with area 13-340 px, aspect <= 2.4 and
       circularity >= 0.24; the centroid of each contour is one fruit.
   It depends on the fruit being red and the leaves green.

2. Classic single-plane synthetic aperture (Vaish et al. 2006): mean or median
   of the refocused stack, counted with the colour-free 2-D detector of
   detect.py so the de-occlusion step is compared on equal terms.

Dependencies: numpy, opencv-python.
"""

import numpy as np
import cv2

from . import render


def finn_leaf_mask(bgr01):
    """Green leaf mask of the prototype (HSV interval), dilated 3x3; float32 in {0, 1}."""
    u8 = np.clip(bgr01 * 255 + 0.5, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array([28, 45, 30], np.uint8), np.array([102, 255, 255], np.uint8))
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    return (m > 0).astype(np.float32)


def finn_sai(rig, fruit_depth):
    """Leaf-masked mean synthetic aperture image on one fruit plane (prototype method)."""
    masks = [finn_leaf_mask(v.image) for v in rig.views]
    img, sup = render.refocus_plane(rig, fruit_depth, "mean", masks, min_views=2)
    return img, sup


def finn_colour_detect(bgr01, roi, a_min=140, min_area=13, max_area=340):
    """
    Red/pink contour detector of the prototype; returns detections with x, y, r_px
    and score = -area.  With a large max_area the score lets the benchmark
    calibrate the area cap (score >= t  <=>  area <= -t) on the calibration
    scenes, a tuned variant that gives the prototype its best chance.
    """
    u8 = np.clip(bgr01 * 255 + 0.5, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_BGR2LAB)
    r = u8[:, :, 2].astype(np.float32)
    g = u8[:, :, 1].astype(np.float32)
    b = u8[:, :, 0].astype(np.float32)
    x0, y0, x1, y1 = roi
    roi_m = np.zeros(r.shape, bool)
    roi_m[y0:y1, x0:x1] = True
    colour = (lab[:, :, 1] >= a_min) & (r > g * 1.18 + 1) & (r > b * 1.12 + 1) & (r >= 26) & roi_m
    m = colour.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = float(cv2.contourArea(c))
        bx, by, bw, bh = cv2.boundingRect(c)
        if not (min_area <= area <= max_area) or bw < 3 or bh < 3 or max(bw, bh) / max(1, min(bw, bh)) > 2.4:
            continue
        per = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (per * per) if per > 0 else 0
        if circ < 0.24:
            continue
        mm = cv2.moments(c)
        if mm["m00"] <= 0:
            continue
        out.append(dict(x=float(mm["m10"] / mm["m00"]), y=float(mm["m01"] / mm["m00"]),
                        r_px=float(np.sqrt(area / np.pi)), score=-float(area), feat={}))
    return out


def colour_mask(bgr01, roi=None, a_min=140):
    """The prototype's red/pink pixel rule (Lab a and R/G, R/B ratios), 3x3 open/close."""
    u8 = np.clip(bgr01 * 255 + 0.5, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_BGR2LAB)
    r = u8[:, :, 2].astype(np.float32)
    g = u8[:, :, 1].astype(np.float32)
    b = u8[:, :, 0].astype(np.float32)
    m = (lab[:, :, 1] >= a_min) & (r > g * 1.18 + 1) & (r > b * 1.12 + 1) & (r >= 26)
    if roi is not None:
        x0, y0, x1, y1 = roi
        keep = np.zeros(m.shape, bool)
        keep[y0:y1, x0:x1] = True
        m &= keep
    m = m.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)


def colour_dt_detect(bgr01, roi, r_px):
    """
    Strong colour baseline: the prototype's colour mask, split into fruit by the
    peaks of its distance transform (touching twins give two peaks).  Peaks
    closer than one fruit radius are merged; score = distance-transform value.
    """
    m = colour_mask(bgr01, roi)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    rr = max(2, int(round(0.9 * r_px)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))
    peaks = (dist >= cv2.dilate(dist, ker)) & (dist >= 0.35 * r_px)
    ys, xs = np.nonzero(peaks)
    order = np.argsort(-dist[ys, xs])
    out = []
    for k in order:
        x, y = float(xs[k]), float(ys[k])
        if any(np.hypot(x - o["x"], y - o["y"]) < r_px for o in out):
            continue
        out.append(dict(x=x, y=y, r_px=float(dist[ys[k], xs[k]]), score=float(dist[ys[k], xs[k]]), feat={}))
    return out


def finn_vote(rig, fruit_depth, roi, min_support=2):
    """
    The prototype's multi-view fusion (cherry_count_compare.fuse_multi_view):
    the colour mask of every photo is warped onto the reference through the
    fruit plane, pixels supported by at least `min_support` photos are kept and
    each connected component (area 12-350 px, aspect <= 2.8, circularity >= 0.28)
    is one fruit.  Score = peak number of supporting photos.
    """
    from .camera import plane_homography, relative_pose, warp_by_homography
    ref = rig.views[rig.ref]
    h, w = ref.image.shape[:2]
    votes = np.zeros((h, w), np.float32)
    for v in rig.views:
        m = colour_mask(v.image).astype(np.float32) / 255.0
        R, t = relative_pose(ref, v)
        H = plane_homography(ref.K, v.K, R, t, fruit_depth)
        mw = warp_by_homography(m, H, (w, h), border_value=0.0, interp=cv2.INTER_NEAREST)
        mw = cv2.dilate((mw > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        votes += mw
    x0, y0, x1, y1 = roi
    fused = np.zeros((h, w), np.uint8)
    fused[y0:y1, x0:x1] = (votes[y0:y1, x0:x1] >= min_support).astype(np.uint8) * 255
    fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(fused, 8)
    out = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if not (12 <= area <= 350) or ww < 3 or hh < 3 or max(ww, hh) / max(1, min(ww, hh)) > 2.8:
            continue
        comp = (lab[y:y + hh, x:x + ww] == i).astype(np.uint8)
        cc, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        per = cv2.arcLength(max(cc, key=cv2.contourArea), True) if cc else 0
        circ = 4 * np.pi * area / (per * per) if per else 0
        if circ < 0.28:
            continue
        out.append(dict(x=float(cents[i][0]), y=float(cents[i][1]), r_px=float(np.sqrt(area / np.pi)),
                        score=float(votes[lab == i].max()), feat={}))
    return out


def free_vote(rig, fruit_depth, roi, fruit="cherry", min_support=3):
    """
    Colour-free detect-then-integrate rival: the colour-free 2-D detector runs on
    every photo, each detection is mapped onto the reference through the fruit
    plane, and detections are clustered within one fruit radius.  A cluster
    seen in at least `min_support` photos is one fruit; score = number of photos.
    """
    from .camera import plane_homography, relative_pose
    from . import detect
    ref = rig.views[rig.ref]
    pts = []
    for v in rig.views:
        dets = detect.detect_2d(v.image, v.K, fruit_depth, fruit=fruit)
        if not dets:
            continue
        R, t = relative_pose(ref, v)
        Hinv = np.linalg.inv(plane_homography(ref.K, v.K, R, t, fruit_depth))
        uv = np.array([[d["x"], d["y"], 1.0] for d in dets]) @ Hinv.T
        uv = uv[:, :2] / uv[:, 2:3]
        sc = np.array([d["score"] for d in dets])
        # keep each photo's stronger half of candidates (2-D detector has no threshold here)
        keep = sc >= np.median(sc)
        for (x, y), r in zip(uv[keep], [d["r_px"] for d, k in zip(dets, keep) if k]):
            pts.append((x, y, r))
    if not pts:
        return []
    P = np.array(pts)
    r0 = float(np.median(P[:, 2]))
    used = np.zeros(len(P), bool)
    out = []
    # greedy clustering around the densest points
    dens = np.array([np.sum(np.hypot(P[:, 0] - p[0], P[:, 1] - p[1]) < r0) for p in P])
    for k in np.argsort(-dens):
        if used[k]:
            continue
        near = (~used) & (np.hypot(P[:, 0] - P[k, 0], P[:, 1] - P[k, 1]) < r0)
        used |= near
        if near.sum() >= min_support:
            x, y = P[near, 0].mean(), P[near, 1].mean()
            out.append(dict(x=float(x), y=float(y), r_px=r0, score=float(near.sum()), feat={}))
    x0, y0, x1, y1 = roi
    return [d for d in out if x0 <= d["x"] < x1 and y0 <= d["y"] < y1]
