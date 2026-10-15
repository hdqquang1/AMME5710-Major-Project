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
    """Red/pink contour detector of the prototype; returns detections with x, y, r_px and score."""
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
                        r_px=float(np.sqrt(area / np.pi)), score=float(area), feat={}))
    return out
