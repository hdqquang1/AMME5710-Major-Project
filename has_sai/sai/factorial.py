"""
factorial.py - separate the contribution of de-occlusion from that of detection.

Every leaf-free image source is crossed with every detector, so a gain can be
credited to the stage that produced it:

  image sources: centre photo, prototype leaf-masked mean SAI, median SAI,
                 occlusion-aware all-in-focus image (ours), foliage-free photo
  detectors:     prototype colour detector (as published), the same with its
                 contour area cap calibrated, the prototype colour mask split by
                 distance-transform peaks (a strong colour baseline), colour-free 2-D detector
                 (luminance blob x radial symmetry); the colour-free 3-D detector
                 needs a depth map and is reported only for ours and the oracle

The images are the 8-bit panels saved by run_synthetic_benchmark.py.

Dependencies: numpy, opencv-python.
"""

import os

import numpy as np
import cv2

from . import baselines, detect, evaluate

SOURCES = ["single", "finn_sai", "median_sai", "ours", "clean"]
DETECTORS = ["colour", "colour_tuned", "colour_dt", "free2d"]


def factorial_detections(scene_dir, roi, K, z_fruit, fruit):
    """Candidate lists for every (image source, detector) pair of one scene."""
    out = {}
    r_px = K[0, 0] * float(np.sqrt(np.prod(detect.FRUIT_RADII[fruit]))) / z_fruit
    for src in SOURCES:
        p = os.path.join(scene_dir, "panels", f"{src}.png")
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            continue
        img = im.astype(np.float32) / 255.0
        out[f"{src}|colour"] = baselines.finn_colour_detect(img, roi)
        out[f"{src}|colour_tuned"] = baselines.finn_colour_detect(img, roi, max_area=1e5)
        out[f"{src}|colour_dt"] = baselines.colour_dt_detect(img, roi, r_px)
        out[f"{src}|free2d"] = detect.nms_3d(evaluate.in_roi(detect.detect_2d(img, K, z_fruit, fruit=fruit), roi))
    return out
