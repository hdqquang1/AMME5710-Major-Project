"""
verify.py - learned, colour-free verification of fruit candidates.

The candidate features of detect.detect_fruit (relief height and blob response,
estimated radius, disc fill, annulus occupancy, luminance radial symmetry and
blob strength, view support, depth confidence) are combined by a random forest
(cv2.ml.RTrees, Breiman 2001) trained on the CALIBRATION scenes only.  Lengths
are divided by the nominal fruit radius, so one model serves cherries, grapes
and limes.  The verifier outputs the fraction of trees voting "fruit", used as
the detection score.

Dependencies: numpy, opencv-python.
"""

import numpy as np
import cv2

from .detect import FRUIT_RADII

FEATURES = ["resp_mm", "relief_mm", "R_mm", "fill", "annulus", "frst", "app", "support", "conf"]


def feature_matrix(dets, fruit):
    """N x F float32 matrix; length features normalised by the nominal fruit radius."""
    r0 = 1e3 * float(np.sqrt(np.prod(FRUIT_RADII[fruit])))
    rows = []
    for d in dets:
        f = d["feat"]
        rows.append([f["resp_mm"] / r0, f["relief_mm"] / r0, f["R_mm"] / r0, f["fill"], f["annulus"],
                     f["frst"], f["app"], f["support"], f["conf"]])
    return np.array(rows, np.float32).reshape(-1, len(FEATURES))


def train(X, y, n_trees=150, max_depth=8, seed=0):
    """Random-forest classifier (labels 0/1)."""
    cv2.setRNGSeed(seed)
    rt = cv2.ml.RTrees_create()
    rt.setMaxDepth(max_depth)
    rt.setMinSampleCount(4)
    rt.setActiveVarCount(3)
    rt.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, n_trees, 0))
    rt.train(cv2.ml.TrainData_create(X, cv2.ml.ROW_SAMPLE, y.astype(np.int32)))
    return rt


def fruit_probability(model, X):
    """Fraction of trees voting class 1 for every row of X."""
    if len(X) == 0:
        return np.zeros(0, np.float32)
    votes = model.getVotes(X, 0)
    labels = votes[0]
    counts = votes[1:].astype(np.float32)
    total = np.maximum(counts.sum(axis=1), 1.0)
    if 1 in labels:
        return counts[:, list(labels).index(1)] / total
    return np.zeros(len(X), np.float32)
