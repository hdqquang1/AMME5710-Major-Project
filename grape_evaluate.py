#!/usr/bin/env python3
"""Evaluate a restoration: pixel-wise against a leaf-free photo, or by berry count.

    python grape_evaluate.py label --image PHOTO --out labels.csv
    python grape_evaluate.py count --detections RESULT/count_canny/berries.csv --labels labels.csv --registration pixel.json
    python grape_evaluate.py pixel --result RESULT --truth PHOTO --truth-grapes MASK --out pixel.json

Missing pixels are never counted as recovered. Count labels mark visible fruit in
one image, so their count can differ from the number of physical berries.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def evaluate(image, truth, roi, observed, reliable, baseline=None):
    """Colour fidelity on supported fruit pixels, plus mask agreement with the truth fruit."""
    if image.shape != truth.shape or any(
        m.shape != image.shape[:2] for m in [roi, observed, reliable]
    ):
        raise ValueError("Truth, reconstruction and masks must have matching dimensions")
    if not roi.any():
        raise ValueError("Ground-truth grape ROI is empty")
    supported = roi & observed
    result = {
        "grape_roi_pixels": int(roi.sum()),
        "supported_grape_pixels": int(supported.sum()),
        "supported_grape_fraction": float(supported.sum() / roi.sum()),
        "less_ambiguous_grape_fraction": float((roi & reliable).sum() / roi.sum()),
        "mask_iou": float(supported.sum() / max((roi | observed).sum(), 1)),
        "mask_precision": float(supported.sum() / max(observed.sum(), 1)),
        "mask_recall": float(supported.sum() / roi.sum()),
        "mae_on_supported_grapes": None,
        "psnr_db_on_supported_grapes": None,
    }
    if supported.any():
        error = image[supported].astype(float) - truth[supported].astype(float)
        mse = float(np.mean(error**2))
        result.update(
            mae_on_supported_grapes=float(np.mean(np.abs(error))),
            psnr_db_on_supported_grapes=float(10 * np.log10(255**2 / mse))
            if mse > 0
            else "infinite",
        )
        if baseline is not None:
            if baseline.shape != image.shape:
                raise ValueError("Baseline dimensions differ")
            result["baseline_mae_on_same_supported_grapes"] = float(
                np.abs(baseline[supported].astype(float) - truth[supported].astype(float)).mean()
            )
    result["warning"] = (
        "Coverage and fidelity are separate; missing pixels are never counted as successful recovery."
    )
    return result


def register_truth(truth, truth_fruit, reference, reference_fruit, grow=25, threshold=4.0):
    """Fit truth -> reference with a similarity on features near fruit, matching the fruit plane."""
    kernel = np.ones((2 * grow + 1,) * 2, np.uint8)
    sift = cv2.SIFT_create(nfeatures=20000, contrastThreshold=0.01)
    def features(image, fruit):
        region = cv2.dilate(np.uint8(fruit) * 255, kernel)
        return sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), region)
    kt, dt = features(truth, truth_fruit)
    kr, dr = features(reference, reference_fruit)
    if dt is None or dr is None or len(dt) < 2 or len(dr) < 2:
        raise ValueError("Too few features near fruit to register the truth photo")
    pairs = cv2.BFMatcher().knnMatch(dt, dr, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 8:
        raise ValueError(f"Only {len(good)} truth-to-reference matches; registration refused")
    a = np.float32([kt[m.queryIdx].pt for m in good])
    b = np.float32([kr[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.estimateAffinePartial2D(
        a, b, method=cv2.RANSAC, ransacReprojThreshold=threshold, maxIters=20000, confidence=0.999
    )
    if matrix is None or inliers.sum() < 8:
        raise ValueError("Truth registration failed; no metrics written")
    keep = inliers.ravel() > 0
    error = np.linalg.norm(a[keep] @ matrix[:, :2].T + matrix[:, 2] - b[keep], axis=1)
    return matrix, {
        "matches": len(good),
        "inliers": int(keep.sum()),
        "median_residual_px": float(np.median(error)),
        "scale": float(np.hypot(matrix[0, 0], matrix[1, 0])),
        "rotation_deg": float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))),
    }


def read_points(path):
    with open(path, newline="", encoding="utf8") as f:
        return [(float(row["x_px"]), float(row["y_px"])) for row in csv.DictReader(f)]


def match_points(preds, truth, r=12):
    """Greedy one-to-one matching within r pixels, closest pairs first."""
    candidates = []
    for a, p in enumerate(preds):
        for b, t in enumerate(truth):
            d = np.hypot(p[0] - t[0], p[1] - t[1])
            if d <= r:
                candidates.append((d, a, b))
    matched_pred, matched_truth = set(), set()
    for _, a, b in sorted(candidates):
        if a not in matched_pred and b not in matched_truth:
            matched_pred.add(a)
            matched_truth.add(b)
    tp = len(matched_pred)
    fp, fn = len(preds) - tp, len(truth) - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "true_count": len(truth),
        "predicted_count": len(preds),
        "count_error": len(preds) - len(truth),
        "relative_count_error": (len(preds) - len(truth)) / len(truth) if truth else None,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
        "matching_tolerance_px": r,
    }


def label(image_path, out, view_size=(1100, 800)):
    """Click berry centres on the unmodified image. Wheel zooms; W, A, Z, D or arrows pan.

    Left add, right remove nearest, S save, U undo, Q quit. Labels are original pixels.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    truth = [(round(x), round(y)) for x, y in read_points(out)] if out.exists() else []
    history = []
    h, w = image.shape[:2]
    view = {"zoom": 1.0, "x": 0.0, "y": 0.0}
    window = "Mark berries | wheel=zoom wazd or arrows=pan left=add right=remove s=save u=undo q=quit"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def visible():
        fit = min(view_size[0] / w, view_size[1] / h)
        scale = fit * view["zoom"]
        vw, vh = min(w, view_size[0] / scale), min(h, view_size[1] / scale)
        view["x"] = float(np.clip(view["x"], 0, w - vw))
        view["y"] = float(np.clip(view["y"], 0, h - vh))
        return scale, vw, vh

    def to_image(x, y):
        scale, _, _ = visible()
        return view["x"] + x / scale, view["y"] + y / scale

    def callback(event, x, y, flags, param):
        ix, iy = to_image(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            history.append(truth.copy())
            truth.append((round(ix), round(iy)))
        elif event == cv2.EVENT_RBUTTONDOWN and truth:
            history.append(truth.copy())
            truth.pop(int(np.argmin([(ix - a) ** 2 + (iy - b) ** 2 for a, b in truth])))
        elif event == cv2.EVENT_MOUSEWHEEL:
            factor = 1.25 if flags > 0 else 0.8  # wheel direction is the sign of flags
            view["zoom"] = float(np.clip(view["zoom"] * factor, 1, 16))
            scale, _, _ = visible()
            view["x"], view["y"] = ix - x / scale, iy - y / scale

    cv2.setMouseCallback(window, callback)
    pan = {ord("a"): (-1, 0), ord("d"): (1, 0), ord("w"): (0, -1), ord("z"): (0, 1), 2424832: (-1, 0), 2555904: (1, 0), 2490368: (0, -1), 2621440: (0, 1)}
    while True:
        scale, vw, vh = visible()
        x0, y0 = int(view["x"]), int(view["y"])
        crop = image[y0 : y0 + max(1, round(vh)), x0 : x0 + max(1, round(vw))]
        canvas = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        for i, (x, y) in enumerate(truth, 1):
            p = (round((x - view["x"]) * scale), round((y - view["y"]) * scale))
            cv2.drawMarker(canvas, p, (40, 255, 40), cv2.MARKER_CROSS, 11, 1, cv2.LINE_AA)
            cv2.putText(canvas, str(i), (p[0] + 7, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 255, 40), 1, cv2.LINE_AA)
        cv2.imshow(window, canvas)
        key = cv2.waitKeyEx(30)
        if key in pan:
            dx, dy = pan[key]
            view["x"] += dx * 0.2 * vw
            view["y"] += dy * 0.2 * vh
        elif key in (ord("s"), ord("S")):
            with out.open("w", newline="", encoding="utf8") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "x_px", "y_px"])
                writer.writerows((i, x, y) for i, (x, y) in enumerate(truth, 1))
            print(f"Saved {len(truth)} labels to {out}")
        elif key in (ord("u"), ord("U")) and history:
            truth[:] = history.pop()
        elif key in (ord("q"), ord("Q"), 27):
            break
    cv2.destroyAllWindows()


def read_image(path, mask=False):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if mask else cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"Cannot read {path}")
    return im > 127 if mask else im


def pixel(args):
    result_dir = args.result
    image = read_image(result_dir / "grapes_rgb.png")
    reference = read_image(result_dir / "reference.png")
    observed = read_image(result_dir / "observed_mask.png", True)
    reliable = read_image(result_dir / "analysis_mask.png", True)
    truth = read_image(args.truth)
    truth_fruit = read_image(args.truth_grapes, True)
    if truth_fruit.shape != truth.shape[:2]:
        raise ValueError("--truth-grapes must match the truth photo's size")
    if not truth_fruit.any():
        raise ValueError("--truth-grapes is empty; check the SAM prompt and that this photo matches the capture")
    reference_fruit = read_image(result_dir / "reference_fruit_mask.png", True) | observed
    matrix, registration = register_truth(truth, truth_fruit, reference, reference_fruit)
    h, w = image.shape[:2]
    area = abs(np.linalg.det(matrix[:2, :2]))
    warped = cv2.warpAffine(
        truth, matrix, (w, h), flags=cv2.INTER_AREA if area < 1 else cv2.INTER_LINEAR
    )
    roi = cv2.warpAffine(np.float32(truth_fruit), matrix, (w, h), flags=cv2.INTER_LINEAR) > 0.5
    result = evaluate(image, warped, roi, observed, reliable, reference)
    result["registration"] = registration
    result["truth_to_result_matrix"] = matrix.tolist()
    result["truth"] = str(args.truth)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    overlay = cv2.addWeighted(reference, 0.5, warped, 0.5, 0)
    cv2.imwrite(str(args.out.with_suffix(".registration.png")), overlay)
    # Same pixel frame as berries.csv, so count labels can include fruit the reference hides.
    cv2.imwrite(str(args.out.with_suffix(".truth_registered.png")), warped)
    print(json.dumps(result, indent=2))


def count(args):
    truth = read_points(args.labels)
    if args.registration:
        matrix = np.asarray(
            json.loads(args.registration.read_text(encoding="utf8"))["truth_to_result_matrix"]
        )
        truth = [tuple(matrix[:, :2] @ np.array(p) + matrix[:, 2]) for p in truth]
    result = {"restored": match_points(read_points(args.detections), truth, args.tolerance)}
    if args.baseline_detections:
        result["single_view_baseline"] = match_points(
            read_points(args.baseline_detections), truth, args.tolerance
        )
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf8")
    print(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    px = sub.add_parser("pixel", help="Score against a leaf-free photo")
    px.add_argument("--result", required=True, type=Path)
    px.add_argument("--truth", required=True, type=Path, help="Leaf-free photo of this capture")
    px.add_argument("--truth-grapes", required=True, type=Path, help="White=grapes in the truth photo")
    px.add_argument("--out", required=True, type=Path)
    lb = sub.add_parser("label", help="Click true berry centres")
    lb.add_argument("--image", required=True, type=Path)
    lb.add_argument("--out", required=True, type=Path)
    ct = sub.add_parser("count", help="Score detections against clicked labels")
    ct.add_argument("--detections", required=True, type=Path)
    ct.add_argument("--labels", required=True, type=Path)
    ct.add_argument("--registration", type=Path, help="pixel JSON; maps labels made on the truth photo into the result frame")
    ct.add_argument("--baseline-detections", type=Path, help="berries.csv from the single reference view")
    ct.add_argument("--tolerance", type=float, default=12)
    ct.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    if args.mode == "pixel":
        pixel(args)
    elif args.mode == "count":
        count(args)
    else:
        label(args.image, args.out)


if __name__ == "__main__":
    main()
