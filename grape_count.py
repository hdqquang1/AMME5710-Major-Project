#!/usr/bin/env python3
"""Count visible restored berries with edges, watershed, Hough circles or SAM 3 instances.

Example:
    python grape_count.py --result results/CAPTURE --method canny_watershed

Counts cover what the reconstruction shows, not hidden berries. Berry size is
estimated per image, so tune thresholds on one capture and report on others.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.segmentation import watershed

METHODS = ("canny", "canny_watershed", "hough", "sam3", "grapesam")
GRAPESAM_DIR = Path(__file__).resolve().parent / "third_party"
_GRAPESAM = {}


def interior_edges(image, mask, low=40, high=120, border=2):
    """Canny inside the mask, without the edge the mask's own outline would create."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    # Fill outside the fruit with its median so the outline gives no gradient.
    if mask.any():
        gray[~mask] = int(np.median(gray[mask]))
    edges = cv2.Canny(gray, low, high) > 0
    inside = cv2.erode(np.uint8(mask), np.ones((2 * border + 1,) * 2, np.uint8)) > 0
    return edges & inside


def _blob(mask_u8):
    """Area, centre, equivalent radius and circularity of the largest contour."""
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area, perimeter = cv2.contourArea(contour), cv2.arcLength(contour, True)
    moments = cv2.moments(contour)
    if area <= 0 or perimeter <= 0 or moments["m00"] <= 0:
        return None
    return {
        "x_px": moments["m10"] / moments["m00"],
        "y_px": moments["m01"] / moments["m00"],
        "area_px": float(area),
        "radius_px": float(np.sqrt(area / np.pi)),
        "circularity": float(min(1, 4 * np.pi * area / perimeter**2)),
    }


def berry_radius(mask, floor=4.0):
    """Typical radius from distance peaks of the intact mask; thin slivers would skew it low."""
    distance = cv2.distanceTransform(np.uint8(mask), cv2.DIST_L2, 5)
    peaks = (distance >= cv2.dilate(distance, np.ones((9, 9), np.uint8)) - 1e-3) & (
        distance >= floor
    )
    return float(np.percentile(distance[peaks], 75)) if peaks.any() else floor


def watershed_regions(mask, edges, radius):
    """Seed from berry-sized peaks of the intact mask; Canny edges act as flood barriers."""
    distance = cv2.GaussianBlur(cv2.distanceTransform(np.uint8(mask), cv2.DIST_L2, 5), (5, 5), 1)
    size = max(3, 2 * round(0.7 * radius) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    peaks = (distance >= cv2.dilate(distance, kernel) - 1e-3) & (distance >= 0.45 * radius) & mask
    count, markers = cv2.connectedComponents(np.uint8(peaks))
    labels = watershed(radius * np.float32(edges) - distance, markers, mask=mask)
    return [labels == k for k in range(1, count)]


def count_berries(
    image,
    mask,
    method="canny",
    canny=(40, 120),
    min_area=30,
    min_circularity=0.35,
):
    """Return berry records and the estimated berry area in pixels."""
    if method not in ("canny", "canny_watershed"):
        raise ValueError(f"Unknown method: {method}")
    mask = mask.astype(bool)
    edges = interior_edges(image, mask, *canny)
    cut = mask & ~(cv2.dilate(np.uint8(edges), np.ones((3, 3), np.uint8)) > 0)
    radius = berry_radius(mask)
    area = float(np.pi * radius**2)
    if method == "canny":
        count, labels = cv2.connectedComponents(np.uint8(cut), connectivity=4)
        regions = [labels == k for k in range(1, count)]
    else:
        regions = watershed_regions(mask, edges, radius)
    berries = []
    for region in regions:
        blob = _blob(np.uint8(region))
        if blob is None or blob["area_px"] < max(min_area, 0.2 * area):
            continue
        # Only small slivers are rejected for shape: a merged cluster is not round but is fruit.
        if blob["circularity"] < min_circularity and blob["area_px"] <= area:
            continue
        berries.append(blob)
    return berries, area


def fill_small_holes(mask, max_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(~mask), 4)
    out = mask.copy()
    for k in range(1, count):
        if stats[k, cv2.CC_STAT_AREA] <= max_area:
            out[labels == k] = True
    return out


def count_hough(image, mask, canny_high=60, votes=8):
    """Edge fragments vote for circle centres, so outlines need not be closed."""
    mask = fill_small_holes(mask.astype(bool), 0.25 * np.pi * berry_radius(mask) ** 2)
    radius = berry_radius(mask)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    gray[~mask] = int(np.median(gray[mask]))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        1,
        minDist=1.2 * radius,
        param1=canny_high,
        param2=votes,
        minRadius=max(3, round(0.7 * radius)),
        maxRadius=round(1.4 * radius),
    )
    berries = []
    h, w = mask.shape
    for x, y, r in [] if circles is None else circles[0]:
        if mask[min(int(y), h - 1), min(int(x), w - 1)]:
            area = float(np.pi * r * r)
            berries.append(
                {"x_px": float(x), "y_px": float(y), "radius_px": float(r), "area_px": area, "circularity": 1.0}
            )
    return berries, float(np.pi * radius**2)


def count_sam3(image, mask, prompt="grape", threshold=0.2, background=255, max_overlap=0.5):
    """SAM 3 instances on the restored fruit, so each detected object is one berry."""
    from grape_segment import sam3_instances

    canvas = image.copy()
    canvas[~mask.astype(bool)] = background
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    instances = sam3_instances(rgb, prompt, threshold)
    berries, kept = [], []
    for region in sorted(instances, key=lambda m: int(m.sum())):
        region = region & mask.astype(bool)
        area = int(region.sum())
        # Prompts can also return a whole bunch; a mask covering kept berries is dropped.
        if not area or any((region & k).sum() > max_overlap * min(area, k.sum()) for k in kept):
            continue
        blob = _blob(np.uint8(region))
        if blob:
            kept.append(region)
            berries.append(blob)
    return berries, float(np.median([b["area_px"] for b in berries])) if berries else 0.0


def _grapesam_model(weights):
    """The published ViViD-5K point model; weights_only keeps the checkpoint from running code."""
    import importlib.util

    import torch

    if weights not in _GRAPESAM:
        path = GRAPESAM_DIR / "GrapeSAM" / "model" / "point" / "models" / "vgg.py"
        spec = importlib.util.spec_from_file_location("grapesam_vgg", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.vgg19().eval().cuda()
        state = torch.load(weights, map_location="cuda", weights_only=True)
        model.load_state_dict(state.get("model_state_dict", state))
        _GRAPESAM[weights] = model
    return _GRAPESAM[weights]


def count_grapesam(image, weights, threshold=0.05, max_points=1024):
    """Berry centres from GrapeSAM's point model, using its published post-processing."""
    import torch
    import torch.nn.functional as F

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    rgb = (rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        heat = _grapesam_model(str(weights))(tensor)
    heat = F.interpolate(heat.float(), scale_factor=8, mode="bilinear")
    keep = F.max_pool2d(heat, 3, stride=1, padding=1) == heat
    heat = (heat * keep)[0, 0]
    ys, xs = torch.nonzero(heat > threshold, as_tuple=True)
    scores = heat[ys, xs]
    order = torch.argsort(scores, descending=True)[:max_points]
    torch.cuda.empty_cache()
    return [
        {"x_px": float(xs[i]), "y_px": float(ys[i]), "radius_px": 8.0, "area_px": 0.0, "circularity": 0.0, "score": float(scores[i])}
        for i in order
    ]


def draw(image, berries):
    out = image.copy()
    for i, b in enumerate(berries, 1):
        centre = (round(b["x_px"]), round(b["y_px"]))
        cv2.circle(out, centre, max(2, round(b["radius_px"])), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, str(i), centre, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save(out, image, berries, median, method):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fields = ["id", "x_px", "y_px", "radius_px", "area_px", "circularity"]
    with (out / "berries.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for i, b in enumerate(berries, 1):
            writer.writerow({"id": i, **{k: round(b[k], 3) for k in fields[1:]}})
    if not cv2.imwrite(str(out / "count_overlay.png"), draw(image, berries)):
        raise OSError(f"Cannot write overlay in {out}")
    summary = {
        "method": method,
        "count": len(berries),
        "estimated_berry_area_px": median,
        "note": "Automated count of visible regions; not verified and excludes hidden berries.",
    }
    (out / "count.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result", type=Path, help="grape_restore_autoref.py output folder")
    p.add_argument("--image", type=Path, help="Override image; default RESULT/grapes_rgb.png")
    p.add_argument("--mask", type=Path, help="Override mask; default RESULT/analysis_mask.png")
    p.add_argument("--out", type=Path, help="Default RESULT/count_<method>")
    p.add_argument("--method", choices=METHODS, default="canny_watershed")
    p.add_argument("--canny", type=int, nargs=2, default=(40, 120), metavar=("LOW", "HIGH"))
    p.add_argument("--min-area", type=int, default=30)
    p.add_argument("--min-circularity", type=float, default=0.35)
    p.add_argument("--hough-votes", type=int, default=8)
    p.add_argument("--hough-canny", type=int, default=60)
    p.add_argument("--fruit-prompt", default="grape", help="sam3 method")
    p.add_argument("--weights", type=Path, help="grapesam method: point model checkpoint")
    p.add_argument("--sam-threshold", type=float, default=0.2)
    args = p.parse_args(argv)
    if not args.result and not (args.image and args.mask and args.out):
        p.error("Give --result, or all of --image, --mask and --out")
    image_path = args.image or args.result / "grapes_rgb.png"
    mask_path = args.mask or args.result / "analysis_mask.png"
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise ValueError(f"Cannot read {image_path} or {mask_path}")
    if args.method == "hough":
        berries, median = count_hough(image, mask > 127, args.hough_canny, args.hough_votes)
    elif args.method == "grapesam":
        if not args.weights:
            p.error("--method grapesam needs --weights")
        canvas = image.copy()
        canvas[mask <= 127] = 0
        berries, median = count_grapesam(canvas, args.weights), 0.0
    elif args.method == "sam3":
        berries, median = count_sam3(image, mask > 127, args.fruit_prompt, args.sam_threshold)
    else:
        berries, median = count_berries(
            image, mask > 127, args.method, tuple(args.canny), args.min_area, args.min_circularity
        )
    out = args.out or args.result / f"count_{args.method}"
    print(json.dumps(save(out, image, berries, median, args.method), indent=2))


if __name__ == "__main__":
    main()
