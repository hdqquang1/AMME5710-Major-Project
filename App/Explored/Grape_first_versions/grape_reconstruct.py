#!/usr/bin/env python3
"""Earlier experimental reconstruction for a parallel camera rail.

Depth-limited plane sweep + robust photometric consensus. This reconstructs
observed surfaces, not hidden grape geometry. Use grape_restore.py for the
measured-motion workflow validated on the real-grape Drive sequence.
See README.md for the comparison and assumptions.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def natural_key(name):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(name))
    ]


def read_rgb(blob):
    with Image.open(io.BytesIO(blob)) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB")).copy()


def load_sequence(source, scale=0.25, masks=None, mask_dilate=2, calibration=None):
    """Read EXIF-oriented images; white masks exclude occluders of any colour.

    Mask filenames mirror input relative paths, replacing the extension with
    .png. Masks must be reviewed and have the oriented ORIGINAL image size.
    """
    source = Path(source)
    if not 0 < scale <= 1:
        raise ValueError("scale must be in (0, 1]")
    archive = None
    if source.is_dir():
        names = sorted(
            (
                p.relative_to(source).as_posix()
                for p in source.rglob("*")
                if p.is_file() and p.suffix.lower() in EXTENSIONS
            ),
            key=natural_key,
        )
        read = lambda name: (source / name).read_bytes()
    elif source.is_file() and source.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(source)
        names = sorted(
            (
                n
                for n in archive.namelist()
                if Path(n).suffix.lower() in EXTENSIONS
                and not n.startswith("__MACOSX/")
            ),
            key=natural_key,
        )
        read = archive.read
    else:
        raise ValueError(f"Image directory or ZIP not found: {source}")
    if len(names) < 3:
        if archive:
            archive.close()
        raise ValueError("Need at least three views, in capture order")
    if len(names) > 65534:
        if archive:
            archive.close()
        raise ValueError("Too many views for uint16 provenance output")
    images, allowed, size = [], [], None
    try:
        if len(set(names)) != len(names) or any(
            Path(name).is_absolute() or ".." in Path(name).parts for name in names
        ):
            raise ValueError(
                "Input names must be unique relative paths without parent traversal"
            )
        if masks and len({str(Path(n).with_suffix(".png")) for n in names}) != len(
            names
        ):
            raise ValueError("Input image stems collide in the mask folder")
        for name in names:
            rgb = read_rgb(read(name))
            h, w = rgb.shape[:2]
            if size is None:
                size = (w, h)
            if size != (w, h):
                raise ValueError(f"Image sizes differ after EXIF orientation: {name}")
            valid = np.ones((h, w), np.uint8)
            if masks:
                mask_path = Path(masks) / Path(name).with_suffix(".png")
                if not mask_path.is_file():
                    raise ValueError(
                        f"Missing occluder mask: {mask_path}; supply a black mask for clear views"
                    )
                with Image.open(mask_path) as mask:
                    mask = np.asarray(mask.convert("L"))
                if mask.shape != (h, w):
                    raise ValueError(
                        f"Mask must match EXIF-oriented original dimensions: {mask_path}"
                    )
                valid = np.uint8(mask < 128)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if calibration:
                if [w, h] != calibration["image_size"]:
                    raise ValueError(
                        "Calibration image_size must match EXIF-oriented originals"
                    )
                k = np.asarray(calibration["K"], np.float64)
                dist = np.asarray(
                    calibration.get("distortion", [0, 0, 0, 0, 0]), np.float64
                )
                mx, my = cv2.initUndistortRectifyMap(
                    k, dist, None, k, (w, h), cv2.CV_32FC1
                )
                bgr = cv2.remap(bgr, mx, my, cv2.INTER_LINEAR)
                valid = cv2.remap(valid, mx, my, cv2.INTER_NEAREST, borderValue=0)
                valid &= np.uint8((mx >= 0) & (mx < w - 1) & (my >= 0) & (my < h - 1))
            target = (max(1, round(w * scale)), max(1, round(h * scale)))
            images.append(cv2.resize(bgr, target, interpolation=cv2.INTER_AREA))
            # Preserve any occluder intersecting a downsampled pixel.
            valid = (
                cv2.resize(
                    valid.astype(np.float32), target, interpolation=cv2.INTER_AREA
                )
                > 0.999
            )
            if mask_dilate:
                valid = (
                    cv2.erode(
                        valid.astype(np.uint8),
                        np.ones((2 * mask_dilate + 1,) * 2, np.uint8),
                        borderType=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    > 0
                )
            allowed.append(valid)
    finally:
        if archive:
            archive.close()
    return names, images, allowed, size


@dataclass
class Geometry:
    fx: float  # processing pixels
    positions: np.ndarray  # signed camera x coordinates, mm
    corrections: (
        np.ndarray
    )  # additional source-to-reference translation, processing pixels
    reference: int


def estimate_direction(images):
    sift = cv2.SIFT_create(nfeatures=2500)
    shifts = []
    for a, b in zip(images[:-1], images[1:]):
        ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
        kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
        if da is None or db is None or len(db) < 2:
            continue
        for pair in cv2.BFMatcher().knnMatch(da, db, k=2):
            if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance:
                m = pair[0]
                dx, dy = np.subtract(kb[m.trainIdx].pt, ka[m.queryIdx].pt)
                if abs(dy) < 3 and abs(dx) > 0.5:
                    shifts.append(dx)
    if len(shifts) < 12 or abs(np.median(shifts)) < 0.5:
        raise ValueError(
            "Cannot infer rail direction; set --direction right or left after checking capture order"
        )
    return -1 if np.median(shifts) > 0 else 1


def refine_translations(images, allowed, geometry, depth, gate=2.5):
    """Small target-motion-guided corrections; never unrestricted leaf alignment."""
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02)
    features = [
        sift.detectAndCompute(
            cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), np.uint8(mask) * 255
        )
        for im, mask in zip(images, allowed)
    ]
    kr, dr = features[geometry.reference]
    diagnostics = []
    for i, (ki, di) in enumerate(features):
        record = {"view_1based": i + 1, "accepted": False, "target_matches": 0}
        diagnostics.append(record)
        if i == geometry.reference:
            record.update(accepted=True, reason="reference")
            continue
        if dr is None or di is None or len(dr) < 2:
            record["reason"] = "insufficient descriptors"
            continue
        expected = (
            geometry.fx
            * (geometry.positions[i] - geometry.positions[geometry.reference])
            / depth
        )
        residuals = []
        for pair in cv2.BFMatcher().knnMatch(di, dr, k=2):
            if len(pair) != 2 or pair[0].distance >= 0.7 * pair[1].distance:
                continue
            m = pair[0]
            delta = np.subtract(kr[m.trainIdx].pt, ki[m.queryIdx].pt) - [expected, 0]
            if np.linalg.norm(delta) <= gate:
                residuals.append(delta)
        record["target_matches"] = len(residuals)
        if len(residuals) < 12:
            record["reason"] = (
                "too few matches consistent with target depth; geometry retained"
            )
            continue
        residuals = np.asarray(residuals)
        correction = np.median(residuals, axis=0)
        inliers = np.linalg.norm(residuals - correction, axis=1) < 1.0
        if inliers.sum() < 10:
            record["reason"] = "inconsistent residuals; geometry retained"
            continue
        geometry.corrections[i] = np.median(residuals[inliers], axis=0)
        record.update(
            accepted=True,
            correction_px=geometry.corrections[i].tolist(),
            inliers=int(inliers.sum()),
        )
    return diagnostics


def warp_tile(images, allowed, geometry, depth, bounds):
    x0, y0, x1, y1 = bounds
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    stack, valid = [], []
    h, w = images[0].shape[:2]
    for i, (im, mask) in enumerate(zip(images, allowed)):
        dx = (
            geometry.fx
            * (geometry.positions[i] - geometry.positions[geometry.reference])
            / depth
        )
        mx = (xx - dx - geometry.corrections[i, 0]).astype(np.float32)
        my = (yy - geometry.corrections[i, 1]).astype(np.float32)
        stack.append(
            cv2.remap(im, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        )
        # An interpolated colour is usable only when all contributing mask pixels are clear.
        clear = (
            cv2.remap(mask.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderValue=0)
            > 0.999
        )
        valid.append(clear & (mx >= 0) & (mx < w - 1) & (my >= 0) & (my < h - 1))
    return np.asarray(stack), np.asarray(valid)


def fuse_consensus(stack, valid, positions, tolerance=18.0, seed_views=0):
    """Choose a locally coherent colour mode, then robustly blend its inliers.

    No colour class is assumed. Large uniform occluders can also agree: depth
    ambiguity is exported separately, and reviewed masks remain recommended.
    """
    n, h, w = valid.shape
    values = stack.astype(np.float32)
    smooth = np.asarray(
        [
            cv2.GaussianBlur(im * mask[..., None], (3, 3), 0.7)
            / np.maximum(
                cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0.7)[..., None], 1e-6
            )
            for im, mask in zip(values, valid)
        ]
    )
    best = np.full((h, w), -np.inf, np.float32)
    best_seed = np.zeros((h, w), np.int32)
    # All candidates by default; optional subsampling trades recovery for speed.
    seeds = n if seed_views == 0 else min(n, seed_views)
    for seed in np.unique(np.linspace(0, n - 1, seeds).round().astype(int)):
        distance = np.sqrt(np.mean((smooth - smooth[seed]) ** 2, axis=-1))
        inliers = valid & valid[seed] & (distance < tolerance)
        count = inliers.sum(axis=0)
        residual = np.sum(np.where(inliers, distance, 0), axis=0) / np.maximum(count, 1)
        score = count - 0.25 * residual / tolerance
        update = score > best
        best[update] = score[update]
        best_seed[update] = seed
    yy, xx = np.mgrid[:h, :w]
    distance = np.sqrt(np.mean((smooth - smooth[best_seed, yy, xx]) ** 2, axis=-1))
    inliers = valid & valid[best_seed, yy, xx] & (distance < tolerance)
    weights = np.where(inliers, np.maximum(1 - distance / tolerance, 0) ** 2, 0)
    weight_sum = weights.sum(axis=0)
    result = (values * weights[..., None]).sum(axis=0) / np.maximum(
        weight_sum[..., None], 1e-6
    )
    residual = np.sum(weights * distance, axis=0) / np.maximum(weight_sum, 1e-6)
    support = inliers.sum(axis=0).astype(np.uint16)
    span = np.max(
        np.where(inliers, positions[:, None, None], -np.inf), axis=0
    ) - np.min(np.where(inliers, positions[:, None, None], np.inf), axis=0)
    span = np.where(support > 0, span, 0)
    full_span = max(float(np.ptp(positions)), 1e-6)
    # Reward repeat observations across a useful baseline, penalize disagreement.
    score = (
        support.astype(np.float32)
        + 0.25 * span / full_span
        - 0.25 * residual / tolerance
    )
    score[support == 0] = -1
    confidence = (
        support
        / np.maximum(valid.sum(axis=0), 1)
        * np.maximum(1 - residual / tolerance, 0)
    )
    return {
        "image": np.uint8(np.clip(result, 0, 255)),
        "support": support,
        "score": score,
        "agreement": confidence.astype(np.float32),
        "source": np.where(support > 0, best_seed + 1, 0).astype(np.uint16),
        "span": span.astype(np.float32),
    }


def canvas_bounds(images, geometry, depths, union=False):
    h, w = images[0].shape[:2]
    if not union:
        return (0, 0, w, h)
    shifts = np.asarray(
        [
            geometry.fx
            * (geometry.positions - geometry.positions[geometry.reference])
            / z
            + geometry.corrections[:, 0]
            for z in depths
        ]
    )
    y = geometry.corrections[:, 1]
    return (
        int(np.floor(shifts.min())),
        int(np.floor(y.min())),
        w + int(np.ceil(shifts.max())),
        h + int(np.ceil(y.max())),
    )


def reconstruct(
    images,
    allowed,
    geometry,
    depths,
    tile_size=192,
    min_views=3,
    tolerance=18.0,
    seed_views=0,
    min_margin=0.15,
    progress=None,
    bounds=None,
    min_baseline=40.0,
):
    ox, oy, ex_canvas, ey_canvas = bounds or canvas_bounds(images, geometry, depths)
    h, w = ey_canvas - oy, ex_canvas - ox
    result = {
        "image": np.zeros((h, w, 3), np.uint8),
        "depth_mm": np.zeros((h, w), np.float32),
        "support": np.zeros((h, w), np.uint16),
        "agreement": np.zeros((h, w), np.float32),
        "source": np.zeros((h, w), np.uint16),
        "depth_margin": np.zeros((h, w), np.float32),
        "span": np.zeros((h, w), np.float32),
    }
    total = ((h + tile_size - 1) // tile_size) * ((w + tile_size - 1) // tile_size)
    done = 0
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            x1, y1 = min(x + tile_size, w), min(y + tile_size, h)
            # Padding makes patch consensus invariant to processing tile boundaries.
            bx, by, ex, ey = (
                max(0, x - 2),
                max(0, y - 2),
                min(w, x1 + 2),
                min(h, y1 + 2),
            )
            inner = np.s_[y - by : y1 - by, x - bx : x1 - bx]
            candidates = []
            for depth in depths:
                stack, valid = warp_tile(
                    images,
                    allowed,
                    geometry,
                    depth,
                    (bx + ox, by + oy, ex + ox, ey + oy),
                )
                candidate = fuse_consensus(
                    stack, valid, geometry.positions, tolerance, seed_views
                )
                candidates.append({k: v[inner] for k, v in candidate.items()})
            scores = np.asarray([c["score"] for c in candidates])
            winner = scores.argmax(axis=0)
            yy, xx = np.mgrid[: y1 - y, : x1 - x]
            dst = np.s_[y:y1, x:x1]
            for key in ["image", "support", "agreement", "source", "span"]:
                result[key][dst] = np.asarray([c[key] for c in candidates])[
                    winner, yy, xx
                ]
            result["depth_mm"][dst] = np.asarray(depths)[winner]
            # Compare distant hypotheses, not an adjacent near-identical focal plane.
            competitors = np.abs(np.arange(len(depths))[:, None, None] - winner) > 1
            runner = np.max(np.where(competitors, scores, -np.inf), axis=0)
            margin = scores[winner, yy, xx] - runner
            result["depth_margin"][dst] = margin if len(depths) > 1 else 0
            done += 1
            if progress:
                progress(done, total)
    observed = result["support"] >= min_views
    # Depth separation is evidence of geometric consistency, not semantic identification.
    reliable = (
        observed
        & (result["depth_margin"] >= min_margin)
        & (result["agreement"] >= 0.5)
        & (result["span"] >= min_baseline)
    )
    result["observed"] = observed
    result["reliable"] = reliable
    result["image"][~observed] = 0
    for key in ["depth_mm", "source", "agreement", "depth_margin", "span"]:
        result[key][~observed] = 0
    return result


def save_outputs(out, result, reference, diagnostics):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    def write(name, im):
        if not cv2.imwrite(str(out / name), im):
            raise OSError(f"Cannot write {out / name}")

    write("reference.png", reference)
    write("reconstruction.png", result["image"])
    rgba = np.dstack([result["image"], np.uint8(result["observed"]) * 255])
    write("reconstruction_rgba.png", rgba)
    rgba[..., 3] = np.uint8(result["reliable"]) * 255
    write("analysis_ready_rgba.png", rgba)
    write("observed_mask.png", np.uint8(result["observed"]) * 255)
    write("analysis_mask.png", np.uint8(result["reliable"]) * 255)
    write("support_count.png", result["support"])
    write("source_view.png", result["source"])
    write("agreement.png", np.uint8(np.clip(result["agreement"], 0, 1) * 255))
    # NPZ preserves physical depths, margins and support without preview quantisation.
    np.savez_compressed(
        out / "evidence.npz", **{k: v for k, v in result.items() if k != "image"}
    )
    heat = cv2.applyColorMap(np.uint8(result["agreement"] * 255), cv2.COLORMAP_VIRIDIS)
    heat[~result["observed"]] = 0
    write("agreement_preview.jpg", heat)
    diagnostic = result["image"].copy()
    diagnostic[~result["observed"]] = (180, 0, 180)
    diagnostic[result["observed"] & ~result["reliable"]] = (0, 150, 255)
    write("uncertainty_preview.jpg", diagnostic)
    diagnostics.update(
        observed_fraction=float(result["observed"].mean()),
        analysis_fraction=float(result["reliable"].mean()),
        note="Support/agreement are heuristics, not calibrated probabilities or proof a pixel is grape.",
    )
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--images", default="/content/drive/MyDrive/CompVis/Major/2cm_74cm_3branches"
    )
    p.add_argument("--out", default="grape_results")
    p.add_argument("--spacing", type=float, default=20, help="Camera spacing, mm")
    p.add_argument(
        "--leaf-depth", type=float, default=740, help="Approximate foreground depth, mm"
    )
    p.add_argument(
        "--fruit-depth", type=float, default=1100, help="Nominal target depth, mm"
    )
    p.add_argument(
        "--depth-range",
        type=float,
        nargs=2,
        default=(950, 1250),
        metavar=("NEAR", "FAR"),
    )
    p.add_argument("--depth-steps", type=int, default=9)
    focal = p.add_mutually_exclusive_group()
    focal.add_argument(
        "--fx", type=float, help="Calibrated fx, ORIGINAL oriented image pixels"
    )
    focal.add_argument(
        "--equiv-focal",
        type=float,
        help="Approximate full-frame-equivalent focal length, mm",
    )
    focal.add_argument(
        "--calibration",
        type=Path,
        help="JSON: image_size [w,h], K 3x3, OpenCV distortion coefficients",
    )
    p.add_argument("--scale", type=float, default=0.25)
    p.add_argument(
        "--ref", type=int, help="1-based reference view; default central view"
    )
    p.add_argument("--direction", choices=["auto", "left", "right"], default="auto")
    p.add_argument(
        "--positions",
        type=Path,
        help="JSON signed x positions in mm, one per naturally sorted view; overrides direction/spacing",
    )
    p.add_argument(
        "--masks",
        type=Path,
        help="Reviewed per-view occluder masks, white=exclude; no green heuristic",
    )
    p.add_argument(
        "--mask-dilate",
        type=int,
        default=2,
        help="Expand excluded regions at processing resolution",
    )
    p.add_argument(
        "--no-refine",
        action="store_true",
        help="Disable small feature-based rail corrections",
    )
    p.add_argument(
        "--registration-gate",
        type=float,
        default=2.5,
        help="Maximum target motion residual in processing pixels",
    )
    p.add_argument("--min-views", type=int, default=3)
    p.add_argument(
        "--colour-tolerance",
        type=float,
        default=18,
        help="Photometric consensus RMS tolerance, 8-bit BGR",
    )
    p.add_argument(
        "--seed-views",
        type=int,
        default=0,
        help="0 tests every source view; positive values trade recovery for speed",
    )
    p.add_argument(
        "--canvas",
        choices=["union", "reference"],
        default="union",
        help="Union includes target-plane footprints beyond the central view",
    )
    p.add_argument("--tile-size", type=int, default=192)
    p.add_argument("--min-depth-margin", type=float, default=0.15)
    p.add_argument(
        "--min-baseline-mm",
        type=float,
        default=40,
        help="Minimum supporting camera span for the analysis mask",
    )
    args = p.parse_args(argv)
    numeric = [
        args.scale,
        args.spacing,
        args.leaf_depth,
        args.fruit_depth,
        *args.depth_range,
        args.colour_tolerance,
        args.registration_gate,
        args.min_depth_margin,
        args.min_baseline_mm,
    ]
    if not np.isfinite(numeric).all():
        p.error("Geometry and fusion settings must be finite")
    if not (
        0 < args.scale <= 1
        and args.spacing > 0
        and 0 < args.leaf_depth < args.depth_range[0] < args.depth_range[1]
        and args.depth_range[0] <= args.fruit_depth <= args.depth_range[1]
        and args.depth_steps >= 5
        and args.min_views >= 2
        and args.colour_tolerance > 0
        and args.seed_views >= 0
        and args.tile_size >= 8
        and args.mask_dilate >= 0
        and args.registration_gate > 0
        and args.min_depth_margin >= 0
        and args.min_baseline_mm >= 0
    ):
        p.error(
            "Invalid geometry or fusion settings: require leaf < near <= nominal <= far, >=5 depths, >=2 support views"
        )
    if args.fx is None and args.equiv_focal is None and args.calibration is None:
        p.error(
            "Supply --fx, --calibration, or an explicitly approximate --equiv-focal; no camera calibration is assumed"
        )
    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    if args.calibration:
        if not isinstance(calibration, dict):
            p.error("Calibration must be a JSON object")
        k = np.asarray(calibration.get("K", []), float)
        dimensions = np.asarray(calibration.get("image_size", []), float)
        distortion = np.asarray(calibration.get("distortion", [0, 0, 0, 0, 0]), float)
        if (
            k.shape != (3, 3)
            or not np.isfinite(k).all()
            or k[0, 0] <= 0
            or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1])
            or dimensions.shape != (2,)
            or not np.isfinite(dimensions).all()
            or np.any(dimensions < 1)
            or np.any(dimensions != np.floor(dimensions))
            or distortion.ndim != 1
            or len(distortion) not in (4, 5, 8, 12, 14)
            or not np.isfinite(distortion).all()
        ):
            p.error(
                "Calibration requires a valid 3x3 K, image_size [w,h], and OpenCV distortion coefficients"
            )
    if Path(args.images).is_dir() and Path(args.out).resolve().is_relative_to(
        Path(args.images).resolve()
    ):
        p.error("Keep reconstruction outputs outside the input image folder")
    names, images, allowed, size = load_sequence(
        args.images, args.scale, args.masks, args.mask_dilate, calibration
    )
    n = len(images)
    if not args.min_views <= n:
        p.error("--min-views exceeds image count")
    ref = (args.ref - 1) if args.ref is not None else n // 2
    if not 0 <= ref < n:
        p.error("--ref is outside sequence")
    fx = float(calibration["K"][0][0]) if calibration else args.fx
    source = "calibration" if calibration else "explicit fx"
    if args.equiv_focal is not None:
        # Full-frame-equivalent focal length normally describes diagonal FOV.
        # Account for the photo aspect ratio; the old long-side/36 shortcut
        # introduces an additional error for 4:3 photographs.
        fx = args.equiv_focal * float(np.hypot(*size)) / float(np.hypot(36.0, 24.0))
        source = "approximate 35mm equivalent; validate against measured parallax"
    if not np.isfinite(fx) or fx <= 0:
        p.error("Focal length must be positive and finite")
    if args.positions:
        positions = np.asarray(json.loads(args.positions.read_text()), float)
        if (
            positions.shape != (n,)
            or not np.isfinite(positions).all()
            or len(np.unique(positions)) != n
        ):
            p.error(
                "--positions must contain one unique finite signed camera x position per view"
            )
    else:
        sign = (
            estimate_direction(images)
            if args.direction == "auto"
            else (1 if args.direction == "right" else -1)
        )
        positions = np.arange(n, dtype=float) * args.spacing * sign
    geometry = Geometry(
        fx * images[0].shape[1] / size[0], positions, np.zeros((n, 2), np.float32), ref
    )
    registration = (
        []
        if args.no_refine
        else refine_translations(
            images, allowed, geometry, args.fruit_depth, args.registration_gate
        )
    )
    depths = np.linspace(*args.depth_range, args.depth_steps)
    bounds = canvas_bounds(images, geometry, depths, args.canvas == "union")
    print(
        f"{n} views, reference {ref + 1}, fx={fx:.1f}px ({source}); processing {images[0].shape[1]}x{images[0].shape[0]}",
        flush=True,
    )

    def progress(done, total):
        if done == 1 or done == total or done % 10 == 0:
            print(f"Fusion tile {done}/{total}", flush=True)

    result = reconstruct(
        images,
        allowed,
        geometry,
        depths,
        args.tile_size,
        args.min_views,
        args.colour_tolerance,
        args.seed_views,
        args.min_depth_margin,
        progress,
        bounds,
        args.min_baseline_mm,
    )
    reference = np.zeros_like(result["image"])
    ox, oy, _, _ = bounds
    rh, rw = images[ref].shape[:2]
    reference[-oy : rh - oy, -ox : rw - ox] = images[ref]
    diagnostics = {
        "parameters": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "views": names,
        "reference_1based": ref + 1,
        "original_size": size,
        "fx_original_px": fx,
        "fx_source": source,
        "positions_mm": positions.tolist(),
        "target_depths_mm": depths.tolist(),
        "registration": registration,
        "corrections_processing_px": geometry.corrections.tolist(),
        "canvas_origin_reference_processing_px": [ox, oy],
        "adjacent_nominal_leaf_target_parallax_px": geometry.fx
        * args.spacing
        * (1 / args.leaf_depth - 1 / args.fruit_depth),
        "limitations": [
            "Only surfaces observed in multiple views can be recovered.",
            "Parallel horizontal rail, static scene and locked camera intrinsics are assumed.",
            "Uniform leaves, branches and backgrounds can form false consensus; inspect masks and uncertainty.",
            "Depth/support are heuristic; no semantic grape identity or physical berry count is inferred.",
            "Union canvas includes projected view footprints, but edges may lack enough support.",
            "Exposure, specular highlights, leaf motion and imperfect geometry can leave holes or ghosts.",
        ],
    }
    save_outputs(args.out, result, reference, diagnostics)
    print(
        f"Saved {args.out}; supported={result['observed'].mean():.1%}, less ambiguous={result['reliable'].mean():.1%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
