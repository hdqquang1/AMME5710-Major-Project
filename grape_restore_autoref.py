#!/usr/bin/env python3
"""Measured multi-view restoration of grapes behind foliage on a camera rail.

Fruit comes from colour thresholds (red, green or both), SAM 3, or saved masks.
Only observed source colours are exported; invisible surfaces are never filled.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from grape_io import load_sequence
from grape_segment import (
    COLOUR_THRESHOLDS,
    colour_mask,
    load_grape_masks,
    segment_folder_sam3,
)


@dataclass
class MatchEdge:
    i: int
    j: int
    source: np.ndarray
    target: np.ndarray
    check_source: np.ndarray
    check_target: np.ndarray
    summary: dict


def green_occluder_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(
        hsv, np.array([25, 30, 18], np.uint8), np.array([105, 255, 255], np.uint8)
    )
    return cv2.dilate(green, np.ones((5, 5), np.uint8)) > 0


def fit_pair(i, j, features, threshold=1.5, min_inliers=12):
    ka, da = features[i]
    kb, db = features[j]
    summary = {"pair_1based": [i + 1, j + 1], "accepted": False}
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return None, {**summary, "reason": "insufficient target descriptors"}
    matcher = cv2.BFMatcher()

    # Reciprocal ratio matches help reject similar-looking berries.
    def ratios(a, b):
        return {
            p[0].queryIdx: p[0]
            for p in matcher.knnMatch(a, b, k=2)
            if len(p) == 2 and p[0].distance < 0.78 * p[1].distance
        }

    forward = ratios(da, db)
    reverse = ratios(db, da)
    matches = [
        m
        for m in forward.values()
        if m.trainIdx in reverse and reverse[m.trainIdx].trainIdx == m.queryIdx
    ]
    summary["reciprocal_matches"] = len(matches)
    if len(matches) < min_inliers + 4:
        return None, {**summary, "reason": "insufficient reciprocal target matches"}
    a = np.float32([ka[m.queryIdx].pt for m in matches])
    b = np.float32([kb[m.trainIdx].pt for m in matches])
    # Hold these correspondences out of fitting the edge and the pose graph.
    order = np.random.default_rng(1009 * i + 9176 * j + 17).permutation(len(a))
    check = order[::4]
    train = np.setdiff1d(order, check)
    cv2.setRNGSeed(1009 * i + 9176 * j + 17)
    matrix, inliers = cv2.estimateAffinePartial2D(
        a[train],
        b[train],
        method=cv2.RANSAC,
        ransacReprojThreshold=threshold,
        maxIters=10000,
        confidence=0.999,
    )
    if matrix is None:
        return None, {**summary, "reason": "target RANSAC failed"}
    keep = inliers.ravel() > 0
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    rotation = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    error = np.linalg.norm(a[check] @ matrix[:, :2].T + matrix[:, 2] - b[check], axis=1)
    checked = float(np.mean(error <= threshold * 2))
    summary.update(
        train_inliers=int(keep.sum()),
        train_inlier_fraction=float(keep.mean()),
        held_out_matches=len(check),
        held_out_median_error_px=float(np.median(error)),
        held_out_fraction_within_threshold=checked,
        scale=scale,
        rotation_deg=rotation,
    )
    if (
        keep.sum() < min_inliers
        or keep.mean() < 0.5
        or checked < 0.5
        or not 0.94 < scale < 1.06
        or abs(rotation) > 3
    ):
        return None, {**summary, "reason": "target motion failed validation"}
    aa = a[train][keep]
    bb = b[train][keep]
    if np.linalg.norm(np.ptp(aa, axis=0)) < 25:
        return None, {
            **summary,
            "reason": "target correspondences occupy too small a region",
        }
    summary["accepted"] = True
    return MatchEdge(i, j, aa, bb, a[check], b[check], summary), summary


def build_edges(images, targets, neighbours=4, min_inliers=12, progress=None):
    sift = cv2.SIFT_create(nfeatures=14000, contrastThreshold=0.01)
    features = []
    for i, (image, target) in enumerate(zip(images, targets)):
        feature_mask = cv2.dilate(np.uint8(target) * 255, np.ones((3, 3), np.uint8))
        features.append(
            sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), feature_mask)
        )
        if progress:
            progress(f"Target features {i + 1}/{len(images)}: {len(features[-1][0])}")
    edges = []
    summaries = []
    for i in range(len(images)):
        for j in range(i + 1, min(len(images), i + neighbours + 1)):
            edge, summary = fit_pair(i, j, features, min_inliers=min_inliers)
            summaries.append(summary)
            if edge is not None:
                edges.append(edge)
    return edges, summaries


def validated_components(n, edges):
    """Groups connected by ACCEPTED RANSAC matches, including isolated images."""
    adjacency = {i: set() for i in range(n)}
    for edge in edges:
        adjacency[edge.i].add(edge.j)
        adjacency[edge.j].add(edge.i)
    unseen = set(range(n))
    groups = []
    while unseen:
        start = min(unseen)
        todo = [start]
        unseen.remove(start)
        group = []
        while todo:
            i = todo.pop()
            group.append(i)
            neighbours = sorted(adjacency[i] & unseen)
            todo.extend(neighbours)
            unseen.difference_update(neighbours)
        groups.append(sorted(group))
    groups.sort(key=lambda g: (-len(g), g[0]))
    return groups


def save_matching_diagnostics(out, images, targets, names, edges, pair_records, groups):
    """Write useful evidence even if global registration later fails."""
    out.mkdir(parents=True, exist_ok=True)
    report = {
        'images': names,
        'accepted_pairs': len(edges),
        'attempted_pairs': len(pair_records),
        'connected_groups_1based': [[i + 1 for i in group] for group in groups],
        'pair_diagnostics': pair_records,
        'note': 'Groups represent accepted feature matches, NOT necessarily a successful global alignment.',
    }
    (out / 'matching_diagnostics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    thumb_w, thumb_h, cols = 260, 205, 7
    rows = (len(images) + cols - 1) // cols
    sheet = np.full((rows * thumb_h, cols * thumb_w, 3), 238, np.uint8)
    for i, (img, mask) in enumerate(zip(images, targets)):
        h, w = img.shape[:2]
        view = cv2.resize(img, (thumb_w, thumb_h - 31), interpolation=cv2.INTER_AREA)
        highlighted = cv2.resize(mask.astype(np.uint8), (thumb_w, thumb_h - 31), interpolation=cv2.INTER_NEAREST) > 0
        view[highlighted] = np.uint8(0.45 * view[highlighted] + 0.55 * np.array([255, 210, 0]))
        y, x = divmod(i, cols)
        x *= thumb_w
        y *= thumb_h
        sheet[y:y + thumb_h - 31, x:x + thumb_w] = view
        coverage = 100 * float(mask.mean())
        cv2.putText(sheet, f'View {i + 1}: {coverage:.1f}% mask',
                    (x + 7, y + thumb_h - 9), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (30, 35, 40), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(out / 'debug_colour_masks_contact_sheet.jpg'), sheet):
        raise OSError('Could not save diagnostic contact sheet')


def select_verified_reference(n, edges, images, min_views, max_error, groups):
    """Try reference images in the strongest connected groups without relaxing RANSAC."""
    errors = []
    for group in groups:
        if len(group) < max(3, min_views):
            continue
        centre = (group[0] + group[-1]) / 2
        strength = {i: 0 for i in group}
        degree = {i: 0 for i in group}
        for edge in edges:
            if edge.i in strength and edge.j in strength:
                strength[edge.i] += len(edge.source)
                strength[edge.j] += len(edge.source)
                degree[edge.i] += 1
                degree[edge.j] += 1
        candidates = sorted(group, key=lambda i: (-degree[i], -strength[i], abs(i - centre)))
        print('Trying reference images in group:', [i + 1 for i in group], flush=True)
        for ref in candidates:
            try:
                poses, active, validation = solve_motion(
                    n, edges, ref, images[0].shape[:2], max_error
                )
                if len(active) >= min_views:
                    return ref, poses, active, validation
                errors.append({'reference': ref + 1, 'reason': f'Only {len(active)} views survived'})
            except (ValueError, np.linalg.LinAlgError) as exc:
                errors.append({'reference': ref + 1, 'reason': str(exc)})
    tried = ', '.join(str(i['reference']) for i in errors)
    raise ValueError(
        'No connected group could pass global alignment validation. '
        f'Tried references: {tried}. Review matching_diagnostics.json and '
        'debug_colour_masks_contact_sheet.jpg; incorrect masks or target parallax '
        'may require revised segmentation or camera alignment.'
    )


def connected_views(n, edges, reference):
    found = {reference}
    while True:
        previous = len(found)
        for edge in edges:
            if edge.i in found or edge.j in found:
                found.update([edge.i, edge.j])
        if len(found) == previous:
            return sorted(found)


def project(points, matrix):
    return points @ matrix[:2, :2].T + matrix[:2, 2]


def solve_motion(n, edges, reference, shape, max_error=3.0):
    """Robust joint similarity fit, with the reference pose fixed to identity."""
    active = connected_views(n, edges, reference)
    if len(active) < 3:
        raise ValueError(
            "Fewer than three views have validated target alignment; no reconstruction written"
        )
    usable = [e for e in edges if e.i in active and e.j in active]
    slots = {i: k for k, i in enumerate(i for i in active if i != reference)}
    h, w = shape
    normalizer = max(h, w)
    center = np.array([w / 2, h / 2], float)
    rows = []
    rhs = []
    edge_weights = []
    for edge in usable:
        # Limit the influence of an unusually feature-rich pair.
        indexes = (
            np.linspace(0, len(edge.source) - 1, min(80, len(edge.source)))
            .round()
            .astype(int)
        )
        pa = (edge.source[indexes] - center) / normalizer
        pb = (edge.target[indexes] - center) / normalizer
        for a, b in zip(pa, pb):
            block = np.zeros((2, 4 * len(slots)), float)
            value = np.zeros(2)
            for i, point, sign in [(edge.i, a, 1), (edge.j, b, -1)]:
                x, y = point
                coefficients = np.array([[x, -y, 1, 0], [y, x, 0, 1]]) * sign
                if i == reference:
                    value -= coefficients[:, 0]  # Reference [a,b,tx,ty]=[1,0,0,0].
                else:
                    block[:, 4 * slots[i] : 4 * slots[i] + 4] = coefficients
            rows.extend(block)
            rhs.extend(value)
            edge_weights.append(1 / np.sqrt(len(indexes)))
    a = np.asarray(rows)
    b = np.asarray(rhs)
    base = np.repeat(np.asarray(edge_weights), 2)
    weights = base.copy()
    for _ in range(6):
        solution = np.linalg.lstsq(a * weights[:, None], b * weights, rcond=None)[0]
        errors = np.linalg.norm((a @ solution - b).reshape(-1, 2), axis=1) * normalizer
        weights = base * np.repeat(
            np.sqrt(np.minimum(1, 1.5 / np.maximum(errors, 1e-6))), 2
        )
    poses = [None] * n
    poses[reference] = np.eye(3)
    for i, k in slots.items():
        aa, bb, tx, ty = solution[4 * k : 4 * k + 4]
        linear = np.array([[aa, -bb], [bb, aa]])
        matrix = np.eye(3)
        matrix[:2, :2] = linear
        matrix[:2, 2] = center + normalizer * np.array([tx, ty]) - linear @ center
        if not np.isfinite(matrix).all() or not 0.85 < np.hypot(aa, bb) < 1.15:
            raise ValueError(
                f"Joint target fit produced an implausible pose for view {i + 1}"
            )
        poses[i] = matrix
    checks = []
    per_view = {i: [] for i in active}
    for edge in usable:
        errors = np.linalg.norm(
            project(edge.check_source, poses[edge.i])
            - project(edge.check_target, poses[edge.j]),
            axis=1,
        )
        checks.extend(errors.tolist())
        per_view[edge.i].extend(errors.tolist())
        per_view[edge.j].extend(errors.tolist())
    validation = [
        {
            "view_1based": i + 1,
            "correspondences": len(per_view[i]),
            "median_error_px": float(np.median(per_view[i])) if per_view[i] else None,
        }
        for i in active
    ]
    rejected = {
        i
        for i in active
        if i != reference and (not per_view[i] or np.median(per_view[i]) > max_error)
    }
    active = connected_views(
        n, [e for e in usable if e.i not in rejected and e.j not in rejected], reference
    )
    if len(active) < 3:
        raise ValueError("Fewer than three views pass joint alignment validation")
    for i in range(n):
        if i not in active:
            poses[i] = None
    return (
        poses,
        active,
        {
            "connected_views_1based": [i + 1 for i in active],
            "excluded_views_1based": [i + 1 for i in range(n) if i not in active],
            "held_out_correspondences": len(checks),
            "held_out_median_error_px": float(np.median(checks)),
            "held_out_fraction_within_3px": float(np.mean(np.asarray(checks) <= 3)),
            "per_view_validation": validation,
            "max_view_median_error_px": max_error,
            "note": "Held-out feature transfer errors are alignment diagnostics, not labelled grape segmentation accuracy.",
        },
    )


def restoration_bounds(images, poses, union=True):
    h, w = images[0].shape[:2]
    if not union:
        return (0, 0, w, h)
    corners = np.float64([[0, 0], [w, 0], [w, h], [0, h]])
    points = np.vstack([project(corners, p) for p in poses if p is not None])
    lo = np.floor(points.min(axis=0)).astype(int)
    hi = np.ceil(points.max(axis=0)).astype(int)
    return tuple(int(v) for v in (*lo, *hi))


def fuse_tile(
    colours, valid, targets, indices, reference, min_views, spacing, min_baseline,
    rms=30, select="best",
):
    """Robust colour support, followed by selection of an observed source pixel."""
    n, h, w = valid.shape
    eligible = valid & targets
    count = eligible.sum(axis=0).astype(np.uint16)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(
            np.where(eligible[..., None], colours.astype(np.float32), np.nan), axis=0
        )
    median = np.nan_to_num(median)
    distance = np.sqrt(np.mean((colours.astype(np.float32) - median) ** 2, axis=-1))
    inliers = eligible & (distance <= rms)
    support = inliers.sum(axis=0).astype(np.uint16)
    span = np.max(np.where(inliers, indices[:, None, None], -np.inf), axis=0) - np.min(
        np.where(inliers, indices[:, None, None], np.inf), axis=0
    )
    span = np.where(support > 0, span * spacing, 0).astype(np.float32)
    # Smooth the selection cost, not grape colours. This reduces pixel-level
    # switching while preserving actual source texture and provenance.
    costs = []
    for k in range(n):
        errors = cv2.GaussianBlur(
            np.where(eligible[k], distance[k], min(rms, 255)).astype(np.float32), (5, 5), 0.9
        )
        boundary = np.minimum(
            cv2.distanceTransform(inliers[k].astype(np.uint8), cv2.DIST_L2, 3), 4
        )
        cost = (
            errors / min(rms, 255)
            + 0.15 / (1 + boundary)
            + 0.025 * abs(int(indices[k]) - reference) / max(1, len(indices) - 1)
        )
        costs.append(np.where(inliers[k], cost, np.inf))
    choice = np.argmin(costs, axis=0)
    yy, xx = np.mgrid[:h, :w]
    selected = colours[choice, yy, xx].copy()
    if select == "median":  # Ablation only: blends colours instead of copying one view.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            blend = np.nanmedian(np.where(inliers[..., None], colours.astype(np.float32), np.nan), axis=0)
        selected = np.uint8(np.clip(np.nan_to_num(blend), 0, 255))
    observed = (support >= min_views) & (span >= min_baseline)
    disagreement = (distance * inliers).sum(axis=0) / np.maximum(support, 1)
    reliable = observed & (disagreement <= 18)
    selected[~observed] = 0
    return {
        "image": selected,
        "observed": observed,
        "reliable": reliable,
        "support": support,
        "candidate_support": count,
        "source": np.where(observed, indices[choice] + 1, 0).astype(np.uint16),
        "span": span,
        "disagreement": disagreement.astype(np.float32),
    }


def fuse_views(
    images,
    allowed,
    targets,
    poses,
    reference,
    bounds,
    tile_size=192,
    min_views=3,
    spacing=20,
    min_baseline=40,
    progress=None,
    rms=30,
    select="best",
):
    ox, oy, x1, y1 = bounds
    h, w = y1 - oy, x1 - ox
    indices = np.array([i for i, p in enumerate(poses) if p is not None])
    output = {
        "image": np.zeros((h, w, 3), np.uint8),
        "observed": np.zeros((h, w), bool),
        "reliable": np.zeros((h, w), bool),
        "support": np.zeros((h, w), np.uint16),
        "candidate_support": np.zeros((h, w), np.uint16),
        "source": np.zeros((h, w), np.uint16),
        "span": np.zeros((h, w), np.float32),
        "disagreement": np.zeros((h, w), np.float32),
        "masked_mean": np.zeros((h, w, 3), np.uint8),
        "unmasked_mean": np.zeros((h, w, 3), np.uint8),
    }
    inverse = [np.linalg.inv(poses[i]) for i in indices]
    clear_sources = [~green_occluder_mask(images[i]) for i in indices]
    source_h, source_w = images[0].shape[:2]
    total = ((h + tile_size - 1) // tile_size) * ((w + tile_size - 1) // tile_size)
    done = 0
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            ex, ey = min(x + tile_size, w), min(y + tile_size, h)
            bx, by, px, py = (
                max(0, x - 4),
                max(0, y - 4),
                min(w, ex + 4),
                min(h, ey + 4),
            )
            yy, xx = np.mgrid[by + oy : py + oy, bx + ox : px + ox].astype(np.float32)
            colours = []
            valid = []
            votes = []
            clear = []
            for i, inv, source_clear in zip(indices, inverse, clear_sources):
                mx = (inv[0, 0] * xx + inv[0, 1] * yy + inv[0, 2]).astype(np.float32)
                my = (inv[1, 0] * xx + inv[1, 1] * yy + inv[1, 2]).astype(np.float32)
                if inv[2, 0] or inv[2, 1]:  # Projective pose (homography ablation).
                    wz = (inv[2, 0] * xx + inv[2, 1] * yy + inv[2, 2]).astype(np.float32)
                    mx, my = mx / wz, my / wz
                colours.append(cv2.remap(images[i], mx, my, cv2.INTER_LINEAR))
                v = (
                    (
                        cv2.remap(
                            allowed[i].astype(np.float32), mx, my, cv2.INTER_LINEAR
                        )
                        > 0.999
                    )
                    & (mx >= 0)
                    & (mx < source_w - 1)
                    & (my >= 0)
                    & (my < source_h - 1)
                )
                valid.append(v)
                votes.append(
                    cv2.remap(targets[i].astype(np.float32), mx, my, cv2.INTER_LINEAR)
                    > 0.999
                )
                # Green masks are used only for the labelled baseline, not as
                # evidence that a grape is present in the main output.
                clear.append(
                    cv2.remap(source_clear.astype(np.float32), mx, my, cv2.INTER_LINEAR)
                    > 0.999
                )
            colours = np.asarray(colours)
            valid = np.asarray(valid)
            votes = np.asarray(votes)
            clear = np.asarray(clear) & valid
            result = fuse_tile(
                colours,
                valid,
                votes,
                indices,
                reference,
                min_views,
                spacing,
                min_baseline,
                rms,
                select,
            )
            crop = np.s_[y - by : ey - by, x - bx : ex - bx]
            dest = np.s_[y:ey, x:ex]
            for key, value in result.items():
                output[key][dest] = value[crop]
            for key, mask in [("unmasked_mean", valid), ("masked_mean", clear)]:
                count = mask.sum(axis=0)
                mean = (colours.astype(np.float32) * mask[..., None]).sum(
                    axis=0
                ) / np.maximum(count[..., None], 1)
                mean[count < min_views] = 0
                output[key][dest] = np.uint8(np.clip(np.rint(mean[crop]), 0, 255))
            done += 1
            if progress and (done == 1 or done % 10 == 0 or done == total):
                progress(f"Fusion tile {done}/{total}")
    return output


def clean_small_regions(result, min_area):
    """Suppress isolated colour specks; do not close holes or generate pixels."""
    if min_area <= 1:
        return
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.uint8(result["observed"]), 8
    )
    sizes = stats[:, cv2.CC_STAT_AREA]
    keep = sizes >= min_area
    keep[0] = False
    supported = keep[labels]
    result["observed"] &= supported
    result["reliable"] &= supported
    result["image"][~supported] = 0
    result["source"][~supported] = 0


def export_result(out, result, reference, bounds, metadata):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    def write(name, image):
        if not cv2.imwrite(str(out / name), image):
            raise OSError(f"Cannot write {out / name}")

    write("reference.png", reference)
    write("grapes_rgb.png", result["image"])
    write(
        "grapes_rgba.png",
        np.dstack([result["image"], np.uint8(result["observed"]) * 255]),
    )
    write(
        "analysis_ready_rgba.png",
        np.dstack([result["image"], np.uint8(result["reliable"]) * 255]),
    )
    write("observed_mask.png", np.uint8(result["observed"]) * 255)
    write("analysis_mask.png", np.uint8(result["reliable"]) * 255)
    write("support_count.png", result["support"])
    write("source_view.png", result["source"])
    write("aligned_mean.png", result["unmasked_mean"])
    write("leafmasked_mean.png", result["masked_mean"])
    np.savez_compressed(
        out / "evidence.npz",
        **{
            k: v
            for k, v in result.items()
            if k not in ["image", "masked_mean", "unmasked_mean"]
        },
    )
    mask = result["observed"]
    if mask.any():
        yy, xx = np.where(mask)
        x0, y0 = max(0, xx.min() - 15), max(0, yy.min() - 15)
        x1, y1 = min(mask.shape[1], xx.max() + 16), min(mask.shape[0], yy.max() + 16)
        crop = np.s_[y0:y1, x0:x1]
        write(
            "grapes_closeup.png",
            np.dstack([result["image"][crop], np.uint8(mask[crop]) * 255]),
        )
        panels = []
        for title, image in [
            ("Reference", reference),
            ("Measured alignment, green-masked mean", result["masked_mean"]),
            ("Observed grape pixels, selected sources", result["image"]),
        ]:
            panel = image[crop]
            canvas = np.full((panel.shape[0] + 32, panel.shape[1], 3), 245, np.uint8)
            canvas[32:] = panel
            cv2.putText(
                canvas,
                title,
                (8, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            panels.append(canvas)
        write("comparison.jpg", np.vstack(panels))
        metadata["closeup_bounds_canvas_px"] = [int(x0), int(y0), int(x1), int(y1)]
    metadata.update(
        supported_target_pixels=int(mask.sum()),
        reliable_target_pixels=int(result["reliable"].sum()),
        canvas_origin_reference_processing_px=list(bounds[:2]),
        note="Target pixels are fruit candidates supported by multiple aligned views, not verified grape instances or berry counts.",
    )
    (out / "diagnostics.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", required=True, type=Path)
    p.add_argument("--out", default=Path("grape_results_measured"), type=Path)
    p.add_argument("--scale", type=float, default=0.4)
    p.add_argument("--ref", type=int, help="1-based reference; default automatically selects from validated connected groups")
    p.add_argument(
        "--target-roi",
        type=float,
        nargs=4,
        default=(0.03, 0.34, 0.97, 0.60),
        help="Source-image fractions; default excludes red cardboard tape in the Drive scene",
    )
    p.add_argument(
        "--occluder-depth-range",
        type=float,
        nargs=2,
        default=(740, 800),
        help="Approximate mm; recorded as scene context, not a calibrated per-pixel depth estimate",
    )
    p.add_argument("--target-depth-range", type=float, nargs=2, default=(1100, 1200))
    p.add_argument("--spacing", type=float, default=20)
    p.add_argument("--neighbours", type=int, default=4)
    p.add_argument("--min-inliers", type=int, default=12)
    p.add_argument("--min-views", type=int, default=3)
    p.add_argument("--min-baseline-mm", type=float, default=40)
    p.add_argument("--tile-size", type=int, default=192)
    p.add_argument(
        "--min-region-area",
        type=int,
        default=25,
        help="Minimum connected target area in processing pixels; 0 disables speck cleanup",
    )
    p.add_argument(
        "--max-registration-error",
        type=float,
        default=3,
        help="Maximum per-view median validation error in processing pixels",
    )
    p.add_argument("--canvas", choices=["reference", "union"], default="reference")
    p.add_argument(
        "--segment",
        choices=["colour", "sam3"],
        default="colour",
        help="How to find fruit in each view; ignored when --grape-masks is given",
    )
    p.add_argument(
        "--colour",
        choices=["red", "green", "both"],
        default="red",
        help="Fruit colour for --segment colour; choose per capture",
    )
    p.add_argument(
        "--fruit-prompt", default="grape", help="SAM 3 text prompt for --segment sam3"
    )
    p.add_argument(
        "--grape-masks",
        type=Path,
        help="Saved original-size white=grape masks; overrides --segment",
    )
    p.add_argument(
        "--foliage-masks",
        type=Path,
        help="Saved original-size white=foliage masks, named <stem>.png or <stem>_mask.png",
    )
    args = p.parse_args(argv)
    if not np.isfinite(
        [
            args.scale,
            args.spacing,
            args.min_baseline_mm,
            args.max_registration_error,
            *args.target_roi,
            *args.occluder_depth_range,
            *args.target_depth_range,
        ]
    ).all():
        p.error("Geometry, ROI and registration settings must be finite")
    if not (
        0 < args.scale <= 1
        and args.spacing > 0
        and args.neighbours >= 1
        and args.min_inliers >= 6
        and args.min_views >= 2
        and args.min_baseline_mm >= 0
        and args.tile_size >= 16
        and args.min_region_area >= 0
        and args.max_registration_error > 0
        and 0
        < args.occluder_depth_range[0]
        <= args.occluder_depth_range[1]
        < args.target_depth_range[0]
        <= args.target_depth_range[1]
    ):
        p.error("Invalid scale, support, or scene geometry settings")
    if args.images.is_dir() and args.out.resolve().is_relative_to(
        args.images.resolve()
    ):
        p.error("Keep results outside the input image folder")
    progress = lambda value: print(value, flush=True)
    grape_dir, foliage_dir = args.grape_masks, args.foliage_masks
    if grape_dir is None and args.segment == "sam3":
        progress("Segmenting every view with SAM 3...")
        grape_dir, generated = segment_folder_sam3(
            args.images, args.out, args.fruit_prompt, foliage=foliage_dir is None
        )
        foliage_dir = foliage_dir or generated
    progress("Loading original images (EXIF orientation applied)...")
    names, images, allowed, size = load_sequence(
        args.images, args.scale, foliage_dir, mask_dilate=0
    )
    if args.min_views > len(images):
        p.error("Minimum support exceeds image count")
    reference = args.ref - 1 if args.ref is not None else None
    if reference is not None and not 0 <= reference < len(images):
        p.error("Reference outside image sequence")
    targets = (
        load_grape_masks(grape_dir, names, size, images[0].shape[:2])
        if grape_dir
        else [colour_mask(im, args.target_roi, args.colour) for im in images]
    )
    targets = [t & v for t, v in zip(targets, allowed)]
    edges, summaries = build_edges(
        images, targets, args.neighbours, args.min_inliers, progress
    )
    groups = validated_components(len(images), edges)
    progress(f"Validated target edges: {len(edges)}/{len(summaries)}")
    progress("Connected groups (1-based view numbers): " + str([[i + 1 for i in g] for g in groups]))
    save_matching_diagnostics(args.out, images, targets, names, edges, summaries, groups)
    progress(f"Saved diagnostic masks and pair graph to: {args.out}")
    if reference is None:
        reference, poses, active, registration = select_verified_reference(
            len(images), edges, images, args.min_views, args.max_registration_error, groups
        )
        progress(f"Automatically selected reference view: {reference + 1}")
    else:
        progress(f"Using manually specified reference view: {reference + 1}")
        poses, active, registration = solve_motion(
            len(images), edges, reference, images[0].shape[:2], args.max_registration_error
        )
    if len(active) < args.min_views:
        p.error("Too few validated target-aligned views")
    progress(
        f"Connected views: {len(active)}/{len(images)}; held-out median error: {registration['held_out_median_error_px']:.3f} processing px"
    )
    if registration["excluded_views_1based"]:
        progress(f"Excluded unvalidated views: {registration['excluded_views_1based']}")
    bounds = restoration_bounds(images, poses, args.canvas == "union")
    result = fuse_views(
        images,
        allowed,
        targets,
        poses,
        reference,
        bounds,
        args.tile_size,
        args.min_views,
        args.spacing,
        args.min_baseline_mm,
        progress,
    )
    clean_small_regions(result, args.min_region_area)
    if not result["observed"].any():
        raise ValueError(
            "No target regions meet the support criteria. Review target masks/ROI, alignment and minimum support before analysis."
        )
    ox, oy, x1, y1 = bounds
    ref_canvas = np.zeros((y1 - oy, x1 - ox, 3), np.uint8)
    h, w = images[0].shape[:2]
    ref_canvas[-oy : h - oy, -ox : w - ox] = images[reference]
    ref_fruit = np.zeros((y1 - oy, x1 - ox), np.uint8)
    ref_fruit[-oy : h - oy, -ox : w - ox] = np.uint8(targets[reference]) * 255
    metadata = {
        "method": "target-feature pose graph and observed-source selection",
        "parameters": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "views": names,
        "original_size": list(size),
        "processing_size": [w, h],
        "reference_1based": reference + 1,
        "segmentation": {
            "fruit": "saved masks"
            if args.grape_masks
            else ("sam3" if grape_dir else f"colour: {args.colour}"),
            "fruit_prompt": args.fruit_prompt
            if grape_dir and not args.grape_masks
            else None,
            "foliage": str(foliage_dir) if foliage_dir else None,
            "colour_thresholds": None if grape_dir else COLOUR_THRESHOLDS,
        },
        "registration": registration,
        "pair_diagnostics": summaries,
        "source_to_reference_matrices": [
            m.tolist() if m is not None else None for m in poses
        ],
        "limitations": [
            "Fruit masks, colour thresholds and the ROI are scene-specific priors and need review.",
            "The 2D target model approximates a range of grape depths; alignment can still be imperfect.",
            "Neighbouring-view graph consistency does not establish true camera calibration or metric depth.",
            "A surface must have sufficient clear observations; unseen surfaces and backs of berries remain unknown.",
            "No model was trained for grape identity, ripeness, health, or berry counting.",
        ],
    }
    export_result(args.out, result, ref_canvas, bounds, metadata)
    cv2.imwrite(str(args.out / "reference_fruit_mask.png"), ref_fruit)
    progress(
        f"Saved {args.out}; supported candidate pixels: {int(result['observed'].sum())}"
    )


if __name__ == "__main__":
    main()
