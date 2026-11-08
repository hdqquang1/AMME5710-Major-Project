#!/usr/bin/env python3
"""Regenerate the report's experiments: branches x spacing, pixel and count metrics.

    python run_experiments.py --truth data/20261022_034827542_iOS.jpg \
        --truth-grapes results/refactor_check/truth/t1/grape_masks/t1.png \
        --labels labels/t1.csv --weights third_party/weights/point/best_val.pth

Each result is cached in its folder, so an interrupted run resumes. A failed
reconstruction is recorded as failed with its reason, never as 0%.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

import grape_count as count
import grape_evaluate as ev
from grape_io import EXTENSIONS, natural_key

STEPS = {20: 1, 40: 2, 60: 3}
BRANCHES = (3, 4, 5, 6)
SCALE = 0.8
# Tolerance is 12 px at scale 0.4, kept proportional to berry size.
TOLERANCE = 12 * SCALE / 0.4
# Chosen earlier on the 4-branch capture at this scale; secondary rows only.
HOUGH = {"canny_high": 100, "votes": 12}
SAM_COUNT = {"prompt": "cherry", "threshold": 0.1, "background": 0}


def subset_folder(images, out, reference, step):
    """Views at reference +/- k*step; returns the folder and the reference's 1-based position."""
    names = sorted(
        (p for p in Path(images).iterdir() if p.suffix.lower() in EXTENSIONS),
        key=lambda p: natural_key(p.name),
    )
    keep = [p for i, p in enumerate(names) if (i - reference) % step == 0]
    out.mkdir(parents=True, exist_ok=True)
    for p in keep:
        target = out / p.name
        if not target.exists():
            try:
                os.link(p, target)
            except OSError:
                shutil.copy2(p, target)
    return out, keep.index(names[reference]) + 1


def restore(images, out, spacing, ref=None, grape_masks=None, foliage=None, colour=None):
    if (out / "reference_fruit_mask.png").is_file():
        return None
    if (out / "FAILED.txt").is_file():
        return (out / "FAILED.txt").read_text(encoding="utf8")
    cmd = [sys.executable, "grape_restore_autoref.py", "--images", str(images), "--out", str(out),
           "--scale", str(SCALE), "--spacing", str(spacing)]
    if ref:
        cmd += ["--ref", str(ref)]
    if grape_masks:
        cmd += ["--grape-masks", str(grape_masks)]
    if foliage:
        cmd += ["--foliage-masks", str(foliage)]
    if colour:
        cmd += ["--colour", colour]
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode:
        reason = (run.stderr.strip().splitlines() or ["unknown error"])[-1][:400]
        out.mkdir(parents=True, exist_ok=True)
        (out / "FAILED.txt").write_text(reason, encoding="utf8")
        return reason
    return None


def register(truth, truth_fruit, reference, reference_fruit):
    """Fruit-plane fit, widening the feature region; rejects implausible geometry."""
    errors = []
    h, w = reference.shape[:2]
    for grow in (25, 60, 120, 240):
        try:
            matrix, info = ev.register_truth(truth, truth_fruit, reference, reference_fruit, grow=grow)
        except ValueError as exc:
            errors.append(f"grow {grow}: {exc}")
            continue
        roi = cv2.warpAffine(np.float32(truth_fruit), matrix, (w, h)) > 0.5
        inside = roi.sum() / max(truth_fruit.sum() * info["scale"] ** 2, 1)
        if not 0.95 * SCALE < info["scale"] < 1.05 * SCALE or abs(info["rotation_deg"]) > 3 or inside < 0.9:
            errors.append(f"grow {grow}: scale {info['scale']:.3f}, rotation {info['rotation_deg']:.2f}, inside {inside:.2f}")
            continue
        return matrix, {**info, "grow": grow, "fruit_inside_frame": float(inside)}
    raise ValueError("; ".join(errors))


def mask_metrics(pred, truth):
    inter = int((pred & truth).sum())
    return {
        "recall": inter / max(int(truth.sum()), 1),
        "precision": inter / max(int(pred.sum()), 1),
        "iou": inter / max(int((pred | truth).sum()), 1),
    }


def points(berries):
    return [(b["x_px"], b["y_px"]) for b in berries]


def score(name, berries, labels, row):
    m = ev.match_points(points(berries), labels, TOLERANCE)
    row.update({f"{name}_n": m["predicted_count"], f"{name}_err": m["count_error"], f"{name}_p": m["precision"],
                f"{name}_r": m["recall"], f"{name}_f1": m["f1"]})


def evaluate_run(out, truth, truth_fruit, matrix, labels, weights, row, secondary):
    image = cv2.imread(str(out / "grapes_rgb.png"))
    reference = cv2.imread(str(out / "reference.png"))
    observed = cv2.imread(str(out / "observed_mask.png"), 0) > 127
    reliable = cv2.imread(str(out / "analysis_mask.png"), 0) > 127
    floor_mask = cv2.imread(str(out / "reference_fruit_mask.png"), 0) > 127
    h, w = image.shape[:2]
    warped = cv2.warpAffine(truth, matrix, (w, h), flags=cv2.INTER_LINEAR)
    roi = cv2.warpAffine(np.float32(truth_fruit), matrix, (w, h)) > 0.5
    labels = [tuple(matrix[:, :2] @ np.array(p) + matrix[:, 2]) for p in labels]
    fidelity = ev.evaluate(image, warped, roi, observed, reliable, reference)
    row.update({"mae": fidelity["mae_on_supported_grapes"], "psnr": fidelity["psnr_db_on_supported_grapes"],
                "baseline_mae": fidelity.get("baseline_mae_on_same_supported_grapes")})
    for key, value in mask_metrics(floor_mask, roi).items():
        row[f"floor_{key}"] = value
    for key, value in mask_metrics(observed, roi).items():
        row[f"result_{key}"] = value
    restored = image.copy()
    restored[~reliable] = 0
    panels = {"floor": reference, "result": restored, "ceiling": warped}
    for name, im in panels.items():
        berries = count.count_grapesam(im, weights)
        score(f"gs_{name}", berries, labels, row)
        overlay = count.draw(im, berries)
        for x, y in labels:
            cv2.drawMarker(overlay, (round(x), round(y)), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
        cv2.imwrite(str(out / f"grapesam_{name}.jpg"), overlay)
    cv2.imwrite(str(out / "registration.jpg"), cv2.addWeighted(reference, 0.5, warped, 0.5, 0))
    if secondary:
        score("sam3_result", count.count_sam3(image, reliable, **SAM_COUNT)[0], labels, row)
        score("hough_result", count.count_hough(image, reliable, **HOUGH)[0], labels, row)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--truth", required=True, type=Path)
    p.add_argument("--truth-grapes", required=True, type=Path)
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--out", default=Path("results/report"), type=Path)
    args = p.parse_args()
    truth = cv2.imread(str(args.truth))
    truth_fruit = cv2.imread(str(args.truth_grapes), 0) > 127
    labels = ev.read_points(args.labels)
    rows = []
    args.out.mkdir(parents=True, exist_ok=True)
    for branches in BRANCHES:
        capture = Path(f"data/2cm_74cm_{branches}branches")
        foliage, grapes = capture / "removal_masks", capture / "grape_masks"
        base = args.out / f"b{branches}_20mm"
        print(f"== {branches} branches", flush=True)
        failed = restore(capture / "images", base, 20, grape_masks=grapes, foliage=foliage)
        reference, matrix, reg = None, None, {}
        if failed:
            print("  20 mm failed:", failed, flush=True)
        else:
            meta = json.loads((base / "diagnostics.json").read_text(encoding="utf8"))
            reference = meta["reference_1based"] - 1
            try:
                ref_image = cv2.imread(str(base / "reference.png"))
                ref_fruit = (cv2.imread(str(base / "reference_fruit_mask.png"), 0) > 127) | (cv2.imread(str(base / "observed_mask.png"), 0) > 127)
                matrix, reg = register(truth, truth_fruit, ref_image, ref_fruit)
            except ValueError as exc:
                print("  registration failed:", exc, flush=True)
                reg = {"failed": str(exc)}
        configurations = [(mm, "sam") for mm in STEPS] + [(20, "colour")]
        for mm, kind in configurations:
            row = {"branches": branches, "spacing_mm": mm, "masks": kind, "status": "ok", **{f"reg_{k}": v for k, v in reg.items()}}
            rows.append(row)
            out = args.out / (f"b{branches}_{mm}mm" + ("" if kind == "sam" else "_colour"))
            try:
                if failed and not (mm == 20 and kind == "sam"):
                    row["status"] = "not run: no reference from the 20 mm run"
                    continue
                if mm == 20 and kind == "sam":
                    if failed:
                        row["status"] = f"failed: {failed}"
                        continue
                else:
                    folder, ref = subset_folder(capture / "images", args.out / "subsets" / f"b{branches}_{mm}", reference, STEPS[mm])
                    why = restore(folder, out, mm, ref, grapes if kind == "sam" else None, foliage, None if kind == "sam" else "red")
                    if why:
                        row["status"] = f"failed: {why}"
                        print(f"  {mm} mm {kind} failed: {why}", flush=True)
                        continue
                meta = json.loads((out / "diagnostics.json").read_text(encoding="utf8"))
                row.update(views=len(meta["registration"]["connected_views_1based"]),
                           aligned_error_px=meta["registration"]["held_out_median_error_px"])
                if matrix is None:
                    row["status"] = "reconstructed; truth registration failed, no metrics"
                    continue
                evaluate_run(out, truth, truth_fruit, matrix, labels, args.weights, row, mm == 20 and kind == "sam")
                print(f"  {mm} mm {kind}: pixel recall {row['floor_recall']:.2f} -> {row['result_recall']:.2f}; "
                      f"GrapeSAM F1 floor {row['gs_floor_f1']:.2f} result {row['gs_result_f1']:.2f} ceiling {row['gs_ceiling_f1']:.2f}", flush=True)
            except Exception:
                row["status"] = "error: " + traceback.format_exc(limit=1).strip().splitlines()[-1][:300]
                print(f"  {mm} mm {kind}", row["status"], flush=True)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("branches", "spacing_mm", "masks", "status"), k))
    with (args.out / "results.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "manifest.json").write_text(json.dumps({
        "scale": SCALE, "tolerance_px": TOLERANCE, "truth": str(args.truth), "labels": str(args.labels),
        "weights": str(args.weights), "hough": HOUGH, "sam_count": SAM_COUNT,
    }, indent=2) + "\n", encoding="utf8")
    print("done", flush=True)


if __name__ == "__main__":
    main()
