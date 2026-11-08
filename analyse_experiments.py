#!/usr/bin/env python3
"""Add like-for-like count scores (fruit-only inputs) and review sheets to results/report."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np

import grape_count as count
import grape_evaluate as ev
from run_experiments import TOLERANCE, register

REPORT = Path("results/report")
WEIGHTS = "third_party/weights/point/best_val.pth"
truth = cv2.imread("data/20261022_034827542_iOS.jpg")
truth_fruit = cv2.imread("results/refactor_check/truth/t1/grape_masks/t1.png", 0) > 127
labels0 = ev.read_points("labels/t1.csv")
rows = list(csv.DictReader((REPORT / "results.csv").open(encoding="utf8")))
matrices = {}
sheet_reg, sheet_count = [], []


def crop(im, m):
    ys, xs = np.where(m)
    return im[max(ys.min() - 60, 0) : ys.max() + 60, max(xs.min() - 60, 0) : xs.max() + 60]


def tile(im, text, width=620):
    im = cv2.resize(im, (width, round(im.shape[0] * width / im.shape[1])), interpolation=cv2.INTER_AREA)
    im = cv2.copyMakeBorder(im, 34, 6, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(im, text, (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return im


def grid(tiles, cols, path):
    h = max(t.shape[0] for t in tiles)
    w = tiles[0].shape[1]
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    tiles += [np.full((h, w, 3), 255, np.uint8)] * (-len(tiles) % cols)
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 88])


for row in rows:
    tag = f"b{row['branches']}_{row['spacing_mm']}mm" + ("_colour" if row["masks"] == "colour" else "")
    out = REPORT / tag
    if row["status"] != "ok":
        continue
    b = row["branches"]
    base = REPORT / f"b{b}_20mm"
    if b not in matrices:
        ref = cv2.imread(str(base / "reference.png"))
        fruit = (cv2.imread(str(base / "reference_fruit_mask.png"), 0) > 127) | (cv2.imread(str(base / "observed_mask.png"), 0) > 127)
        matrices[b] = register(truth, truth_fruit, ref, fruit)[0]
    M = matrices[b]
    ref = cv2.imread(str(out / "reference.png"))
    h, w = ref.shape[:2]
    warped = cv2.warpAffine(truth, M, (w, h))
    roi = cv2.warpAffine(np.float32(truth_fruit), M, (w, h)) > 0.5
    labels = [tuple(M[:, :2] @ np.array(p) + M[:, 2]) for p in labels0]
    floor_mask = cv2.imread(str(out / "reference_fruit_mask.png"), 0) > 127
    for name, im, m in (("floor_fruit", ref, floor_mask), ("ceiling_fruit", warped, roi)):
        im = im.copy()
        im[~m] = 0
        berries = count.count_grapesam(im, WEIGHTS)
        r = ev.match_points([(q["x_px"], q["y_px"]) for q in berries], labels, TOLERANCE)
        row.update({f"gs_{name}_n": r["predicted_count"], f"gs_{name}_err": r["count_error"], f"gs_{name}_p": r["precision"], f"gs_{name}_r": r["recall"], f"gs_{name}_f1": r["f1"]})
    meta = json.loads((out / "diagnostics.json").read_text(encoding="utf8"))
    row["excluded_views"] = len(meta["registration"]["excluded_views_1based"])
    row["pairs_ok"] = sum(p.get("accepted", False) for p in meta["pair_diagnostics"])
    row["pairs_all"] = len(meta["pair_diagnostics"])
    row["reference_view"] = meta["reference_1based"]
    label = f"{tag} refview {meta['reference_1based']}"
    if row["spacing_mm"] == "20" and row["masks"] == "sam":
        sheet_reg.append(tile(crop(cv2.imread(str(out / "registration.jpg")), roi), label))
    sheet_count.append(tile(crop(cv2.imread(str(out / "grapesam_result.jpg")), roi), label))

fields = list(dict.fromkeys(k for r in rows for k in r))
with (REPORT / "results_full.csv").open("w", newline="", encoding="utf8") as f:
    writer = csv.DictWriter(f, fields)
    writer.writeheader()
    writer.writerows(rows)
grid(sheet_reg, 2, REPORT / "review_registration.jpg")
grid(sheet_count, 4, REPORT / "review_counts.jpg")
fmt = lambda r, k: (f"{float(r[k]):.2f}" if r.get(k) not in (None, "") else "-")
print("br mm masks status | views excl pairs refv reg(inl,res,grow) | floor/res recall prec | gs n floor,res,ceil | f1 floor,res,ceil | fruit-only n,f1 floor ceil | sam3 hough f1")
for r in rows:
    print(r["branches"], r["spacing_mm"], r["masks"], r["status"][:40], "|", r.get("views"), r.get("excluded_views"), f"{r.get('pairs_ok')}/{r.get('pairs_all')}", r.get("reference_view"), r.get("reg_inliers"), fmt(r, "reg_median_residual_px"), r.get("reg_grow"), "|", fmt(r, "floor_recall"), fmt(r, "result_recall"), fmt(r, "floor_precision"), fmt(r, "result_precision"), "|", r.get("gs_floor_n"), r.get("gs_result_n"), r.get("gs_ceiling_n"), "|", fmt(r, "gs_floor_f1"), fmt(r, "gs_result_f1"), fmt(r, "gs_ceiling_f1"), "|", r.get("gs_floor_fruit_n"), fmt(r, "gs_floor_fruit_f1"), r.get("gs_ceiling_fruit_n"), fmt(r, "gs_ceiling_fruit_f1"), "|", fmt(r, "sam3_result_f1"), fmt(r, "hough_result_f1"))
