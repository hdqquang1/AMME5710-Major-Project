#!/usr/bin/env python3
"""Build report tables and figures from results/report/results_full.csv."""

import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix", "pdf.fonttype": 42})

REPORT = Path("results/report")
PAPER = Path("AMME5710_Major_Report_Final")
rows = list(csv.DictReader((REPORT / "results_full.csv").open(encoding="utf8")))
get = lambda b, mm, kind="sam": next(r for r in rows if r["branches"] == str(b) and r["spacing_mm"] == str(mm) and r["masks"] == kind)
num = lambda r, k: float(r[k])
pct = lambda r, k: f"{100 * num(r, k):.0f}"
f2 = lambda r, k: f"{num(r, k):.2f}"

pixel = [r"\begin{tabular}{@{}cccccc@{}}", r"\toprule", r"Branches & mm & Views & Single & Restored & IoU\\", r"\midrule"]
count = [r"\begin{tabular}{@{}cccccc@{}}", r"\toprule", r"Branches & mm & Single & Restored & Leaf free & Count\\", r"\midrule"]
for b in (3, 4, 5, 6):
    for mm in (20, 40, 60):
        r = get(b, mm)
        total = int(r["views"]) + int(r["excluded_views"])
        pixel.append(f"{b} & {mm} & {r['views']}/{total} & {pct(r, 'floor_recall')} & {pct(r, 'result_recall')} & {f2(r, 'result_iou')}\\\\")
        count.append(f"{b} & {mm} & {f2(r, 'gs_floor_fruit_f1')} & {f2(r, 'gs_result_f1')} & {f2(r, 'gs_ceiling_fruit_f1')} & {r['gs_result_n']}\\\\")
    if b != 6:
        pixel.append(r"\addlinespace[2pt]")
        count.append(r"\addlinespace[2pt]")
for t in (pixel, count):
    t += [r"\bottomrule", r"\end{tabular}"]
(PAPER / "sec" / "tab_pixel.tex").write_text("\n".join(pixel) + "\n", encoding="utf8")
(PAPER / "sec" / "tab_count.tex").write_text("\n".join(count) + "\n", encoding="utf8")

second = [r"\begin{tabular}{@{}cccc@{}}", r"\toprule", r"Branches & GrapeSAM & SAM~3 & Hough\\", r"\midrule"]
for b in (3, 4, 5, 6):
    r = get(b, 20)
    second.append(f"{b} & {f2(r, 'gs_result_f1')} & {f2(r, 'sam3_result_f1')} & {f2(r, 'hough_result_f1')}\\\\")
second += [r"\bottomrule", r"\end{tabular}"]
(PAPER / "sec" / "tab_second.tex").write_text("\n".join(second) + "\n", encoding="utf8")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
colors = {20: "#1b6ca8", 40: "#d9822b", 60: "#6a994e"}
xs = [3, 4, 5, 6]
axes[0].plot(xs, [100 * num(get(b, 20), "floor_recall") for b in xs], "k--o", label="Single view")
axes[1].plot(xs, [num(get(b, 20), "gs_floor_fruit_f1") for b in xs], "k--o", label="Single view")
for mm, c in colors.items():
    axes[0].plot(xs, [100 * num(get(b, mm), "result_recall") for b in xs], "-o", color=c, label=f"{mm} mm")
    axes[1].plot(xs, [num(get(b, mm), "gs_result_f1") for b in xs], "-o", color=c, label=f"{mm} mm")
axes[0].set_ylabel("Fruit pixel recall (%)")
axes[1].set_ylabel("GrapeSAM count F1")
axes[0].set_ylim(0, 100)
axes[1].set_ylim(0, 1)
for a in axes:
    a.set_xlabel("Foreground branches")
    a.set_xticks(xs)
    a.grid(alpha=0.25)
    a.spines[["top", "right"]].set_visible(False)
axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(PAPER / "figures" / "performance_chart.pdf", bbox_inches="tight")

r = REPORT / "b5_20mm"
mask = cv2.imread(str(r / "observed_mask.png"), 0) > 127
ys, xs2 = np.where(mask)
crop = lambda a: a[max(ys.min() - 90, 0) : ys.max() + 90, max(xs2.min() - 90, 0) : xs2.max() + 90]
tiles = [crop(cv2.imread(str(r / f"grapesam_{n}.jpg"))) for n in ("floor", "result", "ceiling")]
sep = np.full((tiles[0].shape[0], 16, 3), 255, np.uint8)
cv2.imwrite(str(PAPER / "figures" / "count_floor_result_ceiling.jpg"), np.hstack([tiles[0], sep, tiles[1], sep, tiles[2]]), [cv2.IMWRITE_JPEG_QUALITY, 90])

print("b mm mask | floorP resP resIoU | MAE PSNR baseMAE | n floor(fruit),result,ceil(fruit) | err | F1 whole floor,ceil | res P R")
for r in rows:
    if r["status"] == "ok":
        print(r["branches"], r["spacing_mm"], r["masks"], "|", f2(r, "floor_precision"), f2(r, "result_precision"), f2(r, "result_iou"), "|", f2(r, "mae"), f2(r, "psnr"), f2(r, "baseline_mae"), "|", r["gs_floor_fruit_n"], r["gs_result_n"], r["gs_ceiling_fruit_n"], "|", r["gs_result_err"], "|", f2(r, "gs_floor_f1"), f2(r, "gs_ceiling_f1"), "|", f2(r, "gs_result_p"), f2(r, "gs_result_r"), "| align", f2(r, "aligned_error_px"))
    else:
        print(r["branches"], r["spacing_mm"], r["masks"], r["status"][:70])
