#!/usr/bin/env python3
"""Tables and figures for the A-series experiments (results/ablation -> results/ablation/assets)."""

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

A = Path("results/ablation")
OUT = A / "assets"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({**{"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix", "pdf.fonttype": 42}, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
BL = (3, 4, 5, 6)


def tex(name, lines):
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf8")


def f(x, d=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def pc(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.0f}"


# ---------------- A1 baselines
if (A / "a1.csv").is_file():
    d = pd.read_csv(A / "a1.csv")
    order = [
        ("Single view", None),
        ("Mask union", "mask_union"),
        ("Mask majority", "mask_majority"),
        ("SA mean, depth shift", "depth_shift_mean"),
        ("SA median, depth shift", "depth_shift_median"),
        ("SA masked mean, depth shift", "depth_shift_masked_mean"),
        ("SA mean, our alignment", "features_mean"),
        ("SA median, our alignment", "features_median"),
        ("SA masked mean, our alignment", "features_masked_mean"),
        ("Ours", "ours"),
    ]
    head = " & ".join(rf"\multicolumn{{3}}{{c}}{{{b} branches}}" for b in BL)
    rule = "".join(rf"\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}" for k in range(4))
    lines = [r"\begin{tabular}{@{}l" + "ccc" * 4 + "@{}}", r"\toprule", f"Method & {head}\\\\", rule,
             " & " + " & ".join(["R & P & F1"] * 4) + r"\\", r"\midrule"]
    for label, key in order:
        cells = []
        for b in BL:
            if key is None:
                r = d[(d.branches == b) & (d.method == "ours")].iloc[0]
                cells += [pc(r.floor_recall), pc(r.floor_precision), "--"]
                continue
            s = d[(d.branches == b) & (d.method == key)]
            if s.empty:
                cells += ["--"] * 3
                continue
            r = s.iloc[0]
            cells += [pc(r.result_recall), pc(r.result_precision), f(r.gs_result_f1)]
        lines.append(f"{label} & " + " & ".join(cells) + r"\\")
        if key in (None, "mask_majority", "depth_shift_masked_mean", "features_masked_mean"):
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex("tab_baselines", lines)
    disp = d.dropna(subset=["disparity_px_per_step"]).groupby("branches").disparity_px_per_step.first()
    (OUT / "a1_disparity.json").write_text(disp.to_json(), encoding="utf8")

# ---------------- A2 ablations
if (A / "a2.csv").is_file():
    d = pd.read_csv(A / "a2.csv")
    bs = [b for b in (5, 4) if b in set(d.branches)]
    lines = [r"\begin{tabular}{@{}ll" + "ccc" * len(bs) + "@{}}", r"\toprule",
             "& Variant & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{b} br.}}" for b in bs) + r"\\",
             "".join(rf"\cmidrule(lr){{{3 + 3 * k}-{5 + 3 * k}}}" for k in range(len(bs))),
             "& & " + " & ".join(["R & P & F1"] * len(bs)) + r"\\", r"\midrule"]
    groups = list(dict.fromkeys(d.group))
    for g in groups:
        variants = list(dict.fromkeys(d[d.group == g].variant))
        for k, v in enumerate(variants):
            cells = []
            for b in bs:
                s = d[(d.branches == b) & (d.group == g) & (d.variant == v)]
                if s.empty:
                    cells += ["--"] * 3
                elif str(s.iloc[0].get("status", "")).startswith("failed"):
                    cells += [r"\multicolumn{3}{c}{alignment failed}"]
                else:
                    r = s.iloc[0]
                    cells += [pc(r.result_recall), pc(r.result_precision), f(r.gs_result_f1)]
            name = g.replace("_", " ") if k == 0 else ""
            lines.append(f"{name} & {v} & " + " & ".join(cells) + r"\\")
        lines.append(r"\addlinespace[2pt]")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    tex("tab_ablation", lines)

# ---------------- A3/A4
if (A / "a34.csv").is_file():
    d = pd.read_csv(A / "a34.csv")
    ok = d[d.status == "ok"]
    # Views curve at 20 mm (b4, b5), including the full 28-view run.
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for b, c in ((4, "#d9822b"), (5, "#1b6ca8")):
        s = ok[(ok.branches == b) & (ok.spacing_mm == 20) & ok.exp.isin(["a3", "main"])].sort_values("n_views")
        ax[0].plot(s.n_views, 100 * s.result_recall, "-o", color=c, ms=3, label=f"{b} branches")
        ax[1].plot(s.n_views, s.gs_result_f1, "-o", color=c, ms=3, label=f"{b} branches")
        ax[0].axhline(100 * s.floor_recall.iloc[0], color=c, ls=":", lw=1)
        ax[1].axhline(s.gs_floor_fruit_f1.iloc[0], color=c, ls=":", lw=1)
    ax[0].set_ylabel("Fruit pixel recall (%)")
    ax[1].set_ylabel("GrapeSAM count F1")
    for a in ax:
        a.set_xlabel("Views at 20 mm (centred on reference)")
        a.grid(alpha=0.25)
    ax[0].set_ylim(0, 100)
    ax[1].set_ylim(0, 1)
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "views_curve.pdf", bbox_inches="tight")
    # Fixed N = 9 at 20/40/60 mm.
    lines = [r"\begin{tabular}{@{}c" + "cc" * 3 + "@{}}", r"\toprule",
             r"& \multicolumn{2}{c}{20 mm (160)} & \multicolumn{2}{c}{40 mm (320)} & \multicolumn{2}{c}{60 mm (480)}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}", r"Br. & R & F1 & R & F1 & R & F1\\", r"\midrule"]
    for b in BL:
        cells = []
        for mm in (20, 40, 60):
            s = d[(d.branches == b) & (d.spacing_mm == mm) & (d.n_views == 9)]
            if s.empty or s.iloc[0].status != "ok":
                cells += ["fail" if not s.empty else "--", "--"]
            else:
                r = s.iloc[0]
                cells += [pc(r.result_recall), f(r.gs_result_f1)]
        lines.append(f"{b} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex("tab_fixed_n", lines)
    # Fixed span ~480 mm.
    span = d[(d.branches.isin([4, 5])) & (((d.spacing_mm == 20) & (d.n_views.isin([25]))) | ((d.spacing_mm == 40) & (d.n_views == 13))
                                           | ((d.spacing_mm == 60) & (d.n_views == 9)))]
    span[["branches", "spacing_mm", "n_views", "span_mm", "status", "result_recall", "gs_result_f1"]].to_csv(OUT / "fixed_span.csv", index=False)
    # A4 reference variance (main ref + others).
    lines = [r"\begin{tabular}{@{}ccccc@{}}", r"\toprule", r"Br. & Refs & Recall single $\rightarrow$ restored & F1 single $\rightarrow$ restored\\", r"\midrule"]
    lines = [r"\begin{tabular}{@{}cccc@{}}", r"\toprule", r"Br. & Refs ok & Recall (\%) & Count F1\\", r"\midrule"]
    for b in BL:
        s = d[(d.branches == b) & (((d.exp == "a4")) | ((d.exp == "main") & (d.spacing_mm == 20)))]
        g = s[s.status == "ok"]
        ms = lambda col, scale=1, dd=2: f"{scale * g[col].mean():.{dd}f}$\\pm${scale * g[col].std(ddof=1):.{dd}f}" if len(g) > 1 else "--"
        lines.append(f"{b} & {len(g)}/{len(s)} & {ms('floor_recall', 100, 0)} $\\rightarrow$ {ms('result_recall', 100, 0)} & "
                     f"{ms('gs_floor_fruit_f1')} $\\rightarrow$ {ms('gs_result_f1')}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex("tab_refs", lines)

# ---------------- A5
if (A / "a5_bootstrap.csv").is_file():
    bt = pd.read_csv(A / "a5_bootstrap.csv")
    lines = [r"\begin{tabular}{@{}c" + "cc" * 3 + "c@{}}", r"\toprule",
             r"& \multicolumn{2}{c}{Single} & \multicolumn{2}{c}{Restored} & \multicolumn{2}{c}{Leaf free} & Gain\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-8}",
             r"Br. & F1 [95\% CI] & AP & F1 [95\% CI] & AP & F1 [95\% CI] & AP & 95\% CI\\", r"\midrule"]
    for b in BL:
        cells = []
        for name in ("single", "restored", "leaf_free"):
            r = bt[(bt.branches == b) & (bt.input == name)].iloc[0]
            cells += [f"{r.f1:.2f} [{r.f1_lo:.2f}, {r.f1_hi:.2f}]", f"{r.ap:.2f}"]
        r = bt[(bt.branches == b) & (bt.input == "restored")].iloc[0]
        cells.append(f"[{r.gain_lo:+.2f}, {r.gain_hi:+.2f}]")
        lines.append(f"{b} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex("tab_count_ci", lines)
    tol = pd.read_csv(A / "a5_tolerance.csv")
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.9), sharey=True)
    for k, b in enumerate(BL):
        for name, c in (("single", "k"), ("restored", "#1b6ca8"), ("leaf_free", "#6a994e")):
            s = tol[(tol.branches == b) & (tol.input == name)]
            ax[k].plot(s.tol_px, s.f1, "-o", ms=2.5, color=c, label=name.replace("_", " "))
        ax[k].axvline(24, color="0.6", lw=0.8, ls=":")
        ax[k].set_title(f"{b} branches")
        ax[k].set_xlabel("Match radius (px)")
        ax[k].grid(alpha=0.25)
    ax[0].set_ylabel("Count F1")
    ax[0].set_ylim(0, 1)
    ax[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "tolerance_curve.pdf", bbox_inches="tight")

# ---------------- A6
if (A / "a6.csv").is_file():
    d = pd.read_csv(A / "a6.csv")
    rows = []
    for b in BL:
        base = d[(d.branches == b) & (d.kind == "none")].iloc[0]
        for kind in ("shift1", "shift2", "shift3", "shift5", "rot", "scale"):
            s = d[(d.branches == b) & (d.kind == kind)]
            row = {"branches": b, "kind": kind}
            for c in ("single_recall", "restored_recall", "restored_precision", "single_f1", "restored_f1"):
                row[c] = float((s[c] - base[c]).abs().max())
            lf = s.leaf_free_f1.dropna()
            if len(lf):
                row["leaf_free_f1_range"] = f"{lf.min():.2f}-{lf.max():.2f} (base {base.leaf_free_f1:.2f})"
            rows.append(row)
    pd.DataFrame(rows).round(3).to_csv(OUT / "noise_floor.csv", index=False)

# ---------------- A7
if (A / "a7.csv").is_file():
    v = pd.read_csv(A / "a7.csv")
    bins = [(-1, 0, "0"), (0, 2, "1-2"), (2, 5, "3-5"), (5, 10, "6-10"), (10, 99, ">10")]
    xs, n, cover, cnt, single = [], [], [], [], []
    for lo, hi, lab in bins:
        s = v[(v.views_visible > lo) & (v.views_visible <= hi)]
        xs.append(f"{lab}\n(n={len(s)})")
        cover.append((s.restored_pixel_cover >= 0.5).mean() if len(s) else 0)
        cnt.append(s.restored_counted.mean() if len(s) else 0)
        single.append(s.single_counted.mean() if len(s) else 0)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    x = np.arange(len(bins))
    ax.bar(x - 0.27, single, 0.27, color="0.55", label="Counted, single view")
    ax.bar(x, cnt, 0.27, color="#1b6ca8", label="Counted, restored")
    ax.bar(x + 0.27, cover, 0.27, color="#9ecae1", label="Pixels restored")
    ax.set_xticks(x, xs)
    ax.set_xlabel("Aligned views in which the grape is visible")
    ax.set_ylabel("Fraction of grapes")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "visibility.pdf", bbox_inches="tight")

# ---------------- A8 runtime
if (A / "a8_runtime.json").is_file():
    r = json.loads((A / "a8_runtime.json").read_text(encoding="utf8"))
    lines = [r"\begin{tabular}{@{}lr@{}}", r"\toprule", r"Stage & Time (s)\\", r"\midrule"]
    for k, v in r["stage_seconds"].items():
        lines.append(f"{k} & {v:.1f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex("tab_runtime", lines)


# ---------------- Qualitative grid (branches x spacing) and failure cases
def crop_to(img, box):
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1]


R = Path("results/report")
if not all((R / f"b{b}_{mm}mm" / "grapes_rgb.png").is_file() for b in BL for mm in (20, 40, 60)):
    # The run images are git-ignored; every table and plot above has been regenerated.
    print("qualitative grid skipped: the run images in results/report/ are not present")
    print("assets written to", OUT)
    raise SystemExit(0)
tiles = []
for b in BL:
    m = cv2.imread(str(R / f"b{b}_20mm" / "reference_fruit_mask.png"), 0) > 127
    m |= cv2.imread(str(R / f"b{b}_20mm" / "observed_mask.png"), 0) > 127
    ys, xs = np.where(m)
    box = (max(xs.min() - 40, 0), max(ys.min() - 40, 0), xs.max() + 40, ys.max() + 40)
    row = [crop_to(cv2.imread(str(R / f"b{b}_20mm" / "reference.png")), box)]
    for mm in (20, 40, 60):
        row.append(crop_to(cv2.imread(str(R / f"b{b}_{mm}mm" / "grapes_rgb.png")), box))
    hh = 260
    row = [cv2.resize(t, (round(t.shape[1] * hh / t.shape[0]), hh), interpolation=cv2.INTER_AREA) for t in row]
    pad = lambda t: cv2.copyMakeBorder(t, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    tiles.append(np.hstack([pad(t) for t in row]))
wmax = max(t.shape[1] for t in tiles)
tiles = [cv2.copyMakeBorder(t, 0, 0, 0, wmax - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
cv2.imwrite(str(OUT / "qualitative_grid.jpg"), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
print("assets written to", OUT)
