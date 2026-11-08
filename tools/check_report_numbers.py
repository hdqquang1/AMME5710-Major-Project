#!/usr/bin/env python3
"""Check the numbers in the report against the committed result files.

Reads report/sec/*.tex and compares
  - every cell of Table 1 with results/report/results_full.csv,
  - every row of Table 2 with results/ablation/assets/tab_baselines.tex (and the single-view
    F1 values, which the report adds, with results_full.csv),
  - every (R, P, F1) triple of Table 3 with results/ablation/assets/tab_ablation.tex,
  - the numbers quoted in the text that come from result files (pair counts, the bootstrap
    intervals, the runtime and the repeat-run check).
No data, masks or weights are needed. Run from anywhere:

    python3 tools/check_report_numbers.py
"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEC = ROOT / "report" / "sec"
RES = ROOT / "results"
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))


def table(tex, label):
    start = tex.index("\\label{" + label + "}")
    return tex[start:tex.index("\\end{tabular}", start)]


def nums(s):
    return re.findall(r"(?<![\w.])\d+(?:\.\d+)?", s)


rows = {(r["branches"], r["spacing_mm"]): r
        for r in csv.DictReader((RES / "report" / "results_full.csv").open(encoding="utf8")) if r["masks"] == "sam"}
results_tex = (SEC / "5_results.tex").read_text(encoding="utf8")
exp_tex = (SEC / "4_experiments.tex").read_text(encoding="utf8")
method_tex = (SEC / "3_methodology.tex").read_text(encoding="utf8")
disc_tex = (SEC / "6_discussions.tex").read_text(encoding="utf8")

# ---------------------------------------------------------------- Table 1
pc = lambda r, k: round(100 * float(r[k]))
f2 = lambda r, k: f"{float(r[k]):.2f}"
bad = []
for line in table(results_tex, "tab:main").splitlines():
    m = re.match(r"^(\d) & (\d+) & (\d+)/(\d+) & (\d+) & (\d+) & (\d+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & (\d+)\\\\", line.strip())
    if not m:
        continue
    b, mm, views, _, rs, rr, pp, iou, f1s, f1, n = m.groups()
    r = rows[(b, mm)]
    want = [r["views"], pc(r, "floor_recall"), pc(r, "result_recall"), pc(r, "result_precision"), f2(r, "result_iou"),
            f2(r, "gs_floor_fruit_f1"), f2(r, "gs_result_f1"), r["gs_result_n"]]
    got = [views, rs, rr, pp, iou, f1s, f1, n]
    bad += [f"{b} br {mm} mm: {g} vs {w}" for g, w in zip(got, want) if str(g) != str(w)]
check("Table 1: every cell against results_full.csv", not bad, "; ".join(bad) or "12 rows x 8 cells")
caption = re.search(r"Single-view precision is ([\d, and]+)\\%", results_tex)
want = [pc(rows[(b, "20")], "floor_precision") for b in "3456"]
check("Table 1 caption: single-view precision", caption and [int(x) for x in nums(caption.group(1))] == want, str(want))

# ---------------------------------------------------------------- Table 2
def body(s):
    out = []
    for line in s.splitlines():
        if "&" in line and "\\\\" in line and not line.strip().startswith(("Method", "&", "%")):
            out.append((line.split("&")[0].strip(), nums("&".join(line.split("&")[1:]))))
    return out


rep, asset = body(table(results_tex, "tab:baselines")), body((RES / "ablation" / "assets" / "tab_baselines.tex").read_text())
diff = []
for (lr, nr), (la, na) in zip(rep, asset):
    if lr.startswith("Single"):
        f1 = [f2(rows[(b, "20")], "gs_floor_fruit_f1") for b in "3456"]
        expect = sum(([na[2 * k], na[2 * k + 1], f1[k]] for k in range(4)), [])
        if nr != expect:
            diff.append(lr)
    elif nr != na:
        diff.append(lr)
check("Table 2: every row against tab_baselines.tex", len(rep) == len(asset) and not diff, ", ".join(diff) or f"{len(rep)} rows")

# ---------------------------------------------------------------- Table 3
triples = lambda s: {tuple(re.findall(r"& (\d+) & (\d+) & ([\d.]+)\\\\", line)[0])
                     for line in s.splitlines() if re.search(r"& (\d+) & (\d+) & ([\d.]+)\\\\", line)}
t3 = triples(table(results_tex, "tab:ablation"))
asset3 = triples((RES / "ablation" / "assets" / "tab_ablation.tex").read_text())
check("Table 3: every (R, P, F1) triple appears in tab_ablation.tex", t3 <= asset3, str(sorted(t3 - asset3)) if t3 - asset3 else f"{len(t3)} distinct triples")

# ---------------------------------------------------------------- numbers quoted in the text
pairs = {b: (rows[(b, "20")]["pairs_ok"], rows[(b, "20")]["pairs_all"]) for b in "3456"}
check("Sec. 3.3: 84-95 of 102 pairs at three to five branches, 40 of 98 at six",
      "84--95 of 102" in method_tex and "40 of 98" in method_tex
      and sorted(int(pairs[b][0]) for b in "345") == [84, 88, 95] and pairs["6"] == ("40", "98"), str(pairs))
check("Sec. 3.3: 13 views removed at six branches", "13 views were removed" in method_tex and rows[("6", "20")]["excluded_views"] == "13")
ci = (RES / "ablation" / "assets" / "tab_count_ci.tex").read_text()
quoted = re.findall(r"\$\[\+([\d.]+), \+([\d.]+)\]\$", results_tex)
committed = re.findall(r"\[\+([\d.]+), \+([\d.]+)\]", ci)
check("Sec. 5.5: bootstrap intervals for the F1 gain", quoted and quoted == committed, str(committed))
rt = json.loads((RES / "ablation" / "a8_runtime.json").read_text())["stage_seconds"]
front = rt["load images and masks"] + rt["SIFT and pairwise RANSAC"] + rt["joint pose solve and reference"]
check("Sec. 5.5: runtime 16 s and 209 s", "16~s" in results_tex and "209~s" in results_tex
      and round(front) == 16 and round(rt["fusion"]) == 209, f"{front:.1f} s and {rt['fusion']:.1f} s")
share = {b: (round(100 * float(rows[(b, '20')]['gs_floor_fruit_r'])), round(100 * float(rows[(b, '20')]['gs_result_r']))) for b in "345"}
lo_s, hi_s = min(v[0] for v in share.values()), max(v[0] for v in share.values())
lo_r, hi_r = min(v[1] for v in share.values()), max(v[1] for v in share.values())
check("Sec. 6: share of grapes counted, single view and reconstruction",
      f"{lo_s}--{hi_s}\\%" in disc_tex and f"{lo_r}--{hi_r}\\%" in disc_tex, f"{lo_s}-{hi_s}% to {lo_r}-{hi_r}%")


def decisions(path):
    d = json.loads(path.read_text(encoding="utf8"))
    return d, {tuple(p["pair_1based"]): bool(p["accepted"]) for p in d["pair_diagnostics"]}


same = True
for b in "345":
    a, pa = decisions(RES / "report" / f"b{b}_20mm" / "diagnostics.json")
    c, pc_ = decisions(RES / "refactor_check" / f"74_{b}b_s8" / "diagnostics.json")
    same &= (a["reference_1based"] == c["reference_1based"] and pa == pc_
             and a["supported_target_pixels"] == c["supported_target_pixels"])
check("Sec. 4.3: the second run reproduces the reported runs at three to five branches",
      same and "the same decision on all 102 tested pairs" in exp_tex)

width = max(len(n) for n, _, _ in results)
for name, ok, detail in results:
    print(f"{'ok  ' if ok else 'FAIL'}  {name:<{width}}  {detail}")
print(f"\n{sum(ok for _, ok, _ in results)} of {len(results)} checks passed")
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
