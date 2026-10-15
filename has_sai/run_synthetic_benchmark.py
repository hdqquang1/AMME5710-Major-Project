"""
run_synthetic_benchmark.py - benchmark of occlusion removal and fruit counting on
synthetic rail captures with exact ground truth.

Run from any folder (no path edits needed):

    python3 run_synthetic_benchmark.py            # generate, run, evaluate (skips finished work)
    python3 run_synthetic_benchmark.py --quick    # 4 scenes only, for a smoke test
    python3 run_synthetic_benchmark.py --stage evaluate   # re-evaluate cached results

Stages
------
1. generate  Render the scene set (sai/synth.py) into data/synthetic/<scene>/.
2. run       Run every method on every scene and cache candidates, image
             metrics and timings in data/synthetic/<scene>/results.json.
3. robust    Robustness study on the test scenes: number of views, pose
             noise and focal-length error.
3b. factorial Every image source crossed with every detector, per-view voting
             rivals, and a whole-frame ROI (separates de-occlusion from detection).
4. evaluate  Calibrate each detector's score threshold on the calibration
             scenes (seed 0, best F1), apply it unchanged to the test scenes
             (seed 1) and write tables and figures to results/.

Methods
-------
  single_colour   centre photo + prototype colour detector (finns-branch)
  finn_sai        prototype pipeline: leaf-masked mean SAI on one plane + colour detector
  *_tuned         the same two with the contour area cap calibrated (fairer to the prototype)
  single_2d       centre photo + colour-free 2-D detector
  median_2d       median SAI on the fruit plane + colour-free 2-D detector
  ours            occlusion-aware SAI + colour-free 3-D detector (this work)
  ours_noapp      ... without the radial-symmetry candidates
  ours_nohalf     ... without half-aperture selection
  ours_nosgm      ... window costs instead of semi-global aggregation
  ours_nomask     ... without foliage masks (no visibility reasoning)
  oracle          foliage-free photo + true depth + the same 3-D detector (oracle reference;
                  not a strict upper bound, its threshold is calibrated separately)

Dependencies: numpy, opencv-python, matplotlib.
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sai import synth, camera, pipeline, detect, evaluate, baselines, render  # noqa: E402

DATA = os.path.join(HERE, "data", "synthetic")
RESULTS = os.path.join(HERE, "results")


# ---------------------------------------------------------------------------
# Scene set
# ---------------------------------------------------------------------------
def scene_list(quick=False):
    """
    (name, parameters) of every benchmark scene.  Seeds 0 and 1 are calibration
    scenes (thresholds, verifier training; also used while developing the
    method); seeds 2 and 3 are test scenes never looked at during development.
    """
    out = []
    for seed in (0, 1, 2, 3):
        for cov in (0.1, 0.3, 0.5):
            for gap in (0.15, 0.34):
                out.append((f"cherry_c{int(cov*100):02d}_g{int(gap*100):02d}_s{seed}",
                            dict(fruit="cherry", coverage=cov, foliage_gap=gap, seed=seed)))
        for cov in (0.1, 0.3, 0.5):
            out.append((f"grape_c{int(cov*100):02d}_g34_s{seed}",
                        dict(fruit="grape", coverage=cov, foliage_gap=0.34, seed=seed + 10,
                             n_clusters=5)))
        for cov in (0.3, 0.5):
            out.append((f"lime_c{int(cov*100):02d}_g34_s{seed}",
                        dict(fruit="lime", coverage=cov, foliage_gap=0.34, seed=seed + 20,
                             n_clusters=9)))
    if quick:
        keep = {"cherry_c30_g34_s0", "cherry_c30_g34_s2", "grape_c30_g34_s0", "grape_c30_g34_s2"}
        out = [s for s in out if s[0] in keep]
    return out


def generate_scene(item):
    """Render one benchmark scene (worker process)."""
    n, p = item
    t0 = time.time()
    s = synth.make_scene(os.path.join(DATA, n), **p)
    return f"generated {n}: coverage {s['coverage_ref']:.2f}, {len(s['fruit'])} fruit, {time.time()-t0:.0f} s"


def split_of(name):
    return "calib" if name.endswith(("_s0", "_s1")) else "test"


def fruit_of(name):
    return name.split("_")[0]


# ---------------------------------------------------------------------------
# Per-scene evaluation helpers
# ---------------------------------------------------------------------------
def det_record(dets):
    """Compact candidate list for the cache (position, radius, score, centre, features)."""
    return [dict(x=round(d["x"], 2), y=round(d["y"], 2), r_px=round(d["r_px"], 2),
                 score=round(float(d["score"]), 4),
                 centre=[round(v, 5) for v in d["centre"]] if "centre" in d else None,
                 R=round(d.get("R", 0.0), 5),
                 feat={k: round(float(v), 5) for k, v in d.get("feat", {}).items()})
            for d in dets]


def ours_candidates(rig, sc, out):
    """Colour-free 3-D detection on a de-occlusion result, NMS applied, inside the ROI."""
    K = rig.views[rig.ref].K
    cands, maps = detect.detect_fruit(out["depth"], out["image"], out["support"], K,
                                      fruit=sc["params"]["fruit"], confidence=out["confidence"])
    cands = evaluate.in_roi(cands, sc["roi_ref"])
    return detect.nms_3d(cands), maps


def fruit_plane_depth(sc):
    """Mean depth of the fruit centres (what a tape measure would give the baselines)."""
    return float(np.mean([f["centre"][2] for f in sc["fruit"]]))


def run_scene(name):
    """Run all methods on one scene and cache the results (skips if cached)."""
    sdir = os.path.join(DATA, name)
    cache = os.path.join(sdir, "results.json")
    if os.path.exists(cache):
        return name, "cached"
    t_start = time.time()
    sc = json.load(open(os.path.join(sdir, "scene.json")))
    rig = camera.load_synthetic_rig(sdir)
    ref = rig.ref
    roi = sc["roi_ref"]
    K = rig.views[ref].K
    fruit = sc["params"]["fruit"]
    clean = camera.read_image(os.path.join(sdir, "images_clean", f"view_{ref:02d}.png"))
    z_fruit = fruit_plane_depth(sc)
    res = dict(scene=name, fruit=fruit, split=split_of(name), coverage=sc["coverage_ref"],
               gap=sc["params"]["foliage_gap"], n_countable=len(evaluate.countable_fruit(sc)),
               n_fruit_roi=sum(1 for f in sc["fruit"] if roi[0] <= f["uv_ref"][0] < roi[2]
                               and roi[1] <= f["uv_ref"][1] < roi[3]),
               dets={}, images={}, masks={}, depth={}, timings={})
    # observability ceiling: countable fruit at least half visible in >= 2 photos
    gts = evaluate.countable_fruit(sc)
    res["n_observable_2views"] = int(sum(1 for g in gts if g["n_views_visible_50"] >= 2))
    res["n_visible_ref_foliage"] = int(sum(1 for g in gts if g["visible_frac_with_foliage"][ref] >= 0.5))

    # --- baselines -------------------------------------------------------
    single = rig.views[ref].image
    t0 = time.time()
    mean_img, _ = render.refocus_plane(rig, z_fruit, "mean")
    med_img, _ = render.refocus_plane(rig, z_fruit, "median")
    fsai, _ = baselines.finn_sai(rig, z_fruit)
    res["timings"]["baselines"] = time.time() - t0
    for key, img in (("single", single), ("mean_sai", mean_img), ("median_sai", med_img), ("finn_sai", fsai)):
        res["images"][key] = dict(psnr=evaluate.psnr(img, clean, roi), ssim=evaluate.ssim(img, clean, roi))
    res["dets"]["single_colour"] = det_record(baselines.finn_colour_detect(single, roi))
    res["dets"]["finn_sai"] = det_record(baselines.finn_colour_detect(fsai, roi))
    res["dets"]["single_colour_tuned"] = det_record(baselines.finn_colour_detect(single, roi, max_area=1e5))
    res["dets"]["finn_sai_tuned"] = det_record(baselines.finn_colour_detect(fsai, roi, max_area=1e5))
    res["dets"]["single_2d"] = det_record(detect.nms_3d(evaluate.in_roi(
        detect.detect_2d(single, K, z_fruit, fruit=fruit), roi)))
    res["dets"]["median_2d"] = det_record(detect.nms_3d(evaluate.in_roi(
        detect.detect_2d(med_img, K, z_fruit, fruit=fruit), roi)))

    # --- ours and ablations ----------------------------------------------
    oid_ref = cv2.imread(os.path.join(sdir, "gt", f"oid_{ref:02d}.png"), cv2.IMREAD_UNCHANGED)
    zc = np.load(os.path.join(sdir, "gt", f"depth_clean_{ref:02d}.npy")).astype(np.float32)
    oidc = cv2.imread(os.path.join(sdir, "gt", f"oid_clean_{ref:02d}.png"), cv2.IMREAD_UNCHANGED)
    roi_m = np.zeros(zc.shape, bool)
    roi_m[roi[1]:roi[3], roi[0]:roi[2]] = True
    variants = dict(ours={}, ours_nohalf=dict(half_select=False), ours_nosgm=dict(use_sgm=False),
                    ours_nomask="nomask")
    for key, cfg in variants.items():
        t0 = time.time()
        if cfg == "nomask":
            out = deocclude_without_masks(rig)
        else:
            out = pipeline.deocclude(rig, cfg, verbose=False)
        dets, maps = ours_candidates(rig, sc, out)
        res["timings"][key] = time.time() - t0
        res["dets"][key] = det_record(dets)
        res["images"][key] = dict(psnr=evaluate.psnr(out["image"], clean, roi),
                                  ssim=evaluate.ssim(out["image"], clean, roi),
                                  unobserved=float(np.mean(out["support"][roi_m] < 0.5)))
        err = out["depth"] - zc
        res["depth"][key] = {cls: dict(bias_mm=float(np.median(err[m & roi_m]) * 1e3),
                                       mae_mm=float(np.median(np.abs(err[m & roi_m])) * 1e3))
                             for cls, m in (("backdrop", oidc == synth.ID_BACKDROP),
                                            ("lattice", oidc == synth.ID_LATTICE),
                                            ("fruit", oidc >= synth.ID_FRUIT0)) if np.any(m & roi_m)}
        if key == "ours":
            res["timings"]["ours_stages"] = out["timings"]
            res["masks"]["ref"] = evaluate.mask_scores(out["masks"][ref], (oid_ref == synth.ID_FOLIAGE))
            ious = []
            for j in range(len(rig.views)):
                o = cv2.imread(os.path.join(sdir, "gt", f"oid_{j:02d}.png"), cv2.IMREAD_UNCHANGED)
                ious.append(evaluate.mask_scores(out["masks"][j], o == synth.ID_FOLIAGE))
            res["masks"]["all_mean"] = {k: float(np.mean([d[k] for d in ious])) for k in ious[0]}
            res["z_split"] = out["z_split"]
            # no-appearance ablation reuses this reconstruction
            K_ = rig.views[ref].K
            c2, _ = detect.detect_fruit(out["depth"], out["image"], out["support"], K_, fruit=fruit,
                                        confidence=out["confidence"], use_appearance=False)
            res["dets"]["ours_noapp"] = det_record(detect.nms_3d(evaluate.in_roi(c2, roi)))
            save_panels(sdir, rig, out, single, med_img, fsai, clean)
    # --- oracle: foliage-free photo, true depth --------------------------
    sup_full = np.full(zc.shape, float(len(rig.views)), np.float32)
    c3, _ = detect.detect_fruit(np.minimum(zc, 5.0), clean, sup_full, K, fruit=fruit)
    res["dets"]["oracle"] = det_record(detect.nms_3d(evaluate.in_roi(c3, roi)))
    res["timings"]["total"] = time.time() - t_start
    with open(cache, "w") as f:
        json.dump(res, f)
    return name, f"done in {res['timings']['total']:.0f} s"


def deocclude_without_masks(rig):
    """Ablation: the same sweeps and render with every foliage mask empty (no visibility)."""
    h, w = rig.views[0].image.shape[:2]
    empty = [np.zeros((h, w), np.float32) for _ in rig.views]
    fm = pipeline.planesweep.foreground_masks(rig, 0.40, 1.40)
    lo, hi, z_wall = pipeline.background_range(fm["depths"][rig.ref], fm["z_split"], 0.07, 0.05)
    planes = pipeline.planesweep.inverse_depth_planes(lo, hi, 1.0, rig.views[rig.ref].K[0, 0],
                                                      rig.baselines().max())
    guide_img, _ = render.refocus_plane(rig, z_wall, "mean")
    guide = cv2.cvtColor(np.clip(guide_img, 0, 1), cv2.COLOR_BGR2GRAY)
    va, vs, _ = render.background_cost_volume(rig, empty, planes)
    S = pipeline.sgm.sgm_aggregate(vs, guide)
    depth, cost, conf = pipeline.sgm.wta_parabola(S, planes)
    S = pipeline.sgm.sgm_aggregate(va, guide)
    depth_r, _, _ = pipeline.sgm.wta_parabola(S, planes)
    img, sup = render.render_all_in_focus(rig, depth_r, empty)
    return dict(depth=depth, image=img, support=sup, confidence=conf, masks=empty, timings={})


def save_panels(sdir, rig, out, single, med, fsai, clean):
    """Qualitative images for the report (8-bit PNG)."""
    pdir = os.path.join(sdir, "panels")
    os.makedirs(pdir, exist_ok=True)
    to8 = lambda a: np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8)
    for k, a in (("single", single), ("median_sai", med), ("finn_sai", fsai), ("ours", out["image"]),
                 ("clean", clean)):
        cv2.imwrite(os.path.join(pdir, f"{k}.png"), to8(a))
    cv2.imwrite(os.path.join(pdir, "mask_ref.png"), to8(out["masks"][rig.ref]))
    d = out["depth"]
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    cv2.imwrite(os.path.join(pdir, "depth.png"),
                cv2.applyColorMap(to8(np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)), cv2.COLORMAP_TURBO))


# ---------------------------------------------------------------------------
# Robustness study
# ---------------------------------------------------------------------------
def view_subset(n_total, ref, k):
    """k views evenly spread over the rail, always including the reference."""
    if k >= n_total:
        return list(range(n_total))
    idx = np.unique(np.round(np.linspace(0, n_total - 1, k)).astype(int)).tolist()
    if ref not in idx:
        idx[int(np.argmin([abs(i - ref) for i in idx]))] = ref
    return sorted(set(idx))


def run_robust(name):
    """Views / pose-noise / focal-error study on one test scene (cached)."""
    sdir = os.path.join(DATA, name)
    cache = os.path.join(sdir, "robust.json")
    if os.path.exists(cache):
        return name, "cached"
    sc = json.load(open(os.path.join(sdir, "scene.json")))
    roi = sc["roi_ref"]
    fruit = sc["params"]["fruit"]
    z_fruit = fruit_plane_depth(sc)
    ref_idx = json.load(open(os.path.join(sdir, "cameras.json")))["ref"]
    clean = camera.read_image(os.path.join(sdir, "images_clean", f"view_{ref_idx:02d}.png"))
    out_all = {}
    conditions = [("views", k, None) for k in (3, 5, 7, 9, 13)]
    conditions += [("pose", s, None) for s in (0.1, 0.25, 0.5)]
    conditions += [("fx", e, None) for e in (0.97, 1.03)]
    for kind, val, _ in conditions:
        if kind == "views":
            rig = camera.load_synthetic_rig(sdir)
            rig = rig.subset(view_subset(len(rig.views), rig.ref, val))
        elif kind == "pose":
            rig = camera.load_synthetic_rig(sdir, pose_noise=(val, 0.001, np.random.default_rng(7)))
        else:
            rig = camera.load_synthetic_rig(sdir, K_scale_error=val)
        K = rig.views[rig.ref].K
        t0 = time.time()
        out = pipeline.deocclude(rig, verbose=False)
        dets, _ = ours_candidates(rig, sc, out)
        med, _ = render.refocus_plane(rig, z_fruit, "median")
        mdets = detect.nms_3d(evaluate.in_roi(detect.detect_2d(med, K, z_fruit, fruit=fruit), roi))
        out_all[f"{kind}={val}"] = dict(
            kind=kind, value=val, n_views=len(rig.views), seconds=time.time() - t0,
            ours=det_record(dets), median_2d=det_record(mdets),
            psnr_ours=evaluate.psnr(out["image"], clean, roi), psnr_median=evaluate.psnr(med, clean, roi))
    with open(cache, "w") as f:
        json.dump(out_all, f)
    return name, "done"


# ---------------------------------------------------------------------------
# Factorial study (council request): image source x detector, two ROIs
# ---------------------------------------------------------------------------
def wide_roi(sc):
    """Whole reference frame minus a 5 % margin (scores false positives everywhere)."""
    W, H = sc["params"]["width"], sc["params"]["height"]
    return [int(0.05 * W), int(0.05 * H), int(0.95 * W), int(0.95 * H)]


def run_factorial(name):
    """Every (image source, detector) pair plus per-view voting, on the fruit ROI and the wide ROI."""
    from sai import factorial
    sdir = os.path.join(DATA, name)
    cache = os.path.join(sdir, "factorial.json")
    if os.path.exists(cache):
        return name, "cached"
    sc = json.load(open(os.path.join(sdir, "scene.json")))
    rig = camera.load_synthetic_rig(sdir)
    K = rig.views[rig.ref].K
    fruit = sc["params"]["fruit"]
    z_fruit = fruit_plane_depth(sc)
    rois = dict(roi=sc["roi_ref"], wide=wide_roi(sc))
    out = {}
    for rname, roi in rois.items():
        dets = factorial.factorial_detections(sdir, roi, K, z_fruit, fruit)
        dets["vote|colour"] = baselines.finn_vote(rig, z_fruit, roi)
        dets["vote|free2d"] = baselines.free_vote(rig, z_fruit, roi, fruit=fruit)
        out[rname] = {k: det_record(v) for k, v in dets.items()}
    res = pipeline.deocclude(rig, verbose=False)
    cands, _ = detect.detect_fruit(res["depth"], res["image"], res["support"], K, fruit=fruit,
                                   confidence=res["confidence"])
    zc = np.load(os.path.join(sdir, "gt", f"depth_clean_{rig.ref:02d}.npy")).astype(np.float32)
    clean = camera.read_image(os.path.join(sdir, "images_clean", f"view_{rig.ref:02d}.png"))
    oc, _ = detect.detect_fruit(np.minimum(zc, 5.0), clean, np.full(zc.shape, float(len(rig.views)), np.float32),
                                K, fruit=fruit)
    for rname, roi in rois.items():
        out[rname]["ours|relief3d"] = det_record(detect.nms_3d(evaluate.in_roi(cands, roi)))
        out[rname]["clean|relief3d_truedepth"] = det_record(detect.nms_3d(evaluate.in_roi(oc, roi)))
    np.savez_compressed(os.path.join(sdir, "deocclusion.npz"), depth=res["depth"], image=res["image"],
                        support=res["support"], confidence=res["confidence"])
    with open(cache, "w") as f:
        json.dump(out, f)
    return name, "done"


def evaluate_factorial(names, out_dir):
    """Calibrate each (method, ROI, fruit) on seeds 0-1 and report test F1 per seed layout."""
    caches = {}
    for n in names:
        p = os.path.join(DATA, n, "factorial.json")
        if os.path.exists(p):
            caches[n] = json.load(open(p))
    if not caches:
        return None
    scenes = {n: json.load(open(os.path.join(DATA, n, "scene.json"))) for n in caches}
    keys = sorted({k for c in caches.values() for k in c["roi"]})
    table = {}
    rows = []
    for rname in ("roi", "wide"):
        for fruit in ("cherry", "grape", "lime"):
            for key in keys:
                calib = [(caches[n][rname].get(key, []), evaluate.countable_fruit(scenes[n]))
                         for n in caches if fruit_of(n) == fruit and split_of(n) == "calib"]
                if not calib:
                    continue
                fixed = key.endswith("|colour")          # prototype detector as published: no threshold
                t = None
                if not fixed:
                    scores = sorted({r["score"] for recs, _ in calib for r in recs})
                    best = (-1, None)
                    for tt in (np.unique(np.quantile(scores, np.linspace(0, 1, 120))) if scores else []):
                        tp = fp = fn = 0
                        for recs, gts in calib:
                            m = evaluate.match_counts(to_dets(recs, tt), gts)
                            tp, fp, fn = tp + m["tp"], fp + m["fp"], fn + m["fn"]
                        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
                        if f1 > best[0]:
                            best = (f1, float(tt))
                    t = best[1]
                tp = fp = fn = 0
                per_layout = {}
                for n in caches:
                    if fruit_of(n) != fruit or split_of(n) != "test":
                        continue
                    m = evaluate.match_counts(to_dets(caches[n][rname].get(key, []), t), evaluate.countable_fruit(scenes[n]))
                    tp, fp, fn = tp + m["tp"], fp + m["fp"], fn + m["fn"]
                    lay = n.rsplit("_", 1)[1]
                    a = per_layout.setdefault(lay, [0, 0, 0])
                    a[0] += m["tp"]; a[1] += m["fp"]; a[2] += m["fn"]
                    rows.append(dict(roi=rname, fruit=fruit, method=key, scene=n, coverage=round(scenes[n]["coverage_ref"], 3),
                                     tp=m["tp"], fp=m["fp"], fn=m["fn"], f1=round(m["f1"], 4),
                                     count_error=round(m["count_error"], 4), threshold=t))
                f1 = 2 * tp / max(2 * tp + fp + fn, 1)
                table.setdefault(rname, {}).setdefault(fruit, {})[key] = dict(
                    f1=round(f1, 4), tp=tp, fp=fp, fn=fn, threshold=t,
                    f1_per_layout={k: round(2 * v[0] / max(2 * v[0] + v[1] + v[2], 1), 4) for k, v in per_layout.items()})
    if rows:
        with open(os.path.join(out_dir, "factorial_per_scene.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(out_dir, "factorial.json"), "w") as f:
        json.dump(table, f, indent=1)
    srcs = [("single", "centre photo"), ("finn_sai", "prototype masked mean SAI"), ("median_sai", "median SAI"),
            ("ours", "occlusion-aware SAI (ours)"), ("clean", "foliage-free photo")]
    dets_ = [("colour", "colour (as published)"), ("colour_tuned", "colour, area cap calibrated"),
             ("colour_dt", "colour + distance transform"), ("free2d", "colour-free 2-D")]
    L = ["# Factorial study: image source x detector (test seeds 2-3, simulator)", "",
         "F1 on the test layouts; thresholds calibrated on seeds 0-1.  Per-layout F1 in brackets.  "
         "'roi' scores only the fruit region; 'wide' scores the whole frame minus a 5 % margin.", ""]
    for rname in ("roi", "wide"):
        for fruit in ("cherry", "grape", "lime"):
            T = table.get(rname, {}).get(fruit)
            if not T:
                continue
            L += [f"## {fruit}, {rname}", "", "| Image | " + " | ".join(d for _, d in dets_) + " |",
                  "|---|" + "---|" * len(dets_)]
            for sk, sl in srcs:
                cells = []
                for dk, _ in dets_:
                    v = T.get(f"{sk}|{dk}")
                    cells.append("-" if v is None else f"{v['f1']:.3f} ({', '.join(f'{x:.2f}' for x in v['f1_per_layout'].values())})")
                L.append(f"| {sl} | " + " | ".join(cells) + " |")
            for k, lab in (("vote|colour", "per-view colour voting (prototype fuse_multi_view)"),
                           ("vote|free2d", "per-view colour-free 2-D voting"),
                           ("ours|relief3d", "ours: occlusion-aware SAI + colour-free 3-D detector"),
                           ("clean|relief3d_truedepth", "oracle: foliage-free photo + true depth + 3-D detector")):
                v = T.get(k)
                if v:
                    L.append(f"| {lab} | F1 {v['f1']:.3f} ({', '.join(f'{x:.2f}' for x in v['f1_per_layout'].values())}); "
                             f"TP {v['tp']} FP {v['fp']} FN {v['fn']} | | | |")
            L.append("")
    # heavy cover: the decisive cells
    L += ["## Cherries at 50 % foliage (test), F1 by method", ""]
    for key in ("median_sai|free2d", "finn_sai|colour_dt", "vote|colour", "vote|free2d", "ours|relief3d"):
        sel = [r for r in rows if r["roi"] == "roi" and r["fruit"] == "cherry" and r["method"] == key and r["coverage"] > 0.45]
        if sel:
            tp, fp, fn = sum(r["tp"] for r in sel), sum(r["fp"] for r in sel), sum(r["fn"] for r in sel)
            L.append(f"- {key}: F1 {2*tp/max(2*tp+fp+fn,1):.3f} (TP {tp}, FP {fp}, FN {fn}, {len(sel)} scenes)")
    open(os.path.join(out_dir, "factorial.md"), "w").write("\n".join(L) + "\n")
    return table


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
METHODS = ["single_colour", "single_colour_tuned", "finn_sai", "finn_sai_tuned", "single_2d", "median_2d",
           "ours", "ours_learned", "ours_noapp", "ours_nohalf", "ours_nosgm", "ours_nomask", "oracle"]
FIXED = {"single_colour", "finn_sai"}       # prototype detector has no score threshold to tune


def load_results(names):
    out = {}
    for n in names:
        p = os.path.join(DATA, n, "results.json")
        if os.path.exists(p):
            out[n] = json.load(open(p))
    return out


def to_dets(rec, t=None):
    d = [dict(x=r["x"], y=r["y"], score=r["score"]) for r in rec]
    return d if t is None else [x for x in d if x["score"] >= t]


def calibrate(results, scenes, method, fruits):
    """Score threshold with the best pooled F1 over the calibration scenes of `fruits`."""
    pool = []
    for n in scenes:
        if split_of(n) == "calib" and fruit_of(n) in fruits and n in results:
            sc = json.load(open(os.path.join(DATA, n, "scene.json")))
            pool.append((results[n]["dets"][method], evaluate.countable_fruit(sc)))
    if not pool:
        return None
    scores = sorted({r["score"] for recs, _ in pool for r in recs})
    if not scores:
        return None
    cand = np.unique(np.quantile(scores, np.linspace(0, 1, 200)))
    best_t, best_f1 = cand[0], -1
    for t in cand:
        tp = fp = fn = 0
        for recs, gts in pool:
            m = evaluate.match_counts(to_dets(recs, t), gts)
            tp, fp, fn = tp + m["tp"], fp + m["fp"], fn + m["fn"]
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return float(best_t)


def add_learned_scores(results):
    """
    Score the 'ours' candidates with the random-forest verifier (sai/verify.py).

    Calibration scenes get out-of-fold scores (leave-one-LAYOUT-out: scenes with
    the same seed share a fruit layout, so all of them are held out together),
    so their threshold calibration is not biased by training on the same fruit;
    test scenes are scored by a model trained on every calibration scene.
    """
    from sai import verify
    data = {}
    for n, r in results.items():
        sc = json.load(open(os.path.join(DATA, n, "scene.json")))
        dets = r["dets"]["ours"]
        X = verify.feature_matrix(dets, r["fruit"])
        gts = evaluate.countable_fruit(sc)
        m = evaluate.match_counts(to_dets(dets), gts)
        y = np.zeros(len(dets), np.int32)
        for i, _ in m["matched"]:
            y[i] = 1
        data[n] = (X, y)
    calib = [n for n in results if split_of(n) == "calib"]
    if len(calib) < 2:
        return
    def fit(names_):
        X = np.vstack([data[n][0] for n in names_])
        y = np.concatenate([data[n][1] for n in names_])
        return verify.train(X, y)
    full = fit(calib)
    layout = lambda name: (fruit_of(name), name.rsplit("_", 1)[1])   # same seed = same fruit layout
    for n, r in results.items():
        if split_of(n) == "calib":
            model = fit([c for c in calib if layout(c) != layout(n)])
        else:
            model = full
        p = verify.fruit_probability(model, data[n][0])
        r["dets"]["ours_learned"] = [dict(d, score=float(s_)) for d, s_ in zip(r["dets"]["ours"], p)]


def evaluate_all(names, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    results = load_results(names)
    add_learned_scores(results)
    thresholds = {}
    for m in METHODS:
        if m in FIXED:
            thresholds[m] = {"cherry_only": None, "per_fruit": {f: None for f in ("cherry", "grape", "lime")}}
            continue
        thresholds[m] = {"cherry_only": calibrate(results, names, m, {"cherry"}),
                         "per_fruit": {f: calibrate(results, names, m, {f}) for f in ("cherry", "grape", "lime")}}
    rows = []
    for n, r in results.items():
        sc = json.load(open(os.path.join(DATA, n, "scene.json")))
        gts = evaluate.countable_fruit(sc)
        for m in METHODS:
            if m not in r["dets"]:
                continue
            for mode in ("cherry_only", "per_fruit"):
                t = thresholds[m][mode] if mode == "cherry_only" else thresholds[m][mode][r["fruit"]]
                mm = evaluate.match_counts(to_dets(r["dets"][m], t), gts)
                rows.append(dict(scene=n, fruit=r["fruit"], split=r["split"], coverage=round(r["coverage"], 3),
                                 gap=r["gap"], method=m, calib=mode, threshold=t, n_gt=mm["n_gt"],
                                 n_pred=mm["n_pred"], tp=mm["tp"], fp=mm["fp"], fn=mm["fn"],
                                 precision=round(mm["precision"], 4), recall=round(mm["recall"], 4),
                                 f1=round(mm["f1"], 4), count_error=round(mm["count_error"], 4),
                                 n_observable_2views=r["n_observable_2views"],
                                 n_visible_ref_foliage=r["n_visible_ref_foliage"]))
    with open(os.path.join(out_dir, "counting_per_scene.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # image / mask / depth table
    img_rows = []
    for n, r in results.items():
        for k, v in r["images"].items():
            img_rows.append(dict(scene=n, fruit=r["fruit"], split=r["split"], coverage=round(r["coverage"], 3),
                                 gap=r["gap"], image=k, psnr=round(v["psnr"], 3), ssim=round(v["ssim"], 4),
                                 unobserved=round(v.get("unobserved", float("nan")), 4)))
    with open(os.path.join(out_dir, "images_per_scene.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(img_rows[0].keys()))
        w.writeheader()
        w.writerows(img_rows)
    summary = summarise(rows, img_rows, results, thresholds)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    write_markdown(summary, os.path.join(out_dir, "summary.md"))
    robust = evaluate_robust(names, thresholds, out_dir)
    try:
        make_figures(rows, img_rows, results, robust, out_dir)
    except Exception as e:  # figures are optional; tables are the record
        print("figure generation failed:", e)
    return summary


def pooled(rows, **filt):
    sel = [r for r in rows if all(r[k] == v for k, v in filt.items())]
    tp = sum(r["tp"] for r in sel)
    fp = sum(r["fp"] for r in sel)
    fn = sum(r["fn"] for r in sel)
    n = sum(r["n_gt"] for r in sel)
    npred = sum(r["n_pred"] for r in sel)
    p = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    f1 = 2 * p * rc / max(p + rc, 1e-9)
    mape = float(np.mean([abs(r["count_error"]) for r in sel])) if sel else float("nan")
    return dict(scenes=len(sel), n_gt=n, n_pred=npred, tp=tp, fp=fp, fn=fn, precision=round(p, 4),
                recall=round(rc, 4), f1=round(f1, 4), mean_abs_count_error=round(mape, 4))


def summarise(rows, img_rows, results, thresholds):
    S = dict(thresholds=thresholds, counting={}, images={}, masks={}, depth={}, timings={})
    for mode in ("cherry_only", "per_fruit"):
        S["counting"][mode] = {}
        for fruit in ("cherry", "grape", "lime"):
            S["counting"][mode][fruit] = {m: pooled(rows, method=m, calib=mode, split="test", fruit=fruit)
                                          for m in METHODS}
    # by coverage (cherries, test, per-fruit calibration)
    cov_tab = {}
    for target in (0.1, 0.3, 0.5):
        sel = [r for r in rows if r["fruit"] == "cherry" and r["split"] == "test" and r["calib"] == "per_fruit"
               and abs(r["coverage"] - target) < 0.05]
        cov_tab[str(target)] = {m: pooled(sel, method=m) for m in METHODS}
    S["counting"]["cherry_test_by_coverage"] = cov_tab
    for k in sorted({r["image"] for r in img_rows}):
        for fruit in ("cherry", "grape", "lime"):
            sel = [r for r in img_rows if r["image"] == k and r["fruit"] == fruit]
            if sel:
                S["images"].setdefault(k, {})[fruit] = dict(psnr=round(float(np.mean([r["psnr"] for r in sel])), 2),
                                                            ssim=round(float(np.mean([r["ssim"] for r in sel])), 4))
    S["masks"] = {k: float(np.mean([r["masks"]["ref"][k] for r in results.values()])) for k in ("recall", "precision", "iou")}
    S["masks_all_views"] = {k: float(np.mean([r["masks"]["all_mean"][k] for r in results.values()])) for k in ("recall", "precision", "iou")}
    for v in ("ours", "ours_nohalf", "ours_nosgm", "ours_nomask"):
        d = {}
        for cls in ("backdrop", "lattice", "fruit"):
            vals = [r["depth"][v][cls]["mae_mm"] for r in results.values() if v in r["depth"] and cls in r["depth"][v]]
            if vals:
                d[cls] = round(float(np.mean(vals)), 2)
        S["depth"][v] = d
    S["timings"] = {k: round(float(np.mean([r["timings"][k] for r in results.values() if k in r["timings"]])), 1)
                    for k in ("baselines", "ours", "ours_nohalf", "ours_nosgm", "ours_nomask", "total")}
    S["observability"] = {f: dict(
        countable=sum(r["n_countable"] for r in results.values() if r["fruit"] == f and r["split"] == "test"),
        observable_2views=sum(r["n_observable_2views"] for r in results.values() if r["fruit"] == f and r["split"] == "test"),
        visible_ref_foliage=sum(r["n_visible_ref_foliage"] for r in results.values() if r["fruit"] == f and r["split"] == "test"))
        for f in ("cherry", "grape", "lime")}
    return S


def write_markdown(S, path):
    L = ["# Synthetic benchmark summary", "",
         "Test scenes only (seeds 2 and 3, never looked at while developing the method); score thresholds "
         "and the verifier were calibrated on seeds 0 and 1 (best pooled F1).  Scenes with the same seed "
         "share one fruit layout, so each fruit type has two test layouts.  All numbers are from the "
         "simulator.", ""]
    names = {"single_colour": "Centre photo + colour detector (prototype)",
             "single_colour_tuned": "  same, area cap calibrated",
             "finn_sai": "Leaf-masked mean SAI + colour detector (prototype)",
             "finn_sai_tuned": "  same, area cap calibrated",
             "single_2d": "Centre photo + colour-free 2-D detector",
             "median_2d": "Median SAI + colour-free 2-D detector",
             "ours": "Occlusion-aware SAI + colour-free 3-D detector (ours)",
             "ours_learned": "  ours + learned colour-free verifier (random forest)",
             "ours_noapp": "  ours without radial-symmetry candidates",
             "ours_nohalf": "  ours without half-aperture selection",
             "ours_nosgm": "  ours without semi-global aggregation",
             "ours_nomask": "  ours without foliage masks",
             "oracle": "Foliage-free photo + true depth (oracle reference)"}
    for mode, title in (("per_fruit", "Threshold calibrated per fruit type"),
                        ("cherry_only", "One threshold calibrated on cherries only (cross-fruit generalisation)")):
        L += [f"## Counting: {title}", ""]
        for fruit in ("cherry", "grape", "lime"):
            obs = S["observability"][fruit]
            L += [f"### {fruit} (countable {obs['countable']}, at least half visible in >= 2 photos: "
                  f"{obs['observable_2views']}, visible in the centre photo through foliage: {obs['visible_ref_foliage']})", "",
                  "| Method | Precision | Recall | F1 | Mean abs. count error | n_pred / n_gt |",
                  "|---|---|---|---|---|---|"]
            for m, v in S["counting"][mode][fruit].items():
                if v["scenes"] == 0:
                    continue
                L.append(f"| {names[m]} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | "
                         f"{100*v['mean_abs_count_error']:.1f} % | {v['n_pred']} / {v['n_gt']} |")
            L.append("")
    L += ["## Cherries by foliage coverage (test, per-fruit threshold), F1", "",
          "| Method | 10 % | 30 % | 50 % |", "|---|---|---|---|"]
    tab = S["counting"]["cherry_test_by_coverage"]
    for m in names:
        L.append(f"| {names[m]} | " + " | ".join(f"{tab[c][m]['f1']:.3f}" for c in ("0.1", "0.3", "0.5")) + " |")
    L += ["", "## Leaf-free image quality inside the ROI (all scenes)", "",
          "| Image | PSNR cherry | SSIM cherry | PSNR grape | PSNR lime |", "|---|---|---|---|---|"]
    for k, v in S["images"].items():
        L.append(f"| {k} | {v.get('cherry', {}).get('psnr', float('nan')):.2f} | "
                 f"{v.get('cherry', {}).get('ssim', float('nan')):.4f} | "
                 f"{v.get('grape', {}).get('psnr', float('nan')):.2f} | {v.get('lime', {}).get('psnr', float('nan')):.2f} |")
    L += ["", "## Foliage masks (colour-free, from depth)", "",
          f"Reference view: recall {S['masks']['recall']:.3f}, precision {S['masks']['precision']:.3f}, IoU {S['masks']['iou']:.3f}.  "
          f"All views: recall {S['masks_all_views']['recall']:.3f}, IoU {S['masks_all_views']['iou']:.3f}.", "",
          "## Background depth, median absolute error [mm]", "", "| Variant | Backdrop | Lattice | Fruit |", "|---|---|---|---|"]
    for v, d in S["depth"].items():
        L.append(f"| {v} | {d.get('backdrop', float('nan'))} | {d.get('lattice', float('nan'))} | {d.get('fruit', float('nan'))} |")
    L += ["", "## Mean run time per scene [s]", "", json.dumps(S["timings"]), ""]
    open(path, "w").write("\n".join(L))


def evaluate_robust(names, thresholds, out_dir):
    rows = []
    for n in names:
        p = os.path.join(DATA, n, "robust.json")
        if not os.path.exists(p):
            continue
        sc = json.load(open(os.path.join(DATA, n, "scene.json")))
        gts = evaluate.countable_fruit(sc)
        fruit = sc["params"]["fruit"]
        for cond, v in json.load(open(p)).items():
            for m in ("ours", "median_2d"):
                t = thresholds[m]["per_fruit"][fruit]
                mm = evaluate.match_counts(to_dets(v[m], t), gts)
                rows.append(dict(scene=n, kind=v["kind"], value=v["value"], method=m, tp=mm["tp"], fp=mm["fp"],
                                 fn=mm["fn"], n_gt=mm["n_gt"], f1=round(mm["f1"], 4),
                                 count_error=round(mm["count_error"], 4),
                                 psnr=round(v["psnr_ours"] if m == "ours" else v["psnr_median"], 3)))
    if rows:
        with open(os.path.join(out_dir, "robustness.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


def make_figures(rows, img_rows, results, robust, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # F1 vs coverage for cherries (test, per-fruit calibration)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    show = [("single_colour", "photo + colour"), ("finn_sai", "masked mean SAI + colour"),
            ("median_2d", "median SAI + 2-D"), ("ours", "ours"), ("oracle", "oracle reference")]
    for m, lab in show:
        xs, f1s, ces = [], [], []
        for c in (0.1, 0.3, 0.5):
            sel = [r for r in rows if r["fruit"] == "cherry" and r["split"] == "test" and r["calib"] == "per_fruit"
                   and r["method"] == m and abs(r["coverage"] - c) < 0.05]
            if sel:
                pr = pooled(sel)
                xs.append(100 * c)
                f1s.append(pr["f1"])
                ces.append(100 * pr["mean_abs_count_error"])
        ax[0].plot(xs, f1s, "o-", label=lab)
        ax[1].plot(xs, ces, "o-", label=lab)
    ax[0].set_xlabel("foliage coverage of the centre photo [%]")
    ax[0].set_ylabel("F1 (test scenes)")
    ax[0].set_ylim(0, 1.05)
    ax[1].set_xlabel("foliage coverage of the centre photo [%]")
    ax[1].set_ylabel("mean |count error| [%]")
    ax[0].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "f1_vs_coverage.png"), dpi=160)
    plt.close(fig)
    if robust:
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
        for i, (kind, lab) in enumerate((("views", "number of photos"), ("pose", "pose noise sigma [deg]"),
                                         ("fx", "focal length scale"))):
            for m in ("ours", "median_2d"):
                sel = [r for r in robust if r["kind"] == kind and r["method"] == m]
                vals = sorted({r["value"] for r in sel})
                f1 = []
                for v in vals:
                    s2 = [r for r in sel if r["value"] == v]
                    tp, fp, fn = sum(r["tp"] for r in s2), sum(r["fp"] for r in s2), sum(r["fn"] for r in s2)
                    f1.append(2 * tp / max(2 * tp + fp + fn, 1))
                ax[i].plot(vals, f1, "o-", label=m)
            ax[i].set_xlabel(lab)
            ax[i].set_ylabel("F1")
            ax[i].set_ylim(0, 1.05)
            ax[i].grid(alpha=0.3)
            ax[i].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "robustness.png"), dpi=160)
        plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["all", "generate", "run", "robust", "factorial", "evaluate"], default="all")
    ap.add_argument("--quick", action="store_true", help="4 scenes only (smoke test)")
    ap.add_argument("--workers", type=int, default=3, help="parallel scene processes")
    args = ap.parse_args()
    scenes = scene_list(args.quick)
    names = [n for n, _ in scenes]
    os.makedirs(DATA, exist_ok=True)
    if args.stage in ("all", "generate"):
        todo = [(n, p) for n, p in scenes if not os.path.exists(os.path.join(DATA, n, "scene.json"))]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for msg in ex.map(generate_scene, todo):
                print(msg, flush=True)
    if args.stage in ("all", "run"):
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for n, msg in ex.map(run_scene, names):
                print(f"run {n}: {msg}", flush=True)
    if args.stage in ("all", "robust"):
        targets = [n for n in names if split_of(n) == "test" and n.startswith("cherry") and "_g34_" in n
                   and ("_c30_" in n or "_c50_" in n)]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for n, msg in ex.map(run_robust, targets):
                print(f"robust {n}: {msg}", flush=True)
    if args.stage in ("all", "factorial"):
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for n, msg in ex.map(run_factorial, names):
                print(f"factorial {n}: {msg}", flush=True)
    if args.stage in ("all", "evaluate", "factorial"):
        out_dir = os.path.join(RESULTS, "quick" if args.quick else "synthetic")
        S = evaluate_all(names, out_dir)
        evaluate_factorial(names, out_dir)
        print(open(os.path.join(out_dir, "summary.md")).read())


if __name__ == "__main__":
    main()
