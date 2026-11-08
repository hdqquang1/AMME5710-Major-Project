#!/usr/bin/env python3
"""A-series experiments: baselines, ablations, view-count confound, variance,
count-metric sensitivity, truth noise floor, per-grape visibility and runtime.

    python run_ablations.py launch      # A3/A4 full-pipeline restores (subprocesses)
    python run_ablations.py a1 a2 ...   # in-process experiments
    python run_ablations.py a34         # evaluate A3/A4 restores

Every variant writes the same files as grape_restore_autoref.py, and is scored by
run_experiments.evaluate_run, so all rows are comparable with the main table.
Results are cached per folder; rerunning resumes.
"""

from __future__ import annotations

import csv
import functools
import json
import os
import pickle
import shutil
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

import grape_count as count
import grape_evaluate as ev
import grape_restore_autoref as gr
from grape_io import EXTENSIONS, load_sequence, natural_key
from grape_segment import colour_mask, load_grape_masks
from run_experiments import SCALE, TOLERANCE, evaluate_run, mask_metrics, register, restore

REPORT = Path("results/report")
OUT = Path("results/ablation")
WEIGHTS = "third_party/weights/point/best_val.pth"
TRUTH = cv2.imread("data/20261022_034827542_iOS.jpg")
TRUTH_FRUIT = cv2.imread("results/refactor_check/truth/t1/grape_masks/t1.png", 0) > 127
LABELS = ev.read_points("labels/t1.csv")
ROI = (0.03, 0.34, 0.97, 0.60)
BRANCHES = (3, 4, 5, 6)
ABLATE = (5, 4)  # Branch counts for component ablations, most informative first.


# ---------------------------------------------------------------- shared helpers
def capture_dir(b):
    return Path(f"data/2cm_74cm_{b}branches")


@functools.lru_cache(maxsize=1)
def capture(b):
    cap = capture_dir(b)
    names, images, allowed, size = load_sequence(cap / "images", SCALE, cap / "removal_masks", mask_dilate=0)
    sam = load_grape_masks(cap / "grape_masks", names, size, images[0].shape[:2])
    return names, images, allowed, sam


def base(b):
    d = json.loads((REPORT / f"b{b}_20mm" / "diagnostics.json").read_text(encoding="utf8"))
    return d["reference_1based"] - 1, [np.array(m) if m else None for m in d["source_to_reference_matrices"]]


@functools.lru_cache(maxsize=None)
def truth_matrix(folder):
    folder = Path(folder)
    ref = cv2.imread(str(folder / "reference.png"))
    fruit = (cv2.imread(str(folder / "reference_fruit_mask.png"), 0) > 127) | (
        cv2.imread(str(folder / "observed_mask.png"), 0) > 127
    )
    return register(TRUTH, TRUTH_FRUIT, ref, fruit)[0]


def base_matrix(b):
    return truth_matrix(str(REPORT / f"b{b}_20mm"))


def save(out, image, observed, reliable, reference, ref_fruit, meta):
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "grapes_rgb.png"), np.where(observed[..., None], image, 0).astype(np.uint8))
    cv2.imwrite(str(out / "observed_mask.png"), np.uint8(observed) * 255)
    cv2.imwrite(str(out / "analysis_mask.png"), np.uint8(reliable) * 255)
    cv2.imwrite(str(out / "reference.png"), reference)
    cv2.imwrite(str(out / "reference_fruit_mask.png"), np.uint8(ref_fruit) * 255)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf8")


def scored(out, matrix, extra):
    """Score a variant folder once; cached in row.json."""
    cache = out / "row.json"
    if cache.is_file():
        return {**json.loads(cache.read_text(encoding="utf8")), **extra}
    row = dict(extra)
    evaluate_run(out, TRUTH, TRUTH_FRUIT, matrix, LABELS, WEIGHTS, row, False)
    cache.write_text(json.dumps(row, indent=1, default=float), encoding="utf8")
    return row


def write_csv(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / f"{name}.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT / name}.csv ({len(rows)} rows)", flush=True)


def project(points, matrix):
    h = np.c_[points, np.ones(len(points))] @ matrix.T
    return h[:, :2] / h[:, 2:3]


def fuse_ours(images, allowed, targets, poses, ref, min_views=3, min_baseline=40, rms=30, select="best"):
    h, w = images[0].shape[:2]
    result = gr.fuse_views(images, allowed, targets, poses, ref, (0, 0, w, h), 192, min_views, 20,
                           min_baseline, None, rms, select)
    gr.clean_small_regions(result, 25)
    return result


def run_variant(out, b, images, allowed, targets, poses, ref, meta, **kw):
    if not (out / "grapes_rgb.png").is_file():
        t = time.time()
        r = fuse_ours(images, allowed, targets, poses, ref, **kw)
        save(out, r["image"], r["observed"], r["reliable"], images[ref], targets[ref], {**meta, **kw, "fusion_s": time.time() - t})
    return out


def warp_coords(inv, xx, yy):
    mx = inv[0, 0] * xx + inv[0, 1] * yy + inv[0, 2]
    my = inv[1, 0] * xx + inv[1, 1] * yy + inv[1, 2]
    if inv[2, 0] or inv[2, 1]:
        wz = inv[2, 0] * xx + inv[2, 1] * yy + inv[2, 2]
        mx, my = mx / wz, my / wz
    return mx.astype(np.float32), my.astype(np.float32)


def stack_stats(images, allowed, targets, poses, tile=256):
    """Per-pixel statistics of the aligned stack, for the classical baselines."""
    h, w = images[0].shape[:2]
    idx = [i for i, p in enumerate(poses) if p is not None]
    inv = [np.linalg.inv(poses[i]) for i in idx]
    fr_src = [(targets[i] & allowed[i]).astype(np.float32) for i in idx]
    leaf_src = [allowed[i].astype(np.float32) for i in idx]
    out = {k: np.zeros((h, w, 3), np.uint8) for k in ("mean", "median", "masked_mean", "fruit_rgb")}
    out.update(votes=np.zeros((h, w), np.uint16), frames=np.zeros((h, w), np.uint16))
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            yy, xx = np.mgrid[y : min(y + tile, h), x : min(x + tile, w)].astype(np.float32)
            cols, inb, fr, clear = [], [], [], []
            for k, M in enumerate(inv):
                mx, my = warp_coords(M, xx, yy)
                cols.append(cv2.remap(images[idx[k]], mx, my, cv2.INTER_LINEAR))
                inb.append((mx >= 0) & (mx < w - 1) & (my >= 0) & (my < h - 1))
                fr.append(cv2.remap(fr_src[k], mx, my, cv2.INTER_LINEAR) > 0.999)
                clear.append(cv2.remap(leaf_src[k], mx, my, cv2.INTER_LINEAR) > 0.999)
            C = np.asarray(cols, np.float32)
            V = np.asarray(inb)
            F = np.asarray(fr) & V
            L = np.asarray(clear) & V
            s = np.s_[y : y + C.shape[1], x : x + C.shape[2]]
            out["frames"][s] = V.sum(0)
            out["votes"][s] = F.sum(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out["mean"][s] = np.nan_to_num((C * V[..., None]).sum(0) / np.maximum(V.sum(0), 1)[..., None])
                out["masked_mean"][s] = np.nan_to_num((C * L[..., None]).sum(0) / np.maximum(L.sum(0), 1)[..., None])
                out["median"][s] = np.nan_to_num(np.nanmedian(np.where(V[..., None], C, np.nan), axis=0))
                out["fruit_rgb"][s] = np.nan_to_num(np.nanmedian(np.where(F[..., None], C, np.nan), axis=0))
    return out


def clean(mask, area=25):
    n, lab, st, _ = cv2.connectedComponentsWithStats(np.uint8(mask), 8)
    keep = st[:, cv2.CC_STAT_AREA] >= area
    keep[0] = False
    return keep[lab]


def sam_fruit(bgr):
    """SAM 3 fruit mask of an image, prompt and threshold as for the truth photo."""
    from PIL import Image

    from grape_segment import FRUIT_THRESHOLD, _sam3_processor, _sam3_union

    return _sam3_union(_sam3_processor(), Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), ["cherry"], FRUIT_THRESHOLD)


def depth_shift_poses(images, allowed, sam, ref, n):
    """The prototype: one global horizontal shift per rail step, chosen by autofocus
    on the reference fruit region (no features, no per-view fit)."""
    s = 0.25
    small = [cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for im in images]
    region = cv2.dilate(np.uint8(cv2.resize(np.uint8(sam[ref]), small[0].shape[1::-1]) > 0), np.ones((31, 31), np.uint8)) > 0
    h, w = small[0].shape[:2]

    def focus(d):
        acc = np.zeros((h, w, 3), np.float32)
        cnt = np.zeros((h, w), np.float32)
        for i in range(n):
            M = np.float32([[1, 0, d * s * (ref - i)], [0, 1, 0]])
            acc += cv2.warpAffine(small[i], M, (w, h)).astype(np.float32)
            cnt += cv2.warpAffine(np.ones((h, w), np.float32), M, (w, h))
        mean = acc / np.maximum(cnt, 1)[..., None]
        lap = cv2.Laplacian(cv2.cvtColor(np.uint8(mean), cv2.COLOR_BGR2GRAY), cv2.CV_32F)
        return float(lap[region].var())

    coarse = {d: focus(d) for d in np.arange(-80, 81, 2.0)}
    d0 = max(coarse, key=coarse.get)
    fine = {d: focus(d) for d in np.arange(d0 - 2, d0 + 2.01, 0.25)}
    d = max(fine, key=fine.get)
    poses = []
    for i in range(n):
        M = np.eye(3)
        M[0, 2] = d * (ref - i)
        poses.append(M)
    return poses, float(d)


# ---------------------------------------------------------------- A1 baselines
def a1():
    rows = []
    for b in BRANCHES:
        names, images, allowed, sam = capture(b)
        ref, poses = base(b)
        targets = [t & v for t, v in zip(sam, allowed)]
        M = base_matrix(b)
        root = OUT / "a1" / f"b{b}"
        common = {"branches": b}
        rows.append(scored(REPORT / f"b{b}_20mm", M, {**common, "method": "ours"}))
        n_views = sum(p is not None for p in poses)
        # Classical statistics on our alignment (isolates fusion), then on depth-shift alignment (the prototype).
        dpose, disparity = None, None
        for align in ("features", "depth_shift"):
            if align == "features":
                P = poses
            else:
                if dpose is None:
                    dpose, disparity = depth_shift_poses(images, allowed, sam, ref, len(images))
                P = dpose
            tag = root / align
            if not (tag / "stats.npz").is_file():
                tag.mkdir(parents=True, exist_ok=True)
                st = stack_stats(images, allowed, sam, P)
                np.savez_compressed(tag / "stats.npz", **st)
            st = dict(np.load(tag / "stats.npz"))
            if align == "features":
                for method, mask in (("mask_union", st["votes"] >= 1), ("mask_majority", st["votes"] * 2 > st["frames"])):
                    out = root / method
                    if not (out / "grapes_rgb.png").is_file():
                        m = clean(mask)
                        save(out, st["fruit_rgb"], m, m, images[ref], targets[ref], {"method": method})
                    rows.append(scored(out, M, {**common, "method": method, "views": n_views}))
            for stat in ("mean", "median", "masked_mean"):
                method = f"{align}_{stat}"
                out = root / method
                if not (out / "grapes_rgb.png").is_file():
                    m = clean(sam_fruit(st[stat]))
                    save(out, st[stat], m, m, images[ref], targets[ref], {"method": method, "disparity_px_per_step": disparity})
                    cv2.imwrite(str(out / "fused_full.jpg"), st[stat])
                rows.append(scored(out, M, {**common, "method": method, "views": len(images) if align == "depth_shift" else n_views,
                                            "disparity_px_per_step": disparity}))
        # Segmentation sanity: SAM 3 run directly on the single reference view.
        out = root / "single_sam_on_image"
        if not (out / "grapes_rgb.png").is_file():
            m = clean(sam_fruit(images[ref]))
            save(out, images[ref], m, m, images[ref], targets[ref], {"method": "single view, SAM 3 cherry on image"})
        rows.append(scored(out, M, {**common, "method": "single_sam_on_image", "views": 1}))
        write_csv("a1", rows)
    return rows


# ---------------------------------------------------------------- A2 ablations
def edges_for(b, kind):
    path = OUT / "cache" / f"edges_b{b}_{kind}.pkl"
    if path.is_file():
        return pickle.loads(path.read_bytes())
    names, images, allowed, sam = capture(b)
    if kind == "fruit":
        masks = [t & v for t, v in zip(sam, allowed)]
    else:  # whole image, leaves included
        masks = [np.ones_like(v) for v in allowed]
    t = time.time()
    edges, summaries = gr.build_edges(images, masks, 4, 12)
    data = (edges, summaries, time.time() - t)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(data))
    return data


def validate(n, edges, ref, poses, max_error=3.0):
    """Same per-view held-out check as solve_motion, for any pose model."""
    per = {i: [] for i in range(n) if poses[i] is not None}
    for e in edges:
        if poses[e.i] is None or poses[e.j] is None:
            continue
        err = np.linalg.norm(project(e.check_source, poses[e.i]) - project(e.check_target, poses[e.j]), axis=1)
        per[e.i].extend(err)
        per[e.j].extend(err)
    bad = {i for i in per if i != ref and (not per[i] or np.median(per[i]) > max_error)}
    keep = gr.connected_views(n, [e for e in edges if e.i in per and e.j in per and e.i not in bad and e.j not in bad], ref)
    allerr = [x for i in per for x in per[i]]
    return [poses[i] if i in keep else None for i in range(n)], float(np.median(allerr)) if allerr else None


def chained(n, edges, ref, model):
    T = {ref: np.eye(3)}
    order = sorted(edges, key=lambda e: -len(e.source))
    grown = True
    while grown:
        grown = False
        for e in order:
            if (e.i in T) == (e.j in T):
                continue
            if model == "similarity":
                A, _ = cv2.estimateAffinePartial2D(e.source, e.target, method=cv2.LMEDS)
                H = np.vstack([A, [0, 0, 1]])
            else:
                H, _ = cv2.findHomography(e.source, e.target, cv2.LMEDS)
            if e.j in T:
                T[e.i] = T[e.j] @ H
            else:
                T[e.j] = T[e.i] @ np.linalg.inv(H)
            grown = True
            break
    return [T.get(i) for i in range(n)]


def translation_joint(n, edges, ref):
    act = gr.connected_views(n, edges, ref)
    slots = {i: k for k, i in enumerate(i for i in act if i != ref)}
    A, B = [], []
    for e in edges:
        if e.i not in act or e.j not in act:
            continue
        t = np.median(e.target - e.source, axis=0)  # x_j = x_i + t  ->  T_i - T_j = t
        for axis in (0, 1):
            row = np.zeros(2 * len(slots))
            if e.i != ref:
                row[2 * slots[e.i] + axis] = 1
            if e.j != ref:
                row[2 * slots[e.j] + axis] = -1
            A.append(row)
            B.append(t[axis])
    sol = np.linalg.lstsq(np.asarray(A), np.asarray(B), rcond=None)[0]
    poses = [None] * n
    poses[ref] = np.eye(3)
    for i, k in slots.items():
        M = np.eye(3)
        M[:2, 2] = sol[2 * k : 2 * k + 2]
        poses[i] = M
    return poses


def a2():
    """A2_PART=k/K fuses only part k (no GPU); without it, fuses what is missing and scores all."""
    part = os.environ.get("A2_PART")
    branches = [int(x) for x in os.environ.get("A2_BRANCHES", ",".join(map(str, ABLATE))).split(",")]
    rows = []
    for b in branches:
        names, images, allowed, sam = capture(b)
        ref, poses = base(b)
        targets = [t & v for t, v in zip(sam, allowed)]
        root = OUT / "a2" / f"b{b}"
        n = len(images)
        colour = [colour_mask(im, ROI, "red") & v for im, v in zip(images, allowed)]
        edges = {}

        def E(kind):
            if kind not in edges:
                edges[kind] = edges_for(b, kind)[0]
            return edges[kind]

        def align(kind):
            fe = E("fruit")
            if kind == "whole":
                return gr.solve_motion(n, E("whole"), ref, images[0].shape[:2], 3.0)[0]
            if kind == "novalid":
                return gr.solve_motion(n, fe, ref, images[0].shape[:2], 1e9)[0]
            if kind == "chain_sim":
                return validate(n, fe, ref, chained(n, fe, ref, "similarity"))[0]
            if kind == "chain_h":
                return validate(n, fe, ref, chained(n, fe, ref, "homography"))[0]
            if kind == "trans":
                return validate(n, fe, ref, translation_joint(n, fe, ref))[0]

        V = []  # (group, variant, tag, kwargs, pose kind or None, allowed, targets)
        for m in (1, 2, 3, 5):
            V.append(("min views (B=0)", f"m={m}", f"m{m}_b0", dict(min_views=m, min_baseline=0), None, allowed, targets))
        for bl in (0, 20, 80):
            V.append(("min span (m=3)", f"B={bl} mm", f"m3_b{bl}", dict(min_baseline=bl), None, allowed, targets))
        for r in (15, 60, 1e9):
            V.append(("colour consensus", "rms=off" if r > 1e6 else f"rms={r}", f"rms{int(min(r, 999))}", dict(rms=r), None, allowed, targets))
        V.append(("output colour", "median blend", "blend", dict(select="median"), None, allowed, targets))
        V.append(("no consensus at all", "m=1, B=0, rms=off", "none", dict(min_views=1, min_baseline=0, rms=1e9), None, allowed, targets))
        V.append(("masks", "colour (red) fruit mask", "mask_colour", {}, None, allowed, colour))
        V.append(("masks", "no fruit mask", "mask_none", {}, None, allowed, [v.copy() for v in allowed]))
        V.append(("masks", "no foliage mask", "mask_nofoliage", {}, None, [np.ones_like(v) for v in allowed], sam))
        for name, kind in (("whole-image SIFT", "whole"), ("no held-out view check", "novalid"), ("chained similarity", "chain_sim"),
                           ("chained homography", "chain_h"), ("joint translation only", "trans")):
            V.append(("alignment", name, f"align_{kind}", {}, kind, allowed, targets))
        if part:
            k, K = map(int, part.split("/"))
            plain = [v for v in V if v[4] is None]
            mine = [v for v in V if v[4] is not None] if k == K - 1 else plain[k :: K - 1]
            for g, var, tag, kw, kind, A_, T_ in mine:
                if (root / tag / "grapes_rgb.png").is_file():
                    continue
                try:
                    P = align(kind) if kind else poses
                except ValueError as exc:
                    (root / tag).mkdir(parents=True, exist_ok=True)
                    (root / tag / "FAILED.txt").write_text(str(exc), encoding="utf8")
                    continue
                run_variant(root / tag, b, images, A_, T_, P, ref, {"group": g, "variant": var, "pose": kind}, **kw)
                print(f"b{b} fused {tag}", flush=True)
            continue
        M = base_matrix(b)
        common = {"branches": b}
        rows.append(scored(REPORT / f"b{b}_20mm", M, {**common, "group": "full", "variant": "ours (m=3, B=40, rms=30, copy best)",
                                                      "views": sum(p is not None for p in poses)}))
        for g, var, tag, kw, kind, A_, T_ in V:
            row = {**common, "group": g, "variant": var}
            if (root / tag / "FAILED.txt").is_file():
                rows.append({**row, "status": "failed: " + (root / tag / "FAILED.txt").read_text(encoding="utf8")[:150]})
                continue
            P = align(kind) if kind else poses
            out = run_variant(root / tag, b, images, A_, T_, P, ref, {"group": g, "variant": var, "pose": kind}, **kw)
            r = scored(out, M, {**row, "views": sum(p is not None for p in P)})
            if kind:
                r["held_out_median_px"] = validate(n, E("fruit"), ref, P, 1e9)[1]
                r["pairs_ok"] = len(E("whole" if kind == "whole" else "fruit"))
            rows.append(r)
            print(f"  b{b} {g}: {var} -> recall {r['result_recall']:.3f} prec {r['result_precision']:.3f} F1 {r['gs_result_f1']:.2f}", flush=True)
        rows[-len(V) - 1]["held_out_median_px"] = validate(n, E("fruit"), ref, poses, 1e9)[1]
        rows[-len(V) - 1]["pairs_ok"] = len(E("fruit"))
        write_csv("a2", rows)
    return rows


# ---------------------------------------------------------------- A3/A4 launches
def names_of(b):
    return sorted((p for p in (capture_dir(b) / "images").iterdir() if p.suffix.lower() in EXTENSIONS), key=lambda p: natural_key(p.name))


def window(n, ref, step, count_):
    """count_ views at spacing step that include ref, as centred on ref as the sequence allows."""
    grid = [i for i in range(ref % step, n, step)]
    best = None
    for start in range(len(grid) - count_ + 1):
        idx = grid[start : start + count_]
        if ref in idx:
            off = abs((idx[0] + idx[-1]) / 2 - ref)
            if best is None or off < best[0]:
                best = (off, idx)
    return best[1] if best else None


def link_subset(b, idx, tag):
    folder = OUT / "subsets" / tag
    folder.mkdir(parents=True, exist_ok=True)
    files = names_of(b)
    for i in idx:
        t = folder / files[i].name
        if not t.exists():
            try:
                os.link(files[i], t)
            except OSError:
                shutil.copy2(files[i], t)
    return folder


def a34_jobs():
    jobs = []
    for b in BRANCHES:
        ref, _ = base(b)
        n = len(names_of(b))
        cap = capture_dir(b)
        # A3: fixed N=9 at 20 and 40 mm (60 mm already has 9 views).
        sets = [(20, 1, 9), (40, 2, 9)]
        if b in (4, 5):
            sets += [(20, 1, c) for c in (3, 5, 7, 13, 17, 21)] + [(20, 1, 25), (40, 2, 13)]
        for mm, step, c in sets:
            idx = window(n, ref, step, c)
            if idx is None or len(idx) != c:
                print(f"skip b{b} {mm}mm N={c}: window {idx}", flush=True)
                continue
            tag = f"b{b}_{mm}mm_n{c}"
            jobs.append(("a3", b, tag, mm, idx, idx.index(ref) + 1, cap))
        # A4: other reference views on the full 20 mm sequence.
        for d in (-8, -4, 4, 8):
            r = ref + d
            if 0 <= r < n:
                jobs.append(("a4", b, f"b{b}_20mm_ref{r + 1}", 20, list(range(n)), r + 1, cap))
    return jobs


def launch(workers=4):
    def go(job):
        kind, b, tag, mm, idx, ref1, cap = job
        folder = cap / "images" if kind == "a4" else link_subset(b, idx, tag)
        out = OUT / kind / tag
        t = time.time()
        why = restore(folder, out, mm, ref1, cap / "grape_masks", cap / "removal_masks")
        print(f"{tag}: {'FAILED ' + why if why else 'ok'} ({time.time() - t:.0f}s)", flush=True)

    jobs = a34_jobs()
    print(f"{len(jobs)} restore jobs", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(go, jobs))


def a34():
    rows = []
    for kind, b, tag, mm, idx, ref1, cap in a34_jobs():
        out = OUT / kind / tag
        row = {"exp": kind, "branches": b, "spacing_mm": mm, "n_views": len(idx), "span_mm": (max(idx) - min(idx)) * 20,
               "ref_1based": ref1 if kind == "a3" else ref1, "tag": tag}
        if (out / "FAILED.txt").is_file():
            row["status"] = "failed: " + (out / "FAILED.txt").read_text(encoding="utf8")[:200]
            rows.append(row)
            continue
        if not (out / "diagnostics.json").is_file():
            row["status"] = "missing"
            rows.append(row)
            continue
        meta = json.loads((out / "diagnostics.json").read_text(encoding="utf8"))
        row["views_kept"] = len(meta["registration"]["connected_views_1based"])
        try:
            M = base_matrix(b) if kind == "a3" else truth_matrix(str(out))
        except ValueError as exc:
            row["status"] = f"truth registration failed: {exc}"[:200]
            rows.append(row)
            continue
        row["status"] = "ok"
        r = scored(out, M, row)
        r.update(fruit_floor_ceiling(out, M))
        rows.append(r)
    # Add the main-table runs so curves include them.
    for b in BRANCHES:
        for mm in (20, 40, 60):
            out = REPORT / f"b{b}_{mm}mm"
            meta = json.loads((out / "diagnostics.json").read_text(encoding="utf8"))
            n = len(meta["views"])
            r = scored(out, base_matrix(b), {"exp": "main", "branches": b, "spacing_mm": mm, "n_views": n,
                                             "span_mm": (n - 1) * mm, "tag": out.name, "status": "ok",
                                             "views_kept": len(meta["registration"]["connected_views_1based"])})
            r.update(fruit_floor_ceiling(out, base_matrix(b)))
            rows.append(r)
    write_csv("a34", rows)
    return rows


def fruit_floor_ceiling(out, M):
    """Fruit-only floor and ceiling GrapeSAM F1, as in the main count table."""
    ref = cv2.imread(str(out / "reference.png"))
    h, w = ref.shape[:2]
    warped = cv2.warpAffine(TRUTH, M, (w, h))
    roi = cv2.warpAffine(np.float32(TRUTH_FRUIT), M, (w, h)) > 0.5
    labels = [tuple(M[:, :2] @ np.array(p) + M[:, 2]) for p in LABELS]
    floor_mask = cv2.imread(str(out / "reference_fruit_mask.png"), 0) > 127
    res = {}
    for name, im, m in (("floor_fruit", ref, floor_mask), ("ceiling_fruit", warped, roi)):
        im = im.copy()
        im[~m] = 0
        det = count.count_grapesam(im, WEIGHTS)
        r = ev.match_points([(q["x_px"], q["y_px"]) for q in det], labels, TOLERANCE)
        res[f"gs_{name}_f1"] = r["f1"]
        res[f"gs_{name}_n"] = r["predicted_count"]
    return res


# ---------------------------------------------------------------- A5/A6/A4-bootstrap: count metric checks
def panels(b, M):
    out = REPORT / f"b{b}_20mm"
    ref = cv2.imread(str(out / "reference.png"))
    image = cv2.imread(str(out / "grapes_rgb.png"))
    reliable = cv2.imread(str(out / "analysis_mask.png"), 0) > 127
    floor_mask = cv2.imread(str(out / "reference_fruit_mask.png"), 0) > 127
    observed = cv2.imread(str(out / "observed_mask.png"), 0) > 127
    h, w = ref.shape[:2]
    warped = cv2.warpAffine(TRUTH, M, (w, h))
    roi = cv2.warpAffine(np.float32(TRUTH_FRUIT), M, (w, h)) > 0.5
    f = ref.copy()
    f[~floor_mask] = 0
    r = image.copy()
    r[~reliable] = 0
    c = warped.copy()
    c[~roi] = 0
    return {"single": f, "restored": r, "leaf_free": c}, floor_mask, observed, roi


def match_detail(preds, truth, r):
    cands = sorted((np.hypot(p[0] - t[0], p[1] - t[1]), a, k) for a, p in enumerate(preds) for k, t in enumerate(truth)
                   if np.hypot(p[0] - t[0], p[1] - t[1]) <= r)
    mp, mt = set(), set()
    for _, a, k in cands:
        if a not in mp and k not in mt:
            mp.add(a)
            mt.add(k)
    hit = np.zeros(len(truth), bool)
    hit[list(mt)] = True
    return hit, len(preds) - len(mp)


def f1_of(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)


def a5():
    """Tolerance sweep, threshold sweep (PR/AP), count error, and bootstrap CIs over grapes."""
    rows_tol, rows_pr, rows_boot = [], [], []
    rng = np.random.default_rng(0)
    for b in BRANCHES:
        M = base_matrix(b)
        labels = [tuple(M[:, :2] @ np.array(p) + M[:, 2]) for p in LABELS]
        ims, *_ = panels(b, M)
        for name, im in ims.items():
            det = count.count_grapesam(im, WEIGHTS, threshold=0.01)
            pts = [(d["x_px"], d["y_px"]) for d in det]
            scores = np.array([d["score"] for d in det])
            at = [p for p, s in zip(pts, scores) if s > 0.05]
            for tol in (8, 12, 16, 20, 24, 32, 40):
                hit, fp = match_detail(at, labels, tol)
                rows_tol.append({"branches": b, "input": name, "tol_px": tol, "f1": f1_of(hit.sum(), fp, len(labels) - hit.sum()),
                                 "recall": hit.mean(), "n": len(at)})
            prec, rec = [], []
            for thr in np.r_[np.arange(0.01, 0.1, 0.01), np.arange(0.1, 0.95, 0.05)]:
                sel = [p for p, s in zip(pts, scores) if s > thr]
                hit, fp = match_detail(sel, labels, TOLERANCE)
                p_ = hit.sum() / max(len(sel), 1) if sel else 1.0
                prec.append(p_)
                rec.append(hit.mean())
                rows_pr.append({"branches": b, "input": name, "threshold": round(float(thr), 3), "precision": p_, "recall": hit.mean(),
                                "f1": f1_of(hit.sum(), fp, len(labels) - hit.sum()), "n": len(sel), "count_abs_err": abs(len(sel) - len(labels))})
            # AP: all-point interpolated, recall ascending.
            order = np.argsort(rec)
            r_ = np.r_[0, np.asarray(rec)[order]]
            p_ = np.r_[1, np.asarray(prec)[order]]
            p_ = np.maximum.accumulate(p_[::-1])[::-1]
            ap = float(np.sum(np.diff(r_) * p_[1:]))
            # Bootstrap over grapes at the reported operating point.
            hit, fp = match_detail(at, labels, TOLERANCE)
            boots = []
            for _ in range(2000):
                k = rng.integers(0, len(labels), len(labels))
                tp = hit[k].sum()
                boots.append(f1_of(tp, fp, len(labels) - tp))
            rows_boot.append({"branches": b, "input": name, "f1": f1_of(hit.sum(), fp, len(labels) - hit.sum()),
                              "f1_lo": float(np.percentile(boots, 2.5)), "f1_hi": float(np.percentile(boots, 97.5)),
                              "ap": ap, "n": len(at), "count_abs_err": abs(len(at) - len(labels))})
            print(f"b{b} {name}: F1 {rows_boot[-1]['f1']:.2f} [{rows_boot[-1]['f1_lo']:.2f},{rows_boot[-1]['f1_hi']:.2f}] AP {ap:.2f}", flush=True)
            # Paired bootstrap of the restored-minus-single gain.
            if name == "single":
                single_hit, single_fp = hit, fp
            if name == "restored":
                gains = []
                for _ in range(2000):
                    k = rng.integers(0, len(labels), len(labels))
                    gains.append(f1_of(hit[k].sum(), fp, len(labels) - hit[k].sum()) - f1_of(single_hit[k].sum(), single_fp, len(labels) - single_hit[k].sum()))
                rows_boot[-1].update(gain_lo=float(np.percentile(gains, 2.5)), gain_hi=float(np.percentile(gains, 97.5)),
                                     gain_p_le0=float(np.mean(np.asarray(gains) <= 0)))
    write_csv("a5_tolerance", rows_tol)
    write_csv("a5_pr", rows_pr)
    write_csv("a5_bootstrap", rows_boot)


def a6():
    """Truth-registration noise floor: perturb the truth transform and re-score."""
    rows = []
    for b in BRANCHES:
        M0 = base_matrix(b)
        out = REPORT / f"b{b}_20mm"
        image = cv2.imread(str(out / "grapes_rgb.png"))
        reliable = cv2.imread(str(out / "analysis_mask.png"), 0) > 127
        observed = cv2.imread(str(out / "observed_mask.png"), 0) > 127
        floor_mask = cv2.imread(str(out / "reference_fruit_mask.png"), 0) > 127
        ref = cv2.imread(str(out / "reference.png"))
        h, w = ref.shape[:2]
        r = image.copy()
        r[~reliable] = 0
        f = ref.copy()
        f[~floor_mask] = 0
        det_r = [(d["x_px"], d["y_px"]) for d in count.count_grapesam(r, WEIGHTS)]
        det_f = [(d["x_px"], d["y_px"]) for d in count.count_grapesam(f, WEIGHTS)]
        ys, xs = np.where(observed | floor_mask)
        cx, cy = xs.mean(), ys.mean()
        perturb = [("none", 0, 0, 0, 1)]
        for rad in (1, 2, 3, 5):
            for a in range(8):
                perturb.append((f"shift{rad}", rad * np.cos(a * np.pi / 4), rad * np.sin(a * np.pi / 4), 0, 1))
        for deg in (-1, -0.5, 0.5, 1):
            perturb.append(("rot", 0, 0, deg, 1))
        for sc in (0.99, 1.01):
            perturb.append(("scale", 0, 0, 0, sc))
        for kind, dx, dy, deg, sc in perturb:
            P = cv2.getRotationMatrix2D((cx, cy), deg, sc)
            P[:, 2] += (dx, dy)
            M = np.vstack([P, [0, 0, 1]]) @ np.vstack([M0, [0, 0, 1]])
            M = M[:2]
            roi = cv2.warpAffine(np.float32(TRUTH_FRUIT), M, (w, h)) > 0.5
            labels = [tuple(M[:, :2] @ np.array(p) + M[:, 2]) for p in LABELS]
            row = {"branches": b, "kind": kind, "dx": dx, "dy": dy, "deg": deg, "scale": sc}
            for name, m in (("single", floor_mask), ("restored", observed)):
                for k, v in mask_metrics(m, roi).items():
                    row[f"{name}_{k}"] = v
            for name, det in (("single", det_f), ("restored", det_r)):
                row[f"{name}_f1"] = ev.match_points(det, labels, TOLERANCE)["f1"]
            if kind in ("none", "shift3", "rot"):
                c = cv2.warpAffine(TRUTH, M, (w, h))
                c[~roi] = 0
                row["leaf_free_f1"] = ev.match_points([(d["x_px"], d["y_px"]) for d in count.count_grapesam(c, WEIGHTS)], labels, TOLERANCE)["f1"]
            rows.append(row)
        print(f"b{b} perturbations done", flush=True)
    write_csv("a6", rows)


# ---------------------------------------------------------------- A7 visibility
def a7(radius=10):
    rows = []
    for b in BRANCHES:
        names, images, allowed, sam = capture(b)
        ref, poses = base(b)
        M = base_matrix(b)
        out = REPORT / f"b{b}_20mm"
        observed = cv2.imread(str(out / "observed_mask.png"), 0) > 127
        reliable = cv2.imread(str(out / "analysis_mask.png"), 0) > 127
        image = cv2.imread(str(out / "grapes_rgb.png"))
        r = image.copy()
        r[~reliable] = 0
        f = images[ref].copy()
        f[~(sam[ref] & allowed[ref])] = 0
        labels = [tuple(M[:, :2] @ np.array(p) + M[:, 2]) for p in LABELS]
        hit_r, _ = match_detail([(d["x_px"], d["y_px"]) for d in count.count_grapesam(r, WEIGHTS)], labels, TOLERANCE)
        hit_f, _ = match_detail([(d["x_px"], d["y_px"]) for d in count.count_grapesam(f, WEIGHTS)], labels, TOLERANCE)
        h, w = observed.shape
        disc = np.zeros((2 * radius + 1,) * 2, np.uint8)
        cv2.circle(disc, (radius, radius), radius, 1, -1)
        disc = disc > 0

        def cover(mask, x, y):
            x, y = int(round(x)), int(round(y))
            if not (radius <= x < mask.shape[1] - radius and radius <= y < mask.shape[0] - radius):
                return 0.0
            return float(mask[y - radius : y + radius + 1, x - radius : x + radius + 1][disc].mean())

        for k, (x, y) in enumerate(labels):
            vis = 0
            for i, P in enumerate(poses):
                if P is None:
                    continue
                sx, sy = project(np.array([[x, y]]), np.linalg.inv(P))[0]
                if cover(sam[i] & allowed[i], sx, sy) >= 0.5:
                    vis += 1
            rows.append({"branches": b, "grape": k, "views_aligned": sum(p is not None for p in poses), "views_visible": vis,
                         "visible_in_reference": cover(sam[ref] & allowed[ref], x, y) >= 0.5,
                         "restored_pixel_cover": cover(observed, x, y), "restored_counted": bool(hit_r[k]),
                         "single_counted": bool(hit_f[k])})
        print(f"b{b} visibility done", flush=True)
    write_csv("a7", rows)


# ---------------------------------------------------------------- A8 runtime
def a8():
    b = 5
    cap = capture_dir(b)
    t = {}
    s = time.time()
    names, images, allowed, size = load_sequence(cap / "images", SCALE, cap / "removal_masks", mask_dilate=0)
    sam = load_grape_masks(cap / "grape_masks", names, size, images[0].shape[:2])
    targets = [a & v for a, v in zip(sam, allowed)]
    t["load images and masks"] = time.time() - s
    s = time.time()
    edges, summ = gr.build_edges(images, targets, 4, 12)
    t["SIFT and pairwise RANSAC"] = time.time() - s
    s = time.time()
    groups = gr.validated_components(len(images), edges)
    ref, poses, active, reg = gr.select_verified_reference(len(images), edges, images, 3, 3.0, groups)
    t["joint pose solve and reference"] = time.time() - s
    s = time.time()
    r = fuse_ours(images, allowed, targets, poses, ref)
    t["fusion"] = time.time() - s
    s = time.time()
    im = r["image"].copy()
    im[~r["reliable"]] = 0
    count.count_grapesam(im, WEIGHTS)  # warm-up
    s = time.time()
    count.count_grapesam(im, WEIGHTS)
    t["GrapeSAM count (one image)"] = time.time() - s
    from grape_io import read_rgb
    from PIL import Image

    from grape_segment import FOLIAGE_PROMPTS, FOLIAGE_THRESHOLD, FRUIT_THRESHOLD, _sam3_processor, _sam3_union

    proc = _sam3_processor()
    img = Image.fromarray(read_rgb(names_of(b)[0].read_bytes()))
    _sam3_union(proc, img, ["grape"], FRUIT_THRESHOLD)  # warm-up
    s = time.time()
    _sam3_union(proc, img, ["grape"], FRUIT_THRESHOLD)
    _sam3_union(proc, img, FOLIAGE_PROMPTS, FOLIAGE_THRESHOLD)
    t["SAM 3 masks (one view, fruit + foliage)"] = time.time() - s
    import torch

    info = {"stage_seconds": t, "views": len(images), "processing_size": list(images[0].shape[1::-1]),
            "gpu": torch.cuda.get_device_name(0), "cpu_threads": os.cpu_count(), "pairs_ok": len(edges), "pairs": len(summ)}
    (OUT / "a8_runtime.json").write_text(json.dumps(info, indent=2), encoding="utf8")
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    steps = {"launch": launch, "a1": a1, "a2": a2, "a34": a34, "a5": a5, "a6": a6, "a7": a7, "a8": a8}
    for arg in sys.argv[1:]:
        print(f"=== {arg}", flush=True)
        steps[arg]()
