"""
run_real.py - occlusion-aware synthetic aperture and colour-free counting on a real rail capture.

Run from any folder.  With no arguments it searches for captures in the
team's data folders (repo data/ and has_sai/data/real/) and processes each:

    python3 run_real.py
    python3 run_real.py --images "../data/Sequence 1/images" --spacing 0.05
    python3 run_real.py --images photos/ --clean photos_no_foliage/ --labels labels_ref.csv

A capture is a folder of photos taken left to right along the rail (file
names sort in rail order).  Poses come from, in order of preference:
  1. a COLMAP sparse model found inside the capture folder (cameras/images .bin
     or .txt; the dense/0 undistorted layout is preferred when present),
  2. the OpenCV rail structure from motion of sai/rail_sfm.py, with K from
     --fx-full, else the EXIF 35 mm-equivalent focal length, else 26 mm.
The rail spacing (--spacing, default 0.05 m) fixes the metric scale.

Outputs (results/real/<capture>/): leaf-free image, depth map, foliage masks,
detection overlay, detections.csv (pixel and 3-D positions), a coloured point
cloud of the leaf-free surface (leaf_free_cloud.ply) and summary.json.
With --clean (the same rail positions without foliage) the leaf-free image is
compared with the clean centre photo (PSNR/SSIM after ECC alignment) and the
clean capture is counted too.  With --labels (CSV of x_px, y_px in
full-resolution pixels of the centre photo) precision/recall/F1 are reported.

Dependencies: numpy, opencv-python.
"""

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sai import camera, pipeline, detect, evaluate, rail_sfm  # noqa: E402

REPO = os.path.dirname(HERE)


def find_captures():
    """Folders under data/ that hold at least 3 photos (COLMAP image folders preferred)."""
    roots = [os.path.join(REPO, "data"), os.path.join(HERE, "data", "real")]
    found = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if not os.path.isdir(d):
                continue
            for cand in (os.path.join(d, "dense", "0", "images"), os.path.join(d, "images"), d):
                if os.path.isdir(cand) and len(camera.list_images(cand)) >= 3:
                    found.append((os.path.basename(d.rstrip("/")), cand, d))
                    break
    return found


def build_rig(image_dir, search_root, scale, spacing, fx_full, ref, poses, args=None):
    """Rig with poses from COLMAP (if available) or the OpenCV rail SfM."""
    info = {}
    model = None
    if poses in ("auto", "colmap"):
        dense = os.path.join(os.path.dirname(image_dir), "sparse")
        if os.path.isdir(dense) and camera.find_colmap_model(dense):
            model = camera.find_colmap_model(dense)
        else:
            model = camera.find_colmap_model(search_root)
    if model is not None:
        rig = camera.rig_from_colmap(image_dir, model, scale=scale, ref=ref, spacing=spacing)
        info["poses"] = f"COLMAP model {os.path.relpath(model, REPO)}"
        return rig, info
    if poses == "colmap":
        raise FileNotFoundError("no COLMAP model found")
    paths = camera.list_images(image_dir)
    first = cv2.imread(paths[0], cv2.IMREAD_COLOR)
    H, W = first.shape[:2]
    if fx_full:
        K = np.array([[fx_full, 0, (W - 1) / 2], [0, fx_full, (H - 1) / 2], [0, 0, 1]], np.float64)
        info["K_source"] = "--fx-full"
    else:
        f35 = camera.exif_focal_35mm(paths[0])
        info["K_source"] = f"EXIF 35 mm equivalent {f35} mm" if f35 else "default 26 mm equivalent (no EXIF)"
        K = camera.K_from_35mm(f35 or 26.0, W, H)
    views = [camera.View(os.path.basename(p), camera.read_image(p, scale), camera.scale_K(K, scale),
                         np.eye(3), np.zeros(3)) for p in paths]
    ref = len(views) // 2 if ref is None else ref
    if poses == "nominal":
        # pure translation along the camera X axis at the nominal spacing (no estimation)
        for i, v in enumerate(views):
            v.t = -np.array([(i - ref) * spacing, 0.0, 0.0])
        info["poses"] = "nominal rail (pure X translation)"
    else:
        sfm_scale = min(1.0, 2000.0 / max(W, H))   # SIFT on ~2000 px wide photos
        grays = [cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), None, fx=sfm_scale, fy=sfm_scale,
                            interpolation=cv2.INTER_AREA) for p in paths]
        r = rail_sfm.estimate_rail_poses(grays, camera.scale_K(K, sfm_scale), spacing, ref=ref,
                                         max_pair_gap=args.sfm_gap, n_features=args.sfm_features)
        for v, R, t in zip(views, r["R"], r["t"]):
            v.R, v.t = R, t
        info["poses"] = "OpenCV rail SfM (translation-only)"
        info["sfm"] = r["diagnostics"]
    info["fx_full_px"] = float(K[0, 0])
    return camera.Rig(views, ref), info


def rig_with_poses(image_dir, rig, scale):
    """
    Rig of a second pass at the same rail positions (e.g. the foliage-free pass),
    reusing the poses and intrinsics of `rig`.  The carriage stops at the same
    tape marks, so re-estimating poses is unnecessary, and the foliage-free
    pass is a hard case for structure from motion (the repetitive lattice
    produces matches that are consistent with the epipolar geometry but wrong).
    """
    paths = camera.list_images(image_dir)
    if len(paths) != len(rig.views):
        raise ValueError(f"{image_dir} has {len(paths)} photos, the main pass has {len(rig.views)}")
    views = [camera.View(os.path.basename(p), camera.read_image(p, scale), v.K, v.R, v.t)
             for p, v in zip(paths, rig.views)]
    return camera.Rig(views, rig.ref)


def default_threshold(fruit):
    """Score threshold calibrated on the synthetic calibration scenes, if available."""
    p = os.path.join(HERE, "results", "synthetic", "summary.json")
    try:
        t = json.load(open(p))["thresholds"]["ours"]["per_fruit"][fruit]
        if t is not None:
            return float(t), "synthetic calibration (results/synthetic/summary.json)"
    except (OSError, KeyError, ValueError):
        pass
    return 15.0, "default 15 mm (no synthetic calibration found)"


def to8(a):
    return np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8)


def process(name, image_dir, search_root, args, out_root):
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    rig, info = build_rig(image_dir, search_root, args.scale, args.spacing, args.fx_full,
                          args.ref, args.poses, args)
    print(f"[{name}] {len(rig.views)} photos, poses: {info['poses']}")
    if args.exposure_norm:
        gains = camera.normalise_exposure(rig, args.gain_depth)
        info["exposure_gain_range"] = [float(np.min(gains)), float(np.max(gains))]
    cfg = dict(z_near=args.z_near, z_far=args.z_far, wall_front=args.wall_front)
    if args.z_split:
        cfg["z_split"] = args.z_split
    res = pipeline.deocclude(rig, cfg)
    K = rig.views[rig.ref].K
    rr = (args.radius_mm[0] / 1e3, args.radius_mm[1] / 1e3) if args.radius_mm else None
    cands, maps = detect.detect_fruit(res["depth"], res["image"], res["support"], K, fruit=args.fruit,
                                      radius_range=rr, confidence=res["confidence"],
                                      wall_front=args.wall_front)
    kept = detect.nms_3d(cands)
    thr, thr_src = (args.threshold, "--threshold") if args.threshold else default_threshold(args.fruit)
    dets = [d for d in kept if d["score"] >= thr]
    h, w = res["image"].shape[:2]
    margin = int(0.02 * w)
    roi = args.roi if args.roi else [margin, margin, w - margin, h - margin]
    roi = [int(v) for v in roi]
    dets = evaluate.in_roi(dets, roi)
    # outputs
    ref_img = rig.views[rig.ref].image
    cv2.imwrite(os.path.join(out, "centre_photo.png"), to8(ref_img))
    cv2.imwrite(os.path.join(out, "leaf_free.png"), to8(res["image"]))
    d = res["depth"]
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    cv2.imwrite(os.path.join(out, "depth.png"),
                cv2.applyColorMap(to8(np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)), cv2.COLORMAP_TURBO))
    cv2.imwrite(os.path.join(out, "relief.png"),
                cv2.applyColorMap(to8(np.clip(maps["relief"] / 0.03, 0, 1)), cv2.COLORMAP_TURBO))
    strip = np.hstack([to8(cv2.resize(m, (w // 4, h // 4))) for m in res["masks"]])
    cv2.imwrite(os.path.join(out, "foliage_masks.png"), strip)
    vis = to8(res["image"]).copy()
    cv2.rectangle(vis, (roi[0], roi[1]), (roi[2], roi[3]), (255, 200, 0), 1)
    for k, dd in enumerate(dets, 1):
        cv2.circle(vis, (int(dd["x"]), int(dd["y"])), int(dd["r_px"]) + 3, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(vis, str(k), (int(dd["x"] + dd["r_px"] + 3), int(dd["y"])), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(vis, f"{len(dets)} fruit", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(out, "detections.png"), vis)
    write_ply(os.path.join(out, "leaf_free_cloud.ply"), res["depth_render"], res["image"], K, res["support"] >= 1)
    with open(os.path.join(out, "detections.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "x_px", "y_px", "x_px_full", "y_px_full", "X_m", "Y_m", "Z_m", "radius_mm", "score"])
        for k, dd in enumerate(dets, 1):
            wr.writerow([k, round(dd["x"], 2), round(dd["y"], 2), round(dd["x"] / args.scale, 1),
                         round(dd["y"] / args.scale, 1)] + [round(v, 4) for v in dd["centre"]]
                        + [round(dd["R"] * 1e3, 2), round(dd["score"], 3)])
    summary = dict(capture=name, images=os.path.relpath(image_dir, REPO), n_photos=len(rig.views),
                   count=len(dets), threshold=thr, threshold_source=thr_src, roi=roi, scale=args.scale,
                   z_split_m=res["z_split"], z_wall_m=res["z_wall"], bg_range_m=res["bg_range"],
                   unobserved_fraction=float(np.mean(res["support"] < 0.5)), timings=res["timings"],
                   seconds_total=time.time() - t0, **info)
    # foliage-free comparison
    if args.clean:
        crig = rig_with_poses(args.clean, rig, args.scale)
        if args.exposure_norm:
            camera.normalise_exposure(crig, args.gain_depth)
        clean = crig.views[crig.ref].image
        aligned = align_ecc(res["image"], clean)
        summary["psnr_vs_clean"] = evaluate.psnr(aligned, clean, roi)
        summary["ssim_vs_clean"] = evaluate.ssim(aligned, clean, roi)
        summary["psnr_centre_photo_vs_clean"] = evaluate.psnr(align_ecc(ref_img, clean), clean, roi)
        cres = pipeline.deocclude(crig, cfg, verbose=False)
        cc, _ = detect.detect_fruit(cres["depth"], cres["image"], cres["support"], crig.views[crig.ref].K,
                                    fruit=args.fruit, radius_range=rr, confidence=cres["confidence"],
                                    wall_front=args.wall_front)
        cdets = evaluate.in_roi([q for q in detect.nms_3d(cc) if q["score"] >= thr], roi)
        summary["count_clean_capture"] = len(cdets)
        cv2.imwrite(os.path.join(out, "clean_centre_photo.png"), to8(clean))
    if args.labels:
        gts = []
        for row in csv.DictReader(open(args.labels)):
            gts.append(dict(uv_ref=[float(row["x_px"]) * args.scale, float(row["y_px"]) * args.scale],
                            r_px_ref=args.label_radius_px * args.scale))
        m = evaluate.match_counts(dets, gts)
        summary["vs_labels"] = {k: v for k, v in m.items() if k != "matched"}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(f"[{name}] counted {len(dets)} fruit (threshold {thr:.1f}, {thr_src}); outputs in {os.path.relpath(out, REPO)}")
    return summary


def write_ply(path, depth, image, K, valid, step=2):
    """
    Coloured point cloud of the leaf-free surface (ASCII PLY, metres, reference
    camera frame): every `step`-th pixel back-projected at its background depth.
    Opens in MeshLab, CloudCompare or Open3D for a 3-D figure.
    """
    h, w = depth.shape
    v, u = np.mgrid[0:h:step, 0:w:step]
    z = depth[::step, ::step]
    ok = valid[::step, ::step] & np.isfinite(z) & (z > 0)
    Ki = np.linalg.inv(K)
    x = (Ki[0, 0] * u + Ki[0, 2]) * z
    y = (Ki[1, 1] * v + Ki[1, 2]) * z
    col = np.clip(image[::step, ::step] * 255, 0, 255).astype(np.uint8)
    pts = np.stack([x[ok], y[ok], z[ok]], 1)
    rgb = col[ok][:, ::-1]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for (a, b, c), (r_, g_, b_) in zip(pts, rgb):
            f.write(f"{a:.4f} {b:.4f} {c:.4f} {r_} {g_} {b_}\n")


def align_ecc(img, target):
    """Align img to target with an ECC homography on luminance (for image metrics only)."""
    a = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(np.clip(target, 0, 1).astype(np.float32), cv2.COLOR_BGR2GRAY)
    Hm = np.eye(3, dtype=np.float32)
    try:
        _, Hm = cv2.findTransformECC(b, a, Hm, cv2.MOTION_HOMOGRAPHY,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6), None, 5)
    except cv2.error:
        return img
    return cv2.warpPerspective(img, Hm, (img.shape[1], img.shape[0]),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", help="folder of rail photos (default: search data folders)")
    ap.add_argument("--clean", help="folder of the same rail positions without foliage")
    ap.add_argument("--labels", help="CSV with x_px, y_px of every fruit in the full-resolution centre photo")
    ap.add_argument("--label-radius-px", type=float, default=32.0, help="fruit radius in full-res pixels for matching")
    ap.add_argument("--spacing", type=float, default=0.05, help="rail step [m]")
    ap.add_argument("--scale", type=float, default=0.25, help="processing scale of the photos")
    ap.add_argument("--ref", type=int, default=None, help="index of the reference photo (default: centre)")
    ap.add_argument("--fx-full", type=float, default=None, help="focal length in full-resolution pixels")
    ap.add_argument("--poses", choices=["auto", "colmap", "sfm", "nominal"], default="auto")
    ap.add_argument("--sfm-gap", type=int, default=6, help="match photos at most this many rail steps apart")
    ap.add_argument("--sfm-features", type=int, default=8000, help="SIFT features kept per photo in the rail SfM")
    ap.add_argument("--fruit", default="cherry", choices=list(detect.FRUIT_RADII))
    ap.add_argument("--threshold", type=float, default=None, help="detection score threshold [mm]")
    ap.add_argument("--radius-mm", type=float, nargs=2, default=None, metavar=("RMIN", "RMAX"),
                    help="fruit radius range [mm] (default: per fruit type)")
    ap.add_argument("--wall-front", type=float, default=0.07,
                    help="how far fruit can hang in front of the fence [m]")
    ap.add_argument("--exposure-norm", action="store_true",
                    help="match per-photo exposure on the fence plane (only if exposure was not locked)")
    ap.add_argument("--gain-depth", type=float, default=0.9, help="fence depth used for --exposure-norm [m]")
    ap.add_argument("--z-near", type=float, default=0.30)
    ap.add_argument("--z-far", type=float, default=1.50)
    ap.add_argument("--z-split", type=float, default=None, help="foliage/background split depth [m] (default: Otsu)")
    ap.add_argument("--roi", type=float, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"),
                    help="count only inside this box (processing-scale pixels)")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "real"))
    args = ap.parse_args()
    if args.images:
        caps = [(os.path.basename(os.path.normpath(args.images)), args.images, args.images)]
    else:
        caps = find_captures()
        if not caps:
            print("No captures found. Put the rail photos in <repo>/data/<capture>/images/ "
                  "(or pass --images).  data/ is git-ignored, so the photos stay local.")
            return
    for name, image_dir, root in caps:
        process(name, image_dir, root, args, args.out)


if __name__ == "__main__":
    main()
