#!/usr/bin/env python3
"""First synthetic aperture refocusing experiment for the AMME5710 cherry scene.

INPUTS: 12 sequential rail photographs (images.zip or an image directory)
GEOMETRY: spacing 50 mm, leaf depth 572 mm, fruit depth 912 mm

No dense COLMAP point cloud is used.  The script accepts a focal length from
COLMAP cameras.bin / cameras.txt, a focal length derived from the iPhone metadata (24 mm full-frame
equivalent), or can estimate a scene-based fx from green leaf parallax.

Example:
    python cherry_sai_refocus.py --images images.zip --out sai_results_24mm
    python cherry_sai_refocus.py --images images.zip --colmap-cameras path/to/cameras.bin
    python cherry_sai_refocus.py --images images.zip --fx 2700
    python cherry_sai_refocus.py --images images.zip --auto-fx
"""
import argparse
import json
import math
import struct
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MODEL_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12,
                7: 5, 8: 4, 9: 5, 10: 12}
MODEL_NAMES = {
    0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL",
    4: "OPENCV", 5: "OPENCV_FISHEYE", 6: "FULL_OPENCV", 7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE", 9: "RADIAL_FISHEYE", 10: "THIN_PRISM_FISHEYE",
}


def colmap_fx(path, image_width):
    """Load fx in pixels from the first COLMAP camera; warn on inconsistent sizes."""
    path = Path(path)
    if path.is_dir():
        paths = [path / "cameras.bin", path / "cameras.txt"]
        path = next((p for p in paths if p.exists()), None)
        if path is None:
            raise FileNotFoundError("COLMAP folder has no cameras.bin or cameras.txt")
    if path.suffix == ".txt":
        entries = [line.split() for line in path.read_text().splitlines()
                   if line.strip() and not line.startswith("#")]
        if not entries:
            raise ValueError("No cameras in COLMAP cameras.txt")
        e = entries[0]
        model, width, params = e[1], int(e[2]), [float(v) for v in e[4:]]
    else:
        with path.open("rb") as f:
            n, = struct.unpack("<Q", f.read(8))
            if n == 0:
                raise ValueError("No COLMAP cameras")
            cid, mid, width, height = struct.unpack("<iiQQ", f.read(24))
            if mid not in MODEL_PARAMS:
                raise ValueError(f"Unsupported COLMAP camera model id {mid}")
            model = MODEL_NAMES[mid]
            npar = MODEL_PARAMS[mid]
            params = struct.unpack("<" + "d" * npar, f.read(8 * npar))
    print(f"  COLMAP: model={model}, width={width}, fx={params[0]:.1f} px")
    if abs(width - image_width) > 1:
        raise ValueError(
            f"COLMAP calibration width={width}, but source PNG width={image_width}. "
            "Use the matching ORIGINAL camera model, or provide --fx in source-image pixels."
        )
    if model != "PINHOLE" and model != "SIMPLE_PINHOLE":
        print("  NOTE: Input lens may have distortion. Refocusing only translates images; "
              "undistorting source PNGs would improve precise alignment.")
    return float(params[0])


def load_images(path, scale):
    path = Path(path)
    suffixes = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    if path.is_dir():
        names = sorted(x for x in path.rglob("*") if x.suffix.lower() in suffixes)
        blobs = ((x.name, x.read_bytes()) for x in names)
    elif path.suffix.lower() == ".zip":
        z = zipfile.ZipFile(path)
        names = sorted(x for x in z.namelist() if x.lower().endswith(suffixes))
        blobs = ((Path(x).name, z.read(x)) for x in names)
    else:
        raise ValueError("--images must be a directory or a zip archive")
    output = []
    full_size = None
    for name, data in blobs:
        raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if raw is None:
            raise ValueError(f"Cannot decode {name}")
        h, w = raw.shape[:2]
        if full_size is None:
            full_size = (w, h)
        if (w, h) != full_size:
            raise ValueError("All input images must have matching dimensions")
        scaled = cv2.resize(raw, (int(round(w * scale)), int(round(h * scale))),
                            interpolation=cv2.INTER_AREA)
        output.append((name, scaled))
    if len(output) < 3:
        raise ValueError("Need at least three camera views")
    return output, full_size


def leaf_mask(bgr):
    """Loose GREEN mask, used only for approximate focal calibration and optional masking."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Broad green interval for the physical green pipe-cleaner leaves.
    return (cv2.inRange(hsv, np.array([28, 45, 30], dtype=np.uint8),
                            np.array([102, 255, 255], dtype=np.uint8)) > 0)


def match_pair(a, b, sift):
    g1 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    kp1, d1 = sift.detectAndCompute(g1, None)
    kp2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    matches = [m for m, n in pairs if m.distance < .75 * n.distance]
    pts_a = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 2)
    pts_b = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 2)
    return pts_a, pts_b


def estimate_fx_and_vertical_offsets(images_bgr, scale, spacing_mm, leaf_depth_mm):
    """Estimate approximate fx from green feature motion on consecutive rail frames.

    Assumes some green feature matches are at the supplied leaf plane depth.
    Individual leaves may be nearer/farther; this is NOT formal camera calibration.
    """
    sift = cv2.SIFT_create(nfeatures=4800, contrastThreshold=.018)
    masks = [leaf_mask(im) for im in images_bgr]
    focal_candidates = []
    vertical_steps = []
    per_pair = []
    for i, (a, b) in enumerate(zip(images_bgr[:-1], images_bgr[1:])):
        pa, pb = match_pair(a, b, sift)
        if not len(pa):
            vertical_steps.append(0.0)
            per_pair.append({"pair": [i + 1, i + 2], "matches": 0, "green_matches": 0})
            continue
        delta = pb - pa
        finite = (np.abs(delta[:, 1]) < 25) & (np.abs(delta[:, 0]) < 250)
        dy = float(np.median(delta[finite, 1])) if np.sum(finite) >= 15 else 0.0
        vertical_steps.append(dy)

        u1 = np.rint(pa).astype(int)
        u2 = np.rint(pb).astype(int)
        h, w = a.shape[:2]
        good1 = (u1[:, 0] >= 0) & (u1[:, 0] < w) & (u1[:, 1] >= 0) & (u1[:, 1] < h)
        good2 = (u2[:, 0] >= 0) & (u2[:, 0] < w) & (u2[:, 1] >= 0) & (u2[:, 1] < h)
        inds = np.where(good1 & good2)[0]
        leaf = np.zeros(len(pa), dtype=bool)
        leaf[inds] = masks[i][u1[inds, 1], u1[inds, 0]] & masks[i + 1][u2[inds, 1], u2[inds, 0]]
        # The camera was translated RIGHT, so scene features move LEFT.
        # Restrict possible fx to a very loose smartphone-camera range.
        dx = delta[leaf, 0]
        dy_resid = delta[leaf, 1] - dy
        max_dx = 6000.0 * scale * spacing_mm / leaf_depth_mm
        min_dx = 900.0 * scale * spacing_mm / leaf_depth_mm
        plausible = (dx < -min_dx) & (dx > -max_dx) & (np.abs(dy_resid) < 7)
        dx = dx[plausible]
        if len(dx) >= 12:
            # Find the dominant green-feature displacement and reject other depths.
            hist, edges = np.histogram(dx, bins=np.arange(-max_dx - 2, -min_dx + 3, 3.0))
            peak = int(np.argmax(hist))
            center = 0.5 * (edges[peak] + edges[peak + 1])
            inliers = dx[np.abs(dx - center) < 8]
            if len(inliers) >= 9:
                dx_leaves = float(np.median(inliers))
                fx_est = (-dx_leaves) * leaf_depth_mm / (spacing_mm * scale)
                focal_candidates.append(fx_est)
                per_pair.append({"pair": [i + 1, i + 2], "matches": len(pa),
                                 "green_matches": int(len(dx)), "green_dx_px_scaled": dx_leaves,
                                 "fx_from_leaves_px": fx_est, "vertical_step_px_scaled": dy})
                continue
        per_pair.append({"pair": [i + 1, i + 2], "matches": len(pa),
                         "green_matches": int(len(dx)), "vertical_step_px_scaled": dy,
                         "note": "insufficient consistent green correspondences"})
    if len(focal_candidates) < 3:
        raise RuntimeError("Automatic focal estimate is not reliable. "
                           "Provide a COLMAP cameras.bin file with --colmap-cameras, "
                           "or a calibrated focal length in pixels using --fx.")
    fx = float(np.median(focal_candidates))
    spread = float(np.std(focal_candidates))
    y_positions = np.concatenate(([0.0], np.cumsum(vertical_steps)))
    return fx, spread, y_positions, per_pair


def warp_stack(images_bgr, positions_mm, y_positions_scaled, reference_index,
               depth_mm, fx_scaled):
    """Warp each source image to the chosen plane in the reference camera frame."""
    h, w = images_bgr[0].shape[:2]
    stack = []
    valids = []
    leaf_valid = []
    ones = np.ones((h, w), np.uint8)
    for i, im in enumerate(images_bgr):
        # As camera moves right, objects shift left; undo that motion.
        delta_x = fx_scaled * (positions_mm[i] - positions_mm[reference_index]) / depth_mm
        delta_y = -(y_positions_scaled[i] - y_positions_scaled[reference_index])
        M = np.float32([[1, 0, delta_x], [0, 1, delta_y]])
        wi = cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        v = cv2.warpAffine(ones, M, (w, h), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        # Conservative dilation of leaf occlusion mask to reduce green edge artefacts.
        green_dilated = cv2.dilate((leaf_mask(im) * 255).astype(np.uint8),
                                   np.ones((3, 3), np.uint8))
        allowed = cv2.warpAffine(green_dilated, M, (w, h),
                                 flags=cv2.INTER_NEAREST, borderValue=255) == 0
        stack.append(wi)
        valids.append(v)
        leaf_valid.append(v & allowed)
    return np.stack(stack, axis=0), np.stack(valids, axis=0), np.stack(leaf_valid, axis=0)


def average_images(stack, valid, min_contributors):
    cnt = np.sum(valid, axis=0).astype(np.uint8)
    # Accumulate float, mask invalid contributions, never interpret black pixels as data.
    weighted_sum = np.sum(stack.astype(np.float32) * valid[..., None], axis=0)
    avg = weighted_sum / np.maximum(cnt[..., None], 1)
    avg[cnt < min_contributors] = 0
    return np.clip(avg, 0, 255).astype(np.uint8), cnt


def median_images(stack, valid, min_contributors):
    cnt = np.sum(valid, axis=0).astype(np.uint8)
    rgb = np.where(valid[..., None], stack.astype(np.float32), np.nan)
    with np.errstate(all="ignore"):
        # Suppress expected 'All-NaN slice' warning for pixels never seen by any view.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(rgb, axis=0)
    median[cnt < min_contributors] = 0
    return np.clip(np.nan_to_num(median), 0, 255).astype(np.uint8)


def crop_from_reference(im, scale, box=(0.10, 0.27, 0.91, 0.78)):
    """Optional display-only crop as a fraction of image dimensions."""
    h, w = im.shape[:2]
    x0, y0, x1, y1 = box
    return im[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def write_comparison(out, imgs, titles):
    thumbs = []
    for im, title in zip(imgs, titles):
        thumb = cv2.resize(im, (600, 800), interpolation=cv2.INTER_AREA)
        canvas = np.full((858, 600, 3), 245, np.uint8)
        canvas[58:] = thumb
        cv2.putText(canvas, title, (18, 37), cv2.FONT_HERSHEY_SIMPLEX,
                    .70, (30, 30, 30), 2, cv2.LINE_AA)
        thumbs.append(canvas)
    # Two per row, safe preview dimensions 1200 x 1716.
    if len(thumbs) % 2:
        thumbs.append(np.full_like(thumbs[0], 245))
    rows = [np.hstack(thumbs[i:i + 2]) for i in range(0, len(thumbs), 2)]
    cv2.imwrite(str(out / "comparison.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", default="images.zip", help="Image zip or directory")
    parser.add_argument("--out", default="sai_results", help="Output folder")
    parser.add_argument("--spacing", type=float, default=50.0, help="Rail step in mm")
    parser.add_argument("--leaf-depth", type=float, default=572.0, help="Foreground depth in mm")
    parser.add_argument("--fruit-depth", type=float, default=912.0, help="Fruit depth in mm")
    parser.add_argument("--fx", type=float, help="Override focal length in SOURCE image pixels")
    parser.add_argument("--equiv-focal", type=float, default=24.0,
                        help="35 mm full-frame-equivalent focal length in mm (iPhone photo: 24)")
    parser.add_argument("--auto-fx", action="store_true",
                        help="Use focal length estimated from green-leaf parallax instead of 24 mm equivalent")
    parser.add_argument("--colmap-cameras", type=str,
                        help="COLMAP cameras.bin or cameras.txt (must match input image resolution)")
    parser.add_argument("--scale", type=float, default=0.25, help="Preview image scale, 0.25 or 0.5")
    parser.add_argument("--ref", type=int, default=6, help="Reference frame, 1-based")
    parser.add_argument("--first", type=int, default=1, help="First included frame, 1-based")
    parser.add_argument("--last", type=int, default=0, help="Last included frame, 1-based; 0 means all")
    parser.add_argument("--no-median", action="store_true", help="Skip median (reduces RAM)")
    parser.add_argument("--sweep-steps", type=int, default=7, help="Number of focal depths from leaves to fruit")
    args = parser.parse_args()
    if not (0 < args.scale <= 1):
        parser.error("--scale must be >0 and <=1")
    if min(args.spacing, args.leaf_depth, args.fruit_depth, args.equiv_focal) <= 0:
        parser.error("Geometry values must be positive")
    if not args.leaf_depth < args.fruit_depth:
        parser.error("Leaf depth should be less than fruit depth")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("Loading photographs...")
    named_imgs, (original_w, original_h) = load_images(args.images, args.scale)
    names = [x[0] for x in named_imgs]
    frames = [x[1] for x in named_imgs]
    n = len(frames)
    if not 1 <= args.ref <= n:
        parser.error("--ref must be inside the image sequence")
    if not 1 <= args.first <= (args.last or n) <= n:
        parser.error("--first/--last is an invalid range")

    print("Estimating camera vertical drift and checking focal length with green leaf matches...")
    try:
        estimated_fx, fx_spread, y_offsets, pairs = estimate_fx_and_vertical_offsets(
            frames, args.scale, args.spacing, args.leaf_depth)
        print(f"  Green-feature cross-check fx = {estimated_fx:.1f} px "
              f"(pair scatter ~{fx_spread:.1f} px)")
    except RuntimeError as e:
        print(f"  Leaf feature cross-check unavailable: {e}")
        estimated_fx, fx_spread = None, None
        y_offsets = np.zeros(n, dtype=np.float32)
        pairs = []
    if args.fx is not None:
        fx = args.fx
        fx_source = "Explicit fx supplied in source-image pixels"
    elif args.colmap_cameras:
        fx = colmap_fx(args.colmap_cameras, original_w)
        fx_source = "COLMAP intrinsics"
    elif args.auto_fx:
        if estimated_fx is None:
            parser.error("--auto-fx needs successful green feature matching")
        fx = estimated_fx
        fx_source = "Approximate fx from green-feature parallax (NOT camera calibration)"
    else:
        # 35mm film / full-frame reference is 36 mm across its long side.
        # The camera produced 12 MP 4:3 frames, 4032x3024, here stored portrait.
        # Sensor equivalent width is matched to the LONG image dimension.
        fx = args.equiv_focal / 36.0 * max(original_w, original_h)
        fx_source = (f"Approximate fx from {args.equiv_focal:g} mm full-frame-"
                     f"equivalent focal length, 36 mm reference width")
    print(f"Using fx={fx:.1f} full-resolution pixels: {fx_source}")
    print(f"Expected adjacent-view shift at leaves: {fx * args.spacing / args.leaf_depth:.1f} full px")
    print(f"Expected adjacent-view shift at fruit:  {fx * args.spacing / args.fruit_depth:.1f} full px")

    indexes = list(range(args.first - 1, (args.last or n)))
    if args.ref - 1 not in indexes:
        parser.error("Reference frame must be within included range")
    imgs = [frames[i] for i in indexes]
    y_selected = y_offsets[indexes]
    positions = np.array(indexes, np.float32) * args.spacing
    ref_selected = indexes.index(args.ref - 1)
    fx_scaled = fx * args.scale
    cv2.imwrite(str(out / "reference.png"), frames[args.ref - 1])

    outputs = {}
    for key, depth in [("leaves", args.leaf_depth), ("fruit", args.fruit_depth)]:
        print(f"Refocusing at {depth:.0f} mm ({key})...")
        stack, valid, vis_valid = warp_stack(imgs, positions, y_selected,
                                             ref_selected, depth, fx_scaled)
        mean_img, count = average_images(stack, valid, min_contributors=2)
        cv2.imwrite(str(out / f"focus_{key}_mean.png"), mean_img)
        outputs[f"focus_{key}_mean"] = mean_img
        cv2.imwrite(str(out / f"support_{key}.png"), ((count.astype(np.float32) * 255.0 / len(imgs))).astype(np.uint8))
        if key == "fruit":
            mask_mean, mask_count = average_images(stack, vis_valid, min_contributors=2)
            cv2.imwrite(str(out / "focus_fruit_leafmasked.png"), mask_mean)
            cv2.imwrite(str(out / "support_fruit_leafmasked.png"),
                        (mask_count.astype(np.float32) * 255.0 / len(imgs)).astype(np.uint8))
            outputs["focus_fruit_leafmasked"] = mask_mean
        if not args.no_median:
            med = median_images(stack, valid, min_contributors=2)
            cv2.imwrite(str(out / f"focus_{key}_median.png"), med)
            outputs[f"focus_{key}_median"] = med
        del stack, valid, vis_valid

    print("Generating focus depth sweep...")
    gif_frames = []
    for depth in np.linspace(args.leaf_depth, args.fruit_depth, args.sweep_steps):
        stack, valid, _ = warp_stack(imgs, positions, y_selected,
                                      ref_selected, float(depth), fx_scaled)
        frame, cnt = average_images(stack, valid, min_contributors=2)
        # Only visualisation: small label in upper left.
        annotated = frame.copy()
        cv2.rectangle(annotated, (4, 4), (280, 46), (20, 20, 20), -1)
        cv2.putText(annotated, f"Focal depth: {depth:.0f} mm", (11, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        gif_frames.append(Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)))
        del stack, valid
    gif_frames[0].save(out / "focus_sweep.gif", save_all=True,
                       append_images=gif_frames[1:], duration=800, loop=0, optimize=False)
    previews = [frames[args.ref - 1], outputs["focus_leaves_mean"],
                outputs["focus_fruit_mean"], outputs["focus_fruit_leafmasked"]]
    titles = ["Reference (single view)", "Focus: leaves (mean)",
              "Focus: fruit (mean)", "Focus: fruit, leaves masked"]
    write_comparison(out, previews, titles)
    from PIL import ImageDraw
    crops = [crop_from_reference(im, args.scale, (0.10, 0.30, 0.88, 0.71))
             for im in previews]
    labelled = []
    for cr, title in zip(crops, titles):
        canvas = Image.new("RGB", (780, 460), "white")
        pil = Image.fromarray(cv2.cvtColor(cr, cv2.COLOR_BGR2RGB))
        pil.thumbnail((760, 415))
        canvas.paste(pil, ((780 - pil.width)//2, 40))
        ImageDraw.Draw(canvas).text((10, 12), title, fill=(0,0,0))
        labelled.append(canvas)
    contact = Image.new("RGB", (1560, 920), "white")
    for i, cr in enumerate(labelled):
        contact.paste(cr, ((i % 2)*780, (i // 2)*460))
    contact.save(out / "comparison_closeup.jpg", quality=92)
    dx_sep = fx_scaled * args.spacing * (1 / args.leaf_depth - 1 / args.fruit_depth)
    info = {
        "image_count_total": n, "image_indices_used_1based": [i + 1 for i in indexes],
        "image_size_original": [original_w, original_h],
        "image_size_processed": list(reversed(imgs[0].shape[:2])),
        "reference_image_1based": args.ref, "camera_step_mm": args.spacing,
        "leaf_depth_mm": args.leaf_depth, "fruit_depth_mm": args.fruit_depth,
        "fx_source": fx_source, "focal_equivalent_35mm_mm": args.equiv_focal,
        "fx_original_px_used": float(fx),
        "fx_original_px_estimated_from_leaves": estimated_fx,
        "estimated_fx_pair_scatter_px": fx_spread,
        "parallax_leaf_vs_fruit_px_adjacent_at_processing_scale": float(dx_sep),
        "vertical_offsets_px_at_processing_scale": y_offsets.tolist(),
        "pair_feature_diagnostics": pairs,
        "limitations": [
            "Leaf and fruit depths are representative planes, not per-object depths.",
            "Automatic fx is a scene-derived estimate, NOT an intrinsic calibration.",
            "SIFT vertical correction approximates small camera-height variations.",
            "The horizontal translation model assumes no camera rotation and modest distortion.",
            "Masked means cannot fill regions occluded in every view.",
        ],
    }
    (out / "diagnostics.json").write_text(json.dumps(info, indent=2), encoding="utf8")
    print(f"Saved outputs to {out.resolve()}")
    print(f"Differential leaf/fruit parallax per 50 mm step: {dx_sep:.1f} preview pixels")


if __name__ == "__main__":
    main()
