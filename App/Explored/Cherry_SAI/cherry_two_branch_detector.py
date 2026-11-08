#!/usr/bin/env python3
r"""AMME5710 cherry detection: two complementary classical computer vision branches.

Branch A: red/pink candidate segmentation using Lab/RGB, morphology and contours.
Branch B: purple/dark candidate proposals using multi-scale blob contrast, colour
          hints, circular shape tests and local annular (surround) contrast.
Overlap suppression merges the branches without counting the same proposal twice.

Requires: numpy, opencv-python. No trained weights, SciPy or internet required.

Examples from your VS Code PowerShell project directory:
  python .\cherry_two_branch_detector.py --sai .\sai_results_24mm --out .\fruit_dual_results
  python .\cherry_two_branch_detector.py --sai .\sai_results_24mm.zip --out .\fruit_dual_results
  python .\cherry_two_branch_detector.py --image .\sai_results_24mm\focus_fruit_leafmasked.png --out .\fruit_dual_results

All results are UNVERIFIED detection candidates, not accurate physical fruit counts.
The ROI is a configurable prototype crop, not an automated box or fruit detector.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np


def read_image(source: Path) -> np.ndarray:
    im = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"Could not open image: {source}")
    return im


def load_variants(args):
    if args.image:
        return {"focus_fruit_leafmasked": read_image(Path(args.image))}
    folder = Path(args.sai)
    filenames = ["reference", "focus_fruit_mean", "focus_fruit_median", "focus_fruit_leafmasked"]
    images = {}
    if folder.is_dir():
        for name in filenames:
            choices = list(folder.rglob(name + ".png"))
            if choices:
                images[name] = read_image(choices[0])
    elif folder.is_file() and folder.suffix.lower() == ".zip":
        with zipfile.ZipFile(folder) as z:
            for name in filenames:
                choices = [n for n in z.namelist() if Path(n).name == name + ".png"]
                if choices:
                    bgr = cv2.imdecode(np.frombuffer(z.read(choices[0]), np.uint8), cv2.IMREAD_COLOR)
                    if bgr is None: raise ValueError(f"Could not decode {name}.png")
                    images[name] = bgr
    else:
        raise FileNotFoundError(f"SAI folder/zip not found: {folder}")
    if "focus_fruit_leafmasked" not in images:
        raise FileNotFoundError("Missing focus_fruit_leafmasked.png in SAI results")
    dimensions = {tuple(v.shape[:2]) for v in images.values()}
    if len(dimensions) != 1:
        raise ValueError("All input images must have the same processed resolution")
    return images


def make_roi(shape, frac):
    h, w = shape[:2]
    x0, y0, x1, y1 = [float(x) for x in frac]
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("--roi expects fractions X0 Y0 X1 Y1, each between 0 and 1")
    rect = tuple(map(int, [round(x0*w), round(y0*h), round(x1*w), round(y1*h)]))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[rect[1]:rect[3], rect[0]:rect[2]] = 255
    return mask, rect


def brightness_branch(im, roi, args):
    """Colour branch: preserve semantics of the previous Lab colour baseline."""
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    r = im[:, :, 2].astype(np.float32)
    g = im[:, :, 1].astype(np.float32)
    b = im[:, :, 0].astype(np.float32)
    pixels = ((lab[:, :, 1] >= args.lab_a_min) &
              (r > 1.18 * g + 1) & (r > 1.12 * b + 1) &
              (r >= args.bright_min_red) & (roi > 0))
    binary = pixels.astype(np.uint8)*255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if not (args.bright_min_area <= area <= args.bright_max_area): continue
        if min(w, h) < 3 or max(w, h) / max(1, min(w, h)) > args.bright_aspect_max: continue
        p = cv2.arcLength(c, True)
        circ = float(4 * np.pi * area / max(1, p*p))
        if circ < args.bright_min_circularity: continue
        M = cv2.moments(c)
        if M["m00"] <= 0: continue
        xc, yc = M["m10"]/M["m00"], M["m01"]/M["m00"]
        objects.append({"x": round(xc, 2), "y": round(yc, 2),
                        "radius": round(float(np.sqrt(area / np.pi)), 2),
                        "area": round(float(area), 2), "circularity": round(circ, 3),
                        "branch": "bright_colour", "review": "candidate",
                        "contrast": "", "redness": "", "shape_support": ""})
    return sorted(objects, key=lambda o: (o["y"], o["x"])), binary


def blob_proposals(gray, roi, args):
    """Find dark, circular-ish local minima at several scales (DoG + Hough)."""
    f = gray.astype(np.float32)
    xy = []
    # Positive Gaussian background-minus-foreground response for dark blobs.
    for sigma in args.dark_sigmas:
        small = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma)
        large = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma*2.25)
        response = large - small
        maxima = cv2.dilate(response, np.ones((11, 11), np.uint8))
        yy, xx = np.where((response >= args.dark_blob_response) &
                          (response >= maxima - 0.01) & (roi > 0))
        for x, y in zip(xx.tolist(), yy.tolist()):
            xy.append((x, y, "multiscale_blob", float(response[y, x])))
    # Hough proposals help on partial rounded boundaries and provide an
    # independent shape cue; proposals still have to pass our tests below.
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.25)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,
                               dp=1, minDist=10, param1=55, param2=args.hough_votes,
                               minRadius=args.dark_min_radius, maxRadius=args.dark_max_radius)
    if circles is not None:
        for x, y, r in circles[0]:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < roi.shape[0] and 0 <= xi < roi.shape[1] and roi[yi, xi]:
                xy.append((xi, yi, "hough", float(r)))
    return xy


def local_dark_features(im, gray, lab, x, y, args):
    """Score a possible dark cherry using circular local appearance.

    We avoid declaring a dark pixel alone a cherry. Round dark cherries should
    generally be darker inside a disk than the immediately surrounding ring,
    and should exhibit some red/purple chromatic signal. Both tests can fail
    in very heavy shadow, so borderline detections are REVIEW candidates.
    """
    H, W = gray.shape
    best = None
    for radius in range(args.dark_min_radius, args.dark_max_radius + 1, 2):
        outer = radius * 1.65
        pad = int(np.ceil(outer + 2))
        if x-pad < 0 or y-pad < 0 or x+pad >= W or y+pad >= H: continue
        ys, xs = np.ogrid[-pad:pad+1, -pad:pad+1]
        rho = np.sqrt(xs*xs + ys*ys)
        core = rho <= max(2.5, radius*.65)
        inner = rho <= radius*.88
        surround = (rho >= radius*1.2) & (rho <= outer)
        patchgray = gray[y-pad:y+pad+1, x-pad:x+pad+1].astype(np.float32)
        patch = im[y-pad:y+pad+1, x-pad:x+pad+1]
        patchlab = lab[y-pad:y+pad+1, x-pad:x+pad+1]
        core_gray = float(np.median(patchgray[core]))
        ring_gray = float(np.median(patchgray[surround]))
        contrast = ring_gray - core_gray
        core_r = float(np.median(patch[:, :, 2][core]))
        core_g = float(np.median(patch[:, :, 1][core]))
        core_b = float(np.median(patch[:, :, 0][core]))
        lab_a = float(np.median(patchlab[:, :, 1][core]))
        # A filled dark disk occupies most of its inner area. A narrow black
        # pipe cleaner occupies a smaller fraction, even if locally darker.
        dark_fraction = float(np.mean(patchgray[inner] < ring_gray - 7))
        # Contrast measured around the circle at eight directions.
        ang = np.arange(16, dtype=np.float32)*np.pi/8
        xp = np.clip(np.rint(pad + np.cos(ang)*radius*1.30).astype(int), 0, 2*pad)
        yp = np.clip(np.rint(pad + np.sin(ang)*radius*1.30).astype(int), 0, 2*pad)
        radial_support = float(np.mean(patchgray[yp, xp] > core_gray + 6))
        # A permissive red/purple hint still rejects most neutral stems. It is
        # not a guarantee of object identity and cannot detect every fruit.
        red_green = core_r - core_g
        red_blue = core_r - core_b
        redness = max(0., min(1., (lab_a-127)/13))*.6 + max(0., min(1., red_green/15))*.4
        circle_score = 0.43*min(1., contrast/24) + 0.26*dark_fraction + 0.20*radial_support + 0.11*redness
        candidate = {
            "x": float(x), "y": float(y), "radius": float(radius),
            "area": float(np.pi * radius * radius),
            "circularity": round(radial_support, 3),
            "branch": "dark_blob", "review": "candidate",
            "contrast": round(contrast, 2), "redness": round(lab_a, 2),
            "shape_support": round(dark_fraction, 3),
            "radial_support": round(radial_support, 3),
            "red_green": round(red_green, 2), "red_blue": round(red_blue, 2),
            "core_gray": round(core_gray, 2),
            "heuristic_score": round(circle_score, 3),
        }
        if best is None or candidate["heuristic_score"] > best["heuristic_score"]:
            best = candidate
    return best


def dark_branch(im, roi, bright, args):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    proposals = blob_proposals(gray, roi, args)
    candidates = []
    for x, y, source, signal in proposals:
        if not (args.dark_min_radius <= x < im.shape[1] - args.dark_min_radius and
                args.dark_min_radius <= y < im.shape[0] - args.dark_min_radius): continue
        ob = local_dark_features(im, gray, lab, x, y, args)
        if ob is None: continue
        # Dark burgundy/purple fruit has a weaker blue-red difference than
        # brown cardboard shadows; reject extreme brown ratios. This is
        # deliberately tunable, and the heuristic can still fail.
        core5 = im[max(0, y-2):y+3, max(0, x-2):x+3]
        B, G, R = [float(np.median(core5[:, :, k])) for k in range(3)]
        red_green_core = R-G
        red_blue_core = R-B
        if red_blue_core < args.dark_core_min_rb: continue
        if red_green_core < args.dark_core_min_rg: continue
        ratio = red_blue_core / max(1.0, red_green_core)
        if ratio > args.dark_max_brown_ratio: continue

        if ob["core_gray"] >= args.dark_max_gray: continue
        if ob["contrast"] < args.dark_min_contrast: continue
        if ob["shape_support"] < args.dark_min_fill: continue
        if ob["radial_support"] < args.dark_radial_support: continue
        if ob["red_green"] < args.dark_min_rg or ob["redness"] < args.dark_lab_a: continue
        if ob["red_blue"] < args.dark_min_rb: continue
        if ob["heuristic_score"] < args.dark_min_score: continue
        # Conservative separation from already detected red/pink objects.
        if any(np.hypot(ob["x"] - q["x"], ob["y"] - q["y"]) <
               max(7, min(17, (ob["radius"] + q["radius"])*.8)) for q in bright):
            continue
        ob["proposal"] = source
        candidates.append(ob)
    # Greedy non-maximum suppression; prefer strongest proposals.
    candidates.sort(key=lambda o: -o["heuristic_score"])
    selected = []
    for ob in candidates:
        if any(np.hypot(ob["x"]-q["x"], ob["y"]-q["y"]) <
               max(8, min(18, (ob["radius"] + q["radius"])*.9)) for q in selected):
            continue
        selected.append(ob)
    return sorted(selected, key=lambda o: (o["y"], o["x"])), proposals


def draw(im, rect, bright, dark):
    result = im.copy()
    cv2.rectangle(result, (rect[0], rect[1]), (rect[2], rect[3]), (50, 200, 255), 1)
    for prefix, items, color in [("B", bright, (15, 220, 255)), ("D", dark, (255, 220, 0))]:
        for i, o in enumerate(items, 1):
            x, y = int(round(o["x"])), int(round(o["y"]))
            r = max(5, int(round(o["radius"] + 3)))
            cv2.circle(result, (x, y), r, color, 2, cv2.LINE_AA)
            cv2.putText(result, f"{prefix}{i}", (x+r+2, y-2),
                        cv2.FONT_HERSHEY_SIMPLEX, .34, color, 1, cv2.LINE_AA)
    cv2.rectangle(result, (0, 0), (min(result.shape[1], 756), 45), (17, 17, 17), -1)
    cv2.putText(result, f"Bright B:{len(bright)}  Dark D:{len(dark)}  Combined candidates:{len(bright)+len(dark)}",
                (8, 29), cv2.FONT_HERSHEY_SIMPLEX, .51, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def save_csv(path, objects):
    fields = ["id", "branch", "x_px", "y_px", "radius_px", "area_px2", "circularity",
              "local_contrast", "lab_a", "dark_fill_fraction", "radial_support",
              "heuristic_score", "review_status"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, o in enumerate(objects, 1):
            writer.writerow({"id": i, "branch": o["branch"], "x_px": o["x"], "y_px": o["y"],
                             "radius_px": o["radius"], "area_px2": o["area"],
                             "circularity": o["circularity"], "local_contrast": o.get("contrast", ""),
                             "lab_a": o.get("redness", ""), "dark_fill_fraction": o.get("shape_support", ""),
                             "radial_support": o.get("radial_support", ""),
                             "heuristic_score": o.get("heuristic_score", ""),
                             "review_status": "unverified"})


def comparison_strip(variants, annotated, rect, target):
    names = [x for x in ["reference", "focus_fruit_mean", "focus_fruit_median", "focus_fruit_leafmasked"] if x in annotated]
    panels = []
    for name in names:
        im = annotated[name]
        x0,y0,x1,y1=rect
        region=im[max(0,y0-20):min(im.shape[0],y1+20),max(0,x0-12):min(im.shape[1],x1+12)]
        panel=cv2.resize(region,(740,480),interpolation=cv2.INTER_AREA)
        frame=np.full((520,740,3),245,dtype=np.uint8)
        frame[40:]=panel
        cv2.putText(frame, name.replace('_',' '), (12,28),
                    cv2.FONT_HERSHEY_SIMPLEX,.69,(30,30,30),2,cv2.LINE_AA)
        panels.append(frame)
    if len(panels)==4:
        cv2.imwrite(str(target),np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])]))
    elif panels:
        cv2.imwrite(str(target), np.vstack(panels))


def run(args):
    variants=load_variants(args)
    output=Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    roi, rect=make_roi(next(iter(variants.values())).shape,args.roi)
    results={}; annotated={}
    for name,im in variants.items():
        bright, mask=brightness_branch(im,roi,args)
        dark,proposals=dark_branch(im,roi,bright,args)
        all_objects=sorted(bright+dark,key=lambda o:(o["y"],o["x"]))
        save_csv(output/(name+'_bright.csv'),bright)
        save_csv(output/(name+'_dark.csv'),dark)
        save_csv(output/(name+'_all_candidates.csv'),all_objects)
        cv2.imwrite(str(output/(name+'_colour_mask.png')),mask)
        pic=draw(im,rect,bright,dark)
        cv2.imwrite(str(output/(name+'_two_branch_annotated.png')),pic)
        cv2.imwrite(str(output/(name+'_bright_only.png')),draw(im,rect,bright,[]))
        cv2.imwrite(str(output/(name+'_dark_review_only.png')),draw(im,rect,[],dark))
        annotated[name]=pic
        results[name]={"bright_colour_candidates":len(bright),
                       "dark_blob_candidates_unverified":len(dark),
                       "combined_candidates_unverified":len(all_objects),
                       "raw_dark_proposals_before_filtering":len(proposals)}
        print(f"{name:25s} bright={len(bright):2} dark={len(dark):2} combined={len(all_objects):2}")
    comparison_strip(variants,annotated,rect,output/'two_branch_comparison.jpg')
    record={"description":"Two-branch heuristic detector (NOT a verified physical fruit count)",
            "roi_fraction":list(args.roi),"roi_pixels":list(rect),
            "parameters":vars(args).copy(),"results":results,
            "interpretation":[
                "Branch A identifies red/pink connected components in Lab/RGB colour space.",
                "Branch B proposes dark blobs through multi-scale contrast and circular shape, then checks weak red/purple appearance.",
                "Dark blobs may be shadows or stems. All predictions require human ground-truth review.",
                "This does not perform cross-view identity matching, full 3D triangulation or deep learning.",
                "The manually selected scene ROI is a prototype convenience, not a general-purpose tree detector.",
                "The single 912 mm SAI plane cannot align fruit at other depths perfectly."]}
    (output/'two_branch_summary.json').write_text(json.dumps(record,indent=2,default=str),encoding='utf8')
    (output/'README.txt').write_text(
        'AMME5710 - Two-branch cherry detection\n\n'
        'Open two_branch_comparison.jpg and focus_fruit_leafmasked_two_branch_annotated.png.\n'
        'Yellow numbered B labels = bright colour candidates. Cyan numbered D labels = dark blob candidates.\n'
        'Every candidate remains unverified. Dark blobs can include dark stems and shadows.\n'
        'CSV files provide positions and scores for manual review.\n'
        'Tune --dark-min-contrast, --dark-min-score, --dark-min-fill and --lab-a-min.\n'
        'ROI: configurable fractions, not an automatic scene detector.\n',encoding='utf8')
    print("\nSaved outputs to:",output.resolve())
    print("WARNING: These are detections, not a validated count of physical cherries.")


def main():
    parser=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sai',default='sai_results_24mm',help='SAI results folder or ZIP')
    parser.add_argument('--image',default=None,help='Optional specific image; overrides --sai')
    parser.add_argument('--out',default='fruit_dual_results')
    parser.add_argument('--roi',nargs=4,type=float,default=(.20,.35,.82,.61),metavar=('X0','Y0','X1','Y1'))
    parser.add_argument('--lab-a-min',type=int,default=140)
    parser.add_argument('--bright-min-red',type=float,default=26)
    parser.add_argument('--bright-min-area',type=float,default=13)
    parser.add_argument('--bright-max-area',type=float,default=340)
    parser.add_argument('--bright-aspect-max',type=float,default=2.4)
    parser.add_argument('--bright-min-circularity',type=float,default=.24)
    parser.add_argument('--dark-min-radius',type=int,default=5)
    parser.add_argument('--dark-max-radius',type=int,default=16)
    parser.add_argument('--dark-sigmas',type=float,nargs='+',default=(4.,6.,8.))
    parser.add_argument('--dark-blob-response',type=float,default=9)
    parser.add_argument('--hough-votes',type=int,default=14)
    parser.add_argument('--dark-max-gray',type=float,default=75)
    parser.add_argument('--dark-min-contrast',type=float,default=10)
    parser.add_argument('--dark-min-fill',type=float,default=.70)
    parser.add_argument('--dark-radial-support',type=float,default=.65)
    parser.add_argument('--dark-lab-a',type=float,default=132)
    parser.add_argument('--dark-min-rg',type=float,default=5)
    parser.add_argument('--dark-min-rb',type=float,default=-2)
    parser.add_argument('--dark-min-score',type=float,default=.70)
    parser.add_argument('--dark-core-min-rb',type=float,default=-2)
    parser.add_argument('--dark-core-min-rg',type=float,default=4)
    parser.add_argument('--dark-max-brown-ratio',type=float,default=1.65)

    args=parser.parse_args()
    if args.dark_min_radius<2 or args.dark_max_radius<=args.dark_min_radius:
        parser.error('Invalid dark radius range')
    run(args)


if __name__=='__main__':
    main()
