"""
detect.py - colour-free fruit detection on the leaf-free depth map and image.

No colour thresholds are used.  Fruit is found as compact protrusions of the
reconstructed surface, so the same detector works for red cherries, green
grapes and limes.

1. Wall relief by directional morphology
   The fence, wires, stems and slats are long and thin; fruit is compact.  A
   morphological closing of the depth map with a line structuring element of
   length L at angle theta fills every protrusion that is shorter than L along
   theta (protrusions are local minima of depth).  The residual
       r_theta = close_theta(Z) - Z
   is therefore zero on a slat that runs along theta and positive on compact
   bumps.  Taking the minimum over orientations,
       relief = min_theta r_theta,
   keeps only structures that are short in EVERY direction (fruit) and
   suppresses slats, lattice crossings, wires and stems (each is long along at
   least one orientation).  relief is in metres: the height of the fruit above
   whatever lies behind it.

2. Scale-adapted blob detection
   Fruit radius R (metres) is known to within a range, and depth gives the
   pixel radius r = f R / Z, so the scale-normalised Laplacian of Gaussian
       L(p, s) = -s^2 (d2/dx2 + d2/dy2)(G_s * relief)(p),  s = r / sqrt(2),
   is evaluated only at plausible scales.  Its local maxima above a relief
   height are candidates; twins 2R apart give two maxima.

3. Candidate features (all colour-free)
   relief height, blob response, estimated metric radius, disc fill, annulus
   occupancy (a cherry is isolated, a slat crossing extends beyond its disc),
   luminance blob strength and radial symmetry (Loy and Zelinsky 2003) of the
   leaf-free image, view support and photo-consistency.

4. Score and 3-D non-maximum suppression
   score = relief blob response, optionally re-weighted by a classifier
   (cv2.ml.RTrees) trained on the synthetic training scenes.  Candidates are
   back-projected to 3-D (centre = surface point + R along the ray) and
   suppressed within one radius, so each fruit is counted once.

Dependencies: numpy, opencv-python.
"""

import math

import numpy as np
import cv2


FRUIT_RADII = {"cherry": (0.0085, 0.0125), "grape": (0.0065, 0.0095), "lime": (0.022, 0.031)}
# Length of the relief line element in units of R_max: cherries hang in rows of up
# to four (about 8 R), so the line must span a whole row; grape berries are judged
# against their own bunch, so a short element (4 R) keeps each berry's dome.
RELIEF_FACTOR = {"cherry": 10.0, "grape": 4.0, "lime": 4.0}


def line_kernel(length, angle_deg):
    """Binary line structuring element of `length` pixels at `angle_deg` (uint8)."""
    L = int(length) | 1
    k = np.zeros((L, L), np.uint8)
    c = L // 2
    a = math.radians(angle_deg)
    dx, dy = math.cos(a) * c, math.sin(a) * c
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1, 1)
    return k


def fill_invalid_depth(depth, valid):
    """Inpaint depth where `valid` is False (cv2.inpaint accepts float32, one channel)."""
    if np.all(valid):
        return depth
    hole = (~valid).astype(np.uint8)
    return cv2.inpaint(depth.astype(np.float32), hole, 5, cv2.INPAINT_TELEA)


def directional_relief(depth, length_px, n_orient=12, scale=0.5):
    """
    relief = min over orientations of (line closing of depth - depth), clipped at 0.

    Computed at `scale` of the input resolution for speed (the relief is smooth
    at fruit scale) and resized back.
    """
    h, w = depth.shape
    if scale != 1.0:
        d = cv2.resize(depth, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        L = max(5, int(round(length_px * scale)))
    else:
        d, L = depth, int(length_px)
    rel = None
    for k in range(n_orient):
        ker = line_kernel(L, 180.0 * k / n_orient)
        clo = cv2.morphologyEx(d, cv2.MORPH_CLOSE, ker, borderType=cv2.BORDER_REPLICATE)
        r = clo - d
        rel = r if rel is None else np.minimum(rel, r)
    rel = np.maximum(rel, 0)
    if scale != 1.0:
        rel = cv2.resize(rel, (w, h), interpolation=cv2.INTER_LINEAR)
    return rel


def log_response(img, sigma):
    """Scale-normalised negative Laplacian of Gaussian: -sigma^2 * LoG(img)."""
    g = cv2.GaussianBlur(img, (0, 0), sigma)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    return -(sigma ** 2) * lap


def radial_symmetry(gray, radius, alpha=2.0, beta=0.05):
    """
    Fast radial symmetry transform (Loy and Zelinsky 2003) at one radius.

    Each pixel with gradient magnitude above beta * max votes at the pixel
    `radius` away along its gradient (positively affected pixel) and against it
    (negatively affected).  Bright and dark radially symmetric blobs both score
    (absolute orientation projection), so the cue is independent of the fruit
    being lighter or darker than its surroundings.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    h, w = gray.shape
    sel = mag > beta * mag.max()
    ys, xs = np.nonzero(sel)
    ux, uy = gx[sel] / mag[sel], gy[sel] / mag[sel]
    O = np.zeros((h, w), np.float32)
    M = np.zeros((h, w), np.float32)
    for sgn in (1, -1):
        px = np.clip(np.rint(xs + sgn * ux * radius).astype(int), 0, w - 1)
        py = np.clip(np.rint(ys + sgn * uy * radius).astype(int), 0, h - 1)
        np.add.at(O, (py, px), sgn)
        np.add.at(M, (py, px), sgn * mag[sel])
    kn = 9.9 if radius > 1 else 8.0
    Ot = np.minimum(np.abs(O), kn)
    F = (M / kn) * (Ot / kn) ** alpha
    return cv2.GaussianBlur(np.abs(F), (0, 0), 0.25 * radius + 0.5)


def disc_stats(img, x, y, r_in, r_out=None):
    """Values of img inside the disc of radius r_in (or the annulus r_in..r_out) at (x, y)."""
    h, w = img.shape
    R = int(math.ceil(r_out if r_out else r_in)) + 1
    x0, x1 = max(0, int(x) - R), min(w, int(x) + R + 1)
    y0, y1 = max(0, int(y) - R), min(h, int(y) + R + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - x, yy - y)
    m = (d <= r_in) if r_out is None else ((d >= r_in) & (d <= r_out))
    return img[y0:y1, x0:x1][m]


def detect_fruit(depth, image, support, K, fruit="cherry", radius_range=None, confidence=None,
                 cost=None, min_support=3.0, min_conf=0.15, h_min_frac=0.4, k_noise=4.0,
                 relief_len_factor=None, n_scales=4, relief_scale=1.0, use_appearance=True,
                 wall_front=0.07):
    """
    Colour-free fruit detection on the leaf-free reconstruction of the reference view.

    Inputs:
      depth         per-pixel background depth [m] (reference view)
      image         leaf-free image (float32 BGR); only its luminance is used
      support       views used per pixel; pixels seen by fewer than min_support
                    views are not trusted
      K             reference intrinsics
      fruit         key of FRUIT_RADII, or give radius_range=(R_min, R_max) [m]
      confidence    depth confidence from the sweep; pixels below min_conf are
                    not trusted
      h_min_frac    minimum relief as a fraction of R_min ...
      k_noise       ... and at least k_noise times the robust noise level of
                    the relief map (1.4826 x median absolute deviation)
    Untrusted depths are inpainted from their neighbours, then median filtered
    (5x5), before the relief is computed.  Candidates whose relief exceeds
    wall_front + 10 mm (the background sweep never searches nearer than
    wall_front in front of the wall, so larger reliefs are artefacts) are rejected.
    Returns (candidates, maps): candidates is a list of dicts with pixel
    position, metric radius, 3-D centre, features and score; maps holds the
    relief and response images for display.
    """
    R_min, R_max = radius_range if radius_range else FRUIT_RADII[fruit]
    if relief_len_factor is None:
        relief_len_factor = RELIEF_FACTOR.get(fruit, 10.0)
    f = K[0, 0]
    valid = support >= min_support
    if confidence is not None:
        valid &= confidence >= min_conf
    Z = fill_invalid_depth(depth, valid)
    Z = cv2.medianBlur(Z.astype(np.float32), 5)
    z_med = float(np.median(Z[valid])) if np.any(valid) else float(np.median(Z))
    L_px = f * relief_len_factor * R_max / z_med
    relief = directional_relief(Z, L_px, scale=relief_scale)
    rv = relief[valid] if np.any(valid) else relief.ravel()
    noise = 1.4826 * float(np.median(np.abs(rv - np.median(rv))))
    h_min = max(h_min_frac * R_min, k_noise * noise)
    h_max = wall_front + 0.01

    # scale-normalised LoG of the relief over the plausible pixel radii
    radii = np.geomspace(f * R_min / z_med, f * R_max / z_med, n_scales)
    resp = np.full(Z.shape, -np.inf, np.float32)
    best_r = np.zeros(Z.shape, np.float32)
    for r in radii:
        L = log_response(relief, r / math.sqrt(2.0))
        better = L > resp
        resp[better] = L[better]
        best_r[better] = r
    # local maxima, at least one minimum radius apart
    r0 = max(2, int(round(radii[0] * 0.9)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r0 + 1, 2 * r0 + 1))
    gate = (relief > h_min) & (relief < h_max) & valid
    peaks = (resp >= cv2.dilate(resp, ker)) & gate & (resp > 0)

    # appearance cue (luminance only): radial symmetry at the plausible radii.
    # Fruit touching in a row forms one relief "bar" whose interior has no relief
    # maximum; each sphere's rim still votes for its own centre, so radial-symmetry
    # maxima inside the relief-gated region split the bar into individual fruit.
    gray = cv2.cvtColor(np.clip(image, 0, 1).astype(np.float32), cv2.COLOR_BGR2GRAY)
    frst = np.zeros(Z.shape, np.float32)
    for r in radii:
        fr = radial_symmetry(gray, r)
        frst = np.maximum(frst, fr / max(float(fr.max()), 1e-9))
    r_mid = float(np.median(radii))
    app = np.abs(log_response(gray, r_mid / math.sqrt(2.0)))
    if use_appearance:
        fill_ok = cv2.erode(gate.astype(np.uint8), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * r0 + 1, 2 * r0 + 1))) > 0
        peaks |= (frst >= cv2.dilate(frst, ker)) & fill_ok & (frst > 0.02) & (resp > 0)
    ys, xs = np.nonzero(peaks)

    validf = valid.astype(np.float32)
    Kinv = np.linalg.inv(K)
    cands = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        r = float(best_r[y, x])
        h = float(relief[y, x])
        z_s = float(Z[y, x])
        R_hat = r * z_s / f
        disc = disc_stats(relief, x, y, r)
        ann = disc_stats(relief, x, y, 1.5 * r, 2.2 * r)
        ray = Kinv @ np.array([x, y, 1.0])
        c3 = ray * (z_s + R_hat)
        feat = dict(relief_mm=h * 1e3, resp_mm=float(resp[y, x]) * 1e3, r_px=r, R_mm=R_hat * 1e3,
                    fill=float(np.mean(disc > 0.4 * h)) if disc.size else 0.0,
                    annulus=float(np.mean(ann > 0.5 * h)) if ann.size else 0.0,
                    frst=float(frst[y, x]), app=float(app[y, x]),
                    support=float(np.median(disc_stats(support, x, y, r))),
                    valid_frac=float(np.mean(disc_stats(validf, x, y, r))),
                    conf=float(np.median(disc_stats(confidence, x, y, r))) if confidence is not None else 1.0,
                    z=z_s, noise_mm=noise * 1e3)
        cands.append(dict(x=float(x), y=float(y), r_px=r, R=R_hat, centre=c3.tolist(),
                          feat=feat, score=feat["resp_mm"]))
    return cands, dict(relief=relief, resp=resp, frst=frst, app=app, depth_filled=Z, valid=valid,
                       h_min=h_min, noise=noise)


def nms_3d(cands, radius_factor=1.0, min_dist=None):
    """
    Greedy 3-D non-maximum suppression: keep the best-scoring candidate, drop
    every other candidate whose centre lies within radius_factor * R of it.
    """
    order = sorted(cands, key=lambda c: -c["score"])
    kept = []
    for c in order:
        p = np.array(c["centre"])
        ok = True
        for k in kept:
            dmin = min_dist if min_dist else radius_factor * max(c["R"], k["R"])
            if np.linalg.norm(p - np.array(k["centre"])) < dmin:
                ok = False
                break
        if ok:
            kept.append(c)
    return kept


def detect_2d(image, K, depth_plane, fruit="cherry", radius_range=None, thresh=None):
    """
    Colour-free 2-D detector for images without a depth map (single photo,
    single-plane synthetic aperture): luminance blob strength x radial symmetry
    at the pixel radius the fruit would have on the focal plane.
    """
    R_min, R_max = radius_range if radius_range else FRUIT_RADII[fruit]
    f = K[0, 0]
    gray = cv2.cvtColor(np.clip(image, 0, 1).astype(np.float32), cv2.COLOR_BGR2GRAY)
    radii = np.geomspace(f * R_min / depth_plane, f * R_max / depth_plane, 3)
    score = np.zeros(gray.shape, np.float32)
    best_r = np.zeros(gray.shape, np.float32)
    for r in radii:
        s = np.abs(log_response(gray, r / math.sqrt(2.0))) * np.sqrt(radial_symmetry(gray, r))
        better = s > score
        score[better] = s[better]
        best_r[better] = r
    r0 = max(2, int(round(radii[0] * 0.9)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r0 + 1, 2 * r0 + 1))
    peaks = (score >= cv2.dilate(score, ker)) & (score > 0)
    ys, xs = np.nonzero(peaks)
    Kinv = np.linalg.inv(K)
    out = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        r = float(best_r[y, x])
        R_hat = r * depth_plane / f
        c3 = (Kinv @ np.array([x, y, 1.0])) * (depth_plane + R_hat)
        out.append(dict(x=float(x), y=float(y), r_px=r, R=R_hat, centre=c3.tolist(),
                        score=float(score[y, x]), feat={}))
    if thresh is not None:
        out = [c for c in out if c["score"] >= thresh]
    return out
