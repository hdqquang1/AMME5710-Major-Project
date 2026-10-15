"""
planesweep.py - multi-baseline plane-sweep stereo and colour-free occluder masks.

Idea
----
Every photo of the rail is also a stereo partner of every other photo.  For a
candidate depth Z, warping the neighbours onto view i through the plane Z
(synthetic-aperture refocusing) lines up the points that really lie at Z, so
their photometric difference to view i is small.  Scanning Z (a plane sweep,
Collins 1996) and keeping the most photo-consistent depth per pixel gives a
depth map; with several baselines this is multiple-baseline stereo (Okutomi and
Kanade 1993).

Occlusion handling (Kang, Szeliski and Chai 2001): a background pixel of view
i is hidden by foliage in some neighbours, so the per-neighbour matching costs
are sorted and only the best fraction `keep` is averaged ("temporal
selection").  Foliage pixels are seen by every neighbour and need no selection.

The foreground layer (foliage) is then everything nearer than a split depth
found automatically by Otsu's method on the inverse-depth histogram of the
reference view.  Nothing here uses colour classes or colour thresholds: the
masks come from geometry (depth) alone, so the same code removes green leaves
in front of green limes.

Dependencies: numpy, opencv-python.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

from .camera import plane_homography, relative_pose, warp_by_homography, scale_K


def box_mean(img, win):
    """Normalised box filter (window win x win), the fast cost aggregation."""
    return cv2.boxFilter(img, -1, (win, win), normalize=True, borderType=cv2.BORDER_REFLECT)


def guided_filter(guide, src, radius, eps):
    """
    Grey-guide guided filter (He, Sun and Tang 2010) from box filters.

    Edge-aware cost aggregation (Hosni et al. 2013): costs are smoothed within
    regions of similar guide intensity, which limits the "foreground fattening"
    of window-based stereo at depth edges.
    """
    w = 2 * radius + 1
    mI = box_mean(guide, w)
    mp = box_mean(src, w)
    cov = box_mean(guide * src, w) - mI * mp
    var = box_mean(guide * guide, w) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return box_mean(a, w) * guide + box_mean(b, w)


def choose_neighbours(n_views, i, k=6):
    """
    Neighbour set for view i: a mix of short and long baselines on both sides
    (offsets 1, 2, 4, 6, ... clipped to the rail), at most k views.
    """
    offs = [1, -1, 2, -2, 4, -4, 6, -6, 8, -8, 3, -3, 5, -5, 7, -7, 9, -9, 10, -10, 11, -11, 12, -12]
    out = []
    for o in offs:
        j = i + o
        if 0 <= j < n_views and j not in out:
            out.append(j)
        if len(out) >= k:
            break
    return out


def robust_sweep(views, i, depths, neighbours, win=5, keep=0.5, trunc=0.12,
                 aggregation="box", gf_eps=1e-3):
    """
    Plane-sweep depth map of view i with occlusion-robust cost selection.

    Inputs:
      views       list of View (all at the same processing scale)
      i           index of the view whose depth map is computed
      depths      depth planes (metres, near to far, uniform in inverse depth)
      neighbours  indices of the partner views
      win         aggregation window (pixels)
      keep        fraction of the per-neighbour costs averaged (best first)
      trunc       truncation of the per-pixel absolute difference (robust to
                  specular highlights and residual misalignment)
    Cost per neighbour j and plane k: truncated mean absolute colour difference
    between view i and neighbour j warped through plane k, aggregated over the
    window.  Plane cost = mean of the `keep` best neighbour costs.
    Outputs dict:
      depth   sub-plane depth (parabola fit of the cost around the minimum)
      cost    minimum aggregated cost
      kbest   index of the best plane
    """
    vi = views[i]
    I = vi.image
    h, w = I.shape[:2]
    rel = [relative_pose(vi, views[j]) for j in neighbours]
    n_keep = max(1, int(round(keep * len(neighbours))))
    guide = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY) if aggregation == "guided" else None

    best_c = np.full((h, w), np.inf, np.float32)
    best_k = np.zeros((h, w), np.int32)
    c_before = np.full((h, w), np.inf, np.float32)
    c_after = np.full((h, w), np.inf, np.float32)
    prev = None
    stack = np.empty((len(neighbours), h, w), np.float32)
    for k, d in enumerate(depths):
        for n, j in enumerate(neighbours):
            H = plane_homography(vi.K, views[j].K, rel[n][0], rel[n][1], d)
            wj = warp_by_homography(views[j].image, H, (w, h), border_value=-1.0)
            e = cv2.absdiff(wj, I).sum(axis=2) * (1.0 / 3.0)
            np.minimum(e, trunc, out=e)
            e[wj[..., 0] < -0.5] = trunc          # outside neighbour j: maximal cost
            if aggregation == "guided":
                stack[n] = guided_filter(guide, e, win // 2, gf_eps)
            else:
                stack[n] = box_mean(e, win)
        if n_keep < len(neighbours):
            C = np.sort(stack, axis=0)[:n_keep].mean(axis=0)
        else:
            C = stack.mean(axis=0)
        # running arg-min with the two costs around it (for the parabola fit)
        if prev is not None:
            just_after = best_k == (k - 1)
            c_after[just_after] = C[just_after]
        better = C < best_c
        c_before[better] = prev[better] if prev is not None else np.inf
        c_after[better] = np.inf
        best_c[better] = C[better]
        best_k[better] = k
        prev = C
    from .render import parabola_depth
    depth = parabola_depth(depths, best_k, c_before, best_c, c_after)
    return dict(depth=depth, cost=best_c, kbest=best_k)


def otsu_split_depth(depth, valid=None, n_bins=256):
    """
    Split depth between the foreground (foliage) and background layers.

    Otsu's method (cv2.threshold with THRESH_OTSU) on the inverse depth
    quantised to 8 bits: inverse depth is proportional to image parallax, so
    the two layers form two well separated histogram modes.  Returns the split
    depth in metres.
    """
    inv = 1.0 / np.maximum(depth, 1e-6)
    if valid is not None:
        inv = inv[valid]
    inv = inv.ravel()
    lo, hi = np.percentile(inv, 0.5), np.percentile(inv, 99.5)
    q = np.clip((inv - lo) / max(hi - lo, 1e-9) * 255.0, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(q.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv_t = lo + (t + 0.5) / 255.0 * (hi - lo)
    return float(1.0 / inv_t)


def scaled_views(views, s):
    """Shallow copies of views resized by factor s (for the low-resolution mask sweep)."""
    from .camera import View
    out = []
    for v in views:
        im = cv2.resize(v.image, (int(round(v.image.shape[1] * s)), int(round(v.image.shape[0] * s))),
                        interpolation=cv2.INTER_AREA)
        out.append(View(v.name, im, scale_K(v.K, s), v.R, v.t))
    return out


def foreground_masks(rig, z_near, z_far, mask_scale=0.5, n_neighbours=6, win=5, keep=0.5,
                     step_px=2.0, z_split=None, dilate_px=2, verbose=False, max_offset=3,
                     n_threads=4, wall_front=None):
    """
    Colour-free occluder (foliage) masks for every view.

    1. Downsample the views by `mask_scale` (foliage is large; thin pipe
       cleaners remain several pixels wide).
    2. Robust plane sweep of every view against `n_neighbours` partners.
    3. Split depth = Otsu threshold of the reference depth map (unless given).
       With `wall_front` (how far fruit can hang in front of the wall) the split
       is never placed behind z_wall - wall_front - 10 mm, so fruit is never
       masked and a capture without foliage (unimodal depth histogram) gives
       empty masks instead of splitting the wall.
    4. Foreground = depth nearer than the split; a 3x3 opening removes speckle,
       then the mask is upsampled and dilated by `dilate_px` so defocused or
       fuzzy pipe-cleaner rims are also excluded.
    Returns dict(masks=[float32 HxW in {0,1}], depths=[low-res depth maps],
                 z_split, planes).
    """
    small = scaled_views(rig.views, mask_scale)
    f = small[rig.ref].K[0, 0]
    out_depths = [None] * len(small)

    def sweep_one(i):
        neigh = [j for j in choose_neighbours(len(small), i, 4 * n_neighbours)
                 if abs(j - i) <= max_offset][:n_neighbours]
        if len(neigh) < 2:
            neigh = choose_neighbours(len(small), i, n_neighbours)
        b_max = max(np.linalg.norm(rig.views[j].C - rig.views[i].C) for j in neigh)
        planes_i = inverse_depth_planes(z_near, z_far, step_px, f, b_max)
        return robust_sweep(small, i, planes_i, neigh, win=win, keep=keep)["depth"], planes_i

    # reference first (it fixes the split depth), then the other views in parallel threads
    out_depths[rig.ref], planes = sweep_one(rig.ref)
    if z_split is None:
        z_split = otsu_split_depth(out_depths[rig.ref])
        if wall_front is not None:
            d_ref = out_depths[rig.ref]
            far = d_ref[d_ref > z_split]
            z_wall = float(np.median(far)) if far.size else float(np.median(d_ref))
            z_split = min(z_split, z_wall - wall_front - 0.01)
    if verbose:
        print(f"    split depth (Otsu on inverse depth): {z_split:.3f} m")
    others = [i for i in range(len(small)) if i != rig.ref]
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        for i, (d, _) in zip(others, ex.map(sweep_one, others)):
            out_depths[i] = d
    h, w = rig.views[0].image.shape[:2]
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    masks = []
    for d in out_depths:
        m = (d < z_split).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        if dilate_px > 0:
            m = cv2.dilate(m, kd)
        masks.append(m.astype(np.float32))
    return dict(masks=masks, depths=out_depths, z_split=z_split, planes=planes)


def inverse_depth_planes(z_near, z_far, step_px, f_px, baseline):
    """Depth planes uniform in inverse depth, `step_px` pixels apart at `baseline`."""
    inv0, inv1 = 1.0 / z_near, 1.0 / z_far
    d_inv = step_px / (f_px * max(baseline, 1e-6))
    n = max(2, int(np.ceil((inv0 - inv1) / d_inv)) + 1)
    return 1.0 / np.linspace(inv0, inv1, n)
