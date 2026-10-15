"""
render.py - occlusion-aware synthetic-aperture rendering of the leaf-free image.

Baselines (single-plane synthetic aperture, Isaksen et al. 2000; Vaish et al.
2006): every photo is warped onto the reference view through ONE focal plane
and the stack is combined per pixel by the mean or the median.  The mean
leaves ghosts of the leaves; the median fails wherever leaves cover a point in
half of the photos or more, and fruit off the focal plane is blurred.

Occlusion-aware method (this work)
----------------------------------
1. Visibility: view j can see a background point only where its foliage mask
   (planesweep.foreground_masks, from depth, not colour) is empty at the
   point's projection.  The mask is warped with the same homography as the
   photo, so v_j(p, Z) = 1 - M_j(H_j(Z) p) (0 outside the photo).
2. Background depth: a second plane sweep over the background depth range
   only.  Its cost is the visibility-weighted, truncated colour variance of the
   visible samples, aggregated over a window:
       C(p, Z) = sum_win sum_j v_j min(|s_j - mu|^2, tau^2) / sum_win sum_j v_j,
   with mu the visibility-weighted mean of the samples s_j at (p, Z).  The
   minimum over Z (refined by a parabola) is the per-pixel background depth.
3. All-in-focus leaf-free image: each reference pixel is reprojected into
   every view at ITS OWN depth (cv2.remap with per-pixel maps) and the visible
   samples are averaged.  Fruit at different depths stays sharp, unlike a
   single focal plane.
4. Pixels that no photo sees are flagged (`support` < 1) and inpainted only
   for display (Telea 2004); detections there are not trusted.

Dependencies: numpy, opencv-python.
"""

import numpy as np
import cv2

from .camera import plane_homography, relative_pose, warp_by_homography, depth_maps_to_remap
from .planesweep import box_mean


def warp_view_and_visibility(rig, j, H, size, mask=None):
    """Warp photo j and its visibility (1 - foliage mask) through homography H."""
    vj = rig.views[j]
    img = warp_by_homography(vj.image, H, size, border_value=-1.0)
    inside = (img[..., 0] > -0.5).astype(np.float32)
    if mask is None:
        vis = inside
    else:
        vis = warp_by_homography(1.0 - mask, H, size, border_value=0.0) * inside
    return img, vis


def refocus_plane(rig, depth, mode="mean", masks=None, min_views=2):
    """
    Single-plane synthetic-aperture image at `depth` (the classic baseline).

    mode "mean" or "median"; with `masks`, only visible samples are combined.
    Returns (image float32 BGR, support count).
    """
    ref = rig.views[rig.ref]
    h, w = ref.image.shape[:2]
    imgs, viss = [], []
    for j, vj in enumerate(rig.views):
        R, t = relative_pose(ref, vj)
        H = plane_homography(ref.K, vj.K, R, t, depth)
        img, vis = warp_view_and_visibility(rig, j, H, (w, h), None if masks is None else masks[j])
        imgs.append(img)
        viss.append(vis)
    imgs = np.stack(imgs)
    viss = np.stack(viss)
    support = viss.sum(0)
    if mode == "mean":
        out = (imgs * viss[..., None]).sum(0) / np.maximum(support, 1e-6)[..., None]
    else:
        stack = np.where(viss[..., None] > 0.5, imgs, np.nan)
        with np.errstate(all="ignore"):
            out = np.nanmedian(stack, axis=0)
        out = np.nan_to_num(out, nan=0.0)
    out[support < min_views - 1e-3] = 0
    return out.astype(np.float32), support


def background_sweep(rig, masks, depths, wins=(3, 7), tau=0.15, min_support=2.0):
    """
    Visibility-weighted plane sweep of the reference view over background depths.

    Two passes per plane, one view at a time (no large stacks in memory):
    pass 1 accumulates the visibility-weighted mean mu of the warped samples,
    pass 2 accumulates the truncated squared deviations from mu.  The plane
    cost is the window-aggregated ratio of the two sums (see module notes).

    Several aggregation windows are tracked in the same pass (`wins`): a small
    window keeps small fruit sharp in depth, a larger one gives the cleaner
    colour render.  Returns dict win -> dict(depth, cost, support, confidence),
    with confidence = 1 - C_min / mean_Z C (near 0 where the cost curve is flat,
    e.g. untextured or barely observed pixels).
    """
    ref = rig.views[rig.ref]
    h, w = ref.image.shape[:2]
    rel = [relative_pose(ref, v) for v in rig.views]
    st = {}
    for win in wins:
        st[win] = dict(best_c=np.full((h, w), np.inf, np.float32), best_k=np.zeros((h, w), np.int32),
                       best_s=np.zeros((h, w), np.float32), c_before=np.full((h, w), np.inf, np.float32),
                       c_after=np.full((h, w), np.inf, np.float32), prev=None,
                       c_sum=np.zeros((h, w), np.float32))
    n = len(rig.views)
    S1 = np.empty((h, w, 3), np.float32)
    W = np.empty((h, w), np.float32)
    num = np.empty((h, w), np.float32)
    tmp3 = np.empty((h, w, 3), np.float32)
    for k, d in enumerate(depths):
        Hs = [plane_homography(ref.K, rig.views[j].K, rel[j][0], rel[j][1], d) for j in range(n)]
        S1.fill(0)
        W.fill(0)
        for j in range(n):
            img, vis = warp_view_and_visibility(rig, j, Hs[j], (w, h), masks[j])
            np.multiply(img, vis[..., None], out=tmp3)
            S1 += tmp3
            W += vis
        mu = S1 / np.maximum(W, 1e-6)[..., None]
        num.fill(0)
        for j in range(n):
            img, vis = warp_view_and_visibility(rig, j, Hs[j], (w, h), masks[j])
            np.subtract(img, mu, out=tmp3)
            np.multiply(tmp3, tmp3, out=tmp3)
            dev = tmp3.sum(axis=-1)
            np.minimum(dev, tau * tau, out=dev)
            num += dev * vis
        # pixels seen by fewer than min_support views get the maximal cost
        num_c = np.where(W >= min_support, num, tau * tau * np.maximum(W, 1.0))
        Wc = np.maximum(W, 1.0)
        for win in wins:
            s = st[win]
            C = box_mean(num_c, win) / box_mean(Wc, win)
            if s["prev"] is not None:
                just_after = s["best_k"] == (k - 1)
                s["c_after"][just_after] = C[just_after]
            better = C < s["best_c"]
            s["c_before"][better] = s["prev"][better] if s["prev"] is not None else np.inf
            s["c_after"][better] = np.inf
            s["best_c"][better] = C[better]
            s["best_k"][better] = k
            s["best_s"][better] = W[better]
            s["c_sum"] += C
            s["prev"] = C
    out = {}
    for win in wins:
        s = st[win]
        c_mean = s["c_sum"] / len(depths)
        conf = 1.0 - s["best_c"] / np.maximum(c_mean, 1e-12)
        out[win] = dict(depth=parabola_depth(depths, s["best_k"], s["c_before"], s["best_c"], s["c_after"]),
                        cost=s["best_c"], support=s["best_s"], confidence=conf.astype(np.float32))
    return out


def rail_sides(rig):
    """
    Split the views into the two sides of the reference along the rail.

    The rail direction is the principal axis of the camera centres; a view is
    on the left (right) when its centre projects before (after) the reference.
    The reference belongs to both sides.  Returns (left, right) index lists.
    """
    C = np.array([v.C for v in rig.views])
    c0 = C[rig.ref]
    _, _, Vt = np.linalg.svd(C - C.mean(0))
    a = Vt[0]
    s = (C - c0) @ a
    left = [j for j in range(len(C)) if s[j] <= 1e-9]
    right = [j for j in range(len(C)) if s[j] >= -1e-9]
    return left, right


def background_cost_volume(rig, masks, depths, tau=0.15, min_support=2.0, pre_win=3, half_select=True):
    """
    Visibility-weighted photo-consistency cost volume of the reference view.

    Per-pixel cost: unbiased truncated variance of the visible samples,
        c = sum_j v_j min(|s_j - mu|^2, tau^2) / (sum_j v_j - 1),
    lightly aggregated over pre_win x pre_win and normalised by tau^2.

    Half-occlusion handling (Kang, Szeliski and Chai 2001): fruit hanging in
    front of the backdrop hides the backdrop beside it from the views on ONE
    side of the rail only.  With half_select=True the cost is also computed from
    the left and the right half of the rail and the smallest of the three is
    kept; the unbiased variance makes the full and half apertures comparable.

    Returns (vol_all, vol_sel, support), each D x H x W float32: the
    full-aperture cost, the half-selected cost (equal to vol_all when
    half_select is False) and the visible-view count.
    """
    ref = rig.views[rig.ref]
    h, w = ref.image.shape[:2]
    rel = [relative_pose(ref, v) for v in rig.views]
    n = len(rig.views)
    groups = [list(range(n))]
    if half_select:
        left, right = rail_sides(rig)
        if len(left) >= 2 and len(right) >= 2:
            groups += [left, right]
    member = np.zeros((len(groups), n), bool)
    for g, idx in enumerate(groups):
        member[g, idx] = True
    vol = np.empty((len(depths), h, w), np.float32)
    vol_all = np.empty((len(depths), h, w), np.float32) if len(groups) > 1 else vol
    sup = np.empty((len(depths), h, w), np.float32)
    S1 = np.zeros((len(groups), h, w, 3), np.float32)
    W = np.zeros((len(groups), h, w), np.float32)
    num = np.zeros((len(groups), h, w), np.float32)
    tmp3 = np.empty((h, w, 3), np.float32)
    for k, d in enumerate(depths):
        Hs = [plane_homography(ref.K, rig.views[j].K, rel[j][0], rel[j][1], d) for j in range(n)]
        S1.fill(0)
        W.fill(0)
        num.fill(0)
        for j in range(n):
            img, vis = warp_view_and_visibility(rig, j, Hs[j], (w, h), masks[j])
            np.multiply(img, vis[..., None], out=tmp3)
            for g in range(len(groups)):
                if member[g, j]:
                    S1[g] += tmp3
                    W[g] += vis
        mu = S1 / np.maximum(W, 1e-6)[..., None]
        for j in range(n):
            img, vis = warp_view_and_visibility(rig, j, Hs[j], (w, h), masks[j])
            for g in range(len(groups)):
                if member[g, j]:
                    np.subtract(img, mu[g], out=tmp3)
                    np.multiply(tmp3, tmp3, out=tmp3)
                    dev = tmp3.sum(axis=-1)
                    np.minimum(dev, tau * tau, out=dev)
                    num[g] += dev * vis
        best = None
        for g in range(len(groups)):
            # unbiased variance; fewer than min_support visible views -> maximal cost
            c = np.where(W[g] >= min_support, num[g] / np.maximum(W[g] - 1.0, 1e-6), tau * tau)
            c = box_mean(c, pre_win) / (tau * tau)
            if g == 0 and len(groups) > 1:
                vol_all[k] = np.minimum(c, 1.0)
            best = c if best is None else np.minimum(best, c)
        vol[k] = np.minimum(best, 1.0)
        sup[k] = W[0]
    return vol_all, vol, sup


def parabola_depth(depths, best_k, c_before, best_c, c_after):
    """
    Sub-plane depth from a parabola through the costs of planes k-1, k, k+1.
    Planes are uniform in inverse depth, so the offset is applied in inverse depth.
    """
    a, b, c = c_before, best_c, c_after
    ok = np.isfinite(a) & np.isfinite(c)
    den = np.where(ok, a - 2 * b + c, 0.0)
    safe = ok & (den > 1e-9)
    off = np.zeros_like(b)
    off[safe] = 0.5 * (a[safe] - c[safe]) / den[safe]
    off = np.clip(off, -0.5, 0.5)
    inv = 1.0 / np.asarray(depths, np.float64)
    kk = np.clip(best_k + off, 0, len(depths) - 1)
    k0 = np.floor(kk).astype(np.int32)
    k1 = np.minimum(k0 + 1, len(depths) - 1)
    frac = kk - k0
    inv_d = (1 - frac) * inv[k0] + frac * inv[k1]
    return (1.0 / inv_d).astype(np.float32)


def render_all_in_focus(rig, depth, masks=None, robust=True, k_sigma=2.5):
    """
    Leaf-free, all-in-focus image: reproject every reference pixel into each view
    at its own depth (per-pixel cv2.remap) and average the visible samples.

    With robust=True one reweighting pass drops samples further than k_sigma
    standard deviations from the first mean (residual leaf pixels that slipped
    past the masks, specular highlights).
    Returns (image, support) with support = number of views used per pixel.
    """
    ref = rig.views[rig.ref]
    h, w = ref.image.shape[:2]
    n = len(rig.views)
    imgs = np.empty((n, h, w, 3), np.float32)
    viss = np.empty((n, h, w), np.float32)
    for j, vj in enumerate(rig.views):
        R, t = relative_pose(ref, vj)
        mx, my = depth_maps_to_remap(ref.K, vj.K, R, t, depth, (w, h))
        img = cv2.remap(vj.image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(-1.0, -1.0, -1.0, -1.0))
        inside = (img[..., 0] > -0.5).astype(np.float32)
        if masks is not None:
            vis = cv2.remap(1.0 - masks[j], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0.0) * inside
        else:
            vis = inside
        imgs[j] = img
        viss[j] = vis
    W = viss.sum(0)
    mu = (imgs * viss[..., None]).sum(0) / np.maximum(W, 1e-6)[..., None]
    if robust:
        dev = np.sqrt(np.sum((imgs - mu[None]) ** 2, axis=-1))
        sd = np.sqrt((viss * dev ** 2).sum(0) / np.maximum(W, 1e-6))
        keep = (dev <= k_sigma * np.maximum(sd, 0.02)[None]).astype(np.float32) * viss
        W2 = keep.sum(0)
        mu2 = (imgs * keep[..., None]).sum(0) / np.maximum(W2, 1e-6)[..., None]
        mu = np.where((W2 >= 1.0)[..., None], mu2, mu)
        W = np.where(W2 >= 1.0, W2, W)
    return mu.astype(np.float32), W.astype(np.float32)


def inpaint_holes(img, support, min_support=0.5, radius=5):
    """Display-only fill of pixels that no photo sees (cv2.inpaint, Telea 2004)."""
    hole = (support < min_support).astype(np.uint8)
    if not np.any(hole):
        return img.copy(), hole
    u8 = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
    out = cv2.inpaint(u8, hole, radius, cv2.INPAINT_TELEA)
    return out.astype(np.float32) / 255.0, hole
