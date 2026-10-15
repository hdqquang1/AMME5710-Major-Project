"""
sgm.py - semi-global aggregation of a plane-sweep cost volume (Hirschmueller 2008).

Window costs alone are noisy on weakly textured surfaces (the canopy print) and
smear depth across object edges.  Semi-global matching adds a smoothness prior
by dynamic programming along several 1-D paths through the image:

    L_r(p, k) = C(p, k) + min( L_r(p - r, k),
                               L_r(p - r, k +/- 1) + P1,
                               min_i L_r(p - r, i) + P2 ) - min_i L_r(p - r, i)

and S(p, k) = sum_r L_r(p, k).  P1 penalises a change of one depth plane
(slanted surfaces), P2 any larger jump (depth edges).  P2 is divided by
(1 + alpha |dI|) along each path, so jumps are cheap across intensity edges of
the guide image and expensive inside smooth regions.  The plane index with the
smallest S, refined by a parabola, is the depth.

The same recurrence runs inside OpenCV's StereoSGBM for rectified pairs; here it
is applied to the multi-view plane-sweep volume (as in plane-sweep stereo with
SGM, e.g. Haene et al. 2014), written with NumPy and vectorised across each
scan line.

Dependencies: numpy.
"""

import numpy as np


def _path(C, G, P1, P2, alpha, reverse):
    """
    Aggregate along the first axis of C (shape N x D x M: N steps along the path,
    D planes, M independent scan lines).  G (N x M) is the guide intensity.
    Returns L with the same shape.
    """
    N, D, M = C.shape
    L = np.empty_like(C)
    rng = range(N - 1, -1, -1) if reverse else range(N)
    first = True
    prev = None
    g_prev = None
    big = np.float32(1e9)
    for n in rng:
        c = C[n]
        if first:
            L[n] = c
            first = False
        else:
            m = prev.min(axis=0)                                        # (M,)
            up = np.empty_like(prev)
            up[1:] = prev[:-1]
            up[0] = big
            dn = np.empty_like(prev)
            dn[:-1] = prev[1:]
            dn[-1] = big
            p2 = P2 / (1.0 + alpha * np.abs(G[n] - g_prev))              # edge-aware jump penalty
            best = np.minimum(prev, np.minimum(up, dn) + P1)
            best = np.minimum(best, (m + p2)[None, :])
            L[n] = c + best - m[None, :]
        prev = L[n]
        g_prev = G[n]
    return L


def sgm_aggregate(cost, guide, P1=0.08, P2=0.6, alpha=8.0):
    """
    Four-path semi-global aggregation of cost (D x H x W, normalised to ~[0, 1]).

    guide: H x W grey image in [0, 1] for the edge-aware P2.
    Returns S (D x H x W), the sum of the four path costs.
    """
    D, H, W = cost.shape
    C = cost.astype(np.float32)
    S = np.zeros_like(C)
    # horizontal paths: iterate over columns -> arrays (W, D, H)
    Ch = np.ascontiguousarray(C.transpose(2, 0, 1))
    Gh = np.ascontiguousarray(guide.T.astype(np.float32))
    for rev in (False, True):
        S += _path(Ch, Gh, P1, P2, alpha, rev).transpose(1, 2, 0)
    del Ch
    # vertical paths: iterate over rows -> arrays (H, D, W)
    Cv = np.ascontiguousarray(C.transpose(1, 0, 2))
    Gv = guide.astype(np.float32)
    for rev in (False, True):
        S += _path(Cv, Gv, P1, P2, alpha, rev).transpose(1, 0, 2)
    return S


def wta_parabola(S, depths):
    """Winner-take-all plane index with a parabola refinement; returns (depth, S_min, confidence)."""
    D = S.shape[0]
    k = np.argmin(S, axis=0)
    H, W = k.shape
    ii, jj = np.mgrid[0:H, 0:W]
    s0 = S[k, ii, jj]
    km = np.clip(k - 1, 0, D - 1)
    kp = np.clip(k + 1, 0, D - 1)
    a = S[km, ii, jj]
    c = S[kp, ii, jj]
    den = a - 2 * s0 + c
    off = np.zeros_like(s0)
    ok = (k > 0) & (k < D - 1) & (den > 1e-9)
    off[ok] = 0.5 * (a[ok] - c[ok]) / den[ok]
    off = np.clip(off, -0.5, 0.5)
    inv = 1.0 / np.asarray(depths, np.float64)
    kk = np.clip(k + off, 0, D - 1)
    k0 = np.floor(kk).astype(np.int32)
    k1 = np.minimum(k0 + 1, D - 1)
    frac = kk - k0
    inv_d = (1 - frac) * inv[k0] + frac * inv[k1]
    conf = 1.0 - s0 / np.maximum(S.mean(axis=0), 1e-12)
    return (1.0 / inv_d).astype(np.float32), s0, conf.astype(np.float32)
