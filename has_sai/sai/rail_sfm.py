"""
rail_sfm.py - camera poses of a rail capture with OpenCV only (no COLMAP needed).

Structure from motion specialised to a camera on a straight rail:

1. SIFT features (Lowe 2004) in every photo, matched between every pair with
   Lowe's ratio test and a mutual-nearest check.
2. Geometric verification of each pair with the essential matrix
   (cv2.findEssentialMat, RANSAC; Nister 2004).  Pure translation along the
   rail is not degenerate for E (it is the skew matrix of t).
3. Tracks: verified matches are chained across photos with union-find; a track
   holding two keypoints of the same photo is dropped.
4. Initialisation from the widest reliable pair through the reference photo
   (cv2.recoverPose), scaled so the two centres are the known rail distance
   apart; tracks seen by both are triangulated.
5. Incremental registration of the remaining photos with cv2.solvePnPRansac,
   then alternating refinement: N-view DLT re-triangulation of every track and
   iterative PnP of every camera (a light bundle adjustment).  The reference
   camera stays fixed (it is the world frame).
6. Metric scale: the mean distance between consecutive camera centres is set
   to the rail spacing (50 mm), so depths come out in metres.

Diagnostics (reprojection error, straightness of the camera path, spacing
error) are returned so the rail quality can be reported.

Intrinsics must be known (COLMAP, EXIF 35 mm-equivalent focal length or a
calibration): a camera that only translates cannot calibrate itself.

Dependencies: numpy, opencv-python.
"""

import numpy as np
import cv2


class _UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, a):
        p = self.p
        p.setdefault(a, a)
        root = a
        while p[root] != root:
            root = p[root]
        while p[a] != root:
            p[a], a = root, p[a]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def detect_features(gray_images, n_features=0):
    """SIFT keypoints (pixel coordinates) and descriptors for every image."""
    sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=0.01)
    out = []
    for g in gray_images:
        kp, des = sift.detectAndCompute(g, None)
        pts = np.array([k.pt for k in kp], np.float64).reshape(-1, 2)
        out.append((pts, des))
    return out


def match_pair(des1, des2, ratio=0.75):
    """Ratio-test + mutual nearest-neighbour matches; returns (idx1, idx2) arrays."""
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return np.zeros(0, int), np.zeros(0, int)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    m12 = bf.knnMatch(des1, des2, k=2)
    m21 = bf.knnMatch(des2, des1, k=1)
    back = {m[0].queryIdx: m[0].trainIdx for m in m21 if m}
    i1, i2 = [], []
    for m in m12:
        if len(m) < 2:
            continue
        a, b = m
        if a.distance < ratio * b.distance and back.get(a.trainIdx) == a.queryIdx:
            i1.append(a.queryIdx)
            i2.append(a.trainIdx)
    return np.array(i1, int), np.array(i2, int)


def triangulate_nview(Ps, xs):
    """Linear N-view triangulation (DLT) of one point; Ps list of 3x4, xs list of (u, v)."""
    A = []
    for P, (u, v) in zip(Ps, xs):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3]


def solve_centre_translation_only(K, X, uv):
    """
    Camera centre C of a camera with R = I from 3-D points X (M x 3) and their
    pixels uv (M x 2).  With x_c = X - C the projection equations are linear in C:
        fx C_x - (u - cx) C_z = fx X - (u - cx) Z
        fy C_y - (v - cy) C_z = fy Y - (v - cy) Z
    solved by least squares, then once more without the 10 % worst residuals.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    du, dv = uv[:, 0] - cx, uv[:, 1] - cy
    A = np.zeros((2 * len(X), 3))
    b = np.zeros(2 * len(X))
    A[0::2, 0] = fx
    A[0::2, 2] = -du
    b[0::2] = fx * X[:, 0] - du * X[:, 2]
    A[1::2, 1] = fy
    A[1::2, 2] = -dv
    b[1::2] = fy * X[:, 1] - dv * X[:, 2]
    C = np.linalg.lstsq(A, b, rcond=None)[0]
    r = np.abs(A @ C - b).reshape(-1, 2).sum(1)
    keep = np.repeat(r <= np.quantile(r, 0.9), 2)
    if keep.sum() >= 6:
        C = np.linalg.lstsq(A[keep], b[keep], rcond=None)[0]
    return C


def global_translation_ls(K, tracks, feats, n, ref, spacing, X0, C0, n_iter=6, outlier_px=2.0):
    """
    Global known-rotation (R = I) structure from motion by linear least squares.

    For camera i and point j the pixel (u, v) satisfies A_ij (X_j - C_i) = 0 with
        A_ij = [[fx, 0, -(u - cx)], [0, fy, -(v - cy)]],
    which is LINEAR in every unknown.  Each equation is weighted by 1 / depth
    (from the previous iterate) so the algebraic residual approximates the pixel
    error.  Points are eliminated with the Schur complement
        N_j = sum_i M_ij,   M_ij = A_ij^T A_ij w_ij^2,
        S C = 0 with S_ik = delta_ik sum_j M_ij - sum_j M_ij N_j^-1 M_kj,
    and the camera system is solved with the gauge fixed (C_ref = 0) and the
    scale fixed by the rail length along the current rail direction.
    Observations with reprojection error above `outlier_px` are dropped and the
    solve repeated.  Returns (C list, X dict track -> point, rmse).
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    obs = {ti: dict(tr) for ti, tr in enumerate(tracks) if ti in X0}
    C = np.array(C0, np.float64)
    X = {ti: np.array(x, np.float64) for ti, x in X0.items()}
    rmse = None
    for it in range(n_iter):
        S = np.zeros((3 * n, 3 * n))
        Nj_inv, Mj = {}, {}
        for ti, tr in obs.items():
            if len(tr) < 2:
                continue
            Ms = {}
            N = np.zeros((3, 3))
            for i, kp in tr.items():
                u, v = feats[i][0][kp]
                A = np.array([[fx, 0, -(u - cx)], [0, fy, -(v - cy)]])
                z = max(X[ti][2] - C[i][2], 1e-3)
                M = (A.T @ A) / (z * z)
                Ms[i] = M
                N += M
            Ni = np.linalg.inv(N + 1e-12 * np.eye(3))
            Nj_inv[ti], Mj[ti] = Ni, Ms
            for i, Mi in Ms.items():
                S[3 * i:3 * i + 3, 3 * i:3 * i + 3] += Mi
                for k, Mk in Ms.items():
                    S[3 * i:3 * i + 3, 3 * k:3 * k + 3] -= Mi @ Ni @ Mk
        # constraints: C_ref = 0 and g . C = L (rail length along the current direction)
        d = C[-1] - C[0]
        d = d / max(np.linalg.norm(d), 1e-9)
        g = np.zeros(3 * n)
        g[3 * (n - 1):3 * n] = d
        g[0:3] -= d
        free = [k for k in range(3 * n) if k // 3 != ref]
        Sf = S[np.ix_(free, free)]
        gf = g[free]
        L = spacing * (n - 1)
        KKT = np.zeros((len(free) + 1, len(free) + 1))
        KKT[:-1, :-1] = 2 * Sf
        KKT[:-1, -1] = gf
        KKT[-1, :-1] = gf
        rhs = np.zeros(len(free) + 1)
        rhs[-1] = L
        sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
        Cv = np.zeros(3 * n)
        Cv[free] = sol[:-1]
        C = Cv.reshape(n, 3)
        # back-substitute points, then prune outlier observations
        errs = []
        for ti in list(Mj.keys()):
            rhs_p = sum(Mj[ti][i] @ C[i] for i in Mj[ti])
            X[ti] = Nj_inv[ti] @ rhs_p
            for i, kp in list(obs[ti].items()):
                xc = X[ti] - C[i]
                if xc[2] <= 1e-3:
                    e = 1e9
                else:
                    e = np.hypot(fx * xc[0] / xc[2] + cx - feats[i][0][kp][0],
                                 fy * xc[1] / xc[2] + cy - feats[i][0][kp][1])
                errs.append(min(e, 1e3))
                if it < n_iter - 1 and e > outlier_px:
                    del obs[ti][i]
        rmse = float(np.sqrt(np.mean(np.square(errs)))) if errs else None
    X = {ti: X[ti] for ti in obs if len(obs[ti]) >= 2}
    return [C[i] for i in range(n)], X, rmse


def estimate_rail_poses(gray_images, K, spacing, ref=None, max_pair_gap=None, n_iter=5, verbose=True,
                        model="translation"):
    """
    Poses (R_i, t_i, world = reference camera frame) of a rail capture.

    Inputs:  gray_images (uint8 list, same size), K (3x3 for that size),
             spacing (rail step, metres), ref (reference index, default centre),
             model "translation" (rail prior: a rigid carriage does not rotate, so
             every R_i = I and only the centres are estimated; removes the
             rotation-translation ambiguity of a short lateral baseline) or
             "full" (independent rotation per photo).
    Output:  dict(R=[...], t=[...], points (M x 3), diagnostics).
    """
    n = len(gray_images)
    ref = n // 2 if ref is None else ref
    feats = detect_features(gray_images)
    pairs = {}
    for i in range(n):
        for j in range(i + 1, n):
            if max_pair_gap is not None and j - i > max_pair_gap:
                continue
            a, b = match_pair(feats[i][1], feats[j][1])
            if len(a) < 30:
                continue
            p1, p2 = feats[i][0][a], feats[j][0][b]
            E, inl = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is None or inl is None:
                continue
            inl = inl.ravel().astype(bool)
            if inl.sum() < 25:
                continue
            pairs[(i, j)] = (a[inl], b[inl])
    if verbose:
        print(f"    rail SfM: {len(pairs)} verified pairs, "
              f"median {int(np.median([len(v[0]) for v in pairs.values()])) if pairs else 0} inliers")
    # tracks by union-find over (image, keypoint)
    uf = _UnionFind()
    for (i, j), (a, b) in pairs.items():
        for x, y in zip(a, b):
            uf.union((i, int(x)), (j, int(y)))
    groups = {}
    for node in list(uf.p.keys()):
        groups.setdefault(uf.find(node), []).append(node)
    tracks = []
    for nodes in groups.values():
        imgs = [nd[0] for nd in nodes]
        if len(nodes) >= 2 and len(set(imgs)) == len(imgs):
            tracks.append({im: kp for im, kp in nodes})
    # initial pair: reference with the farthest well-matched photo
    best = None
    for (i, j), (a, b) in pairs.items():
        if ref not in (i, j):
            continue
        other = j if i == ref else i
        score = len(a) * abs(other - ref)
        if best is None or score > best[0]:
            best = (score, other)
    if best is None:
        raise RuntimeError("no verified pair through the reference photo")
    other = best[1]
    key = (min(ref, other), max(ref, other))
    a, b = pairs[key]
    p_ref = feats[ref][0][a if key[0] == ref else b]
    p_oth = feats[other][0][b if key[0] == ref else a]
    _, R, t, _ = cv2.recoverPose(cv2.findEssentialMat(p_ref, p_oth, K, cv2.RANSAC, 0.999, 1.0)[0],
                                 p_ref, p_oth, K)
    t = t.ravel() / np.linalg.norm(t) * spacing * abs(other - ref)
    Rs = [None] * n
    ts = [None] * n
    Rs[ref], ts[ref] = np.eye(3), np.zeros(3)
    Rs[other], ts[other] = R, t
    P = lambda i: K @ np.hstack([Rs[i], ts[i].reshape(3, 1)])
    pts3 = {}

    def retriangulate(min_views=2):
        for ti, tr in enumerate(tracks):
            reg = [im for im in tr if Rs[im] is not None]
            if len(reg) < min_views:
                continue
            X = triangulate_nview([P(im) for im in reg], [feats[im][0][tr[im]] for im in reg])
            # cheirality and reprojection check
            ok = True
            errs = []
            for im in reg:
                xc = Rs[im] @ X + ts[im]
                if xc[2] <= 0:
                    ok = False
                    break
                uv = (K @ xc)[:2] / xc[2]
                errs.append(np.linalg.norm(uv - feats[im][0][tr[im]]))
            if ok and np.median(errs) < 4.0:
                pts3[ti] = X
            elif ti in pts3:
                del pts3[ti]

    retriangulate()
    order = sorted([i for i in range(n) if Rs[i] is None], key=lambda i: abs(i - ref))
    for i in order:
        obj, img = [], []
        for ti, X in pts3.items():
            if i in tracks[ti]:
                obj.append(X)
                img.append(feats[i][0][tracks[ti][i]])
        if len(obj) < 12:
            continue
        ok, rvec, tvec, inl = cv2.solvePnPRansac(np.array(obj), np.array(img), K, None,
                                                 reprojectionError=2.0, iterationsCount=2000,
                                                 flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            continue
        Rs[i] = cv2.Rodrigues(rvec)[0]
        ts[i] = tvec.ravel()
        retriangulate()
    missing = [i for i in range(n) if Rs[i] is None]
    if missing:
        raise RuntimeError(f"could not register photos {missing}")
    if model == "translation":
        # rail prior: keep only the centres, set every rotation to the identity
        Cs = [-Rs[i].T @ ts[i] for i in range(n)]
        Rs = [np.eye(3) for _ in range(n)]
        ts = [-C for C in Cs]
    # alternating refinement (light bundle adjustment), reference fixed
    for _ in range(n_iter if model == "full" else 0):
        retriangulate()
        for i in range(n):
            if i == ref:
                continue
            obj, img = [], []
            for ti, X in pts3.items():
                if i in tracks[ti]:
                    obj.append(X)
                    img.append(feats[i][0][tracks[ti][i]])
            rvec = cv2.Rodrigues(Rs[i])[0]
            ok, rvec, tvec = cv2.solvePnP(np.array(obj), np.array(img), K, None, rvec,
                                          ts[i].reshape(3, 1).copy(), True, cv2.SOLVEPNP_ITERATIVE)
            if ok:
                Rs[i] = cv2.Rodrigues(rvec)[0]
                ts[i] = tvec.ravel()
        # metric scale from the rail spacing
        C = np.array([-Rs[i].T @ ts[i] for i in range(n)])
        s = spacing / np.mean(np.linalg.norm(np.diff(C, axis=0), axis=1))
        ts = [t_ * s for t_ in ts]
        for ti in pts3:
            pts3[ti] = pts3[ti] * s
    if model == "translation":
        retriangulate()
        Cs, pts_new, rmse = global_translation_ls(K, tracks, feats, n, ref, spacing, pts3,
                                                  [-t_ for t_ in ts])
        ts = [-C for C in Cs]
        pts3.clear()
        pts3.update(pts_new)
    retriangulate()
    # diagnostics
    errs = []
    for ti, X in pts3.items():
        for im, kp in tracks[ti].items():
            xc = Rs[im] @ X + ts[im]
            uv = (K @ xc)[:2] / xc[2]
            errs.append(np.linalg.norm(uv - feats[im][0][kp]))
    C = np.array([-Rs[i].T @ ts[i] for i in range(n)])
    _, _, Vt = np.linalg.svd(C - C.mean(0))
    off_line = (C - C.mean(0)) - np.outer((C - C.mean(0)) @ Vt[0], Vt[0])
    steps = np.linalg.norm(np.diff(C, axis=0), axis=1)
    rot_deg = [float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))) for R in Rs]
    diag = dict(n_points=len(pts3), reproj_rmse_px=float(np.sqrt(np.mean(np.square(errs)))) if errs else None,
                path_straightness_mm=float(np.sqrt(np.mean(np.sum(off_line ** 2, 1))) * 1e3),
                spacing_mm=(steps * 1e3).round(2).tolist(), rotation_deg=np.round(rot_deg, 3).tolist())
    if verbose:
        print(f"    rail SfM: {len(pts3)} points, reprojection RMSE {diag['reproj_rmse_px']:.2f} px, "
              f"path deviation from a line {diag['path_straightness_mm']:.2f} mm")
    return dict(R=Rs, t=ts, points=np.array(list(pts3.values())), diagnostics=diag)
