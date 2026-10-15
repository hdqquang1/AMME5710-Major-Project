"""
synth.py - synthetic rail captures of a fruit wall behind foliage, with ground truth.

Why this exists
---------------
The real captures have no per-cherry ground truth for occlusion, depth or
visibility, so every stage of the pipeline is benchmarked first on scenes whose
answer is known exactly.  The scenes copy the physical rig of the pitch
(Appendix A): a camera translating along a straight rail, toy cherries hung in
twins on a timber diamond lattice with a printed canopy backdrop behind it, and
pipe-cleaner foliage (bamboo stakes, helical vines and wound leaves) hanging in
front of the fence.

Renderer
--------
A small ray caster written with NumPy.  Each pixel of each view casts a ray
X(t) = C + t d, with d = R^T K^-1 [u, v, 1]^T, so the ray parameter t equals the
depth along the camera Z axis.  Every primitive (backdrop plane, lattice slats,
spheres, tubes, vertical cylinders, planar leaves) is intersected only inside
its projected bounding box, and a z-buffer keeps the nearest hit.  The scene is
rendered twice per view, with and without foliage, which gives:

* the input photos and the foliage-free photos (upper bound / image ground truth);
* exact depth maps and object-id maps (occluder masks, per-fruit visibility);
* the 3-D centre and radius of every fruit.

Shading is Lambertian plus a Blinn-Phong specular term, so glossy fruit has
view-dependent highlights (the real toy cherries are glossy plastic, which
breaks the Lambertian assumption of photo-consistency).  Images are rendered
with 2x2 supersampling, blurred slightly (lens PSF) and corrupted with sensor
noise before 8-bit quantisation.

All lengths are in metres.  World frame = reference (centre) camera frame:
X right, Y down, Z forward (OpenCV convention).

Dependencies: numpy, opencv-python.
"""

import json
import math
import os

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Default geometry: matches the first real capture (leaves ~0.57 m, fruit
# ~0.91 m, 50 mm rail steps) and the 1x lens of an iPhone at 1/4 resolution.
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    width=1008, height=756,      # 4032 x 3024 photo at 1/4 scale (landscape)
    f_px=725.0,                  # ~2900 px at full resolution (26 mm-equivalent lens)
    n_views=13, spacing=0.05,    # 13 photos, 50 mm apart, over a 0.6 m rail
    z_fence=0.925,               # front face of the front lattice slats [m]
    slat_width=0.030, slat_thick=0.008, slat_pitch=0.095,
    backdrop_gap=0.010,          # canopy print sits 10 mm behind the lattice
    foliage_gap=0.34,            # foliage centre plane in front of the fence [m]
    foliage_spread=0.03,         # +/- depth spread of the foliage about its plane
    coverage=0.30,               # target fraction of the fruit ROI hidden by foliage (centre view)
    fruit="cherry",              # "cherry", "grape" or "lime"
    n_clusters=10,               # fruit clusters (cherry twins/quads, grape bunches, limes)
    pose_rot_deg=0.0,            # 1-sigma rotation jitter per view [deg]
    pose_trans_m=0.0,            # 1-sigma translation jitter per view [m]
    y_drift_m=0.0,               # linear vertical drift end to end [m] (rail not level)
    noise_sigma=1.5,             # sensor noise [8-bit grey levels]
    psf_sigma=0.45,              # lens blur [output px]
    supersample=2,
    seed=0,
)

# Fruit sizes and appearance.  Colours are linear-ish RGB in [0, 1].
FRUIT_SPECS = {
    "cherry": dict(radius=(0.0095, 0.0115), shininess=60.0, ks=(0.45, 0.75),
                   colours=[(0.62, 0.05, 0.08), (0.42, 0.02, 0.07), (0.78, 0.10, 0.12)]),
    "grape": dict(radius=(0.0075, 0.0090), shininess=35.0, ks=(0.20, 0.35),
                  colours=[(0.56, 0.70, 0.26), (0.50, 0.66, 0.22), (0.62, 0.74, 0.30)]),
    "lime": dict(radius=(0.024, 0.029), shininess=12.0, ks=(0.08, 0.15),
                 colours=[(0.36, 0.58, 0.16), (0.44, 0.64, 0.20), (0.32, 0.52, 0.14)]),
}

# Object-id ranges in the id buffer
ID_BACKDROP, ID_LATTICE, ID_STEM = 1, 2, 3
ID_FRUIT0 = 1000          # fruit k has id ID_FRUIT0 + k
ID_FOLIAGE = 900          # all foliage parts share one id (stakes, vines, leaves)


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------
def rodrigues(rvec):
    """Rotation matrix from an axis-angle vector (thin wrapper on cv2.Rodrigues)."""
    return cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))[0]


def make_rail_cameras(p, rng):
    """
    Camera intrinsics and extrinsics of a rail capture.

    Inputs:  p (parameter dict, see DEFAULTS), rng (np.random.Generator).
    Output:  K (3x3), list of dicts {R, t, C} with world->camera rotation R,
             translation t = -R C and centre C, and the reference index.
    Method:  centres sit on the X axis at x_i = (i - ref) * spacing; small
             Gaussian rotation/translation jitter and a linear vertical drift
             emulate an imperfect carriage.  The reference camera is exact
             (it defines the world frame).
    """
    n = p["n_views"]
    ref = n // 2
    K = np.array([[p["f_px"], 0, (p["width"] - 1) / 2.0],
                  [0, p["f_px"], (p["height"] - 1) / 2.0],
                  [0, 0, 1]], np.float64)
    cams = []
    for i in range(n):
        x = (i - ref) * p["spacing"]
        y = p["y_drift_m"] * (i - ref) / max(1, n - 1)
        C = np.array([x, y, 0.0])
        R = np.eye(3)
        if i != ref:
            C = C + rng.normal(0, p["pose_trans_m"], 3)
            R = rodrigues(np.deg2rad(rng.normal(0, p["pose_rot_deg"], 3)))
        cams.append(dict(R=R, t=-R @ C, C=C))
    return K, cams, ref


# ---------------------------------------------------------------------------
# Procedural textures
# ---------------------------------------------------------------------------
def canopy_texture(rng, half_x=1.3, half_y=1.0, res=0.001):
    """
    Printed-canopy backdrop: dark foliage with lighter leaves, as a texture
    image covering X in [-half_x, half_x], Y in [-half_y, half_y] at `res`
    metres per texel.  Returns (texture float32 HxWx3, (x0, y0, res)).
    """
    tw, th = int(2 * half_x / res), int(2 * half_y / res)
    tex = np.empty((th, tw, 3), np.float32)
    tex[:] = (0.08, 0.15, 0.07)
    n_leaves = int(tw * th / 260)
    for _ in range(n_leaves):
        c = (int(rng.uniform(0, tw)), int(rng.uniform(0, th)))
        L = rng.uniform(0.020, 0.060) / res
        ax = (max(2, int(L / 2)), max(1, int(L * rng.uniform(0.15, 0.30))))
        g = rng.uniform(0.55, 1.6)
        col = (0.09 * g, 0.19 * g * rng.uniform(0.9, 1.15), 0.07 * g)
        cv2.ellipse(tex, c, ax, rng.uniform(0, 180), 0, 360, col, -1, cv2.LINE_AA)
    # sparse bright gaps ("sky") and fine print grain
    for _ in range(int(tw * th / 40000)):
        c = (int(rng.uniform(0, tw)), int(rng.uniform(0, th)))
        cv2.circle(tex, c, int(rng.uniform(2, 9)), (0.55, 0.62, 0.50), -1, cv2.LINE_AA)
    tex *= (1.0 + 0.06 * rng.standard_normal((th, tw, 1)).astype(np.float32))
    tex = cv2.GaussianBlur(tex, (0, 0), 0.7)
    return np.clip(tex, 0, 1), (-half_x, -half_y, res)


def value_noise_1d(rng, n=4096):
    """Smooth 1-D noise table in [0, 1] (for wood grain and leaf fibres)."""
    v = rng.uniform(0, 1, n // 16 + 2).astype(np.float32)
    return cv2.resize(v.reshape(1, -1), (n, 1), interpolation=cv2.INTER_CUBIC).ravel()


# ---------------------------------------------------------------------------
# Scene sampling
# ---------------------------------------------------------------------------
def lattice_planes(p):
    """Depths of the lattice layers and the backdrop: (front A, front B, backdrop)."""
    zA = p["z_fence"]
    zB = zA + p["slat_thick"]
    zBack = zB + p["slat_thick"] + p["backdrop_gap"]
    return zA, zB, zBack


def on_front_slat(x, y, p):
    """True where (x, y) lies on a front-layer (+45 deg) slat of the lattice."""
    q = (-x * math.sin(math.radians(45)) + y * math.cos(math.radians(45)))
    return (q / p["slat_pitch"]) % 1.0 < p["slat_width"] / p["slat_pitch"]


def sample_fruit(p, rng):
    """
    Fruit spheres hung on the lattice.

    Cherries hang in twins from a front slat: each cluster is one twin (2) or a
    double twin (4).  Grapes form conical bunches of packed berries.  Limes
    hang singly.  Returns a list of dicts {c (3,), r, colour, ks, shin, cluster}
    and a list of stems [(p0, p1, radius)].
    """
    spec = FRUIT_SPECS[p["fruit"]]
    zA, _, _ = lattice_planes(p)
    fruit, stems = [], []
    anchors = []
    tries = 0
    while len(anchors) < p["n_clusters"] and tries < 5000:
        tries += 1
        x, y = rng.uniform(-0.30, 0.30), rng.uniform(-0.20, 0.17)
        if not on_front_slat(x, y, p):
            continue
        if any(math.hypot(x - a[0], y - a[1]) < (0.085 if p["fruit"] != "grape" else 0.15)
               for a in anchors):
            continue
        anchors.append((x, y))
    for k, (ax, ay) in enumerate(anchors):
        A = np.array([ax, ay, zA - 0.002])
        if p["fruit"] == "cherry":
            twins = 2 if rng.uniform() < 0.5 else 1
            for tw in range(twins):
                drop = rng.uniform(0.028, 0.045)
                for side in (-1, 1):
                    r = rng.uniform(*spec["radius"])
                    dx = side * (r + rng.uniform(0.0005, 0.003)) + (tw - 0.5) * 0.05 * (twins - 1)
                    c = np.array([ax + dx, ay + drop + rng.uniform(-0.004, 0.004),
                                  zA - r - rng.uniform(0.002, 0.012)])
                    col = np.array(spec["colours"][rng.integers(len(spec["colours"]))])
                    col = col * rng.uniform(0.85, 1.1)
                    fruit.append(dict(c=c, r=r, colour=col, ks=rng.uniform(*spec["ks"]),
                                      shin=spec["shininess"], cluster=k))
                    stems.append((A, c - np.array([0, r * 0.9, 0]), 0.0011))
        elif p["fruit"] == "grape":
            top = np.array([ax, ay + 0.02, zA - 0.035])
            stems.append((A, top, 0.002))
            berries = []
            n_target = int(rng.integers(16, 26))
            for _ in range(4000):
                if len(berries) >= n_target:
                    break
                s = rng.uniform(0, 1)                      # 0 at top, 1 at tip
                half_w = 0.045 * (1 - 0.75 * s)
                c = top + np.array([rng.uniform(-half_w, half_w), 0.012 + s * 0.12,
                                    rng.uniform(-0.022, 0.022)])
                r = rng.uniform(*spec["radius"])
                if c[2] + r > zA - 0.001:
                    continue
                if all(np.linalg.norm(c - b[0]) > 0.98 * (r + b[1]) for b in berries):
                    berries.append((c, r))
            for c, r in berries:
                col = np.array(spec["colours"][rng.integers(len(spec["colours"]))])
                fruit.append(dict(c=c, r=r, colour=col * rng.uniform(0.9, 1.08),
                                  ks=rng.uniform(*spec["ks"]), shin=spec["shininess"], cluster=k))
        else:  # lime
            r = rng.uniform(*spec["radius"])
            c = np.array([ax, ay + 0.012 + r, zA - r - rng.uniform(0.002, 0.01)])
            col = np.array(spec["colours"][rng.integers(len(spec["colours"]))])
            fruit.append(dict(c=c, r=r, colour=col, ks=rng.uniform(*spec["ks"]),
                              shin=spec["shininess"], cluster=k))
            stems.append((A, c - np.array([0, r * 0.95, 0]), 0.0018))
    return fruit, stems


def sample_foliage(p, rng, density):
    """
    Pipe-cleaner foliage: bamboo stakes with green vines wound round them and
    wound pipe-cleaner leaves along the vines.

    `density` scales the number of plants and leaves (tuned by bisection to hit
    the requested coverage).  Returns dict(stakes=[(x, z, r, y0, y1)],
    tubes=[(p0, p1, r, colour)], leaves=[dict(c, e1, e2, A, B, colour)]).
    """
    zF = p["z_fence"] - p["foliage_gap"]
    sp = p["foliage_spread"]
    n_plants = max(1, int(round(7 * density)))
    xs = np.linspace(-0.55, 0.55, n_plants) + rng.uniform(-0.04, 0.04, n_plants)
    stakes, tubes, leaves = [], [], []
    for xs_i in xs:
        zs = zF + rng.uniform(-sp, sp)
        stakes.append((xs_i, zs, 0.004, -0.40, 0.40))
        # helical vine around the stake
        amp = rng.uniform(0.010, 0.028)
        omega = 2 * math.pi / rng.uniform(0.12, 0.22)
        ph = rng.uniform(0, 2 * math.pi)
        ys = np.arange(-0.36, 0.36, 0.020)
        pts = np.stack([xs_i + amp * np.cos(omega * ys + ph), ys,
                        zs + amp * np.sin(omega * ys + ph)], axis=1)
        green = np.array([0.20, 0.55, 0.16]) * rng.uniform(0.8, 1.15)
        for a, b in zip(pts[:-1], pts[1:]):
            tubes.append((a, b, 0.0030, green))
        # leaves attached along the vine
        n_leaves = int(round(rng.uniform(10, 16) * density / max(1.0, n_plants / 7)))
        for _ in range(n_leaves):
            j = int(rng.integers(len(pts)))
            base = pts[j]
            A = rng.uniform(0.024, 0.036)
            B = A * rng.uniform(0.38, 0.50)
            yaw = rng.uniform(-0.9, 0.9)           # leaf plane turned about the vertical
            ang = rng.uniform(0, 2 * math.pi)      # leaf axis direction within its plane
            n = np.array([math.sin(yaw), rng.uniform(-0.3, 0.3), -math.cos(yaw)])
            n /= np.linalg.norm(n)
            tmp = np.array([math.cos(ang), math.sin(ang), 0.0])
            e1 = tmp - n * (tmp @ n)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(n, e1)
            c = base + e1 * A * 0.95
            col = np.array([0.18, 0.52, 0.15]) * rng.uniform(0.75, 1.2)
            leaves.append(dict(c=c, e1=e1, e2=e2, A=A, B=B, colour=col,
                               seed=int(rng.integers(1 << 30))))
    return dict(stakes=stakes, tubes=tubes, leaves=leaves)


# ---------------------------------------------------------------------------
# Ray caster
# ---------------------------------------------------------------------------
class Renderer:
    """
    Z-buffer ray caster for one camera at supersampled resolution.

    Buffers: depth `z` (camera Z of the nearest hit), colour `rgb`, object id
    `oid`.  Each `add_*` method intersects one primitive inside its projected
    bounding box and keeps hits nearer than the current buffer.
    """

    LIGHT = np.array([-0.35, -0.55, -0.75]) / np.linalg.norm([-0.35, -0.55, -0.75])
    AMBIENT = 0.38

    def __init__(self, K, R, t, W, H, ss):
        self.ss = ss
        self.W, self.H = W * ss, H * ss
        Ks = K.copy()
        Ks[0, 0] *= ss
        Ks[1, 1] *= ss
        Ks[0, 2] = ss * (K[0, 2] + 0.5) - 0.5
        Ks[1, 2] = ss * (K[1, 2] + 0.5) - 0.5
        self.K, self.R, self.t = Ks, R, t
        self.C = -R.T @ t
        self.z = np.full((self.H, self.W), np.inf, np.float32)
        self.rgb = np.zeros((self.H, self.W, 3), np.float32)
        self.oid = np.zeros((self.H, self.W), np.int32)

    # -- helpers ----------------------------------------------------------
    def rays(self, u0, u1, v0, v1):
        """World ray directions (camera-Z normalised) for the pixel block."""
        u, v = np.meshgrid(np.arange(u0, u1, dtype=np.float32),
                           np.arange(v0, v1, dtype=np.float32))
        K = self.K
        dc = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
        return dc @ self.R          # = (R^T dc^T)^T

    def bbox(self, pts_world, pad_px):
        """Pixel bounding box of world points (+ padding), clipped; None if empty/behind."""
        pc = pts_world @ self.R.T + self.t
        if np.any(pc[:, 2] <= 1e-3):
            return None
        uv = pc[:, :2] / pc[:, 2:3] * [self.K[0, 0], self.K[1, 1]] + self.K[:2, 2]
        u0 = int(max(0, math.floor(uv[:, 0].min() - pad_px)))
        u1 = int(min(self.W, math.ceil(uv[:, 0].max() + pad_px) + 1))
        v0 = int(max(0, math.floor(uv[:, 1].min() - pad_px)))
        v1 = int(min(self.H, math.ceil(uv[:, 1].max() + pad_px) + 1))
        if u0 >= u1 or v0 >= v1:
            return None
        return u0, u1, v0, v1

    def commit(self, box, t, rgb, oid):
        """Z-test the block hits `t` (inf = miss) and write colour/id where nearer."""
        u0, u1, v0, v1 = box
        zs = self.z[v0:v1, u0:u1]
        m = t < zs
        if not np.any(m):
            return
        zs[m] = t[m]
        self.rgb[v0:v1, u0:u1][m] = rgb[m]
        self.oid[v0:v1, u0:u1][m] = oid

    def shade(self, n, d, albedo, ks=0.0, shin=1.0, two_sided=False):
        """Lambert + Blinn-Phong shading; n unit normals, d ray directions."""
        L = self.LIGHT
        ndl = n @ L
        if two_sided:
            ndl = np.abs(ndl)
        lam = np.clip(ndl, 0, 1)
        out = albedo * (self.AMBIENT + (1 - self.AMBIENT) * lam[..., None])
        if ks > 0:
            V = -d / np.linalg.norm(d, axis=-1, keepdims=True)
            Hh = V + L
            Hh /= np.linalg.norm(Hh, axis=-1, keepdims=True)
            spec = ks * np.clip(np.sum(n * Hh, -1), 0, 1) ** shin
            out = out + spec[..., None]
        return out

    # -- primitives -------------------------------------------------------
    def add_backdrop(self, z_plane, tex, tex_origin):
        """Infinite textured plane Z = z_plane (the printed canopy)."""
        box = (0, self.W, 0, self.H)
        d = self.rays(*box)
        t = (z_plane - self.C[2]) / d[..., 2]
        X = self.C[0] + t * d[..., 0]
        Y = self.C[1] + t * d[..., 1]
        x0, y0, res = tex_origin
        mapx = ((X - x0) / res).astype(np.float32)
        mapy = ((Y - y0) / res).astype(np.float32)
        col = cv2.remap(tex, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        col = col * (self.AMBIENT + (1 - self.AMBIENT) * max(0.0, -self.LIGHT[2]))
        self.commit(box, t.astype(np.float32), col, ID_BACKDROP)

    def add_lattice(self, p, grain):
        """Two layers of diagonal slats (front +45 deg, back -45 deg); front faces."""
        zA, zB, _ = lattice_planes(p)
        box = (0, self.W, 0, self.H)
        d = self.rays(*box)
        wood = np.array([0.62, 0.45, 0.28])
        for z_layer, ang, shift in ((zB, -45.0, 0.31), (zA, 45.0, 0.0)):
            t = (z_layer - self.C[2]) / d[..., 2]
            X = self.C[0] + t * d[..., 0]
            Y = self.C[1] + t * d[..., 1]
            a = math.radians(ang)
            q = (-X * math.sin(a) + Y * math.cos(a)) / p["slat_pitch"] + shift
            s = (X * math.cos(a) + Y * math.sin(a))          # along-slat coordinate
            fq = q % 1.0
            wfrac = p["slat_width"] / p["slat_pitch"]
            hit = fq < wfrac
            slat_idx = np.floor(q).astype(np.int64)
            gi = ((s * 260.0 + slat_idx * 397).astype(np.int64)) % len(grain)
            g = 0.82 + 0.25 * grain[gi]
            edge = np.minimum(fq, wfrac - fq) / wfrac        # 0 at the slat edges
            bevel = np.clip(edge * 9.0, 0.55, 1.0)
            col = wood[None, None, :] * (g * bevel)[..., None]
            col = col * (self.AMBIENT + (1 - self.AMBIENT) * max(0.0, -self.LIGHT[2]))
            tt = np.where(hit, t, np.inf).astype(np.float32)
            self.commit(box, tt, col, ID_LATTICE)

    def add_sphere(self, c, r, albedo, ks, shin, oid, sphere_seed=0):
        """Shaded sphere with a slight mottle fixed to the surface (view independent)."""
        corners = c + r * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        box = self.bbox(corners, 2)
        if box is None:
            return
        d = self.rays(*box)
        oc = self.C - c
        a = np.sum(d * d, -1)
        b = 2 * (d @ oc)
        cc = oc @ oc - r * r
        disc = b * b - 4 * a * cc
        hit = disc > 0
        sq = np.sqrt(np.where(hit, disc, 0))
        t = (-b - sq) / (2 * a)
        hit &= t > 1e-4
        if not np.any(hit):
            return
        P = self.C + t[..., None] * d
        n = (P - c) / r
        mott = 1.0 + 0.05 * np.sin(11 * n[..., 0] + sphere_seed) * np.cos(9 * n[..., 1] + 2 * sphere_seed)
        col = self.shade(n, d, albedo[None, None, :] * mott[..., None], ks, shin)
        self.commit(box, np.where(hit, t, np.inf).astype(np.float32), col, oid)

    def add_tube(self, p0, p1, r, albedo, oid, fuzz=0.0, rng=None):
        """Thin tube (vine or stem) as a capsule: closest approach of ray and segment."""
        box = self.bbox(np.stack([p0, p1]), self.K[0, 0] * r / 0.3 + 3)
        if box is None:
            return
        d = self.rays(*box)
        v = p1 - p0
        w0 = self.C - p0
        a = np.sum(d * d, -1)
        bb = d @ v
        c = v @ v
        dd = d @ w0
        e = v @ w0
        den = a * c - bb * bb
        sc = np.where(den > 1e-12, (bb * e - c * dd) / np.maximum(den, 1e-12), 0)
        tc = np.where(den > 1e-12, (a * e - bb * dd) / np.maximum(den, 1e-12), 0)
        tc = np.clip(tc, 0, 1)
        sc = (bb * tc - dd) / a
        Pr = self.C + sc[..., None] * d
        Q = p0 + tc[..., None] * v
        dist = np.linalg.norm(Pr - Q, axis=-1)
        rad = r
        if fuzz > 0 and rng is not None:
            rad = r * (1.0 + fuzz * rng.standard_normal(dist.shape))
        hit = (dist < rad) & (sc > 1e-4)
        if not np.any(hit):
            return
        back = np.sqrt(np.clip(r * r - dist * dist, 0, None)) / np.sqrt(a)
        t = sc - back
        P = self.C + t[..., None] * d
        n = P - Q
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
        col = self.shade(n, d, albedo[None, None, :] * (0.9 + 0.2 * (rng.random(dist.shape)[..., None] if rng is not None else 0.5)))
        self.commit(box, np.where(hit, t, np.inf).astype(np.float32), col, oid)

    def add_stake(self, xs, zs, r, y0, y1, oid):
        """Vertical bamboo stake (cylinder along world Y)."""
        pts = np.array([[xs - r, y0, zs - r], [xs + r, y0, zs - r],
                        [xs - r, y1, zs - r], [xs + r, y1, zs - r]])
        box = self.bbox(pts, 3)
        if box is None:
            return
        d = self.rays(*box)
        ox, oz = self.C[0] - xs, self.C[2] - zs
        a = d[..., 0] ** 2 + d[..., 2] ** 2
        b = 2 * (d[..., 0] * ox + d[..., 2] * oz)
        c = ox * ox + oz * oz - r * r
        disc = b * b - 4 * a * c
        hit = disc > 0
        t = (-b - np.sqrt(np.where(hit, disc, 0))) / (2 * a)
        Y = self.C[1] + t * d[..., 1]
        hit &= (Y > y0) & (Y < y1) & (t > 1e-4)
        if not np.any(hit):
            return
        P = self.C + t[..., None] * d
        n = np.stack([(P[..., 0] - xs) / r, np.zeros_like(t), (P[..., 2] - zs) / r], -1)
        node = (np.abs(((Y + 0.03) % 0.15) - 0.075) < 0.004)
        alb = np.array([0.74, 0.64, 0.40])[None, None, :] * np.where(node, 0.7, 1.0)[..., None]
        col = self.shade(n, d, alb)
        self.commit(box, np.where(hit, t, np.inf).astype(np.float32), col, oid)

    def add_leaf(self, lf, oid):
        """Wound pipe-cleaner leaf: planar pointed ellipse with loop texture and fuzzy rim."""
        c, e1, e2, A, B = lf["c"], lf["e1"], lf["e2"], lf["A"], lf["B"]
        corners = np.array([c + sa * A * e1 + sb * B * e2 for sa in (-1, 1) for sb in (-1, 1)])
        box = self.bbox(corners, 3)
        if box is None:
            return
        d = self.rays(*box)
        n = np.cross(e1, e2)
        den = d @ n
        t = ((c - self.C) @ n) / np.where(np.abs(den) > 1e-9, den, 1e-9)
        P = self.C + t[..., None] * d
        a = (P - c) @ e1
        b = (P - c) @ e2
        prof = B * np.clip(1 - (a / A) ** 2, 0, 1) ** 0.75
        rng = np.random.default_rng(lf["seed"] + int(box[0]) * 7 + int(box[2]) * 13)
        fuzz = 1.0 + 0.10 * rng.standard_normal(a.shape)
        hit = (np.abs(a) < A) & (np.abs(b) < prof * fuzz) & (t > 1e-4)
        if not np.any(hit):
            return
        # concentric winding loops of the pipe cleaner
        rho = np.sqrt((a / A) ** 2 + (b / np.maximum(B, 1e-6)) ** 2)
        loops = 0.82 + 0.18 * np.sin(rho * 2 * math.pi * A / 0.004)
        alb = lf["colour"][None, None, :] * loops[..., None] * (0.92 + 0.16 * rng.random(a.shape)[..., None])
        nn = np.broadcast_to(n, d.shape)
        col = self.shade(nn, d, alb, two_sided=True)
        self.commit(box, np.where(hit, t, np.inf).astype(np.float32), col, oid)

    # -- output -----------------------------------------------------------
    def finish(self, psf_sigma, noise_sigma, rng):
        """Downsample, blur, add noise; return (uint8 BGR, depth, id map)."""
        ss = self.ss
        W, H = self.W // ss, self.H // ss
        rgb = cv2.resize(self.rgb, (W, H), interpolation=cv2.INTER_AREA)
        if psf_sigma > 0:
            rgb = cv2.GaussianBlur(rgb, (0, 0), psf_sigma)
        rgb = rgb + rng.normal(0, noise_sigma / 255.0, rgb.shape).astype(np.float32)
        bgr = np.clip(rgb[..., ::-1] * 255.0 + 0.5, 0, 255).astype(np.uint8)
        # depth / id at output resolution: centre sample of each ss x ss block
        o = ss // 2
        z = self.z[o::ss, o::ss][:H, :W].copy()
        oid = self.oid[o::ss, o::ss][:H, :W].copy()
        return bgr, z, oid


# ---------------------------------------------------------------------------
# Scene assembly
# ---------------------------------------------------------------------------
def render_view(p, K, cam, statics, fruit, stems, foliage, rng):
    """
    Render one view twice: without foliage, then with foliage composited on top.

    Returns dict(img, img_clean, z, z_clean, oid, oid_clean).
    """
    r = Renderer(K, cam["R"], cam["t"], p["width"], p["height"], p["supersample"])
    tex, tex_origin, grain = statics
    _, _, zBack = lattice_planes(p)
    r.add_backdrop(zBack, tex, tex_origin)
    r.add_lattice(p, grain)
    for s0, s1, sr in stems:
        r.add_tube(s0, s1, sr, np.array([0.30, 0.38, 0.14]), ID_STEM)
    for k, fr in enumerate(fruit):
        r.add_sphere(fr["c"], fr["r"], fr["colour"], fr["ks"], fr["shin"], ID_FRUIT0 + k, sphere_seed=k)
    clean = dict(z=r.z.copy(), rgb=r.rgb.copy(), oid=r.oid.copy())
    # foliage on top (same buffers)
    for xs, zs, sr, y0, y1 in foliage["stakes"]:
        r.add_stake(xs, zs, sr, y0, y1, ID_FOLIAGE)
    for p0, p1, tr, col in foliage["tubes"]:
        r.add_tube(p0, p1, tr, col, ID_FOLIAGE, fuzz=0.18, rng=rng)
    for lf in foliage["leaves"]:
        r.add_leaf(lf, ID_FOLIAGE)
    img, z, oid = r.finish(p["psf_sigma"], p["noise_sigma"], rng)
    r.z, r.rgb, r.oid = clean["z"], clean["rgb"], clean["oid"]
    img_c, z_c, oid_c = r.finish(p["psf_sigma"], p["noise_sigma"], rng)
    return dict(img=img, img_clean=img_c, z=z, z_clean=z_c, oid=oid, oid_clean=oid_c)


def foliage_coverage(p, K, cam_ref, foliage, roi):
    """Fraction of the reference-view fruit ROI covered by foliage (foliage-only render)."""
    q = dict(p)
    q["supersample"] = 1
    r = Renderer(K, cam_ref["R"], cam_ref["t"], p["width"] // 2, p["height"] // 2, 1)
    r.K = K.copy()
    r.K[:2] *= 0.5
    for xs, zs, sr, y0, y1 in foliage["stakes"]:
        r.add_stake(xs, zs, sr, y0, y1, ID_FOLIAGE)
    for p0, p1, tr, col in foliage["tubes"]:
        r.add_tube(p0, p1, tr, col, ID_FOLIAGE)
    for lf in foliage["leaves"]:
        r.add_leaf(lf, ID_FOLIAGE)
    x0, y0, x1, y1 = [v // 2 for v in roi]
    return float(np.mean(r.oid[y0:y1, x0:x1] == ID_FOLIAGE))


def fruit_roi(p, K):
    """Reference-view pixel box (x0, y0, x1, y1) that bounds the fruit region of the wall."""
    zA = p["z_fence"]
    X0, X1, Y0, Y1 = -0.34, 0.34, -0.22, 0.25
    u0 = int(K[0, 0] * X0 / zA + K[0, 2])
    u1 = int(K[0, 0] * X1 / zA + K[0, 2])
    v0 = int(K[1, 1] * Y0 / zA + K[1, 2])
    v1 = int(K[1, 1] * Y1 / zA + K[1, 2])
    return max(0, u0), max(0, v0), min(p["width"], u1), min(p["height"], v1)


def make_scene(out_dir, **overrides):
    """
    Generate and save one synthetic scene.

    Writes to out_dir:
      images/view_XX.png          input photos (with foliage)
      images_clean/view_XX.png    the same views without foliage
      gt/depth_XX.npy, gt/depth_clean_XX.npy  float16 depth [m]
      gt/oid_XX.png, gt/oid_clean_XX.png      object-id maps (uint16)
      cameras.json                K, R, t per view, reference index
      scene.json                  parameters, fruit list, per-fruit visibility
    Returns the scene dict.
    """
    p = dict(DEFAULTS)
    p.update(overrides)
    rng = np.random.default_rng(p["seed"])
    K, cams, ref = make_rail_cameras(p, rng)
    fruit, stems = sample_fruit(p, rng)
    roi = fruit_roi(p, K)

    # Foliage density tuned by bisection so the centre view's ROI coverage hits the target
    fol_seed = int(rng.integers(1 << 30))
    lo, hi = 0.05, 8.0
    foliage, cov = None, 0.0
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        foliage = sample_foliage(p, np.random.default_rng(fol_seed), mid)
        cov = foliage_coverage(p, K, cams[ref], foliage, roi)
        if abs(cov - p["coverage"]) < 0.015:
            break
        if cov < p["coverage"]:
            lo = mid
        else:
            hi = mid

    tex, tex_origin = canopy_texture(np.random.default_rng(p["seed"] + 101))
    grain = value_noise_1d(np.random.default_rng(p["seed"] + 202))
    statics = (tex, tex_origin, grain)

    for sub in ("images", "images_clean", "gt"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    n = len(cams)
    vis = np.zeros((len(fruit), n), np.float32)        # visible fraction with foliage
    vis_clean = np.zeros((len(fruit), n), np.float32)  # visible pixels without foliage
    for i, cam in enumerate(cams):
        out = render_view(p, K, cam, statics, fruit, stems, foliage, rng)
        cv2.imwrite(os.path.join(out_dir, "images", f"view_{i:02d}.png"), out["img"])
        cv2.imwrite(os.path.join(out_dir, "images_clean", f"view_{i:02d}.png"), out["img_clean"])
        np.save(os.path.join(out_dir, "gt", f"depth_{i:02d}.npy"), np.minimum(out["z"], 60).astype(np.float16))
        np.save(os.path.join(out_dir, "gt", f"depth_clean_{i:02d}.npy"), np.minimum(out["z_clean"], 60).astype(np.float16))
        cv2.imwrite(os.path.join(out_dir, "gt", f"oid_{i:02d}.png"), out["oid"].astype(np.uint16))
        cv2.imwrite(os.path.join(out_dir, "gt", f"oid_clean_{i:02d}.png"), out["oid_clean"].astype(np.uint16))
        ids_f, cnt_f = np.unique(out["oid"], return_counts=True)
        ids_c, cnt_c = np.unique(out["oid_clean"], return_counts=True)
        cf = dict(zip(ids_f.tolist(), cnt_f.tolist()))
        cc = dict(zip(ids_c.tolist(), cnt_c.tolist()))
        for k in range(len(fruit)):
            nc = cc.get(ID_FRUIT0 + k, 0)
            vis_clean[k, i] = nc
            vis[k, i] = cf.get(ID_FRUIT0 + k, 0) / nc if nc > 0 else 0.0

    # Expected projected area of each fruit in the reference view (unoccluded disc)
    fruit_out = []
    for k, fr in enumerate(fruit):
        c = fr["c"]
        u = K[0, 0] * c[0] / c[2] + K[0, 2]
        v = K[1, 1] * c[1] / c[2] + K[1, 2]
        r_px = K[0, 0] * fr["r"] / c[2]
        disc_area = math.pi * r_px ** 2
        fruit_out.append(dict(
            id=k, cluster=int(fr["cluster"]), centre=c.tolist(), radius=float(fr["r"]),
            uv_ref=[float(u), float(v)], r_px_ref=float(r_px),
            visible_frac_ref_clean=float(vis_clean[k, ref] / max(disc_area, 1e-6)),
            visible_frac_with_foliage=vis[k].tolist(),
            n_views_visible_50=int(np.sum(vis[k] >= 0.5)),
            n_views_visible_any=int(np.sum(vis[k] > 0.05)),
            colour=np.asarray(fr["colour"]).tolist()))

    cams_out = dict(K=K.tolist(), ref=ref, width=p["width"], height=p["height"],
                    views=[dict(name=f"view_{i:02d}.png", R=c["R"].tolist(), t=c["t"].tolist(),
                                C=c["C"].tolist()) for i, c in enumerate(cams)])
    zA, zB, zBack = lattice_planes(p)
    scene = dict(params=p, roi_ref=list(roi), coverage_ref=cov,
                 depths=dict(foliage=p["z_fence"] - p["foliage_gap"], fence_front=zA,
                             fence_back=zB + p["slat_thick"], backdrop=zBack),
                 fruit=fruit_out)
    with open(os.path.join(out_dir, "cameras.json"), "w") as f:
        json.dump(cams_out, f, indent=1)
    with open(os.path.join(out_dir, "scene.json"), "w") as f:
        json.dump(scene, f, indent=1)
    return scene
