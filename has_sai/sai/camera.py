"""
camera.py - cameras, rail rigs, COLMAP model readers and plane-induced homographies.

Conventions (OpenCV): pixel (u, v) with v down; camera frame X right, Y down,
Z forward.  A view stores the world->camera transform X_cam = R X_world + t,
so its centre is C = -R^T t.  The world frame is the reference camera frame,
so the reference view has R = I, t = 0.

Plane-induced homography
------------------------
For the fronto-parallel plane n^T X = d of the reference frame (n = [0, 0, 1],
d = depth), a reference pixel p maps into view j by

    p_j ~ K_j (R_j + t_j n^T / d) K_ref^-1 p,

because X_j = R_j X + t_j and n^T X / d = 1 on the plane.  This is the
synthetic-aperture refocusing warp; for a camera that only translates along
the rail it reduces to a pure shift of f * x_j / d pixels.

Dependencies: numpy, opencv-python (and the Python standard library).
"""

import glob
import json
import os
import struct

import cv2
import numpy as np


IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


class View:
    """One photo: image (float32 BGR in [0, 1]), intrinsics K, world->camera R, t."""

    def __init__(self, name, image, K, R, t):
        self.name = name
        self.image = image
        self.K = np.asarray(K, np.float64)
        self.R = np.asarray(R, np.float64)
        self.t = np.asarray(t, np.float64).reshape(3)

    @property
    def C(self):
        """Camera centre in world coordinates."""
        return -self.R.T @ self.t

    @property
    def shape(self):
        return self.image.shape[:2]


class Rig:
    """A set of views with one reference view (index `ref`)."""

    def __init__(self, views, ref):
        self.views = views
        self.ref = ref

    def __len__(self):
        return len(self.views)

    def baselines(self):
        """Distance of every camera centre from the reference centre [m]."""
        c0 = self.views[self.ref].C
        return np.array([np.linalg.norm(v.C - c0) for v in self.views])

    def subset(self, indices):
        """New rig with the views in `indices` (the reference must be included)."""
        idx = list(indices)
        if self.ref not in idx:
            raise ValueError("view subset must contain the reference view")
        return Rig([self.views[i] for i in idx], idx.index(self.ref))


# ---------------------------------------------------------------------------
# Scaling and image loading
# ---------------------------------------------------------------------------
def scale_K(K, s):
    """Intrinsics of an image resized by factor s (pixel-centre convention)."""
    K2 = np.array(K, np.float64).copy()
    K2[0, 0] *= s
    K2[1, 1] *= s
    K2[0, 2] = s * (K2[0, 2] + 0.5) - 0.5
    K2[1, 2] = s * (K2[1, 2] + 0.5) - 0.5
    return K2


def read_image(path, scale=1.0):
    """Read an image as float32 BGR in [0, 1], resized by `scale` with area averaging."""
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"cannot read image {path}")
    if scale != 1.0:
        im = cv2.resize(im, (int(round(im.shape[1] * scale)), int(round(im.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA)
    return im.astype(np.float32) / 255.0


def list_images(folder):
    """Sorted image paths in a folder (natural order of the file names)."""
    paths = [p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(IMAGE_EXT)]
    return sorted(paths)


def normalise_exposure(rig, plane_depth, max_gain=1.3):
    """
    Per-photo, per-channel exposure gain, measured on the fence plane.

    Each photo is warped onto the reference view through the plane at
    `plane_depth` (the fence), so the two images show the same surface point at
    most pixels; the gain is the median of the per-pixel ratios I_ref / I_j over
    pixels that are valid in both and not near black.  Foliage, which moves
    between photos, only adds outliers to the median.  (A whole-image median
    ratio is biased by that motion: on synthetic photos with no exposure change
    it produced gains of up to 5 %.)  Use only when exposure was not locked.
    Returns the gains applied.
    """
    ref = rig.views[rig.ref]
    h, w = ref.image.shape[:2]
    gains = []
    for v in rig.views:
        R, t = relative_pose(ref, v)
        H = plane_homography(ref.K, v.K, R, t, plane_depth)
        wj = warp_by_homography(v.image, H, (w, h), border_value=-1.0)
        ok = (wj[..., 0] > -0.5) & (ref.image.min(axis=2) > 0.05) & (wj.min(axis=2) > 0.05)
        if ok.sum() < 1000:
            gains.append([1.0, 1.0, 1.0])
            continue
        ratio = ref.image[ok] / wj[ok]
        g = np.clip(np.median(ratio, axis=0), 1.0 / max_gain, max_gain).astype(np.float32)
        v.image = np.clip(v.image * g[None, None, :], 0.0, 1.0)
        gains.append(g.tolist())
    return gains


# ---------------------------------------------------------------------------
# Synthetic scenes (cameras.json written by synth.py)
# ---------------------------------------------------------------------------
def load_synthetic_rig(scene_dir, scale=1.0, image_folder="images", K_scale_error=1.0,
                       pose_noise=None):
    """
    Load a synthetic scene as a Rig.

    Inputs:  scene_dir; scale (processing scale relative to the rendered size);
             image_folder ("images" or "images_clean"); K_scale_error multiplies
             the focal length handed to the pipeline (calibration error study);
             pose_noise = (rot_deg, trans_m, rng) perturbs the poses handed to the
             pipeline (pose-estimation error study) while the images keep the
             true geometry.
    Output:  Rig.
    """
    with open(os.path.join(scene_dir, "cameras.json")) as f:
        cams = json.load(f)
    K = scale_K(np.array(cams["K"]), scale)
    K[0, 0] *= K_scale_error
    K[1, 1] *= K_scale_error
    views = []
    for i, v in enumerate(cams["views"]):
        img = read_image(os.path.join(scene_dir, image_folder, v["name"]), scale)
        R = np.array(v["R"])
        t = np.array(v["t"])
        if pose_noise is not None and i != cams["ref"]:
            rot_deg, trans_m, rng = pose_noise
            dR = cv2.Rodrigues(np.deg2rad(rng.normal(0, rot_deg, 3)).reshape(3, 1))[0]
            C = -R.T @ t + rng.normal(0, trans_m, 3)
            R = dR @ R
            t = -R @ C
        views.append(View(v["name"], img, K.copy(), R, t))
    return Rig(views, cams["ref"])


# ---------------------------------------------------------------------------
# COLMAP sparse models (text or binary), no pycolmap required
# ---------------------------------------------------------------------------
_COLMAP_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
                  3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
                  6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
                  9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12)}


def _qvec_to_R(q):
    """COLMAP quaternion (qw, qx, qy, qz) to a rotation matrix."""
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def _camera_K_dist(model, params):
    """K and OpenCV distortion coefficients (k1, k2, p1, p2) of a COLMAP camera."""
    p = list(params)
    dist = np.zeros(4)
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
        if model == "SIMPLE_RADIAL":
            dist[0] = p[3]
        elif model == "RADIAL":
            dist[0], dist[1] = p[3], p[4]
    elif model in ("PINHOLE", "OPENCV", "FULL_OPENCV"):
        fx, fy, cx, cy = p[:4]
        if model in ("OPENCV", "FULL_OPENCV"):
            dist[:] = p[4:8]
    else:
        raise ValueError(f"unsupported COLMAP camera model {model}; undistort with COLMAP first")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    return K, dist


def read_colmap_model(model_dir):
    """
    Read cameras and images of a COLMAP sparse model (cameras/images .bin or .txt).

    Returns (cameras, images): cameras[id] = dict(model, width, height, params);
    images[id] = dict(name, R, t, camera_id).
    """
    def rd(f, fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))

    cams, imgs = {}, {}
    cb, ib = os.path.join(model_dir, "cameras.bin"), os.path.join(model_dir, "images.bin")
    ct, it = os.path.join(model_dir, "cameras.txt"), os.path.join(model_dir, "images.txt")
    if os.path.exists(cb) and os.path.exists(ib):
        with open(cb, "rb") as f:
            (n,) = rd(f, "Q")
            for _ in range(n):
                cid, mid, w, h = rd(f, "iiQQ")
                name, npar = _COLMAP_MODELS[mid]
                cams[cid] = dict(model=name, width=w, height=h, params=rd(f, "d" * npar))
        with open(ib, "rb") as f:
            (n,) = rd(f, "Q")
            for _ in range(n):
                (iid,) = rd(f, "i")
                q = rd(f, "dddd")
                t = rd(f, "ddd")
                (cid,) = rd(f, "i")
                name = b""
                while (c := f.read(1)) != b"\x00":
                    name += c
                (npts,) = rd(f, "Q")
                f.seek(24 * npts, 1)
                imgs[iid] = dict(name=name.decode(), R=_qvec_to_R(q), t=np.array(t), camera_id=cid)
    elif os.path.exists(ct) and os.path.exists(it):
        for line in open(ct):
            if line.startswith("#") or not line.strip():
                continue
            e = line.split()
            cams[int(e[0])] = dict(model=e[1], width=int(e[2]), height=int(e[3]),
                                   params=[float(x) for x in e[4:]])
        lines = [l for l in open(it) if not l.startswith("#")]
        for k in range(0, len(lines), 2):
            e = lines[k].split()
            if len(e) < 10:
                continue
            q = [float(x) for x in e[1:5]]
            t = [float(x) for x in e[5:8]]
            imgs[int(e[0])] = dict(name=e[9], R=_qvec_to_R(q), t=np.array(t), camera_id=int(e[8]))
    else:
        raise FileNotFoundError(f"no COLMAP cameras/images files in {model_dir}")
    return cams, imgs


def find_colmap_model(root):
    """Search `root` recursively for a folder holding a COLMAP sparse model; None if absent."""
    for dirpath, _, files in os.walk(root):
        names = set(files)
        if ({"cameras.bin", "images.bin"} <= names) or ({"cameras.txt", "images.txt"} <= names):
            return dirpath
    return None


def rig_from_colmap(image_dir, model_dir, scale=1.0, ref=None, spacing=None):
    """
    Build a Rig from photos and a COLMAP model.

    * Poses are re-expressed in the reference camera frame (the centre photo by
      default), so the reference has R = I, t = 0.
    * COLMAP reconstructions are up to scale.  If `spacing` (rail step in metres)
      is given, the scale is fixed so that the mean distance between consecutive
      camera centres equals `spacing`; depths are then in metres.
    * Lens distortion (if the model has any) is removed with cv2.undistort.
    """
    cams, imgs = read_colmap_model(model_dir)
    entries = sorted(imgs.values(), key=lambda e: e["name"])
    if ref is None:
        ref = len(entries) // 2
    Rr, tr = entries[ref]["R"], entries[ref]["t"]
    # re-express: X_cam_j = R_j X_w + t_j, with X_w = Rr^T (X_ref - tr)
    poses = []
    for e in entries:
        Rj = e["R"] @ Rr.T
        tj = e["t"] - Rj @ tr
        poses.append((Rj, tj))
    centres = np.array([-R.T @ t for R, t in poses])
    s = 1.0
    if spacing is not None and len(centres) > 1:
        steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
        s = spacing / np.mean(steps)
    views = []
    for e, (Rj, tj) in zip(entries, poses):
        cam = cams[e["camera_id"]]
        K, dist = _camera_K_dist(cam["model"], cam["params"])
        path = os.path.join(image_dir, e["name"])
        if not os.path.exists(path):
            path = os.path.join(image_dir, os.path.basename(e["name"]))
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(path)
        if np.any(dist != 0):
            im = cv2.undistort(im, K, dist)
        if scale != 1.0:
            im = cv2.resize(im, (int(round(im.shape[1] * scale)), int(round(im.shape[0] * scale))),
                            interpolation=cv2.INTER_AREA)
        views.append(View(e["name"], im.astype(np.float32) / 255.0, scale_K(K, scale), Rj, tj * s))
    return Rig(views, ref)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def relative_pose(view_ref, view_j):
    """Pose of view j relative to view_ref: X_j = R X_ref + t."""
    R = view_j.R @ view_ref.R.T
    t = view_j.t - R @ view_ref.t
    return R, t


def plane_homography(K_ref, K_j, R, t, depth):
    """Homography taking reference pixels to view-j pixels via the plane Z = depth (reference frame)."""
    n = np.array([0.0, 0.0, 1.0])
    H = K_j @ (R + np.outer(t, n) / depth) @ np.linalg.inv(K_ref)
    return H / H[2, 2]


def warp_by_homography(img, H, size, border_value=-1.0, interp=cv2.INTER_LINEAR):
    """
    Sample `img` at H p for every pixel p of an output of `size` (w, h):
    out(p) = img(H p).  Pixels that fall outside `img` get `border_value`.
    """
    return cv2.warpPerspective(img, H, size, flags=interp | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(border_value,) * 4 if img.ndim == 3 else border_value)


def depth_maps_to_remap(K_ref, K_j, R, t, depth, size):
    """
    Per-pixel reprojection maps for cv2.remap: reference pixel p with depth
    Z(p) maps to x_j = K_j (R Z(p) K_ref^-1 p + t).

    Inputs:  depth (H x W float32, metres), size (w, h) of the reference.
    Output:  (map_x, map_y) float32, ready for cv2.remap(img_j, map_x, map_y, ...).
    """
    w, h = size
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    Ki = np.linalg.inv(K_ref)
    rx = Ki[0, 0] * u + Ki[0, 1] * v + Ki[0, 2]
    ry = Ki[1, 1] * v + Ki[1, 2]
    X = rx * depth
    Y = ry * depth
    Z = depth
    M = (K_j @ R).astype(np.float32)
    m = (K_j @ t).astype(np.float32)
    xh = M[0, 0] * X + M[0, 1] * Y + M[0, 2] * Z + m[0]
    yh = M[1, 0] * X + M[1, 1] * Y + M[1, 2] * Z + m[1]
    zh = M[2, 0] * X + M[2, 1] * Y + M[2, 2] * Z + m[2]
    zh = np.where(np.abs(zh) < 1e-9, 1e-9, zh)
    return (xh / zh).astype(np.float32), (yh / zh).astype(np.float32)


def inverse_depth_samples(z_near, z_far, step_px, f_px, baseline):
    """
    Depth planes uniform in inverse depth, with spacing chosen so that the
    image shift between consecutive planes is `step_px` pixels at the largest
    `baseline`: d(1/Z) = step_px / (f * B).  Returned from near to far.
    """
    inv0, inv1 = 1.0 / z_near, 1.0 / z_far
    d_inv = step_px / (f_px * max(baseline, 1e-6))
    n = max(2, int(np.ceil((inv0 - inv1) / d_inv)) + 1)
    return 1.0 / np.linspace(inv0, inv1, n)


# ---------------------------------------------------------------------------
# EXIF focal length (JPEG only), without extra libraries
# ---------------------------------------------------------------------------
def exif_focal_35mm(path):
    """
    FocalLengthIn35mmFilm (EXIF tag 0xA405) of a JPEG, or None.

    Minimal TIFF/EXIF walk: APP1 'Exif' segment -> IFD0 -> Exif sub-IFD (0x8769).
    """
    try:
        with open(path, "rb") as f:
            data = f.read(256 * 1024)
    except OSError:
        return None
    i = data.find(b"Exif\x00\x00")
    if i < 0:
        return None
    t = data[i + 6:]
    if t[:2] == b"II":
        end = "<"
    elif t[:2] == b"MM":
        end = ">"
    else:
        return None

    def u16(o):
        return struct.unpack(end + "H", t[o:o + 2])[0]

    def u32(o):
        return struct.unpack(end + "I", t[o:o + 4])[0]

    def ifd_entries(off):
        n = u16(off)
        for k in range(n):
            e = off + 2 + 12 * k
            yield u16(e), u16(e + 2), u32(e + 4), e + 8

    try:
        exif_off = None
        for tag, typ, cnt, vo in ifd_entries(u32(4)):
            if tag == 0x8769:
                exif_off = u32(vo)
        if exif_off is None:
            return None
        for tag, typ, cnt, vo in ifd_entries(exif_off):
            if tag == 0xA405:
                return float(u16(vo))
    except (struct.error, IndexError):
        return None
    return None


def K_from_35mm(f35, width, height):
    """
    Pinhole K from a 35 mm-equivalent focal length.  The equivalence is defined on
    the image diagonal (43.27 mm for the 36 x 24 mm frame), so
    f_px = f35 * sqrt(W^2 + H^2) / 43.27.  Principal point at the image centre.
    """
    f = f35 * np.hypot(width, height) / np.hypot(36.0, 24.0)
    return np.array([[f, 0, (width - 1) / 2.0], [0, f, (height - 1) / 2.0], [0, 0, 1]], np.float64)
