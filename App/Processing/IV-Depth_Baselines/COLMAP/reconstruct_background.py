"""Middle view of a COLMAP sequence without foliage, rebuilt from the other views' depth maps."""
import argparse
import struct
from pathlib import Path

import cv2
import numpy as np

STRIDE, SPLAT, MEDIAN = 2, 5, 7  # cloud subsampling, splat size (px), speckle filter size
CAMERA_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
                 3: ("RADIAL", 5), 4: ("OPENCV", 8)}


# ---------------------------------------------------------------- COLMAP I/O
def read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_cameras_bin(path):
    cameras = {}
    with open(path, "rb") as f:
        (n,) = read(f, "Q")
        for _ in range(n):
            cam_id, model_id, width, height = read(f, "iiQQ")
            model, n_params = CAMERA_MODELS[model_id]
            cameras[cam_id] = dict(model=model, width=width, height=height,
                                   params=np.array(read(f, "d" * n_params)))
    return cameras


def read_images_bin(path):
    images = []
    with open(path, "rb") as f:
        (n,) = read(f, "Q")
        for _ in range(n):
            read(f, "i")  # image id
            qvec = np.array(read(f, "dddd"))
            tvec = np.array(read(f, "ddd"))
            (cam_id,) = read(f, "i")
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            (n_pts,) = read(f, "Q")
            f.seek(24 * n_pts, 1)
            images.append(dict(name=name.decode(), qvec=qvec, tvec=tvec, camera_id=cam_id))
    return sorted(images, key=lambda i: i["name"])


def read_depth_map(path):
    with open(path, "rb") as f:
        header, amp = b"", 0
        while amp < 3:
            ch = f.read(1)
            header += ch
            amp += ch == b"&"
        w, h, c = map(int, header.decode().strip("&").split("&"))
        data = np.fromfile(f, np.float32)
    return data.reshape((c, h, w) if c > 1 else (h, w))


def qvec_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def intrinsics(cam):
    p = cam["params"]
    if cam["model"] == "PINHOLE":
        return np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]])
    return np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]])



# ------------------------------------------- masked depth -> background cloud
def background_cloud(dense, images, cameras, masks, stride):
    pts, cols = [], []
    for img in images:
        cam = cameras[img["camera_id"]]
        K = intrinsics(cam)
        R, t = qvec_to_rotmat(img["qvec"]), img["tvec"]
        depth = read_depth_map(dense / "stereo" / "depth_maps" / f"{img['name']}.geometric.bin").copy()
        h, w = depth.shape
        # COLMAP may have run stereo at reduced size: bring K, mask and colours to depth resolution
        K = np.diag([w / cam["width"], h / cam["height"], 1]) @ K
        mask = cv2.resize(masks[img["name"]].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        depth[mask] = 0  # foliage -> invalid depth
        rgb = cv2.cvtColor(cv2.imread(str(dense / "images" / img["name"])), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
        d = depth[v, u]
        ok = d > 0
        u, v, d = u[ok], v[ok], d[ok]
        xc = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], 1)
        pts.append(((xc - t) @ R).astype(np.float32))  # camera -> world: R^T (x - t)
        cols.append(rgb[v, u])
    return np.concatenate(pts), np.concatenate(cols)


# ------------------------------------------------------- project to middle view
def project_zbuffer(points, colors, cam, img, splat, chunk=2_000_000):
    """Splat points into the camera, nearest point wins. Returns (image, covered mask)."""
    K, R, t = intrinsics(cam), qvec_to_rotmat(img["qvec"]), img["tvec"]
    R, t = R.astype(np.float32), t.astype(np.float32)
    w, h = cam["width"], cam["height"]
    zbuf = np.full(h * w, np.inf, np.float32)
    owner = np.zeros(h * w, np.int64)
    r = splat // 2
    for start in range(0, len(points), chunk):
        pc = points[start:start + chunk] @ R.T + t
        front = np.flatnonzero(pc[:, 2] > 1e-6)
        pc = pc[front]
        z = pc[:, 2]
        u0 = np.round(pc[:, 0] / z * K[0, 0] + K[0, 2]).astype(np.int64)
        v0 = np.round(pc[:, 1] / z * K[1, 1] + K[1, 2]).astype(np.int64)
        for dv in range(-r, r + 1):
            for du in range(-r, r + 1):
                u, v = u0 + du, v0 + dv
                ok = np.flatnonzero((u >= 0) & (u < w) & (v >= 0) & (v < h))
                idx, zz = v[ok] * w + u[ok], z[ok]
                order = np.lexsort((zz, idx))  # nearest point first within each pixel
                idx, zz, who = idx[order], zz[order], ok[order]
                first = np.r_[True, idx[1:] != idx[:-1]]
                idx, zz, who = idx[first], zz[first], who[first]
                better = zz < zbuf[idx]
                zbuf[idx[better]] = zz[better]
                owner[idx[better]] = start + front[who[better]]
    covered = np.isfinite(zbuf)
    out = np.zeros((h * w, 3), np.uint8)
    out[covered] = colors[owner[covered]]
    return out.reshape(h, w, 3), covered.reshape(h, w)


# ------------------------------------------------------------------ hole filling
def fill_holes(render, holes, scale=0.25, small_px=9):
    """Small holes: inpaint at full res. Large holes: inpaint at reduced res, then upsample."""
    h, w = holes.shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small_px, small_px))
    big = cv2.dilate(cv2.morphologyEx(holes.astype(np.uint8), cv2.MORPH_OPEN, k), k) > 0
    small = holes & ~big
    out = render.copy()
    if small.any():
        out = cv2.inpaint(out, small.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    if big.any():
        size = (int(w * scale), int(h * scale))
        low = cv2.resize(out, size, interpolation=cv2.INTER_AREA)
        m = cv2.resize(big.astype(np.uint8) * 255, size, interpolation=cv2.INTER_NEAREST)
        low = cv2.inpaint(low, m, 7, cv2.INPAINT_TELEA)
        out[big] = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)[big]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True, help="COLMAP dense folder (undistorted images, sparse/, depth maps)")
    ap.add_argument("--masks", required=True, help="removal masks folder from make_masks.py")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--inpaint", action="store_true", help="inpaint pixels the cloud does not cover")
    args = ap.parse_args()

    dense, out = Path(args.dense), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cameras = read_cameras_bin(dense / "sparse" / "cameras.bin")
    images = read_images_bin(dense / "sparse" / "images.bin")
    masks = {i["name"]: cv2.imread(str(Path(args.masks) / f"{Path(i['name']).stem}_mask.png"), 0) > 0 for i in images}
    mid = images[len(images) // 2]  # middle pose

    points, colors = background_cloud(dense, images, cameras, masks, STRIDE)
    render, covered = project_zbuffer(points, colors, cameras[mid["camera_id"]], mid, SPLAT)

    # keep the original image, replace only the removed pixels from the cloud
    region = masks[mid["name"]]
    result = cv2.cvtColor(cv2.imread(str(dense / "images" / mid["name"])), cv2.COLOR_BGR2RGB)
    result[region & covered] = render[region & covered]
    holes = region & ~covered
    print(f"removed {region.mean():.2%} of the middle view, not covered by the cloud: {holes.mean():.2%}")

    if args.inpaint:
        result = fill_holes(result, holes)
        smoothed = cv2.medianBlur(result, MEDIAN)  # de-speckle the replaced pixels only
        result[region] = smoothed[region]
    else:
        result[holes] = 0
    bgr = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out / "background.png"), bgr(result))
    cv2.imwrite(str(out / "hole_mask.png"), holes.astype(np.uint8) * 255)


if __name__ == "__main__":
    main()
