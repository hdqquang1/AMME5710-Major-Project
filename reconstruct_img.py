import argparse
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

# COLMAP camera model id -> (name, number of params)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}


def read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_cameras_bin(path):
    cameras = {}
    with open(path, "rb") as f:
        (n,) = read(f, "Q")
        for _ in range(n):
            cam_id, model_id, width, height = read(f, "iiQQ")
            model, n_params = CAMERA_MODELS[model_id]
            params = np.array(read(f, "d" * n_params))
            cameras[cam_id] = dict(model=model, width=width, height=height, params=params)
    return cameras


def read_images_bin(path):
    images = {}
    with open(path, "rb") as f:
        (n,) = read(f, "Q")
        for _ in range(n):
            image_id, = read(f, "i")
            qvec = np.array(read(f, "dddd"))  # qw, qx, qy, qz
            tvec = np.array(read(f, "ddd"))
            cam_id, = read(f, "i")
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            (n_pts,) = read(f, "Q")
            f.seek(24 * n_pts, 1)  # skip 2D points (x, y, point3D_id)
            images[image_id] = dict(name=name.decode(), qvec=qvec, tvec=tvec, camera_id=cam_id)
    return images


def qvec_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def intrinsic_matrix(cam):
    p = cam["params"]
    if cam["model"] in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
                        "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "FOV"):
        fx = fy = p[0]
        cx, cy = p[1], p[2]
    else:
        fx, fy, cx, cy = p[:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def render_view(points, colors, cam, img, scale=0.25):
    """Project coloured 3D points into a camera with a z-buffer -> HxWx3 image."""
    R = qvec_to_rotmat(img["qvec"])
    K = intrinsic_matrix(cam)
    K[:2] *= scale  # render at reduced resolution so the sparse points fill the image
    w, h = int(cam["width"] * scale), int(cam["height"] * scale)

    pc = points @ R.T + img["tvec"]  # world -> camera
    front = pc[:, 2] > 1e-6
    pc, col = pc[front], colors[front]
    uv = pc @ K.T
    u = np.round(uv[:, 0] / uv[:, 2]).astype(int)
    v = np.round(uv[:, 1] / uv[:, 2]).astype(int)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z, col = u[inside], v[inside], pc[inside, 2], col[inside]

    # Draw far-to-near so the nearest point at each pixel is written last
    order = np.argsort(-z)
    image = np.zeros((h, w, 3))
    image[v[order], u[order]] = col[order]
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="data/Sequence 1", help="COLMAP sequence folder")
    parser.add_argument("--frustum-scale", type=float, default=0.1, help="camera frustum size")
    parser.add_argument("--views", type=int, nargs="+", default=[6, 7],
                        help="1-based camera indices (sorted by image name) to render")
    parser.add_argument("--render-scale", type=float, default=0.25,
                        help="output resolution as a fraction of the original image")
    args = parser.parse_args()

    dense = Path(args.seq) / "dense" / "0"
    # Use the undistorted model in dense/0/sparse so poses share the frame of fused.ply
    sparse = dense / "sparse"

    pcd = o3d.io.read_point_cloud(str(dense / "fused.ply"))
    print(f"Loaded {len(pcd.points)} points (colours: {pcd.has_colors()})")

    # Threshold by z
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    mask = points[:,2] > 25
    points = points[mask]
    colors = colors[mask]
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    cameras = read_cameras_bin(sparse / "cameras.bin")
    images = read_images_bin(sparse / "images.bin")
    print(f"Loaded {len(cameras)} cameras, {len(images)} images")

    # Re-render the thresholded cloud from selected camera poses (1-based, capture order)
    ordered = sorted(images.values(), key=lambda im: im["name"])
    fig, axes = plt.subplots(1, len(args.views), figsize=(6 * len(args.views), 8))
    for ax, idx in zip(np.atleast_1d(axes), args.views):
        img = ordered[idx - 1]
        render = render_view(points, colors, cameras[img["camera_id"]], img, args.render_scale)
        out = dense / f"render_view{idx}_{Path(img['name']).stem}.png"
        plt.imsave(out, render)
        print(f"Saved {out}")
        ax.imshow(render)
        ax.set_title(f"Camera {idx}: {img['name']}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    geometries = [pcd]
    for img in images.values():
        cam = cameras[img["camera_id"]]
        extrinsic = np.eye(4)  # world -> camera
        extrinsic[:3, :3] = qvec_to_rotmat(img["qvec"])
        extrinsic[:3, 3] = img["tvec"]
        frustum = o3d.geometry.LineSet.create_camera_visualization(
            cam["width"], cam["height"], intrinsic_matrix(cam), extrinsic,
            scale=args.frustum_scale)
        frustum.paint_uniform_color([1.0, 0.0, 0.0])
        geometries.append(frustum)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="COLMAP fused cloud + cameras")
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = 0  # Open3D clamps this to its minimum (1.0)
    opt.point_color_option = o3d.visualization.PointColorOption.Default  # keep PLY colours
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
