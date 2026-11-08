#!/usr/bin/env python3
"""Read a rail sequence: EXIF-oriented images and optional white=exclude foliage masks."""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def natural_key(name):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(name))
    ]


def read_rgb(blob):
    with Image.open(io.BytesIO(blob)) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB")).copy()


def foliage_mask_path(folder, name):
    """Accept <stem>.png or the stored removal_masks naming, <stem>_mask.png."""
    plain = Path(folder) / Path(name).with_suffix(".png")
    suffixed = plain.with_name(f"{plain.stem}_mask.png")
    for path in (plain, suffixed):
        if path.is_file():
            return path
    raise ValueError(
        f"Missing foliage mask: {plain}; supply a black mask for clear views"
    )


def load_sequence(source, scale=0.25, masks=None, mask_dilate=2, calibration=None):
    """Read EXIF-oriented images; white masks exclude occluders of any colour.

    Mask filenames mirror input relative paths, replacing the extension with
    .png. Masks must be reviewed and have the oriented ORIGINAL image size.
    """
    source = Path(source)
    if not 0 < scale <= 1:
        raise ValueError("scale must be in (0, 1]")
    archive = None
    if source.is_dir():
        names = sorted(
            (
                p.relative_to(source).as_posix()
                for p in source.rglob("*")
                if p.is_file() and p.suffix.lower() in EXTENSIONS
            ),
            key=natural_key,
        )
        read = lambda name: (source / name).read_bytes()
    elif source.is_file() and source.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(source)
        names = sorted(
            (
                n
                for n in archive.namelist()
                if Path(n).suffix.lower() in EXTENSIONS
                and not n.startswith("__MACOSX/")
            ),
            key=natural_key,
        )
        read = archive.read
    else:
        raise ValueError(f"Image directory or ZIP not found: {source}")
    if len(names) < 3:
        if archive:
            archive.close()
        raise ValueError("Need at least three views, in capture order")
    if len(names) > 65534:
        if archive:
            archive.close()
        raise ValueError("Too many views for uint16 provenance output")
    images, allowed, size = [], [], None
    try:
        if len(set(names)) != len(names) or any(
            Path(name).is_absolute() or ".." in Path(name).parts for name in names
        ):
            raise ValueError(
                "Input names must be unique relative paths without parent traversal"
            )
        if masks and len({str(Path(n).with_suffix(".png")) for n in names}) != len(
            names
        ):
            raise ValueError("Input image stems collide in the mask folder")
        for name in names:
            rgb = read_rgb(read(name))
            h, w = rgb.shape[:2]
            if size is None:
                size = (w, h)
            if size != (w, h):
                raise ValueError(f"Image sizes differ after EXIF orientation: {name}")
            valid = np.ones((h, w), np.uint8)
            if masks:
                mask_path = foliage_mask_path(masks, name)
                with Image.open(mask_path) as mask:
                    mask = np.asarray(mask.convert("L"))
                if mask.shape != (h, w):
                    raise ValueError(
                        f"Mask must match EXIF-oriented original dimensions: {mask_path}"
                    )
                valid = np.uint8(mask < 128)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if calibration:
                if [w, h] != calibration["image_size"]:
                    raise ValueError(
                        "Calibration image_size must match EXIF-oriented originals"
                    )
                k = np.asarray(calibration["K"], np.float64)
                dist = np.asarray(
                    calibration.get("distortion", [0, 0, 0, 0, 0]), np.float64
                )
                mx, my = cv2.initUndistortRectifyMap(
                    k, dist, None, k, (w, h), cv2.CV_32FC1
                )
                bgr = cv2.remap(bgr, mx, my, cv2.INTER_LINEAR)
                valid = cv2.remap(valid, mx, my, cv2.INTER_NEAREST, borderValue=0)
                valid &= np.uint8((mx >= 0) & (mx < w - 1) & (my >= 0) & (my < h - 1))
            target = (max(1, round(w * scale)), max(1, round(h * scale)))
            images.append(cv2.resize(bgr, target, interpolation=cv2.INTER_AREA))
            # Preserve any occluder intersecting a downsampled pixel.
            valid = (
                cv2.resize(
                    valid.astype(np.float32), target, interpolation=cv2.INTER_AREA
                )
                > 0.999
            )
            if mask_dilate:
                valid = (
                    cv2.erode(
                        valid.astype(np.uint8),
                        np.ones((2 * mask_dilate + 1,) * 2, np.uint8),
                        borderType=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    > 0
                )
            allowed.append(valid)
    finally:
        if archive:
            archive.close()
    return names, images, allowed, size
