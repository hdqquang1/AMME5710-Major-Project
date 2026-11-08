#!/usr/bin/env python3
"""Fruit and foliage masks: per-capture colour thresholds or SAM 3 text prompts.

Colour masks work at processing resolution inside grape_restore_autoref.py.
SAM 3 masks are written at original size, white = selected:
    python grape_segment.py --images data/CAPTURE/images --out masks/CAPTURE
Masks are proposals, not verified grape labels; review them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from grape_io import EXTENSIONS, natural_key, read_rgb

COLOUR_THRESHOLDS = {
    "red": {"lab_a_min": 137, "r_over_g": 1.12, "r_over_b": 1.05, "r_min": 25},
    # Grid-searched on 2cm_80cm_1 and _2 against stored SAM masks, 0.4 scale, OpenCV units.
    "green": {
        "hue_min": 22,
        "hue_max": 34,
        "sat_min": 110,
        "lab_l_min": 50,
        "lab_a_min": 115,
        "lab_b_min": 150,
    },
    "max_hole_area": 49,
}
FOLIAGE_PROMPTS, FOLIAGE_THRESHOLD = ["green leaf", "green vine", "green stem"], 0.35
FRUIT_THRESHOLD = 0.2
FOLIAGE_DILATE = 25


def roi_mask(shape, box):
    if len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ValueError("ROI must contain x0 y0 x1 y1 fractions within 0..1")
    h, w = shape
    result = np.zeros(shape, np.uint8)
    result[
        round(box[1] * h) : round(box[3] * h), round(box[0] * w) : round(box[2] * w)
    ] = 255
    return result


def red_seed(image):
    t = COLOUR_THRESHOLDS["red"]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    b, g, r = cv2.split(image.astype(np.float32))
    return (
        (lab[:, :, 1] >= t["lab_a_min"])
        & (r > g * t["r_over_g"])
        & (r > b * t["r_over_b"])
        & (r > t["r_min"])
    )


def green_seed(image):
    """Pale yellow-green berries. Green fruit overlaps foliage in colour, so expect
    leaf false positives and missed fruit; held-out IoU against SAM masks was about 0.3."""
    t = COLOUR_THRESHOLDS["green"]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    seed = (
        (hsv[..., 0] >= t["hue_min"])
        & (hsv[..., 0] <= t["hue_max"])
        & (hsv[..., 1] >= t["sat_min"])
        & (lab[..., 0] >= t["lab_l_min"])
        & (lab[..., 1] >= t["lab_a_min"])
        & (lab[..., 2] >= t["lab_b_min"])
    )
    # Thin leaf veins and edges pass the colour test; berries survive the opening.
    return cv2.morphologyEx(np.uint8(seed), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0


def colour_mask(image, box, colour="red"):
    """Colour candidates in the ROI, with small enclosed highlights filled in."""
    seeds = {"red": [red_seed], "green": [green_seed], "both": [red_seed, green_seed]}
    if colour not in seeds:
        raise ValueError(f"Unknown colour: {colour}")
    seed = np.zeros(image.shape[:2], bool)
    for rule in seeds[colour]:
        seed |= rule(image)
    mask = np.uint8(seed) * 255
    mask &= roi_mask(mask.shape, box)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # Include small enclosed highlights in candidate masks. No image colours
    # are generated; these are not separate berry-instance labels.
    holes = cv2.bitwise_not(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(holes, 8)
    for k in range(1, count):
        x, y, w, h, area = stats[k]
        if (
            area <= COLOUR_THRESHOLDS["max_hole_area"]
            and x > 0
            and y > 0
            and x + w < mask.shape[1]
            and y + h < mask.shape[0]
        ):
            mask[labels == k] = 255
    mask &= roi_mask(mask.shape, box)
    return mask > 0


def load_grape_masks(folder, names, original_size, shape):
    """Saved fruit masks: white includes grapes, black excludes them."""
    output = []
    if len({str(Path(n).with_suffix(".png")) for n in names}) != len(names):
        raise ValueError("Input names collide in the grape-mask folder")
    for name in names:
        path = Path(folder) / Path(name).with_suffix(".png")
        if not path.is_file():
            raise ValueError(f"Missing grape mask: {path}")
        with Image.open(path) as image:
            if image.size != original_size:
                raise ValueError(
                    f"Grape mask must match oriented original dimensions: {path}"
                )
            mask = np.asarray(image.convert("L")) >= 128
        output.append(
            cv2.resize(
                mask.astype(np.float32),
                (shape[1], shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            > 0.999
        )
    return output


_PROCESSOR = None


def _sam3_processor():
    global _PROCESSOR
    if _PROCESSOR is None:
        # The clone at third_party/sam3 would otherwise shadow its own inner package.
        sys.path.insert(0, str(Path(__file__).resolve().parent / "third_party" / "sam3"))
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        _PROCESSOR = Sam3Processor(build_sam3_image_model())
    return _PROCESSOR


def sam3_instances(rgb, prompt="grape", threshold=0.2):
    """Separate boolean masks, one per SAM 3 instance, for an RGB array."""
    import torch

    processor = _sam3_processor()
    processor.confidence_threshold = threshold
    image = Image.fromarray(rgb)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        masks = processor.set_text_prompt(state=state, prompt=prompt)["masks"].detach()
        output = [(m > 0.5).cpu().numpy() for m in masks.reshape(-1, *masks.shape[-2:])]
    torch.cuda.empty_cache()
    return output


def _sam3_union(processor, image, prompts, threshold):
    """Union of all SAM 3 instances over all prompts."""
    import torch

    processor.confidence_threshold = threshold
    union = np.zeros(image.size[::-1], bool)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        for prompt in prompts:
            masks = processor.set_text_prompt(state=state, prompt=prompt)[
                "masks"
            ].detach()
            # One instance at a time to bound GPU memory.
            for m in masks.reshape(-1, *masks.shape[-2:]):
                union |= (m > 0.5).cpu().numpy()
            torch.cuda.empty_cache()
    return union


def segment_folder_sam3(images, out, fruit_prompt="grape", foliage=True):
    """Write <out>/grape_masks and optionally <out>/foliage_masks; finished views are reused."""
    images, out = Path(images), Path(out)
    if images.is_dir() and out.resolve().is_relative_to(images.resolve()):
        raise ValueError("Keep masks outside the input image folder")
    paths = (
        [images]
        if images.is_file()
        else sorted(
            (
                p
                for p in images.rglob("*")
                if p.is_file() and p.suffix.lower() in EXTENSIONS
            ),
            key=natural_key,
        )
    )
    if not paths:
        raise ValueError(f"No images found: {images}")
    grape_dir, foliage_dir = out / "grape_masks", out / "foliage_masks"
    processor = None
    dilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (FOLIAGE_DILATE, FOLIAGE_DILATE)
    )
    for path in paths:
        name = Path(path.name) if images.is_file() else path.relative_to(images)
        grape_path = (grape_dir / name).with_suffix(".png")
        foliage_path = (foliage_dir / name).with_suffix(".png")
        if grape_path.is_file() and (not foliage or foliage_path.is_file()):
            continue
        if processor is None:
            processor = _sam3_processor()
        image = Image.fromarray(read_rgb(path.read_bytes()))
        grape = _sam3_union(processor, image, [fruit_prompt], FRUIT_THRESHOLD)
        grape_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(grape_path), np.uint8(grape) * 255)
        if foliage:
            leaves = _sam3_union(processor, image, FOLIAGE_PROMPTS, FOLIAGE_THRESHOLD)
            # Grapes win over foliage: the vine/stem prompts also fire on grapes.
            removal = (cv2.dilate(np.uint8(leaves), dilate) > 0) & ~grape
            foliage_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(foliage_path), np.uint8(removal) * 255)
        print(f"{name}: fruit {grape.mean():.2%}", flush=True)
    return grape_dir, (foliage_dir if foliage else None)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", required=True, type=Path, help="Image folder or one image")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--fruit-prompt", default="grape")
    p.add_argument("--grape-only", action="store_true", help="Skip foliage masks")
    args = p.parse_args(argv)
    segment_folder_sam3(args.images, args.out, args.fruit_prompt, not args.grape_only)


if __name__ == "__main__":
    main()
