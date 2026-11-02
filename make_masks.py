"""SAM 3 masks for one image folder: COLMAP masks (foliage ignored, grapes kept) and removal masks."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

FOLIAGE, FOLIAGE_THR = ["green leaf", "green vine", "green stem"], 0.35
GRAPE, GRAPE_THR = ["grape"], 0.2


def segment(processor, img, prompts, thr):
    # union of all SAM 3 instances over all prompts
    processor.confidence_threshold = thr
    union = np.zeros(img.size[::-1], bool)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(img)
        for prompt in prompts:
            masks = processor.set_text_prompt(state=state, prompt=prompt)["masks"].detach()
            for m in masks.reshape(-1, *masks.shape[-2:]):  # one instance at a time to bound GPU memory
                union |= (m > 0.5).cpu().numpy()
            torch.cuda.empty_cache()
    return union


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input image folder")
    ap.add_argument("--output", required=True, help="output folder (gets masks/ and removal_masks/)")
    ap.add_argument("--overlay", action="store_true", help="also save overlay/ (blue = removed, red = grape)")
    args = ap.parse_args()

    out = Path(args.output)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "removal_masks").mkdir(exist_ok=True)
    if args.overlay:
        (out / "overlay").mkdir(exist_ok=True)
    processor = Sam3Processor(build_sam3_image_model())
    erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

    for p in sorted(Path(args.input).iterdir()):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        img = Image.open(p).convert("RGB")
        foliage = segment(processor, img, FOLIAGE, FOLIAGE_THR).astype(np.uint8)
        grape = segment(processor, img, GRAPE, GRAPE_THR)
        # grapes win over foliage: the vine/stem prompts also fire on grapes
        colmap = (cv2.erode(foliage, erode) > 0) & ~grape  # COLMAP: 0 = ignore, named IMG.png.png
        removal = (cv2.dilate(foliage, dilate) > 0) & ~grape  # depth maps: 255 = drop
        cv2.imwrite(str(out / "masks" / (p.name + ".png")), (~colmap).astype(np.uint8) * 255)
        cv2.imwrite(str(out / "removal_masks" / f"{p.stem}_mask.png"), removal.astype(np.uint8) * 255)
        if args.overlay:
            im = cv2.imread(str(p))
            im[removal] = (0.5 * im[removal] + 0.5 * np.array([255, 128, 0])).astype(np.uint8)  # blue
            im[grape] = (0.4 * im[grape] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)  # red
            cv2.imwrite(str(out / "overlay" / f"{p.stem}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{p.name}: foliage {foliage.mean():.1%}, grape {grape.mean():.1%}", flush=True)


if __name__ == "__main__":
    main()
