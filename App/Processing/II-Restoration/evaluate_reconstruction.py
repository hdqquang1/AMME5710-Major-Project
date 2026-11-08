#!/usr/bin/env python3
"""Evaluate only observed pixels against an unobscured, registered reference.

Truth and ROI must be in the SAME camera frame, scale, crop and exposure.
A different viewpoint or an unregistered photograph is not a valid target.
"""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def evaluate(image, truth, roi, observed, reliable, baseline=None):
    if image.shape != truth.shape or any(
        m.shape != image.shape[:2] for m in [roi, observed, reliable]
    ):
        raise ValueError(
            "Truth, reconstruction and masks must have matching registered dimensions"
        )
    if not roi.any():
        raise ValueError("Ground-truth grape ROI is empty")
    supported = roi & observed
    result = {
        "grape_roi_pixels": int(roi.sum()),
        "supported_grape_pixels": int(supported.sum()),
        "supported_grape_fraction": float(supported.sum() / roi.sum()),
        "less_ambiguous_grape_fraction": float((roi & reliable).sum() / roi.sum()),
        "mae_on_supported_grapes": None,
        "psnr_db_on_supported_grapes": None,
    }
    if supported.any():
        error = image[supported].astype(float) - truth[supported].astype(float)
        mse = float(np.mean(error**2))
        result.update(
            mae_on_supported_grapes=float(np.mean(np.abs(error))),
            psnr_db_on_supported_grapes=float(10 * np.log10(255**2 / mse))
            if mse > 0
            else "infinite",
        )
        if baseline is not None:
            if baseline.shape != image.shape:
                raise ValueError("Baseline dimensions differ")
            result["baseline_mae_on_same_supported_grapes"] = float(
                np.abs(
                    baseline[supported].astype(float) - truth[supported].astype(float)
                ).mean()
            )
    result["warning"] = (
        "Coverage and fidelity are separate; missing pixels are never counted as successful recovery."
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result", required=True, type=Path)
    p.add_argument("--truth", required=True, type=Path)
    p.add_argument(
        "--grape-roi",
        required=True,
        type=Path,
        help="White=actual grapes in registered unobscured truth",
    )
    p.add_argument("--baseline", type=Path)
    p.add_argument(
        "--image-name",
        help="Override the result image filename; otherwise detect measured or experimental output",
    )
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    def read(path, mask=False):
        im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if mask else cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError(f"Cannot read {path}")
        return im > 127 if mask else im

    image_name = args.image_name or (
        "grapes_rgb.png"
        if (args.result / "grapes_rgb.png").is_file()
        else "reconstruction.png"
    )
    result = evaluate(
        read(args.result / image_name),
        read(args.truth),
        read(args.grape_roi, True),
        read(args.result / "observed_mask.png", True),
        read(args.result / "analysis_mask.png", True),
        read(args.baseline) if args.baseline else None,
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
