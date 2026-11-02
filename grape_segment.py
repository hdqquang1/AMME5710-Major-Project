#!/usr/bin/env python3
"""Optional prompt-guided SAM 2 masks for occluders or reconstructed grape bunches.

SAM 2 segments prompted objects; it does not identify grapes, assess health,
classify ripeness, or infer hidden berries. Review every proposed mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from grape_reconstruct import EXTENSIONS, natural_key, read_rgb


def validate_prompt(prompt, width, height):
    if not isinstance(prompt, dict):
        raise ValueError("Each object prompt must be a JSON object")
    points = np.asarray(prompt.get("points", []), np.float32).reshape(-1, 2)
    labels = np.asarray(prompt.get("labels", [1] * len(points)))
    box = np.asarray(prompt["box"], np.float32) if "box" in prompt else None
    if len(points) == 0 and box is None:
        raise ValueError(
            "Each object needs points and/or a box in oriented original pixels"
        )
    if labels.shape != (len(points),) or not np.isin(labels, [0, 1]).all():
        raise ValueError(
            "Point labels must be 0 (background) or 1 (foreground), one per point"
        )
    if (
        not np.isfinite(points).all()
        or np.any(points < 0)
        or (
            len(points)
            and (np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height))
        )
    ):
        raise ValueError("Prompt points are outside the oriented image")
    if box is not None and (
        box.shape != (4,)
        or not np.isfinite(box).all()
        or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)
    ):
        raise ValueError("Box must be [x0,y0,x1,y1] within the oriented image")
    return {
        "point_coords": points if len(points) else None,
        "point_labels": labels.astype(np.int32) if len(points) else None,
        "box": box,
    }


def instance_statistics(rgb, proposal, valid):
    """Only measure observed pixels; report how much of the proposal was excluded."""
    visible = proposal & valid
    area = int(visible.sum())
    stats = {
        "supported_area_px": area,
        "proposal_area_px": int(proposal.sum()),
        "supported_fraction_of_proposal": area / max(int(proposal.sum()), 1),
        "mean_rgb": rgb[visible].mean(axis=0).tolist() if area else None,
        "interpretation": "Visible projected pixels only; no berry count, ripeness or health inference.",
    }
    return visible, stats


def make_predictor(model_id, device):
    try:
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise RuntimeError(
            "Optional SAM 2 is not installed. Follow README.md; core reconstruction does not need it."
        ) from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA requested but unavailable; choose --device cpu or a GPU runtime"
        )
    predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
    return predictor


def predict_objects(predictor, rgb, objects):
    import torch

    outputs = []
    with torch.inference_mode():
        predictor.set_image(rgb)
        for obj in objects:
            kwargs = validate_prompt(obj, rgb.shape[1], rgb.shape[0])
            masks, scores, _ = predictor.predict(**kwargs, multimask_output=True)
            index = int(np.argmax(scores))
            outputs.append((masks[index] > 0, float(scores[index])))
    return outputs


def save_mask(path, mask):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(mask) * 255).save(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    template = sub.add_parser(
        "template", help="Write an occluder-prompt manifest; no model needed"
    )
    template.add_argument("--images", required=True, type=Path)
    template.add_argument("--out", required=True, type=Path)
    occ = sub.add_parser(
        "occluders", help="Segment prompted leaves AND branches per original view"
    )
    occ.add_argument("--images", required=True, type=Path)
    targets = sub.add_parser(
        "targets", help="Segment prompted grape targets in every original view"
    )
    targets.add_argument("--images", required=True, type=Path)
    grape = sub.add_parser(
        "grapes", help="Segment prompted grape bunches in the reconstruction"
    )
    grape.add_argument("--image", required=True, type=Path)
    grape.add_argument(
        "--valid-mask",
        required=True,
        type=Path,
        help="analysis_mask.png or deliberately selected observed_mask.png",
    )
    for command in [occ, targets, grape]:
        command.add_argument("--prompts", required=True, type=Path)
        command.add_argument("--out", required=True, type=Path)
        command.add_argument(
            "--model-id",
            default="facebook/sam2.1-hiera-small",
            help="Explicit public model ID; downloads on first use",
        )
        command.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args(argv)
    if args.mode == "template":
        names = sorted(
            (
                x.relative_to(args.images).as_posix()
                for x in args.images.rglob("*")
                if x.is_file() and x.suffix.lower() in EXTENSIONS
            ),
            key=natural_key,
        )
        if not names:
            p.error("No images found")
        if args.out.exists():
            p.error("Refusing to overwrite an existing prompt manifest")
        args.out.write_text(json.dumps({name: None for name in names}, indent=2) + "\n")
        print(
            "Replace null with a list of object prompts per view; [] explicitly means no occluders."
        )
        return
    prompts = json.loads(args.prompts.read_text())
    # Validate everything before starting any model download.
    if args.mode in ("occluders", "targets"):
        if args.out.resolve().is_relative_to(args.images.resolve()):
            p.error("Keep mask outputs outside the input image folder")
        names = sorted(
            (
                x.relative_to(args.images).as_posix()
                for x in args.images.rglob("*")
                if x.is_file() and x.suffix.lower() in EXTENSIONS
            ),
            key=natural_key,
        )
        if not names or set(names) != set(prompts):
            p.error("Manifest must cover exactly every input image")
        if len({str(Path(n).with_suffix(".png")) for n in names}) != len(names):
            p.error("Input image stems collide in the mask folder")
        paths = [args.images / name for name in names]
        prompt_lists = [prompts[name] for name in names]
    else:
        names = [args.image.name]
        paths = [args.image]
        prompt_lists = [prompts]
    grape_valid = None
    for path, objects in zip(paths, prompt_lists):
        if not isinstance(objects, list):
            p.error(f"{path.name}: prompts must be a list, not null")
        rgb = read_rgb(path.read_bytes())
        if args.mode == "grapes":
            with Image.open(args.valid_mask) as mask:
                grape_valid = np.asarray(mask.convert("L")) >= 128
            if grape_valid.shape != rgb.shape[:2]:
                p.error("Validity mask dimensions differ from reconstruction")
        for obj in objects:
            validate_prompt(obj, rgb.shape[1], rgb.shape[0])
    predictor = make_predictor(args.model_id, args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for name, path, objects in zip(names, paths, prompt_lists):
        rgb = read_rgb(path.read_bytes())
        predicted = predict_objects(predictor, rgb, objects)
        union = np.zeros(rgb.shape[:2], bool)
        preview = rgb.copy()
        valid = np.ones(rgb.shape[:2], bool)
        if args.mode == "grapes":
            valid = grape_valid
        for i, (proposal, score) in enumerate(predicted, 1):
            visible, stats = instance_statistics(rgb, proposal, valid)
            stats.update(
                image=name,
                prompt_index=i,
                model_mask_score=score,
                model_id=args.model_id,
            )
            records.append(stats)
            union |= visible
            if args.mode == "grapes":
                save_mask(args.out / f"instance_{i:03d}.png", visible)
        preview[union] = (
            preview[union].astype(float) * 0.55 + np.array([255, 0, 255]) * 0.45
        ).astype(np.uint8)
        if args.mode in ("occluders", "targets"):
            target = Path(name).with_suffix(".png")
            save_mask(args.out / target, union)
            preview_path = args.out / "_previews" / Path(name).with_suffix(".jpg")
        else:
            save_mask(args.out / "grape_mask.png", union)
            Image.fromarray(np.dstack([rgb, np.uint8(union) * 255])).save(
                args.out / "grapes_rgba.png"
            )
            preview_path = args.out / "preview.jpg"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(preview).save(preview_path)
    (args.out / "segments.json").write_text(
        json.dumps(
            {
                "segments": records,
                "warning": "Prompt-guided segmentation proposals require review. SAM 2 scores are not grape classification probabilities.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved proposals and review overlays to {args.out}; review before analysis.")


if __name__ == "__main__":
    main()
