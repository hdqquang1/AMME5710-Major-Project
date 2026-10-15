# has_sai: occlusion-aware synthetic aperture imaging and colour-free fruit counting

Has's branch of the AMME5710 major project.  It removes the foliage in front of
a fruit wall using the parallax of a rail capture, and counts the fruit without
any colour thresholds, so the same code counts red cherries and green grapes.

Each teammate explores a different technique:

* `finns-branch`: shift-and-add synthetic aperture at one depth, HSV/Lab colour
  masks and a red-blob detector;
* `main`: COLMAP dense point cloud, depth threshold and re-rendering;
* this branch: per-view visibility from multi-view depth, a per-pixel-depth
  (all-in-focus) leaf-free image, and fruit found as compact 3-D protrusions.

The method, equations and design decisions are in [docs/method.md](docs/method.md).
Verified literature reviews are in [docs/lit_sai_draft.md](docs/lit_sai_draft.md)
and [docs/lit_fruit_draft.md](docs/lit_fruit_draft.md).

## Pipeline

1. **Poses**: a COLMAP sparse model if one exists, otherwise OpenCV rail
   structure from motion with a no-rotation carriage prior (one linear solve).
2. **Foliage masks**: robust multi-baseline plane sweep of every photo; foliage is
   what lies nearer than an automatic (Otsu) split depth.  No colour.
3. **Background depth**: visibility-weighted photo-consistency sweep within a
   band around the fence, half-aperture selection, semi-global aggregation.
4. **Leaf-free image**: every pixel rendered at its own depth from the photos that
   see it.
5. **Detection and count**: directional morphological relief of the depth map
   (slats, wires and stems vanish, compact fruit stays), scale-adapted LoG plus
   radial symmetry, 3-D non-maximum suppression, optional random-forest verifier.

## Running

Python 3 with `opencv-python`, `numpy` and `matplotlib`.  Scripts resolve paths
from their own location, so they run from any folder.

```bash
python3 has_sai/run_synthetic_benchmark.py
```

renders the synthetic benchmark (44 scenes, about 20 min on 3 cores), runs all
methods and writes tables and figures to `has_sai/results/synthetic/`.
Add `--quick` for a 4-scene smoke test.

```bash
python3 has_sai/run_real.py
```

searches `data/<capture>/` (and `has_sai/data/real/<capture>/`) for rail photos,
uses a COLMAP model inside the capture folder if there is one, and writes the
leaf-free image, depth, masks, detections and `summary.json` to
`has_sai/results/real/<capture>/`.  Useful options: `--images DIR`,
`--clean DIR` (the foliage-free pass, for PSNR/SSIM and a reference count),
`--labels CSV`, `--spacing 0.05`, `--fx-full PX`, `--fruit grape`.

```bash
python3 has_sai/label_points.py path/to/centre_photo.jpg
```

click-to-label tool that writes ground-truth fruit centres for `--labels`.

Photos and generated scenes live in `data/` folders, which git ignores.

## Results

See `results/synthetic/summary.md` once the benchmark has run.

## Layout

```
has_sai/
  run_synthetic_benchmark.py   benchmark entry point
  run_real.py                  real-capture entry point
  label_points.py              ground-truth labelling tool
  sai/
    synth.py       ray-cast synthetic rail captures with ground truth
    camera.py      views, COLMAP reader, EXIF focal length, homographies
    rail_sfm.py    OpenCV rail structure from motion
    planesweep.py  robust plane sweep, colour-free foliage masks
    render.py      visibility-weighted cost volume, all-in-focus render
    sgm.py         semi-global aggregation of the cost volume
    pipeline.py    the de-occlusion stages end to end
    detect.py      colour-free fruit detection, 3-D NMS
    verify.py      random-forest verifier of candidates
    baselines.py   prototype colour pipeline (finns-branch) for comparison
    evaluate.py    counting, image and mask metrics
  docs/            method notes and literature reviews
```
