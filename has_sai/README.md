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

## Results so far (simulation only)

The numbers below are from the synthetic benchmark.  One real capture has since been
processed, qualitatively only: see [results/real/README.md](results/real/README.md) (the
leaf-free render works, the count does not without real-photo calibration).  Benchmark:
44 ray-cast scenes, test seeds 2 and 3 never looked at during development,
detector frozen at tag `has-sai-freeze-1` before they were read.  Full tables:
`results/synthetic/summary.md`, `factorial.md`, `realpath_study.csv`.

**Leaf-free image (PSNR inside the fruit region, against the foliage-free photo)**

| | cherry | grape | lime |
|---|---|---|---|
| centre photo | 19.9 dB | 19.8 dB | 18.1 dB |
| prototype leaf-masked mean SAI | 17.9 | 15.2 | 15.4 |
| median SAI | 25.0 | 23.6 | 22.3 |
| occlusion-aware SAI (this branch) | **31.0** | **32.0** | **31.7** |

**Counting F1 on the test scenes**

| | cherry | grape | lime |
|---|---|---|---|
| prototype pipeline (masked mean SAI + colour detector) | 0.55 | 0.00 | 0.00 |
| centre photo + prototype colour detector | 0.70 | 0.00 | 0.00 |
| prototype SAI + colour mask split by distance transform | **0.997** | 0.00 | 0.00 |
| median SAI + colour-free 2-D detector | 0.58 | 0.79 | 0.74 |
| ours: occlusion-aware SAI + colour-free 3-D relief detector | 0.81 | 0.40 | 0.88 |
| ours + random-forest verifier | 0.87 | 0.55 | 0.90 |
| ours leaf-free image + colour-free 2-D detector | 0.62 | **0.84** | **0.91** |
| foliage-free photo + colour-free 2-D detector (reference) | 0.75 | 0.87 | 0.93 |

What this says, honestly:

* The occlusion-aware render is the clear win: 6 to 9 dB above median synthetic
  aperture on every fruit, and every colour-free detector scores higher on it
  than on the photo or the median image.
* For red cherries in simulation a simple colour mask split by its distance
  transform, on any synthetic-aperture image, is close to perfect.  The simulator
  has one clean cherry red and one leaf green, so it cannot show where colour
  fails (dark-purple cherries, shadows, reddish stems, tan stakes); only real
  photos can.
* For green fruit, where colour gets nothing, the leaf-free image with the
  colour-free 2-D detector is the best counter (grapes 0.84, limes 0.91).
* The 3-D relief detector works for separate round fruit at up to 30 % cover
  (cherries 0.99 at 10 %, 0.88 at 30 %) but drops to 0.54 at 50 % cover, fails on
  packed grape bunches, and needs accurate poses: 0.1 degree of pose error drops
  it to 0.12, and it needs at least 9 photos.
* Through `run_real.py`'s exact path (rail SfM poses, frozen threshold) on new
  scenes with rendered carriage wobble, drift and exposure jitter, it counted
  cherries at F1 0.95 to 1.00 (10 and 30 % cover), and the foliage-free control
  counted within one cherry.

A five-seat review (`docs/council_verdict.md`) concluded: keep the geometric
de-occlusion, test on real captures first, and claim only what survives.

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
