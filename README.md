<div align="center">

# AMME5710 2026: Smartphone-Only Occlusion Removal for Counting Grapes Behind Foliage

**One phone sliding past a vine row sees around the leaves. These scripts register the
photos on the fruit, keep only the fruit pixels that several views agree on, and count the
grapes that no single photo shows.**

AMME5710 Computer Vision and Image Processing · The University of Sydney · 2026
Has Salvesen · Finn Panko-Jones · Dinh Quang Huynh

[![Python 3](https://img.shields.io/badge/Python-3-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SAM 3](https://img.shields.io/badge/SAM%203-masks-0467DF?style=for-the-badge)](https://github.com/facebookresearch/sam3)
[![Licence: MIT](https://img.shields.io/badge/Licence-MIT-F7DF1E?style=for-the-badge)](LICENSE)

[Report](report/report.pdf) · [Layout](#repository-layout) · [Rig](#the-rig-and-the-captures) ·
[Setup](#setup) · [Run](#running-the-pipeline) · [Figures](#regenerating-the-tables-and-figures) ·
[Gaps](#what-the-report-shows-that-this-repository-cannot-regenerate) · [Pitfalls](#known-pitfalls)

<table>
  <tr>
    <td align="center" width="50%">
      <b>The rail rig</b><br><br>
      <img src="report/figures/capture_setup.png" height="230" alt="The enclosure with the camera rail, and the phone, rail and lit scene"><br>
      <sub>A phone on a horizontal rail photographs red grapes behind removable foreground branches.</sub>
    </td>
    <td align="center" width="50%">
      <b>Occluded view, then the fruit recovered from 28 views</b><br><br>
      <img src="report/figures/qualitative_grid.jpg" height="230" alt="Rows of 3 to 6 branches; the occluded view and the reconstructions at 20, 40 and 60 mm spacing"><br>
      <sub>Rows: 3 to 6 branches. Columns: the occluded reference view, then reconstructions at 20, 40 and 60 mm spacing. Black means no view saw fruit there.</sub>
    </td>
  </tr>
</table>

</div>

## Overview

Leaves hide fruit from any single camera position, so a camera-only crop estimate undercounts.
A vehicle driving along a straight vine row cannot move its camera to look around each leaf,
but it does supply a dense sweep of lateral viewpoints as it goes, and the near foliage shifts
more between those views than the fruit behind it. The pipeline uses that parallax:

1. **Segment** every photo with SAM 3 text prompts into fruit and foliage masks.
2. **Register** the views with SIFT features found only on the fruit, a pairwise RANSAC
   similarity with a quarter of the matches held out, and one joint pose solve.
3. **Fuse** by consensus: a pixel is fruit when at least three views spanning at least 40 mm
   agree on its colour, and its colour is copied from one of those views. Nothing unseen is
   synthesised; empty means unobserved.
4. **Count** the result with GrapeSAM, a berry point model trained on the ViViD-5K vineyard
   dataset and left untuned, against 25 hand-labelled grapes.

Indoors, on one arrangement of red grapes with 3 to 6 foreground branches and 28 photos at
20 mm steps, fruit pixel recall rose from 42 % to 87 % and count F1 from 0.42 to 0.77 at five
branches. Recovery fell as views got sparser and occlusion heavier. The
[report](report/report.pdf) has the method, the baselines, the ablations and the limits; this
page is the build and run guide.

| Stage | Folder | Status |
| --- | --- | --- |
| Capture and HEIC conversion | [App/Capturing](App/Capturing) | ![Complete](https://img.shields.io/badge/complete-2EA44F?style=flat-square) |
| SAM 3 fruit and foliage masks | [App/Processing/I-Segmentation](App/Processing/I-Segmentation) | ![Complete](https://img.shields.io/badge/complete-2EA44F?style=flat-square) |
| Fruit-only registration and consensus fusion (the method) | [App/Processing/II-Restoration](App/Processing/II-Restoration) | ![Complete](https://img.shields.io/badge/complete-2EA44F?style=flat-square) |
| GrapeSAM counting, metrics, ablations and figures | [App/Processing/III-Evaluation](App/Processing/III-Evaluation) | ![Being pushed](https://img.shields.io/badge/being%20pushed-D29922?style=flat-square) |
| Depth-based baselines and earlier lines | [App/Processing/IV-Depth_Baselines](App/Processing/IV-Depth_Baselines), [has_sai](has_sai) | ![Explored](https://img.shields.io/badge/explored-6E7781?style=flat-square) |

## Repository layout

| Folder | Contents |
| --- | --- |
| [App/Capturing](App/Capturing) | The rig and capture procedure, and `convert_heic.py`, which turns the phone's HEIC files into PNGs. |
| [App/Processing/I-Segmentation](App/Processing/I-Segmentation) | `make_masks.py`: SAM 3 masks for fruit (`grape`) and foliage (`green leaf`, `green vine`, `green stem`). |
| [App/Processing/II-Restoration](App/Processing/II-Restoration) | `grape_restore_autoref.py`, the method; `grape_reconstruct.py`, the depth-sweep prototype and the image loader; `evaluate_reconstruction.py`; two superseded tools. [Its README](App/Processing/II-Restoration/README.md) lists every input, output and flag. |
| [App/Processing/III-Evaluation](App/Processing/III-Evaluation) | The GrapeSAM counting, pixel metrics, bootstrap, reference and registration sweeps, baselines, ablations and the figure scripts behind Tables 1 to 3 and Figures 4 to 6. See its README for what is there. |
| [App/Processing/IV-Depth_Baselines](App/Processing/IV-Depth_Baselines) | The October cherry prototype (shift-and-add synthetic aperture) and the COLMAP background fill. Both are depth-based and were superseded by the fruit-feature registration. |
| [has_sai](has_sai) | A colour-free, occlusion-aware synthetic aperture benchmarked in simulation only (44 ray-cast scenes). A parallel line of work, not the reported method; [its README](has_sai/README.md) says what it showed. |
| [Captures](Captures) | What the four captures are and where the photographs live. The photos themselves are not in git. |
| [Out](Out) | What the scripts write, one subfolder per stage. Git keeps the small result files and the figures the README shows. |
| [report](report) | The LaTeX source and PDF of the final report. |

## The rig and the captures

| Part | Use |
| --- | --- |
| Smartphone (iPhone, main camera, held in portrait: 3024 x 4032) | Takes every photo; focus, exposure and white balance locked. |
| Horizontal rail with a phone holder | Stands in for the vehicle. The carriage stops every 20 mm. |
| Cardboard enclosure with a viewing slot | Holds the scene and blocks stray light. |
| Red grapes on branches | The fruit layer, about 1.1 m from the camera. |
| Removable foreground branches with leaves, 3 to 6 of them | The occluders, about 0.75 m from the camera. |
| Lamp | Fixed during a sequence; the consensus test compares raw RGB between views. |

Four captures were used for the report, named by step, fruit distance and branch count:

| Capture | Photos | Spacing | What it tests |
| --- | --- | --- | --- |
| `2cm_74cm_3branches` | 28 | 20 mm | light occlusion |
| `2cm_74cm_4branches` | 28 | 20 mm | |
| `2cm_74cm_5branches` | 28 | 20 mm | the ablation condition |
| `2cm_74cm_6branches` | 27 | 20 mm | heavy occlusion |

The 40 and 60 mm conditions keep every second or third photo of the same captures around the
same reference view. A leaf-free photograph of the same fruit, taken after the branches were
removed, is the truth for the pixel metrics and carries the 25 grape labels.
[Captures/README.md](Captures/README.md) gives the folder layout the scripts expect and where
the photographs are hosted.

## Setup

Python 3.9 or newer (the group used 3.13). The restoration, baselines and the synthetic
benchmark need only the packages in `requirements.txt`:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

SAM 3 masks and GrapeSAM counting need PyTorch with a CUDA GPU (the group used an RTX 2000
Ada laptop GPU), the [SAM 3 repository](https://github.com/facebookresearch/sam3) and its
checkpoint on the Python path, and the GrapeSAM weights released with
[ViViD-5K](https://arxiv.org/abs/2605.24353). The COLMAP baseline needs the
[COLMAP](https://colmap.github.io/) binary with CUDA.

## Running the pipeline

Each block starts from the repository root. Put a capture at `data/<capture>/images/`
(`data/` is git-ignored).

```bash
# 1. masks (GPU): writes data/<capture>/grape_masks, removal_masks, masks and overlay
.venv/bin/python App/Processing/I-Segmentation/make_masks.py \
    --input data/2cm_74cm_5branches/images --output data/2cm_74cm_5branches --overlay --grape-masks

# 2. the method: the fruit-only image, its masks, provenance maps and diagnostics
.venv/bin/python App/Processing/II-Restoration/grape_restore_autoref.py \
    --images data/2cm_74cm_5branches/images --target-masks data/2cm_74cm_5branches/grape_masks \
    --scale 0.8 --spacing 20 --out Out/Restoration/2cm_74cm_5branches

# 3. counting and metrics: see App/Processing/III-Evaluation/README.md
```

Without `--target-masks` the restoration falls back to a red/purple colour rule, which is the
"colour rule for fruit" ablation of the report (Table 3), not the method.

## Regenerating the tables and figures

`./reproduce.sh` builds one virtual environment and runs every stage whose inputs are present,
printing `SKIPPED` with the reason for the rest, so the boundary between what this repository
regenerates and what it does not is visible:

```bash
./reproduce.sh                 # everything that can run on this machine
./reproduce.sh --synthetic     # only has_sai's 44-scene synthetic benchmark (CPU, about 20 min)
```

| Result in the report | Script | Needs |
| --- | --- | --- |
| Fig. 3, the depth-shift prototype | `App/Processing/II-Restoration/grape_reconstruct.py` | one capture |
| Fig. 4 reconstructions, the view counts of Sec. 3.3, `observed_mask.png` behind Table 1's recall and precision | `App/Processing/II-Restoration/grape_restore_autoref.py` | captures and SAM 3 masks |
| GrapeSAM counts, Tables 1 to 3, Figs. 5 and 6, the bootstrap and sensitivity numbers of Sec. 5.5 | `App/Processing/III-Evaluation/` | the restoration outputs, the leaf-free truth, the labels and the GrapeSAM weights |
| `has_sai/results/synthetic/` (cited in Sec. 3.5 as the simulation-only variant) | `has_sai/run_synthetic_benchmark.py` | nothing: the scenes are rendered |

## What the report shows that this repository cannot regenerate

- **The photographs and masks are not in git.** They are hosted on the university OneDrive
  ([Captures/README.md](Captures/README.md)); every table can be checked against the result
  files in `Out/` once they are downloaded.
- **The SAM 3 and GrapeSAM weights** are third-party releases and are not redistributed.
- **The presentation's red and green separation** (Sec. 3.2 of the report) was a qualitative
  demonstration; its images are in `report/figures/ripe_*.png`, and no script in this repository
  reproduces them.
- **The October cherry prototype's captures** are not included, so `IV-Depth_Baselines/Cherry_SAI`
  documents the design history but cannot be re-run here.

## Known pitfalls

- `grape_restore_autoref.py` defaults to `--scale 0.4`; the report used 0.8. The pixel
  thresholds (RANSAC 1.5 px, per-view gate 3 px, speck filter 25 px) are in processing pixels.
- The minimum-baseline rule multiplies photo index differences by `--spacing`. A 40 or 60 mm
  run needs a folder holding only every second or third photo and `--spacing 40` or `60`.
- Fewer than three views surviving alignment ends the run with a message; the six-branch
  capture at 60 mm does this. `matching_diagnostics.json` and the contact sheet show why.
- Lock the phone's exposure. Agreement is tested in raw RGB; an exposure step between photos
  rejects true fruit.
- `grape_reconstruct.py`'s default `--images` is a Google Drive path from development; pass
  `--images` explicitly.
- The Windows instructions in `IV-Depth_Baselines/COLMAP/README_Dinh_original.md` use a
  hard-coded COLMAP path; the folder README gives the same commands for macOS and Linux.

## Third-party code

| Code | Use | Licence |
| --- | --- | --- |
| [SAM 3](https://github.com/facebookresearch/sam3) (Carion et al., 2025) | fruit and foliage masks | SAM License |
| GrapeSAM / [ViViD-5K](https://arxiv.org/abs/2605.24353) (Tong et al., 2026) | berry counting | as released by its authors |
| [OpenCV](https://github.com/opencv/opencv), [NumPy](https://github.com/numpy/numpy), [Pillow](https://github.com/python-pillow/Pillow), [PyTorch](https://github.com/pytorch/pytorch) | everything else | Apache 2.0, BSD, HPND, BSD |
| [COLMAP](https://colmap.github.io/) | the depth baseline only | BSD |

The group's own scripts are under the [MIT licence](LICENSE).
