# Restoration: fruit-only registration and consensus fusion

This folder holds the method the report describes (Sec. 3) and the two earlier versions it
grew out of. Keep the five files together: `grape_restore_autoref.py` and `grape_segment.py`
import `load_sequence` and the image reader from `grape_reconstruct.py`.

| Script | Role |
| --- | --- |
| [grape_restore_autoref.py](grape_restore_autoref.py) | **The method.** SIFT features found only inside the fruit masks, pairwise RANSAC similarity with a quarter of the matches held out, a view graph, one joint IRLS pose solve with the reference fixed, consensus fusion (at least 3 agreeing views spanning at least 40 mm, RMS RGB within 30), one observed source pixel copied per output pixel. The reference view is chosen automatically from the best-connected group. |
| [grape_restore.py](grape_restore.py) | The same pipeline before automatic reference selection and the matching diagnostics were added; `--ref` picks the reference by hand. Superseded. |
| [grape_reconstruct.py](grape_reconstruct.py) | The intermediate prototype of Sec. 3.5: a plane sweep over nine depth hypotheses with per-pixel colour consensus. Also provides `load_sequence` (EXIF-oriented reading, optional occluder masks) to the other scripts. Its default `--images` points at a Google Drive path; always pass `--images`. |
| [grape_segment.py](grape_segment.py) | Optional prompt-guided SAM 2 masks for occluders or bunches. Not used for the report; the report's masks come from SAM 3 ([../I-Segmentation](../I-Segmentation)). |
| [evaluate_reconstruction.py](evaluate_reconstruction.py) | Coverage, MAE and PSNR of the observed pixels against a truth image that is already registered to the same frame. |

## Running the method on one capture

```bash
python3 grape_restore_autoref.py \
  --images ../../../data/2cm_74cm_5branches/images \
  --target-masks ../../../data/2cm_74cm_5branches/grape_masks \
  --scale 0.8 --spacing 20 --out ../../../Out/Restoration/2cm_74cm_5branches
```

- `--images`: the rail photos in capture order (PNG or JPEG; HEIC is converted first by
  [../../Capturing/convert_heic.py](../../Capturing/convert_heic.py)). File names must sort in
  rail order.
- `--target-masks`: one white-on-black PNG per photo, named `<photo stem>.png`, at the
  original photo size, white where SAM 3 found fruit. Without it the script falls back to the
  red/purple colour rule inside `--target-roi`, which is the "colour rule for fruit" ablation
  of Table 3, not the method.
- `--masks`: optional occluder masks of the same shape (white = foliage), used to exclude
  foliage pixels from the fruit evidence (the $E_i = C_i(1-F_i)$ rule of Sec. 3.2).
- `--scale 0.8`: the report processed the photos at 0.8 of full resolution. The script's
  default is 0.4, so pass 0.8 explicitly.
- `--spacing`: the rail step in millimetres between consecutive photos in the folder. The
  baseline rule multiplies index differences by this value, so a 40 or 60 mm run made by
  keeping every second or third photo needs `--spacing 40` or `60` and a folder holding only
  those photos.

Outputs in `--out`:

| File | Contents |
| --- | --- |
| `grapes_rgb.png`, `grapes_rgba.png` | The fruit-only image; alpha = observed |
| `observed_mask.png` | Pixels that met the support and span rules (the mask the report scores) |
| `analysis_mask.png`, `analysis_ready_rgba.png` | The subset whose agreeing views also disagree by at most 18 RMS |
| `support_count.png`, `source_view.png`, `span` in `evidence.npz` | How many views agreed, which view each pixel was copied from, and the baseline span |
| `aligned_mean.png`, `leafmasked_mean.png` | The synthetic aperture means on the same alignment (the "SA mean" and "SA masked mean, our alignment" rows of Table 2; the leaf mask here is an HSV green rule) |
| `comparison.jpg`, `grapes_closeup.png` | Reference, masked mean and result side by side |
| `diagnostics.json`, `matching_diagnostics.json`, `debug_colour_masks_contact_sheet.jpg` | Every parameter, accepted and rejected pairs, per-view held-out errors, the chosen reference and the excluded views |

Runtime for 28 views at scale 0.8 is dominated by the tiled NumPy fusion (about 3.5 min on a
laptop CPU); SIFT and RANSAC take about 10 s.

## Pitfalls

- Fewer than three views surviving validation ends the run with a message rather than an image;
  read `matching_diagnostics.json` and the contact sheet to see which pairs failed.
- The RANSAC threshold (1.5 px), the per-view gate (3 px) and the 25 px speck filter are in
  processing pixels, so they scale with `--scale`.
- A repeat run can flip the accept decision of one pair out of a hundred (SIFT keeps the 14,000
  strongest features and ties are broken by thread order); the output pixels matched exactly in
  our repeat.
