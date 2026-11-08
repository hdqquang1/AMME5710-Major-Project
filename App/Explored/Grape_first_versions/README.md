# First grape versions (superseded)

Finn Panko-Jones's first grape scripts, kept for the design history behind Sec. 3.5 of the report.
The method itself is `grape_restore_autoref.py` at the repository root. Keep these three files
together: `grape_restore.py` imports its image loader from `grape_reconstruct.py`.

| Script | What it is |
| --- | --- |
| [grape_reconstruct.py](grape_reconstruct.py) | An intermediate version between the first prototype and the method: a plane sweep over nine depth hypotheses with per-pixel colour consensus. It needs the focal length and the rail step, and uniform foliage can form a false consensus, which is why it was left; the report does not describe it. Its default `--images` is a Google Drive path from development, so always pass `--images`. |
| [grape_restore.py](grape_restore.py) | The first fruit-only registration and consensus fusion, with the reference view picked by hand (`--ref`) and a colour/ROI fruit prior. `grape_restore_autoref.py` added automatic reference selection, matching diagnostics and saved SAM 3 masks. |
| [evaluate_reconstruction.py](evaluate_reconstruction.py) | Coverage, MAE and PSNR of the observed pixels against a truth image already registered to the same frame. Replaced by `grape_evaluate.py`. |

```bash
cd App/Explored/Grape_first_versions
python3 grape_reconstruct.py --images ../../../data/2cm_74cm_5branches/images --out ../../../data/out/sweep_5br
```
