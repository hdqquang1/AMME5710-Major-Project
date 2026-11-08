# Segmentation: SAM 3 fruit and foliage masks

[make_masks.py](make_masks.py) (Dinh Quang Huynh) runs SAM 3 on every photo of a capture with
text prompts and writes binary masks. It needs PyTorch with a CUDA GPU and the SAM 3 code,
which this repository links as a submodule in `third_party/sam3`
([../../third_party/README.md](../../third_party/README.md) covers the patch and the gated
checkpoint).

| Class | Prompts | Threshold | Rule |
| --- | --- | --- | --- |
| Fruit | `grape` | 0.20 | Union of every instance |
| Foliage | `green leaf`, `green vine`, `green stem` | 0.35 | Union of every instance, dilated 25 px, with fruit pixels removed (the vine and stem prompts also fire on grapes) |

Run from the repository root. `make_masks.py` imports `sam3` directly, so put the submodule
on the Python path (or `pip install -e third_party/sam3`):

```bash
PYTHONPATH=third_party/sam3 python3 App/Segmentation/make_masks.py \
    --input data/2cm_74cm_5branches/images --output data/2cm_74cm_5branches --overlay --grape-masks
```

| Folder | File name | Contents | Used by |
| --- | --- | --- | --- |
| `grape_masks/` | `<stem>.png` | White where SAM 3 found fruit, at the photo's size (written with `--grape-masks`) | `grape_restore_autoref.py --grape-masks`, `run_experiments.py`, `run_ablations.py` |
| `removal_masks/` | `<stem>_mask.png` | Foliage dilated 25 px, fruit removed | the same scripts, as foliage masks; the COLMAP background fill |
| `masks/` | `<name>.png` | COLMAP feature mask: black on eroded foliage, white elsewhere | COLMAP `--ImageReader.mask_path` |
| `overlay/` | `<stem>.jpg` | Blue = removed foliage, red = fruit | checking the masks by eye |

`grape_segment.py` at the repository root wraps the same model and prompts and writes the same
fruit masks (`grape_masks/`), or foliage masks into `foliage_masks/` with `<stem>.png` names.

The masks are not validated against hand labels (Sec. 3.2 of the report), and the truth mask of
the leaf-free photograph comes from the same model, so the pixel scores are not independent of
this stage. Check the overlays before trusting a run.
