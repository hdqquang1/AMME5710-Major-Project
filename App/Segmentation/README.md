# Segmentation: SAM 3 fruit and foliage masks

[make_masks.py](make_masks.py) runs SAM 3 (Segment Anything with Concepts) on every photo of
a capture with text prompts and writes binary masks. It needs PyTorch with CUDA and the SAM 3
repository on the Python path (`git clone https://github.com/facebookresearch/sam3`, then follow
its README for the checkpoint; if `import sam3` fails set `PYTHONPATH` to the clone).

Prompts and confidence thresholds, as in the code:

| Class | Prompts | Threshold | Rule |
| --- | --- | --- | --- |
| Fruit | `grape` | 0.20 | Union of every instance |
| Foliage | `green leaf`, `green vine`, `green stem` | 0.35 | Union of every instance, with fruit pixels removed (the vine and stem prompts also fire on grapes) |

```bash
python3 make_masks.py --input ../../../data/2cm_74cm_5branches/images \
                      --output ../../../data/2cm_74cm_5branches --overlay --grape-masks
```

Outputs, one file per photo:

| Folder | File name | Contents | Used by |
| --- | --- | --- | --- |
| `grape_masks/` | `<stem>.png` | White where SAM 3 found fruit, at the photo's size | `--target-masks` of the restoration (`--grape-masks` flag) |
| `removal_masks/` | `<stem>_mask.png` | Foliage dilated by 25 px, fruit removed | the COLMAP background fill |
| `masks/` | `<name>.png` | COLMAP feature mask: black (ignore) on eroded foliage, white elsewhere | COLMAP `--ImageReader.mask_path` |
| `overlay/` | `<stem>.jpg` | Blue = removed foliage, red = fruit, for checking the masks by eye | nobody; look at it |

The report does not validate these masks against hand labels (Sec. 3.2), and the truth masks
of the leaf-free photographs come from the same model, so check the overlays before trusting
a run.
