# Third-party code and weights

## Code (git submodules)

| Folder | Source | Used for |
|---|---|---|
| `sam3/` | https://github.com/facebookresearch/sam3 | Fruit and foliage masks (`grape_segment.py`) and SAM 3 counting |
| `GrapeSAM/` | https://github.com/tong-xz/GrapeSAM | ViViD-5K point model for berry counting (`grape_count.py --method grapesam`) |

After cloning this repo:

```
git submodule update --init
git -C third_party/sam3 apply ../patches/sam3_local.patch
```

`patches/sam3_local.patch` makes SAM 3 run on Windows and an 8 GB GPU:
- It falls back to OpenCV when `triton` is unavailable.
- It upsamples masks in fp16 chunks.
- It loads the BPE vocabulary without `pkg_resources`.

## Weights (not in git, too large)

- **SAM 3:** downloaded automatically from Hugging Face on first use. You must request access to `facebook/sam3` and run `huggingface-cli login`.
- **GrapeSAM point model:**
  1. Download `weights.zip` from the authors' Google Drive folder linked in `GrapeSAM/README.md`.
  2. Extract it so that the checkpoint is at `third_party/weights/point/best_val.pth`.
