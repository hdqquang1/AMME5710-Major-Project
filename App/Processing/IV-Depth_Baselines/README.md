# Depth-based baselines and earlier lines of work

The report's final method aligns views on fruit features and needs no depth. These folders hold
the depth-based approaches the team built before and beside it. They are kept because the
report discusses them (Secs. 3.5 and 5.2) and because they show the design process.

| Folder | What it is | Status |
| --- | --- | --- |
| [Cherry_SAI](Cherry_SAI) | October prototype on the toy-cherry scene: shift-and-add synthetic aperture at a nominal fruit depth with HSV leaf masks, a two-branch colour detector and a multi-view vote. The "depth shift" baseline of Table 2 descends from this. | Superseded |
| [COLMAP](COLMAP) | SAM 3 foliage masks, a COLMAP sparse and dense reconstruction with the foliage masked out, and a middle view whose foliage pixels are filled from the other views' depth maps (`reconstruct_background.py`); `reconstruct_img.py` renders and shows a thresholded fused point cloud. | Explored, not the reported method |
| [../../../has_sai](../../../has_sai) | A colour-free, occlusion-aware synthetic aperture: per-view foliage masks from multi-view depth, a visibility-weighted all-in-focus render and a shape-based detector, with a 44-scene ray-cast benchmark. Benchmarked in simulation; run once on the five-branch capture ([has_sai/results/real](../../../has_sai/results/real)), where the leaf-free render worked and the count did not. | Explored, not the reported method |

## COLMAP background fill

Dinh's original instructions are kept verbatim in
[COLMAP/README_Dinh_original.md](COLMAP/README_Dinh_original.md) (PowerShell, Windows COLMAP
path). The same steps on macOS or Linux, from a capture folder `$S` holding `images/`:

```bash
python3 ../I-Segmentation/make_masks.py --input $S/images --output $S --overlay
colmap feature_extractor --database_path $S/database.db --image_path $S/images \
    --ImageReader.mask_path $S/masks --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_PINHOLE
colmap exhaustive_matcher --database_path $S/database.db
mkdir -p $S/sparse && colmap mapper --database_path $S/database.db --image_path $S/images --output_path $S/sparse
colmap image_undistorter --image_path $S/images --input_path $S/sparse/0 --output_path $S/dense --output_type COLMAP
colmap patch_match_stereo --workspace_path $S/dense --workspace_format COLMAP \
    --PatchMatchStereo.max_image_size 2000 --PatchMatchStereo.geom_consistency 1
python3 COLMAP/reconstruct_background.py --dense $S/dense --masks $S/removal_masks --out $S/background
```

`reconstruct_background.py` writes `background.png` (black where no view saw behind the
leaves) and `hole_mask.png`; `--inpaint` fills the holes with Telea inpainting for display.
COLMAP with CUDA is required for `patch_match_stereo`.

## Cherry prototype

`cherry_sai_refocus.py --images <folder or zip> --out <dir>` refocuses the rail photos on the
leaf and fruit depths (defaults 572 and 912 mm, 50 mm steps) and writes mean, median and
leaf-masked images plus a focus sweep GIF; `cherry_two_branch_detector.py` and
`cherry_count_compare.py` count red blobs in those images; `cherry_review_labels.py` is a
click-to-label tool. These scripts expect the October cherry captures, which are not in this
repository.
