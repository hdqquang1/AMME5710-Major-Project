# Explored approaches

The reported method (the scripts at the repository root) aligns the views on fruit features and
needs no scene depth. The folders here hold what the team built before it and beside it. They
are kept because the report discusses them (Secs. 3.5 and 5.2) and because they show how the
method was arrived at; none of them produced a number in the report's tables.

| Folder | Author | What it is | Status |
| --- | --- | --- | --- |
| [Cherry_SAI](Cherry_SAI) | Finn Panko-Jones | The October prototype on a scene of toy cherries: shift-and-add synthetic aperture at a nominal fruit depth from known geometry, HSV leaf masks, a two-branch colour detector and a multi-view vote. | Superseded |
| [Grape_first_versions](Grape_first_versions) | Finn Panko-Jones | The first grape versions: a nine-depth plane sweep with per-pixel colour consensus (`grape_reconstruct.py`, the intermediate prototype of Sec. 3.5), the first fruit-only registration with a hand-picked reference (`grape_restore.py`) and an early pixel evaluator. | Superseded by the root pipeline |
| [COLMAP_background](COLMAP_background) | Dinh Quang Huynh | SAM 3 foliage masks, a COLMAP sparse and dense reconstruction with the foliage masked out, and a middle view whose foliage pixels are filled from the other views' depth maps. | Explored, not used in the report |
| [../../has_sai](../../has_sai) | Has Salvesen | A colour-free, occlusion-aware synthetic aperture: per-view foliage masks from multi-view plane-sweep depth, a visibility-weighted all-in-focus render and a shape-based detector, with a 44-scene ray-cast benchmark. Run once on the five-branch capture, where the leaf-free render worked and the count did not. | Explored, simulation only |

The depth-shift baseline of Table 2 is not one of these scripts: `depth_shift_poses` in
[../../run_ablations.py](../../run_ablations.py) reimplements the prototype's alignment (one
horizontal shift per rail step, chosen by maximising focus over the reference fruit region) so
that it runs on the same views and masks as every other row.

## COLMAP background fill

Dinh's original instructions are kept verbatim in
[COLMAP_background/README_Dinh_original.md](COLMAP_background/README_Dinh_original.md)
(PowerShell, Windows COLMAP path). The same steps on macOS or Linux, from the repository root,
for a capture folder `$S` that holds `images/`:

```bash
PYTHONPATH=third_party/sam3 python3 App/Segmentation/make_masks.py --input $S/images --output $S --overlay
colmap feature_extractor --database_path $S/database.db --image_path $S/images \
    --ImageReader.mask_path $S/masks --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_PINHOLE
colmap exhaustive_matcher --database_path $S/database.db
mkdir -p $S/sparse && colmap mapper --database_path $S/database.db --image_path $S/images --output_path $S/sparse
colmap image_undistorter --image_path $S/images --input_path $S/sparse/0 --output_path $S/dense --output_type COLMAP
colmap patch_match_stereo --workspace_path $S/dense --workspace_format COLMAP \
    --PatchMatchStereo.max_image_size 2000 --PatchMatchStereo.geom_consistency 1
python3 App/Explored/COLMAP_background/reconstruct_background.py --dense $S/dense --masks $S/removal_masks --out $S/background
```

`reconstruct_background.py` writes `background.png` (black where no view saw behind the leaves)
and `hole_mask.png`; `--inpaint` fills the holes with Telea inpainting for display only.
`reconstruct_img.py` is Dinh's first version, which thresholds the fused point cloud in depth
and re-renders it. COLMAP needs CUDA for `patch_match_stereo`.

## Cherry prototype

`cherry_sai_refocus.py --images <folder or zip> --out <dir>` refocuses the rail photos on the
leaf and fruit depths you give it (defaults 572 and 912 mm, 50 mm steps) and writes mean, median
and leaf-masked images and a focus-sweep GIF; `cherry_two_branch_detector.py` and
`cherry_count_compare.py` count red blobs in them; `cherry_review_labels.py` is a click-to-label
tool. They expect the October cherry captures, which are not in this repository.
