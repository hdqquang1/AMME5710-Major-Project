# Foliage-free middle view

Run in the `uni` environment. Per sequence (`$S = "data/<seq>"`, images in `$S/images`):

```powershell
$S = "data/5cm_60cm_0"
$colmap = "C:/Program Files/colmap-x64-windows-cuda/bin/colmap.exe"

# 1. SAM 3: COLMAP masks (foliage ignored, grapes kept) + removal masks
# (if `import sam3` fails here, set $env:PYTHONPATH="sam3" first)
python make_masks.py --input $S/images --output $S --overlay

# 2. COLMAP sparse (foliage masked)
& $colmap feature_extractor --database_path $S/database.db --image_path $S/images --ImageReader.mask_path $S/masks --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_PINHOLE
& $colmap exhaustive_matcher --database_path $S/database.db
mkdir $S/sparse
& $colmap mapper --database_path $S/database.db --image_path $S/images --output_path $S/sparse

# 3. COLMAP dense depth maps (use the sparse model with the most images)
& $colmap image_undistorter --image_path $S/images --input_path $S/sparse/0 --output_path $S/dense --output_type COLMAP
& $colmap patch_match_stereo --workspace_path $S/dense --workspace_format COLMAP --PatchMatchStereo.max_image_size 2000 --PatchMatchStereo.geom_consistency 1

# 4. Middle view: foliage pixels filled from the point cloud (black where no view saw behind the leaves)
python reconstruct_background.py --dense $S/dense --masks $S/removal_masks --out $S/background
# optional: also inpaint the remaining holes
python reconstruct_background.py --dense $S/dense --masks $S/removal_masks --out $S/background_inpaint --inpaint
```

