# Captures

The photographs are not in git (about 90 MB per capture). They live on the University of
Sydney OneDrive of Dinh Quang Huynh, shared with the team and the lecturer on request, at
`Year 4/AMME5710/Major Project/data/2cm_60cm/`. Each capture is a folder holding an `images/`
subfolder of JPEGs named by the phone's timestamp (`20261022_033632103_iOS.jpeg`, ...), which
sort in rail order.

## The captures the report uses

| Folder | Photos | Rail step | Fruit distance | Foreground branches |
| --- | --- | --- | --- | --- |
| `2cm_74cm_3branches` | 28 | 20 mm | about 0.74 m to the enclosure front; fruit about 1.1 m | 3 |
| `2cm_74cm_4branches` | 28 | 20 mm | | 4 |
| `2cm_74cm_5branches` | 28 | 20 mm | | 5 |
| `2cm_74cm_6branches` | 27 | 20 mm | | 6 |

The leaf-free photograph of the same fruit (branches removed) and the 25 grape labels were
taken and labelled after these captures; they are the truth for the pixel and count metrics.

## Other captures in the same OneDrive folder

`2cm_74cm_*_30deg` and `*_60deg` repeat the four conditions with the camera turned 30 or 60
degrees from the rail normal; `2cm_80cm_1` to `_5` and `5cm_60cm_*` are earlier captures at
other steps and distances; `Sequence 1` and `Sequence 2` are the first October rail
sequences. None of these is used in the report.

## Layout the scripts expect

```
data/
  2cm_74cm_5branches/
    images/          the rail photos (PNG or JPEG), names sorting in rail order
    grape_masks/     SAM 3 fruit masks, <stem>.png, from make_masks.py --grape-masks
    removal_masks/   SAM 3 foliage masks dilated 25 px, <stem>_mask.png (COLMAP fill)
    masks/           COLMAP feature masks, <name>.png
    overlay/         mask overlays for checking by eye
```

`App/Capturing/convert_heic.py` produces `data/<capture>/images/` from `raw/<capture>/` when the
phone saved HEIC files; the OneDrive copies are already JPEG.
