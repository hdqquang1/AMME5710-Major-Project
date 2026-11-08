# Data: captures, masks and the leaf-free photograph

Nothing in this folder except this README is in git: the photographs are about 90 MB per
capture, and the masks are regenerated from them. The scripts at the repository root expect
the layout below and are run from the repository root.

## Where the photographs are

The rail photographs are on Dinh Quang Huynh's University of Sydney OneDrive, shared with the
team, at `Year 4/AMME5710/Major Project/data/2cm_60cm/<capture>/images/`. The phone's file
names (`20261022_033606632_iOS.jpeg`, ...) are timestamps, so they sort in rail order.

The leaf-free photograph used as the truth, `20261022_034827542_iOS.jpg`, was taken at 03:48
on 22 October 2026, about two minutes after the last three-branch photo, with every foreground
branch removed. It is **not** in the shared OneDrive folder; ask Finn Panko-Jones for it. The
25 berry centres labelled on it are in git, in [../labels/t1.csv](../labels/t1.csv).

## The four captures in the report

| Folder | Photos | Rail step | Foreground branches |
| --- | --- | --- | --- |
| `2cm_74cm_3branches` | 28 | 20 mm | 3 |
| `2cm_74cm_4branches` | 28 | 20 mm | 4 |
| `2cm_74cm_5branches` | 28 | 20 mm | 5 |
| `2cm_74cm_6branches` | 27 | 20 mm | 6 |

All four share one arrangement of red grapes. `grape_restore_autoref.py` records the scene as
foliage at roughly 0.74 to 0.80 m and fruit at roughly 1.1 to 1.2 m from the rail; these are
approximate tape measurements kept as context, and no script uses them. The 40 and 60 mm
conditions are subsets of these captures (every second or third photo around the same
reference view), made by `run_experiments.py`.

The shared folder also holds captures the report does not use: the same branch counts with the
phone turned 30 or 60 degrees (`*_30deg`, `*_60deg`), `2cm_80cm_1` to `_5` and `5cm_60cm_*`
at other steps and distances, and the first October sequences (`Sequence 1`, `Sequence 2`).
`results/refactor_check/80_*` holds exploratory runs on two of the `2cm_80cm` captures.

## Layout the scripts expect

```
data/
  20261022_034827542_iOS.jpg       the leaf-free truth photograph (ask Finn)
  2cm_74cm_5branches/
    images/                        the rail photos, names sorting in rail order
    grape_masks/                   SAM 3 fruit masks, <stem>.png, white = grape
    removal_masks/                 SAM 3 foliage masks, <stem>_mask.png, white = foliage
  2cm_74cm_3branches/ ...          the same for the other captures
```

- `images/` comes from OneDrive, or from `App/Capturing/convert_heic.py` for HEIC files.
- `grape_masks/` and `removal_masks/` come from
  `App/Segmentation/make_masks.py --grape-masks` (prompt `grape` for fruit; `green leaf`,
  `green vine`, `green stem` for foliage, dilated by 25 px with fruit pixels removed).
  `grape_segment.py` at the root writes the same fruit masks.
- The truth mask that `run_experiments.py` takes as `--truth-grapes` lives at
  `results/refactor_check/truth/t1/grape_masks/t1.png` and is git-ignored like every image in
  `results/`; it is a SAM 3 fruit mask of the leaf-free photograph.
