# Results

Every run the report quotes, as small text files: parameters, poses, per-pair and per-view
registration evidence, and the scores. Images and arrays are git-ignored because the scripts
regenerate them, so a folder here holds `diagnostics.json` and `matching_diagnostics.json`
(and `row.json` for scored variants) but not `grapes_rgb.png`.

Two commands use these files without any photographs, masks or weights:

```bash
python3 tools/check_report_numbers.py   # compares the report's Tables 1-3 and quoted numbers with these files
python3 make_ablation_assets.py         # rebuilds results/ablation/assets (tables byte-identical; plots
                                        # can differ by a pixel with other fonts)
```

## [report/](report): the main study (Table 1)

Written by `run_experiments.py`, then extended by `analyse_experiments.py`.

| File or folder | Contents |
| --- | --- |
| `b{3,4,5,6}_{20,40,60}mm/` | One reconstruction per branch count and spacing, with SAM 3 masks. 40 and 60 mm keep every second or third view around the same reference. |
| `b{3,4,5,6}_20mm_colour/` | The same at 20 mm with the red colour rule in place of the SAM 3 fruit masks; the SAM 3 foliage masks are kept. At six branches it failed: fewer than three views passed alignment validation (`FAILED.txt`). |
| `results.csv` | One row per run: pixel recall, precision and IoU against the leaf-free photograph, GrapeSAM, Hough and SAM 3 count scores, registration statistics. |
| `results_full.csv` | `results.csv` plus the like-for-like count scores on fruit-only inputs (`gs_floor_fruit_*` for the single view, `gs_ceiling_fruit_*` for the leaf-free photograph), the excluded views, the accepted pairs and the reference view. **Table 1 is this file.** |
| `manifest.json` | Processing scale 0.8, match radius 24 px, and the truth, label and weight paths. |
| `report_draft.pdf` | A draft of the report kept with the runs. The submitted report is [../report/report.pdf](../report/report.pdf). |

## [ablation/](ablation): baselines, ablations and uncertainty (Tables 2 and 3, Figs. 5 and 6, Sec. 5.5)

Written by `run_ablations.py`; each scored variant has a `row.json` and is collected in a CSV.

| File or folder | Experiment |
| --- | --- |
| `a1.csv`, `a1/` | Fusion baselines at 20 mm for 3 to 6 branches: mask union and majority, synthetic aperture mean, median and foliage-masked mean on the depth-shift alignment and on our alignment, and SAM 3 on the single view (Table 2). |
| `a2.csv`, `a2/` | Component ablations at five branches: minimum views, minimum span, colour threshold, output colour, masks and alignment variants (Table 3). |
| `a3/`, `a4/`, `a34.csv` | Number of views against spacing (`a3`, Fig. 5 and the fixed-span study of Sec. 5.4) and other reference views at offsets of 4 and 8 (`a4`, Sec. 5.5). |
| `a5_bootstrap.csv`, `a5_pr.csv`, `a5_tolerance.csv` | Bootstrap intervals over the 25 grapes, precision and recall over confidence thresholds (AP), and the match-radius sweep. |
| `a6.csv` | Truth-registration perturbations (shifts of 1 to 5 px, rotations, scale): the noise floor of the evaluation. |
| `a7.csv` | Per-grape visibility: in how many aligned views each labelled grape is visible, and whether it was recovered and counted (Fig. 6). |
| `a8_runtime.json` | Stage times for five branches, 28 views, on an RTX 2000 Ada laptop. |
| `assets/` | What `make_ablation_assets.py` builds from the above: `tab_baselines.tex` (Table 2), `tab_ablation.tex` (Table 3), `views_curve.pdf` (Fig. 5), `visibility.pdf` (Fig. 6), and the tables behind the numbers of Secs. 5.4 and 5.5. |
| `logs/` | Console logs of the long runs. |

## [refactor_check/](refactor_check): repeat runs

Runs of the refactored code, kept as a check of the code.

| Folder | Contents |
| --- | --- |
| `74_{3,4,5}b_s8/` | Scale 0.8 re-runs of the three- to five-branch captures. They match the reported runs exactly: the same reference view, the same accept or reject decision for all 102 tested pairs and the same pixel counts (checked by `tools/check_report_numbers.py`). |
| `74_{3,4,5,6}b/` | The same captures at scale 0.4 (for example 101 of 102 pairs accepted and 13,395 supported pixels at four branches). |
| `80_*` | Exploratory scale-0.4 runs on two `2cm_80cm` captures with the green, red and two-colour rules and with SAM 3 masks. Not used in the report. |
| `truth/` | The registration of the leaf-free photograph onto the scale-0.4 runs and its pixel scores. |
