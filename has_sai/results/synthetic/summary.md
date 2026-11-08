# Synthetic benchmark summary

Test scenes only (seeds 2 and 3, never looked at while developing the method); score thresholds and the verifier were calibrated on seeds 0 and 1 (best pooled F1).  Scenes with the same seed share one fruit layout, so each fruit type has two test layouts.  All numbers are from the simulator.

## Counting: Threshold calibrated per fruit type

### cherry (countable 336, at least half visible in >= 2 photos: 331, visible in the centre photo through foliage: 226)

| Method | Precision | Recall | F1 | Mean abs. count error | n_pred / n_gt |
|---|---|---|---|---|---|
| Centre photo + colour detector (prototype) | 0.931 | 0.562 | 0.701 | 39.6 % | 203 / 336 |
|   same, area cap calibrated | 0.927 | 0.571 | 0.707 | 38.4 % | 207 / 336 |
| Leaf-masked mean SAI + colour detector (prototype) | 0.977 | 0.384 | 0.551 | 60.7 % | 132 / 336 |
|   same, area cap calibrated | 0.992 | 0.381 | 0.550 | 61.6 % | 129 / 336 |
| Centre photo + colour-free 2-D detector | 0.381 | 0.431 | 0.405 | 24.7 % | 381 / 336 |
| Median SAI + colour-free 2-D detector | 0.700 | 0.494 | 0.579 | 33.0 % | 237 / 336 |
| Occlusion-aware SAI + colour-free 3-D detector (ours) | 0.861 | 0.771 | 0.813 | 16.4 % | 301 / 336 |
|   ours + learned colour-free verifier (random forest) | 1.000 | 0.762 | 0.865 | 23.8 % | 256 / 336 |
|   ours without radial-symmetry candidates | 0.835 | 0.691 | 0.756 | 20.2 % | 278 / 336 |
|   ours without half-aperture selection | 0.741 | 0.783 | 0.761 | 22.9 % | 355 / 336 |
|   ours without semi-global aggregation | 0.232 | 0.610 | 0.336 | 163.1 % | 884 / 336 |
|   ours without foliage masks | 0.098 | 0.530 | 0.166 | 439.0 % | 1811 / 336 |
| Foliage-free photo + true depth (oracle reference) | 1.000 | 1.000 | 1.000 | 0.0 % | 336 / 336 |

### grape (countable 561, at least half visible in >= 2 photos: 561, visible in the centre photo through foliage: 435)

| Method | Precision | Recall | F1 | Mean abs. count error | n_pred / n_gt |
|---|---|---|---|---|---|
| Centre photo + colour detector (prototype) | 0.000 | 0.000 | 0.000 | 100.0 % | 0 / 561 |
|   same, area cap calibrated | 0.000 | 0.000 | 0.000 | 100.0 % | 0 / 561 |
| Leaf-masked mean SAI + colour detector (prototype) | 0.000 | 0.000 | 0.000 | 100.0 % | 0 / 561 |
|   same, area cap calibrated | 0.000 | 0.000 | 0.000 | 100.0 % | 0 / 561 |
| Centre photo + colour-free 2-D detector | 0.908 | 0.597 | 0.720 | 34.2 % | 369 / 561 |
| Median SAI + colour-free 2-D detector | 0.876 | 0.717 | 0.788 | 18.2 % | 459 / 561 |
| Occlusion-aware SAI + colour-free 3-D detector (ours) | 0.430 | 0.378 | 0.402 | 12.1 % | 493 / 561 |
|   ours + learned colour-free verifier (random forest) | 0.935 | 0.387 | 0.547 | 58.6 % | 232 / 561 |
|   ours without radial-symmetry candidates | 0.432 | 0.378 | 0.403 | 12.5 % | 491 / 561 |
|   ours without half-aperture selection | 0.539 | 0.403 | 0.461 | 25.3 % | 419 / 561 |
|   ours without semi-global aggregation | 0.230 | 0.307 | 0.263 | 42.3 % | 748 / 561 |
|   ours without foliage masks | 0.121 | 0.406 | 0.186 | 239.6 % | 1889 / 561 |
| Foliage-free photo + true depth (oracle reference) | 0.895 | 0.638 | 0.745 | 28.7 % | 400 / 561 |

### lime (countable 36, at least half visible in >= 2 photos: 36, visible in the centre photo through foliage: 21)

| Method | Precision | Recall | F1 | Mean abs. count error | n_pred / n_gt |
|---|---|---|---|---|---|
| Centre photo + colour detector (prototype) | 0.000 | 0.000 | 0.000 | 100.0 % | 0 / 36 |
|   same, area cap calibrated | 0.000 | 0.000 | 0.000 | 100.0 % | 0 / 36 |
| Leaf-masked mean SAI + colour detector (prototype) | 0.000 | 0.000 | 0.000 | 100.0 % | 0 / 36 |
|   same, area cap calibrated | 0.000 | 0.000 | 0.000 | 100.0 % | 0 / 36 |
| Centre photo + colour-free 2-D detector | 0.667 | 0.444 | 0.533 | 33.3 % | 24 / 36 |
| Median SAI + colour-free 2-D detector | 0.926 | 0.694 | 0.794 | 30.6 % | 27 / 36 |
| Occlusion-aware SAI + colour-free 3-D detector (ours) | 0.938 | 0.833 | 0.882 | 16.7 % | 32 / 36 |
|   ours + learned colour-free verifier (random forest) | 0.968 | 0.833 | 0.895 | 13.9 % | 31 / 36 |
|   ours without radial-symmetry candidates | 0.935 | 0.806 | 0.866 | 19.4 % | 31 / 36 |
|   ours without half-aperture selection | 0.931 | 0.750 | 0.831 | 25.0 % | 29 / 36 |
|   ours without semi-global aggregation | 0.938 | 0.833 | 0.882 | 16.7 % | 32 / 36 |
|   ours without foliage masks | 0.487 | 0.528 | 0.507 | 75.0 % | 39 / 36 |
| Foliage-free photo + true depth (oracle reference) | 1.000 | 1.000 | 1.000 | 0.0 % | 36 / 36 |

## Counting: One threshold calibrated on cherries only (cross-fruit generalisation)

### cherry (countable 336, at least half visible in >= 2 photos: 331, visible in the centre photo through foliage: 226)

| Method | Precision | Recall | F1 | Mean abs. count error | n_pred / n_gt |
|---|---|---|---|---|---|
| Centre photo + colour detector (prototype) | 0.931 | 0.562 | 0.701 | 39.6 % | 203 / 336 |
|   same, area cap calibrated | 0.927 | 0.571 | 0.707 | 38.4 % | 207 / 336 |
| Leaf-masked mean SAI + colour detector (prototype) | 0.977 | 0.384 | 0.551 | 60.7 % | 132 / 336 |
|   same, area cap calibrated | 0.992 | 0.381 | 0.550 | 61.6 % | 129 / 336 |
| Centre photo + colour-free 2-D detector | 0.381 | 0.431 | 0.405 | 24.7 % | 381 / 336 |
| Median SAI + colour-free 2-D detector | 0.700 | 0.494 | 0.579 | 33.0 % | 237 / 336 |
| Occlusion-aware SAI + colour-free 3-D detector (ours) | 0.861 | 0.771 | 0.813 | 16.4 % | 301 / 336 |
|   ours + learned colour-free verifier (random forest) | 1.000 | 0.762 | 0.865 | 23.8 % | 256 / 336 |
|   ours without radial-symmetry candidates | 0.835 | 0.691 | 0.756 | 20.2 % | 278 / 336 |
|   ours without half-aperture selection | 0.741 | 0.783 | 0.761 | 22.9 % | 355 / 336 |
|   ours without semi-global aggregation | 0.232 | 0.610 | 0.336 | 163.1 % | 884 / 336 |
|   ours without foliage masks | 0.098 | 0.530 | 0.166 | 439.0 % | 1811 / 336 |
| Foliage-free photo + true depth (oracle reference) | 1.000 | 1.000 | 1.000 | 0.0 % | 336 / 336 |

### grape (countable 561, at least half visible in >= 2 photos: 561, visible in the centre photo through foliage: 435)

| Method | Precision | Recall | F1 | Mean abs. count error | n_pred / n_gt |
|---|---|---|---|---|---|
| Centre photo + colour detector (prototype) | 0.000 | 0.000 | 0.000 | 100.0 % | 0 / 561 |
|   same, area cap calibrated | 0.000 | 0.000 | 0.000 | 100.0 % | 0 / 561 |
| Leaf-masked mean SAI + colour detector (prototype) | 0.000 | 0.000 | 0.000 | 100.0 % | 0 / 561 |
|   same, area cap calibrated | 0.000 | 0.000 | 0.000 | 100.0 % | 0 / 561 |
| Centre photo + colour-free 2-D detector | 0.527 | 0.659 | 0.586 | 25.1 % | 702 / 561 |
| Median SAI + colour-free 2-D detector | 0.896 | 0.706 | 0.790 | 21.2 % | 442 / 561 |
| Occlusion-aware SAI + colour-free 3-D detector (ours) | 0.641 | 0.226 | 0.335 | 64.7 % | 198 / 561 |
|   ours + learned colour-free verifier (random forest) | 0.995 | 0.353 | 0.521 | 64.5 % | 199 / 561 |
|   ours without radial-symmetry candidates | 0.642 | 0.246 | 0.356 | 61.7 % | 215 / 561 |
|   ours without half-aperture selection | 0.615 | 0.267 | 0.373 | 56.5 % | 244 / 561 |
|   ours without semi-global aggregation | 0.225 | 0.127 | 0.162 | 43.8 % | 315 / 561 |
|   ours without foliage masks | 0.112 | 0.317 | 0.166 | 210.5 % | 1588 / 561 |
| Foliage-free photo + true depth (oracle reference) | 0.905 | 0.561 | 0.693 | 38.0 % | 348 / 561 |

### lime (countable 36, at least half visible in >= 2 photos: 36, visible in the centre photo through foliage: 21)

| Method | Precision | Recall | F1 | Mean abs. count error | n_pred / n_gt |
|---|---|---|---|---|---|
| Centre photo + colour detector (prototype) | 0.000 | 0.000 | 0.000 | 100.0 % | 0 / 36 |
|   same, area cap calibrated | 0.000 | 0.000 | 0.000 | 100.0 % | 0 / 36 |
| Leaf-masked mean SAI + colour detector (prototype) | 0.000 | 0.000 | 0.000 | 100.0 % | 0 / 36 |
|   same, area cap calibrated | 0.000 | 0.000 | 0.000 | 100.0 % | 0 / 36 |
| Centre photo + colour-free 2-D detector | 0.737 | 0.389 | 0.509 | 47.2 % | 19 / 36 |
| Median SAI + colour-free 2-D detector | 1.000 | 0.556 | 0.714 | 44.4 % | 20 / 36 |
| Occlusion-aware SAI + colour-free 3-D detector (ours) | 0.882 | 0.833 | 0.857 | 11.1 % | 34 / 36 |
|   ours + learned colour-free verifier (random forest) | 1.000 | 0.833 | 0.909 | 16.7 % | 30 / 36 |
|   ours without radial-symmetry candidates | 0.879 | 0.806 | 0.841 | 13.9 % | 33 / 36 |
|   ours without half-aperture selection | 0.871 | 0.750 | 0.806 | 19.4 % | 31 / 36 |
|   ours without semi-global aggregation | 0.882 | 0.833 | 0.857 | 16.7 % | 34 / 36 |
|   ours without foliage masks | 0.146 | 0.556 | 0.231 | 330.6 % | 137 / 36 |
| Foliage-free photo + true depth (oracle reference) | 1.000 | 1.000 | 1.000 | 0.0 % | 36 / 36 |

## Cherries by foliage coverage (test, per-fruit threshold), F1

| Method | 10 % | 30 % | 50 % |
|---|---|---|---|
| Centre photo + colour detector (prototype) | 0.610 | 0.682 | 0.796 |
|   same, area cap calibrated | 0.623 | 0.689 | 0.796 |
| Leaf-masked mean SAI + colour detector (prototype) | 0.463 | 0.600 | 0.584 |
|   same, area cap calibrated | 0.466 | 0.600 | 0.579 |
| Centre photo + colour-free 2-D detector | 0.612 | 0.376 | 0.274 |
| Median SAI + colour-free 2-D detector | 0.673 | 0.582 | 0.447 |
| Occlusion-aware SAI + colour-free 3-D detector (ours) | 0.987 | 0.884 | 0.543 |
|   ours + learned colour-free verifier (random forest) | 0.991 | 0.918 | 0.626 |
|   ours without radial-symmetry candidates | 0.940 | 0.818 | 0.490 |
|   ours without half-aperture selection | 0.966 | 0.856 | 0.461 |
|   ours without semi-global aggregation | 0.620 | 0.315 | 0.174 |
|   ours without foliage masks | 0.562 | 0.116 | 0.036 |
| Foliage-free photo + true depth (oracle reference) | 1.000 | 1.000 | 1.000 |

## Leaf-free image quality inside the ROI (all scenes)

| Image | PSNR cherry | SSIM cherry | PSNR grape | PSNR lime |
|---|---|---|---|---|
| finn_sai | 17.93 | 0.3730 | 15.20 | 15.41 |
| mean_sai | 23.49 | 0.7296 | 22.05 | 21.38 |
| median_sai | 24.99 | 0.7746 | 23.55 | 22.27 |
| ours | 31.02 | 0.9051 | 32.02 | 31.72 |
| ours_nohalf | 31.02 | 0.9051 | 32.02 | 31.72 |
| ours_nomask | 26.16 | 0.8336 | 27.17 | 24.25 |
| ours_nosgm | 30.62 | 0.9003 | 31.61 | 31.34 |
| single | 19.88 | 0.7184 | 19.79 | 18.12 |

## Foliage masks (colour-free, from depth)

Reference view: recall 0.995, precision 0.715, IoU 0.713.  All views: recall 0.994, IoU 0.687.

## Background depth, median absolute error [mm]

| Variant | Backdrop | Lattice | Fruit |
|---|---|---|---|
| ours | 20.21 | 1.53 | 5.69 |
| ours_nohalf | 16.25 | 1.19 | 4.57 |
| ours_nosgm | 8.27 | 5.35 | 3.56 |
| ours_nomask | 24.67 | 15.19 | 12.01 |

## Mean run time per scene [s]

{"baselines": 3.1, "ours": 48.3, "ours_nohalf": 42.0, "ours_nosgm": 41.6, "ours_nomask": 54.7, "total": 196.2}
