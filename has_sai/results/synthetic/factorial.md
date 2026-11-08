# Factorial study: image source x detector (test seeds 2-3, simulator)

F1 on the test layouts; thresholds calibrated on seeds 0-1.  Per-layout F1 in brackets.  'roi' scores only the fruit region; 'wide' scores the whole frame minus a 5 % margin.

## cherry, roi

| Image | colour (as published) | colour, area cap calibrated | colour + distance transform | colour-free 2-D |
|---|---|---|---|---|
| centre photo | 0.701 (0.69, 0.71) | 0.707 (0.71, 0.71) | 0.893 (0.86, 0.92) | 0.409 (0.41, 0.41) |
| prototype masked mean SAI | 0.551 (0.52, 0.58) | 0.570 (0.56, 0.58) | 0.997 (1.00, 1.00) | 0.282 (0.32, 0.25) |
| median SAI | 0.701 (0.61, 0.78) | 0.705 (0.62, 0.78) | 0.924 (0.91, 0.94) | 0.584 (0.54, 0.63) |
| occlusion-aware SAI (ours) | 0.347 (0.32, 0.37) | 0.346 (0.32, 0.37) | 0.968 (0.95, 0.98) | 0.623 (0.58, 0.66) |
| foliage-free photo | 0.600 (0.61, 0.59) | 0.612 (0.63, 0.59) | 0.990 (0.98, 1.00) | 0.753 (0.74, 0.77) |
| per-view colour voting (prototype fuse_multi_view) | F1 0.044 (0.05, 0.04); TP 8 FP 23 FN 328 | | | |
| per-view colour-free 2-D voting | F1 0.303 (0.28, 0.33); TP 159 FP 553 FN 177 | | | |
| ours: occlusion-aware SAI + colour-free 3-D detector | F1 0.812 (0.81, 0.82); TP 264 FP 50 FN 72 | | | |
| oracle: foliage-free photo + true depth + 3-D detector | F1 1.000 (1.00, 1.00); TP 336 FP 0 FN 0 | | | |

## grape, roi

| Image | colour (as published) | colour, area cap calibrated | colour + distance transform | colour-free 2-D |
|---|---|---|---|---|
| centre photo | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.721 (0.72, 0.72) |
| prototype masked mean SAI | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.314 (0.31, 0.32) |
| median SAI | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.785 (0.77, 0.80) |
| occlusion-aware SAI (ours) | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.836 (0.83, 0.84) |
| foliage-free photo | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.872 (0.87, 0.88) |
| per-view colour voting (prototype fuse_multi_view) | F1 0.000 (0.00, 0.00); TP 0 FP 14 FN 561 | | | |
| per-view colour-free 2-D voting | F1 0.602 (0.58, 0.63); TP 362 FP 280 FN 199 | | | |
| ours: occlusion-aware SAI + colour-free 3-D detector | F1 0.402 (0.40, 0.40); TP 212 FP 281 FN 349 | | | |
| oracle: foliage-free photo + true depth + 3-D detector | F1 0.745 (0.76, 0.73); TP 358 FP 42 FN 203 | | | |

## lime, roi

| Image | colour (as published) | colour, area cap calibrated | colour + distance transform | colour-free 2-D |
|---|---|---|---|---|
| centre photo | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.533 (0.48, 0.58) |
| prototype masked mean SAI | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.234 (0.24, 0.23) |
| median SAI | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.737 (0.67, 0.80) |
| occlusion-aware SAI (ours) | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.909 (0.94, 0.88) |
| foliage-free photo | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.927 (0.91, 0.94) |
| per-view colour voting (prototype fuse_multi_view) | F1 0.000 (0.00, 0.00); TP 0 FP 1 FN 36 | | | |
| per-view colour-free 2-D voting | F1 0.333 (0.33, 0.33); TP 30 FP 114 FN 6 | | | |
| ours: occlusion-aware SAI + colour-free 3-D detector | F1 0.866 (0.85, 0.88); TP 29 FP 2 FN 7 | | | |
| oracle: foliage-free photo + true depth + 3-D detector | F1 1.000 (1.00, 1.00); TP 36 FP 0 FN 0 | | | |

## cherry, wide

| Image | colour (as published) | colour, area cap calibrated | colour + distance transform | colour-free 2-D |
|---|---|---|---|---|
| centre photo | 0.705 (0.70, 0.71) | 0.704 (0.70, 0.71) | 0.893 (0.86, 0.92) | 0.277 (0.27, 0.28) |
| prototype masked mean SAI | 0.548 (0.52, 0.58) | 0.551 (0.52, 0.58) | 0.997 (1.00, 1.00) | 0.201 (0.24, 0.16) |
| median SAI | 0.699 (0.60, 0.78) | 0.695 (0.60, 0.78) | 0.924 (0.91, 0.94) | 0.512 (0.49, 0.54) |
| occlusion-aware SAI (ours) | 0.342 (0.31, 0.37) | 0.341 (0.31, 0.37) | 0.970 (0.95, 0.98) | 0.442 (0.39, 0.50) |
| foliage-free photo | 0.600 (0.61, 0.59) | 0.600 (0.61, 0.59) | 0.999 (1.00, 1.00) | 0.636 (0.61, 0.66) |
| per-view colour voting (prototype fuse_multi_view) | F1 0.040 (0.04, 0.04); TP 8 FP 53 FN 328 | | | |
| per-view colour-free 2-D voting | F1 0.252 (0.25, 0.25); TP 102 FP 373 FN 234 | | | |
| ours: occlusion-aware SAI + colour-free 3-D detector | F1 0.729 (0.73, 0.73); TP 259 FP 116 FN 77 | | | |
| oracle: foliage-free photo + true depth + 3-D detector | F1 1.000 (1.00, 1.00); TP 336 FP 0 FN 0 | | | |

## grape, wide

| Image | colour (as published) | colour, area cap calibrated | colour + distance transform | colour-free 2-D |
|---|---|---|---|---|
| centre photo | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.664 (0.65, 0.68) |
| prototype masked mean SAI | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.220 (0.22, 0.22) |
| median SAI | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.738 (0.72, 0.76) |
| occlusion-aware SAI (ours) | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.779 (0.77, 0.79) |
| foliage-free photo | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.813 (0.80, 0.83) |
| per-view colour voting (prototype fuse_multi_view) | F1 0.000 (0.00, 0.00); TP 0 FP 27 FN 561 | | | |
| per-view colour-free 2-D voting | F1 0.536 (0.52, 0.55); TP 323 FP 320 FN 238 | | | |
| ours: occlusion-aware SAI + colour-free 3-D detector | F1 0.336 (0.32, 0.36); TP 154 FP 202 FN 407 | | | |
| oracle: foliage-free photo + true depth + 3-D detector | F1 0.719 (0.72, 0.72); TP 358 FP 77 FN 203 | | | |

## lime, wide

| Image | colour (as published) | colour, area cap calibrated | colour + distance transform | colour-free 2-D |
|---|---|---|---|---|
| centre photo | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.417 (0.35, 0.48) |
| prototype masked mean SAI | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.128 (0.15, 0.11) |
| median SAI | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.667 (0.62, 0.71) |
| occlusion-aware SAI (ours) | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.862 (0.88, 0.85) |
| foliage-free photo | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.000 (0.00, 0.00) | 0.879 (0.91, 0.85) |
| per-view colour voting (prototype fuse_multi_view) | F1 0.000 (0.00, 0.00); TP 0 FP 13 FN 36 | | | |
| per-view colour-free 2-D voting | F1 0.179 (0.08, 0.27); TP 10 FP 66 FN 26 | | | |
| ours: occlusion-aware SAI + colour-free 3-D detector | F1 0.849 (0.79, 0.91); TP 28 FP 2 FN 8 | | | |
| oracle: foliage-free photo + true depth + 3-D detector | F1 1.000 (1.00, 1.00); TP 36 FP 0 FN 0 | | | |

## Cherries at 50 % foliage (test), F1 by method

- median_sai|free2d: F1 0.450 (TP 36, FP 12, FN 76, 4 scenes)
- finn_sai|colour_dt: F1 0.991 (TP 110, FP 0, FN 2, 4 scenes)
- vote|colour: F1 0.132 (TP 8, FP 1, FN 104, 4 scenes)
- vote|free2d: F1 0.301 (TP 26, FP 35, FN 86, 4 scenes)
- ours|relief3d: F1 0.531 (TP 55, FP 40, FN 57, 4 scenes)
