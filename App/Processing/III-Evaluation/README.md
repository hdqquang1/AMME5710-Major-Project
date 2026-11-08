# Evaluation: counting, metrics, baselines, ablations and figures

This folder is for the scripts that turn the restoration outputs into the numbers of the
report: GrapeSAM counting with one-to-one matching to the 25 labels (24 px radius), fruit-pixel
recall, precision and IoU against the registered leaf-free truth, the mask-union and
mask-majority baselines, the synthetic aperture baselines segmented by SAM 3, the ablation
variants of Table 3, the view-count and spacing sweeps (Fig. 5), the per-grape visibility
analysis (Fig. 6), and the bootstrap, reference and registration-perturbation studies of
Sec. 5.5.

They were run on Finn Panko-Jones's laptop and are being added to this folder; until they
appear here, the numbers in the report can be checked against the result files in
[../../../Out](../../../Out) and the restoration outputs are reproducible with
[../II-Restoration](../II-Restoration).

What each script must do, so the layout is clear whoever runs it next:

| Step | Inputs | Outputs |
| --- | --- | --- |
| Register the leaf-free photograph to the reference view | leaf-free photo, reference view, SAM 3 fruit masks of both | a similarity transform and the registered truth mask |
| Pixel metrics | `observed_mask.png`, registered truth mask | recall, precision, IoU per condition |
| GrapeSAM counting | the occluded view, the reconstruction and the registered leaf-free photo as fruit-only images; the 25 labels | detections, count F1 at confidence 0.05, AP over confidence thresholds, radius sweep |
| Baselines | the aligned stack from the restoration (`evidence.npz`, poses in `diagnostics.json`) | mask union, mask majority, SA mean, median and masked mean, each segmented by SAM 3 |
| Ablations | the five-branch 20 mm capture | one run per row of Table 3 |
| Sweeps and uncertainty | the four 20 mm captures | views curve, span-fixed runs, bootstrap intervals, reference sweep, truth perturbation |
