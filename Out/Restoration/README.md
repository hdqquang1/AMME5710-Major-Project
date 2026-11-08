# Restoration outputs

One folder per run of `App/Processing/II-Restoration/grape_restore_autoref.py`. Images and the
`evidence.npz` arrays are git-ignored; `diagnostics.json` and `matching_diagnostics.json` are
kept because they record every parameter, the accepted and rejected pairs, the per-view
held-out errors and the chosen reference.

| Folder | Run | Outcome |
| --- | --- | --- |
| `2cm_74cm_5branches_colour_rule/` | The five-branch capture at scale 0.8 **without SAM 3 masks**, so the script fell back to its red/purple colour rule inside the default `--target-roi` (a fraction box designed for a landscape framing; these photos are portrait). Run on 8 November 2026 from this repository to check that the script runs on the real capture. | Only 11 of 102 pairs passed validation and 7 of 28 views stayed connected (reference view 24); 21,293 fruit pixels were supported. This is the colour-rule ablation path, not the reported method, and it shows why the method needs the SAM 3 fruit masks: without them the fruit features are too few and too unreliable to align the sweep. |

The report's runs (SAM 3 masks, 28 views connected at 20 mm for three to five branches) were made
on Finn Panko-Jones's laptop; their diagnostics are to be added here with the evaluation scripts.
