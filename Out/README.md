# Out

What the scripts write, one subfolder per stage:

| Subfolder | Written by |
| --- | --- |
| `Restoration/<capture>/` | `App/Processing/II-Restoration/grape_restore_autoref.py`: the fruit-only image, its masks, provenance maps and `diagnostics.json` |
| `Evaluation/` | the counting and metric scripts of `App/Processing/III-Evaluation` |
| `Depth_Baselines/` | the COLMAP background fill and the cherry prototype |

Images and arrays are git-ignored; the small JSON and CSV result files are kept so the
numbers in the report can be checked without re-running anything.
