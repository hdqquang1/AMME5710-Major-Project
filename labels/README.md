# Labels

[t1.csv](t1.csv) holds the 25 grape centres clicked on the full-resolution leaf-free
photograph (`data/20261022_034827542_iOS.jpg`, 3024 x 4032 px, portrait), one row per berry:
`id, x_px, y_px`. The click-to-label tool in [../grape_evaluate.py](../grape_evaluate.py)
writes this format:

```bash
python3 grape_evaluate.py label --image data/20261022_034827542_iOS.jpg --out labels/t1.csv
```

The evaluation maps them into each reconstruction's frame through the truth registration and
matches detections one-to-one, closest first, within 24 px at the 0.8 processing scale (about
one berry radius). The labels mark berries visible in the leaf-free photograph, so 25 is the
number of visible berries in that view, not a count of every berry on the branches.
