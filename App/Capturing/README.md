# Capturing: the rail rig and the photo conversion

## Rig

A smartphone sits in a holder on a horizontal rail inside a cardboard enclosure (Fig. 2 of the
report). Red grapes hang on branches at the back of the enclosure; removable foreground
branches with leaves hang in front of them. A lamp lights the scene and stays fixed during a
sequence. The rail stands in for a vehicle driving along a vine row.

## Procedure for one capture

1. Hang the fruit and choose the number of foreground branches (the report used 3, 4, 5 and 6).
2. Lock focus, exposure and white balance on the phone. The consensus test compares raw RGB
   between views, so exposure changes between photos reject true fruit.
3. Slide the carriage to one end. Take a photo every 20 mm for 28 positions (27 at six
   branches). Do not touch the scene between photos.
4. Remove every foreground branch and take the leaf-free photograph of the same fruit. It is the
   truth for the pixel metrics and the image on which the 25 grape centres were labelled.
5. Copy the HEIC files into `raw/<capture>/` and convert them:

```bash
python3 convert_heic.py
```

`convert_heic.py` walks `raw/` and writes `data/<capture>/images/<name>.png` for every HEIC
(8-bit, EXIF orientation applied by `pillow-heif`), skipping files that already exist. Run it
from the repository root. Photos that are already PNG or JPEG can be placed in
`data/<capture>/images/` directly.

The 40 and 60 mm spacings in the report are subsets of the 20 mm captures (every second or
third photo around the same reference), not separate captures. See
[../../Captures/README.md](../../Captures/README.md) for the folder layout and where the photos
are kept.
