import os
import cv2
import numpy as np
import pillow_heif

raw_dir = "raw"
data_dir = "data"

for root, _, files in os.walk(raw_dir):
    output_dir = os.path.join(data_dir, os.path.relpath(root, raw_dir))
    for filename in files:
        stem, ext = os.path.splitext(filename)
        if ext.lower() != ".heic":
            continue

        output_path = os.path.join(output_dir, "images", stem + ".png")
        if os.path.exists(output_path):
            continue
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img = np.asarray(
            pillow_heif.open_heif(
                os.path.join(root, filename),
                convert_hdr_to_8bit=True,
                bgr_mode=True,
            )
        )
        cv2.imwrite(output_path, img)
