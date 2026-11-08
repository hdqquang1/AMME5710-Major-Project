"""
label_points.py - click the centre of every fruit in a photo to make ground truth.

    python3 label_points.py path/to/centre_photo_without_foliage.jpg
    python3 label_points.py photo.jpg --out labels_ref.csv

Left click adds a point, right click removes the nearest, U undoes, S saves,
Q or Esc quits (asks nothing; save first).  Points are stored in FULL-resolution
pixels of the photo (x_px, y_px), the format run_real.py --labels reads.  Label
the centre photo of the foliage-free pass: it shows every countable fruit.

Dependencies: numpy, opencv-python.
"""

import argparse
import csv
import os

import numpy as np
import cv2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--out", default=None, help="CSV path (default: <image>_labels.csv)")
    ap.add_argument("--display-width", type=int, default=1400)
    args = ap.parse_args()
    im = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if im is None:
        raise SystemExit(f"cannot read {args.image}")
    out = args.out or os.path.splitext(args.image)[0] + "_labels.csv"
    s = min(1.0, args.display_width / im.shape[1])
    disp = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    pts, history = [], []
    if os.path.exists(out):
        pts = [(float(r["x_px"]), float(r["y_px"])) for r in csv.DictReader(open(out))]
        print(f"loaded {len(pts)} points from {out}")

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            history.append(list(pts))
            pts.append((x / s, y / s))
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            history.append(list(pts))
            k = int(np.argmin([(px * s - x) ** 2 + (py * s - y) ** 2 for px, py in pts]))
            pts.pop(k)

    win = "label fruit: left add, right remove, U undo, S save, Q quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        canvas = disp.copy()
        for k, (px, py) in enumerate(pts, 1):
            c = (int(px * s), int(py * s))
            cv2.circle(canvas, c, 7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, str(k), (c[0] + 9, c[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{len(pts)} fruit", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("s"), ord("S")):
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["x_px", "y_px"])
                w.writerows([(round(x, 1), round(y, 1)) for x, y in pts])
            print(f"saved {len(pts)} points to {out}")
        elif key in (ord("u"), ord("U")) and history:
            pts[:] = history.pop()
        elif key in (ord("q"), ord("Q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
