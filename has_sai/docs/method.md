# Occlusion-aware synthetic aperture imaging and colour-free fruit counting

Method notes for Has's branch (`Has/occlusion-aware-sai`).  The code is in
`has_sai/sai/`; every stage below names the function that implements it.

## 1. Where this sits in the team

| | Prototype on `finns-branch` | Quang (`main`) | This branch |
|---|---|---|---|
| Poses | nominal rail shifts + SIFT vertical drift | COLMAP | COLMAP model, or OpenCV rail SfM (no COLMAP needed) |
| Occluders found by | HSV green threshold | depth threshold on the dense cloud | multi-view photo-consistency (depth), no colour |
| Leaf-free image | mean of views shifted to ONE depth | splat the thresholded cloud | per-pixel depth, visibility-weighted, all in focus |
| Detection | Lab/RGB red thresholds + contours | (planned YOLO) | depth relief + radial symmetry, no colour |
| Counting | 2-D contours, optional tracks | (planned) | 3-D positions, 3-D non-maximum suppression |
| Green fruit | fails by design | depends on detector | same code path |

The tutor's advice ("synthetic aperture imaging is the way to go") is followed:
every stage is a synthetic-aperture refocus, a plane sweep through the stack of
rail photos.  What is new relative to classic synthetic aperture is that
visibility is decided per view from geometry, so the reconstruction does not
depend on the median (which fails once a point is hidden in half the photos,
Vaish *et al.* 2006) or on colour.

## 2. Capture and notation

Photos $I_1 \dots I_N$ (13 planned, 12 in the first capture) are taken $B = 50$ mm
apart along the rail.  View $i$ maps world points by $\mathbf{X}_i = R_i\mathbf{X} + \mathbf{t}_i$
with intrinsics $K_i$; the world frame is the centre (reference) camera $r$.
Processing runs at 1/4 resolution (1008 x 756 px, $f \approx 725$ px).

## 3. Camera poses (`camera.rig_from_colmap`, `rail_sfm.estimate_rail_poses`)

A COLMAP sparse model is used when present; its scale is fixed by setting the
mean distance between consecutive camera centres to $B$.  Without COLMAP the
OpenCV rail SfM runs: SIFT, ratio test and mutual check, essential-matrix
RANSAC per pair, union-find tracks, initialisation from the widest pair through
the reference, PnP registration of the rest.

Rail prior.  A rigid carriage on a straight rail does not rotate, so after
initialisation every $R_i = I$.  The projection equations then become linear in
all unknowns (points $\mathbf{X}_j$ and centres $\mathbf{C}_i$):

$$
\begin{bmatrix} f_x & 0 & -(u_{ij}-c_x) \\ 0 & f_y & -(v_{ij}-c_y)\end{bmatrix}(\mathbf{X}_j - \mathbf{C}_i) = \mathbf{0}.
$$

Each equation is weighted by the inverse depth from the previous iterate so the
residual approximates pixels; points are eliminated with the Schur complement
and the $3N$ camera unknowns are solved in one linear system with the gauge
fixed ($\mathbf{C}_r = 0$) and the scale fixed by the rail length.  Outliers
above 2 px are dropped between iterations (`rail_sfm.global_translation_ls`).
On a synthetic capture with exact ground truth: centre error at most 0.41 mm,
reprojection RMSE 0.31 px, spacing 50.00 +/- 0.03 mm.  Free rotations instead
gave 0.3 to 0.65 degree spurious rotations (the rotation-translation ambiguity of
a short lateral baseline), i.e. about 6 px misregistration on the fence.

## 4. Refocusing warp

The fronto-parallel plane $Z = d$ of the reference frame induces

$$
\mathbf{p}_i \sim K_i\left(R_i + \mathbf{t}_i\mathbf{n}^\top / d\right)K_r^{-1}\mathbf{p}, \qquad \mathbf{n} = [0,0,1]^\top,
$$

(`camera.plane_homography`, applied with `cv2.warpPerspective`).  Planes are
spaced uniformly in inverse depth so consecutive planes shift the widest
baseline by a fixed number of pixels (`planesweep.inverse_depth_planes`).

## 5. Colour-free foliage masks (`planesweep.foreground_masks`)

For every view $i$, a plane sweep against 6 partners within three rail steps
computes, per plane, the truncated absolute colour difference to each partner
warped through that plane, box-aggregated over 5 x 5 px.  Background pixels of
view $i$ are hidden by leaves in some partners, so only the best half of the
partner costs is averaged (temporal selection, Kang, Szeliski and Chai 2001).
The arg-min, refined by a parabola, is the depth map of view $i$ (computed at
half resolution, views in parallel threads).

The split depth between foliage and background is Otsu's threshold of the
reference view's inverse-depth histogram (inverse depth is proportional to
parallax, so the two layers are two modes).  It is clamped to lie at least
$w_f + 10$ mm in front of the wall ($w_f = 70$ mm, how far fruit can hang in
front of the fence), so fruit is never masked and a capture without foliage
gives empty masks.  Mask $M_i$ = depth nearer than the split, opened 3 x 3,
upsampled and dilated 2 px (fuzzy pipe-cleaner rims).

No colour is used.  On the synthetic benchmark the reference masks have
recall 0.99 and IoU 0.73 against the true foliage (they are deliberately fat:
a missed leaf pixel leaks leaf colour, an extra masked pixel only costs one
sample).

## 6. Background depth with visibility (`render.background_cost_volume`, `sgm.py`)

Visibility of a reference pixel $\mathbf{p}$ at depth $d$ in view $j$:
$v_j(\mathbf{p},d) = 1 - M_j(H_j(d)\,\mathbf{p})$ (zero outside photo $j$).

Cost of plane $d$: the unbiased truncated colour variance of the visible samples,

$$
c(\mathbf{p},d) = \frac{\sum_j v_j \min\left(\lVert s_j - \mu\rVert^2, \tau^2\right)}{\sum_j v_j - 1},\qquad
\mu = \frac{\sum_j v_j s_j}{\sum_j v_j},
$$

with $s_j = I_j(H_j(d)\mathbf{p})$, $\tau = 0.15$, and maximal cost where fewer
than two views see the pixel.

* Wall prior.  The sweep only spans $[z_w - 0.07, z_w + 0.05]$ m, with $z_w$ the
  median background depth: fruit hangs on the fence.  Without it, weakly
  observed pixels settled on spurious depths up to 130 mm in front of the
  wall, which looked like fruit (396 false detections on one scene).
* Half-occlusion.  Fruit 20 to 40 mm in front of the backdrop hides the backdrop
  beside it from the views on one side only.  The cost is also computed from
  the left and right halves of the rail and the smallest of the three is kept;
  the unbiased variance keeps full and half apertures comparable.
* Semi-global aggregation (Hirschmueller 2008) over four paths with
  $P_1 = 0.08$, $P_2 = 0.6/(1 + 8\lvert\Delta g\rvert)$ on the normalised cost,
  guide $g$ = visibility-masked mean refocused on the wall.  Lattice depth error
  fell from 2.8 mm to 0.6 mm (MAD) and the share of lattice pixels more than
  10 mm off from 27 % to 4 %.

Two depth maps come from one pass: the half-selected one (sharp fruit edges)
for detection and the full-aperture one for the image.

## 7. All-in-focus leaf-free image (`render.render_all_in_focus`)

Each reference pixel is reprojected into every view at its own depth,
$\mathbf{x}_j = K_j(R_j Z(\mathbf{p})K_r^{-1}\mathbf{p} + \mathbf{t}_j)$, with one `cv2.remap`
per view, and the visible samples are averaged; one reweighting pass drops
samples beyond 2.5 standard deviations (specular highlights, leaf pixels that
slipped past the masks).  Fruit at different depths stays sharp, unlike a
single focal plane.  Pixels no view sees are flagged and only inpainted for
display (Telea 2004).

## 8. Colour-free detection (`detect.detect_fruit`)

1. Untrusted depths (fewer than 3 views, confidence $1 - c_{min}/\bar c < 0.15$)
   are inpainted, then median filtered 5 x 5.
2. Directional relief.  With a line structuring element $\ell_\theta$ of length
   $L = 10R_{max}$ (in pixels at the wall depth) and 12 orientations,

   $$
   h(\mathbf{p}) = \min_\theta\left[(Z \bullet \ell_\theta)(\mathbf{p}) - Z(\mathbf{p})\right]_+ ,
   $$

   where $\bullet$ is grey-level closing.  Closing along $\theta$ fills every
   protrusion shorter than $L$ along $\theta$, so slats, wires, stems and
   lattice crossings (each long along some $\theta$) give $h = 0$, and compact
   fruit gives its height above whatever is behind it, in metres.
3. Scale-adapted blobs.  The fruit radius range $[R_{min}, R_{max}]$ and the
   depth give the pixel radius $r = fR/Z$; the scale-normalised LoG
   $-\sigma^2\nabla^2(G_\sigma * h)$, $\sigma = r/\sqrt2$, is evaluated at four
   radii.  Its local maxima with $R_{min}\cdot0.4 < h < w_f + 10$ mm are candidates.
4. Radial symmetry (Loy and Zelinsky 2003) of the luminance at the same radii:
   fruit touching in a row forms one relief bar with no interior maximum, but
   each sphere's rim votes for its own centre.  Radial-symmetry maxima inside
   the relief-gated region are added as candidates.
5. Each candidate is back-projected to 3-D (centre = surface point + $R$ along
   the ray) and greedy 3-D non-maximum suppression within one radius keeps one
   detection per fruit (twins, 2R apart, survive).
6. Score = relief LoG response [mm]; the threshold is calibrated on separate
   calibration scenes (best pooled F1) and applied unchanged to the test scenes
   and the real capture.

## 9. Evaluation protocol (`evaluate.py`, `run_synthetic_benchmark.py`)

* Countable fruit: centre in the ROI and at least 25 % of its disc visible in
  the foliage-free centre view.
* A detection is correct within 0.8 fruit radii of a countable fruit's
  centre (one-to-one greedy matching).  Precision, recall, F1, count and
  counting error $(\hat n - n)/n$.
* Image: PSNR and SSIM inside the ROI against the foliage-free photo.
* Masks: recall, precision, IoU against the true foliage; depth: median absolute
  error by surface class.
* Scenes: cherries (10/30/50 % foliage, gap 0.15/0.34 m), green grapes and limes;
  seed 0 calibrates, seed 1 tests.  Robustness: 3 to 13 photos, pose noise
  0.1 to 0.5 degrees with 1 mm, focal length +/-3 %.
* Baselines: centre photo; the prototype colour pipeline as published and with
  its area cap calibrated; median synthetic aperture with the colour-free 2-D
  detector; ablations of each stage; an oracle (foliage-free photo, true depth).

## 10. Libraries

OpenCV (features, matching, essential matrix, PnP, warps, remap, morphology,
inpainting, Otsu), NumPy, Matplotlib (figures).  Nothing else: no SciPy,
PyTorch, Open3D or COLMAP is required (a COLMAP model is used when present).

## 11. Known limits

* A fruit hidden in every photo cannot be recovered; the benchmark reports how
  many countable fruit are at least half visible in two or more photos.
* The wall prior assumes the fruit hangs within 70 mm of a roughly fronto-parallel
  fence; a deep canopy needs a wider search range.
* Untextured backdrop regions take the lattice depth under SGM (backdrop depth
  error about 20 mm); this does not create protrusions but limits backdrop PSNR.
* Glossy fruit breaks Lambertian photo-consistency at highlights; truncation
  and the robust render limit the damage.
