# Council review of the occlusion-aware branch (15 October 2026)

A five-seat review with the Council of High Intelligence skill (full mode:
restatement, independent analysis, anonymised cross-examination, final
positions, weighted tally, chairman synthesis).  Members: Feynman (first
principles), Karpathy (empirical evaluation; domain-weight seat, 1.5x), Taleb
(tail risk), Kahneman (bias), Torvalds (shipping).  Single model provider, so
the provider spread is 1 of 5.

## Outcome

Unanimous (weighted 5.0 of 5.5, threshold 3.67, no dealbreakers):
**keep the geometric de-occlusion, get real captures first, claim only what
survives; write "in simulation" until sealed real results exist.**

## Kill criteria (recorded 15 October 2026)

- If `run_real.py` scores below F1 0.85, or misses the hand count by more than
  15 %, on a real foliage-free pass by 28 October 2026, the real-photo claim is
  invalidated; report simulation as ablations and the real run as failure analysis.
- If ours fails to beat median-SAI + 2-D beyond the seed-to-seed spread at 50 %
  cover on the frozen seeds 2-3, or at 0.25 degree pose noise, limit the
  simulation claim to localisation and image quality.
  **Status: triggered.** At 50 % cover ours scores 0.53 against 0.45; at 0.25 degree
  pose noise ours scores 0.00 against 0.12.
- If at least four sealed hangings, scored once on 29 October 2026, give a
  layout-level paired bootstrap interval for the F1 difference that includes zero
  against the strongest baseline (colour + distance transform, per-view voting,
  the prototype run by its owner), drop "beats the prototype" and report the tie.
- If colour + distance transform on classic synthetic aperture comes within 0.05
  F1 of ours on red and dark-purple cherries on 29 October, the geometric
  counting bet is invalidated; headline colour + distance transform and keep
  depth as an ablation.  **In simulation it is already 0.19 above ours on red
  cherries** (0.997 vs 0.812); the simulator cannot model colour failures, so
  the real test decides.

## Concrete next step

Run the frozen `run_real.py`, untuned, on the existing 12-photo real capture,
saving every intermediate map and comparing rail-SfM with COLMAP poses, by
17 October 2026.

## Recommended plan

1. Reproduce colour + distance transform, per-view voting, the render x detector
   factorial and a whole-frame ROI (done 15 October, `results/synthetic/factorial.md`),
   and test `run_real.py` on rotated, exposure-jittered scenes (done,
   `results/synthetic/realpath_study.csv`).
2. 18-22 October: shoot five or six hangings arranged by a teammate from a
   randomised grid, each with a clean and a foliage pass, AE/AF/WB locked, remote
   trigger, JPEG, dark-purple cherries included; one calibrates, the rest stay
   sealed and are labelled by someone other than Has.
3. 23-28 October: fix the three worst real failures on the calibration hanging
   only; use COLMAP poses if the rail-SfM RMSE exceeds 1 px; re-tag.
4. 29 October: score the sealed hangings once; report per-hanging F1, false
   positives and misses, red and dark-purple cherries separately.

## Points of disagreement

- Relief detector: headline only if it beats colour + distance transform with an
  interval excluding zero (Karpathy); keep but do not headline (Torvalds); demote
  only on a tie (Kahneman); untestable in a fixed-palette simulator (Taleb).
- Decision rule: a 0.03-0.05 F1 margin, 0.10 on three of four hangings, or wins
  on all of at least five hangings.
- PSNR may judge rendering, never counting: a 36.5 dB image sat beside a 0 of 28
  count before the SfM fix.

## Minority report

Karpathy gave about 40 % odds that depth is the wrong bet for counting red
cherries: classic synthetic aperture with colour + distance transform scored
1.00 on the simulated scenes while the geometric detector fails abruptly under
heavy cover.  Taleb's variant credits the 2-D detector and keeps depth for
localisation unless ours wins under heavy cover.

## What the review changed

- Detector frozen and tagged (`has-sai-freeze-1`) before the test seeds were read.
- The foliage-free control exposed a real bug: the rail SfM solved a badly scaled
  KKT system, the scale constraint was dropped and cameras collapsed onto the
  reference (count 0 of 28).  Fixed; the control now counts 29 for 28.
- The whole-image exposure gain was biased by foliage parallax (up to 5 % on
  photos with no exposure change); it is now opt-in and measured on the fence plane.
- Added: colour + distance-transform and per-view voting rivals, the factorial
  study, a whole-frame ROI, layout-level verifier folds, scenes with rendered
  carriage rotation and exposure drift scored through `run_real.py`.
