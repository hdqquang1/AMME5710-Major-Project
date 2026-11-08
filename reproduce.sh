#!/usr/bin/env bash
# Runs every stage whose inputs are present and prints SKIPPED with the reason for the rest.
# Run from anywhere: ./reproduce.sh            (all stages)
#                    ./reproduce.sh --synthetic (only the has_sai synthetic benchmark)
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install --quiet -r requirements.txt
fi
P="$PWD/.venv/bin/python"
export OPENCV_FOR_THREADS_NUM=1   # one pair decision can flip between runs when SIFT ties are threaded

if [ "${1:-}" = "--synthetic" ]; then
  echo "== has_sai synthetic benchmark (renders 44 scenes; about 20 min on 3 cores)"
  "$P" has_sai/run_synthetic_benchmark.py
  exit 0
fi

CAPS="2cm_74cm_3branches 2cm_74cm_4branches 2cm_74cm_5branches 2cm_74cm_6branches"
echo "== I. SAM 3 masks"
if "$P" -c "import torch, sam3" 2>/dev/null; then
  for c in $CAPS; do
    if [ -d "data/$c/images" ] && [ ! -d "data/$c/grape_masks" ]; then
      "$P" App/Processing/I-Segmentation/make_masks.py --input "data/$c/images" --output "data/$c" --overlay --grape-masks
    fi
  done
else
  echo "SKIPPED: PyTorch with CUDA and the sam3 package are not installed (see App/Processing/I-Segmentation/README.md)"
fi

echo "== II. Restoration (the method), 28 views at scale 0.8, about 4 min per capture"
for c in $CAPS; do
  if [ ! -d "data/$c/images" ]; then
    echo "SKIPPED $c: data/$c/images is missing (see Captures/README.md)"
  elif [ ! -d "data/$c/grape_masks" ]; then
    echo "SKIPPED $c: data/$c/grape_masks is missing; without SAM 3 masks the script would run the colour-rule ablation, not the method"
  else
    "$P" App/Processing/II-Restoration/grape_restore_autoref.py --images "data/$c/images" \
        --target-masks "data/$c/grape_masks" --scale 0.8 --spacing 20 --out "Out/Restoration/$c"
  fi
done

echo "== III. Counting, metrics and figures"
if ls App/Processing/III-Evaluation/*.py >/dev/null 2>&1; then
  echo "See App/Processing/III-Evaluation/README.md for the order of the evaluation scripts."
else
  echo "SKIPPED: the evaluation scripts are not in the repository yet (App/Processing/III-Evaluation/README.md)"
fi

echo "== IV. has_sai synthetic benchmark"
if [ -f has_sai/results/synthetic/summary.md ]; then
  echo "Results are committed in has_sai/results/synthetic/; run ./reproduce.sh --synthetic to regenerate them."
else
  "$P" has_sai/run_synthetic_benchmark.py
fi
echo "Done."
