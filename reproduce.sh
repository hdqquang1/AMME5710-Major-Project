#!/usr/bin/env bash
# Reproduce what this machine can. Run from anywhere.
#   ./reproduce.sh              check the report's tables against results/ and rebuild the ablation
#                               tables and plots from the committed results (no data, no GPU)
#   ./reproduce.sh --synthetic  also run has_sai's 44-scene synthetic benchmark (about 20 min on 3 cores)
#   ./reproduce.sh --full       also re-run the whole study: needs data/ (see data/README.md), the
#                               submodules and weights (third_party/README.md) and a CUDA GPU
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install --quiet -r requirements.txt
fi
P="$PWD/.venv/bin/python"

echo "== 1. The report's Tables 1-3 and quoted numbers against results/"
"$P" tools/check_report_numbers.py

echo "== 2. results/ablation/assets rebuilt from the committed per-run results"
"$P" make_ablation_assets.py
git diff --stat -- results/ablation/assets 2>/dev/null | tail -n 1 || true

case "${1:-}" in
  --synthetic)
    echo "== 3. has_sai synthetic benchmark"
    "$P" has_sai/run_synthetic_benchmark.py
    ;;
  --full)
    T=data/20261022_034827542_iOS.jpg
    M=results/refactor_check/truth/t1/grape_masks/t1.png
    W=third_party/weights/point/best_val.pth
    for f in "$T" "$M" "$W"; do
      [ -f "$f" ] || { echo "missing $f: see data/README.md and third_party/README.md"; exit 1; }
    done
    for b in 3 4 5 6; do
      d=data/2cm_74cm_${b}branches
      [ -d "$d/images" ] || { echo "missing $d/images: see data/README.md"; exit 1; }
      if [ ! -d "$d/grape_masks" ] || [ ! -d "$d/removal_masks" ]; then
        echo "== SAM 3 masks for $d"
        PYTHONPATH=third_party/sam3 "$P" App/Segmentation/make_masks.py --input "$d/images" --output "$d" --grape-masks
      fi
    done
    echo "== 3. Main study (Table 1), about 4 min per reconstruction"
    "$P" run_experiments.py --truth "$T" --truth-grapes "$M" --labels labels/t1.csv --weights "$W"
    "$P" analyse_experiments.py
    echo "== 4. Baselines, ablations and uncertainty (Tables 2-3, Figs. 5-6)"
    "$P" run_ablations.py launch
    "$P" run_ablations.py a1 a2 a34 a5 a6 a7 a8
    "$P" make_ablation_assets.py
    ;;
esac
echo "Done."
