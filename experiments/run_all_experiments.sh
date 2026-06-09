#!/usr/bin/env bash
# run_all_experiments.sh — LIGHT experiments (< 4 min wall time)
# Runs the four decisive experiments at modest scale with a fixed seed.
# Writes results/*.json and figures to ../paper/figures/.

set -euo pipefail

# Navigate to the experiments directory (so relative paths work)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

echo "=========================================="
echo " Distributed Sub-Threshold Sabotage"
echo " Frontier -- Light Experiments"
echo "=========================================="
echo " Code dir: ${CODE_DIR}"
echo " Started: $(date)"
echo ""

SEED=42
T=300
N_TRAJ=500

# Create results and figures directories if needed
mkdir -p "${CODE_DIR}/results"
mkdir -p "${CODE_DIR}/results/figures"
mkdir -p "${CODE_DIR}/../paper/figures"

echo "[1/4] Realizability: KS-vs-harm Pareto frontier"
python "${CODE_DIR}/experiments/exp_realizability.py" \
    --seed ${SEED} --T ${T} --n_traj ${N_TRAJ}
echo ""

echo "[2/4] Monitor A vs B: ROC/AUC comparison"
python "${CODE_DIR}/experiments/exp_monitor_ab.py" \
    --seed ${SEED} --T ${T} --n_traj ${N_TRAJ}
echo ""

echo "[3/4] Power vs spread: THE money figure"
python "${CODE_DIR}/experiments/exp_power_vs_spread.py" \
    --seed ${SEED} --T ${T} --n_traj ${N_TRAJ}
echo ""

echo "[4/5] Marginal check: KS distance validation"
python "${CODE_DIR}/experiments/exp_marginal_check.py" \
    --seed ${SEED} --T ${T} --n_traj ${N_TRAJ}
echo ""

echo "[5/5] Adaptive red-teamer: detectability frontier + base-rate robustness"
python "${CODE_DIR}/experiments/exp_adaptive_redteam.py" \
    --seed ${SEED} --T ${T} --n_traj ${N_TRAJ}
echo ""

echo "=========================================="
echo " All experiments complete."
echo " Results: ${CODE_DIR}/results/"
echo " Figures: ${CODE_DIR}/../paper/figures/"
echo " Finished: $(date)"
echo "=========================================="

# Print a brief summary of key numbers
echo ""
echo "--- Key Results Summary ---"
python "${CODE_DIR}/experiments/summarize_results.py" "${CODE_DIR}/results"
