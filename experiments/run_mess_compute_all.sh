#!/usr/bin/env bash
# run_mess_compute_all.sh — HEAVY experiments (may take hours)
#
# Extends the light experiments with:
#   1. Large-N sweeps at lower base rates (down to 0.1%)
#   2. Longer trajectories (T=1000, 2000)
#   3. More bootstrap repetitions for tight confidence intervals
#   4. Adaptive red-teamer loop (attacker optimizes against deployed Monitor B)
#   5. Stub for ingesting real agent-monitor traces
#
# NOTE: This script requires large compute.  On a single CPU core, expect
# 30-120 minutes depending on options.

set -euo pipefail

cd "$(dirname "$0")"
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=========================================="
echo " Distributed Sub-Threshold Sabotage"
echo " Frontier — HEAVY Experiments"
echo "=========================================="
echo " Started: $(date)"
echo ""

SEED=42

# ---------------------------------------------------------------------------
# 1. Large-N sweep at low base rates
# ---------------------------------------------------------------------------
echo "[1] Large-N / low base-rate sweep ..."
python "${CODE_DIR}/experiments/exp_monitor_ab.py" \
    --seed ${SEED} --T 500 --n_traj 2000
python "${CODE_DIR}/experiments/exp_power_vs_spread.py" \
    --seed ${SEED} --T 500 --n_traj 2000

# ---------------------------------------------------------------------------
# 2. Very long trajectories
# ---------------------------------------------------------------------------
echo "[2] Long trajectory experiments (T=1000) ..."
python "${CODE_DIR}/experiments/exp_monitor_ab.py" \
    --seed ${SEED} --T 1000 --n_traj 1000
python "${CODE_DIR}/experiments/exp_power_vs_spread.py" \
    --seed ${SEED} --T 1000 --n_traj 1000

# ---------------------------------------------------------------------------
# 3. Adaptive red-teamer loop
#    The attacker iteratively adjusts stealth/spread to minimise Monitor B
#    AUC while keeping Monitor A AUC low (evading detection).
# ---------------------------------------------------------------------------
echo "[3] Adaptive red-teamer loop ..."
python - << 'PYEOF'
import sys, os, numpy as np
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in dir() else \
    os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, CODE_DIR)

from src.benign import default_benign_model, sample_benign
from src.attack import marginal_preserving_attack
from src.monitors import score_trajectories, SequenceClassifier
from src.frontier import compute_auc
import json

model = default_benign_model(mean=0.25, var=0.02, phi=0.05)
T = 300
n_benign = 300
n_attack = 150
harm_budget = 1.5
seed = 42

# Train a fixed Monitor B (SPRT + autocorr) on default attack
rng = np.random.default_rng(seed)
benign_arr = sample_benign(model, n_benign, T, rng)
attack_trajs = marginal_preserving_attack(model, n_attack, T,
    harm_budget=harm_budget, stealth=0.8, spread=T//2, rng=rng)
b_list = [benign_arr[i] for i in range(n_benign)]
a_list = [t.scores for t in attack_trajs]

n_train = n_benign // 2
seq_clf = SequenceClassifier(window=min(20, T // 5))
seq_clf.fit(b_list[:n_train], a_list[:n_train])

print("Adaptive red-teamer: searching for optimal (stealth, spread) ...")
best_b_auc = 1.0
best_params = None

stealth_grid = [0.2, 0.4, 0.6, 0.8]
spread_grid  = [T//4, T//2, T]

adaptive_results = []
for stealth in stealth_grid:
    for spread in spread_grid:
        rng_i = np.random.default_rng(seed + int(stealth*100) + spread)
        b_arr = sample_benign(model, 200, T, rng_i)
        a_trajs = marginal_preserving_attack(model, 100, T,
            harm_budget=harm_budget, stealth=stealth, spread=spread,
            rng=rng_i)
        b_sc = score_trajectories([b_arr[i] for i in range(200)], model, seq_clf=seq_clf)
        a_sc = score_trajectories([t.scores for t in a_trajs], model, seq_clf=seq_clf)
        aucs = compute_auc(b_sc, a_sc)
        b_auc = aucs.get("b_seq_clf", aucs.get("b_sprt", 0.5))
        a_auc_mean = aucs.get("a_mean", 0.5)
        adaptive_results.append({
            "stealth": stealth, "spread": spread,
            "b_seq_clf_auc": float(b_auc), "a_mean_auc": float(a_auc_mean),
        })
        print(f"  stealth={stealth:.1f} spread={spread}: b_auc={b_auc:.3f} a_mean_auc={a_auc_mean:.3f}")
        if b_auc < best_b_auc and a_auc_mean < 0.55:
            best_b_auc = b_auc
            best_params = {"stealth": stealth, "spread": spread}

print(f"\nBest params for attacker (min Monitor B AUC with Monitor A at chance):")
print(f"  {best_params} -> Monitor B AUC = {best_b_auc:.3f}")

results_dir = os.path.join(CODE_DIR, "results")
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, "adaptive_redteamer.json"), "w") as fout:
    json.dump({
        "adaptive_results": adaptive_results,
        "best_params": best_params,
        "best_b_auc": best_b_auc,
    }, fout, indent=2)
PYEOF

# ---------------------------------------------------------------------------
# 4. Stub for ingesting real agent-monitor traces
#    Replace the TODO URLs with actual trace files from SHADE-Arena /
#    control-arena when available.
# ---------------------------------------------------------------------------
echo "[4] Real-trace ingestion stub (no-op without data) ..."
python - << 'PYEOF'
"""
Real agent-monitor trace ingestion stub.

TODO: Replace the placeholder URLs below with actual trace files from:
  - SHADE-Arena: https://github.com/redwoodresearch/SHADE-Arena
    (paper: https://arxiv.org/abs/2406.02875)
  - control-arena: https://github.com/UKGovernmentBEIS/control-arena
    (resource: https://www.aisi.gov.uk/work/control-arena)

Once real traces are available:
  1. Download JSON-line or CSV trace files to data/real_traces/.
  2. Each trace should be a sequence of (step_id, monitor_score, metadata) tuples.
  3. Run the fit_benign() function on confirmed-benign traces to calibrate the model.
  4. Run score_trajectories() on the full dataset to get per-trajectory monitor scores.
  5. Compute ROC/AUC against ground-truth labels (benign vs. attack).

Example placeholder:
  SHADE_ARENA_URL = "https://github.com/redwoodresearch/SHADE-Arena/raw/main/traces/..."
  CONTROL_ARENA_URL = "https://huggingface.co/datasets/ukaisi/control-arena/..."
"""
import os, sys
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) \
    if "__file__" in dir() else os.path.abspath(os.path.join(os.getcwd(), ".."))

data_dir = os.path.join(CODE_DIR, "data", "real_traces")
os.makedirs(data_dir, exist_ok=True)

stub_readme = os.path.join(data_dir, "README.txt")
with open(stub_readme, "w") as f:
    f.write(
        "Place real agent-monitor trace files here.\n"
        "Expected format: CSV with columns [traj_id, step, monitor_score, label]\n"
        "Sources:\n"
        "  SHADE-Arena: https://github.com/redwoodresearch/SHADE-Arena\n"
        "  control-arena: https://github.com/UKGovernmentBEIS/control-arena\n"
    )
print(f"  Stub README written to {stub_readme}")
print("  (No real traces available — skipping ingestion)")
PYEOF

echo ""
echo "=========================================="
echo " HEAVY experiments complete."
echo " Finished: $(date)"
echo "=========================================="

# ----------------------------------------------------------------------
# Real-data validation: live LLM agent + LLM monitor traces (GPU).
# Generates real per-step monitor-score trajectories then runs the
# marginal-preserving attack / Monitor-A-vs-B analysis on the real benign
# distribution. Requires: pip install torch transformers; a GPU; cached
# Qwen2.5-7B-Instruct + Llama-3.2-3B-Instruct. ~40 min.
# ----------------------------------------------------------------------
echo "[real-data] generating live agent/monitor traces ..."
python gpu/gen_real_traces.py --n_traj 120 --T 24 --out gpu/results/real_traces.json \
    && python experiments/exp_realdata_detect.py \
    || echo "  [SKIPPED] needs GPU + cached models for real-trace generation"
