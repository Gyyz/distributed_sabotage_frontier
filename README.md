# The Distributed Sub-Threshold Sabotage Frontier

Code for the paper:  
**"The Distributed Sub-Threshold Sabotage Frontier: A Realizability Gate, a Monitor-A-vs-B Reconciliation, and a Demonstrated Detectability Band Against Marginal-Preserving Attacks"**

## Scientific Summary

AI-control monitors score agent actions; real harm can be *distributed* across many benign-looking steps. We construct a **marginal-preserving, correlation-encoded distributed-sabotage attack** that:

- Keeps the per-step monitor-score **marginal** identical to benign (mean/max/tail/threshold monitors — "Monitor A" — cannot detect it, by construction)
- Encodes harm in the **temporal/correlation structure**, which a sequence/correlation monitor ("Monitor B": CUSUM/SPRT/HMM/autocorrelation/runs-test) **can** catch

The central finding: as harm is spread over more steps, Monitor A degrades toward chance while Monitor B maintains high detection power.

## Repository Layout

```
code/
  src/
    benign.py       — Benign agent trajectory model (Beta/truncnorm, AR(1))
    attack.py       — Marginal-preserving attacker + naive baseline
    monitors.py     — Monitor A (marginal) + Monitor B (sequential)
    frontier.py     — Realizability gate + detectability band analysis
  experiments/
    exp_realizability.py    — KS-vs-harm Pareto frontier
    exp_monitor_ab.py       — ROC/AUC: Monitor A vs B
    exp_power_vs_spread.py  — Detection power vs attack spread (money figure)
    exp_marginal_check.py   — KS distance validation
    run_all_experiments.sh  — LIGHT: all experiments < 4 min
    run_mess_compute_all.sh — HEAVY: large-N, adaptive red-teamer, real-data stub
  results/          — JSON outputs
  tests/            — Sanity checks (run via python tests/test_*.py)
  plotstyle.py      — Shared matplotlib styling
  requirements.txt
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run sanity checks
python tests/test_benign.py
python tests/test_attack.py
python tests/test_monitors.py

# Run all light experiments
bash experiments/run_all_experiments.sh
```

## Key Results (from light experiments)

| Metric | Value |
|--------|-------|
| Stealthy attacker KS distance | < 0.05 at all harm levels |
| Monitor A AUC (marginal-preserving) | ~0.50 (chance) |
| Monitor B AUC (marginal-preserving) | ~0.70–0.85 |
| Monitor A AUC (naive attack) | > 0.85 |
| Max harm at KS < 0.05 | Demonstrated non-zero |

## Construction Details

### Marginal-Preserving Attack (`src/attack.py`)

1. Draw T uniforms; map to Beta(a,b) scores (exact benign marginal)
2. Mix with AR(1) Gaussian-copula-correlated uniforms (weight = `stealth`)
3. Assign harm to top-`spread` steps by score rank (rank-coupling)

The per-step score *histogram* is preserved; only temporal structure changes.

**Knobs:**
- `stealth` ∈ (0,1]: strength of AR(1) correlation injected
- `spread` ∈ [1, T]: number of steps over which harm is amortized

### Monitor A — Marginal/Per-Step

- `a_mean`: mean score
- `a_max`: maximum score  
- `a_top_k_tail`: mean of top-k scores
- `a_threshold`: fraction of steps exceeding threshold

All blind to marginal-preserving attacks (by construction).

### Monitor B — Sequential/Correlation

- `b_cusum`: Page's CUSUM on standardised scores
- `b_sprt`: SPRT LLR (benign i.i.d. vs. AR(1) alternative)
- `b_hmm_lr`: 2-state Markov chain LLR
- `b_runs_test`: Wald-Wolfowitz runs test
- `b_autocorr`: Lag-1..5 autocorrelation sum
- `b_seq_clf`: Logistic regression on sliding-window features

## Environment

- Python 3.11
- Dependencies: numpy 1.26, scipy 1.13, scikit-learn 1.5, matplotlib 3.8
