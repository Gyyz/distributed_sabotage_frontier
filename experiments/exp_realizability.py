"""exp_realizability.py — KS-vs-harm Pareto frontier experiment.

Sweeps harm budget for both the stealthy (marginal-preserving) attacker and
the naive (score-lifting) attacker.  For each:
  - Measures KS distance of per-step score marginal vs. benign
  - Records achieved harm

Writes: results/realizability.json
Figures: paper/figures/fig_realizability
"""
import sys
import os
import json
import argparse
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

import plotstyle
plotstyle.apply()

import matplotlib.pyplot as plt
from src.benign import default_benign_model
from src.attack import marginal_preserving_attack, naive_attack, ks_distance_to_benign


def main(seed: int = 42, T: int = 300, n_traj: int = 500):
    model = default_benign_model(mean=0.286, var=0.025, phi=0.5)

    harm_values = np.linspace(0.0, 3.0, 16)

    print("Running stealthy attacker realizability sweep...")
    stealthy_results = []
    for h in harm_values:
        trajs = marginal_preserving_attack(
            model, n_traj, T,
            harm_budget=h, stealth=0.2, spread=T // 2,
            rng=np.random.default_rng(seed),
        )
        ks = ks_distance_to_benign(trajs, model, rng=np.random.default_rng(seed + 100))
        harms = [t.total_harm for t in trajs]
        stealthy_results.append({
            "harm_budget": float(h),
            "ks_distance": ks,
            "mean_harm": float(np.mean(harms)),
            "std_harm": float(np.std(harms)),
        })
        print(f"  stealthy harm={h:.2f}: KS={ks:.4f}")

    print("Running naive attacker realizability sweep...")
    naive_results = []
    lifts = np.linspace(0.0, 0.40, 16)
    for i, (h, lift) in enumerate(zip(harm_values, lifts)):
        trajs = naive_attack(
            model, n_traj, T,
            harm_budget=h,
            score_lift=float(lift),
            rng=np.random.default_rng(seed + 1),
        )
        ks = ks_distance_to_benign(trajs, model, rng=np.random.default_rng(seed + 200))
        harms = [t.total_harm for t in trajs]
        naive_results.append({
            "harm_budget": float(h),
            "ks_distance": ks,
            "mean_harm": float(np.mean(harms)),
            "score_lift": float(lift),
        })

    output = {
        "stealthy": stealthy_results,
        "naive": naive_results,
        "config": {"T": T, "n_traj": n_traj, "seed": seed},
    }

    results_dir = os.path.join(CODE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "realizability.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  wrote {json_path}")

    # --- Figure ---
    def make_fig():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        ax = axes[0]
        h_s = [r["harm_budget"] for r in stealthy_results]
        ks_s = [r["ks_distance"] for r in stealthy_results]
        h_n = [r["harm_budget"] for r in naive_results]
        ks_n = [r["ks_distance"] for r in naive_results]

        ax.plot(h_s, ks_s, "o-", label="Stealthy (marginal-preserving)",
                color=plotstyle.COLORS["blue"])
        ax.plot(h_n, ks_n, "s--", label="Naive (score-lifting)",
                color=plotstyle.COLORS["red"])
        ax.axhline(0.05, color="gray", linestyle=":", linewidth=1.0, label="KS=0.05")
        ax.set_xlabel("Harm budget per trajectory")
        ax.set_ylabel("KS distance to benign marginal")
        ax.set_title("KS Distance vs. Harm Budget")
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.scatter(ks_s, h_s, marker="o", s=60, label="Stealthy",
                   color=plotstyle.COLORS["blue"])
        ax.scatter(ks_n, h_n, marker="s", s=60, label="Naive",
                   color=plotstyle.COLORS["red"])
        ax.axvline(0.05, color="gray", linestyle=":", linewidth=1.0, label="KS=0.05")
        ax.set_xlabel("KS distance (lower = more stealthy)")
        ax.set_ylabel("Harm budget")
        ax.set_title("Realizability Gate: KS vs. Harm Pareto Frontier")
        ax.legend(fontsize=8)

        fig.tight_layout()
        return fig

    for fig_dir in [
        os.path.join(CODE_DIR, "..", "paper", "figures"),
        os.path.join(CODE_DIR, "results", "figures"),
    ]:
        os.makedirs(fig_dir, exist_ok=True)
        fig = make_fig()
        plotstyle.save_fig(fig, os.path.join(fig_dir, "fig_realizability"))

    stealthy_low_ks = [(r["harm_budget"], r["ks_distance"])
                       for r in stealthy_results if r["ks_distance"] < 0.05]
    print(f"\nRealizability summary:")
    print(f"  Stealthy: {len(stealthy_low_ks)} of {len(stealthy_results)} harm levels have KS < 0.05")
    if stealthy_low_ks:
        print(f"  Max stealthy harm at KS < 0.05: {max(h for h, _ in stealthy_low_ks):.2f}")
    print(f"  Naive max KS: {max(r['ks_distance'] for r in naive_results):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--n_traj", type=int, default=500)
    args = parser.parse_args()
    main(args.seed, args.T, args.n_traj)
