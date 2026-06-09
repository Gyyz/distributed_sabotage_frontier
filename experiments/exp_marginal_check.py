"""exp_marginal_check.py — KS distance check of attack vs benign marginals.

Key claim:
  - Marginal-preserving attacker: KS ~0 at all harm/stealth/spread settings
  - Naive attacker: KS grows with score_lift (easily detectable by KS test)
  - Semi-stealthy attacker: KS ~0 at large spread, grows at small spread

Writes: results/marginal.json
Figures: paper/figures/fig_marginal
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
from src.attack import marginal_preserving_attack, naive_attack, semi_stealthy_attack, ks_distance_to_benign


def main(seed: int = 42, T: int = 300, n_traj: int = 500):
    model = default_benign_model(mean=0.286, var=0.025, phi=0.5)
    harm_budget = 1.5

    # --- Stealthy attack: vary stealth and spread ---
    stealth_values = [0.2, 0.4, 0.6, 0.8, 0.9]
    spread_values = [T // 4, T // 2, T]

    results_stealthy = []
    for stealth in stealth_values:
        for sp in spread_values:
            trajs = marginal_preserving_attack(
                model, n_traj, T,
                harm_budget=harm_budget, stealth=stealth, spread=sp,
                rng=np.random.default_rng(seed + int(stealth * 100) + sp),
            )
            ks = ks_distance_to_benign(trajs, model,
                                       rng=np.random.default_rng(seed + 999))
            results_stealthy.append({
                "stealth": float(stealth), "spread": int(sp), "ks_distance": ks,
            })
            print(f"  stealthy stealth={stealth:.1f} spread={sp:4d}: KS={ks:.4f}")

    # --- Semi-stealthy: vary spread (shows KS grows as spread shrinks) ---
    spreads = [1, 2, 5, 10, 20, 50, 100, T]
    results_semi = []
    for sp in spreads:
        trajs = semi_stealthy_attack(
            model, n_traj, T,
            harm_budget=harm_budget, stealth=0.8, spread=sp,
            rng=np.random.default_rng(seed + sp * 13),
        )
        ks = ks_distance_to_benign(trajs, model, rng=np.random.default_rng(seed + 888))
        results_semi.append({"spread": int(sp), "ks_distance": ks})
        print(f"  semi-stealthy spread={sp:4d}: KS={ks:.4f}")

    # --- Naive attack: vary score_lift ---
    lifts = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25]
    results_naive = []
    for lift in lifts:
        trajs = naive_attack(
            model, n_traj, T,
            harm_budget=harm_budget, score_lift=lift,
            rng=np.random.default_rng(seed + 77),
        )
        ks = ks_distance_to_benign(trajs, model, rng=np.random.default_rng(seed + 888))
        results_naive.append({"score_lift": float(lift), "ks_distance": ks})
        print(f"  naive lift={lift:.2f}: KS={ks:.4f}")

    output = {
        "stealthy": results_stealthy,
        "semi_stealthy": results_semi,
        "naive": results_naive,
        "config": {"T": T, "n_traj": n_traj, "seed": seed},
    }

    results_dir = os.path.join(CODE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "marginal.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  wrote {json_path}")

    # --- Figure ---
    def make_fig():
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        colors = plotstyle.CYCLE

        # Left: stealthy KS vs stealth (per spread)
        ax = axes[0]
        for j, sp in enumerate(spread_values):
            ks_vals = [r["ks_distance"] for r in results_stealthy if r["spread"] == sp]
            ax.plot(stealth_values, ks_vals, "o-", color=colors[j], label=f"spread={sp}")
        ax.axhline(0.05, color="gray", linestyle=":", linewidth=1.0, label="KS=0.05")
        ax.set_xlabel("Stealth coefficient"); ax.set_ylabel("KS distance")
        ax.set_title("Stealthy: KS vs. Stealth\n(marginal exactly preserved)")
        ax.legend(fontsize=7)

        # Middle: semi-stealthy KS vs spread
        ax = axes[1]
        sp_arr = [r["spread"] for r in results_semi]
        ks_arr = [r["ks_distance"] for r in results_semi]
        ax.plot(sp_arr, ks_arr, "o-", color=colors[0], label="Semi-stealthy")
        ax.axhline(0.05, color="gray", linestyle=":", linewidth=1.0, label="KS=0.05")
        ax.set_xlabel("Spread (# harm steps)"); ax.set_ylabel("KS distance")
        ax.set_title("Semi-Stealthy: KS vs. Spread\n(KS grows as spread shrinks)")
        ax.legend(fontsize=7)

        # Right: naive KS vs score lift
        ax = axes[2]
        lifts_arr = [r["score_lift"] for r in results_naive]
        ks_naive = [r["ks_distance"] for r in results_naive]
        ax.bar([f"{l:.2f}" for l in lifts_arr], ks_naive,
               color=plotstyle.COLORS["red"], alpha=0.75)
        best_s = min(r["ks_distance"] for r in results_stealthy)
        ax.axhline(best_s, color=plotstyle.COLORS["blue"], linestyle="--",
                   linewidth=1.5, label=f"Best stealthy KS={best_s:.3f}")
        ax.set_xlabel("Score lift"); ax.set_ylabel("KS distance")
        ax.set_title("Naive: KS vs. Score Lift\n(marginal visibly shifted)")
        ax.legend(fontsize=7)

        fig.tight_layout()
        return fig

    for fig_dir in [
        os.path.join(CODE_DIR, "..", "paper", "figures"),
        os.path.join(CODE_DIR, "results", "figures"),
    ]:
        os.makedirs(fig_dir, exist_ok=True)
        fig = make_fig()
        plotstyle.save_fig(fig, os.path.join(fig_dir, "fig_marginal"))

    print(f"\nMarginal check summary:")
    print(f"  Best stealthy KS: {min(r['ks_distance'] for r in results_stealthy):.4f}")
    print(f"  Worst stealthy KS: {max(r['ks_distance'] for r in results_stealthy):.4f}")
    print(f"  Naive max KS: {max(r['ks_distance'] for r in results_naive):.4f}")
    print(f"  Semi-stealthy large-spread KS: "
          f"{min(r['ks_distance'] for r in results_semi if r['spread'] >= T//2):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--n_traj", type=int, default=500)
    args = parser.parse_args()
    main(args.seed, args.T, args.n_traj)
