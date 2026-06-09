"""exp_power_vs_spread.py — Detection power vs. attack spread (THE money figure).

Uses the semi-stealthy attack: harm is concentrated on `spread` steps,
with a score lift proportional to harm_budget/spread.

As spread increases:
  - Per-step harm -> smaller score lift -> Monitor A (a_max, a_threshold) FAILS
  - Monitor B (b_sprt, b_autocorr) detects elevated phi at ALL spreads

This demonstrates the central paper claim: Monitor B is robust to spread,
Monitor A degrades to chance as the attack distributes harm more.

Writes: results/power_vs_spread.json
Figures: paper/figures/fig_power_spread
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
from src.benign import default_benign_model, sample_benign
from src.attack import semi_stealthy_attack, marginal_preserving_attack
from src.monitors import score_trajectories, SequenceClassifier
from src.frontier import compute_auc, compute_detection_power_at_fpr


def main(seed: int = 42, T: int = 300, n_traj: int = 500):
    model = default_benign_model(mean=0.286, var=0.025, phi=0.5)
    n_benign = n_traj
    n_attack = n_traj // 2
    harm_budget = 2.0
    stealth = 0.2
    target_fpr = 0.01

    # Spread values: from very concentrated to fully distributed
    spreads = [1, 2, 5, 10, 20, 50, 100, 150, T]

    print(f"Power vs spread sweep (T={T}, n_benign={n_benign}, n_attack={n_attack})")

    all_results = []

    for sp in spreads:
        rng = np.random.default_rng(seed + sp * 7)

        benign_arr = sample_benign(model, n_benign, T, rng)
        attack_trajs = semi_stealthy_attack(
            model, n_attack, T,
            harm_budget=harm_budget, stealth=stealth, spread=sp,
            rng=rng,
        )

        benign_list = [benign_arr[i] for i in range(n_benign)]
        attack_list = [t.scores for t in attack_trajs]

        n_b_train = n_benign // 2
        n_a_train = n_attack // 2
        b_train, b_test = benign_list[:n_b_train], benign_list[n_b_train:]
        a_train, a_test = attack_list[:n_a_train], attack_list[n_a_train:]

        seq_clf = SequenceClassifier(window=min(20, T // 5), model=model)
        try:
            seq_clf.fit(b_train, a_train)
        except Exception:
            seq_clf = None

        benign_scores = score_trajectories(b_test, model, seq_clf=seq_clf)
        attack_scores = score_trajectories(a_test, model, seq_clf=seq_clf)

        all_names = list(benign_scores.keys())
        aucs = compute_auc(benign_scores, attack_scores, all_names)
        powers = {n: compute_detection_power_at_fpr(benign_scores[n], attack_scores[n], target_fpr)
                  for n in all_names}

        all_results.append({"spread": int(sp), "aucs": aucs, "powers": powers})

        print(f"  spread={sp:4d}: a_max={aucs.get('a_max', 0):.3f} "
              f"a_thresh={aucs.get('a_threshold', 0):.3f} | "
              f"b_sprt={aucs.get('b_sprt', 0):.3f} "
              f"b_autocorr={aucs.get('b_autocorr', 0):.3f}")

    # Save JSON
    output = {
        "results": all_results,
        "config": {
            "T": T, "n_benign": n_benign, "n_attack": n_attack,
            "harm_budget": harm_budget, "stealth": stealth,
            "target_fpr": target_fpr, "seed": seed,
        },
    }
    results_dir = os.path.join(CODE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "power_vs_spread.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  wrote {json_path}")

    # --- THE money figure ---
    spreads_arr = [r["spread"] for r in all_results]
    a_show = ["a_max", "a_threshold", "a_mean"]
    b_show = ["b_sprt", "b_autocorr", "b_cusum"]
    if any("b_seq_clf" in r["aucs"] for r in all_results):
        b_show.append("b_seq_clf")
    colors = plotstyle.CYCLE

    def make_fig():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: AUC vs spread
        ax = axes[0]
        for i, name in enumerate(a_show):
            vals = [r["aucs"].get(name, 0.5) for r in all_results]
            ax.plot(spreads_arr, vals, "o--", color=colors[i], label=name, alpha=0.85)
        for i, name in enumerate(b_show):
            vals = [r["aucs"].get(name, 0.5) for r in all_results]
            ax.plot(spreads_arr, vals, "s-", color=colors[len(a_show) + i],
                    label=name, linewidth=2.0)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, label="Chance")
        ax.set_xlabel("Attack spread (# steps with harm)")
        ax.set_ylabel("ROC AUC")
        ax.set_title("AUC vs. Attack Spread\n(Monitor A collapses; Monitor B holds)")
        ax.set_ylim([0.3, 1.05])
        ax.legend(fontsize=7)

        # Right: Detection power @ target FPR
        ax = axes[1]
        for i, name in enumerate(a_show):
            vals = [r["powers"].get(name, 0.0) for r in all_results]
            ax.plot(spreads_arr, vals, "o--", color=colors[i], label=name, alpha=0.85)
        for i, name in enumerate(b_show):
            vals = [r["powers"].get(name, 0.0) for r in all_results]
            ax.plot(spreads_arr, vals, "s-", color=colors[len(a_show) + i],
                    label=name, linewidth=2.0)
        ax.axhline(target_fpr, color="gray", linestyle=":", linewidth=1.0,
                   label=f"Chance ({target_fpr*100:.0f}%)")
        ax.set_xlabel("Attack spread (# steps with harm)")
        ax.set_ylabel(f"Detection power @ FPR={target_fpr*100:.0f}%")
        ax.set_title(f"Detection Power vs. Attack Spread")
        ax.set_ylim([-0.05, 1.05])
        ax.legend(fontsize=7)

        fig.tight_layout()
        return fig

    for fig_dir in [
        os.path.join(CODE_DIR, "..", "paper", "figures"),
        os.path.join(CODE_DIR, "results", "figures"),
    ]:
        os.makedirs(fig_dir, exist_ok=True)
        fig = make_fig()
        plotstyle.save_fig(fig, os.path.join(fig_dir, "fig_power_spread"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--n_traj", type=int, default=500)
    args = parser.parse_args()
    main(args.seed, args.T, args.n_traj)
