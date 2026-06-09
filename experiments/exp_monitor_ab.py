"""exp_monitor_ab.py — Monitor A vs B ROC/AUC comparison.

At the marginal-preserving attack setting (stealth=0.2, spread=T/2):
  - Monitor A should achieve AUC ~0.5 (near chance, beta marginal is identical)
  - Monitor B should achieve AUC ~1.0 (detects elevated autocorrelation)

Also runs the naive attacker as a sanity check:
  - Both Monitor A and B detect the naive attack (AUC > 0.9).

This demonstrates the DETECTABILITY BAND: the regime where A is at chance
but B is powerful.

Writes: results/monitor_ab.json
Figures: paper/figures/fig_roc
"""
import sys
import os
import json
import argparse
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

import plotstyle
plotstyle.apply()

import matplotlib.pyplot as plt
from src.benign import default_benign_model, sample_benign
from src.attack import marginal_preserving_attack, naive_attack, ks_distance_to_benign
from src.monitors import score_trajectories, SequenceClassifier
from src.frontier import compute_auc, compute_detection_power_at_fpr


def run_comparison(model, T, n_benign, n_attack, harm_budget, stealth, spread,
                   rng_seed, target_fpr=0.01, label="stealthy",
                   use_naive=False, score_lift=0.15):
    rng = np.random.default_rng(rng_seed)
    benign_arr = sample_benign(model, n_benign, T, rng)
    if use_naive:
        attack_trajs = naive_attack(model, n_attack, T,
                                    harm_budget=harm_budget, score_lift=score_lift, rng=rng)
    else:
        attack_trajs = marginal_preserving_attack(
            model, n_attack, T,
            harm_budget=harm_budget, stealth=stealth, spread=spread, rng=rng,
        )

    ks = ks_distance_to_benign(attack_trajs, model, rng=rng)
    print(f"  [{label}] KS distance to benign: {ks:.4f}")

    benign_list = [benign_arr[i] for i in range(n_benign)]
    attack_list = [t.scores for t in attack_trajs]

    # Train/test split (per class independently)
    n_b_train = n_benign // 2
    n_a_train = n_attack // 2
    b_train, b_test = benign_list[:n_b_train], benign_list[n_b_train:]
    a_train, a_test = attack_list[:n_a_train], attack_list[n_a_train:]

    seq_clf = SequenceClassifier(window=min(20, T // 5), model=model)
    try:
        seq_clf.fit(b_train, a_train)
    except Exception as e:
        print(f"  Warning: seq_clf fit failed: {e}")
        seq_clf = None

    benign_scores = score_trajectories(b_test, model, seq_clf=seq_clf)
    attack_scores = score_trajectories(a_test, model, seq_clf=seq_clf)

    all_names = list(benign_scores.keys())
    aucs = compute_auc(benign_scores, attack_scores, all_names)
    powers = {n: compute_detection_power_at_fpr(benign_scores[n], attack_scores[n], target_fpr)
              for n in all_names}

    return benign_scores, attack_scores, aucs, powers, ks


def main(seed: int = 42, T: int = 300, n_traj: int = 600):
    model = default_benign_model(mean=0.286, var=0.025, phi=0.5)
    n_benign = n_traj
    n_attack = n_traj // 2

    harm_budget = 1.5
    stealth = 0.2
    spread = T // 2
    target_fpr = 0.01

    print("Running marginal-preserving attack comparison...")
    b_scores, a_scores, aucs_mp, powers_mp, ks_mp = run_comparison(
        model, T, n_benign, n_attack, harm_budget, stealth, spread,
        rng_seed=seed, target_fpr=target_fpr, label="marginal-preserving",
    )

    print("\nRunning naive attack comparison...")
    b_scores2, n_scores2, aucs_naive, powers_naive, ks_naive = run_comparison(
        model, T, n_benign, n_attack, harm_budget, stealth, spread,
        rng_seed=seed + 10, target_fpr=target_fpr, label="naive",
        use_naive=True, score_lift=0.15,
    )

    print("\n=== Monitor AUCs — Marginal-Preserving Attack ===")
    a_names = [n for n in aucs_mp if n.startswith("a_")]
    b_names = [n for n in aucs_mp if n.startswith("b_")]
    print("  Monitor A:")
    for n in a_names:
        print(f"    {n:20s}: AUC={aucs_mp[n]:.3f}  Power@{target_fpr*100:.0f}%FPR={powers_mp[n]:.3f}")
    print("  Monitor B:")
    for n in b_names:
        print(f"    {n:20s}: AUC={aucs_mp[n]:.3f}  Power@{target_fpr*100:.0f}%FPR={powers_mp[n]:.3f}")

    print("\n=== Monitor AUCs — Naive Attack ===")
    for n in a_names + b_names:
        if n in aucs_naive:
            print(f"  {n:20s}: AUC={aucs_naive[n]:.3f}  Power={powers_naive.get(n, 0):.3f}")

    # Save JSON
    output = {
        "marginal_preserving": {
            "aucs": aucs_mp,
            "powers": powers_mp,
            "ks_distance": ks_mp,
        },
        "naive": {
            "aucs": aucs_naive,
            "powers": powers_naive,
            "ks_distance": ks_naive,
        },
        "config": {
            "T": T, "n_benign": n_benign, "n_attack": n_attack,
            "harm_budget": harm_budget, "stealth": stealth, "spread": spread,
            "target_fpr": target_fpr, "seed": seed,
        },
    }

    results_dir = os.path.join(CODE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "monitor_ab.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  wrote {json_path}")

    # --- ROC figure ---
    def make_roc_fig(b_sc, a_sc, names, title, ax):
        colors = list(plotstyle.CYCLE)
        for i, n in enumerate(names):
            if n not in b_sc or n not in a_sc:
                continue
            b = b_sc[n]
            a = a_sc[n]
            y_true = np.concatenate([np.zeros(len(b)), np.ones(len(a))])
            y_score = np.concatenate([b, a])
            try:
                auc = roc_auc_score(y_true, y_score)
            except Exception:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_score)
            label_str = f"{n} (AUC={auc:.2f})"
            ax.plot(fpr, tpr, linewidth=1.5, label=label_str,
                    color=colors[i % len(colors)])
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
        ax.axvline(target_fpr, color="gray", linestyle=":", linewidth=1.0)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(fontsize=6.5)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])

    # Show a subset of monitors to keep figure legible
    a_show = a_names[:4]
    b_show = [n for n in b_names if n not in ("b_seq_clf",)][:4]
    show_names = a_show + b_show

    for fig_dir in [
        os.path.join(CODE_DIR, "..", "paper", "figures"),
        os.path.join(CODE_DIR, "results", "figures"),
    ]:
        os.makedirs(fig_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        make_roc_fig(b_scores, a_scores, show_names,
                     "Marginal-Preserving Attack", axes[0])
        make_roc_fig(b_scores2, n_scores2, show_names,
                     "Naive (Score-Lifting) Attack", axes[1])
        fig.tight_layout()
        plotstyle.save_fig(fig, os.path.join(fig_dir, "fig_roc"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--n_traj", type=int, default=600)
    args = parser.parse_args()
    main(args.seed, args.T, args.n_traj)
