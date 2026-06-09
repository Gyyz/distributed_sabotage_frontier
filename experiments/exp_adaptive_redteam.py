"""Experiment: Adaptive red-teamer & the detectability frontier.

Closes the loop left open by the static monitor-A-vs-B comparison: what does a
*reactive* attacker do once it knows a correlation monitor (Monitor B) is
deployed? Its only lever is the coordination/stealth parameter delta_phi (the
AR(1) elevation that encodes distributed harm). Lowering delta_phi evades
Monitor B but also removes the temporal coordination the attack relies on.

We characterise three things, all CPU-only:

(1) DETECTABILITY FRONTIER. Sweep delta_phi downward. Monitor A stays at
    chance (marginal-preserving by construction); Monitor B's AUC degrades
    toward 0.5 as the attacker becomes stealthier. We locate the evasion
    threshold delta_phi* below which the realistic (parameter-free)
    autocorrelation detector drops under AUC 0.65 -- the honest message that
    Monitor B is not a panacea against a patient adaptive attacker.

(2) EVIDENCE COUNTER. For a stealthy attacker (small delta_phi) the defender's
    counter is to accumulate more steps. We sweep trajectory length T and show
    Monitor B's AUC rises with T at fixed delta_phi -- the attacker/defender
    trade-off (stealth vs evidence).

(3) BASE-RATE / FPR ROBUSTNESS. At the headline operating point we report
    Monitor B detection power at strict FPRs (0.1%, 1%, 5%) matching realistic
    rare-sabotage base rates.

Outputs: results/adaptive_redteam.json, ../paper/figures/fig_adaptive_frontier.{pdf,png}
"""
import sys, os, json, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import plotstyle
plotstyle.apply()
import matplotlib.pyplot as plt

from src.benign import default_benign_model, sample_benign
from src.attack import marginal_preserving_attack
from src.monitors import score_trajectories, SequenceClassifier
from src.frontier import compute_auc, compute_detection_power_at_fpr

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

PHI_BENIGN = 0.5
PHI_MAX = 0.95
# attack stealth knob `stealth` maps to delta_phi = stealth * (PHI_MAX - PHI_BENIGN)
def stealth_for_delta(delta_phi):
    return float(np.clip(delta_phi / (PHI_MAX - PHI_BENIGN), 0.0, 1.0))


def auc_at(model, T, delta_phi, n_traj, harm_budget, spread, rng, fit_clf=True):
    """Return dict of monitor->AUC and ->power@1%FPR for one (T, delta_phi)."""
    stealth = stealth_for_delta(delta_phi)
    benign = sample_benign(model, n_traj, T, rng)
    atk = marginal_preserving_attack(model, n_traj, T, harm_budget=harm_budget,
                                     stealth=stealth, spread=spread, rng=rng)
    benign_list = [benign[i] for i in range(n_traj)]
    atk_list = [t.scores for t in atk]
    nb, na = n_traj // 2, n_traj // 2
    b_tr, b_te = benign_list[:nb], benign_list[nb:]
    a_tr, a_te = atk_list[:na], atk_list[na:]
    seq_clf = None
    if fit_clf:
        seq_clf = SequenceClassifier(window=min(20, T // 5), model=model)
        try:
            seq_clf.fit(b_tr, a_tr)
        except Exception:
            seq_clf = None
    bs = score_trajectories(b_te, model, seq_clf=seq_clf)
    as_ = score_trajectories(a_te, model, seq_clf=seq_clf)
    names = list(bs.keys())
    aucs = compute_auc(bs, as_, names)
    powers = {n: compute_detection_power_at_fpr(bs[n], as_[n], 0.01) for n in names}
    return aucs, powers, bs, as_


def main(seed=42, T=300, n_traj=600):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
    model = default_benign_model(mean=0.286, var=0.025, phi=PHI_BENIGN)
    harm_budget, spread = 1.5, T // 2

    # ---- (1) Detectability frontier: sweep delta_phi ----
    deltas = [0.02, 0.05, 0.09, 0.15, 0.25, 0.40]
    a_best, b_autocorr, b_sprt, b_seqclf, b_best = [], [], [], [], []
    for dp in deltas:
        rng = np.random.default_rng(seed + int(dp * 1000))
        aucs, _, _, _ = auc_at(model, T, dp, n_traj, harm_budget, spread, rng)
        a_keys = [k for k in aucs if k.startswith("a_")]
        b_keys = [k for k in aucs if k.startswith("b_")]
        a_best.append(max(abs(aucs[k] - 0.5) + 0.5 for k in a_keys))  # best A (dist from chance)
        b_autocorr.append(aucs["b_autocorr"])
        b_sprt.append(aucs["b_sprt"])
        b_seqclf.append(aucs.get("b_seq_clf", float("nan")))
        b_best.append(max(aucs[k] for k in b_keys))
    # evasion threshold: smallest delta_phi with realistic (autocorr) AUC >= 0.65
    evasion_idx = [i for i, v in enumerate(b_autocorr) if v >= 0.65]
    delta_star = deltas[min(evasion_idx)] if evasion_idx else None

    # ---- (2) Evidence counter: sweep T at a stealthy delta_phi ----
    dp_stealthy = 0.05
    Ts = [100, 200, 400, 800, 1600]
    b_autocorr_vs_T, b_sprt_vs_T, a_vs_T = [], [], []
    for Tv in Ts:
        rng = np.random.default_rng(seed + Tv)
        aucs, _, _, _ = auc_at(model, Tv, dp_stealthy, n_traj, harm_budget, Tv // 2, rng)
        b_autocorr_vs_T.append(aucs["b_autocorr"])
        b_sprt_vs_T.append(aucs["b_sprt"])
        a_vs_T.append(aucs["a_max"])

    # ---- (3) Base-rate / FPR robustness at headline operating point ----
    rng = np.random.default_rng(seed + 7)
    _, _, bs, as_ = auc_at(model, T, 0.09, n_traj, harm_budget, spread, rng)
    fprs = [0.001, 0.01, 0.05]
    power_by_fpr = {}
    for det in ["b_autocorr", "b_cusum", "b_sprt", "b_seq_clf", "a_max"]:
        if det in bs:
            power_by_fpr[det] = {f"{f}": compute_detection_power_at_fpr(bs[det], as_[det], f) for f in fprs}

    results = {
        "config": {"T": T, "n_traj": n_traj, "phi_benign": PHI_BENIGN, "seed": seed,
                   "harm_budget": harm_budget},
        "frontier": {"delta_phi": deltas, "a_best_auc": a_best,
                     "b_autocorr_auc": b_autocorr, "b_sprt_auc": b_sprt,
                     "b_seq_clf_auc": b_seqclf, "b_best_auc": b_best,
                     "evasion_threshold_delta_phi": delta_star,
                     "evasion_rule": "smallest delta_phi with parameter-free autocorr AUC >= 0.65"},
        "evidence_counter": {"delta_phi": dp_stealthy, "T": Ts,
                             "b_autocorr_auc": b_autocorr_vs_T, "b_sprt_auc": b_sprt_vs_T,
                             "a_max_auc": a_vs_T},
        "base_rate_robustness": {"delta_phi": 0.09, "power_at_fpr": power_by_fpr},
    }
    out = os.path.join(RESULTS_DIR, "adaptive_redteam.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"  saved {out}")
    print(f"  evasion threshold delta_phi* = {delta_star} (autocorr AUC>=0.65)")
    print(f"  frontier b_autocorr AUC: {[round(x,3) for x in b_autocorr]}")
    print(f"  evidence counter (delta_phi=0.05) b_autocorr AUC vs T: {[round(x,3) for x in b_autocorr_vs_T]}")

    # ---- Figure: 2 panels ----
    def make_fig():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
        ax = axes[0]
        ax.plot(deltas, a_best, "s--", color=plotstyle.COLORS["gray"], label="Monitor A (best)")
        ax.plot(deltas, b_autocorr, "o-", color=plotstyle.COLORS["blue"], label="Monitor B: autocorr (param-free)")
        ax.plot(deltas, b_sprt, "^-", color=plotstyle.COLORS["red"], label="Monitor B: SPRT (oracle $\\phi$)")
        ax.plot(deltas, b_seqclf, "D-", color=plotstyle.COLORS["green"], label="Monitor B: seq. classifier")
        ax.axhline(0.5, color="k", lw=0.8, ls=":")
        ax.axhline(0.65, color=plotstyle.COLORS["orange"], lw=1.0, ls="--", label="evasion line (AUC=0.65)")
        if delta_star is not None:
            ax.axvline(delta_star, color=plotstyle.COLORS["orange"], lw=0.8, alpha=0.6)
        ax.set_xlabel("Attacker coordination / stealth  $\\delta_\\phi$")
        ax.set_ylabel("Detection AUC")
        ax.set_title("(a) Detectability frontier (adaptive attacker)")
        ax.legend(fontsize=7.5, loc="center right")
        ax.set_ylim(0.45, 1.02)

        ax2 = axes[1]
        ax2.plot(Ts, a_vs_T, "s--", color=plotstyle.COLORS["gray"], label="Monitor A (max)")
        ax2.plot(Ts, b_autocorr_vs_T, "o-", color=plotstyle.COLORS["blue"], label="Monitor B: autocorr")
        ax2.plot(Ts, b_sprt_vs_T, "^-", color=plotstyle.COLORS["red"], label="Monitor B: SPRT")
        ax2.axhline(0.5, color="k", lw=0.8, ls=":")
        ax2.set_xscale("log")
        ax2.set_xlabel("Trajectory length $T$ (evidence)")
        ax2.set_ylabel("Detection AUC")
        ax2.set_title(f"(b) Defender's counter: evidence\n(stealthy attacker $\\delta_\\phi$={dp_stealthy})")
        ax2.legend(fontsize=8, loc="lower right")
        ax2.set_ylim(0.45, 1.02)
        fig.suptitle("Adaptive Red-Teaming: the Sub-Threshold Sabotage Detectability Frontier", fontsize=12, y=1.02)
        fig.tight_layout()
        return fig

    plotstyle.save_fig(make_fig(), os.path.join(FIGURES_DIR, "fig_adaptive_frontier"))
    plotstyle.save_fig(make_fig(), os.path.join(RESULTS_DIR, "figures", "fig_adaptive_frontier"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--T", type=int, default=300)
    p.add_argument("--n_traj", type=int, default=600)
    a = p.parse_args()
    main(a.seed, a.T, a.n_traj)
