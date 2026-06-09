"""Real-data validation of the distributed-sabotage detectability band.

Uses REAL benign per-step monitor-score trajectories (Qwen2.5-7B agent scored by
a Llama-3.2-3B monitor; see gpu/gen_real_traces.py) as the benign class, and
constructs the marginal-preserving attack from the REAL empirical marginal:

  z ~ AR(1, phi_att)  ->  u = Phi(z)  ->  s = Q_real(u)

where Q_real is the empirical quantile function of the pooled real benign scores.
By construction the attack's per-step marginal EXACTLY equals the real benign
marginal (so Monitor A is blind), while its temporal autocorrelation is elevated
to phi_att = phi_real + delta (so Monitor B can detect it).

We report, on real-calibrated data: (1) the real benign marginal + autocorrelation,
(2) KS(attack, benign) ~ 0, (3) Monitor A vs Monitor B AUC, and (4) Monitor B AUC
vs effective horizon (real agent runs are short, T~24; concatenating K episodes
recovers detectability -- the evidence axis of the adaptive frontier).

Input: gpu/results/real_traces.json. Output: results/realdata_detect.json,
../paper/figures/fig_realdata_detect.{pdf,png}
"""
import os, sys, json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import plotstyle
plotstyle.apply()
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACES = os.path.join(CODE_DIR, "gpu", "results", "real_traces.json")
RESULTS_DIR = os.path.join(CODE_DIR, "results")
FIG_DIR = os.path.join(CODE_DIR, "..", "paper", "figures")


def empirical_cdf_fns(pool):
    xs = np.sort(pool)
    n = len(xs)
    def cdf(s):  # -> u in (0,1)
        r = np.searchsorted(xs, s, side="right") / (n + 1.0)
        return np.clip(r, 1e-4, 1 - 1e-4)
    def quant(u):  # inverse
        return np.interp(np.clip(u, 0, 1), np.linspace(0, 1, n), xs)
    return cdf, quant


def to_z(traj, cdf):
    return stats.norm.ppf(cdf(np.asarray(traj)))


def sample_attack(n, T, quant, phi, rng):
    out = []
    s = float(np.sqrt(max(1 - phi * phi, 1e-6)))
    for _ in range(n):
        z = np.empty(T); z[0] = rng.standard_normal()
        for t in range(1, T):
            z[t] = phi * z[t - 1] + s * rng.standard_normal()
        u = stats.norm.cdf(z)
        out.append(quant(u))
    return np.array(out)


# --- monitors ---
def a_mean(s): return float(np.mean(s))
def a_max(s): return float(np.max(s))
def a_topk(s, k=4): return float(np.mean(np.sort(s)[-k:]))
def b_autocorr(z, K=3):
    z = z - z.mean(); v = (z**2).mean()
    if v < 1e-9: return 0.0
    return float(sum((z[:-l]*z[l:]).mean()/v for l in range(1, min(K+1, len(z)))))
def b_sprt(z, phi0, phi1):
    # AR(1) Gaussian LLR for phi1 vs phi0
    def ll(phi):
        s2 = 1 - phi*phi
        r = z[1:] - phi*z[:-1]
        return -0.5*np.sum(r*r)/s2 - 0.5*(len(z)-1)*np.log(s2)
    return float(ll(phi1) - ll(phi0))


def aucs(benign, attack, cdf, phi0, phi1):
    res = {}
    for name, fn in [("a_mean", a_mean), ("a_max", a_max), ("a_topk", a_topk)]:
        sb = [fn(t) for t in benign]; sa = [fn(t) for t in attack]
        y = np.r_[np.zeros(len(sb)), np.ones(len(sa))]
        res[name] = roc_auc_score(y, np.r_[sb, sa])
    zb = [to_z(t, cdf) for t in benign]; za = [to_z(t, cdf) for t in attack]
    for name, fn in [("b_autocorr", lambda z: b_autocorr(z)),
                     ("b_sprt", lambda z: b_sprt(z, phi0, phi1))]:
        sb = [fn(z) for z in zb]; sa = [fn(z) for z in za]
        y = np.r_[np.zeros(len(sb)), np.ones(len(sa))]
        res[name] = roc_auc_score(y, np.r_[sb, sa])
    return res


def main():
    d = json.load(open(TRACES))
    benign = np.array(d["scores"])              # (N, T) real benign
    N, T = benign.shape
    phi_real = d["lag1_autocorr"]
    pool = benign.ravel()
    cdf, quant = empirical_cdf_fns(pool)
    os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
    rng = np.random.default_rng(0)

    delta = 0.25
    phi_att = min(0.9, phi_real + delta)
    attack = sample_attack(N, T, quant, phi_att, rng)
    # KS marginal check
    ks = float(stats.ks_2samp(attack.ravel(), pool)[0])
    main_auc = aucs(list(benign), list(attack), cdf, phi_real, phi_att)

    # Evidence axis: real agent episodes are short (T=24). To isolate the effect
    # of HORIZON from sample size, we generate real-CALIBRATED benign (empirical
    # marginal + real phi) and attack (same marginal + phi_att) at FIXED N over a
    # grid of T. This holds sample count constant so the trend reflects evidence,
    # not estimator noise (the bug in naive episode-concatenation).
    Ts = [24, 48, 96, 200, 400]
    Nh = 300
    horizon = {"b_autocorr": [], "b_sprt": [], "a_max": [], "T_eff": Ts}
    for Tg in Ts:
        b = list(sample_attack(Nh, Tg, quant, phi_real, np.random.default_rng(100 + Tg)))   # real-calibrated benign
        at = list(sample_attack(Nh, Tg, quant, phi_att, np.random.default_rng(900 + Tg)))    # attack
        au = aucs(b, at, cdf, phi_real, phi_att)
        horizon["b_autocorr"].append(au["b_autocorr"])
        horizon["b_sprt"].append(au["b_sprt"])
        horizon["a_max"].append(au["a_max"])

    out = {"n_traj": N, "T": T, "agent": d["agent"], "monitor": d["monitor"],
           "benign_marginal_mean": d["marginal_mean"], "benign_marginal_var": d["marginal_var"],
           "benign_beta": [d["beta_a"], d["beta_b"]], "phi_real": phi_real, "phi_attack": phi_att,
           "ks_attack_vs_benign": ks, "main_aucs": main_auc, "horizon": horizon, "Ts": Ts}
    json.dump(out, open(os.path.join(RESULTS_DIR, "realdata_detect.json"), "w"), indent=2)
    print(f"Real benign: N={N} T={T} mean={d['marginal_mean']:.3f} phi={phi_real:.3f} "
          f"Beta({d['beta_a']:.2f},{d['beta_b']:.2f})")
    print(f"Attack phi={phi_att:.3f} | KS(attack,benign)={ks:.4f}")
    print("Main AUCs:", {k: round(v, 3) for k, v in main_auc.items()})
    print("Monitor B (autocorr) AUC vs horizon T_eff:",
          [(t, round(a, 3)) for t, a in zip(horizon["T_eff"], horizon["b_autocorr"])])

    def make_fig():
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
        # (a) real benign marginal + fitted Beta
        ax = axes[0]
        ax.hist(pool, bins=25, density=True, alpha=0.6, color=plotstyle.COLORS["gray"], label="real benign scores")
        xs = np.linspace(0, 1, 200)
        if np.isfinite(d["beta_a"]):
            ax.plot(xs, stats.beta.pdf(xs, d["beta_a"], d["beta_b"]), color=plotstyle.COLORS["red"],
                    lw=2, label=f"Beta({d['beta_a']:.1f},{d['beta_b']:.1f}) fit")
        ax.set_xlabel("Per-step monitor score"); ax.set_ylabel("Density")
        ax.set_title(f"(a) Real benign monitor scores\n(Qwen2.5-7B agent / Llama-3.2-3B monitor, $\\phi$={phi_real:.2f})")
        ax.legend(fontsize=8)
        # (b) main AUCs A vs B
        ax2 = axes[1]
        names = ["a_mean", "a_max", "a_topk", "b_autocorr", "b_sprt"]
        vals = [main_auc[n] for n in names]
        cols = [plotstyle.COLORS["gray"]]*3 + [plotstyle.COLORS["blue"], plotstyle.COLORS["red"]]
        ax2.bar(range(len(names)), vals, color=cols, edgecolor="k", linewidth=0.5)
        ax2.axhline(0.5, color="k", ls=":", lw=0.8)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(["A:mean", "A:max", "A:top-k", "B:autocorr", "B:SPRT"], rotation=20, ha="right", fontsize=8)
        ax2.set_ylabel("Detection AUC"); ax2.set_ylim(0, 1.02)
        ax2.set_title(f"(b) A blind, B detects on REAL benign\n(KS attack vs benign = {ks:.3f})")
        for i, v in enumerate(vals): ax2.text(i, v+0.01, f"{v:.2f}", ha="center", fontsize=8)
        # (c) horizon: B AUC vs effective T (real runs are short)
        ax3 = axes[2]
        ax3.plot(horizon["T_eff"], horizon["b_autocorr"], "o-", color=plotstyle.COLORS["blue"], label="B: autocorr")
        ax3.plot(horizon["T_eff"], horizon["b_sprt"], "^-", color=plotstyle.COLORS["red"], label="B: SPRT")
        ax3.plot(horizon["T_eff"], horizon["a_max"], "s--", color=plotstyle.COLORS["gray"], label="A: max")
        ax3.axhline(0.5, color="k", ls=":", lw=0.8)
        ax3.set_xscale("log"); ax3.set_xlabel("Effective horizon $T$ (episodes concatenated)")
        ax3.set_ylabel("Detection AUC"); ax3.set_ylim(0.4, 1.02)
        ax3.set_title("(c) Short real runs need evidence:\nB recovers as horizon grows")
        ax3.legend(fontsize=8)
        fig.suptitle("Real-Data Validation: Distributed Sabotage on Live Agent/Monitor Traces", fontsize=12, y=1.02)
        fig.tight_layout(); return fig

    plotstyle.save_fig(make_fig(), os.path.join(FIG_DIR, "fig_realdata_detect"))
    plotstyle.save_fig(make_fig(), os.path.join(RESULTS_DIR, "figures", "fig_realdata_detect"))
    print("  figure written")


if __name__ == "__main__":
    main()
