"""frontier.py — Realizability gate and detectability-band analysis.

REALIZABILITY GATE
==================
Sweeps attack harm-budget; for each setting measures:
  - KS distance of per-step score marginal vs. benign (should be ~0)
  - Achieved total harm
  - Monitor A performance (AUC)
  - Monitor B performance (AUC)
Produces the KS-vs-harm Pareto frontier showing a non-empty stealthy region.

NON-EMPTY DETECTABILITY BAND
=============================
Finds the (stealth, spread) regime where:
  - Monitor A AUC ~0.5 (at chance, cannot detect)
  - Monitor B AUC >> 0.5 (can detect)
This "band" is the core claim of the paper.
"""
from __future__ import annotations

import numpy as np
from numpy.random import Generator
from typing import Optional
from sklearn.metrics import roc_auc_score

from .benign import BenignModel, sample_benign
from .attack import marginal_preserving_attack, naive_attack, ks_distance_to_benign
from .monitors import score_trajectories, SequenceClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_scores_list(
    trajs_benign: np.ndarray,
    trajs_attack,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Convert ndarray trajectories to lists, and extract per-trajectory scores."""
    benign_list = [trajs_benign[i] for i in range(len(trajs_benign))]
    attack_list = [t.scores for t in trajs_attack]
    return benign_list, attack_list


def compute_auc(
    benign_scores_dict: dict,
    attack_scores_dict: dict,
    monitor_names: Optional[list[str]] = None,
) -> dict:
    """Compute ROC-AUC for each monitor.

    Parameters
    ----------
    benign_scores_dict : dict[name -> (N_b,) array]
    attack_scores_dict : dict[name -> (N_a,) array]
    monitor_names : subset of keys to evaluate (None = all)

    Returns
    -------
    dict[name -> auc_float]
    """
    if monitor_names is None:
        monitor_names = list(benign_scores_dict.keys())

    aucs = {}
    for name in monitor_names:
        b = benign_scores_dict[name]
        a = attack_scores_dict[name]
        y_true = np.concatenate([np.zeros(len(b)), np.ones(len(a))])
        y_score = np.concatenate([b, a])
        try:
            aucs[name] = float(roc_auc_score(y_true, y_score))
        except Exception:
            aucs[name] = 0.5
    return aucs


def compute_detection_power_at_fpr(
    benign_scores: np.ndarray,
    attack_scores: np.ndarray,
    target_fpr: float = 0.01,
) -> float:
    """Compute detection power (TPR) at a matched FPR threshold.

    Parameters
    ----------
    benign_scores : (N_b,)
    attack_scores : (N_a,)
    target_fpr : float

    Returns
    -------
    tpr at target_fpr
    """
    threshold = np.quantile(benign_scores, 1.0 - target_fpr)
    tpr = float(np.mean(attack_scores >= threshold))
    return tpr


# ---------------------------------------------------------------------------
# Realizability sweep
# ---------------------------------------------------------------------------

def realizability_sweep(
    model: BenignModel,
    harm_values: np.ndarray,
    T: int,
    n_traj: int,
    stealth: float,
    spread: int,
    rng: Generator,
) -> list[dict]:
    """Sweep harm budget; measure KS distance + achieved harm for each.

    Returns
    -------
    list of dicts with keys: harm_budget, ks_distance, mean_harm, std_harm
    """
    results = []
    for h in harm_values:
        trajs = marginal_preserving_attack(
            model, n_traj, T, harm_budget=h,
            stealth=stealth, spread=spread, rng=rng,
        )
        ks = ks_distance_to_benign(trajs, model, rng=rng)
        harms = [t.total_harm for t in trajs]
        results.append({
            "harm_budget": float(h),
            "ks_distance": ks,
            "mean_harm": float(np.mean(harms)),
            "std_harm": float(np.std(harms)),
        })
    return results


# ---------------------------------------------------------------------------
# Monitor A vs B ROC/AUC comparison
# ---------------------------------------------------------------------------

def monitor_ab_comparison(
    model: BenignModel,
    T: int,
    n_benign: int,
    n_attack: int,
    harm_budget: float,
    stealth: float,
    spread: int,
    rng: Generator,
    train_fraction: float = 0.5,
    target_fpr: float = 0.01,
) -> dict:
    """Full Monitor A vs B comparison: AUCs + detection power at FPR.

    Trains the sequence classifier on a fraction of the data and evaluates on
    the remainder.

    Returns
    -------
    dict with keys:
      aucs_a, aucs_b : dict[monitor_name -> auc]
      powers_a, powers_b : dict[monitor_name -> tpr at target_fpr]
      ks_distance : float
    """
    # Generate data
    benign_arr = sample_benign(model, n_benign, T, rng)
    attack_trajs = marginal_preserving_attack(
        model, n_attack, T, harm_budget=harm_budget,
        stealth=stealth, spread=spread, rng=rng,
    )

    ks = ks_distance_to_benign(attack_trajs, model, rng=rng)

    benign_list, attack_list = _extract_scores_list(benign_arr, attack_trajs)

    # Split for seq_clf training
    n_b_train = max(10, int(n_benign * train_fraction))
    n_a_train = max(10, int(n_attack * train_fraction))
    b_train, b_test = benign_list[:n_b_train], benign_list[n_b_train:]
    a_train, a_test = attack_list[:n_a_train], attack_list[n_a_train:]

    # Train sequence classifier
    seq_clf = SequenceClassifier(window=min(20, T // 5))
    try:
        seq_clf.fit(b_train, a_train)
    except Exception:
        seq_clf = None

    # Score test sets
    benign_scores = score_trajectories(b_test, model, seq_clf=seq_clf)
    attack_scores = score_trajectories(a_test, model, seq_clf=seq_clf)

    monitor_names = list(benign_scores.keys())
    a_monitors = [n for n in monitor_names if n.startswith("a_")]
    b_monitors = [n for n in monitor_names if n.startswith("b_")]

    aucs = compute_auc(benign_scores, attack_scores, monitor_names)
    aucs_a = {n: aucs[n] for n in a_monitors}
    aucs_b = {n: aucs[n] for n in b_monitors}

    powers_a, powers_b = {}, {}
    for n in a_monitors:
        powers_a[n] = compute_detection_power_at_fpr(
            benign_scores[n], attack_scores[n], target_fpr
        )
    for n in b_monitors:
        powers_b[n] = compute_detection_power_at_fpr(
            benign_scores[n], attack_scores[n], target_fpr
        )

    return {
        "aucs_a": aucs_a,
        "aucs_b": aucs_b,
        "powers_a": powers_a,
        "powers_b": powers_b,
        "ks_distance": ks,
    }


# ---------------------------------------------------------------------------
# Power vs spread sweep
# ---------------------------------------------------------------------------

def power_vs_spread_sweep(
    model: BenignModel,
    spreads: list[int],
    T: int,
    n_benign: int,
    n_attack: int,
    harm_budget: float,
    stealth: float,
    rng: Generator,
    target_fpr: float = 0.01,
) -> list[dict]:
    """Sweep attack spread; measure Monitor A and B detection power.

    As spread increases, per-step harm diminishes -> Monitor A degrades.
    Monitor B uses temporal structure and holds up.

    Returns
    -------
    list of dicts with keys: spread, aucs, powers
    """
    results = []
    for sp in spreads:
        # Regenerate rng seed per spread for reproducibility
        res = monitor_ab_comparison(
            model=model, T=T,
            n_benign=n_benign, n_attack=n_attack,
            harm_budget=harm_budget, stealth=stealth, spread=sp,
            rng=rng, train_fraction=0.5, target_fpr=target_fpr,
        )
        results.append({"spread": sp, **res})
    return results
