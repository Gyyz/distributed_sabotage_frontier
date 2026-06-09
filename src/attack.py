"""attack.py — Marginal-preserving distributed-sabotage attacker.

THE CORE CONSTRUCTION
=====================
A benign marginal-preserving attacker that accumulates harm across T steps
without changing the per-step score *histogram* (so mean/max/tail monitors
cannot detect it), but encodes harm in the temporal / correlation structure.

Key idea — Gaussian-copula autocorrelation elevation:
  1. Generate a Gaussian AR(1) process z_1..z_T with phi_attack > phi_benign.
  2. Map z_t -> u_t = Phi(z_t) (uniform margins).
  3. Map u_t -> s_t = F^{-1}(u_t) where F = Beta(a,b).

  The per-step marginal is EXACTLY Beta(a,b) for ALL values of phi.
  Monitor A (functions of the marginal) cannot distinguish benign from attack.
  Monitor B (correlation tests) detects the elevated phi.

ATTACK VARIANTS
==============
1. marginal_preserving_attack — pure autocorrelation elevation (stealthy)
2. naive_attack               — score-lifting offset (non-stealthy, Monitor A detects)
3. semi_stealthy_attack       — harm concentrated on top-spread steps;
                                used for the "power vs spread" experiment where
                                spread modulates per-step harm size.

The semi-stealthy attack adds a small per-step score boost proportional to
harm_budget / spread, which IS detectable by Monitor A when spread is small
(large per-step harm = large score lift) but becomes undetectable when spread
approaches T (tiny per-step lift, near-marginal-preserving).
Meanwhile Monitor B detects it via the autocorrelation structure at all spreads.

Attack parameters
-----------------
stealth : float in [0, 1]
    phi_attack = phi_benign + stealth * (0.95 - phi_benign).

spread : int in [1, T]
    For semi_stealthy_attack only.  Harm is amortized across spread steps.
    Small spread -> large per-step score lift -> Monitor A detects.
    Large spread -> tiny per-step lift -> Monitor A fails.
"""
from __future__ import annotations

import numpy as np
from numpy.random import Generator
from dataclasses import dataclass
from typing import Optional
from scipy import stats

from .benign import BenignModel


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class AttackTrajectory:
    """One attacked trajectory.

    Attributes
    ----------
    scores : ndarray, shape (T,)
        Per-step monitor scores.
    harm : ndarray, shape (T,)
        Latent per-step harm (not observed by monitors).
    total_harm : float
    phi_used : float
        The AR(1) coefficient used to generate this trajectory.
    """
    scores: np.ndarray
    harm: np.ndarray
    total_harm: float
    phi_used: float = 0.0


# ---------------------------------------------------------------------------
# Core Gaussian-copula AR(1) sampler (exact marginal preservation)
# ---------------------------------------------------------------------------

def _sample_ar1_copula(
    T: int,
    a: float,
    b: float,
    phi: float,
    rng: Generator,
) -> np.ndarray:
    """Sample T steps with Beta(a,b) marginal and AR(1) Gaussian copula.

    The per-step marginal is EXACTLY Beta(a,b) for any phi in [0, 1).
    Only the temporal autocorrelation differs from phi=0 (i.i.d.).
    """
    innov_std = float(np.sqrt(max(1.0 - phi * phi, 1e-8)))
    z = np.empty(T)
    z[0] = rng.standard_normal()
    for t in range(1, T):
        z[t] = phi * z[t - 1] + innov_std * rng.standard_normal()
    u = np.clip(stats.norm.cdf(z), 1e-7, 1.0 - 1e-7)
    return np.clip(stats.beta.ppf(u, a, b), 0.0, 1.0)


# ---------------------------------------------------------------------------
# 1. Marginal-preserving (stealthy) attacker
# ---------------------------------------------------------------------------

def marginal_preserving_attack(
    model: BenignModel,
    n_traj: int,
    T: int,
    harm_budget: float,
    stealth: float,
    spread: int,
    rng: Generator,
) -> list[AttackTrajectory]:
    """Marginal-preserving attack via Gaussian-copula phi elevation.

    Per-step marginal is exactly Beta(a,b).  Harm is assigned to top-spread
    steps by score rank but this does NOT affect the scores themselves.

    The only detectable signal is the elevated autocorrelation in the z-space:
      phi_attack = phi_benign + stealth * (0.95 - phi_benign)

    Parameters
    ----------
    model : BenignModel
    n_traj : int
    T : int
    harm_budget : float — total harm per trajectory
    stealth : float — controls phi elevation (0=benign, 1=max)
    spread : int — number of steps receiving harm (affects harm distribution,
                   NOT score distribution)
    rng : Generator
    """
    spread = max(1, min(spread, T))
    a = model.params.get("a", 2.0)
    b = model.params.get("b", 5.0)
    phi_benign = model.phi
    phi_attack = phi_benign + float(stealth) * (0.95 - phi_benign)
    phi_attack = float(np.clip(phi_attack, phi_benign, 0.95))

    results = []
    for _ in range(n_traj):
        scores = _sample_ar1_copula(T, a, b, phi_attack, rng)
        harm = np.zeros(T)
        if harm_budget > 0.0:
            rank_order = np.argsort(scores)
            top_idx = rank_order[T - spread:]
            w = np.argsort(np.argsort(scores[top_idx])).astype(float) + 1.0
            w /= w.sum()
            harm[top_idx] = harm_budget * w
        results.append(AttackTrajectory(scores, harm, float(harm.sum()), phi_attack))
    return results


# ---------------------------------------------------------------------------
# 2. Naive (non-stealthy) attacker
# ---------------------------------------------------------------------------

def naive_attack(
    model: BenignModel,
    n_traj: int,
    T: int,
    harm_budget: float,
    score_lift: float,
    rng: Generator,
) -> list[AttackTrajectory]:
    """Naive attacker: uniform score lift.  Monitor A catches it immediately.

    Uses benign phi so temporal structure is unchanged; only marginal shifts.
    """
    a = model.params.get("a", 2.0)
    b_param = model.params.get("b", 5.0)
    phi = model.phi

    results = []
    for _ in range(n_traj):
        base = _sample_ar1_copula(T, a, b_param, phi, rng)
        scores = np.clip(base + score_lift, 0.0, 1.0)
        harm = np.full(T, harm_budget / T)
        results.append(AttackTrajectory(scores, harm, harm_budget, phi))
    return results


# ---------------------------------------------------------------------------
# 3. Semi-stealthy attack for "power vs spread" experiment
# ---------------------------------------------------------------------------

def semi_stealthy_attack(
    model: BenignModel,
    n_traj: int,
    T: int,
    harm_budget: float,
    stealth: float,
    spread: int,
    rng: Generator,
) -> list[AttackTrajectory]:
    """Semi-stealthy attack that blends score-lifting with autocorrelation.

    Harm is concentrated on 'spread' steps.  Each harm step adds a score lift
    proportional to harm_budget / spread.  This means:

    - Small spread: large per-step harm -> large per-step score lift
      -> Monitor A (a_max, a_threshold) DETECTS.
    - Large spread (spread = T): tiny per-step harm -> tiny per-step lift
      -> Monitor A FAILS.
    - Monitor B (b_sprt, b_autocorr): detects elevated phi_attack at ALL spreads.

    This is the key experiment: as spread increases, Monitor A degrades toward
    chance while Monitor B maintains high detection power.

    Score lift per harm step = (harm_budget / spread) * score_lift_factor.
    score_lift_factor is calibrated so that at spread=1, the lift is detectable.
    """
    spread = max(1, min(spread, T))
    a = model.params.get("a", 2.0)
    b = model.params.get("b", 5.0)
    phi_benign = model.phi
    phi_attack = phi_benign + float(stealth) * (0.95 - phi_benign)
    phi_attack = float(np.clip(phi_attack, phi_benign, 0.95))

    # Score lift factor: harm_budget / T is per-step harm at full spread;
    # we scale so that max lift = 0.30 at spread=1
    max_lift = 0.30
    per_step_lift = min(max_lift, harm_budget / spread * 0.3)

    results = []
    for _ in range(n_traj):
        # Base: AR(1) with elevated phi (temporal structure, marginal-preserving)
        scores = _sample_ar1_copula(T, a, b, phi_attack, rng)

        # Additional: add a score lift to the top-spread-ranked steps
        # This is the "naive" component that Monitor A can detect when spread is small
        harm = np.zeros(T)
        if harm_budget > 0.0 and per_step_lift > 0.0:
            rank_order = np.argsort(scores)
            top_idx = rank_order[T - spread:]
            scores[top_idx] = np.clip(scores[top_idx] + per_step_lift, 0.0, 1.0)
            harm[top_idx] = harm_budget / spread

        results.append(AttackTrajectory(scores, harm, float(harm.sum()), phi_attack))
    return results


# ---------------------------------------------------------------------------
# KS distance utility
# ---------------------------------------------------------------------------

def ks_distance_to_benign(
    attack_trajs: list[AttackTrajectory],
    model: BenignModel,
    n_ref: int = 5000,
    rng: Optional[Generator] = None,
) -> float:
    """KS distance between attack scores and benign marginal.

    For marginal_preserving_attack: should be ~0 (same Beta marginal).
    For semi_stealthy_attack at large spread: should also be ~0.
    For naive_attack: large (shifted marginal).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    attack_scores = np.concatenate([t.scores for t in attack_trajs])
    a = model.params.get("a", 2.0)
    b_param = model.params.get("b", 5.0)
    benign_ref = rng.beta(a, b_param, size=n_ref)
    ks_stat, _ = stats.ks_2samp(attack_scores, benign_ref)
    return float(ks_stat)
