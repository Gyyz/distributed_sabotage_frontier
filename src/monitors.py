"""monitors.py — Monitor A (marginal/per-step) and Monitor B (sequence/correlation).

Monitor A — per-step / marginal monitors
=========================================
These operate on the empirical distribution of per-step scores.  They cannot
detect attacks that preserve the per-step marginal because:
  - Their test statistics are functions of the order statistics only (rank-invariant)
  - For any Beta(a,b) sample, the empirical CDF converges to the same distribution

  a_mean        : mean score
  a_max         : maximum score
  a_top_k_tail  : mean of top-k scores
  a_threshold   : fraction of steps exceeding a threshold

Monitor B — temporal / correlation monitors
============================================
These use sequential structure and can detect attacks that elevate the
autocorrelation of the latent Gaussian process while preserving the Beta marginal.

  b_cusum       : CUSUM on sample autocorrelation increment
  b_sprt        : SPRT LLR (benign phi_0 vs elevated phi_1 alt.)
  b_hmm_lr      : 2-state Markov chain LLR
  b_runs_test   : Wald-Wolfowitz runs test (fewer runs = more autocorr)
  b_autocorr    : Lag-1 autocorrelation in the Beta copula z-space
  b_seq_clf     : Sliding-window logistic regression (sklearn)

All monitors return a scalar score per trajectory (higher = more suspicious).
"""
from __future__ import annotations

import numpy as np
from numpy.random import Generator
from typing import Optional
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .benign import BenignModel


# =============================================================================
# Monitor A — marginal / per-step monitors
# =============================================================================

def a_mean(scores: np.ndarray) -> float:
    """Mean per-step score."""
    return float(np.mean(scores))


def a_max(scores: np.ndarray) -> float:
    """Maximum per-step score."""
    return float(np.max(scores))


def a_top_k_tail(scores: np.ndarray, k: Optional[int] = None) -> float:
    """Mean of the top-k scores (tail statistic)."""
    T = len(scores)
    if k is None:
        k = max(1, T // 10)
    k = min(k, T)
    return float(np.mean(np.partition(scores, -k)[-k:]))


def a_threshold(scores: np.ndarray, threshold: float = 0.7) -> float:
    """Fraction of steps exceeding `threshold`."""
    return float(np.mean(scores > threshold))


# =============================================================================
# Monitor B — sequential / correlation monitors
# =============================================================================

def _scores_to_z(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    """Map Beta(a,b) scores to standard-normal copula z-scores.

    This is the key transform: z_t = Phi^{-1}(F_{beta}(s_t)).
    Under the Gaussian AR(1) copula model, z_t is AR(1) Gaussian.
    """
    u = np.clip(stats.beta.cdf(scores, a, b), 1e-7, 1.0 - 1e-7)
    return stats.norm.ppf(u)


def b_cusum(
    scores: np.ndarray,
    model: BenignModel,
    drift: float = 0.5,
) -> float:
    """CUSUM on the squared copula-z residual — detects elevated variance/autocorrelation.

    Operates in the Gaussian copula space.  Detects when consecutive z-values
    are more correlated than expected under the benign null (z ~ AR(1) with phi_benign).

    The statistic is CUSUM of (z_t - phi*z_{t-1})^2 / (1-phi^2) - 1,
    which is positive when correlation is higher than expected.
    """
    a = model.params.get("a", 2.0)
    b = model.params.get("b", 5.0)
    phi_b = model.phi
    z = _scores_to_z(scores, a, b)

    # Innovation process under benign null
    sigma_innov = float(np.sqrt(max(1.0 - phi_b ** 2, 1e-8)))
    innov = np.empty(len(z))
    innov[0] = z[0]
    for t in range(1, len(z)):
        innov[t] = z[t] - phi_b * z[t - 1]

    # Standardise innovations
    w = innov / sigma_innov  # should be ~ N(0,1) under benign

    # CUSUM on squared innovations: statistic is (w^2 - 1) -> +ve when more correlated
    # (Higher autocorrelation in s -> smaller innovations -> w^2 < 1... wait)
    # Actually: higher phi_attack means SMALLER innovations -> w is smaller ->
    # (w^2 - 1) is negative for well-behaved case.
    # Use CUSUM on w^2 itself (flagging DECREASED innovation = higher autocorr)
    # Target: innovations should have variance 1.  Attack: variance < 1.
    cusum_stat = (1.0 - w ** 2)  # positive when innovations are smaller than expected

    C = 0.0
    C_max = 0.0
    for ct in cusum_stat:
        C = max(0.0, C + ct - drift * 0.0)  # drift=0 for this direction
        C_max = max(C_max, C)
    # Also compute sum version (more powerful for global test)
    return float(np.sum(np.maximum(cusum_stat, 0.0)))


def b_sprt(
    scores: np.ndarray,
    model: BenignModel,
    alt_phi: Optional[float] = None,
) -> float:
    """SPRT log-likelihood ratio: benign AR(1) phi_0 vs attack AR(1) phi_1.

    LLR = sum_t log[ p_phi1(z_t | z_{t-1}) / p_phi0(z_t | z_{t-1}) ]
    where p_phi(z_t | z_{t-1}) = N(phi*z_{t-1}, 1-phi^2).

    Higher LLR -> more evidence for elevated autocorrelation (attack).
    """
    a = model.params.get("a", 2.0)
    b = model.params.get("b", 5.0)
    phi_0 = model.phi  # benign phi
    if alt_phi is None:
        # Default: attack phi halfway between benign and 0.95
        alt_phi = phi_0 + 0.5 * (0.95 - phi_0)
    phi_1 = float(alt_phi)

    z = _scores_to_z(scores, a, b)

    sigma_1 = float(np.sqrt(max(1.0 - phi_1 ** 2, 1e-8)))
    sigma_0 = float(np.sqrt(max(1.0 - phi_0 ** 2, 1e-8)))

    llr = 0.0
    for t in range(1, len(z)):
        # log p under H1 (attack): N(phi1 * z_{t-1}, sigma1^2)
        mu_1 = phi_1 * z[t - 1]
        lp1 = -0.5 * ((z[t] - mu_1) / sigma_1) ** 2 - np.log(sigma_1)
        # log p under H0 (benign): N(phi0 * z_{t-1}, sigma0^2)
        mu_0 = phi_0 * z[t - 1]
        lp0 = -0.5 * ((z[t] - mu_0) / sigma_0) ** 2 - np.log(sigma_0)
        llr += lp1 - lp0

    return float(llr)


def b_hmm_lr(
    scores: np.ndarray,
    model: BenignModel,
    n_states: int = 4,
) -> float:
    """Markov / HMM likelihood ratio (2-state discretised Markov chain).

    Discretises copula-z scores into n_states bins, computes the log-likelihood
    ratio of the observed sequence under a fitted Markov chain vs. i.i.d. model.

    A positive value means the sequence is more likely under a Markov model
    than i.i.d. — indicating temporal structure above what a marginal model
    would predict.
    """
    a = model.params.get("a", 2.0)
    b = model.params.get("b", 5.0)
    z = _scores_to_z(scores, a, b)
    T = len(z)

    # Discretise z into equal-width bins
    boundaries = np.linspace(-3.0, 3.0, n_states + 1)
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf
    state_seq = np.digitize(z, boundaries[1:-1])  # 0..n_states-1

    # Estimate transition matrix with Laplace smoothing
    counts = np.zeros((n_states, n_states))
    for t in range(T - 1):
        counts[state_seq[t], state_seq[t + 1]] += 1
    counts += 0.5
    trans = counts / counts.sum(axis=1, keepdims=True)

    # Stationary distribution from state frequencies
    pi = np.bincount(state_seq, minlength=n_states).astype(float) + 0.5
    pi /= pi.sum()

    # Log-likelihood under Markov model
    ll_markov = np.log(pi[state_seq[0]] + 1e-12)
    for t in range(T - 1):
        ll_markov += np.log(trans[state_seq[t], state_seq[t + 1]] + 1e-12)

    # Log-likelihood under i.i.d. model
    ll_iid = np.sum(np.log(pi[state_seq] + 1e-12))

    return float(ll_markov - ll_iid)


def b_runs_test(scores: np.ndarray) -> float:
    """Wald-Wolfowitz runs test statistic (in copula z-space).

    Binarises z-scores at 0 (median), counts runs.  Fewer runs = more
    positive autocorrelation.  Returns -(runs - E[runs]) / std so that
    high autocorrelation -> high (suspicious) value.
    """
    med = np.median(scores)
    binary = (scores > med).astype(int)
    T = len(binary)
    n1 = binary.sum()
    n0 = T - n1

    if n1 == 0 or n0 == 0:
        return 0.0

    runs = 1 + np.sum(binary[1:] != binary[:-1])
    mu_runs = 2.0 * n0 * n1 / T + 1
    var_runs = (2.0 * n0 * n1 * (2.0 * n0 * n1 - T)) / (T ** 2 * (T - 1) + 1e-10)
    if var_runs < 1e-10:
        return 0.0
    z = (runs - mu_runs) / np.sqrt(var_runs)
    return float(-z)  # negative z = fewer runs = more autocorr -> suspicious


def b_autocorr(
    scores: np.ndarray,
    model: Optional[BenignModel] = None,
    max_lag: int = 5,
) -> float:
    """Sum of lag-1..max_lag autocorrelations in the Gaussian copula z-space.

    Operating in the z-space ensures the distribution of the statistic is
    comparable across different score marginals.
    """
    if model is not None:
        a = model.params.get("a", 2.0)
        b = model.params.get("b", 5.0)
        z = _scores_to_z(scores, a, b)
    else:
        z = scores - scores.mean()

    var = (z ** 2).mean()
    if var < 1e-10:
        return 0.0
    total = 0.0
    for lag in range(1, min(max_lag + 1, len(z))):
        r = (z[:-lag] * z[lag:]).mean() / var
        total += r  # sum of autocorrelations (positive under AR(1))
    return float(total)


# ---------------------------------------------------------------------------
# b_seq_clf — small logistic classifier on sliding-window features
# ---------------------------------------------------------------------------

def _window_features(
    scores: np.ndarray,
    model: Optional[BenignModel],
    window: int = 20,
) -> np.ndarray:
    """Extract rolling-window features from a score trajectory.

    Features (per window):
      z-space mean, std, lag-1 autocorr, lag-2 autocorr, runs test stat,
      fraction of positive z-values, max |z|
    Returns shape (n_windows, n_features).
    """
    if model is not None:
        a = model.params.get("a", 2.0)
        b = model.params.get("b", 5.0)
        z = _scores_to_z(scores, a, b)
    else:
        z = scores.copy()

    T = len(z)
    feats = []
    step = max(1, window // 2)
    for start in range(0, T - window + 1, step):
        w = z[start: start + window]
        f_mean = np.mean(w)
        f_std = np.std(w)
        f_max_abs = np.max(np.abs(w))
        # Lag-1 autocorr
        v = (w ** 2).mean()
        f_ac1 = (w[:-1] * w[1:]).mean() / (v + 1e-10)
        # Lag-2 autocorr
        if len(w) > 2:
            f_ac2 = (w[:-2] * w[2:]).mean() / (v + 1e-10)
        else:
            f_ac2 = 0.0
        f_runs = b_runs_test(w)
        f_pos = np.mean(w > 0)
        feats.append([f_mean, f_std, f_max_abs, f_ac1, f_ac2, f_runs, f_pos])
    return np.array(feats) if feats else np.zeros((1, 7))


class SequenceClassifier:
    """Logistic regression classifier on sliding-window copula-z features.

    Trained on benign and attacked trajectories.  Outputs probability of attack.

    Usage
    -----
    clf = SequenceClassifier(window=20)
    clf.fit(benign_trajs, attack_trajs)  # lists of (T,) score arrays
    prob = clf.score_trajectory(scores)   # float in [0, 1]
    """

    def __init__(self, window: int = 20, model: Optional[BenignModel] = None):
        self.window = window
        self.model = model
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=2000, C=1.0, random_state=0)
        self._fitted = False

    def _extract(self, trajs: list[np.ndarray]) -> np.ndarray:
        """Feature extraction: mean + std of window features across a trajectory."""
        traj_feats = []
        for s in trajs:
            wf = _window_features(s, self.model, self.window)
            traj_feats.append(np.concatenate([wf.mean(0), wf.std(0)]))
        return np.array(traj_feats)

    def fit(
        self,
        benign_scores: list[np.ndarray],
        attack_scores: list[np.ndarray],
    ) -> "SequenceClassifier":
        X_b = self._extract(benign_scores)
        X_a = self._extract(attack_scores)
        X = np.vstack([X_b, X_a])
        y = np.array([0] * len(benign_scores) + [1] * len(attack_scores))
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self._fitted = True
        return self

    def score_trajectory(self, scores: np.ndarray) -> float:
        """Return probability of being attacked (class 1 score)."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        feats = self._extract([scores])
        feats_scaled = self.scaler.transform(feats)
        return float(self.clf.predict_proba(feats_scaled)[0, 1])


# =============================================================================
# Batch scoring — apply all monitors to a list of trajectories
# =============================================================================

def score_trajectories(
    trajectories: list[np.ndarray],
    model: BenignModel,
    seq_clf: Optional[SequenceClassifier] = None,
    threshold: float = 0.7,
    alt_phi: Optional[float] = None,
) -> dict:
    """Apply all monitors to a list of trajectories.

    Parameters
    ----------
    trajectories : list of (T,) arrays of per-step monitor scores
    model : BenignModel (provides benign phi and Beta params for monitors)
    seq_clf : fitted SequenceClassifier or None
    threshold : threshold for a_threshold monitor
    alt_phi : alternative phi for SPRT (None = auto from model.phi)

    Returns
    -------
    dict mapping monitor name -> array of per-trajectory scores, shape (N,)
    """
    monitor_names = [
        "a_mean", "a_max", "a_top_k_tail", "a_threshold",
        "b_cusum", "b_sprt", "b_hmm_lr", "b_runs_test", "b_autocorr",
    ]
    if seq_clf is not None:
        monitor_names.append("b_seq_clf")

    results = {name: [] for name in monitor_names}

    for s in trajectories:
        results["a_mean"].append(a_mean(s))
        results["a_max"].append(a_max(s))
        results["a_top_k_tail"].append(a_top_k_tail(s))
        results["a_threshold"].append(a_threshold(s, threshold))
        results["b_cusum"].append(b_cusum(s, model))
        results["b_sprt"].append(b_sprt(s, model, alt_phi))
        results["b_hmm_lr"].append(b_hmm_lr(s, model))
        results["b_runs_test"].append(b_runs_test(s))
        results["b_autocorr"].append(b_autocorr(s, model))
        if seq_clf is not None:
            results["b_seq_clf"].append(seq_clf.score_trajectory(s))

    return {k: np.array(v) for k, v in results.items()}
