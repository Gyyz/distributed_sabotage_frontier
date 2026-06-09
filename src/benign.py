"""benign.py — Benign-agent trajectory model.

Models per-step monitor scores s_1 .. s_T from a Beta(a,b) marginal with
AR(1) autocorrelation via a Gaussian copula.  The latent process is:

    z_t = phi * z_{t-1} + sqrt(1-phi^2) * epsilon_t,   epsilon_t ~ N(0,1)
    u_t = Phi(z_t)                                        (uniform copula)
    s_t = F^{-1}(u_t)                                     (Beta quantile)

This guarantees:
  - Per-step marginal of s_t is exactly Beta(a, b)
  - Lag-k autocorrelation in the z-space is phi^k
  - Any function of the marginal only (mean, max, quantile) is identically
    distributed for all values of phi -- so Monitor A cannot distinguish
    two Gaussian-copula processes that differ only in phi.

Provides:
  BenignModel        — fitted model container
  fit_benign         — estimate marginal + autocorrelation from data
  sample_benign      — draw a batch of trajectories
  default_benign_model — construct without data
"""
from __future__ import annotations

import numpy as np
from numpy.random import Generator
from dataclasses import dataclass, field
from typing import Tuple, Optional
from scipy import stats


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class BenignModel:
    """Container for a fitted benign-score distribution.

    Attributes
    ----------
    dist_name : str
        One of ``"beta"`` | ``"truncnorm"``.
    params : dict
        Distribution parameters.
    phi : float
        AR(1) coefficient for temporal autocorrelation, in [0, 1).
        This is the parameter the SPRT tests under the benign hypothesis.
    mean : float
        Marginal mean of the fitted distribution.
    var : float
        Marginal variance.
    """
    dist_name: str
    params: dict
    phi: float = 0.0
    mean: float = 0.0
    var: float = 0.0


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_benign(
    data: np.ndarray,
    dist_name: str = "beta",
    fit_autocorr: bool = True,
) -> BenignModel:
    """Fit a benign-score model to observed trajectory data.

    Parameters
    ----------
    data : array, shape (N, T) or (N*T,)
        Observed per-step monitor scores in (0, 1).
    dist_name : str
        Distribution family to fit.
    fit_autocorr : bool
        Whether to estimate the AR(1) coefficient from trajectory residuals
        in the copula (Gaussian) space.

    Returns
    -------
    BenignModel
    """
    data = np.asarray(data, dtype=float)
    flat = data.ravel()

    # Clip to avoid Beta boundary issues
    flat = np.clip(flat, 1e-6, 1 - 1e-6)

    if dist_name == "beta":
        a, b, loc, scale = stats.beta.fit(flat, floc=0, fscale=1)
        params = {"a": float(a), "b": float(b)}
        mean = a / (a + b)
        var = a * b / ((a + b) ** 2 * (a + b + 1))
    elif dist_name == "truncnorm":
        mu = float(np.mean(flat))
        sigma = float(np.std(flat))
        params = {"mu": mu, "sigma": sigma}
        mean = mu
        var = sigma ** 2
    else:
        raise ValueError(f"Unknown dist_name: {dist_name!r}")

    # Estimate AR(1) from copula-space (Gaussian) z-scores
    phi = 0.0
    if fit_autocorr and data.ndim == 2 and data.shape[1] > 2:
        a_param = params.get("a", 2.0)
        b_param = params.get("b", 5.0)
        data_clipped = np.clip(data, 1e-7, 1 - 1e-7)
        u = stats.beta.cdf(data_clipped, a_param, b_param)
        u = np.clip(u, 1e-7, 1 - 1e-7)
        z = stats.norm.ppf(u)
        # Pooled lag-1 autocorrelation
        r0 = np.mean(z[:, :-1] * z[:, 1:])
        r_var = np.mean(z ** 2)
        phi = float(np.clip(r0 / (r_var + 1e-12), 0.0, 0.99))

    return BenignModel(
        dist_name=dist_name,
        params=params,
        phi=phi,
        mean=float(mean),
        var=float(var),
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_benign(
    model: BenignModel,
    n_traj: int,
    T: int,
    rng: Generator,
) -> np.ndarray:
    """Sample a batch of benign trajectories using the Gaussian AR(1) copula.

    The per-step marginal is exactly the fitted distribution (Beta or truncnorm).
    The temporal autocorrelation in the Gaussian copula space is model.phi.

    Parameters
    ----------
    model : BenignModel
    n_traj : int
    T : int
    rng : Generator

    Returns
    -------
    scores : ndarray, shape (n_traj, T), values in [0, 1]
    """
    phi = model.phi
    a = model.params.get("a", 2.0)
    b = model.params.get("b", 5.0)
    innov_std = float(np.sqrt(max(1.0 - phi * phi, 1e-8)))

    scores = np.empty((n_traj, T))

    for i in range(n_traj):
        # AR(1) in Gaussian space
        z = np.empty(T)
        z[0] = rng.standard_normal()
        for t in range(1, T):
            z[t] = phi * z[t - 1] + innov_std * rng.standard_normal()
        # Gaussian copula -> Beta marginal
        u = stats.norm.cdf(z)
        u = np.clip(u, 1e-7, 1.0 - 1e-7)
        if model.dist_name == "beta":
            scores[i] = stats.beta.ppf(u, a, b)
        elif model.dist_name == "truncnorm":
            mu = model.params["mu"]
            sigma = model.params["sigma"]
            s_raw = stats.norm.ppf(u) * sigma + mu
            scores[i] = np.clip(s_raw, 0.0, 1.0)

    return np.clip(scores, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Convenience: build a default benign model without data
# ---------------------------------------------------------------------------

def default_benign_model(
    mean: float = 0.286,
    var: float = 0.025,
    phi: float = 0.5,
) -> BenignModel:
    """Construct a BenignModel from desired mean/variance without fitting.

    Uses Beta method-of-moments for (a, b).

    Parameters
    ----------
    phi : float
        AR(1) autocorrelation coefficient for the Gaussian copula.
        Default 0.5 (moderate autocorrelation typical for agent traces).
    """
    # Beta MOM: a/(a+b) = mean, a*b/((a+b)^2*(a+b+1)) = var
    ab = mean * (1 - mean) / var - 1
    a = mean * ab
    b = (1 - mean) * ab
    return BenignModel(
        dist_name="beta",
        params={"a": float(a), "b": float(b)},
        phi=phi,
        mean=mean,
        var=var,
    )
