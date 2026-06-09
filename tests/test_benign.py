"""tests/test_benign.py — Sanity checks for the benign module."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.benign import default_benign_model, fit_benign, sample_benign


def test_sample_shape():
    model = default_benign_model()
    rng = np.random.default_rng(0)
    scores = sample_benign(model, n_traj=10, T=50, rng=rng)
    assert scores.shape == (10, 50), f"Expected (10, 50), got {scores.shape}"
    assert np.all(scores >= 0) and np.all(scores <= 1), "Scores must be in [0, 1]"
    print("test_sample_shape: PASS")


def test_sample_marginal():
    """Check that sampled scores have roughly the expected mean."""
    model = default_benign_model(mean=0.25, var=0.02, phi=0.0)
    rng = np.random.default_rng(1)
    scores = sample_benign(model, n_traj=1000, T=100, rng=rng)
    sample_mean = scores.mean()
    assert abs(sample_mean - 0.25) < 0.02, (
        f"Expected mean ~0.25, got {sample_mean:.4f}"
    )
    print(f"test_sample_marginal: PASS (mean={sample_mean:.4f})")


def test_fit_benign():
    """Check that fitting recovers approximate parameters."""
    model_true = default_benign_model(mean=0.3, var=0.025, phi=0.0)
    rng = np.random.default_rng(2)
    data = sample_benign(model_true, n_traj=200, T=100, rng=rng)
    model_fit = fit_benign(data, dist_name="beta", fit_autocorr=False)
    assert abs(model_fit.mean - 0.3) < 0.05, (
        f"Expected mean ~0.3, got {model_fit.mean:.4f}"
    )
    print(f"test_fit_benign: PASS (fitted mean={model_fit.mean:.4f})")


def test_ar1_autocorr():
    """Check that AR(1) model introduces positive autocorrelation."""
    model_iid = default_benign_model(mean=0.25, var=0.02, phi=0.0)
    model_ar1 = default_benign_model(mean=0.25, var=0.02, phi=0.5)
    rng = np.random.default_rng(3)
    scores_iid = sample_benign(model_iid, n_traj=200, T=100, rng=rng)
    scores_ar1 = sample_benign(model_ar1, n_traj=200, T=100, rng=rng)

    # Compute mean lag-1 autocorr
    def mean_lag1(arr):
        s = arr - arr.mean(axis=1, keepdims=True)
        return np.mean(s[:, :-1] * s[:, 1:]) / (np.mean(s ** 2) + 1e-10)

    ac_iid = mean_lag1(scores_iid)
    ac_ar1 = mean_lag1(scores_ar1)
    assert ac_ar1 > ac_iid, (
        f"AR(1) model should have higher autocorr: {ac_ar1:.4f} vs {ac_iid:.4f}"
    )
    print(f"test_ar1_autocorr: PASS (ar1={ac_ar1:.4f} > iid={ac_iid:.4f})")


if __name__ == "__main__":
    test_sample_shape()
    test_sample_marginal()
    test_fit_benign()
    test_ar1_autocorr()
    print("\nAll benign tests passed.")
