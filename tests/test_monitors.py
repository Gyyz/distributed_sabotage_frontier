"""tests/test_monitors.py — Sanity checks for the monitors module."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.benign import default_benign_model, sample_benign
from src.attack import marginal_preserving_attack, naive_attack
from src.monitors import (
    a_mean, a_max, a_top_k_tail, a_threshold,
    b_cusum, b_sprt, b_hmm_lr, b_runs_test, b_autocorr,
    score_trajectories, SequenceClassifier,
)
from sklearn.metrics import roc_auc_score


def test_monitor_a_scalar():
    """All Monitor A functions should return a scalar."""
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.5])
    assert isinstance(a_mean(s), float)
    assert isinstance(a_max(s), float)
    assert isinstance(a_top_k_tail(s, k=2), float)
    assert isinstance(a_threshold(s, threshold=0.5), float)
    print("test_monitor_a_scalar: PASS")


def test_monitor_b_scalar():
    """All Monitor B functions should return a scalar."""
    model = default_benign_model()
    s = np.array([0.2] * 20)
    assert isinstance(b_cusum(s, model), float)
    assert isinstance(b_sprt(s, model), float)
    assert isinstance(b_hmm_lr(s, model), float)
    assert isinstance(b_runs_test(s), float)
    assert isinstance(b_autocorr(s, model), float)
    print("test_monitor_b_scalar: PASS")


def test_cusum_detects_trend():
    """CUSUM should give higher score for correlated trajectory."""
    from src.attack import marginal_preserving_attack
    model = default_benign_model(mean=0.286, var=0.025, phi=0.5)
    rng = np.random.default_rng(0)
    s_benign = sample_benign(model, 1, 100, rng)[0]
    # Attack: elevated phi
    from src.attack import _sample_ar1_copula
    s_attack = _sample_ar1_copula(100, model.params["a"], model.params["b"], 0.95, rng)
    c_benign = b_cusum(s_benign, model)
    c_attack = b_cusum(s_attack, model)
    assert c_attack > c_benign, (
        f"CUSUM attack ({c_attack:.3f}) should > benign ({c_benign:.3f})"
    )
    print(f"test_cusum_detects_trend: PASS (attack={c_attack:.2f} > benign={c_benign:.2f})")


def test_autocorr_detects_structure():
    """b_autocorr should be higher for autocorrelated trajectories."""
    rng = np.random.default_rng(5)
    s_iid = rng.beta(2, 5, 100)
    # Smooth with rolling mean to introduce autocorrelation
    from scipy.ndimage import uniform_filter1d
    s_corr = uniform_filter1d(s_iid, size=10)
    s_corr = np.clip(s_corr, 0, 1)
    ac_iid = b_autocorr(s_iid)
    ac_corr = b_autocorr(s_corr)
    assert ac_corr > ac_iid, (
        f"Correlated ({ac_corr:.4f}) should > iid ({ac_iid:.4f})"
    )
    print(f"test_autocorr_detects_structure: PASS (corr={ac_corr:.4f} > iid={ac_iid:.4f})")


def test_naive_attack_detectable_by_monitor_a():
    """Monitor A (mean) should achieve AUC > 0.7 on naive attack."""
    model = default_benign_model(mean=0.286, var=0.025, phi=0.5)
    rng = np.random.default_rng(42)
    benign = sample_benign(model, 200, 100, rng)
    naive_trajs = naive_attack(model, 200, 100, 1.0, 0.2, rng=rng)
    benign_means = [a_mean(benign[i]) for i in range(200)]
    attack_means = [a_mean(t.scores) for t in naive_trajs]
    y_true = [0] * 200 + [1] * 200
    y_score = benign_means + attack_means
    auc = roc_auc_score(y_true, y_score)
    assert auc > 0.7, f"Monitor A AUC on naive attack should be > 0.7, got {auc:.3f}"
    print(f"test_naive_attack_detectable_by_monitor_a: PASS (AUC={auc:.3f})")


def test_sequence_classifier_trains():
    """SequenceClassifier should train without error."""
    model = default_benign_model(mean=0.286, var=0.025, phi=0.5)
    rng = np.random.default_rng(99)
    benign = sample_benign(model, 100, 100, rng)
    attack_trajs = marginal_preserving_attack(model, 100, 100, 1.0, 0.8, 50, rng)
    benign_list = [benign[i] for i in range(100)]
    attack_list = [t.scores for t in attack_trajs]
    clf = SequenceClassifier(window=20, model=model)
    clf.fit(benign_list, attack_list)
    score = clf.score_trajectory(benign[0])
    assert 0.0 <= score <= 1.0, f"Score should be in [0,1], got {score}"
    print(f"test_sequence_classifier_trains: PASS (score={score:.3f})")


if __name__ == "__main__":
    test_monitor_a_scalar()
    test_monitor_b_scalar()
    test_cusum_detects_trend()
    test_autocorr_detects_structure()
    test_naive_attack_detectable_by_monitor_a()
    test_sequence_classifier_trains()
    print("\nAll monitor tests passed.")
