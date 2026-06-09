"""tests/test_attack.py — Sanity checks for the attack module."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats
from src.benign import default_benign_model
from src.attack import marginal_preserving_attack, naive_attack, ks_distance_to_benign


def test_marginal_preserving_ks():
    """Core claim: marginal-preserving attack should have low KS distance."""
    model = default_benign_model(mean=0.25, var=0.02, phi=0.05)
    rng = np.random.default_rng(42)
    trajs = marginal_preserving_attack(
        model, n_traj=500, T=200,
        harm_budget=1.0, stealth=0.8, spread=100,
        rng=rng,
    )
    ks = ks_distance_to_benign(trajs, model, rng=np.random.default_rng(0))
    assert ks < 0.08, f"Expected KS < 0.08, got {ks:.4f}"
    print(f"test_marginal_preserving_ks: PASS (KS={ks:.4f})")


def test_naive_ks_higher():
    """Naive attacker should have higher KS distance than stealthy."""
    model = default_benign_model(mean=0.25, var=0.02, phi=0.05)
    rng = np.random.default_rng(42)
    trajs_stealthy = marginal_preserving_attack(
        model, n_traj=300, T=200,
        harm_budget=1.0, stealth=0.8, spread=100, rng=rng,
    )
    trajs_naive = naive_attack(
        model, n_traj=300, T=200,
        harm_budget=1.0, score_lift=0.2, rng=np.random.default_rng(1),
    )
    ks_s = ks_distance_to_benign(trajs_stealthy, model, rng=np.random.default_rng(2))
    ks_n = ks_distance_to_benign(trajs_naive, model, rng=np.random.default_rng(3))
    assert ks_n > ks_s, (
        f"Naive KS ({ks_n:.4f}) should be > stealthy KS ({ks_s:.4f})"
    )
    print(f"test_naive_ks_higher: PASS (naive KS={ks_n:.4f} > stealthy KS={ks_s:.4f})")


def test_attack_harm_budget():
    """Total harm should match the requested budget."""
    model = default_benign_model()
    rng = np.random.default_rng(7)
    budget = 2.0
    trajs = marginal_preserving_attack(
        model, n_traj=100, T=100,
        harm_budget=budget, stealth=0.5, spread=50, rng=rng,
    )
    for t in trajs:
        assert abs(t.total_harm - budget) < 1e-6, (
            f"Expected total_harm={budget}, got {t.total_harm:.6f}"
        )
    print(f"test_attack_harm_budget: PASS (all trajectories have harm={budget})")


def test_scores_in_range():
    """All scores should be in [0, 1]."""
    model = default_benign_model()
    rng = np.random.default_rng(11)
    trajs = marginal_preserving_attack(model, 50, 100, 1.0, 0.9, 50, rng)
    for t in trajs:
        assert np.all(t.scores >= 0.0) and np.all(t.scores <= 1.0), (
            "Scores outside [0, 1]"
        )
    print("test_scores_in_range: PASS")


if __name__ == "__main__":
    test_marginal_preserving_ks()
    test_naive_ks_higher()
    test_attack_harm_budget()
    test_scores_in_range()
    print("\nAll attack tests passed.")
