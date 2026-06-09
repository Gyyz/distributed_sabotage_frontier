"""summarize_results.py — Print a brief summary of key results."""
import json
import os
import sys


def main(results_dir: str):
    # realizability
    try:
        with open(os.path.join(results_dir, "realizability.json")) as f:
            real = json.load(f)
        stealthy = real["stealthy"]
        low_ks = [r for r in stealthy if r["ks_distance"] < 0.05]
        if low_ks:
            max_harm = max(r["harm_budget"] for r in low_ks)
            print(f"Realizability: {len(low_ks)} harm levels with KS<0.05 "
                  f"(max harm={max_harm:.2f})")
        else:
            print("Realizability: no harm levels with KS<0.05 found")
    except Exception as e:
        print(f"Realizability: (could not parse: {e})")

    # monitor ab
    try:
        with open(os.path.join(results_dir, "monitor_ab.json")) as f:
            mab = json.load(f)
        mp = mab["marginal_preserving"]
        best_a = max(mp["aucs"].get(n, 0) for n in mp["aucs"] if n.startswith("a_"))
        best_b = max(mp["aucs"].get(n, 0) for n in mp["aucs"] if n.startswith("b_"))
        print(f"Monitor AB (marginal-preserving): best Monitor A AUC={best_a:.3f}, "
              f"best Monitor B AUC={best_b:.3f}")
    except Exception as e:
        print(f"Monitor AB: (could not parse: {e})")

    # marginal check
    try:
        with open(os.path.join(results_dir, "marginal.json")) as f:
            marg = json.load(f)
        best_ks = min(r["ks_distance"] for r in marg["stealthy"])
        naive_max = max(r["ks_distance"] for r in marg["naive"])
        print(f"Marginal: best stealthy KS={best_ks:.4f}, naive max KS={naive_max:.4f}")
    except Exception as e:
        print(f"Marginal: (could not parse: {e})")

    # power vs spread
    try:
        with open(os.path.join(results_dir, "power_vs_spread.json")) as f:
            pvs = json.load(f)
        results = pvs["results"]
        # At max spread, compare Monitor A vs B
        last = results[-1]
        a_auc = max(last["aucs"].get(n, 0) for n in last["aucs"] if n.startswith("a_"))
        b_auc = max(last["aucs"].get(n, 0) for n in last["aucs"] if n.startswith("b_"))
        print(f"Power vs spread (max spread={last['spread']}): "
              f"Monitor A AUC={a_auc:.3f}, Monitor B AUC={b_auc:.3f}")
    except Exception as e:
        print(f"Power vs spread: (could not parse: {e})")


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    main(results_dir)
