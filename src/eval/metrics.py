"""
Aggregate metrics with variance/confidence intervals — a single accuracy number is
exactly what the brief says NOT to present alone.
"""
import numpy as np


def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=42):
    """Bootstrap confidence interval for a mean metric over a small golden set.
    With only 150-250 examples, this matters: report the interval, not just the point estimate."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    boots = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return {"mean": float(values.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def intent_accuracy_by_bucket(df, pred_col="pred_intent", true_col="true_intent",
                               bucket_col="difficulty_bucket"):
    """Break accuracy down by difficulty bucket, not just overall — this is what
    usually reveals that a headline number is inflated by easy cases."""
    results = {}
    for bucket, group in df.groupby(bucket_col):
        acc = (group[pred_col] == group[true_col]).mean()
        results[bucket] = {"n": len(group), "accuracy": float(acc)}
    return results


def compare_to_baselines(agent_metric: float, trivial_metric: float, simple_metric: float) -> dict:
    return {
        "agent": agent_metric,
        "trivial_baseline": trivial_metric,
        "simple_baseline": simple_metric,
        "lift_over_trivial": agent_metric - trivial_metric,
        "lift_over_simple": agent_metric - simple_metric,
    }
