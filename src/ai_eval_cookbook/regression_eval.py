"""Regression evaluation: compare two runs against a golden set and classify each
example as a regression, an improvement, or unchanged.

Aggregate accuracy can stay flat while individual examples silently flip from right
to wrong. A golden-set diff makes the *churn* visible: the same overall score can
hide a 5% regression masked by a 5% improvement elsewhere. This is the metric you
gate releases on.

You supply, per example, the score (or correctness) of the baseline run and the
candidate run. An example is:

- **regression** if the candidate is worse than the baseline by more than a
  tolerance,
- **improvement** if better by more than the tolerance,
- **unchanged** otherwise.

Works with boolean correctness or any numeric score. Pure standard library, offline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

Score = Union[bool, float, int]


def _to_float(x: Score) -> float:
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    return float(x)


def classify(baseline: Score, candidate: Score, tol: float = 0.0) -> str:
    """Classify one example as 'regression', 'improvement', or 'unchanged'.

    Higher is better. ``tol`` is a dead-band: differences within +/- tol count as
    unchanged (use it to ignore numeric noise).
    """
    b, c = _to_float(baseline), _to_float(candidate)
    diff = c - b
    if diff > tol:
        return "improvement"
    if diff < -tol:
        return "regression"
    return "unchanged"


def diff_runs(
    baseline_scores: Sequence[Score],
    candidate_scores: Sequence[Score],
    ids: Optional[Sequence[Any]] = None,
    tol: float = 0.0,
) -> Dict[str, object]:
    """Diff two runs over a golden set.

    Returns counts of each category, the net change (improvements − regressions),
    baseline/candidate mean scores, and the example ids in each category (using
    positional indices if ``ids`` is not given).
    """
    if len(baseline_scores) != len(candidate_scores):
        raise ValueError("baseline and candidate must have equal length")
    n = len(baseline_scores)
    ident = list(ids) if ids is not None else list(range(n))
    if len(ident) != n:
        raise ValueError("ids must match the number of examples")

    buckets: Dict[str, List[Any]] = {
        "regression": [],
        "improvement": [],
        "unchanged": [],
    }
    for i in range(n):
        cat = classify(baseline_scores[i], candidate_scores[i], tol)
        buckets[cat].append(ident[i])

    base_mean = (sum(_to_float(x) for x in baseline_scores) / n) if n else 0.0
    cand_mean = (sum(_to_float(x) for x in candidate_scores) / n) if n else 0.0
    return {
        "n": n,
        "regressions": len(buckets["regression"]),
        "improvements": len(buckets["improvement"]),
        "unchanged": len(buckets["unchanged"]),
        "net_change": len(buckets["improvement"]) - len(buckets["regression"]),
        "baseline_mean": base_mean,
        "candidate_mean": cand_mean,
        "mean_delta": cand_mean - base_mean,
        "regression_ids": buckets["regression"],
        "improvement_ids": buckets["improvement"],
    }


def has_regressions(
    baseline_scores: Sequence[Score],
    candidate_scores: Sequence[Score],
    tol: float = 0.0,
    max_allowed: int = 0,
) -> bool:
    """CI gate: True if the number of regressions exceeds ``max_allowed``."""
    report = diff_runs(baseline_scores, candidate_scores, tol=tol)
    return report["regressions"] > max_allowed


if __name__ == "__main__":
    # baseline vs candidate correctness on a 6-example golden set
    baseline = [True, True, False, True, False, True]
    candidate = [True, False, True, True, False, True]
    report = diff_runs(baseline, candidate, ids=["q1", "q2", "q3", "q4", "q5", "q6"])
    print("regressions:", report["regressions"], report["regression_ids"])
    print("improvements:", report["improvements"], report["improvement_ids"])
    print("net change:", report["net_change"], "mean delta:", report["mean_delta"])
    print("would fail CI (0 allowed)?", has_regressions(baseline, candidate))
