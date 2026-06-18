"""Calibration metrics for predicted confidences: ECE, MCE, Brier score, and
reliability bins.

A model is *calibrated* when its stated confidence matches its empirical accuracy:
of the predictions it makes with 80% confidence, about 80% should be correct. This
matters for LLMs that emit verbalized or token-probability confidences used for
abstention, routing, or risk control.

Given paired (confidence, correct) records this module computes:

- **Reliability bins**: split [0, 1] into bins, report each bin's mean confidence,
  accuracy, and gap.
- **Expected Calibration Error (ECE)**: sample-weighted average |confidence −
  accuracy| across bins.
- **Maximum Calibration Error (MCE)**: the worst bin gap.
- **Brier score**: mean squared error between confidence and outcome (a proper
  scoring rule that rewards both calibration and sharpness).

Pure standard library, offline.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def _validate(confidences: Sequence[float], correct: Sequence[bool]) -> None:
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have equal length")
    for c in confidences:
        if not 0.0 <= c <= 1.0:
            raise ValueError("confidences must be in [0, 1], got %r" % (c,))


def reliability_bins(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    """Bin predictions by confidence and report per-bin statistics.

    Each bin dict has: ``lower``, ``upper``, ``count``, ``mean_confidence``,
    ``accuracy``, ``gap`` (|accuracy − mean_confidence|). Bins are equal-width over
    [0, 1]; a confidence of exactly 1.0 falls in the last bin.
    """
    _validate(confidences, correct)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    bins: List[Dict[str, float]] = []
    width = 1.0 / n_bins
    # accumulate
    sums_conf = [0.0] * n_bins
    sums_acc = [0.0] * n_bins
    counts = [0] * n_bins
    for c, y in zip(confidences, correct):
        b = min(int(c / width), n_bins - 1)
        sums_conf[b] += c
        sums_acc[b] += 1.0 if y else 0.0
        counts[b] += 1
    for b in range(n_bins):
        count = counts[b]
        mean_conf = sums_conf[b] / count if count else 0.0
        acc = sums_acc[b] / count if count else 0.0
        bins.append(
            {
                "lower": b * width,
                "upper": (b + 1) * width,
                "count": count,
                "mean_confidence": mean_conf,
                "accuracy": acc,
                "gap": abs(acc - mean_conf),
            }
        )
    return bins


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """Sample-weighted mean bin gap (ECE). 0.0 = perfectly calibrated.

    ECE = sum over bins of (count_b / N) * |accuracy_b − mean_confidence_b|.
    Empty input -> 0.0.
    """
    _validate(confidences, correct)
    n = len(confidences)
    if n == 0:
        return 0.0
    bins = reliability_bins(confidences, correct, n_bins)
    return sum((b["count"] / n) * b["gap"] for b in bins)


def maximum_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """Largest gap of any non-empty bin (MCE). Empty input -> 0.0."""
    _validate(confidences, correct)
    if not confidences:
        return 0.0
    bins = reliability_bins(confidences, correct, n_bins)
    gaps = [b["gap"] for b in bins if b["count"] > 0]
    return max(gaps) if gaps else 0.0


def brier_score(
    confidences: Sequence[float], correct: Sequence[bool]
) -> float:
    """Mean squared error between confidence and outcome (0 best, 1 worst).

    For binary outcomes: mean((confidence − y)^2) with y in {0, 1}.
    """
    _validate(confidences, correct)
    n = len(confidences)
    if n == 0:
        return 0.0
    return sum((c - (1.0 if y else 0.0)) ** 2 for c, y in zip(confidences, correct)) / n


def calibration_report(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> Dict[str, object]:
    """Bundle ECE, MCE, Brier, overall accuracy, mean confidence, and bins."""
    _validate(confidences, correct)
    n = len(confidences)
    return {
        "ece": expected_calibration_error(confidences, correct, n_bins),
        "mce": maximum_calibration_error(confidences, correct, n_bins),
        "brier": brier_score(confidences, correct),
        "accuracy": (sum(1 for y in correct if y) / n) if n else 0.0,
        "mean_confidence": (sum(confidences) / n) if n else 0.0,
        "bins": reliability_bins(confidences, correct, n_bins),
    }


if __name__ == "__main__":
    # Overconfident model: says 0.9 but is only right ~60% of the time.
    confidences = [0.9, 0.9, 0.9, 0.9, 0.9, 0.6, 0.6, 0.6, 0.3, 0.3]
    correct = [True, True, True, False, False, True, False, False, False, True]
    report = calibration_report(confidences, correct, n_bins=5)
    print("ECE:", round(report["ece"], 4))
    print("MCE:", round(report["mce"], 4))
    print("Brier:", round(report["brier"], 4))
    print("accuracy:", report["accuracy"], "mean conf:", report["mean_confidence"])
