"""A rubric-grading harness for LLM-as-a-judge evaluation.

LLM-as-a-judge (e.g. G-Eval, MT-Bench) scores free-form outputs against a rubric.
Because the model is the judge, the *harness* — not the model — is what you test and
reproduce: it defines the rubric, calls the judge, validates and clamps scores,
aggregates per-criterion and overall, and supports averaging repeated trials to
reduce judge variance.

This module is provider-agnostic: you inject a ``judge`` callable with the signature
``judge(prompt, response, criterion) -> score``. For offline testing it ships a
**deterministic mock judge** so the harness runs and is fully testable without any
model. In production, wrap your LLM call in a function with the same signature.

Offline, standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Callable, Dict, List, Optional, Sequence

# A judge maps (prompt, response, criterion) -> raw numeric score.
Judge = Callable[[str, str, "Criterion"], float]


@dataclass
class Criterion:
    """One rubric dimension."""

    name: str
    description: str = ""
    min_score: float = 1.0
    max_score: float = 5.0
    weight: float = 1.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def grade(
    prompt: str,
    response: str,
    rubric: Sequence[Criterion],
    judge: Judge,
    trials: int = 1,
) -> Dict[str, object]:
    """Grade one response against a rubric.

    Each criterion is scored by ``judge`` ``trials`` times; per-criterion scores are
    the mean of trials, clamped to the criterion range. The overall score is the
    weighted average of per-criterion scores, also normalized to [0, 1] via each
    criterion's own range.

    Returns ``{"per_criterion": {name: score}, "overall": weighted_mean,
    "overall_normalized": [0,1]}``.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if not rubric:
        raise ValueError("rubric must contain at least one criterion")

    per_criterion: Dict[str, float] = {}
    normalized: Dict[str, float] = {}
    for crit in rubric:
        raw = [judge(prompt, response, crit) for _ in range(trials)]
        score = _clamp(mean(raw), crit.min_score, crit.max_score)
        per_criterion[crit.name] = score
        span = crit.max_score - crit.min_score
        normalized[crit.name] = (
            (score - crit.min_score) / span if span > 0 else 1.0
        )

    total_weight = sum(c.weight for c in rubric)
    if total_weight == 0:
        raise ValueError("rubric weights must not sum to zero")
    overall = sum(
        per_criterion[c.name] * c.weight for c in rubric
    ) / total_weight
    overall_norm = sum(
        normalized[c.name] * c.weight for c in rubric
    ) / total_weight
    return {
        "per_criterion": per_criterion,
        "overall": overall,
        "overall_normalized": overall_norm,
    }


def grade_dataset(
    samples: Sequence[Dict[str, str]],
    rubric: Sequence[Criterion],
    judge: Judge,
    trials: int = 1,
) -> Dict[str, object]:
    """Grade many samples (each a dict with ``prompt`` and ``response``).

    Returns per-sample results plus mean per-criterion and mean overall-normalized
    scores across the dataset.
    """
    results = [
        grade(s["prompt"], s["response"], rubric, judge, trials) for s in samples
    ]
    if not results:
        return {"results": [], "mean_per_criterion": {}, "mean_overall_normalized": 0.0}
    mean_per_criterion = {
        c.name: mean(r["per_criterion"][c.name] for r in results) for c in rubric
    }
    mean_overall = mean(r["overall_normalized"] for r in results)
    return {
        "results": results,
        "mean_per_criterion": mean_per_criterion,
        "mean_overall_normalized": mean_overall,
    }


def keyword_mock_judge(
    positive: Optional[Sequence[str]] = None,
    negative: Optional[Sequence[str]] = None,
) -> Judge:
    """Build a deterministic mock judge for offline testing.

    Starts at the criterion midpoint, adds for each ``positive`` keyword found in the
    response and subtracts for each ``negative`` keyword, then clamps. Fully
    deterministic — same inputs always give the same score — so it is suitable for
    unit tests that must not call a real model.
    """
    pos = [w.lower() for w in (positive or [])]
    neg = [w.lower() for w in (negative or [])]

    def judge(prompt: str, response: str, criterion: Criterion) -> float:
        text = response.lower()
        mid = (criterion.min_score + criterion.max_score) / 2.0
        score = mid
        for w in pos:
            if w in text:
                score += 1.0
        for w in neg:
            if w in text:
                score -= 1.0
        return _clamp(score, criterion.min_score, criterion.max_score)

    return judge


if __name__ == "__main__":
    rubric = [
        Criterion("accuracy", "Factually correct", 1, 5, weight=2.0),
        Criterion("clarity", "Easy to understand", 1, 5, weight=1.0),
    ]
    judge = keyword_mock_judge(
        positive=["because", "clearly"], negative=["maybe", "dunno"]
    )
    good = "The sky is blue because of Rayleigh scattering, clearly explained."
    poor = "Maybe the sky is blue, dunno really."
    print("good:", grade("Why is the sky blue?", good, rubric, judge))
    print("poor:", grade("Why is the sky blue?", poor, rubric, judge))
