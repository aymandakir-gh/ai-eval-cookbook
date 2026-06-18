"""Self-consistency: aggregate multiple sampled answers by majority vote and measure
how much the samples agree.

Self-consistency (Wang et al., 2022) samples several reasoning paths for the same
prompt and takes the most common final answer. Beyond picking the answer, the
*agreement* among samples is a cheap, label-free confidence signal: high agreement
suggests the model is sure; a near-even split flags an uncertain or ambiguous item.

This module operates on already-sampled answer strings (you do the sampling). It
provides majority vote, an agreement fraction, normalized Shannon entropy of the
answer distribution, and per-sample correctness of the voted answer when a
reference is supplied.

Offline, standard library only.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence

Normalizer = Callable[[str], str]


def _normalize_all(samples: Sequence[str], normalizer: Optional[Normalizer]) -> List[str]:
    if normalizer is None:
        return [s.strip() for s in samples]
    return [normalizer(s) for s in samples]


def majority_vote(
    samples: Sequence[str], normalizer: Optional[Normalizer] = None
) -> str:
    """Return the most frequent answer. Ties are broken by first appearance.

    Raises ``ValueError`` on empty input.
    """
    if not samples:
        raise ValueError("samples must be non-empty")
    norm = _normalize_all(samples, normalizer)
    counts = Counter(norm)
    best = max(counts.values())
    for ans in norm:  # first-appearance tie-break
        if counts[ans] == best:
            return ans
    return norm[0]  # unreachable


def agreement(
    samples: Sequence[str], normalizer: Optional[Normalizer] = None
) -> float:
    """Fraction of samples equal to the majority answer, in (0, 1].

    1.0 = all samples agree; 1/n = maximally split. Empty input -> 0.0.
    """
    if not samples:
        return 0.0
    norm = _normalize_all(samples, normalizer)
    counts = Counter(norm)
    return max(counts.values()) / len(norm)


def vote_distribution(
    samples: Sequence[str], normalizer: Optional[Normalizer] = None
) -> Dict[str, float]:
    """Probability distribution over distinct answers (counts / total)."""
    if not samples:
        return {}
    norm = _normalize_all(samples, normalizer)
    counts = Counter(norm)
    total = len(norm)
    return {ans: c / total for ans, c in counts.items()}


def normalized_entropy(
    samples: Sequence[str], normalizer: Optional[Normalizer] = None
) -> float:
    """Shannon entropy of the answer distribution, normalized to [0, 1].

    0.0 = all samples identical (no uncertainty); 1.0 = every sample distinct
    (maximum uncertainty given the number of samples). A high value flags items
    where the model is unsure.
    """
    dist = vote_distribution(samples, normalizer)
    if len(dist) <= 1:
        return 0.0
    entropy = -sum(p * math.log2(p) for p in dist.values())
    max_entropy = math.log2(len(samples))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def self_consistency(
    samples: Sequence[str],
    reference: Optional[str] = None,
    normalizer: Optional[Normalizer] = None,
) -> Dict[str, object]:
    """Bundle the voted answer, agreement, entropy, distribution, and (if a
    reference is given) whether the voted answer is correct."""
    if not samples:
        raise ValueError("samples must be non-empty")
    voted = majority_vote(samples, normalizer)
    result: Dict[str, object] = {
        "answer": voted,
        "agreement": agreement(samples, normalizer),
        "entropy": normalized_entropy(samples, normalizer),
        "distribution": vote_distribution(samples, normalizer),
        "n_samples": len(samples),
    }
    if reference is not None:
        ref = normalizer(reference) if normalizer else reference.strip()
        result["correct"] = voted == ref
    return result


if __name__ == "__main__":
    samples = ["42", "42", "forty-two", "42", "41"]
    print("confident-ish:", self_consistency(samples, reference="42"))
    split = ["yes", "no", "yes", "no"]
    print("split:", self_consistency(split))
    norm = lambda s: s.strip().lower().rstrip(".")
    print("normalized:", self_consistency(["Paris", "paris.", "PARIS"], normalizer=norm))
