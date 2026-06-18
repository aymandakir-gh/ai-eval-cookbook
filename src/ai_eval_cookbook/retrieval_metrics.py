"""Retrieval / ranking metrics: precision@k, recall@k, MRR, and nDCG.

These evaluate a *ranked list of retrieved item ids* against a set of relevant ids
(or graded relevances). They underpin RAG retriever evaluation and search quality.

- **precision@k**: of the top-k retrieved, the fraction that are relevant.
- **recall@k**: of all relevant items, the fraction found in the top-k.
- **reciprocal rank**: 1 / (rank of the first relevant item); MRR averages it.
- **nDCG@k**: rank-aware, gain-discounted score normalized by the ideal ordering.
  Supports binary relevance (relevant set) or graded relevance (id -> grade).

Pure standard library, offline. No model is called; you supply rankings and labels.
"""

from __future__ import annotations

import math
from typing import Dict, Hashable, List, Mapping, Sequence, Set, Union

Item = Hashable
RelevanceSet = Union[Set[Item], Sequence[Item]]
GradedRelevance = Mapping[Item, float]


def _as_set(relevant: RelevanceSet) -> Set[Item]:
    return relevant if isinstance(relevant, set) else set(relevant)


def precision_at_k(ranked: Sequence[Item], relevant: RelevanceSet, k: int) -> float:
    """Fraction of the top-k retrieved items that are relevant.

    Denominator is ``k`` (the standard definition), so retrieving fewer than k
    items dilutes precision. ``k`` must be positive.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    rel = _as_set(relevant)
    top = ranked[:k]
    hits = sum(1 for item in top if item in rel)
    return hits / k


def recall_at_k(ranked: Sequence[Item], relevant: RelevanceSet, k: int) -> float:
    """Fraction of all relevant items that appear in the top-k.

    Returns 0.0 when there are no relevant items (nothing to recall)."""
    if k <= 0:
        raise ValueError("k must be positive")
    rel = _as_set(relevant)
    if not rel:
        return 0.0
    top = set(ranked[:k])
    return len(top & rel) / len(rel)


def reciprocal_rank(ranked: Sequence[Item], relevant: RelevanceSet) -> float:
    """1 / rank of the first relevant item (1-indexed); 0.0 if none relevant."""
    rel = _as_set(relevant)
    for i, item in enumerate(ranked, start=1):
        if item in rel:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(
    rankings: Sequence[Sequence[Item]], relevants: Sequence[RelevanceSet]
) -> float:
    """Average reciprocal rank across queries."""
    if len(rankings) != len(relevants):
        raise ValueError("rankings and relevants must have equal length")
    if not rankings:
        return 0.0
    return sum(
        reciprocal_rank(r, rel) for r, rel in zip(rankings, relevants)
    ) / len(rankings)


def _gain(item: Item, relevance: Union[Set[Item], GradedRelevance]) -> float:
    if isinstance(relevance, Mapping):
        return float(relevance.get(item, 0.0))
    return 1.0 if item in relevance else 0.0


def dcg_at_k(
    ranked: Sequence[Item],
    relevance: Union[RelevanceSet, GradedRelevance],
    k: int,
) -> float:
    """Discounted cumulative gain over the top-k using the (2^rel - 1) gain form.

    Discount is log2(rank + 1) with 1-indexed rank.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    rel = relevance if isinstance(relevance, Mapping) else _as_set(relevance)
    dcg = 0.0
    for i, item in enumerate(ranked[:k], start=1):
        g = _gain(item, rel)
        if g:
            dcg += (2 ** g - 1) / math.log2(i + 1)
    return dcg


def ndcg_at_k(
    ranked: Sequence[Item],
    relevance: Union[RelevanceSet, GradedRelevance],
    k: int,
) -> float:
    """Normalized DCG@k = DCG@k / ideal-DCG@k, in [0, 1].

    The ideal DCG sorts all relevant items by descending grade. Returns 0.0 when no
    relevant items exist (ideal DCG is 0).
    """
    dcg = dcg_at_k(ranked, relevance, k)
    if isinstance(relevance, Mapping):
        ideal_grades = sorted(
            (g for g in relevance.values() if g > 0), reverse=True
        )
        ideal_order = list(range(len(ideal_grades)))  # placeholder ids
        idcg = 0.0
        for i, g in enumerate(ideal_grades[:k], start=1):
            idcg += (2 ** g - 1) / math.log2(i + 1)
    else:
        rel = _as_set(relevance)
        n_ideal = min(len(rel), k)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_ideal + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def retrieval_report(
    ranked: Sequence[Item],
    relevant: RelevanceSet,
    k: int,
) -> Dict[str, float]:
    """Bundle precision@k, recall@k, reciprocal rank, and binary nDCG@k."""
    return {
        "precision@k": precision_at_k(ranked, relevant, k),
        "recall@k": recall_at_k(ranked, relevant, k),
        "reciprocal_rank": reciprocal_rank(ranked, relevant),
        "ndcg@k": ndcg_at_k(ranked, relevant, k),
    }


if __name__ == "__main__":
    ranked = ["d1", "d2", "d3", "d4", "d5"]
    relevant = {"d2", "d4"}
    print("retrieval_report @3:", retrieval_report(ranked, relevant, k=3))

    graded = {"d1": 3.0, "d2": 2.0, "d3": 0.0, "d4": 1.0}
    print("graded nDCG@4:", round(ndcg_at_k(ranked, graded, 4), 4))

    rankings = [["a", "b"], ["x", "y", "z"]]
    rels = [{"b"}, {"z"}]
    print("MRR:", round(mean_reciprocal_rank(rankings, rels), 4))
