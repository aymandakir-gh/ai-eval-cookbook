import math

import pytest

from ai_eval_cookbook.retrieval_metrics import (
    dcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_report,
)

RANKED = ["d1", "d2", "d3", "d4", "d5"]
RELEVANT = {"d2", "d4"}


def test_precision_at_k():
    # top-3 = d1,d2,d3 ; only d2 relevant -> 1/3
    assert precision_at_k(RANKED, RELEVANT, 3) == pytest.approx(1 / 3)
    # top-5 -> d2,d4 relevant -> 2/5
    assert precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(2 / 5)


def test_recall_at_k():
    # top-3 finds d2 of {d2,d4} -> 1/2
    assert recall_at_k(RANKED, RELEVANT, 3) == pytest.approx(0.5)
    # top-5 finds both -> 1.0
    assert recall_at_k(RANKED, RELEVANT, 5) == pytest.approx(1.0)


def test_recall_no_relevant_is_zero():
    assert recall_at_k(RANKED, set(), 3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, RELEVANT) == pytest.approx(0.5)  # d2 at rank 2
    assert reciprocal_rank(RANKED, {"d1"}) == pytest.approx(1.0)
    assert reciprocal_rank(RANKED, {"zzz"}) == 0.0


def test_mean_reciprocal_rank():
    rankings = [["a", "b"], ["x", "y", "z"]]
    rels = [{"b"}, {"z"}]
    # (1/2 + 1/3) / 2
    assert mean_reciprocal_rank(rankings, rels) == pytest.approx((0.5 + 1 / 3) / 2)


def test_dcg_binary():
    # only d2 relevant at rank 2 -> (2^1-1)/log2(3)
    assert dcg_at_k(RANKED, RELEVANT, 3) == pytest.approx(1 / math.log2(3))


def test_ndcg_binary_hand_computed():
    dcg = 1 / math.log2(3)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)  # 2 relevant placed ideally
    assert ndcg_at_k(RANKED, RELEVANT, 3) == pytest.approx(dcg / idcg)


def test_ndcg_perfect_ordering_is_one():
    ranked = ["d2", "d4", "d1"]
    assert ndcg_at_k(ranked, RELEVANT, 3) == pytest.approx(1.0)


def test_ndcg_graded():
    graded = {"d1": 3.0, "d2": 2.0, "d3": 0.0, "d4": 1.0}
    score = ndcg_at_k(RANKED, graded, 4)
    assert 0.99 < score < 1.0  # near-ideal but d4 misplaced vs ideal


def test_ndcg_no_relevant_is_zero():
    assert ndcg_at_k(RANKED, set(), 3) == 0.0


def test_report_bundle():
    rep = retrieval_report(RANKED, RELEVANT, 3)
    assert set(rep) == {"precision@k", "recall@k", "reciprocal_rank", "ndcg@k"}
    assert rep["precision@k"] == pytest.approx(1 / 3)


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        precision_at_k(RANKED, RELEVANT, 0)
    with pytest.raises(ValueError):
        recall_at_k(RANKED, RELEVANT, -1)


def test_mrr_length_mismatch():
    with pytest.raises(ValueError):
        mean_reciprocal_rank([["a"]], [{"a"}, {"b"}])
