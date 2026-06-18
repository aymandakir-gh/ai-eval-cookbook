import math

import pytest

from ai_eval_cookbook.self_consistency import (
    agreement,
    majority_vote,
    normalized_entropy,
    self_consistency,
    vote_distribution,
)


def test_majority_vote():
    assert majority_vote(["42", "42", "41", "42"]) == "42"


def test_majority_vote_tie_first_appearance():
    # "no" appears first among the tied (2 vs 2)
    assert majority_vote(["no", "yes", "no", "yes"]) == "no"


def test_majority_vote_with_normalizer():
    norm = lambda s: s.strip().lower().rstrip(".")
    assert majority_vote(["Paris", "paris.", "PARIS"], normalizer=norm) == "paris"


def test_majority_vote_empty_raises():
    with pytest.raises(ValueError):
        majority_vote([])


def test_agreement():
    assert agreement(["42", "42", "41", "42"]) == pytest.approx(3 / 4)
    assert agreement(["a", "a", "a"]) == pytest.approx(1.0)
    assert agreement(["a", "b", "c"]) == pytest.approx(1 / 3)


def test_agreement_empty_is_zero():
    assert agreement([]) == 0.0


def test_vote_distribution():
    dist = vote_distribution(["a", "a", "b"])
    assert dist == {"a": pytest.approx(2 / 3), "b": pytest.approx(1 / 3)}


def test_entropy_identical_is_zero():
    assert normalized_entropy(["x", "x", "x"]) == 0.0


def test_entropy_uniform_is_one():
    assert normalized_entropy(["a", "b", "c", "d"]) == pytest.approx(1.0)


def test_entropy_even_split_half():
    # 2 answers, 2 each, of 4 samples -> H=1 bit, max=log2(4)=2 -> 0.5
    assert normalized_entropy(["yes", "no", "yes", "no"]) == pytest.approx(0.5)


def test_self_consistency_bundle_with_reference():
    res = self_consistency(["42", "42", "forty-two", "42", "41"], reference="42")
    assert res["answer"] == "42"
    assert res["agreement"] == pytest.approx(0.6)
    assert res["n_samples"] == 5
    assert res["correct"] is True


def test_self_consistency_incorrect_consensus():
    # consistently wrong
    res = self_consistency(["wrong", "wrong", "wrong"], reference="right")
    assert res["agreement"] == pytest.approx(1.0)
    assert res["correct"] is False


def test_self_consistency_no_reference_omits_correct():
    res = self_consistency(["a", "b"])
    assert "correct" not in res
