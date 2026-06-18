import pytest

from ai_eval_cookbook.exact_match_accuracy import (
    accuracy,
    correctness_mask,
    exact_match,
    normalize_text,
)


def test_normalize_lowercases_strips_punct_and_articles():
    assert normalize_text("The capital is Paris.") == "capital is paris"
    assert normalize_text("  An  Apple ") == "apple"
    assert normalize_text("YES!!!") == "yes"


def test_normalize_handles_none_and_empty():
    assert normalize_text(None) == ""
    assert normalize_text("") == ""


def test_exact_match_single_reference():
    assert exact_match("Paris", "paris.") is True
    assert exact_match("the Paris", "Paris") is True
    assert exact_match("London", "Paris") is False


def test_exact_match_multi_reference():
    assert exact_match("green", ["red", "Green!"]) is True
    assert exact_match("blue", ["red", "green"]) is False


def test_exact_match_custom_normalizer_is_case_sensitive():
    # Identity normalizer -> raw comparison.
    assert exact_match("Paris", "paris", normalizer=str) is False
    assert exact_match("Paris", "Paris", normalizer=str) is True


def test_accuracy_hand_computed():
    preds = ["Paris", "yes", "blue", "42"]
    refs = ["paris.", "Yes", ["red", "green"], "42"]
    # matches: Paris/paris. yes/Yes 42/42 -> 3 of 4
    assert accuracy(preds, refs) == pytest.approx(3 / 4)


def test_correctness_mask():
    preds = ["a", "b"]
    refs = ["A", "z"]
    assert correctness_mask(preds, refs) == [True, False]


def test_accuracy_empty_is_zero():
    assert accuracy([], []) == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        correctness_mask(["a"], [])
