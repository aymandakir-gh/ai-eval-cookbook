import pytest

from ai_eval_cookbook.answer_relevance import (
    answer_relevance,
    batch_answer_relevance,
    direct_relevance,
    echo_question_generator,
)

QUESTION = "What is the capital of France?"
GOOD = "The capital of France is Paris."
BAD = "France is a country with a rich culinary tradition."


def test_direct_relevance_in_unit_range():
    s = direct_relevance(QUESTION, GOOD)
    assert 0.0 <= s <= 1.0


def test_good_answer_scores_higher_than_off_topic():
    assert direct_relevance(QUESTION, GOOD) > direct_relevance(QUESTION, BAD)


def test_identical_question_answer_is_one():
    assert direct_relevance(QUESTION, QUESTION) == pytest.approx(1.0)


def test_echo_generator_reduces_to_direct():
    assert answer_relevance(QUESTION, GOOD) == pytest.approx(
        direct_relevance(QUESTION, GOOD)
    )


def test_echo_generator_returns_answer():
    assert echo_question_generator("foo bar") == ["foo bar"]


def test_injected_generator_averages_reverse_questions():
    gen = lambda a: ["What is the capital of France?", "Where is Paris located?"]
    score = answer_relevance(QUESTION, GOOD, question_generator=gen)
    # one reconstructed question is identical to the original -> high mean
    assert 0.0 <= score <= 1.0
    # mean of [1.0, something<=1.0] must be >= 0.5 here since first is exact match
    assert score >= 0.5


def test_empty_generator_output_is_zero():
    assert answer_relevance(QUESTION, GOOD, question_generator=lambda a: []) == 0.0


def test_injected_embedder():
    # constant embedder -> all cosines 1.0 -> relevance 1.0
    score = direct_relevance(QUESTION, BAD, embedder=lambda t: [1.0, 1.0])
    assert score == pytest.approx(1.0)


def test_batch_mean():
    qs = [QUESTION, QUESTION]
    ans = [GOOD, GOOD]
    expected = direct_relevance(QUESTION, GOOD)
    assert batch_answer_relevance(qs, ans) == pytest.approx(expected)


def test_batch_length_mismatch():
    with pytest.raises(ValueError):
        batch_answer_relevance(["a"], ["a", "b"])


def test_batch_empty_is_zero():
    assert batch_answer_relevance([], []) == 0.0


def test_negative_cosine_clamped():
    # embedder yielding opposite vectors -> cosine -1 -> clamped to 0
    flip = lambda t: [1.0, 0.0] if t == QUESTION else [-1.0, 0.0]
    assert direct_relevance(QUESTION, "x", embedder=flip) == 0.0
