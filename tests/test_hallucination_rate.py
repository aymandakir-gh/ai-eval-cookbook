import pytest

from ai_eval_cookbook.hallucination_rate import (
    hallucination_rate,
    hallucination_rate_single,
    hallucination_report,
    unsupported_claims,
)

CONTEXTS = [
    "Mercury is the closest planet to the Sun.",
    "Water boils at 100 degrees Celsius at sea level.",
]
ANSWERS = [
    "Mercury is closest to the Sun. Mercury has 12 moons.",
    "Water boils at 100 degrees Celsius at sea level.",
]


def test_single_rate_half():
    # 2 claims, 1 unsupported ("12 moons") -> 0.5
    assert hallucination_rate_single(ANSWERS[0], CONTEXTS[0]) == pytest.approx(0.5)


def test_single_rate_zero():
    assert hallucination_rate_single(ANSWERS[1], CONTEXTS[1]) == 0.0


def test_unsupported_claims_listed():
    bad = unsupported_claims(ANSWERS[0], CONTEXTS[0])
    assert bad == ["Mercury has 12 moons."]


def test_micro_rate():
    # total claims = 3, unsupported = 1 -> 1/3
    assert hallucination_rate(ANSWERS, CONTEXTS, aggregate="micro") == pytest.approx(1 / 3)


def test_macro_rate():
    # mean of [0.5, 0.0] -> 0.25
    assert hallucination_rate(ANSWERS, CONTEXTS, aggregate="macro") == pytest.approx(0.25)


def test_report_structure():
    rep = hallucination_report(ANSWERS, CONTEXTS)
    assert rep["micro"] == pytest.approx(1 / 3)
    assert rep["macro"] == pytest.approx(0.25)
    assert len(rep["per_example"]) == 2
    assert rep["per_example"][0]["unsupported"] == ["Mercury has 12 moons."]


def test_empty_answer_zero_rate():
    assert hallucination_rate_single("", "anything") == 0.0


def test_all_hallucinated():
    rate = hallucination_rate_single("Dragons rule the galaxy.", "Cats are mammals.")
    assert rate == 1.0


def test_injected_scorer():
    # everything supported -> rate 0
    assert hallucination_rate_single(
        "x. y.", "ctx", scorer=lambda c, ctx: 1.0
    ) == 0.0
    # nothing supported -> rate 1
    assert hallucination_rate_single(
        "x. y.", "ctx", scorer=lambda c, ctx: 0.0
    ) == 1.0


def test_length_mismatch_and_bad_aggregate():
    with pytest.raises(ValueError):
        hallucination_rate(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        hallucination_rate(ANSWERS, CONTEXTS, aggregate="weird")


def test_empty_corpus_is_zero():
    assert hallucination_rate([], []) == 0.0
