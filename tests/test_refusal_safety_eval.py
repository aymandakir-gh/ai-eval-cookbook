import pytest

from ai_eval_cookbook.refusal_safety_eval import (
    attack_success_rate,
    is_refusal,
    keyword_refusal_detector,
    refusal_rate,
    safety_report,
)


def test_detector_explicit_refusals():
    assert keyword_refusal_detector("I'm sorry, but I can't help with that.") is True
    assert keyword_refusal_detector("I cannot provide that information.") is True
    assert keyword_refusal_detector("As an AI, I must decline.") is True


def test_detector_compliance():
    assert keyword_refusal_detector("Sure! Here is how you do it.") is False
    assert keyword_refusal_detector("The capital of France is Paris.") is False


def test_detector_custom_patterns():
    assert keyword_refusal_detector("NOPE not doing that", patterns=["nope"]) is True
    assert keyword_refusal_detector("I cannot", patterns=["nope"]) is False


def test_refusal_rate():
    responses = [
        "I cannot help with that.",  # refusal
        "Sure, step 1...",  # comply
        "I'm sorry, I won't do that.",  # refusal
    ]
    assert refusal_rate(responses) == pytest.approx(2 / 3)


def test_attack_success_rate_is_complement():
    responses = ["I cannot.", "Sure.", "Sure."]
    assert attack_success_rate(responses) == pytest.approx(2 / 3)
    assert refusal_rate(responses) + attack_success_rate(responses) == pytest.approx(1.0)


def test_safety_report_harmful():
    responses = ["I cannot do that.", "Sure! step 1...", "I'm sorry, no."]
    rep = safety_report(responses, harmful=True)
    assert rep["refusal_rate"] == pytest.approx(2 / 3)
    assert rep["attack_success_rate"] == pytest.approx(1 / 3)
    assert rep["refusals"] == [True, False, True]
    assert "over_refusal_rate" not in rep


def test_safety_report_benign_over_refusal():
    responses = ["Here is a cake recipe.", "I'm sorry, I can't help with baking."]
    rep = safety_report(responses, harmful=False)
    assert rep["over_refusal_rate"] == pytest.approx(0.5)
    assert "attack_success_rate" not in rep


def test_injected_detector():
    # detector that flags anything containing 'decline'
    detector = lambda r: "decline" in r.lower()
    responses = ["I decline.", "Sure thing."]
    assert refusal_rate(responses, detector=detector) == pytest.approx(0.5)
    assert is_refusal("I decline.", detector=detector) is True


def test_empty_inputs():
    assert refusal_rate([]) == 0.0
    assert attack_success_rate([]) == 0.0
    rep = safety_report([], harmful=True)
    assert rep["n"] == 0
    assert rep["refusal_rate"] == 0.0
