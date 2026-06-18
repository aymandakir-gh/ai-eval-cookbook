import pytest

from ai_eval_cookbook.faithfulness_nli import (
    claim_verdicts,
    faithfulness,
    lexical_entailment,
    sentence_claims,
)

CONTEXT = (
    "The Eiffel Tower is located in Paris. It was completed in 1889 "
    "and was designed by Gustave Eiffel."
)


def test_sentence_claims_splits():
    claims = sentence_claims("A is true. B is false! Is C? ")
    assert claims == ["A is true.", "B is false!", "Is C?"]


def test_lexical_entailment_full_and_none():
    assert lexical_entailment("Eiffel Tower Paris", CONTEXT) == pytest.approx(1.0)
    assert lexical_entailment("banana spaceship", CONTEXT) == 0.0


def test_lexical_entailment_partial():
    # content tokens: paris, london -> only paris in context -> 1/2
    assert lexical_entailment("Paris London", CONTEXT) == pytest.approx(0.5)


def test_lexical_entailment_no_content_tokens_is_one():
    # only stopwords -> vacuously supported
    assert lexical_entailment("it is the", CONTEXT) == 1.0


def test_faithfulness_two_of_three():
    answer = (
        "The Eiffel Tower is in Paris. It was completed in 1889. "
        "It is the tallest building in the world."
    )
    assert faithfulness(answer, CONTEXT) == pytest.approx(2 / 3)


def test_claim_verdicts_flags():
    answer = "The Eiffel Tower is in Paris. It is the tallest building in the world."
    v = claim_verdicts(answer, CONTEXT)
    assert v[0]["supported"] is True
    assert v[1]["supported"] is False
    assert 0.0 <= v[0]["support"] <= 1.0


def test_fully_faithful_and_fully_unfaithful():
    assert faithfulness("It was designed by Gustave Eiffel.", CONTEXT) == pytest.approx(1.0)
    assert faithfulness("Bananas orbit distant moons.", CONTEXT) == 0.0


def test_empty_answer_is_vacuously_faithful():
    assert faithfulness("", CONTEXT) == 1.0


def test_injected_scorer_is_used():
    # always-supported scorer -> faithfulness 1.0 regardless of content
    score = faithfulness(
        "Totally unrelated claim.", CONTEXT, scorer=lambda c, ctx: 1.0
    )
    assert score == 1.0
    # always-zero scorer -> 0.0
    score0 = faithfulness(
        "The Eiffel Tower is in Paris.", CONTEXT, scorer=lambda c, ctx: 0.0
    )
    assert score0 == 0.0


def test_injected_claim_splitter():
    # split on ';' instead of sentences
    answer = "Paris; banana spaceship"
    score = faithfulness(
        answer, CONTEXT, claim_splitter=lambda a: [s.strip() for s in a.split(";")]
    )
    assert score == pytest.approx(0.5)


def test_threshold_changes_verdicts():
    answer = "Paris London"  # support 0.5
    assert faithfulness(answer, CONTEXT, threshold=0.5) == 1.0  # 0.5 >= 0.5
    assert faithfulness(answer, CONTEXT, threshold=0.6) == 0.0  # 0.5 < 0.6
