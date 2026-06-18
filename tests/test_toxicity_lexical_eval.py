import pytest

from ai_eval_cookbook.toxicity_lexical_eval import (
    flagged_terms,
    is_toxic,
    lexical_toxicity_score,
    toxicity_rate,
    toxicity_report,
)


def test_flagged_terms():
    assert flagged_terms("you idiot, stupid move") == ["idiot", "stupid"]
    assert flagged_terms("thank you very much") == []


def test_flagged_terms_duplicates_and_order():
    assert flagged_terms("idiot idiot stupid") == ["idiot", "idiot", "stupid"]


def test_lexical_score():
    # 2 flagged of 6 tokens
    assert lexical_toxicity_score("you are an idiot and loser") == pytest.approx(2 / 6)
    assert lexical_toxicity_score("thank you very much") == 0.0


def test_lexical_score_empty():
    assert lexical_toxicity_score("") == 0.0


def test_is_toxic_default_threshold():
    assert is_toxic("you idiot") is True
    assert is_toxic("hello friend") is False


def test_is_toxic_threshold():
    # one flagged of 4 tokens -> score 0.25
    text = "do not be dumb"
    assert lexical_toxicity_score(text) == pytest.approx(0.25)
    assert is_toxic(text, threshold=0.2) is True
    assert is_toxic(text, threshold=0.3) is False


def test_toxicity_rate():
    texts = ["you idiot", "hello there", "dumb move"]
    assert toxicity_rate(texts) == pytest.approx(2 / 3)


def test_custom_lexicon_via_scorer():
    scorer = lambda t: lexical_toxicity_score(t, {"foo"})
    assert is_toxic("foo bar", scorer=scorer) is True
    assert is_toxic("idiot bar", scorer=scorer) is False  # not in custom lexicon


def test_injected_scorer():
    # always-toxic scorer
    assert toxicity_rate(["anything", "clean text"], scorer=lambda t: 1.0) == 1.0
    # always-clean scorer
    assert toxicity_rate(["you idiot"], scorer=lambda t: 0.0) == 0.0


def test_report_structure_and_caveat():
    rep = toxicity_report(["you idiot", "hello"])
    assert rep["n"] == 2
    assert rep["toxicity_rate"] == pytest.approx(0.5)
    assert rep["per_text"][0]["toxic"] is True
    assert "idiot" in rep["per_text"][0]["terms"]
    assert rep["per_text"][1]["toxic"] is False
    assert "Lexical screen only" in rep["caveat"]


def test_report_with_injected_scorer_omits_terms():
    rep = toxicity_report(["you idiot"], scorer=lambda t: 0.9)
    assert rep["per_text"][0]["toxic"] is True
    assert rep["per_text"][0]["terms"] == []  # terms only from lexical path


def test_empty_inputs():
    assert toxicity_rate([]) == 0.0
    assert toxicity_report([])["toxicity_rate"] == 0.0
