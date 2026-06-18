import pytest

from ai_eval_cookbook.rag_triad import (
    answer_faithfulness,
    context_precision,
    context_recall,
    lexical_relevance,
    rag_triad,
)

QUESTION = "Eiffel Tower Paris Gustave Eiffel designed"
CHUNKS = [
    "The Eiffel Tower is located in Paris, France.",
    "It was designed by the engineer Gustave Eiffel.",
    "Bananas are a good source of potassium.",
]
ANSWER = "The Eiffel Tower is in Paris and was designed by Gustave Eiffel."
REFERENCE = "The Eiffel Tower is in Paris. It was designed by Gustave Eiffel."


def test_context_precision_two_of_three():
    # injected scorer: first two chunks relevant, third not
    scorer = lambda chunk, q: 1.0 if "Eiffel" in chunk else 0.0
    assert context_precision(QUESTION, CHUNKS, relevance_scorer=scorer) == pytest.approx(2 / 3)


def test_context_precision_empty_chunks():
    assert context_precision(QUESTION, [], relevance_scorer=lambda c, q: 1.0) == 0.0


def test_context_precision_default_lexical():
    # default lexical relevance: chunk2 has no query words -> excluded
    p = context_precision(QUESTION, CHUNKS)
    assert p == pytest.approx(2 / 3)


def test_context_recall_full():
    # both reference claims supported by union of chunks
    assert context_recall(REFERENCE, CHUNKS) == pytest.approx(1.0)


def test_context_recall_partial():
    # only the first chunk is retrieved -> only "in Paris" claim covered
    assert context_recall(REFERENCE, [CHUNKS[0]]) == pytest.approx(0.5)


def test_context_recall_empty_reference_is_one():
    assert context_recall("", CHUNKS) == 1.0


def test_answer_faithfulness_grounded():
    assert answer_faithfulness(ANSWER, CHUNKS) == pytest.approx(1.0)


def test_answer_faithfulness_hallucination():
    bad = "The Eiffel Tower is in Paris. It is made entirely of gold."
    score = answer_faithfulness(bad, CHUNKS)
    assert score == pytest.approx(0.5)  # 1 of 2 claims grounded


def test_lexical_relevance_symmetry_helper():
    # question terms present in chunk -> high relevance
    assert lexical_relevance("Paris is the capital of France", "Paris France") == pytest.approx(1.0)
    assert lexical_relevance("Bananas are yellow", "Paris France") == 0.0


def test_rag_triad_bundle():
    triad = rag_triad(QUESTION, CHUNKS, ANSWER, REFERENCE)
    assert set(triad) == {"context_precision", "context_recall", "answer_faithfulness"}
    assert triad["context_recall"] == pytest.approx(1.0)
    assert triad["answer_faithfulness"] == pytest.approx(1.0)
    assert triad["context_precision"] == pytest.approx(2 / 3)


def test_injected_scorers_override_defaults():
    triad = rag_triad(
        QUESTION,
        CHUNKS,
        ANSWER,
        REFERENCE,
        relevance_scorer=lambda c, q: 1.0,  # everything relevant
        entailment_scorer=lambda c, ctx: 0.0,  # nothing supported
    )
    assert triad["context_precision"] == pytest.approx(1.0)
    assert triad["context_recall"] == 0.0
    assert triad["answer_faithfulness"] == 0.0
