"""The RAG triad: context precision, context recall, and answer faithfulness.

A retrieval-augmented generation pipeline has three failure surfaces, and the triad
puts one number on each:

- **Context precision** — of the retrieved chunks, how many are actually relevant?
  (Did the retriever bring back junk?) Here: fraction of retrieved chunks judged
  relevant by an injectable relevance scorer.
- **Context recall** — of the information needed to answer (the ground-truth
  reference), how much is covered by the retrieved chunks? (Did the retriever miss
  needed evidence?) Here: claim-level coverage of the reference by the union of
  retrieved chunks.
- **Answer faithfulness** — is the generated answer grounded in the retrieved
  context? (Did the generator make things up?) Reuses ``faithfulness_nli``.

Every judge (relevance scorer, entailment scorer, claim splitter) is injectable;
offline lexical defaults keep the recipe runnable. Provider-agnostic and offline.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

from .faithfulness_nli import (
    ClaimSplitter,
    EntailmentScorer,
    faithfulness,
    lexical_entailment,
    sentence_claims,
)

# A relevance scorer maps (chunk, question) -> relevance score in [0, 1].
RelevanceScorer = Callable[[str, str], float]


def lexical_relevance(chunk: str, question: str) -> float:
    """Offline relevance: fraction of the question's content words present in the
    chunk (reuses the faithfulness lexical heuristic with arguments swapped)."""
    return lexical_entailment(question, chunk)


def context_precision(
    question: str,
    retrieved_chunks: Sequence[str],
    relevance_scorer: RelevanceScorer = lexical_relevance,
    threshold: float = 0.5,
) -> float:
    """Fraction of retrieved chunks judged relevant to the question.

    Returns 0.0 when nothing was retrieved.
    """
    if not retrieved_chunks:
        return 0.0
    relevant = sum(
        1 for c in retrieved_chunks if relevance_scorer(c, question) >= threshold
    )
    return relevant / len(retrieved_chunks)


def context_recall(
    reference: str,
    retrieved_chunks: Sequence[str],
    scorer: EntailmentScorer = lexical_entailment,
    claim_splitter: ClaimSplitter = sentence_claims,
    threshold: float = 0.5,
) -> float:
    """Fraction of the reference's claims supported by the retrieved context.

    The reference (ground-truth answer or required evidence) is decomposed into
    claims; each is checked against the *concatenation* of retrieved chunks.
    Returns 1.0 for a reference with no extractable claims.
    """
    claims = claim_splitter(reference)
    if not claims:
        return 1.0
    context = "\n".join(retrieved_chunks)
    supported = sum(1 for c in claims if scorer(c, context) >= threshold)
    return supported / len(claims)


def answer_faithfulness(
    answer: str,
    retrieved_chunks: Sequence[str],
    scorer: EntailmentScorer = lexical_entailment,
    claim_splitter: ClaimSplitter = sentence_claims,
    threshold: float = 0.5,
) -> float:
    """Faithfulness of the answer to the retrieved context (see faithfulness_nli)."""
    context = "\n".join(retrieved_chunks)
    return faithfulness(answer, context, scorer, claim_splitter, threshold)


def rag_triad(
    question: str,
    retrieved_chunks: Sequence[str],
    answer: str,
    reference: str,
    relevance_scorer: RelevanceScorer = lexical_relevance,
    entailment_scorer: EntailmentScorer = lexical_entailment,
    claim_splitter: ClaimSplitter = sentence_claims,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute all three triad metrics for one RAG example."""
    return {
        "context_precision": context_precision(
            question, retrieved_chunks, relevance_scorer, threshold
        ),
        "context_recall": context_recall(
            reference, retrieved_chunks, entailment_scorer, claim_splitter, threshold
        ),
        "answer_faithfulness": answer_faithfulness(
            answer, retrieved_chunks, entailment_scorer, claim_splitter, threshold
        ),
    }


if __name__ == "__main__":
    # Note: the offline lexical default is a coarse stand-in (it matches words, not
    # meaning). For a clean illustration we use a relevance scorer with a keyword
    # query whose terms appear verbatim in the relevant chunks.
    question = "Eiffel Tower Paris Gustave Eiffel designed"
    chunks = [
        "The Eiffel Tower is located in Paris, France.",
        "It was designed by the engineer Gustave Eiffel.",
        "Bananas are a good source of potassium.",  # irrelevant chunk
    ]
    answer = "The Eiffel Tower is in Paris and was designed by Gustave Eiffel."
    reference = "The Eiffel Tower is in Paris. It was designed by Gustave Eiffel."

    triad = rag_triad(question, chunks, answer, reference)
    for name, val in triad.items():
        print(name, round(val, 3))
    # 2 of 3 chunks relevant -> precision ~0.67; reference fully covered -> recall 1.0;
    # answer grounded in chunks -> faithfulness 1.0.
