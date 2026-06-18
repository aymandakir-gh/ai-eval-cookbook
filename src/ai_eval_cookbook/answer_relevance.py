"""Answer relevance: does the answer actually address the question?

Two complementary approaches are provided:

1. **Direct similarity** (`direct_relevance`): similarity between the question and
   the answer using an injectable embedder (offline hashing embedder by default).
   Simple and fast.

2. **Reverse-question relevance** (`answer_relevance`, the RAGAS pattern):
   reverse-engineer ``n`` candidate questions *from the answer* using an injected
   question generator, embed them, and take the mean cosine similarity to the
   original question. Intuition: if the answer addresses the question, the question
   should be reconstructable from the answer. This penalizes evasive, off-topic, or
   padded answers even when they share words with the question.

Both the question generator and the embedder are injection points so you can plug
in an LLM and a real embedder. The offline defaults keep everything runnable.

Offline, standard library only.
"""

from __future__ import annotations

from typing import Callable, List, Sequence

from .semantic_similarity import (
    Embedder,
    cosine,
    embedding_cosine_similarity,
    hashing_embedder,
)

# A question generator maps an answer -> list of candidate questions.
QuestionGenerator = Callable[[str], List[str]]


def direct_relevance(
    question: str, answer: str, embedder: Embedder = hashing_embedder
) -> float:
    """Cosine similarity between question and answer embeddings, in [0, 1].

    Negative cosines (possible for some embedders) are clamped to 0.0 so the score
    stays in [0, 1].
    """
    sim = embedding_cosine_similarity(question, answer, embedder)
    return max(0.0, sim)


def echo_question_generator(answer: str) -> List[str]:
    """Offline default question generator.

    Without an LLM we cannot truly reverse-engineer questions, so we treat the
    answer itself as the single "reconstructed question". This reduces
    reverse-question relevance to direct question/answer similarity — a transparent
    fallback. Inject a real generator for the full RAGAS behavior.
    """
    return [answer]


def answer_relevance(
    question: str,
    answer: str,
    question_generator: QuestionGenerator = echo_question_generator,
    embedder: Embedder = hashing_embedder,
) -> float:
    """RAGAS-style answer relevance: mean cosine similarity between the original
    question and questions reconstructed from the answer.

    Returns a value in [0, 1] (negative cosines clamped to 0). With the default
    offline generator this equals ``direct_relevance``.
    """
    generated = question_generator(answer)
    if not generated:
        return 0.0
    q_vec = list(embedder(question))
    sims = []
    for gq in generated:
        sims.append(max(0.0, cosine(q_vec, list(embedder(gq)))))
    return sum(sims) / len(sims)


def batch_answer_relevance(
    questions: Sequence[str],
    answers: Sequence[str],
    question_generator: QuestionGenerator = echo_question_generator,
    embedder: Embedder = hashing_embedder,
) -> float:
    """Mean answer relevance across a list of (question, answer) pairs."""
    if len(questions) != len(answers):
        raise ValueError("questions and answers must have equal length")
    if not questions:
        return 0.0
    scores = [
        answer_relevance(q, a, question_generator, embedder)
        for q, a in zip(questions, answers)
    ]
    return sum(scores) / len(scores)


if __name__ == "__main__":
    q = "What is the capital of France?"
    good = "The capital of France is Paris."
    bad = "France is a country with a rich culinary tradition."

    print("direct (good):", round(direct_relevance(q, good), 3))
    print("direct (bad): ", round(direct_relevance(q, bad), 3))

    # Inject a toy generator that turns a declarative answer into a question.
    def toy_gen(answer: str) -> List[str]:
        return ["What " + answer.lower().rstrip(".") + "?"]

    print("reverse-question (good):", round(answer_relevance(q, good, toy_gen), 3))
    print("reverse-question (bad): ", round(answer_relevance(q, bad, toy_gen), 3))
