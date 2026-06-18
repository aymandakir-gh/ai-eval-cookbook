"""Faithfulness: are an answer's claims supported by the provided context?

Following the RAGAS / FActScore pattern, faithfulness is computed in two steps:

1. **Decompose** the answer into atomic claims (sentence-level by default; pass your
   own ``claim_splitter`` to plug in an LLM claim extractor).
2. **Verify** each claim against the context with an **entailment scorer** — an
   injected ``Callable[[claim, context], float]`` returning a support score in
   [0, 1]. The faithfulness score is the fraction of claims whose support meets a
   threshold.

The entailment scorer is the conceptual "NLI model". To stay offline, the default
is a transparent **lexical entailment** heuristic (token-overlap of the claim with
the context). Swap in a real NLI model or LLM judge by passing ``scorer=...`` —
the signature is intentionally minimal.

Offline, standard library only.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Sequence

_WORD = re.compile(r"\w+", re.UNICODE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of",
    "to", "in", "on", "at", "and", "or", "but", "for", "with", "as", "by",
    "that", "this", "it", "its", "from", "has", "have", "had", "not", "no",
}

# An entailment scorer maps (claim, context) -> support score in [0, 1].
EntailmentScorer = Callable[[str, str], float]
ClaimSplitter = Callable[[str], List[str]]


def _content_tokens(text: str) -> List[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOPWORDS]


def sentence_claims(answer: str) -> List[str]:
    """Default claim splitter: non-empty sentences."""
    parts = _SENT_SPLIT.split(answer.strip())
    return [p.strip() for p in parts if p.strip()]


def lexical_entailment(claim: str, context: str) -> float:
    """Offline support score: fraction of the claim's content tokens found in context.

    A simple, deterministic stand-in for an NLI model. Returns 1.0 when every
    content word of the claim appears in the context, 0.0 when none do. Returns 1.0
    for a claim with no content tokens (nothing to contradict).
    """
    claim_tokens = _content_tokens(claim)
    if not claim_tokens:
        return 1.0
    context_tokens = set(_content_tokens(context))
    hits = sum(1 for t in claim_tokens if t in context_tokens)
    return hits / len(claim_tokens)


def claim_verdicts(
    answer: str,
    context: str,
    scorer: EntailmentScorer = lexical_entailment,
    claim_splitter: ClaimSplitter = sentence_claims,
    threshold: float = 0.5,
) -> List[Dict[str, object]]:
    """Per-claim support scores and supported/unsupported verdicts."""
    claims = claim_splitter(answer)
    verdicts = []
    for claim in claims:
        score = scorer(claim, context)
        verdicts.append(
            {"claim": claim, "support": score, "supported": score >= threshold}
        )
    return verdicts


def faithfulness(
    answer: str,
    context: str,
    scorer: EntailmentScorer = lexical_entailment,
    claim_splitter: ClaimSplitter = sentence_claims,
    threshold: float = 0.5,
) -> float:
    """Fraction of the answer's claims that are supported by the context.

    1.0 = every claim entailed; 0.0 = no claim entailed. An answer with no
    extractable claims scores 1.0 (vacuously faithful).
    """
    verdicts = claim_verdicts(answer, context, scorer, claim_splitter, threshold)
    if not verdicts:
        return 1.0
    supported = sum(1 for v in verdicts if v["supported"])
    return supported / len(verdicts)


if __name__ == "__main__":
    context = (
        "The Eiffel Tower is located in Paris. It was completed in 1889 "
        "and was designed by Gustave Eiffel."
    )
    answer = (
        "The Eiffel Tower is in Paris. It was completed in 1889. "
        "It is the tallest building in the world."
    )
    for v in claim_verdicts(answer, context):
        print(round(v["support"], 2), v["supported"], "::", v["claim"])
    print("faithfulness:", round(faithfulness(answer, context), 3))
