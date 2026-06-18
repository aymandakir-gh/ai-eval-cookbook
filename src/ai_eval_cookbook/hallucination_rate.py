"""Hallucination rate: the fraction of an answer's claims that are NOT supported
by a reference/context.

This is the complement of claim-level faithfulness. Where ``faithfulness_nli``
returns the supported fraction, this module focuses on the *unsupported* fraction
and aggregates it across many (answer, context) pairs so you can report a single
corpus-level hallucination rate — the headline number for RAG/summarization safety.

It reuses the same decompose-then-verify design and the same injectable entailment
scorer and claim splitter, with an offline lexical default. Provider-agnostic and
offline.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

from .faithfulness_nli import (
    ClaimSplitter,
    EntailmentScorer,
    lexical_entailment,
    sentence_claims,
)


def unsupported_claims(
    answer: str,
    context: str,
    scorer: EntailmentScorer = lexical_entailment,
    claim_splitter: ClaimSplitter = sentence_claims,
    threshold: float = 0.5,
) -> List[str]:
    """Return the claims in ``answer`` not supported by ``context``."""
    out = []
    for claim in claim_splitter(answer):
        if scorer(claim, context) < threshold:
            out.append(claim)
    return out


def hallucination_rate_single(
    answer: str,
    context: str,
    scorer: EntailmentScorer = lexical_entailment,
    claim_splitter: ClaimSplitter = sentence_claims,
    threshold: float = 0.5,
) -> float:
    """Fraction of this answer's claims that are unsupported.

    0.0 = fully grounded, 1.0 = every claim hallucinated. An answer with no
    extractable claims has a hallucination rate of 0.0.
    """
    claims = claim_splitter(answer)
    if not claims:
        return 0.0
    bad = sum(1 for c in claims if scorer(c, context) < threshold)
    return bad / len(claims)


def hallucination_rate(
    answers: Sequence[str],
    contexts: Sequence[str],
    scorer: EntailmentScorer = lexical_entailment,
    claim_splitter: ClaimSplitter = sentence_claims,
    threshold: float = 0.5,
    aggregate: str = "micro",
) -> float:
    """Corpus-level hallucination rate over many (answer, context) pairs.

    ``aggregate``:
      - ``"micro"`` (default): pool all claims, then take unsupported / total.
        Each *claim* counts equally — robust to varying answer lengths.
      - ``"macro"``: average each answer's per-answer rate. Each *answer* counts
        equally.
    """
    if len(answers) != len(contexts):
        raise ValueError("answers and contexts must have equal length")
    if not answers:
        return 0.0
    if aggregate == "micro":
        total = bad = 0
        for ans, ctx in zip(answers, contexts):
            claims = claim_splitter(ans)
            total += len(claims)
            bad += sum(1 for c in claims if scorer(c, ctx) < threshold)
        return bad / total if total else 0.0
    if aggregate == "macro":
        rates = [
            hallucination_rate_single(ans, ctx, scorer, claim_splitter, threshold)
            for ans, ctx in zip(answers, contexts)
        ]
        return sum(rates) / len(rates)
    raise ValueError("aggregate must be 'micro' or 'macro'")


def hallucination_report(
    answers: Sequence[str],
    contexts: Sequence[str],
    scorer: EntailmentScorer = lexical_entailment,
    claim_splitter: ClaimSplitter = sentence_claims,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Both aggregates plus the list of unsupported claims per example."""
    if len(answers) != len(contexts):
        raise ValueError("answers and contexts must have equal length")
    per_example = [
        {
            "rate": hallucination_rate_single(
                ans, ctx, scorer, claim_splitter, threshold
            ),
            "unsupported": unsupported_claims(
                ans, ctx, scorer, claim_splitter, threshold
            ),
        }
        for ans, ctx in zip(answers, contexts)
    ]
    return {
        "micro": hallucination_rate(
            answers, contexts, scorer, claim_splitter, threshold, "micro"
        ),
        "macro": hallucination_rate(
            answers, contexts, scorer, claim_splitter, threshold, "macro"
        ),
        "per_example": per_example,
    }


if __name__ == "__main__":
    contexts = [
        "Mercury is the closest planet to the Sun.",
        "Water boils at 100 degrees Celsius at sea level.",
    ]
    answers = [
        "Mercury is closest to the Sun. Mercury has 12 moons.",
        "Water boils at 100 degrees Celsius at sea level.",
    ]
    report = hallucination_report(answers, contexts)
    print("micro rate:", round(report["micro"], 3))
    print("macro rate:", round(report["macro"], 3))
    for i, ex in enumerate(report["per_example"]):
        print("example", i, "rate", round(ex["rate"], 3), "unsupported:", ex["unsupported"])
