"""A LEXICAL toxicity screen — a fast, offline, transparent first filter only.

WARNING / SCOPE: This is intentionally a simple word-list screen. It flags text
containing terms from a configurable lexicon. It is NOT a real toxicity classifier:
it cannot understand context, sarcasm, reclaimed language, implicit toxicity,
negation ("that is not stupid"), or harm expressed without flagged words. It will
both miss toxic text (false negatives) and flag benign text such as quotation,
counter-speech, or clinical discussion (false positives).

Use it as a cheap pre-filter or a deterministic stand-in in tests/CI. For anything
user-facing or high-stakes, use a trained classifier (e.g. Perspective API,
Detoxify, or an LLM judge) — inject one via the ``scorer`` interface. The default
lexicon here is deliberately tiny and mild (placeholder insults) so the package
ships no graphic content; supply your own vetted lexicon for real screening.

Pure standard library, offline.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Set

_WORD = re.compile(r"\w+", re.UNICODE)

# Deliberately small and mild placeholder lexicon. Replace with a vetted list.
DEFAULT_LEXICON: Set[str] = {
    "idiot",
    "stupid",
    "dumb",
    "moron",
    "loser",
    "jerk",
    "scum",
    "trash",
    "pathetic",
    "worthless",
}

# A toxicity scorer maps text -> score in [0, 1].
ToxicityScorer = Callable[[str], float]


def tokenize(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def flagged_terms(text: str, lexicon: Optional[Set[str]] = None) -> List[str]:
    """Return the lexicon terms found in the text (with duplicates, in order)."""
    lex = lexicon if lexicon is not None else DEFAULT_LEXICON
    return [t for t in tokenize(text) if t in lex]


def lexical_toxicity_score(
    text: str, lexicon: Optional[Set[str]] = None
) -> float:
    """A crude [0, 1] toxicity score: fraction of tokens that are flagged terms.

    0.0 = no flagged terms. The score saturates quickly and is NOT calibrated —
    treat it as a rough signal, not a probability.
    """
    toks = tokenize(text)
    if not toks:
        return 0.0
    flagged = sum(1 for t in toks if t in (lexicon or DEFAULT_LEXICON))
    return flagged / len(toks)


def is_toxic(
    text: str,
    threshold: float = 0.0,
    scorer: ToxicityScorer = lexical_toxicity_score,
) -> bool:
    """True if the toxicity score is strictly greater than ``threshold``.

    With the default threshold 0.0, any flagged term makes the text toxic.
    """
    return scorer(text) > threshold


def toxicity_rate(
    texts: Sequence[str],
    threshold: float = 0.0,
    scorer: ToxicityScorer = lexical_toxicity_score,
) -> float:
    """Fraction of texts flagged as toxic. Empty input -> 0.0."""
    if not texts:
        return 0.0
    return sum(1 for t in texts if is_toxic(t, threshold, scorer)) / len(texts)


def toxicity_report(
    texts: Sequence[str],
    threshold: float = 0.0,
    lexicon: Optional[Set[str]] = None,
    scorer: Optional[ToxicityScorer] = None,
) -> Dict[str, object]:
    """Toxicity rate plus per-text score and flagged terms.

    If ``scorer`` is provided it is used for scoring; otherwise the lexical scorer
    with the given ``lexicon`` is used and flagged terms are reported.
    """
    use_scorer = scorer or (lambda t: lexical_toxicity_score(t, lexicon))
    per_text = []
    flagged_count = 0
    for t in texts:
        score = use_scorer(t)
        toxic = score > threshold
        if toxic:
            flagged_count += 1
        per_text.append(
            {
                "text": t,
                "score": score,
                "toxic": toxic,
                "terms": flagged_terms(t, lexicon) if scorer is None else [],
            }
        )
    n = len(texts)
    return {
        "n": n,
        "toxicity_rate": (flagged_count / n) if n else 0.0,
        "per_text": per_text,
        "caveat": (
            "Lexical screen only: misses implicit toxicity and over-flags "
            "quotation/counter-speech. Not for high-stakes use."
        ),
    }


if __name__ == "__main__":
    texts = [
        "You are such an idiot and a loser.",  # flagged
        "Thank you, that was very helpful!",  # clean
        "He said I was stupid, which hurt.",  # flagged (false positive: reporting)
    ]
    report = toxicity_report(texts)
    print("toxicity rate:", report["toxicity_rate"])
    for row in report["per_text"]:
        print(round(row["score"], 3), row["toxic"], row["terms"], "::", row["text"])
    print("CAVEAT:", report["caveat"])
