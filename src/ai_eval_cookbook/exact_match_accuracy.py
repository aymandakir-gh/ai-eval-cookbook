"""Exact-match accuracy with text normalization.

Exact match (EM) is the strictest correctness metric: a prediction counts as
correct only if it equals the reference. Raw string equality is almost never what
you want, because "Paris", "paris", and "Paris." should all be considered the same
answer for a QA task. This module therefore normalizes both sides before comparing,
following the normalization popularized by the SQuAD evaluation script: lowercase,
strip articles ("a", "an", "the"), remove punctuation, and collapse whitespace.

Provider-agnostic and offline: you pass in predictions and references (strings or,
for multi-reference tasks, lists of acceptable references). No model is called.
"""

from __future__ import annotations

import re
import string
import unicodedata
from typing import Callable, List, Optional, Sequence, Union

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_PUNCT_TABLE = {ord(ch): None for ch in string.punctuation}

ReferenceLike = Union[str, Sequence[str]]


def normalize_text(text: str) -> str:
    """Normalize text for exact-match comparison.

    Steps: Unicode NFKC normalization, lowercase, remove articles, strip
    punctuation, and collapse runs of whitespace. Mirrors the SQuAD-style
    ``normalize_answer`` routine.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def exact_match(
    prediction: str,
    reference: ReferenceLike,
    normalizer: Optional[Callable[[str], str]] = None,
) -> bool:
    """Return ``True`` if ``prediction`` matches ``reference`` after normalization.

    ``reference`` may be a single string or a sequence of acceptable references;
    in the latter case the prediction is correct if it matches *any* of them.
    Pass ``normalizer`` to override the default normalization (e.g. ``str`` for
    raw, case-sensitive comparison).
    """
    norm = normalizer or normalize_text
    npred = norm(prediction)
    if isinstance(reference, str):
        return npred == norm(reference)
    return any(npred == norm(ref) for ref in reference)


def accuracy(
    predictions: Sequence[str],
    references: Sequence[ReferenceLike],
    normalizer: Optional[Callable[[str], str]] = None,
) -> float:
    """Fraction of predictions that exactly match their reference.

    ``predictions`` and ``references`` must be the same length. Returns 0.0 for an
    empty input rather than raising.
    """
    if len(predictions) != len(references):
        raise ValueError(
            "predictions and references must have equal length: "
            "%d != %d" % (len(predictions), len(references))
        )
    if not predictions:
        return 0.0
    hits = sum(
        1
        for pred, ref in zip(predictions, references)
        if exact_match(pred, ref, normalizer)
    )
    return hits / len(predictions)


def correctness_mask(
    predictions: Sequence[str],
    references: Sequence[ReferenceLike],
    normalizer: Optional[Callable[[str], str]] = None,
) -> List[bool]:
    """Per-example correctness flags, useful for slicing/error analysis."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    return [
        exact_match(pred, ref, normalizer)
        for pred, ref in zip(predictions, references)
    ]


if __name__ == "__main__":
    preds = ["The capital is Paris.", "  yes ", "blue", "42"]
    refs = ["paris", "Yes", ["red", "green"], "forty two"]
    print("normalized 'The capital is Paris.' ->", repr(normalize_text(preds[0])))
    print("per-example:", correctness_mask(preds, refs))
    print("accuracy:", accuracy(preds, refs))
