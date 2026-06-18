"""ROUGE-N and ROUGE-L (recall-oriented summarization overlap).

ROUGE (Lin, 2004) measures overlap between a candidate and a reference. Unlike BLEU
(precision-oriented for translation), ROUGE was designed for summarization and is
typically reported recall-first, though modern usage reports precision, recall, and
F1 together.

- **ROUGE-N**: overlap of n-grams. Recall = matched / reference n-grams,
  precision = matched / candidate n-grams.
- **ROUGE-L**: based on the Longest Common Subsequence (LCS), which rewards
  in-order (but not necessarily contiguous) word matches and needs no fixed n.

Offline, standard library only. Simple lowercase/word tokenization by default;
pass your own ``tokenizer`` to match a specific protocol.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Dict, List, Sequence

_WORD = re.compile(r"\w+", re.UNICODE)


def simple_tokenize(text: str) -> List[str]:
    """Lowercase and extract word tokens (drops punctuation)."""
    return _WORD.findall(text.lower())


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(
        tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
    )


def _prf(match: int, cand_total: int, ref_total: int) -> Dict[str, float]:
    precision = match / cand_total if cand_total else 0.0
    recall = match / ref_total if ref_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_n(
    candidate: str,
    reference: str,
    n: int = 1,
    tokenizer: Callable[[str], List[str]] = simple_tokenize,
) -> Dict[str, float]:
    """ROUGE-N precision/recall/F1 between a candidate and a reference.

    Overlap of n-grams is clipped by reference counts (an n-gram can match at most
    as many times as it appears in the reference).
    """
    cand = _ngram_counts(tokenizer(candidate), n)
    ref = _ngram_counts(tokenizer(reference), n)
    match = sum(min(cnt, ref[ng]) for ng, cnt in cand.items())
    return _prf(match, sum(cand.values()), sum(ref.values()))


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence (space-optimized DP)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
        prev = curr
    return prev[len(b)]


def rouge_l(
    candidate: str,
    reference: str,
    tokenizer: Callable[[str], List[str]] = simple_tokenize,
) -> Dict[str, float]:
    """ROUGE-L precision/recall/F1 based on the longest common subsequence."""
    cand = tokenizer(candidate)
    ref = tokenizer(reference)
    lcs = _lcs_length(cand, ref)
    return _prf(lcs, len(cand), len(ref))


def rouge_scores(
    candidate: str,
    reference: str,
    ns: Sequence[int] = (1, 2),
    tokenizer: Callable[[str], List[str]] = simple_tokenize,
) -> Dict[str, Dict[str, float]]:
    """Bundle ROUGE-N for each n in ``ns`` plus ROUGE-L."""
    out: Dict[str, Dict[str, float]] = {}
    for n in ns:
        out["rouge%d" % n] = rouge_n(candidate, reference, n, tokenizer)
    out["rougeL"] = rouge_l(candidate, reference, tokenizer)
    return out


if __name__ == "__main__":
    cand = "the cat was sitting on the mat"
    ref = "the cat sat on the mat"
    scores = rouge_scores(cand, ref, ns=(1, 2))
    for name, vals in scores.items():
        print(name, {k: round(v, 3) for k, v in vals.items()})
