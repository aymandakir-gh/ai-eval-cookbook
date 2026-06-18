"""A compact, correct BLEU implementation (n-gram precision + brevity penalty).

BLEU (Papineni et al., 2002) scores a candidate against one or more references by
combining *modified* n-gram precision (clipped by the maximum count of each n-gram
in any reference) for n = 1..N as a geometric mean, then multiplying by a brevity
penalty that punishes candidates shorter than the reference.

This is corpus-level BLEU (the statistically meaningful form), with a sentence-level
convenience wrapper. Tokenization is deliberately simple (lowercase + whitespace/
word split) so behavior is transparent; pass your own ``tokenizer`` to match a
specific protocol such as the original ``mteval`` or ``sacrebleu``.

Offline, standard library only.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, List, Sequence, Tuple

_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def simple_tokenize(text: str) -> List[str]:
    """Lowercase and split into word/punctuation tokens."""
    return _WORD.findall(text.lower())


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(
        tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
    )


def _closest_ref_len(cand_len: int, ref_lens: Sequence[int]) -> int:
    """Reference length closest to the candidate; ties prefer the shorter one."""
    return min(ref_lens, key=lambda rl: (abs(rl - cand_len), rl))


def _modified_precision(
    cand_tokens: Sequence[str],
    refs_tokens: Sequence[Sequence[str]],
    n: int,
) -> Tuple[int, int]:
    """Return (clipped_match_count, total_candidate_ngrams) for order n."""
    cand_ngrams = _ngrams(cand_tokens, n)
    if not cand_ngrams:
        return 0, 0
    max_ref = Counter()
    for ref in refs_tokens:
        ref_ngrams = _ngrams(ref, n)
        for ng, cnt in ref_ngrams.items():
            if cnt > max_ref[ng]:
                max_ref[ng] = cnt
    clipped = sum(min(cnt, max_ref[ng]) for ng, cnt in cand_ngrams.items())
    total = sum(cand_ngrams.values())
    return clipped, total


def corpus_bleu(
    candidates: Sequence[str],
    references: Sequence[Sequence[str]],
    max_n: int = 4,
    tokenizer: Callable[[str], List[str]] = simple_tokenize,
) -> float:
    """Corpus-level BLEU over a list of candidates and per-candidate reference lists.

    ``references[i]`` is the list of acceptable references for ``candidates[i]``.
    Returns a score in [0, 1]. Uses uniform weights 1/max_n and the standard
    brevity penalty. If any n-gram order has zero clipped matches the geometric
    mean is 0 (no smoothing), matching the original definition.
    """
    if len(candidates) != len(references):
        raise ValueError("candidates and references must have equal length")
    clipped = [0] * (max_n + 1)
    totals = [0] * (max_n + 1)
    cand_total_len = 0
    ref_total_len = 0

    for cand, refs in zip(candidates, references):
        cand_tok = tokenizer(cand)
        refs_tok = [tokenizer(r) for r in refs]
        cand_total_len += len(cand_tok)
        ref_lens = [len(r) for r in refs_tok] or [0]
        ref_total_len += _closest_ref_len(len(cand_tok), ref_lens)
        for n in range(1, max_n + 1):
            c, t = _modified_precision(cand_tok, refs_tok, n)
            clipped[n] += c
            totals[n] += t

    precisions = []
    for n in range(1, max_n + 1):
        if totals[n] == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped[n] / totals[n])

    if min(precisions) == 0.0:
        geo_mean = 0.0
    else:
        log_sum = sum((1.0 / max_n) * math.log(p) for p in precisions)
        geo_mean = math.exp(log_sum)

    bp = brevity_penalty(cand_total_len, ref_total_len)
    return bp * geo_mean


def brevity_penalty(cand_len: int, ref_len: int) -> float:
    """BP = 1 if candidate is at least as long as the reference, else exp(1 - r/c)."""
    if cand_len == 0:
        return 0.0
    if cand_len > ref_len:
        return 1.0
    return math.exp(1.0 - ref_len / cand_len)


def sentence_bleu(
    candidate: str,
    references: Sequence[str],
    max_n: int = 4,
    tokenizer: Callable[[str], List[str]] = simple_tokenize,
) -> float:
    """Convenience: BLEU for a single candidate against its references.

    Note: sentence-level BLEU is noisy (a single missing high-order n-gram can
    zero the score). Prefer ``corpus_bleu`` for system comparison.
    """
    return corpus_bleu([candidate], [list(references)], max_n=max_n, tokenizer=tokenizer)


if __name__ == "__main__":
    cand = "the cat sat on the mat"
    refs = ["the cat is on the mat", "there is a cat on the mat"]
    for n in (1, 2, 3, 4):
        print("sentence BLEU-%d:" % n, round(sentence_bleu(cand, refs, max_n=n), 4))
    print(
        "corpus BLEU-4:",
        round(corpus_bleu([cand, "a dog"], [refs, ["a dog", "the dog"]]), 4),
    )
