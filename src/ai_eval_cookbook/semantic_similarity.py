"""Semantic similarity between texts: lexical overlap (Jaccard) and embedding
cosine, with a pluggable embedder interface and a deterministic offline default.

Production semantic similarity uses a sentence embedder (e.g. a transformer) and
cosine similarity. To stay offline and dependency-free, this module ships a
deterministic *hashing bag-of-n-grams* embedder that captures real lexical signal
without any model download. Swap in a real embedder by passing any
``Callable[[str], Sequence[float]]`` to ``embedding_cosine_similarity`` — the
contract is intentionally minimal.

Also provided: token- and character-n-gram **Jaccard**, a transparent overlap
measure that needs no embedder at all.

Offline, standard library only.
"""

from __future__ import annotations

import math
import re
from typing import Callable, List, Sequence, Set

_WORD = re.compile(r"\w+", re.UNICODE)

Embedder = Callable[[str], Sequence[float]]


def tokenize(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def _token_set(text: str) -> Set[str]:
    return set(tokenize(text))


def token_jaccard(a: str, b: str) -> float:
    """Jaccard similarity over the *sets* of word tokens: |A∩B| / |A∪B|."""
    sa, sb = _token_set(a), _token_set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _char_ngrams(text: str, n: int) -> Set[str]:
    text = text.lower()
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def char_ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    """Jaccard over character n-grams; robust to typos and morphology."""
    sa, sb = _char_ngrams(a, n), _char_ngrams(b, n)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def cosine(u: Sequence[float], v: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors; 0.0 if either is all-zero."""
    if len(u) != len(v):
        raise ValueError("vectors must have equal length")
    dot = sum(x * y for x, y in zip(u, v))
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(y * y for y in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (nu * nv)


def hashing_embedder(text: str, dim: int = 256, ngram: int = 1) -> List[float]:
    """Deterministic offline embedding: L2-normalized hashed bag-of-n-grams.

    Each token n-gram is hashed into one of ``dim`` buckets and increments that
    coordinate. This is a real (if simple) vector-space model — similar texts share
    buckets and get higher cosine — with no model or network. Deterministic across
    runs because it uses a fixed hash over the token string.
    """
    vec = [0.0] * dim
    toks = tokenize(text)
    grams = (
        toks
        if ngram == 1
        else [" ".join(toks[i : i + ngram]) for i in range(len(toks) - ngram + 1)]
    )
    for g in grams:
        # Stable, salted hash (independent of PYTHONHASHSEED).
        h = 1469598103934665603
        for ch in g:
            h ^= ord(ch)
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def embedding_cosine_similarity(
    a: str, b: str, embedder: Embedder = hashing_embedder
) -> float:
    """Cosine similarity between embeddings of two texts.

    Defaults to the offline hashing embedder; pass any real embedder with the same
    ``str -> vector`` signature to use it instead.
    """
    return cosine(list(embedder(a)), list(embedder(b)))


def semantic_similarity(
    a: str,
    b: str,
    embedder: Embedder = hashing_embedder,
    weights: Sequence[float] = (0.5, 0.5),
) -> float:
    """Blend token-Jaccard and embedding cosine into one [0, 1] score.

    ``weights`` are (jaccard_weight, cosine_weight) and are renormalized to sum 1.
    """
    w_j, w_c = weights
    total = w_j + w_c
    if total == 0:
        raise ValueError("weights must not both be zero")
    j = token_jaccard(a, b)
    c = embedding_cosine_similarity(a, b, embedder)
    return (w_j * j + w_c * c) / total


if __name__ == "__main__":
    a = "the quick brown fox"
    b = "a quick brown fox jumps"
    c = "completely unrelated sentence"
    print("token jaccard (a,b):", round(token_jaccard(a, b), 3))
    print("char jaccard  (a,b):", round(char_ngram_jaccard(a, b), 3))
    print("emb cosine    (a,b):", round(embedding_cosine_similarity(a, b), 3))
    print("emb cosine    (a,c):", round(embedding_cosine_similarity(a, c), 3))
    print("blended       (a,b):", round(semantic_similarity(a, b), 3))
