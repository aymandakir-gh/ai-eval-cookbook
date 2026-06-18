import pytest

from ai_eval_cookbook.semantic_similarity import (
    char_ngram_jaccard,
    cosine,
    embedding_cosine_similarity,
    hashing_embedder,
    semantic_similarity,
    token_jaccard,
)


def test_token_jaccard_hand_computed():
    # A={the,quick,brown,fox} B={a,quick,brown,fox,jumps}
    # inter=3 union=6 -> 0.5
    assert token_jaccard("the quick brown fox", "a quick brown fox jumps") == pytest.approx(0.5)


def test_token_jaccard_identical_and_disjoint():
    assert token_jaccard("a b c", "c b a") == pytest.approx(1.0)  # set-based
    assert token_jaccard("a b", "x y") == 0.0


def test_token_jaccard_both_empty_is_one():
    assert token_jaccard("", "") == 1.0


def test_char_ngram_jaccard_typo_robust():
    # "organize"/"organise": trigrams share {org,rga,gan,ani} of 8 distinct -> 4/8
    s = char_ngram_jaccard("organize", "organise", n=3)
    assert s == pytest.approx(0.5)
    # unrelated short words share no trigrams
    assert char_ngram_jaccard("cat", "dog", n=3) == 0.0
    # spelling variant still scores meaningfully above zero
    assert char_ngram_jaccard("color", "colour", n=3) > 0.3


def test_cosine_basic():
    assert cosine([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_zero_vector():
    assert cosine([0, 0], [1, 1]) == 0.0


def test_cosine_length_mismatch():
    with pytest.raises(ValueError):
        cosine([1, 2], [1, 2, 3])


def test_hashing_embedder_is_deterministic():
    assert hashing_embedder("hello world") == hashing_embedder("hello world")
    v = hashing_embedder("hello world", dim=64)
    assert len(v) == 64
    # L2 normalized
    assert sum(x * x for x in v) == pytest.approx(1.0)


def test_embedding_cosine_identical_and_disjoint():
    assert embedding_cosine_similarity("hello world", "hello world") == pytest.approx(1.0)
    assert embedding_cosine_similarity("apple", "zebra xylophone") == 0.0


def test_embedding_accepts_injected_embedder():
    # tiny custom embedder: vector = [len, vowel_count]
    def emb(text):
        vowels = sum(text.lower().count(v) for v in "aeiou")
        return [float(len(text)), float(vowels)]

    s = embedding_cosine_similarity("abc", "abcabc", embedder=emb)
    assert 0.0 <= s <= 1.0


def test_semantic_similarity_blend_bounds_and_weights():
    s = semantic_similarity("the quick brown fox", "a quick brown fox jumps")
    assert 0.0 <= s <= 1.0
    # weight fully on jaccard reproduces token_jaccard
    only_j = semantic_similarity(
        "the quick brown fox", "a quick brown fox jumps", weights=(1.0, 0.0)
    )
    assert only_j == pytest.approx(token_jaccard("the quick brown fox", "a quick brown fox jumps"))


def test_semantic_similarity_zero_weights_raise():
    with pytest.raises(ValueError):
        semantic_similarity("a", "b", weights=(0.0, 0.0))
