import math

import pytest

from ai_eval_cookbook.ngram_bleu import (
    brevity_penalty,
    corpus_bleu,
    sentence_bleu,
    simple_tokenize,
)


def test_tokenize():
    assert simple_tokenize("The cat, sat.") == ["the", "cat", ",", "sat", "."]


def test_identical_is_one():
    assert sentence_bleu("hello world foo", ["hello world foo"], max_n=2) == pytest.approx(1.0)


def test_unigram_precision_hand_computed():
    # cand "the cat sat on the mat" vs "the cat is on the mat"
    # clipped unigram matches = 5, total = 6 ; lengths equal -> BP=1
    score = sentence_bleu("the cat sat on the mat", ["the cat is on the mat"], max_n=1)
    assert score == pytest.approx(5 / 6)


def test_bigram_geomean_hand_computed():
    # p1 = 5/6, p2 = 3/5 ; BLEU-2 = sqrt(p1*p2), BP=1
    score = sentence_bleu("the cat sat on the mat", ["the cat is on the mat"], max_n=2)
    assert score == pytest.approx(math.sqrt((5 / 6) * (3 / 5)))


def test_zero_when_an_order_has_no_match():
    # No 4-gram of candidate appears -> BLEU-4 is 0
    score = sentence_bleu("the cat sat on the mat", ["the cat is on the mat"], max_n=4)
    assert score == 0.0


def test_brevity_penalty():
    assert brevity_penalty(6, 6) == pytest.approx(1.0)
    assert brevity_penalty(8, 6) == pytest.approx(1.0)  # longer than ref
    assert brevity_penalty(3, 6) == pytest.approx(math.exp(1 - 6 / 3))
    assert brevity_penalty(0, 5) == 0.0


def test_brevity_penalty_reduces_short_candidate():
    # candidate is a correct but short fragment -> penalized below unigram precision
    long_ref = ["the quick brown fox jumps over the lazy dog"]
    short = sentence_bleu("the quick brown fox", long_ref, max_n=1)
    full = sentence_bleu("the quick brown fox jumps over the lazy dog", long_ref, max_n=1)
    assert full == pytest.approx(1.0)
    assert short < 1.0  # brevity penalty bites


def test_corpus_bleu_pools_statistics():
    cands = ["the cat sat on the mat", "a dog"]
    refs = [["the cat is on the mat"], ["a dog", "the dog"]]
    score = corpus_bleu(cands, refs, max_n=2)
    assert 0.0 < score <= 1.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        corpus_bleu(["a"], [["a"], ["b"]])


def test_closest_reference_length_used_for_bp():
    # two refs of length 4 and 8; candidate length 4 -> closest is 4 -> BP=1
    score = sentence_bleu("one two three four", ["one two three four", "x x x x x x x x"], max_n=1)
    assert score == pytest.approx(1.0)
