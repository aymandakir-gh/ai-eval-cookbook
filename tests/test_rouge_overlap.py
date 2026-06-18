import pytest

from ai_eval_cookbook.rouge_overlap import (
    rouge_l,
    rouge_n,
    rouge_scores,
    simple_tokenize,
)


def test_tokenize_drops_punct():
    assert simple_tokenize("The cat, sat.") == ["the", "cat", "sat"]


def test_rouge1_hand_computed():
    # cand 3 unigrams all in ref ; ref has 6 unigrams
    r = rouge_n("the cat sat", "the cat sat on the mat", n=1)
    assert r["precision"] == pytest.approx(1.0)
    assert r["recall"] == pytest.approx(0.5)
    assert r["f1"] == pytest.approx(2 / 3)


def test_rouge2_hand_computed():
    # cand bigrams: (the,cat),(cat,sat) -> 2 ; ref has 5 bigrams ; both match
    r = rouge_n("the cat sat", "the cat sat on the mat", n=2)
    assert r["precision"] == pytest.approx(1.0)
    assert r["recall"] == pytest.approx(0.4)


def test_rouge_clipping():
    # candidate repeats a unigram more than the reference contains it
    r = rouge_n("cat cat cat", "cat dog", n=1)
    # reference has 'cat' once -> match clipped to 1
    assert r["precision"] == pytest.approx(1 / 3)
    assert r["recall"] == pytest.approx(1 / 2)


def test_rouge_l_lcs():
    # cand "the cat sat" is a subsequence of ref -> LCS=3
    r = rouge_l("the cat sat", "the cat sat on the mat")
    assert r["precision"] == pytest.approx(1.0)
    assert r["recall"] == pytest.approx(0.5)


def test_rouge_l_rewards_in_order_gapped_match():
    # LCS of [a,b,c,d] and [a,x,c,d] is [a,c,d] -> length 3
    r = rouge_l("a b c d", "a x c d")
    assert r["precision"] == pytest.approx(3 / 4)
    assert r["recall"] == pytest.approx(3 / 4)


def test_identical_is_one():
    for fn in (lambda: rouge_n("a b c", "a b c", 1), lambda: rouge_l("a b c", "a b c")):
        r = fn()
        assert r["precision"] == pytest.approx(1.0)
        assert r["recall"] == pytest.approx(1.0)
        assert r["f1"] == pytest.approx(1.0)


def test_no_overlap_is_zero():
    r = rouge_n("foo bar", "baz qux", 1)
    assert r["f1"] == 0.0


def test_empty_inputs():
    r = rouge_l("", "anything")
    assert r["precision"] == 0.0 and r["recall"] == 0.0 and r["f1"] == 0.0


def test_bundle_keys():
    s = rouge_scores("the cat sat", "the cat sat on the mat", ns=(1, 2))
    assert set(s.keys()) == {"rouge1", "rouge2", "rougeL"}
