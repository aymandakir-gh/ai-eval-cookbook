import math

import pytest

from ai_eval_cookbook.pairwise_preference_elo import (
    bradley_terry,
    elo_ratings,
    expected_score,
    ranking,
    win_rate,
)

MATCHES = [
    ("A", "B", 1.0),
    ("A", "B", 1.0),
    ("A", "C", 1.0),
    ("A", "C", 1.0),
    ("B", "C", 1.0),
    ("B", "C", 1.0),
]


def test_expected_score():
    assert expected_score(1000, 1000) == pytest.approx(0.5)
    assert expected_score(1200, 1000) == pytest.approx(1 / (1 + 10 ** (-0.5)))
    # symmetric
    assert expected_score(1000, 1200) == pytest.approx(1 - expected_score(1200, 1000))


def test_elo_single_match_update():
    r = elo_ratings([("A", "B", 1.0)], k=32, initial=1000)
    assert r["A"] == pytest.approx(1016.0)  # 1000 + 32*(1-0.5)
    assert r["B"] == pytest.approx(984.0)


def test_elo_tie_no_change_when_equal():
    r = elo_ratings([("A", "B", 0.5)])
    assert r["A"] == pytest.approx(1000.0)
    assert r["B"] == pytest.approx(1000.0)


def test_elo_ranking_order():
    r = elo_ratings(MATCHES)
    assert [p for p, _ in ranking(r)] == ["A", "B", "C"]


def test_win_rate():
    wr = win_rate(MATCHES)
    assert wr["A"] == pytest.approx(1.0)
    assert wr["B"] == pytest.approx(0.5)
    assert wr["C"] == pytest.approx(0.0)


def test_bradley_terry_orders_correctly():
    bt = bradley_terry(MATCHES)
    assert bt["A"] > bt["B"] > bt["C"]
    assert [p for p, _ in ranking(bt)] == ["A", "B", "C"]


def test_bradley_terry_order_independent():
    shuffled = list(reversed(MATCHES))
    bt1 = bradley_terry(MATCHES)
    bt2 = bradley_terry(shuffled)
    for k in bt1:
        assert bt1[k] == pytest.approx(bt2[k], abs=1e-6)


def test_bradley_terry_even_matchup_equal_ratings():
    even = [("A", "B", 1.0), ("A", "B", 0.0)]
    bt = bradley_terry(even)
    assert bt["A"] == pytest.approx(bt["B"], abs=1e-6)
    assert bt["A"] == pytest.approx(1000.0, abs=1e-6)


def test_bradley_terry_deterministic():
    assert bradley_terry(MATCHES) == bradley_terry(MATCHES)


def test_bradley_terry_handles_separable_data():
    # A never loses -> would diverge without regularization; reg keeps it finite
    bt = bradley_terry([("A", "B", 1.0), ("A", "B", 1.0), ("A", "B", 1.0)])
    assert math.isfinite(bt["A"]) and math.isfinite(bt["B"])
    assert bt["A"] > bt["B"]


def test_invalid_outcome_raises():
    with pytest.raises(ValueError):
        elo_ratings([("A", "B", 0.7)])
    with pytest.raises(ValueError):
        bradley_terry([("A", "B", 2.0)])


def test_empty_inputs():
    assert elo_ratings([]) == {}
    assert bradley_terry([]) == {}
    assert win_rate([]) == {}
