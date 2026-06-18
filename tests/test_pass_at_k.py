from math import comb

import pytest

from ai_eval_cookbook.pass_at_k import (
    count_passing,
    keyword_mock_runner,
    pass_at_k,
    pass_at_k_estimator,
    pass_at_k_report,
    problem_pass_at_k,
)


def test_estimator_matches_exact_formula():
    # 1 - C(n-c, k) / C(n, k)
    assert pass_at_k_estimator(5, 2, 1) == pytest.approx(1 - comb(3, 1) / comb(5, 1))
    assert pass_at_k_estimator(5, 2, 2) == pytest.approx(1 - comb(3, 2) / comb(5, 2))
    assert pass_at_k_estimator(10, 3, 4) == pytest.approx(1 - comb(7, 4) / comb(10, 4))


def test_estimator_all_fail_is_zero():
    assert pass_at_k_estimator(5, 0, 1) == 0.0
    assert pass_at_k_estimator(5, 0, 5) == 0.0


def test_estimator_more_passes_than_failures_for_k():
    # n - c < k -> guaranteed at least one pass
    assert pass_at_k_estimator(5, 2, 5) == 1.0
    assert pass_at_k_estimator(5, 4, 2) == 1.0


def test_estimator_all_pass_is_one():
    assert pass_at_k_estimator(5, 5, 1) == 1.0


def test_estimator_monotonic_in_k():
    # pass@k should not decrease as k grows
    vals = [pass_at_k_estimator(10, 3, k) for k in range(1, 8)]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))


def test_estimator_errors():
    with pytest.raises(ValueError):
        pass_at_k_estimator(5, 6, 1)  # c > n
    with pytest.raises(ValueError):
        pass_at_k_estimator(5, 2, 6)  # k > n
    with pytest.raises(ValueError):
        pass_at_k_estimator(5, 2, 0)  # k <= 0


def test_mock_runner_and_count():
    runner = keyword_mock_runner("return")
    samples = ["def f(): return 1", "def f(): pass", "x = return_value"]
    assert runner("return x") is True
    assert runner("pass") is False
    assert count_passing(samples, runner) == 2


def test_problem_pass_at_k():
    problem = [
        "return 1",  # pass
        "pass",  # fail
        "return 2",  # pass
        "print()",  # fail
        "pass",  # fail
    ]
    runner = keyword_mock_runner("return")
    # n=5, c=2
    assert problem_pass_at_k(problem, runner, 1) == pytest.approx(0.4)
    assert problem_pass_at_k(problem, runner, 2) == pytest.approx(0.7)
    assert problem_pass_at_k(problem, runner, 5) == pytest.approx(1.0)


def test_pass_at_k_mean_across_problems():
    runner = keyword_mock_runner("return")
    p1 = ["return 1", "return 2"]  # n=2 c=2 -> pass@1 = 1.0
    p2 = ["pass", "pass"]  # n=2 c=0 -> pass@1 = 0.0
    assert pass_at_k([p1, p2], runner, 1) == pytest.approx(0.5)


def test_report():
    runner = keyword_mock_runner("return")
    problem = ["return 1", "pass", "return 2", "print()", "pass"]
    rep = pass_at_k_report([problem], runner, ks=[1, 2, 5])
    assert rep["per_problem"] == [(5, 2)]
    assert rep["pass@1"] == pytest.approx(0.4)
    assert rep["pass@2"] == pytest.approx(0.7)
    assert rep["pass@5"] == pytest.approx(1.0)


def test_report_skips_k_greater_than_n():
    runner = keyword_mock_runner("return")
    rep = pass_at_k_report([["return 1", "pass"]], runner, ks=[5])  # n=2 < 5
    assert rep["pass@5"] == 0.0  # no eligible problems


def test_empty_problems():
    assert pass_at_k([], keyword_mock_runner(), 1) == 0.0


def test_injected_custom_runner():
    # runner that passes if source length is even
    runner = lambda s: len(s) % 2 == 0
    samples = ["ab", "abc", "abcd"]  # 2 pass (len 2, 4)
    assert count_passing(samples, runner) == 2
