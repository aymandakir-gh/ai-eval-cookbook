import pytest

from ai_eval_cookbook.latency_cost_budget import (
    check_budgets,
    percentile,
    record_cost,
    summarize,
)


def test_percentile_interpolation():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile(list(range(1, 11)), 90) == pytest.approx(9.1)
    assert percentile([1, 2, 3, 4], 0) == pytest.approx(1.0)
    assert percentile([1, 2, 3, 4], 100) == pytest.approx(4.0)


def test_percentile_single_and_empty():
    assert percentile([5], 50) == 5.0
    assert percentile([], 95) == 0.0


def test_percentile_bad_p():
    with pytest.raises(ValueError):
        percentile([1, 2], 150)


def test_record_cost_explicit():
    assert record_cost({"cost": 0.5}) == pytest.approx(0.5)


def test_record_cost_from_tokens():
    prices = {"input": 0.003, "output": 0.015}
    cost = record_cost({"input_tokens": 1000, "output_tokens": 200}, prices)
    # 1.0*0.003 + 0.2*0.015 = 0.003 + 0.003
    assert cost == pytest.approx(0.006)


def test_record_cost_default_zero():
    assert record_cost({"latency_ms": 100}) == 0.0


def test_summarize():
    log = [
        {"latency_ms": 100, "cost": 0.01},
        {"latency_ms": 200, "cost": 0.02},
        {"latency_ms": 300, "cost": 0.03},
    ]
    s = summarize(log)
    assert s["count"] == 3
    assert s["latency_p50"] == pytest.approx(200.0)
    assert s["latency_max"] == 300.0
    assert s["total_cost"] == pytest.approx(0.06)
    assert s["mean_cost"] == pytest.approx(0.02)


def test_check_budgets_pass():
    log = [{"id": "a", "latency_ms": 100, "cost": 0.01}]
    res = check_budgets(log, max_latency_p95_ms=200, max_total_cost=0.1)
    assert res["passed"] is True
    assert res["violations"] == []


def test_check_budgets_latency_p95_violation():
    log = [{"latency_ms": 100}, {"latency_ms": 5000}]
    res = check_budgets(log, max_latency_p95_ms=1000)
    assert res["passed"] is False
    assert any("p95 latency" in v for v in res["violations"])


def test_check_budgets_cost_violation():
    log = [{"cost": 1.0}, {"cost": 1.0}]
    res = check_budgets(log, max_total_cost=1.5)
    assert res["passed"] is False
    assert any("total cost" in v for v in res["violations"])


def test_check_budgets_per_request_latency_ids():
    log = [
        {"id": "fast", "latency_ms": 100},
        {"id": "slow", "latency_ms": 3000},
    ]
    res = check_budgets(log, max_request_latency_ms=2000)
    assert res["passed"] is False
    assert res["over_budget_request_ids"] == ["slow"]


def test_check_budgets_uses_index_when_no_id():
    log = [{"latency_ms": 100}, {"latency_ms": 3000}]
    res = check_budgets(log, max_request_latency_ms=2000)
    assert res["over_budget_request_ids"] == [1]


def test_check_budgets_token_pricing():
    log = [{"input_tokens": 1000, "output_tokens": 1000}]
    prices = {"input": 0.01, "output": 0.03}  # cost = 0.04
    res = check_budgets(log, max_total_cost=0.03, price_per_1k=prices)
    assert res["passed"] is False
    res2 = check_budgets(log, max_total_cost=0.05, price_per_1k=prices)
    assert res2["passed"] is True


def test_only_specified_budgets_checked():
    log = [{"latency_ms": 9999, "cost": 9999}]
    # no budgets passed -> always passes
    assert check_budgets(log)["passed"] is True
