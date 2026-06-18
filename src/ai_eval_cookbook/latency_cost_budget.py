"""Latency and cost budgets over a run log.

Quality is not the only dimension that ships an LLM feature: latency and cost are
hard constraints. This recipe summarizes a run log (a list of per-request records)
and checks it against budgets, returning pass/fail plus the offending requests.

A record is a mapping with optional keys:
- ``latency_ms``: wall-clock latency for the request,
- ``cost`` (or computed from ``input_tokens``/``output_tokens`` + a price table),
- ``id``: an identifier for reporting.

It computes latency percentiles (p50/p90/p95/p99), total and mean cost, and asserts
budgets such as "p95 latency <= 1500 ms" and "total cost <= $5".

Pure standard library, offline.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

Record = Mapping[str, Any]


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (``p`` in [0, 100]).

    Matches the common "type 7" definition used by NumPy's default. Empty input
    -> 0.0.
    """
    if not values:
        return 0.0
    if not 0 <= p <= 100:
        raise ValueError("p must be in [0, 100]")
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    rank = (p / 100.0) * (len(s) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(s[lo])
    frac = rank - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


def record_cost(
    record: Record, price_per_1k: Optional[Mapping[str, float]] = None
) -> float:
    """Cost of a single record.

    Uses ``record["cost"]`` if present. Otherwise, if ``price_per_1k`` is given with
    ``input``/``output`` USD-per-1k-token rates, computes cost from
    ``input_tokens``/``output_tokens``. Falls back to 0.0.
    """
    if "cost" in record and record["cost"] is not None:
        return float(record["cost"])
    if price_per_1k:
        in_tok = float(record.get("input_tokens", 0) or 0)
        out_tok = float(record.get("output_tokens", 0) or 0)
        cost = in_tok / 1000.0 * price_per_1k.get("input", 0.0)
        cost += out_tok / 1000.0 * price_per_1k.get("output", 0.0)
        return cost
    return 0.0


def summarize(
    log: Sequence[Record],
    price_per_1k: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Latency percentiles and cost aggregates over the run log."""
    latencies = [
        float(r["latency_ms"]) for r in log if r.get("latency_ms") is not None
    ]
    costs = [record_cost(r, price_per_1k) for r in log]
    n = len(log)
    return {
        "count": n,
        "latency_p50": percentile(latencies, 50),
        "latency_p90": percentile(latencies, 90),
        "latency_p95": percentile(latencies, 95),
        "latency_p99": percentile(latencies, 99),
        "latency_max": max(latencies) if latencies else 0.0,
        "latency_mean": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "total_cost": sum(costs),
        "mean_cost": (sum(costs) / n) if n else 0.0,
    }


def check_budgets(
    log: Sequence[Record],
    *,
    max_latency_p95_ms: Optional[float] = None,
    max_latency_p99_ms: Optional[float] = None,
    max_total_cost: Optional[float] = None,
    max_mean_cost: Optional[float] = None,
    max_request_latency_ms: Optional[float] = None,
    price_per_1k: Optional[Mapping[str, float]] = None,
) -> Dict[str, object]:
    """Assert budgets over the log. Returns ``{passed, summary, violations,
    over_budget_request_ids}``.

    Only budgets you pass are checked; ``None`` budgets are skipped. The per-request
    latency budget reports the ids (or indices) of requests that exceeded it.
    """
    summary = summarize(log, price_per_1k)
    violations: List[str] = []

    if max_latency_p95_ms is not None and summary["latency_p95"] > max_latency_p95_ms:
        violations.append(
            "p95 latency %.1f ms > budget %.1f ms"
            % (summary["latency_p95"], max_latency_p95_ms)
        )
    if max_latency_p99_ms is not None and summary["latency_p99"] > max_latency_p99_ms:
        violations.append(
            "p99 latency %.1f ms > budget %.1f ms"
            % (summary["latency_p99"], max_latency_p99_ms)
        )
    if max_total_cost is not None and summary["total_cost"] > max_total_cost:
        violations.append(
            "total cost %.4f > budget %.4f" % (summary["total_cost"], max_total_cost)
        )
    if max_mean_cost is not None and summary["mean_cost"] > max_mean_cost:
        violations.append(
            "mean cost %.4f > budget %.4f" % (summary["mean_cost"], max_mean_cost)
        )

    over_ids: List[Any] = []
    if max_request_latency_ms is not None:
        for i, r in enumerate(log):
            lat = r.get("latency_ms")
            if lat is not None and float(lat) > max_request_latency_ms:
                over_ids.append(r.get("id", i))
        if over_ids:
            violations.append(
                "%d request(s) exceeded per-request latency budget %.1f ms"
                % (len(over_ids), max_request_latency_ms)
            )

    return {
        "passed": not violations,
        "summary": summary,
        "violations": violations,
        "over_budget_request_ids": over_ids,
    }


if __name__ == "__main__":
    log = [
        {"id": "r1", "latency_ms": 300, "input_tokens": 1000, "output_tokens": 200},
        {"id": "r2", "latency_ms": 850, "input_tokens": 1500, "output_tokens": 400},
        {"id": "r3", "latency_ms": 2200, "input_tokens": 2000, "output_tokens": 800},
    ]
    prices = {"input": 0.003, "output": 0.015}
    result = check_budgets(
        log,
        max_latency_p95_ms=1500,
        max_request_latency_ms=2000,
        max_total_cost=0.05,
        price_per_1k=prices,
    )
    print("passed:", result["passed"])
    print("summary:", {k: round(v, 3) for k, v in result["summary"].items()})
    print("violations:", result["violations"])
    print("over-budget ids:", result["over_budget_request_ids"])
