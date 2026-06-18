"""Tool-call correctness: did the model call the right tool with the right arguments?

For agents and function/tool calling, a "correct" call has two parts: the **tool
name** matches the expected one, and the **arguments** match. This recipe scores a
predicted tool call against an expected one and aggregates over a dataset, with
sensible knobs because exact argument equality is often too strict:

- ``name``: exact match (case-insensitive option).
- ``arguments``: per-key comparison with configurable matching:
  - exact equality (default),
  - case/whitespace-insensitive string comparison,
  - a numeric tolerance for floats,
  - ignoring keys the schema marks optional / not-evaluated.

Aggregates: name accuracy, exact-call accuracy (name AND all args), and mean
argument-level F1 (handling extra and missing arguments).

Pure standard library, offline. You supply parsed tool-call dicts.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

# A tool call: {"name": str, "arguments": {param: value}}
ToolCall = Mapping[str, Any]
ArgMatcher = Callable[[Any, Any], bool]


def _default_arg_match(
    expected: Any,
    predicted: Any,
    *,
    case_insensitive: bool,
    float_tol: float,
) -> bool:
    if isinstance(expected, str) and isinstance(predicted, str) and case_insensitive:
        return expected.strip().lower() == predicted.strip().lower()
    if (
        isinstance(expected, float)
        or isinstance(predicted, float)
    ) and isinstance(expected, (int, float)) and isinstance(predicted, (int, float)):
        return abs(float(expected) - float(predicted)) <= float_tol
    return expected == predicted


def name_matches(
    expected: ToolCall, predicted: ToolCall, case_insensitive: bool = False
) -> bool:
    """True if the predicted tool name equals the expected tool name."""
    e = str(expected.get("name", ""))
    p = str(predicted.get("name", ""))
    if case_insensitive:
        return e.lower() == p.lower()
    return e == p


def argument_scores(
    expected: ToolCall,
    predicted: ToolCall,
    *,
    case_insensitive: bool = False,
    float_tol: float = 0.0,
    ignore_keys: Optional[Sequence[str]] = None,
    arg_matcher: Optional[ArgMatcher] = None,
) -> Dict[str, float]:
    """Per-key argument comparison -> precision/recall/F1 over argument keys.

    A key is a "match" if it is present in both with matching values. Extra
    predicted keys hurt precision; missing expected keys hurt recall.
    """
    ignore = set(ignore_keys or [])
    e_args = {k: v for k, v in expected.get("arguments", {}).items() if k not in ignore}
    p_args = {k: v for k, v in predicted.get("arguments", {}).items() if k not in ignore}

    def match(ev: Any, pv: Any) -> bool:
        if arg_matcher is not None:
            return arg_matcher(ev, pv)
        return _default_arg_match(
            ev, pv, case_insensitive=case_insensitive, float_tol=float_tol
        )

    matched = sum(
        1 for k, ev in e_args.items() if k in p_args and match(ev, p_args[k])
    )
    precision = matched / len(p_args) if p_args else (1.0 if not e_args else 0.0)
    recall = matched / len(e_args) if e_args else (1.0 if not p_args else 0.0)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1, "matched": matched}


def is_correct_call(
    expected: ToolCall,
    predicted: ToolCall,
    *,
    case_insensitive: bool = False,
    float_tol: float = 0.0,
    ignore_keys: Optional[Sequence[str]] = None,
    arg_matcher: Optional[ArgMatcher] = None,
) -> bool:
    """True if the name matches AND all expected arguments match with no missing or
    spurious keys (argument F1 == 1.0)."""
    if not name_matches(expected, predicted, case_insensitive):
        return False
    scores = argument_scores(
        expected,
        predicted,
        case_insensitive=case_insensitive,
        float_tol=float_tol,
        ignore_keys=ignore_keys,
        arg_matcher=arg_matcher,
    )
    return scores["f1"] == 1.0


def evaluate(
    expected_calls: Sequence[ToolCall],
    predicted_calls: Sequence[ToolCall],
    *,
    case_insensitive: bool = False,
    float_tol: float = 0.0,
    ignore_keys: Optional[Sequence[str]] = None,
    arg_matcher: Optional[ArgMatcher] = None,
) -> Dict[str, float]:
    """Aggregate name accuracy, exact-call accuracy, and mean argument F1."""
    if len(expected_calls) != len(predicted_calls):
        raise ValueError("expected and predicted call lists must have equal length")
    n = len(expected_calls)
    if n == 0:
        return {"name_accuracy": 0.0, "exact_call_accuracy": 0.0, "arg_f1": 0.0}
    name_hits = exact_hits = 0
    arg_f1_sum = 0.0
    for exp, pred in zip(expected_calls, predicted_calls):
        if name_matches(exp, pred, case_insensitive):
            name_hits += 1
        if is_correct_call(
            exp,
            pred,
            case_insensitive=case_insensitive,
            float_tol=float_tol,
            ignore_keys=ignore_keys,
            arg_matcher=arg_matcher,
        ):
            exact_hits += 1
        arg_f1_sum += argument_scores(
            exp,
            pred,
            case_insensitive=case_insensitive,
            float_tol=float_tol,
            ignore_keys=ignore_keys,
            arg_matcher=arg_matcher,
        )["f1"]
    return {
        "name_accuracy": name_hits / n,
        "exact_call_accuracy": exact_hits / n,
        "arg_f1": arg_f1_sum / n,
    }


if __name__ == "__main__":
    expected = [
        {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}},
        {"name": "search", "arguments": {"query": "python"}},
    ]
    predicted = [
        {"name": "get_weather", "arguments": {"city": "paris", "unit": "celsius"}},
        {"name": "search", "arguments": {"query": "python", "limit": 5}},
    ]
    print("strict:", evaluate(expected, predicted))
    print("lenient:", evaluate(expected, predicted, case_insensitive=True))
