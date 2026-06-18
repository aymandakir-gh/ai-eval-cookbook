import pytest

from ai_eval_cookbook.tool_call_correctness import (
    argument_scores,
    evaluate,
    is_correct_call,
    name_matches,
)


def test_name_matches():
    a = {"name": "get_weather", "arguments": {}}
    b = {"name": "get_weather", "arguments": {}}
    assert name_matches(a, b) is True
    assert name_matches(a, {"name": "search", "arguments": {}}) is False


def test_name_matches_case_insensitive():
    a = {"name": "GetWeather", "arguments": {}}
    b = {"name": "getweather", "arguments": {}}
    assert name_matches(a, b) is False
    assert name_matches(a, b, case_insensitive=True) is True


def test_argument_scores_exact():
    exp = {"name": "f", "arguments": {"a": 1, "b": 2}}
    pred = {"name": "f", "arguments": {"a": 1, "b": 2}}
    s = argument_scores(exp, pred)
    assert s["f1"] == pytest.approx(1.0)
    assert s["matched"] == 2


def test_argument_scores_partial():
    exp = {"name": "f", "arguments": {"city": "Paris", "unit": "celsius"}}
    pred = {"name": "f", "arguments": {"city": "paris", "unit": "celsius"}}
    s = argument_scores(exp, pred)  # case mismatch on city
    assert s["matched"] == 1
    assert s["precision"] == pytest.approx(0.5)
    assert s["recall"] == pytest.approx(0.5)
    assert s["f1"] == pytest.approx(0.5)


def test_argument_scores_extra_predicted_arg():
    exp = {"name": "f", "arguments": {"query": "python"}}
    pred = {"name": "f", "arguments": {"query": "python", "limit": 5}}
    s = argument_scores(exp, pred)
    assert s["precision"] == pytest.approx(0.5)  # extra key
    assert s["recall"] == pytest.approx(1.0)
    assert s["f1"] == pytest.approx(2 / 3)


def test_case_insensitive_argument_matching():
    exp = {"name": "f", "arguments": {"city": "Paris"}}
    pred = {"name": "f", "arguments": {"city": "paris"}}
    s = argument_scores(exp, pred, case_insensitive=True)
    assert s["f1"] == pytest.approx(1.0)


def test_float_tolerance():
    exp = {"name": "f", "arguments": {"temp": 1.0}}
    pred = {"name": "f", "arguments": {"temp": 1.05}}
    assert argument_scores(exp, pred, float_tol=0.0)["matched"] == 0
    assert argument_scores(exp, pred, float_tol=0.1)["matched"] == 1


def test_ignore_keys():
    exp = {"name": "f", "arguments": {"q": "x", "page": 1}}
    pred = {"name": "f", "arguments": {"q": "x", "page": 9}}
    s = argument_scores(exp, pred, ignore_keys=["page"])
    assert s["f1"] == pytest.approx(1.0)


def test_custom_arg_matcher():
    # substring matcher
    matcher = lambda e, p: isinstance(p, str) and isinstance(e, str) and e in p
    exp = {"name": "f", "arguments": {"q": "cat"}}
    pred = {"name": "f", "arguments": {"q": "cats and dogs"}}
    s = argument_scores(exp, pred, arg_matcher=matcher)
    assert s["f1"] == pytest.approx(1.0)


def test_is_correct_call():
    exp = {"name": "f", "arguments": {"a": 1}}
    assert is_correct_call(exp, {"name": "f", "arguments": {"a": 1}}) is True
    assert is_correct_call(exp, {"name": "g", "arguments": {"a": 1}}) is False
    assert is_correct_call(exp, {"name": "f", "arguments": {"a": 2}}) is False
    assert is_correct_call(exp, {"name": "f", "arguments": {"a": 1, "b": 2}}) is False


def test_evaluate_aggregates():
    exp = [
        {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}},
        {"name": "search", "arguments": {"query": "python"}},
    ]
    pred = [
        {"name": "get_weather", "arguments": {"city": "paris", "unit": "celsius"}},
        {"name": "search", "arguments": {"query": "python", "limit": 5}},
    ]
    strict = evaluate(exp, pred)
    assert strict["name_accuracy"] == pytest.approx(1.0)
    assert strict["exact_call_accuracy"] == pytest.approx(0.0)
    assert strict["arg_f1"] == pytest.approx((0.5 + 2 / 3) / 2)

    lenient = evaluate(exp, pred, case_insensitive=True)
    assert lenient["exact_call_accuracy"] == pytest.approx(0.5)  # call0 now exact


def test_empty_arguments_both():
    exp = {"name": "f", "arguments": {}}
    pred = {"name": "f", "arguments": {}}
    assert argument_scores(exp, pred)["f1"] == pytest.approx(1.0)
    assert is_correct_call(exp, pred) is True


def test_length_mismatch_and_empty():
    with pytest.raises(ValueError):
        evaluate([{"name": "a", "arguments": {}}], [])
    assert evaluate([], []) == {
        "name_accuracy": 0.0,
        "exact_call_accuracy": 0.0,
        "arg_f1": 0.0,
    }
