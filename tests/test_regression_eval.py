import pytest

from ai_eval_cookbook.regression_eval import (
    classify,
    diff_runs,
    has_regressions,
)


def test_classify_boolean():
    assert classify(True, False) == "regression"
    assert classify(False, True) == "improvement"
    assert classify(True, True) == "unchanged"
    assert classify(False, False) == "unchanged"


def test_classify_numeric_with_tolerance():
    assert classify(0.8, 0.5, tol=0.05) == "regression"
    assert classify(0.5, 0.8, tol=0.05) == "improvement"
    assert classify(0.80, 0.82, tol=0.05) == "unchanged"  # within dead-band
    assert classify(0.80, 0.90, tol=0.05) == "improvement"


def test_diff_runs_counts_and_ids():
    baseline = [True, True, False, True, False, True]
    candidate = [True, False, True, True, False, True]
    rep = diff_runs(baseline, candidate, ids=["q1", "q2", "q3", "q4", "q5", "q6"])
    assert rep["regressions"] == 1
    assert rep["improvements"] == 1
    assert rep["unchanged"] == 4
    assert rep["regression_ids"] == ["q2"]
    assert rep["improvement_ids"] == ["q3"]
    assert rep["net_change"] == 0


def test_diff_runs_flat_mean_hides_churn():
    # mean unchanged but there IS a regression and an improvement
    baseline = [True, True, False]
    candidate = [True, False, True]
    rep = diff_runs(baseline, candidate)
    assert rep["mean_delta"] == pytest.approx(0.0)
    assert rep["regressions"] == 1
    assert rep["improvements"] == 1


def test_diff_runs_positional_ids():
    rep = diff_runs([True, False], [False, True])
    assert rep["regression_ids"] == [0]
    assert rep["improvement_ids"] == [1]


def test_diff_runs_means_and_delta():
    baseline = [1.0, 0.0, 0.0]
    candidate = [1.0, 1.0, 0.0]
    rep = diff_runs(baseline, candidate)
    assert rep["baseline_mean"] == pytest.approx(1 / 3)
    assert rep["candidate_mean"] == pytest.approx(2 / 3)
    assert rep["mean_delta"] == pytest.approx(1 / 3)


def test_has_regressions_gate():
    baseline = [True, True, True]
    candidate = [True, False, True]
    assert has_regressions(baseline, candidate) is True
    assert has_regressions(baseline, candidate, max_allowed=1) is False
    assert has_regressions(baseline, baseline) is False


def test_length_mismatch():
    with pytest.raises(ValueError):
        diff_runs([True], [True, False])


def test_ids_length_mismatch():
    with pytest.raises(ValueError):
        diff_runs([True, False], [False, True], ids=["only_one"])


def test_numeric_tolerance_in_diff():
    baseline = [0.80, 0.80]
    candidate = [0.82, 0.50]
    rep = diff_runs(baseline, candidate, tol=0.05)
    assert rep["unchanged"] == 1  # 0.82 within tol
    assert rep["regressions"] == 1  # 0.50 below
