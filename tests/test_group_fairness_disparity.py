import pytest

from ai_eval_cookbook.group_fairness_disparity import (
    demographic_parity,
    fairness_report,
    group_metric_disparity,
    per_group_rates,
    selection_rate,
)

GROUPS = ["A", "A", "A", "A", "B", "B", "B", "B"]
PREDS = [True, True, False, False, True, False, False, False]
LABELS = [True, False, True, False, True, True, False, False]


def test_selection_rate():
    rates = selection_rate(GROUPS, PREDS)
    assert rates["A"] == pytest.approx(0.5)  # 2/4
    assert rates["B"] == pytest.approx(0.25)  # 1/4


def test_demographic_parity_difference_and_ratio():
    dp = demographic_parity(GROUPS, PREDS)
    assert dp["selection_rate_difference"] == pytest.approx(0.25)
    assert dp["selection_rate_ratio"] == pytest.approx(0.5)


def test_per_group_rates():
    rates = per_group_rates(GROUPS, PREDS, LABELS)
    # A: tp=1 fp=1 fn=1 tn=1 -> tpr .5 fpr .5 acc .5
    assert rates["A"]["tpr"] == pytest.approx(0.5)
    assert rates["A"]["fpr"] == pytest.approx(0.5)
    assert rates["A"]["accuracy"] == pytest.approx(0.5)
    # B: tp=1 fp=0 fn=1 tn=2 -> tpr .5 fpr 0 acc .75
    assert rates["B"]["tpr"] == pytest.approx(0.5)
    assert rates["B"]["fpr"] == pytest.approx(0.0)
    assert rates["B"]["accuracy"] == pytest.approx(0.75)


def test_equal_opportunity_and_equalized_odds():
    rep = fairness_report(GROUPS, PREDS, LABELS)
    # TPR equal across groups -> equal opportunity satisfied
    assert rep["equal_opportunity"]["tpr_difference"] == pytest.approx(0.0)
    # but FPR differs -> equalized odds violated
    assert rep["equalized_odds"]["fpr"]["fpr_difference"] == pytest.approx(0.5)
    assert rep["equalized_odds"]["tpr"]["tpr_difference"] == pytest.approx(0.0)


def test_accuracy_disparity():
    rep = fairness_report(GROUPS, PREDS, LABELS)
    assert rep["accuracy"]["accuracy_difference"] == pytest.approx(0.25)


def test_report_without_labels():
    rep = fairness_report(GROUPS, PREDS)
    assert "demographic_parity" in rep
    assert "equal_opportunity" not in rep


def test_group_metric_disparity_default_mean():
    res = group_metric_disparity(["A", "A", "B"], [1.0, 0.0, 1.0])
    assert res["per_group"]["A"] == pytest.approx(0.5)
    assert res["per_group"]["B"] == pytest.approx(1.0)
    assert res["metric_difference"] == pytest.approx(0.5)
    assert res["metric_ratio"] == pytest.approx(0.5)


def test_group_metric_disparity_custom_aggregate():
    res = group_metric_disparity(
        ["A", "A", "B"], [1.0, 3.0, 5.0], aggregate=max
    )
    assert res["per_group"]["A"] == pytest.approx(3.0)
    assert res["per_group"]["B"] == pytest.approx(5.0)


def test_perfect_parity():
    groups = ["A", "A", "B", "B"]
    preds = [True, False, True, False]
    dp = demographic_parity(groups, preds)
    assert dp["selection_rate_difference"] == pytest.approx(0.0)
    assert dp["selection_rate_ratio"] == pytest.approx(1.0)


def test_length_mismatch():
    with pytest.raises(ValueError):
        selection_rate(["A"], [True, False])
    with pytest.raises(ValueError):
        per_group_rates(["A", "B"], [True, False], [True])


def test_single_group():
    dp = demographic_parity(["A", "A"], [True, False])
    assert dp["selection_rate_difference"] == 0.0
    assert dp["selection_rate_ratio"] == 1.0
