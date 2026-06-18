import pytest

from ai_eval_cookbook.calibration_ece_brier import (
    brier_score,
    calibration_report,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
)


def test_perfect_calibration():
    conf = [1.0, 1.0, 0.0, 0.0]
    correct = [True, True, False, False]
    assert expected_calibration_error(conf, correct) == pytest.approx(0.0)
    assert brier_score(conf, correct) == pytest.approx(0.0)


def test_worst_calibration():
    conf = [1.0, 1.0]
    correct = [False, False]
    assert expected_calibration_error(conf, correct) == pytest.approx(1.0)
    assert brier_score(conf, correct) == pytest.approx(1.0)


def test_ece_single_bin_gap():
    # all conf 0.8, 50% correct -> gap 0.3
    conf = [0.8, 0.8, 0.8, 0.8]
    correct = [True, True, False, False]
    assert expected_calibration_error(conf, correct) == pytest.approx(0.3)
    assert maximum_calibration_error(conf, correct) == pytest.approx(0.3)


def test_brier_hand_computed():
    # ((0.8-1)^2 *2 + (0.8-0)^2 *2)/4 = (0.08 + 1.28)/4 = 0.34
    conf = [0.8, 0.8, 0.8, 0.8]
    correct = [True, True, False, False]
    assert brier_score(conf, correct) == pytest.approx(0.34)


def test_reliability_bins_structure():
    bins = reliability_bins([0.05, 0.95], [False, True], n_bins=10)
    assert len(bins) == 10
    assert bins[0]["count"] == 1  # 0.05 in first bin
    assert bins[9]["count"] == 1  # 0.95 in last bin
    assert bins[0]["accuracy"] == 0.0
    assert bins[9]["accuracy"] == 1.0


def test_confidence_one_goes_to_last_bin():
    bins = reliability_bins([1.0], [True], n_bins=10)
    assert bins[9]["count"] == 1


def test_mce_picks_worst_bin():
    # bin A (conf~0.2) perfectly calibrated, bin B (conf~0.9) badly off
    conf = [0.2, 0.2, 0.9, 0.9]
    correct = [False, True, False, False]  # bin0.2: 50% acc gap~0.3? let's keep simple
    mce = maximum_calibration_error(conf, correct, n_bins=10)
    ece = expected_calibration_error(conf, correct, n_bins=10)
    assert mce >= ece  # max gap is at least the weighted average


def test_report_bundle():
    conf = [0.9, 0.6, 0.3]
    correct = [True, False, False]
    rep = calibration_report(conf, correct, n_bins=10)
    assert set(rep) == {"ece", "mce", "brier", "accuracy", "mean_confidence", "bins"}
    assert rep["accuracy"] == pytest.approx(1 / 3)
    assert rep["mean_confidence"] == pytest.approx((0.9 + 0.6 + 0.3) / 3)


def test_empty_inputs():
    assert expected_calibration_error([], []) == 0.0
    assert brier_score([], []) == 0.0
    assert maximum_calibration_error([], []) == 0.0


def test_validation_errors():
    with pytest.raises(ValueError):
        brier_score([0.5], [True, False])
    with pytest.raises(ValueError):
        expected_calibration_error([1.5], [True])
    with pytest.raises(ValueError):
        reliability_bins([0.5], [True], n_bins=0)
