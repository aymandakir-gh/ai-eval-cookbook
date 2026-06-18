import pytest

from ai_eval_cookbook.classification_metrics import (
    accuracy,
    classification_report,
    confusion_matrix,
    macro_average,
    micro_average,
    per_class_metrics,
)


# Hand-built example (spam=positive class of interest):
#   true: spam spam ham ham ham
#   pred: spam ham  ham ham spam
# spam: tp=1 (idx0), fp=1 (idx4), fn=1 (idx1)
# ham : tp=2 (idx2,3), fp=1 (idx1), fn=1 (idx4)
Y_TRUE = ["spam", "spam", "ham", "ham", "ham"]
Y_PRED = ["spam", "ham", "ham", "ham", "spam"]


def test_confusion_matrix():
    labs, mat = confusion_matrix(Y_TRUE, Y_PRED)
    assert labs == ["ham", "spam"]  # sorted
    # rows = true, cols = pred, order [ham, spam]
    # ham true: 2 ham, 1 spam ; spam true: 1 ham, 1 spam
    assert mat == [[2, 1], [1, 1]]


def test_per_class_spam():
    pc = per_class_metrics(Y_TRUE, Y_PRED)
    spam = pc["spam"]
    assert spam["precision"] == pytest.approx(0.5)  # 1/(1+1)
    assert spam["recall"] == pytest.approx(0.5)  # 1/(1+1)
    assert spam["f1"] == pytest.approx(0.5)
    assert spam["support"] == 2


def test_per_class_ham():
    pc = per_class_metrics(Y_TRUE, Y_PRED)
    ham = pc["ham"]
    assert ham["precision"] == pytest.approx(2 / 3)  # 2/(2+1)
    assert ham["recall"] == pytest.approx(2 / 3)  # 2/(2+1)
    assert ham["f1"] == pytest.approx(2 / 3)
    assert ham["support"] == 3


def test_macro_average():
    macro = macro_average(Y_TRUE, Y_PRED)
    # mean of f1 [0.5 (spam), 0.6667 (ham)]
    assert macro["f1"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert macro["precision"] == pytest.approx((0.5 + 2 / 3) / 2)


def test_micro_equals_accuracy_for_single_label():
    micro = micro_average(Y_TRUE, Y_PRED)
    acc = accuracy(Y_TRUE, Y_PRED)
    assert micro["f1"] == pytest.approx(acc)
    assert micro["precision"] == pytest.approx(acc)
    assert acc == pytest.approx(3 / 5)


def test_zero_division_is_zero():
    # Predict a label that never occurs as truth -> precision 0, recall undefined->0
    pc = per_class_metrics(["a", "a"], ["b", "b"])
    assert pc["b"]["precision"] == 0.0
    assert pc["a"]["recall"] == 0.0


def test_report_bundle_keys():
    rep = classification_report(Y_TRUE, Y_PRED)
    assert set(rep.keys()) == {"accuracy", "per_class", "macro", "micro"}


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy(["a"], ["a", "b"])


def test_empty_inputs():
    assert accuracy([], []) == 0.0
    assert macro_average([], []) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
