"""Classification metrics: precision, recall, F1 (per-class, macro, micro) and a
confusion matrix.

Many LLM evaluations reduce to classification: intent detection, routing, content
moderation labels, sentiment, multiple-choice answers. This module computes the
standard metrics from two equal-length sequences of labels (gold vs predicted).
Labels can be any hashable value (strings, ints, enums).

Pure standard library, offline. No model is called.
"""

from __future__ import annotations

from typing import Dict, Hashable, List, Sequence, Tuple

Label = Hashable


def _validate(y_true: Sequence[Label], y_pred: Sequence[Label]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have equal length: %d != %d"
            % (len(y_true), len(y_pred))
        )


def labels_in(y_true: Sequence[Label], y_pred: Sequence[Label]) -> List[Label]:
    """Sorted (where possible) union of labels appearing in either sequence."""
    seen = set(y_true) | set(y_pred)
    try:
        return sorted(seen)
    except TypeError:
        # Mixed/unorderable labels: fall back to first-seen order, stable.
        ordered: List[Label] = []
        for lab in list(y_true) + list(y_pred):
            if lab not in ordered:
                ordered.append(lab)
        return ordered


def confusion_matrix(
    y_true: Sequence[Label], y_pred: Sequence[Label], labels: Sequence[Label] = None
) -> Tuple[List[Label], List[List[int]]]:
    """Return ``(labels, matrix)`` where ``matrix[i][j]`` counts examples whose
    true label is ``labels[i]`` and predicted label is ``labels[j]``."""
    _validate(y_true, y_pred)
    labs = list(labels) if labels is not None else labels_in(y_true, y_pred)
    index = {lab: i for i, lab in enumerate(labs)}
    mat = [[0 for _ in labs] for _ in labs]
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            mat[index[t]][index[p]] += 1
    return labs, mat


def _counts(
    y_true: Sequence[Label], y_pred: Sequence[Label], label: Label
) -> Tuple[int, int, int]:
    """Return (tp, fp, fn) for a single label treated one-vs-rest."""
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if p == label and t == label:
            tp += 1
        elif p == label and t != label:
            fp += 1
        elif p != label and t == label:
            fn += 1
    return tp, fp, fn


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


def per_class_metrics(
    y_true: Sequence[Label], y_pred: Sequence[Label], labels: Sequence[Label] = None
) -> Dict[Label, Dict[str, float]]:
    """Per-class precision/recall/F1 and support (number of true instances)."""
    _validate(y_true, y_pred)
    labs = list(labels) if labels is not None else labels_in(y_true, y_pred)
    out: Dict[Label, Dict[str, float]] = {}
    for lab in labs:
        tp, fp, fn = _counts(y_true, y_pred, lab)
        p, r, f1 = _prf(tp, fp, fn)
        out[lab] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": tp + fn,
        }
    return out


def macro_average(
    y_true: Sequence[Label], y_pred: Sequence[Label], labels: Sequence[Label] = None
) -> Dict[str, float]:
    """Unweighted mean of per-class precision/recall/F1 (each class counts once)."""
    pc = per_class_metrics(y_true, y_pred, labels)
    if not pc:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    n = len(pc)
    return {
        "precision": sum(m["precision"] for m in pc.values()) / n,
        "recall": sum(m["recall"] for m in pc.values()) / n,
        "f1": sum(m["f1"] for m in pc.values()) / n,
    }


def micro_average(
    y_true: Sequence[Label], y_pred: Sequence[Label], labels: Sequence[Label] = None
) -> Dict[str, float]:
    """Globally pooled precision/recall/F1 (each example counts once).

    For single-label classification over the full label set, micro precision,
    recall, and F1 all equal overall accuracy.
    """
    _validate(y_true, y_pred)
    labs = list(labels) if labels is not None else labels_in(y_true, y_pred)
    TP = FP = FN = 0
    for lab in labs:
        tp, fp, fn = _counts(y_true, y_pred, lab)
        TP += tp
        FP += fp
        FN += fn
    p, r, f1 = _prf(TP, FP, FN)
    return {"precision": p, "recall": r, "f1": f1}


def accuracy(y_true: Sequence[Label], y_pred: Sequence[Label]) -> float:
    """Fraction of exactly-correct labels."""
    _validate(y_true, y_pred)
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def classification_report(
    y_true: Sequence[Label], y_pred: Sequence[Label], labels: Sequence[Label] = None
) -> Dict[str, object]:
    """Bundle accuracy, per-class metrics, and macro/micro averages."""
    return {
        "accuracy": accuracy(y_true, y_pred),
        "per_class": per_class_metrics(y_true, y_pred, labels),
        "macro": macro_average(y_true, y_pred, labels),
        "micro": micro_average(y_true, y_pred, labels),
    }


if __name__ == "__main__":
    y_true = ["spam", "spam", "ham", "ham", "ham"]
    y_pred = ["spam", "ham", "ham", "ham", "spam"]
    labs, mat = confusion_matrix(y_true, y_pred)
    print("labels:", labs)
    print("confusion matrix:", mat)
    report = classification_report(y_true, y_pred)
    print("accuracy:", report["accuracy"])
    print("macro:", report["macro"])
    print("micro:", report["micro"])
