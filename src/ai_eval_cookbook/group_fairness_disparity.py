"""Group fairness: measure metric disparities across groups.

When evaluating an LLM/classifier across subgroups (defined by a sensitive
attribute), aggregate quality can hide unequal treatment. This recipe computes a
metric *per group* and summarizes the **disparity** between groups:

- **selection rate** per group -> *demographic parity* difference (max − min) and
  ratio (min / max). Equal selection rates across groups = demographic parity.
- **accuracy / TPR / FPR** per group (when ground-truth labels are supplied) ->
  gaps for *equal opportunity* (TPR) and *equalized odds* (TPR and FPR).
- a generic ``group_metric_disparity`` for any per-group scalar metric.

You supply parallel sequences: group labels, predictions, and (optionally) true
labels. Pure standard library, offline.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence

Group = Hashable


def _validate(*seqs: Sequence[Any]) -> None:
    lengths = {len(s) for s in seqs}
    if len(lengths) > 1:
        raise ValueError("all input sequences must have equal length")


def _group_indices(groups: Sequence[Group]) -> Dict[Group, List[int]]:
    idx: Dict[Group, List[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    return dict(idx)


def selection_rate(
    groups: Sequence[Group],
    predictions: Sequence[bool],
) -> Dict[Group, float]:
    """Per-group rate of the positive ("selected"/favorable) prediction."""
    _validate(groups, predictions)
    out: Dict[Group, float] = {}
    for g, idxs in _group_indices(groups).items():
        out[g] = sum(1 for i in idxs if predictions[i]) / len(idxs)
    return out


def demographic_parity(
    groups: Sequence[Group],
    predictions: Sequence[bool],
) -> Dict[str, object]:
    """Demographic-parity difference and ratio of selection rates across groups.

    difference = max_rate − min_rate (0 = parity).
    ratio = min_rate / max_rate (1 = parity; the "80% rule" uses ratio >= 0.8).
    """
    rates = selection_rate(groups, predictions)
    return _disparity_summary(rates, "selection_rate")


def _rates_for_group(
    idxs: Sequence[int],
    predictions: Sequence[bool],
    labels: Sequence[bool],
) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for i in idxs:
        p, y = predictions[i], labels[i]
        if p and y:
            tp += 1
        elif p and not y:
            fp += 1
        elif not p and y:
            fn += 1
        else:
            tn += 1
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    acc = (tp + tn) / len(idxs) if idxs else 0.0
    return {"accuracy": acc, "tpr": tpr, "fpr": fpr}


def per_group_rates(
    groups: Sequence[Group],
    predictions: Sequence[bool],
    labels: Sequence[bool],
) -> Dict[Group, Dict[str, float]]:
    """Per-group accuracy, TPR, and FPR (requires ground-truth labels)."""
    _validate(groups, predictions, labels)
    gi = _group_indices(groups)
    return {g: _rates_for_group(idxs, predictions, labels) for g, idxs in gi.items()}


def _disparity_summary(values: Dict[Group, float], name: str) -> Dict[str, object]:
    if not values:
        return {"per_group": {}, name + "_difference": 0.0, name + "_ratio": 1.0}
    vmax = max(values.values())
    vmin = min(values.values())
    ratio = (vmin / vmax) if vmax > 0 else 1.0
    return {
        "per_group": values,
        name + "_difference": vmax - vmin,
        name + "_ratio": ratio,
    }


def group_metric_disparity(
    groups: Sequence[Group],
    values: Sequence[float],
    aggregate: Callable[[Sequence[float]], float] = None,
) -> Dict[str, object]:
    """Disparity of any per-example metric, averaged within each group.

    ``aggregate`` maps a group's values to its scalar metric (default: mean).
    Returns per-group metric, the max−min difference, and the min/max ratio.
    """
    _validate(groups, values)
    agg = aggregate or (lambda xs: sum(xs) / len(xs) if xs else 0.0)
    per_group = {
        g: agg([values[i] for i in idxs])
        for g, idxs in _group_indices(groups).items()
    }
    return _disparity_summary(per_group, "metric")


def fairness_report(
    groups: Sequence[Group],
    predictions: Sequence[bool],
    labels: Optional[Sequence[bool]] = None,
) -> Dict[str, object]:
    """Demographic parity, and (if labels given) equal-opportunity (TPR gap) and
    equalized-odds (TPR gap and FPR gap) summaries."""
    report: Dict[str, object] = {"demographic_parity": demographic_parity(groups, predictions)}
    if labels is not None:
        rates = per_group_rates(groups, predictions, labels)
        tpr = {g: r["tpr"] for g, r in rates.items()}
        fpr = {g: r["fpr"] for g, r in rates.items()}
        acc = {g: r["accuracy"] for g, r in rates.items()}
        report["per_group_rates"] = rates
        report["equal_opportunity"] = _disparity_summary(tpr, "tpr")
        report["equalized_odds"] = {
            "tpr": _disparity_summary(tpr, "tpr"),
            "fpr": _disparity_summary(fpr, "fpr"),
        }
        report["accuracy"] = _disparity_summary(acc, "accuracy")
    return report


if __name__ == "__main__":
    groups = ["A", "A", "A", "A", "B", "B", "B", "B"]
    preds = [True, True, False, False, True, False, False, False]
    labels = [True, False, True, False, True, True, False, False]
    report = fairness_report(groups, preds, labels)
    dp = report["demographic_parity"]
    print("selection rates:", dp["per_group"])
    print("demographic parity difference:", dp["selection_rate_difference"])
    print("demographic parity ratio (80% rule >= 0.8):", round(dp["selection_rate_ratio"], 3))
    print("equal opportunity (TPR) difference:", report["equal_opportunity"]["tpr_difference"])
