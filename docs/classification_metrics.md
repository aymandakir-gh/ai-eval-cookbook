# Classification metrics

## What it measures

For tasks where the model emits a label (intent, route, moderation category,
sentiment, multiple-choice letter), the canonical metrics are computed from
counts of true positives (TP), false positives (FP), and false negatives (FN):

- **Precision** = TP / (TP + FP) — of the items predicted as a class, how many
  were right.
- **Recall** = TP / (TP + FN) — of the items truly in a class, how many we found.
- **F1** = harmonic mean of precision and recall.

Per-class scores are combined two ways:

- **Macro average**: mean of per-class F1, **each class weighted equally**.
  Sensitive to performance on rare classes.
- **Micro average**: pool TP/FP/FN across all classes, then compute once.
  Dominated by frequent classes; for single-label classification over the full
  label set, micro-precision = micro-recall = micro-F1 = accuracy.

The **confusion matrix** `M[i][j]` (true class `i`, predicted class `j`) shows
*where* errors go, which a single number cannot.

## When to use it

- Any LLM output reduced to a discrete label.
- **Imbalanced data**: report macro-F1 so a model that ignores a rare-but-critical
  class (e.g. "self-harm" in moderation) cannot hide behind accuracy.
- Debugging: read the confusion matrix to see which classes are conflated.

## Pitfalls

- **Accuracy is misleading under imbalance.** 95% "safe" content makes a
  do-nothing classifier look 95% accurate. Lead with macro-F1 and the matrix.
- **Macro vs micro is a value judgment.** Macro treats a rare class as important
  as a common one; micro does the opposite. State which you report and why.
- **Empty denominators.** A class never predicted has undefined precision; a class
  never present has undefined recall. This module defines those as 0.0 (a common
  convention) — be aware other libraries may warn or use `nan`.
- **Multi-label / multi-class confusion.** This module assumes single-label
  classification. For multi-label, compute per-label one-vs-rest separately.
- **Threshold dependence.** If labels come from thresholding a score, the metrics
  describe only that threshold; sweep it (precision-recall curve) for the full
  picture.

## API

- `confusion_matrix(y_true, y_pred, labels=None)` -> `(labels, matrix)`.
- `per_class_metrics(...)` -> `{label: {precision, recall, f1, support}}`.
- `macro_average(...)`, `micro_average(...)` -> `{precision, recall, f1}`.
- `accuracy(y_true, y_pred)` -> float.
- `classification_report(...)` -> bundle of all of the above.

## References

- Rajpurkar et al., *SQuAD* (2016) — token-level F1 in NLP evaluation.
  https://arxiv.org/abs/1606.05250
- scikit-learn, *Precision, recall and F-measures* (macro vs micro averaging).
  https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
- scikit-learn, `classification_report` reference.
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
- Sasaki, *The truth of the F-measure* (2007).
  https://www.cs.odu.edu/~mukka/cs795sum10dm/Lecturenotes/Day3/F-measure-YS-26Oct07.pdf
