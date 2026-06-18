# Group fairness disparity

## What it measures

Aggregate quality can hide unequal treatment across subgroups defined by a sensitive
attribute. This recipe computes a metric **per group** and summarizes the
**disparity** between the best- and worst-served groups:

- **Selection rate** per group -> **demographic parity**:
  - *difference* = max rate − min rate (0 = parity),
  - *ratio* = min rate / max rate (1 = parity; the "80% / four-fifths rule" flags
    ratio < 0.8).
- **TPR / FPR / accuracy** per group (when ground-truth labels are supplied):
  - **equal opportunity** = TPR gap across groups (equal true-positive rates),
  - **equalized odds** = TPR gap *and* FPR gap (equal TPR and FPR),
  - accuracy gap.
- `group_metric_disparity` — disparity of *any* per-example metric averaged within
  groups (e.g. a quality score, a faithfulness score) for arbitrary LLM evals.

## When to use it

- **Auditing LLM/classifier outputs** (moderation, eligibility, ranking,
  generation quality) for disparate impact across demographic or other groups.
- **Slice-based evaluation** — turn any of this cookbook's metrics into a fairness
  check by passing per-example scores to `group_metric_disparity`.
- **Compliance / responsible-AI reporting** — demographic parity and the
  four-fifths rule are widely referenced.

## Pitfalls

- **Fairness metrics conflict by design.** Impossibility results show you generally
  cannot satisfy demographic parity, equal opportunity, and equalized odds
  simultaneously when base rates differ across groups. Choose the criterion that
  matches your harm model, and state it; the demo shows a model with equal TPR but
  unequal FPR (passes equal opportunity, fails equalized odds).
- **Difference vs ratio disagree.** Small absolute rates make the ratio swing wildly
  (1% vs 2% is ratio 0.5 but difference 0.01). Report both and interpret in context.
- **Disparity is not causation.** A gap may reflect genuine base-rate differences,
  label bias, or sampling, not model unfairness. Investigate before concluding.
- **Small groups are noisy.** Per-group rates from few examples are unstable; report
  group sizes and confidence intervals, and beware acting on tiny slices.
- **Sensitive attributes are sensitive.** Collecting/handling group labels has
  legal and ethical constraints; intersectional groups (combinations) often matter
  more than single attributes.
- **Choice of "favorable" outcome.** Which prediction counts as positive/favorable
  is a value judgment that flips the interpretation of parity. Define it explicitly.

## API

- `selection_rate(groups, predictions)` -> per-group positive rate.
- `demographic_parity(groups, predictions)` -> difference + ratio.
- `per_group_rates(groups, predictions, labels)` -> per-group accuracy/TPR/FPR.
- `group_metric_disparity(groups, values, aggregate=mean)` -> generic disparity.
- `fairness_report(groups, predictions, labels=None)` -> demographic parity, equal
  opportunity, equalized odds, accuracy gaps.

## References

- Hardt et al., *Equality of Opportunity in Supervised Learning* (NeurIPS 2016).
  https://arxiv.org/abs/1610.02413
- Barocas, Hardt & Narayanan, *Fairness and Machine Learning* (book).
  https://fairmlbook.org/
- Microsoft Fairlearn, *Common fairness metrics* (demographic parity, equalized
  odds). https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html
- Chouldechova, *Fair prediction with disparate impact* (impossibility) (2017).
  https://arxiv.org/abs/1703.00056
