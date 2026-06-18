# Regression evaluation (golden-set diff)

## What it measures

A single aggregate score can stay flat while individual examples silently flip from
correct to wrong. Regression evaluation diffs two runs over the same **golden set**
and classifies each example:

- **regression** — the candidate is worse than the baseline (beyond a tolerance),
- **improvement** — the candidate is better (beyond the tolerance),
- **unchanged** — within the tolerance dead-band.

It reports the counts, the **net change** (improvements − regressions), the mean
score delta, and the *ids* of the regressed examples for triage. This is the metric
you gate releases on, because "accuracy went from 84% to 84%" can hide a 5%
regression masked by a 5% improvement elsewhere.

Works with boolean correctness or any numeric score (higher = better).

## When to use it

- **CI gating on prompt / model / pipeline changes** — block a merge if any
  example regresses (`has_regressions`).
- **Model upgrades** — a newer model with the same average can still regress your
  most important cases; the regression id list shows which.
- **A/B of prompt variants** — see the churn, not just the mean.
- **Tolerance for noisy judges** — set `tol` so sub-threshold score wiggle from a
  stochastic grader doesn't register as change.

## Pitfalls

- **Golden set must be stable and representative.** A diff is only as trustworthy
  as the examples in it; stale or biased golden sets give false confidence. Curate
  and version it.
- **Per-example scores must be comparable.** If the grader itself is nondeterministic
  (an LLM judge), re-score both runs with the *same* grader version and use a
  tolerance; otherwise judge noise looks like regressions.
- **"Higher is better" assumption.** For metrics where lower is better (latency,
  loss), negate the scores before diffing.
- **Net change can deceive.** Net zero with high churn means the system is unstable,
  not stable. Look at the gross counts and the regression ids, not just net.
- **Threshold choice is a policy.** A wide tolerance hides small but real
  regressions; a zero tolerance flags noise. Pick it deliberately and document it.

## API

- `classify(baseline, candidate, tol=0.0)` -> 'regression' | 'improvement' |
  'unchanged'.
- `diff_runs(baseline_scores, candidate_scores, ids=None, tol=0.0)` -> counts,
  net change, means, deltas, and regressed/improved ids.
- `has_regressions(baseline_scores, candidate_scores, tol=0.0, max_allowed=0)` -> bool
  (CI gate).

## References

- Google, *Testing on the Toilet: Avoiding Flaky Tests* / regression testing
  practice. https://testing.googleblog.com/2008/04/tott-avoiding-flakey-tests.html
- Microsoft, *Responsible AI: Test sets and regression testing for ML systems*.
  https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-technical-paper
- OpenAI Evals — golden-dataset regression testing for LLM apps.
  https://github.com/openai/evals
- Hutchinson et al., *Towards Accountability for Machine Learning Datasets* (2021).
  https://arxiv.org/abs/2010.13561
