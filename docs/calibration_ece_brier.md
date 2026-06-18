# Calibration (ECE, Brier, reliability bins)

## What it measures

A model is **calibrated** when its confidence matches its accuracy: among the
answers it gives with 80% confidence, ~80% should be correct. Calibration is
distinct from accuracy — a model can be accurate but overconfident, which breaks
any downstream use of confidence for abstention, routing, or risk control. This
matters for LLMs that emit verbalized confidences or token probabilities.

Given paired `(confidence, correct)` records:

- **Reliability bins** — partition confidence into bins; each bin reports mean
  confidence, empirical accuracy, and the gap. This is the data behind a
  reliability diagram (the diagonal = perfect calibration).
- **Expected Calibration Error (ECE)** — sample-weighted average gap across bins.
  `0` = calibrated; larger = worse.
- **Maximum Calibration Error (MCE)** — the worst single bin gap; a tail-risk view.
- **Brier score** — mean squared error between confidence and outcome. A *proper
  scoring rule*: it rewards being both calibrated *and* sharp (confident when
  right), so a model that always predicts 0.5 gets a mediocre Brier even if its ECE
  looks fine.

## When to use it

- **Selective prediction / abstention** — only act when calibrated confidence is
  high; calibration is what makes a threshold meaningful.
- **Routing & cascades** — send low-confidence items to a stronger model or human.
- **Comparing models / decoding settings** for trustworthiness, not just accuracy.
- **Post-hoc calibration tuning** (temperature scaling) — ECE/Brier are the target
  metrics.

## Pitfalls

- **ECE depends on binning.** Bin count and equal-width vs equal-mass binning
  change the number; ECE can also be gamed and is biased with few samples. Report
  the bin count, and prefer Brier (binning-free) alongside it.
- **ECE ignores sharpness.** A model that outputs 0.5 for everything on a balanced
  task has near-zero ECE but is useless. Brier and a reliability diagram expose
  this; never report ECE alone.
- **Needs enough data per bin.** Sparse bins give noisy accuracy estimates and
  unstable gaps. Aggregate or widen bins when data is thin.
- **Confidence must be a probability.** These metrics assume confidences in [0, 1]
  aligned with P(correct). Raw logits or verbalized "9/10" must be mapped first.
- **Class imbalance.** On skewed tasks, a constant high confidence can look
  calibrated overall while being miscalibrated per class. Slice by class.

## API

- `reliability_bins(confidences, correct, n_bins=10)` -> per-bin stats.
- `expected_calibration_error(confidences, correct, n_bins=10)` -> ECE.
- `maximum_calibration_error(confidences, correct, n_bins=10)` -> MCE.
- `brier_score(confidences, correct)` -> Brier.
- `calibration_report(...)` -> ECE + MCE + Brier + accuracy + mean confidence + bins.

## References

- Guo et al., *On Calibration of Modern Neural Networks* (ICML 2017).
  https://arxiv.org/abs/1706.04599
- Naeini et al., *Obtaining Well Calibrated Probabilities Using Bayesian Binning*
  (AAAI 2015). https://ojs.aaai.org/index.php/AAAI/article/view/9602
- Brier, *Verification of forecasts expressed in terms of probability* (1950).
  https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2
- Nixon et al., *Measuring Calibration in Deep Learning* (CVPR Workshops 2019).
  https://arxiv.org/abs/1904.01685
