# Self-consistency

## What it measures

**Self-consistency** (Wang et al., 2022) samples several answers (typically several
chain-of-thought reasoning paths) for the *same* prompt and takes the most common
final answer instead of a single greedy decode. It reliably improves accuracy on
reasoning tasks because correct reasoning tends to converge while errors scatter.

This recipe works on the **already-sampled answers** (you do the sampling at
temperature > 0) and provides:

- **Majority vote** — the plurality answer, with an optional `normalizer` so
  "42", "forty-two", and " 42 " can be pooled.
- **Agreement** — fraction of samples matching the voted answer. A label-free
  **confidence** signal: ~1.0 means the model is sure; near `1/n` means it's split.
- **Normalized entropy** — Shannon entropy of the answer distribution in `[0, 1]`;
  high entropy flags ambiguous or hard items.
- **Distribution** and, if you pass a `reference`, whether the voted answer is
  correct.

## When to use it

- **Reasoning / math / multi-step QA** — improve accuracy via majority vote over
  sampled solutions.
- **Selective prediction** — abstain or escalate items with low agreement / high
  entropy (route them to a human or a stronger model).
- **Uncertainty estimation** without logprobs — agreement is a cheap proxy for
  confidence when you only have sampled text.
- **Flakiness detection** — low agreement on a deterministic-looking task signals
  prompt ambiguity.

## Pitfalls

- **Agreement ≠ correctness.** The model can be *consistently wrong*; all samples
  agree on a false answer. Calibrate agreement against accuracy on labeled data
  before trusting it as confidence.
- **Answer extraction matters.** Self-consistency votes over *final answers*, not
  whole responses. Extract/normalize the answer first (here via `normalizer`);
  voting over raw text fragments rarely agrees.
- **Needs diverse samples.** With temperature 0 (or too low) all samples are
  identical and the method adds nothing. Sample at temperature > 0.
- **Cost scales with samples.** Each vote is N model calls. Diminishing returns set
  in; pick N from an accuracy-vs-cost sweep.
- **Open-ended outputs don't vote well.** Free-form text rarely repeats exactly;
  self-consistency suits tasks with a small, canonicalizable answer space (numbers,
  labels, short spans). For long text, cluster semantically instead.

## API

- `majority_vote(samples, normalizer=None)` -> voted answer.
- `agreement(samples, normalizer=None)` -> fraction matching the vote.
- `normalized_entropy(samples, normalizer=None)` -> uncertainty in [0, 1].
- `vote_distribution(samples, normalizer=None)` -> `{answer: probability}`.
- `self_consistency(samples, reference=None, normalizer=None)` -> bundle.

## References

- Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language
  Models* (ICLR 2023). https://arxiv.org/abs/2203.11171
- Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language
  Models* (NeurIPS 2022). https://arxiv.org/abs/2201.11903
- Aggarwal et al., *Adaptive-Consistency: cost-efficient sampling* (EMNLP 2023).
  https://arxiv.org/abs/2305.11860
- Wikipedia, *Entropy (information theory)*.
  https://en.wikipedia.org/wiki/Entropy_(information_theory)
