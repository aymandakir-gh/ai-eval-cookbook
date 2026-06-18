# LLM-judge rubric grading

## What it measures

For open-ended outputs with no single correct answer (essays, chat replies,
summaries, code explanations), **LLM-as-a-judge** scores a response against a
**rubric** — a set of named criteria (accuracy, clarity, helpfulness, ...) each with
a score range and weight. This recipe is the *harness*, not the judge: it owns the
parts you can test and reproduce.

- Defines criteria (`Criterion`: name, description, min/max, weight).
- Calls an injected `judge(prompt, response, criterion) -> score` (your LLM call).
- Clamps each score to its range, averages **repeated trials** to damp judge
  variance, and aggregates a weighted **overall** plus a `[0, 1]` normalized score.
- Grades whole datasets and reports means per criterion.

A **deterministic mock judge** (`keyword_mock_judge`) ships for offline tests, so
the harness is fully exercisable without any model — exactly what you want in CI.

## When to use it

- **Open-ended quality** where references don't exist or are underspecified.
- **Multi-dimensional grading** (separate accuracy / style / safety scores) rather
  than one opaque number.
- **Scalable human-eval replacement** for ranking prompts, models, or prompt
  versions, with humans auditing a sample.

## Pitfalls

- **Judges are biased.** Documented effects: *position bias* (order of options),
  *verbosity bias* (longer = "better"), *self-preference* (a model favors its own
  style), and *overlap bias* in summary grading. A rubric instruction cannot fully
  remove structural biases — measure them (swap order, vary length) and prefer a
  judge different from the model under test.
- **Calibrate against humans.** A judge score is only meaningful if it correlates
  with human ratings on your task. Validate on a labeled subset before trusting it.
- **Variance is real.** Single-call scores are noisy; average several trials
  (`trials>1`) and/or use chain-of-thought prompting (G-Eval) for stability.
- **Rubric design dominates.** Vague criteria yield vague scores. Give each
  criterion an explicit description and anchored score levels in your judge prompt.
- **Don't over-aggregate.** A single weighted number can hide a failing safety
  criterion behind strong style. Inspect per-criterion scores.

## API

- `Criterion(name, description="", min_score=1, max_score=5, weight=1.0)`.
- `grade(prompt, response, rubric, judge, trials=1)` -> per-criterion + overall +
  overall_normalized.
- `grade_dataset(samples, rubric, judge, trials=1)` -> per-sample results + means.
- `keyword_mock_judge(positive=None, negative=None)` -> deterministic offline judge.

## References

- Liu et al., *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*
  (EMNLP 2023). https://arxiv.org/abs/2303.16634
- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*
  (NeurIPS 2023) — biases, position bias. https://arxiv.org/abs/2306.05685
- Gu et al., *A Survey on LLM-as-a-Judge* (2024). https://arxiv.org/abs/2411.15594
- Confident AI, *LLM-as-a-Judge: The Complete Guide*.
  https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method
