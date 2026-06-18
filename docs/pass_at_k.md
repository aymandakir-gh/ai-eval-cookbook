# pass@k (code evaluation)

## What it measures

For code generation, **pass@k** estimates the probability that **at least one of k
sampled solutions** passes the unit tests. Reporting "did any of the first k pass?"
is high-variance, so HumanEval (Chen et al., 2021) introduced an **unbiased
estimator**: generate `n >= k` samples per problem, count `c` that pass, and compute

```
pass@k = 1 - C(n - c, k) / C(n, k)
```

— one minus the chance that a random k-subset of the n samples contains no passing
solution. Larger `n` gives a more accurate estimate. This module ships the
numerically stable estimator, helpers that take **per-problem sampled solutions** and
an **injected runner** `runner(source) -> bool`, and a deterministic **mock runner**
so the recipe runs offline without executing any code.

## When to use it

- **Code-generation benchmarks** (HumanEval, MBPP, MultiPL-E) and your own coding
  evals.
- **Sampling-budget decisions** — pass@1 vs pass@10 vs pass@100 quantifies the
  benefit of drawing more samples (useful when you can verify/select afterward).
- **Any "generate-and-verify" task** with a pass/fail checker, not only code (e.g.
  outputs validated by a constraint or a test oracle).

## Pitfalls

- **You must execute untrusted code — safely.** A real runner runs model-generated
  code; it MUST be sandboxed (containers, seccomp, no network, CPU/memory/time
  limits). The mock runner here executes nothing; never run raw model code in your
  main process.
- **Tests are the ground truth.** pass@k is only as good as the unit tests. Weak
  tests overstate ability; flaky/nondeterministic tests corrupt `c`. Pin
  environments and seeds.
- **n must be >= k.** With `n < k` the estimator is undefined; the report skips
  those problems. For high k (e.g. 100) you need many samples — that is the cost.
- **pass@k rewards luck at high k.** A model that needs 100 tries to get one pass
  isn't production-ready; pass@1 is the deployment-relevant number unless you have a
  reliable selector (re-ranker, tests) to pick the winning sample.
- **Contamination.** If benchmark problems leaked into training, pass@k is
  inflated; use held-out or freshly authored problems.
- **Per-problem then average.** pass@k is computed per problem and averaged; don't
  pool c and n across problems.

## API

- `pass_at_k_estimator(n, c, k)` -> unbiased estimate for one problem.
- `count_passing(samples, runner)` -> number of passing samples.
- `problem_pass_at_k(samples, runner, k)` -> pass@k for one problem.
- `pass_at_k(problems, runner, k)` -> mean pass@k across problems.
- `pass_at_k_report(problems, runner, ks)` -> pass@k for several k + per-problem
  (n, c).
- `keyword_mock_runner(must_contain="return")` -> deterministic offline runner.

## References

- Chen et al., *Evaluating Large Language Models Trained on Code* (Codex / HumanEval)
  (2021) — defines the unbiased pass@k estimator. https://arxiv.org/abs/2107.03374
- Kulal et al., *SPoC: Search-based Pseudocode to Code* (NeurIPS 2019) — pass@k
  origins. https://arxiv.org/abs/1906.04908
- Austin et al., *Program Synthesis with Large Language Models* (MBPP) (2021).
  https://arxiv.org/abs/2108.07732
- OpenAI `human-eval` reference implementation (`estimate_pass_at_k`).
  https://github.com/openai/human-eval
