"""pass@k for code generation, with the HumanEval unbiased estimator and an
injected test runner.

pass@k estimates the probability that at least one of k sampled solutions passes the
tests. Computing it as "did any of the first k pass?" is high-variance; HumanEval
(Chen et al., 2021) uses an **unbiased estimator**: generate n >= k samples, count
c that pass, and compute

    pass@k = 1 - C(n - c, k) / C(n, k)

i.e. one minus the probability that a random k-subset of the n samples contains no
passing solution. This module provides the numerically stable estimator plus helpers
that take a **list of solution sources per problem** and an **injected runner**
``runner(source) -> bool`` that decides whether a solution passes. A deterministic
**mock runner** is provided so the recipe runs offline with no code execution.

WARNING: running model-generated code is dangerous. A real runner MUST sandbox
untrusted code (containers, seccomp, no network, resource limits). The mock runner
here executes nothing.

Pure standard library, offline.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

# A runner maps a candidate solution source -> True if it passes the tests.
Runner = Callable[[str], bool]


def pass_at_k_estimator(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimate from n samples, c of which pass, for parameter k.

    Returns a probability in [0, 1]. If ``n - c < k`` (fewer than k failing
    samples) at least one of any k must pass, so the estimate is 1.0.
    """
    if n < 0 or c < 0 or k <= 0:
        raise ValueError("require n >= 0, c >= 0, k > 0")
    if c > n:
        raise ValueError("c must not exceed n")
    if k > n:
        raise ValueError("k must not exceed n")
    if n - c < k:
        return 1.0
    # Numerically stable product form: 1 - prod_{i=n-c+1}^{n} (1 - k/i)
    prob_all_fail = 1.0
    for i in range(n - c + 1, n + 1):
        prob_all_fail *= 1.0 - k / i
    return 1.0 - prob_all_fail


def count_passing(samples: Sequence[str], runner: Runner) -> int:
    """Number of samples for which ``runner`` returns True."""
    return sum(1 for s in samples if runner(s))


def problem_pass_at_k(
    samples: Sequence[str], runner: Runner, k: int
) -> float:
    """pass@k for one problem given its sampled solutions and a runner."""
    n = len(samples)
    c = count_passing(samples, runner)
    return pass_at_k_estimator(n, c, k)


def pass_at_k(
    problems: Sequence[Sequence[str]], runner: Runner, k: int
) -> float:
    """Mean pass@k across problems (each problem is a list of sampled solutions)."""
    if not problems:
        return 0.0
    return sum(problem_pass_at_k(p, runner, k) for p in problems) / len(problems)


def pass_at_k_report(
    problems: Sequence[Sequence[str]], runner: Runner, ks: Sequence[int]
) -> Dict[str, object]:
    """pass@k for several k values plus per-problem (n, c) counts."""
    counts = [(len(p), count_passing(p, runner)) for p in problems]
    result: Dict[str, object] = {"per_problem": counts}
    for k in ks:
        scores = []
        for (n, c) in counts:
            if k > n:
                continue
            scores.append(pass_at_k_estimator(n, c, k))
        result["pass@%d" % k] = (sum(scores) / len(scores)) if scores else 0.0
    return result


def keyword_mock_runner(must_contain: str = "return") -> Runner:
    """Deterministic offline 'runner': a solution 'passes' iff it contains the
    given substring. Executes NO code — purely for tests and demos.
    """

    def run(source: str) -> bool:
        return must_contain in source

    return run


if __name__ == "__main__":
    # 5 sampled solutions for one problem; the mock runner passes any with "return".
    problem = [
        "def f(x): return x + 1",  # pass
        "def f(x): pass",  # fail
        "def f(x): return x * 2",  # pass
        "def f(x): print(x)",  # fail
        "def f(x): pass",  # fail
    ]
    runner = keyword_mock_runner("return")
    print("n, c:", len(problem), count_passing(problem, runner))
    for k in (1, 2, 5):
        print("pass@%d:" % k, round(problem_pass_at_k(problem, runner, k), 4))
    print("report:", pass_at_k_report([problem], runner, ks=[1, 2, 5]))
