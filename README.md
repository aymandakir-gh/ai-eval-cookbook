# ai-eval-cookbook

Provider-agnostic, runnable **offline** recipes for evaluating LLM systems:
faithfulness, hallucination rate, rubric grading, retrieval metrics, calibration,
and more.

Every recipe is:

- **Provider-agnostic** — it operates on supplied model outputs/references, or on
  an *injected* callable (a judge, embedder, runner, etc.). It never assumes a
  particular vendor or SDK.
- **Offline by default** — no network calls, no API keys. Where a model/judge/
  embedder is conceptually needed, a small deterministic default is used so the
  code runs anywhere, including CI.
- **Minimal** — pure Python standard library, no runtime dependencies.
- **Tested** — each recipe ships with a pytest suite asserting correctness on
  hand-computed cases.

These recipes are reference implementations meant to teach the metric and give you
a dependency-free starting point. For production you will typically plug in your
own judge/embedder via the documented injection points, or graduate to a
specialized library — the docs note when and why.

## Install

```bash
pip install -e ".[dev]"   # editable install with pytest
pytest                    # run the full test suite
```

Requires Python 3.9+.

## Usage

Each recipe is an importable module under `ai_eval_cookbook` with a small
`if __name__ == "__main__":` demo:

```bash
python -m ai_eval_cookbook.exact_match_accuracy
```

```python
from ai_eval_cookbook.exact_match_accuracy import accuracy
print(accuracy(["Paris"], ["paris."]))  # normalized exact match -> 1.0
```

List installed recipes:

```bash
aiec list
```

## Recipe index

| # | Recipe | Module | What it measures |
|---|--------|--------|------------------|
| 1 | Exact match accuracy | `exact_match_accuracy` | Normalized exact-match accuracy |
| 2 | Classification metrics | `classification_metrics` | Precision/recall/F1, macro/micro, confusion matrix |
| 3 | N-gram BLEU | `ngram_bleu` | N-gram precision with brevity penalty |
| 4 | ROUGE overlap | `rouge_overlap` | ROUGE-N and ROUGE-L |
| 5 | Semantic similarity | `semantic_similarity` | Jaccard + cosine, pluggable embedder |
| 6 | Faithfulness (NLI) | `faithfulness_nli` | Claim-vs-context support via injected entailment scorer |
| 7 | Hallucination rate | `hallucination_rate` | Rate of unsupported claims vs context |
| 8 | Answer relevance | `answer_relevance` | Answer-to-question relevance |
| 9 | Retrieval metrics | `retrieval_metrics` | Precision@k, recall@k, MRR, nDCG |
| 10 | RAG triad | `rag_triad` | Context precision, context recall, answer faithfulness |
| 11 | LLM-judge rubric | `llm_judge_rubric` | Rubric grading via injected judge |
| 12 | Pairwise preference Elo | `pairwise_preference_elo` | Pairwise wins to Elo / Bradley-Terry |
| 13 | Self-consistency | `self_consistency` | Sample agreement / majority vote |
| 14 | Calibration (ECE/Brier) | `calibration_ece_brier` | ECE, Brier score, reliability bins |
| 15 | JSON schema validity | `json_schema_validity` | Structured-output validity rate |
| 16 | Tool-call correctness | `tool_call_correctness` | Correct tool name + argument match |
| 17 | Refusal / safety eval | `refusal_safety_eval` | Refusal rate on a harmful-prompt set |
| 18 | Regression eval | `regression_eval` | Golden-set diff: regressions vs improvements |
| 19 | Toxicity (lexical) | `toxicity_lexical_eval` | Offline lexical safety screen (with caveats) |
| 20 | Latency / cost budget | `latency_cost_budget` | Assert latency/cost budgets over a run log |
| 21 | pass@k | `pass_at_k` | Code-eval pass@k with an injected runner |
| 22 | Group fairness disparity | `group_fairness_disparity` | Metric disparity across groups |

## License

MIT. Copyright (c) ai-eval-cookbook contributors.
