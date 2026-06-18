# Hallucination rate

## What it measures

The **hallucination rate** is the fraction of an answer's claims that are *not*
supported by the provided reference/context. It is the complement of claim-level
faithfulness (`hallucination_rate = 1 − faithfulness`) but framed as a risk metric
and aggregated across a whole evaluation set, which is how teams report and track
RAG/summarization safety over time.

Same decompose-then-verify pipeline as `faithfulness_nli`:

1. split the answer into atomic claims,
2. score each claim's support against the context with an injectable entailment
   scorer (offline lexical default),
3. count claims below the support threshold as hallucinations.

Two corpus aggregates are provided:

- **micro** — pool all claims, then `unsupported / total`. Each *claim* counts
  equally; robust to varying answer lengths. This is the usual headline number.
- **macro** — average each answer's own rate. Each *answer* counts equally.

## When to use it

- **RAG / grounded QA monitoring** — a single dashboard number for "how often does
  the system make things up?" tracked per release.
- **Summarization** — rate of summary statements not entailed by the source.
- **Regression gating** — fail CI if hallucination rate rises above a budget.
- Drill-down: the report lists the actual unsupported claims per example for triage.

## Pitfalls

- **Inherits every faithfulness caveat.** Decomposition quality and scorer quality
  dominate the number; the offline lexical default cannot see contradiction or
  paraphrase. Inject a real NLI model / LLM judge for trustworthy rates. See
  `faithfulness_nli` docs.
- **Micro vs macro can disagree.** A few long, mostly-hallucinated answers move
  micro more than macro. Report which you use; ideally track both.
- **Not the same as factual error rate.** A claim can be unsupported by *this*
  context yet true in the world (and vice versa). This measures grounding to the
  given context, not absolute truth.
- **Denominator games.** Terser answers make fewer claims and can show a lower rate
  without being better. Pair with a coverage/recall metric (`rag_triad` context
  recall, `answer_relevance`).
- **Threshold and splitter are policy.** Tune the support threshold and use an LLM
  atomic-claim extractor for compound sentences when precision matters.

## API

- `hallucination_rate(answers, contexts, scorer=..., claim_splitter=..., threshold=0.5, aggregate="micro"|"macro")` -> float.
- `hallucination_rate_single(answer, context, ...)` -> per-answer rate.
- `unsupported_claims(answer, context, ...)` -> list of offending claims.
- `hallucination_report(answers, contexts, ...)` -> micro + macro + per-example.

## References

- Min et al., *FActScore: Fine-grained Atomic Evaluation of Factual Precision in
  Long Form Text Generation* (EMNLP 2023). https://arxiv.org/abs/2305.14251
- Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation* (2023).
  https://arxiv.org/abs/2309.15217
- Ji et al., *Survey of Hallucination in Natural Language Generation* (ACM CSUR 2023).
  https://arxiv.org/abs/2202.03629
- Edinburgh NLP, *Awesome Hallucination Detection* (curated reading list).
  https://github.com/EdinburghNLP/awesome-hallucination-detection
