# Faithfulness (claim-vs-context entailment)

## What it measures

**Faithfulness** (a.k.a. groundedness) asks: is every claim in the model's answer
*supported by* the provided context? It is the core anti-hallucination metric for
RAG and summarization. Following RAGAS and FActScore, it is a two-step pipeline:

1. **Decompose** the answer into atomic claims (here: sentence-level by default;
   inject your own `claim_splitter`, e.g. an LLM claim extractor).
2. **Verify** each claim against the context with an **entailment scorer** — an
   injected `Callable[[claim, context], float]` returning a support score in
   `[0, 1]`. This is the conceptual NLI/judge component.

The score is the fraction of claims whose support meets a threshold: `1.0` = fully
grounded, `0.0` = every claim unsupported.

To run offline the default scorer is a transparent **lexical entailment** heuristic
(fraction of the claim's content words present in the context). Swap in a real NLI
model (e.g. a DeBERTa entailment classifier) or an LLM judge via `scorer=...`.

## When to use it

- **RAG answer evaluation** — does the answer stick to retrieved evidence?
- **Summarization consistency** — is the summary entailed by the source document
  (the SummaC setting)?
- **Grounded generation / citations** — verifying that asserted facts trace back
  to provided material.
- As one leg of the RAG triad (`rag_triad`) and the basis for
  `hallucination_rate` (1 − faithfulness at the claim level).

## Pitfalls

- **Decomposition quality dominates.** If claim splitting misses or over-splits
  claims, every downstream verdict drifts. Sentence splitting is a coarse default;
  an LLM atomic-claim extractor is materially better for compound sentences.
- **The lexical default is weak.** Token overlap cannot detect contradiction
  ("completed in 1889" vs "completed in 1989" share most words) and misses
  paraphrase ("the capital of France" vs "Paris"). It is a stand-in to keep the
  recipe runnable; inject a real entailment model for trustworthy numbers.
- **Faithfulness ≠ correctness.** An answer can be perfectly faithful to a *wrong*
  context. Faithfulness measures grounding, not truth; pair with reference-based
  correctness when ground truth exists.
- **Threshold sensitivity.** The supported/unsupported cutoff trades precision for
  recall of hallucinations. Calibrate it on labeled examples for your scorer.
- **Abstention and hedging.** "I don't know" has no factual claims and scores
  vacuously faithful (1.0). Decide whether that is desirable for your task.

## API

- `faithfulness(answer, context, scorer=lexical_entailment, claim_splitter=sentence_claims, threshold=0.5)` -> float.
- `claim_verdicts(...)` -> per-claim `{claim, support, supported}` for inspection.
- `sentence_claims(answer)`, `lexical_entailment(claim, context)` defaults.

## References

- Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation* (2023).
  https://arxiv.org/abs/2309.15217
- RAGAS Faithfulness metric documentation.
  https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/faithfulness/
- Min et al., *FActScore: Fine-grained Atomic Evaluation of Factual Precision*
  (EMNLP 2023). https://arxiv.org/abs/2305.14251
- Laban et al., *SummaC: Re-Visiting NLI-based Models for Inconsistency Detection*
  (TACL 2022). https://arxiv.org/abs/2111.09525
