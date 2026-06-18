# RAG triad (context precision, context recall, answer faithfulness)

## What it measures

A RAG pipeline fails in three distinct places; the **RAG triad** assigns one metric
to each so you can localize the problem instead of staring at a single end-to-end
score:

- **Context precision** — of the chunks the retriever returned, what fraction are
  actually relevant to the question? Low precision = the retriever pulls in noise,
  diluting and distracting the generator.
- **Context recall** — of the information needed to answer (the ground-truth
  reference), how much is covered by the retrieved chunks? Low recall = the
  retriever *misses* needed evidence and no generator can recover.
- **Answer faithfulness** (a.k.a. groundedness) — is the generated answer supported
  by the retrieved context? Low faithfulness = the generator hallucinates even when
  given good context.

This recipe composes existing building blocks (`faithfulness_nli` for the claim
checks) and exposes every judge — relevance scorer, entailment scorer, claim
splitter — as an injection point, with offline lexical defaults.

## When to use it

- **Diagnosing RAG quality.** The triad tells you *which stage* to fix:
  - bad context precision -> improve the retriever's ranking / reranker;
  - bad context recall -> improve recall (chunking, embeddings, k, query rewriting);
  - bad answer faithfulness -> improve the generator / prompt / grounding.
- **Reference-light evaluation.** Precision and faithfulness need no gold answer
  (reference-free); recall needs a reference. TruLens's original triad
  (context relevance, groundedness, answer relevance) is fully reference-free if
  you prefer that variant — swap context recall for `answer_relevance`.

## Pitfalls

- **Offline defaults are lexical.** Word-overlap relevance and entailment cannot
  see meaning; "the French capital" will not match "Paris". Inject real judges
  (an LLM or NLI model) for production numbers. The defaults exist to keep the
  recipe runnable and to teach the structure.
- **Recall depends on a good reference.** If the reference omits required facts, or
  is phrased very differently from the chunks, recall is underestimated.
- **Precision threshold is a knob.** A relevant-or-not cutoff hides partial
  relevance; for graded relevance use `retrieval_metrics.ndcg_at_k` instead.
- **The three are not independent.** Perfect recall with terrible precision still
  hurts the generator. Read all three together; do not average them into one
  number.
- **Chunk granularity matters.** Precision is computed per chunk; coarse chunks
  inflate it (a huge chunk is "relevant" if any part is). Keep chunking consistent
  across systems you compare.

## API

- `context_precision(question, retrieved_chunks, relevance_scorer=..., threshold=0.5)`.
- `context_recall(reference, retrieved_chunks, scorer=..., claim_splitter=..., threshold=0.5)`.
- `answer_faithfulness(answer, retrieved_chunks, scorer=..., claim_splitter=..., threshold=0.5)`.
- `rag_triad(question, retrieved_chunks, answer, reference, ...)` -> all three.

## References

- TruLens, *The RAG Triad* (context relevance, groundedness, answer relevance).
  https://www.trulens.org/getting_started/core_concepts/rag_triad/
- Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation* (2023)
  — context precision, context recall, faithfulness. https://arxiv.org/abs/2309.15217
- RAGAS context precision / recall metric docs.
  https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
- Snowflake Engineering, *Eval-Guided Optimization of LLM Judges for the RAG Triad*.
  https://www.snowflake.com/en/engineering-blog/eval-guided-optimization-llm-judges-rag-triad/
