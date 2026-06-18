# Answer relevance

## What it measures

Does the answer actually address the question? Answer relevance is independent of
factual correctness — it penalizes evasive, off-topic, incomplete, or padded
answers regardless of whether the facts are right. Two approaches:

1. **Direct similarity** — embed the question and the answer and take their cosine
   similarity. Fast and intuitive.
2. **Reverse-question relevance** (the RAGAS pattern) — reverse-engineer `n`
   candidate questions *from the answer*, embed them, and average their cosine
   similarity to the original question. The intuition: a relevant answer lets you
   reconstruct the question. This is more discriminative than direct similarity
   because it rewards answers that *target* the question, not merely overlap with
   it lexically.

Both the **question generator** and the **embedder** are injectable. Offline
defaults (an echo generator + the hashing embedder) keep the recipe runnable; with
the echo generator, reverse-question relevance reduces to direct similarity. Inject
an LLM question generator and a real embedder for the full RAGAS behavior.

## When to use it

- **RAG / QA** — the "response relevancy" leg alongside faithfulness and context
  metrics (`rag_triad`).
- **Chat assistants** — detect non-answers, hedging, and topic drift.
- **Routing / refusal analysis** — distinguish "didn't answer" from "answered
  wrong".

## Pitfalls

- **Relevance ≠ correctness.** A confidently wrong but on-topic answer scores high.
  Always pair with faithfulness/correctness.
- **Offline default is lexical.** The hashing embedder rewards shared words; "Paris"
  vs "the French capital" looks dissimilar. Inject a real embedder for meaning.
- **Generator quality matters in the RAGAS variant.** Poor reverse questions make
  the score noisy. Use a capable LLM and several generated questions to average out
  variance.
- **Verbosity bias cuts both ways.** Long answers can dilute cosine similarity
  (lowering the score) or pad with relevant keywords (inflating direct similarity).
  Averaging multiple reverse questions mitigates this.
- **No absolute threshold.** Calibrate a cutoff on labeled relevant/irrelevant
  pairs for your embedder.

## API

- `direct_relevance(question, answer, embedder=hashing_embedder)` -> [0, 1].
- `answer_relevance(question, answer, question_generator=echo_question_generator, embedder=hashing_embedder)` -> [0, 1].
- `batch_answer_relevance(questions, answers, ...)` -> mean over pairs.

## References

- Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation* (2023).
  https://arxiv.org/abs/2309.15217
- RAGAS Response Relevancy metric documentation.
  https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/
- Reimers & Gurevych, *Sentence-BERT* (EMNLP 2019) — embedding + cosine for
  semantic similarity. https://arxiv.org/abs/1908.10084
- Vectara, *Evaluating RAG with RAGAs* (overview of the metric suite).
  https://www.vectara.com/blog/evaluating-rag
