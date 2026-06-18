# Semantic similarity

## What it measures

How close two texts are in meaning (or, more honestly, in surface/representation).
This recipe offers two complementary, offline families:

- **Jaccard overlap** — set similarity `|A∩B| / |A∪B|`.
  - *Token Jaccard*: over the set of word tokens. Fast, interpretable, but blind
    to word order and synonyms.
  - *Character-n-gram Jaccard*: over character trigrams. Robust to typos,
    inflection, and minor edits ("organize"/"organise").
- **Embedding cosine** — embed each text into a vector and take the cosine of the
  angle between them. This is the standard production approach. The module ships a
  **deterministic, offline hashing embedder** (L2-normalized hashed bag of token
  n-grams) so the recipe runs with no model or network, and exposes a one-argument
  **pluggable embedder interface** (`Callable[[str], Sequence[float]]`) so you can
  drop in a real sentence-transformer for production.

A `semantic_similarity` helper blends token-Jaccard and embedding cosine into one
`[0, 1]` score with configurable weights.

## When to use it

- **Answer / summary equivalence** when exact and n-gram overlap (EM, BLEU, ROUGE)
  are too strict and paraphrase should count.
- **Deduplication, clustering, near-duplicate detection** of model outputs or
  retrieved documents.
- **Retrieval relevance** scoring and reranking (with a real embedder).
- As a soft component inside RAG metrics (`answer_relevance`, `rag_triad`).

## Pitfalls

- **The offline default is lexical, not neural.** The hashing embedder captures
  shared words/n-grams, not deep meaning — "good" vs "excellent" will look
  dissimilar. Treat it as a strong baseline and a stand-in; inject a real embedder
  for genuine semantic judgments. The pluggable interface exists precisely for
  this.
- **Cosine has no absolute threshold.** "0.7" means nothing in isolation;
  calibrate a cutoff on labeled pairs for your embedder and domain.
- **Embeddings can be fooled.** High cosine does not guarantee factual agreement
  ("Paris is the capital of France" vs "Paris is not the capital of France" embed
  similarly). For factuality use `faithfulness_nli` / `hallucination_rate`.
- **Jaccard ignores frequency and order.** "dog bites man" and "man bites dog"
  have token-Jaccard 1.0. Use n-gram or embedding measures when order matters.
- **Asymmetric tasks.** Similarity is symmetric; "does the answer entail the
  reference?" is not. Use an entailment/NLI scorer for directional claims.

## API

- `token_jaccard(a, b)`, `char_ngram_jaccard(a, b, n=3)` -> float in [0, 1].
- `cosine(u, v)` -> cosine of two vectors.
- `hashing_embedder(text, dim=256, ngram=1)` -> deterministic offline vector.
- `embedding_cosine_similarity(a, b, embedder=hashing_embedder)` -> float.
- `semantic_similarity(a, b, embedder=..., weights=(0.5, 0.5))` -> blended float.

## References

- Reimers & Gurevych, *Sentence-BERT* (EMNLP 2019) — sentence embeddings + cosine.
  https://arxiv.org/abs/1908.10084
- Zhang et al., *BERTScore: Evaluating Text Generation with BERT* (ICLR 2020).
  https://arxiv.org/abs/1904.09675
- Weinberger et al., *Feature Hashing for Large Scale Multitask Learning* (2009).
  https://arxiv.org/abs/0902.2206
- Wikipedia, *Jaccard index*. https://en.wikipedia.org/wiki/Jaccard_index
- Wikipedia, *Cosine similarity*. https://en.wikipedia.org/wiki/Cosine_similarity
