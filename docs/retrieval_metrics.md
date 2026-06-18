# Retrieval metrics (precision@k, recall@k, MRR, nDCG)

## What it measures

Quality of a **ranked list of retrieved items** against known-relevant items — the
core evaluation for RAG retrievers and search.

- **precision@k** = (relevant items in top-k) / k. Are the top results good?
- **recall@k** = (relevant items in top-k) / (all relevant). Did we find what
  exists?
- **reciprocal rank (RR)** = 1 / (rank of first relevant item). **MRR** averages
  RR across queries — sensitive only to where the *first* hit lands, ideal for
  "find the one right document" tasks.
- **nDCG@k** — rank-aware and **graded**: each item contributes a gain
  `2^rel − 1` discounted by `log2(rank + 1)`, normalized by the ideal ordering's
  DCG so the score is in `[0, 1]`. Works with a binary relevant set or a
  `{item: grade}` map.

precision/recall ignore order within the top-k; MRR and nDCG are rank-aware.

## When to use it

- **RAG retriever evaluation** — does the retriever surface the passages that
  contain the answer? precision@k / recall@k for coverage; nDCG/MRR for ranking.
- **Search & recommendation** ranking quality.
- Choosing **k**: tie it to how many chunks you feed the generator. recall@k tells
  you whether the answer-bearing context is even reachable.

## Pitfalls

- **precision@k denominator is k, not the number returned.** Returning fewer than
  k relevant-but-short lists is penalized — by design. Be explicit about this when
  comparing systems.
- **recall@k needs a complete relevance set.** If you only labeled a few documents,
  recall is overestimated. Pooled/qrels judgments matter.
- **MRR only sees the first hit.** A system that nails one result but buries the
  rest looks great by MRR. Combine with recall@k.
- **nDCG hides behind normalization.** A query with one weakly relevant document
  can hit nDCG 1.0 trivially. Report it with absolute coverage (recall@k).
- **Graded gain choice is a convention.** The `2^rel − 1` form rewards highly
  relevant items steeply; a linear gain (`rel`) is also common. Keep it consistent.
- **Ties and duplicates.** Duplicate ids or tied scores make rankings ambiguous;
  dedupe and break ties deterministically before scoring.

## API

- `precision_at_k(ranked, relevant, k)`, `recall_at_k(ranked, relevant, k)`.
- `reciprocal_rank(ranked, relevant)`, `mean_reciprocal_rank(rankings, relevants)`.
- `dcg_at_k(ranked, relevance, k)`, `ndcg_at_k(ranked, relevance, k)`
  — `relevance` may be a set (binary) or a `{item: grade}` map (graded).
- `retrieval_report(ranked, relevant, k)` -> bundle.

## References

- Järvelin & Kekäläinen, *Cumulated gain-based evaluation of IR techniques*
  (ACM TOIS 2002) — DCG / nDCG. https://doi.org/10.1145/582415.582418
- Manning, Raghavan & Schütze, *Introduction to Information Retrieval* (2008),
  Ch. 8 Evaluation. https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html
- Wikipedia, *Discounted cumulative gain*.
  https://en.wikipedia.org/wiki/Discounted_cumulative_gain
- Wikipedia, *Mean reciprocal rank*.
  https://en.wikipedia.org/wiki/Mean_reciprocal_rank
