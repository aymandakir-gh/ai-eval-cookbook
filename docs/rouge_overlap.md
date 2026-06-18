# ROUGE overlap (ROUGE-N and ROUGE-L)

## What it measures

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures content overlap
between a candidate and a reference. It was built for summarization, where recall
("did we cover the reference content?") matters as much as precision.

- **ROUGE-N**: overlap of n-grams, clipped by reference counts.
  - recall = matched n-grams / reference n-grams
  - precision = matched n-grams / candidate n-grams
  - F1 = their harmonic mean
  ROUGE-1 (unigrams) and ROUGE-2 (bigrams) are the common defaults.
- **ROUGE-L**: uses the **longest common subsequence** (LCS) — the longest
  in-order (gaps allowed) shared word sequence. It captures sentence-level
  structure without committing to a fixed n, and naturally rewards correct
  ordering even when words are inserted between matches.

## When to use it

- **Summarization** and other long-form generation with reference texts — the
  canonical use, where ROUGE-1/2/L are still the standard reported triple.
- **Headline / title generation, paraphrase, simplification** — any task with a
  reference where content coverage matters more than exact phrasing.
- As a fast, reproducible complement to BLEU: ROUGE leans recall (coverage), BLEU
  leans precision (fluency/adequacy).

## Pitfalls

- **Surface overlap, not meaning.** ROUGE cannot credit synonyms or paraphrase;
  an abstractive summary that rewords the reference is undervalued. Pair with a
  semantic metric (`semantic_similarity`) or a judge for abstractive systems.
- **Report precision *and* recall.** Recall alone rewards verbosity (copy the
  whole document and recall is perfect). F1 or precision guards against that.
- **Single reference underestimates quality.** Good summaries vary; multiple
  references (take the max or average) correlate better with humans.
- **Stemming / stopwords change the number.** The original ROUGE optionally stems
  and removes stopwords. This module does neither by default for transparency;
  results are only comparable under the same preprocessing.
- **ROUGE-L variants.** This module implements sentence-level LCS ROUGE-L. The
  summary-level variant (ROUGE-Lsum, union-LCS over sentences) differs; note which
  you report.

## API

- `rouge_n(candidate, reference, n=1, tokenizer=...)` -> `{precision, recall, f1}`.
- `rouge_l(candidate, reference, tokenizer=...)` -> `{precision, recall, f1}`.
- `rouge_scores(candidate, reference, ns=(1, 2), tokenizer=...)` -> bundle with
  `rouge1`, `rouge2`, ..., `rougeL`.

## References

- Lin, *ROUGE: A Package for Automatic Evaluation of Summaries* (ACL Workshop 2004).
  https://aclanthology.org/W04-1013/
- Google Research `rouge` implementation (ROUGE-N, ROUGE-L, ROUGE-Lsum).
  https://github.com/google-research/google-research/tree/master/rouge
- Hugging Face `evaluate` ROUGE metric documentation.
  https://github.com/huggingface/evaluate/blob/main/metrics/rouge/README.md
- Wikipedia, *ROUGE (metric)*.
  https://en.wikipedia.org/wiki/ROUGE_(metric)
