# Exact-match accuracy

## What it measures

Exact match (EM) is the strictest correctness signal: a prediction counts as
correct only if it equals the reference answer. The aggregate **accuracy** is the
fraction of examples that match. Because raw string equality is too brittle for
natural-language tasks ("Paris" vs "paris." vs "the Paris"), the standard practice
— popularized by the SQuAD evaluation script — is to **normalize** both prediction
and reference first:

1. Unicode-normalize (NFKC),
2. lowercase,
3. remove articles (`a`, `an`, `the`),
4. strip punctuation,
5. collapse whitespace.

This module also supports **multi-reference** scoring: if several reference strings
are acceptable, a prediction is correct when it matches any of them.

## When to use it

- **Short-answer / extractive QA** where there is a single canonical answer
  (SQuAD-style spans, trivia, entity lookups).
- **Closed-form outputs**: yes/no, multiple-choice letters, numeric IDs, slot
  values, classification labels expressed as text.
- As a fast, interpretable, zero-cost first-pass metric before reaching for
  fuzzier overlap or model-graded metrics.

EM pairs naturally with token-level **F1** (see `classification_metrics` for the
classification analogue) when partial credit matters.

## Pitfalls

- **Too strict for open-ended generation.** Summaries, explanations, and
  free-form answers will be marked wrong despite being correct. Use overlap
  (`rouge_overlap`, `ngram_bleu`), semantic similarity (`semantic_similarity`),
  or an LLM judge (`llm_judge_rubric`) instead.
- **Normalization is a policy choice.** Stripping articles and punctuation is
  right for English QA but can corrupt answers where punctuation is meaningful
  (code, math, currency, dates). Override the `normalizer` argument when needed.
- **Multiple correct answers.** Without a multi-reference set, paraphrases and
  alias forms ("USA" vs "United States") are penalized. Provide an alias list.
- **Number/unit formatting.** "1,000", "1000", and "one thousand" do not match
  under text normalization; add task-specific canonicalization if numbers matter.

## API

- `normalize_text(text)` -> SQuAD-style normalized string.
- `exact_match(prediction, reference, normalizer=None)` -> bool (reference may be
  a string or a list of acceptable strings).
- `accuracy(predictions, references, normalizer=None)` -> float in [0, 1].
- `correctness_mask(predictions, references, normalizer=None)` -> list of bools
  for error analysis.

## References

- Rajpurkar et al., *SQuAD: 100,000+ Questions for Machine Comprehension of Text*
  (2016) — defines EM and F1 and the normalization procedure.
  https://arxiv.org/abs/1606.05250
- Official SQuAD v1.1 evaluation script (`normalize_answer`, `exact_match_score`).
  https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py
- Hugging Face `evaluate` SQuAD metric documentation.
  https://github.com/huggingface/evaluate/blob/main/metrics/squad_v2/README.md
- Brenndoerfer, *Exact Match and F1: Precision Metrics for NLP Evaluation*.
  https://mbrenndoerfer.com/writing/exact-match-f1-nlp-evaluation-metrics
