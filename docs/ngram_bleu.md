# N-gram BLEU

## What it measures

BLEU (BiLingual Evaluation Understudy) scores a candidate against one or more
references by how many of its n-grams appear in a reference. It combines:

- **Modified n-gram precision** for n = 1..N. "Modified" = each candidate n-gram's
  count is *clipped* to the maximum number of times it appears in any single
  reference, so repeating "the the the" cannot inflate the score.
- A **geometric mean** of the n-gram precisions (default N=4, uniform weights),
  which is harsh: if any order has zero matches, BLEU is 0.
- A **brevity penalty (BP)**: `BP = 1` when the candidate is at least as long as
  the closest reference, otherwise `exp(1 - r/c)`. This stops a model from gaming
  precision by emitting only a few high-confidence words.

This module computes **corpus-level** BLEU (statistics pooled across all sentences,
which is the meaningful form) plus a `sentence_bleu` convenience wrapper.

## When to use it

- **Machine translation**, the task BLEU was designed for.
- Any **constrained generation** with reference outputs where surface-form overlap
  is a reasonable proxy: templated responses, format-locked answers, short
  summaries with canonical phrasings.
- As a cheap, deterministic, reproducible regression signal across model versions.

## Pitfalls

- **Surface-form only.** BLEU rewards exact n-gram overlap and is blind to
  synonyms, paraphrase, and meaning. A perfect paraphrase can score near zero.
  For meaning, see `semantic_similarity` or an LLM judge.
- **Sentence-level BLEU is noisy.** One missing 4-gram zeroes a short sentence.
  Always prefer corpus BLEU for comparing systems; this module uses unsmoothed
  precision, matching the original paper (libraries like `sacrebleu` add smoothing
  for sentence-level use).
- **Tokenization changes the number.** BLEU is only comparable under identical
  tokenization. Use a standardized scheme (e.g. `sacrebleu`) when publishing.
- **Not for open-ended generation.** Chat, reasoning, and creative outputs have no
  single reference; BLEU is the wrong tool.
- **Multiple references help.** More valid references raise correlation with human
  judgment; with one reference BLEU is stricter than the task warrants.

## API

- `corpus_bleu(candidates, references, max_n=4, tokenizer=simple_tokenize)`
  — `references[i]` is the list of references for `candidates[i]`.
- `sentence_bleu(candidate, references, max_n=4, tokenizer=...)`.
- `brevity_penalty(cand_len, ref_len)`, `simple_tokenize(text)` helpers.

## References

- Papineni, Roukos, Ward, Zhu, *BLEU: a Method for Automatic Evaluation of Machine
  Translation* (ACL 2002). https://aclanthology.org/P02-1040/
- Post, *A Call for Clarity in Reporting BLEU Scores* (sacrebleu, WMT 2018).
  https://aclanthology.org/W18-6319/
- NLTK BLEU implementation reference.
  https://www.nltk.org/api/nltk.translate.bleu_score.html
- Wikipedia, *BLEU*. https://en.wikipedia.org/wiki/BLEU
