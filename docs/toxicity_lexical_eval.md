# Toxicity (lexical screen)

> **This is a lexical word-list screen, not a real toxicity classifier.** Read the
> limits section before relying on it for anything user-facing.

## What it measures

A fast, offline, fully transparent **first-pass** toxicity filter: it flags text
that contains terms from a configurable lexicon and reports a crude score (fraction
of tokens that are flagged terms), per-text flags, the matched terms, and an
aggregate **toxicity rate** over a batch.

It exposes an injectable `scorer` so you can replace the word list with a real
classifier (Perspective API, Detoxify, an LLM judge) while keeping the rate/report
plumbing. The bundled lexicon is deliberately tiny and mild (placeholder insults) so
the package ships no graphic content — supply your own vetted lexicon for real use.

## When to use it

- **A cheap pre-filter** before an expensive classifier, or a deterministic
  stand-in in tests/CI where calling a model is undesirable.
- **Quick triage** of obviously abusive content during dataset cleaning.
- **A teaching example** of how brittle keyword moderation is — and a scaffold to
  drop a real model into.

## Limits and pitfalls (important)

A word-list cannot understand meaning, so it is wrong in both directions:

- **False negatives (misses real toxicity).** Implicit toxicity, dog whistles,
  threats and harassment phrased without flagged words, sarcasm, and toxicity in
  other languages all slip through. Attackers trivially evade word lists (spacing,
  homoglyphs, misspellings).
- **False positives (flags benign text).** Quotation, news reporting,
  counter-speech ("don't call people idiots"), reclaimed language, clinical or
  educational discussion, and negation ("that is not stupid") get flagged. The
  bundled demo deliberately shows a false positive.
- **No calibration.** The score is not a probability and saturates quickly; do not
  threshold it as if it were calibrated.
- **Lexicons encode bias.** Term lists over-flag dialects and identity terms and
  can cause discriminatory moderation. Audit any lexicon for fairness (see
  `group_fairness_disparity`).
- **Context window.** It scores text in isolation; conversation-level harm is
  invisible.

**Do not use this as a guardrail or for high-stakes moderation.** Use a trained,
audited classifier and inject it via `scorer`.

## API

- `flagged_terms(text, lexicon=None)` -> matched terms.
- `lexical_toxicity_score(text, lexicon=None)` -> crude [0, 1] score.
- `is_toxic(text, threshold=0.0, scorer=...)` -> bool.
- `toxicity_rate(texts, threshold=0.0, scorer=...)` -> fraction flagged.
- `toxicity_report(texts, threshold=0.0, lexicon=None, scorer=None)` -> rate +
  per-text scores/terms + an explicit caveat string.

## References

- Jigsaw / Google, *Perspective API* (production toxicity model).
  https://perspectiveapi.com/
- Hanu & Unitary, *Detoxify* (open-source toxicity classifier).
  https://github.com/unitaryai/detoxify
- Gehman et al., *RealToxicityPrompts* (NLG toxicity evaluation) (EMNLP Findings
  2020). https://arxiv.org/abs/2009.11462
- Sap et al., *The Risk of Racial Bias in Hate Speech Detection* (ACL 2019) — why
  lexical/biased detectors are unfair. https://aclanthology.org/P19-1163/
