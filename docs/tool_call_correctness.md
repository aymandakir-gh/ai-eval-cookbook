# Tool-call correctness

## What it measures

For agents and function/tool calling, a call is correct only if **both** parts are
right: the **tool name** and the **arguments**. This recipe scores a predicted tool
call against an expected one and aggregates over a dataset:

- **name accuracy** — did the model pick the right tool?
- **exact-call accuracy** — name matches AND every argument matches with no missing
  or extra keys (argument F1 == 1.0).
- **argument F1** — per-key precision/recall over arguments, so partial credit is
  visible: extra arguments hurt precision, missing arguments hurt recall.

Because strict equality is often too harsh, argument matching is configurable:
case/whitespace-insensitive strings, a numeric tolerance for floats, ignored keys
(optional params you don't grade), or a fully custom `arg_matcher`.

## When to use it

- **Agent / tool-use evaluation** — the primary signal for "can this model drive my
  tools correctly?"
- **Function-calling regression tests** — gate releases on name and argument
  accuracy.
- **Prompt / schema debugging** — argument F1 plus name accuracy localizes whether
  the model misroutes (wrong tool) or mis-fills (wrong args).

## Pitfalls

- **Exact equality is brittle.** "Paris" vs "paris", `1` vs `1.0`, `5` vs `"5"`,
  reordered lists — all fail naive equality though they may be acceptable. Use the
  case-insensitive / tolerance / custom-matcher knobs deliberately, and document
  which you used.
- **Argument semantics vary.** Some args are free-text (a search query) where exact
  match is wrong; grade those with similarity, not equality (inject an
  `arg_matcher`). Others are enums/ids where exact match is right.
- **Optional vs required args.** Penalizing a model for omitting a defaulted optional
  argument is usually unfair — list those in `ignore_keys`.
- **Single call vs trajectory.** This module scores one expected call per example.
  Multi-step agents that emit a *sequence* of calls need alignment/ordering logic
  (and partial-trajectory credit) on top.
- **Name-only accuracy hides arg failures.** A model can pick the right tool every
  time yet fill it wrong. Always report exact-call accuracy alongside name accuracy.
- **Type coercion from JSON.** If calls are parsed from JSON, validate structure
  first (`json_schema_validity`) so type mismatches don't masquerade as wrong
  values.

## API

- `name_matches(expected, predicted, case_insensitive=False)` -> bool.
- `argument_scores(expected, predicted, ...)` -> `{precision, recall, f1, matched}`.
- `is_correct_call(expected, predicted, ...)` -> bool (name + full arg match).
- `evaluate(expected_calls, predicted_calls, ...)` -> name/exact/arg-F1 aggregates.
- Knobs: `case_insensitive`, `float_tol`, `ignore_keys`, `arg_matcher`.
- Calls are dicts: `{"name": str, "arguments": {param: value}}`.

## References

- OpenAI, *Function calling guide*.
  https://platform.openai.com/docs/guides/function-calling
- Anthropic, *Tool use with Claude*.
  https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- Berkeley Function-Calling Leaderboard (BFCL) — methodology for grading tool calls.
  https://gorilla.cs.berkeley.edu/leaderboard.html
- Patil et al., *Gorilla: Large Language Model Connected with Massive APIs* (2023).
  https://arxiv.org/abs/2305.15334
