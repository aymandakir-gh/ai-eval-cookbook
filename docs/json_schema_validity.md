# JSON schema validity

## What it measures

For "JSON mode", structured outputs, and tool/function calls, the first quality
gate is: **does the output parse and match the expected shape?** This recipe reports
the **validity rate** over a batch of model outputs against a schema, and separates
the two failure modes:

- **Parse failures** — the text isn't valid JSON (trailing prose, markdown fences,
  unquoted keys, truncation).
- **Schema failures** — it parses but violates the contract (missing required
  fields, wrong types, out-of-range values, unexpected properties).

It includes a small, transparent validator for a practical subset of JSON Schema:
`type`, `required`, `properties`, `items`, `enum`, `minimum`/`maximum`,
`minLength`/`maxLength`, `minItems`/`maxItems`, and `additionalProperties: false`.
It correctly treats JSON booleans as *not* integers/numbers — a common gotcha in
Python validators.

## When to use it

- **Structured extraction / function calling** — measure how often outputs are
  machine-usable before any semantic check.
- **Comparing prompts, models, or decoding constraints** (grammar-constrained vs
  free) on format reliability.
- **CI gating** — fail a release if validity rate drops below a budget.
- Triage: the report's parse-vs-schema split tells you whether to fix prompting
  (parse) or the schema/prompt contract (schema).

## Pitfalls

- **Validity ≠ correctness.** A well-formed object can still contain wrong values.
  This is a *format* metric; pair it with content checks (`exact_match_accuracy`,
  `tool_call_correctness`).
- **This is a JSON Schema *subset*.** No `$ref`, `oneOf`/`anyOf`/`allOf`,
  `patternProperties`, format assertions, etc. For full Draft 2020-12 semantics use
  the `jsonschema` library; swap the validator while keeping the rate/report logic.
- **Preprocessing changes the rate.** Models often wrap JSON in ```json fences or
  add prose. Decide whether to strip fences before scoring (and do it consistently)
  — stripping measures the *content*, not stripping measures *raw protocol
  compliance*.
- **Booleans vs numbers.** In JSON `true`/`false` are not numbers; many naive
  Python checks accept `True` as an int because `bool` subclasses `int`. This module
  guards against that — verify your production validator does too.
- **Partial/streamed output.** Truncated JSON counts as a parse failure; that's
  usually correct, but watch for max-token limits inflating the failure rate.

## API

- `validate(value, schema, path="$")` -> list of error strings (empty = valid).
- `parse_and_validate(text, schema)` -> `(is_valid, errors)`.
- `validity_rate(outputs, schema)` -> fraction valid.
- `validity_report(outputs, schema)` -> rate + parse/schema failure counts +
  per-failure details.

## References

- JSON Schema, *Draft 2020-12 specification*. https://json-schema.org/specification
- OpenAI, *Structured Outputs guide* (JSON schema for model outputs).
  https://platform.openai.com/docs/guides/structured-outputs
- `python-jsonschema` library (full-spec validator for production).
  https://python-jsonschema.readthedocs.io/
- JSON Schema, *Understanding JSON Schema* (keyword reference).
  https://json-schema.org/understanding-json-schema/
