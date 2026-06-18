"""Structured-output validity: does the model's JSON parse and match a schema?

Two common failure modes for "JSON mode" / tool outputs: (1) the text isn't valid
JSON at all, and (2) it parses but violates the expected shape (missing fields,
wrong types, out-of-range values). This recipe measures the **validity rate** over a
batch of model outputs against a schema.

To stay dependency-free it ships a small, transparent validator for a practical
subset of JSON Schema: ``type`` (object/array/string/number/integer/boolean/null),
``required``, ``properties``, ``items``, ``enum``, ``minimum``/``maximum``,
``minLength``/``maxLength``, ``minItems``/``maxItems``, and ``additionalProperties``.
It is intentionally a subset — for full Draft 2020-12 coverage use the ``jsonschema``
library; the docs say when to graduate.

Pure standard library, offline.
"""

from __future__ import annotations

import json
from numbers import Real
from typing import Any, Dict, List, Optional, Sequence, Tuple

_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, Real) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def validate(value: Any, schema: Dict[str, Any], path: str = "$") -> List[str]:
    """Validate a Python value (already parsed from JSON) against a schema subset.

    Returns a list of human-readable error strings; an empty list means valid.
    """
    errors: List[str] = []

    expected_type = schema.get("type")
    if expected_type is not None:
        types = expected_type if isinstance(expected_type, list) else [expected_type]
        if not any(_TYPE_CHECKS.get(t, lambda v: True)(value) for t in types):
            errors.append("%s: expected type %s, got %s" % (path, types, type(value).__name__))
            return errors  # type wrong -> further checks are meaningless

    if "enum" in schema and value not in schema["enum"]:
        errors.append("%s: %r not in enum %r" % (path, value, schema["enum"]))

    if isinstance(value, Real) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append("%s: %r < minimum %r" % (path, value, schema["minimum"]))
        if "maximum" in schema and value > schema["maximum"]:
            errors.append("%s: %r > maximum %r" % (path, value, schema["maximum"]))

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append("%s: string shorter than minLength %d" % (path, schema["minLength"]))
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append("%s: string longer than maxLength %d" % (path, schema["maxLength"]))

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append("%s: fewer than minItems %d" % (path, schema["minItems"]))
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append("%s: more than maxItems %d" % (path, schema["maxItems"]))
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(value):
                errors.extend(validate(item, item_schema, "%s[%d]" % (path, i)))

    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append("%s: missing required property %r" % (path, key))
        for key, subschema in props.items():
            if key in value:
                errors.extend(validate(value[key], subschema, "%s.%s" % (path, key)))
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(props)
            if extra:
                errors.append("%s: unexpected properties %s" % (path, sorted(extra)))

    return errors


def parse_and_validate(
    text: str, schema: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """Parse ``text`` as JSON and validate against ``schema``.

    Returns ``(is_valid, errors)``. A parse failure yields
    ``(False, ["...not valid JSON..."])``.
    """
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        return False, ["$: not valid JSON (%s)" % exc]
    errors = validate(value, schema)
    return (not errors), errors


def validity_rate(
    outputs: Sequence[str], schema: Dict[str, Any]
) -> float:
    """Fraction of raw string outputs that parse AND satisfy the schema."""
    if not outputs:
        return 0.0
    valid = sum(1 for o in outputs if parse_and_validate(o, schema)[0])
    return valid / len(outputs)


def validity_report(
    outputs: Sequence[str], schema: Dict[str, Any]
) -> Dict[str, object]:
    """Validity rate plus a breakdown of parse failures vs schema failures."""
    n = len(outputs)
    parse_fail = schema_fail = valid = 0
    failures: List[Dict[str, object]] = []
    for i, o in enumerate(outputs):
        ok, errors = parse_and_validate(o, schema)
        if ok:
            valid += 1
        else:
            if errors and "not valid JSON" in errors[0]:
                parse_fail += 1
            else:
                schema_fail += 1
            failures.append({"index": i, "errors": errors})
    return {
        "validity_rate": (valid / n) if n else 0.0,
        "valid": valid,
        "parse_failures": parse_fail,
        "schema_failures": schema_fail,
        "failures": failures,
    }


if __name__ == "__main__":
    schema = {
        "type": "object",
        "required": ["name", "age"],
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "age": {"type": "integer", "minimum": 0, "maximum": 130},
            "role": {"type": "string", "enum": ["user", "admin"]},
        },
        "additionalProperties": False,
    }
    outputs = [
        '{"name": "Ada", "age": 36, "role": "admin"}',  # valid
        '{"name": "Bob", "age": -1}',  # age < minimum
        '{"name": "Cy"}',  # missing age
        "{not json}",  # parse failure
        '{"name": "Di", "age": 5, "extra": true}',  # additionalProperties
    ]
    report = validity_report(outputs, schema)
    print("validity rate:", report["validity_rate"])
    print("parse failures:", report["parse_failures"], "schema failures:", report["schema_failures"])
    for f in report["failures"]:
        print(" -", f["index"], f["errors"])
