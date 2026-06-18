import pytest

from ai_eval_cookbook.json_schema_validity import (
    parse_and_validate,
    validate,
    validity_rate,
    validity_report,
)

SCHEMA = {
    "type": "object",
    "required": ["name", "age"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "age": {"type": "integer", "minimum": 0, "maximum": 130},
        "role": {"type": "string", "enum": ["user", "admin"]},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    },
    "additionalProperties": False,
}


def test_valid_object():
    ok, errors = parse_and_validate('{"name": "Ada", "age": 36, "role": "admin"}', SCHEMA)
    assert ok and errors == []


def test_missing_required():
    ok, errors = parse_and_validate('{"name": "Cy"}', SCHEMA)
    assert not ok
    assert any("missing required property 'age'" in e for e in errors)


def test_minimum_violation():
    ok, errors = parse_and_validate('{"name": "Bob", "age": -1}', SCHEMA)
    assert not ok
    assert any("minimum" in e for e in errors)


def test_enum_violation():
    ok, errors = parse_and_validate('{"name": "X", "age": 1, "role": "root"}', SCHEMA)
    assert not ok
    assert any("enum" in e for e in errors)


def test_additional_properties():
    ok, errors = parse_and_validate('{"name": "Di", "age": 5, "extra": true}', SCHEMA)
    assert not ok
    assert any("unexpected properties" in e for e in errors)


def test_parse_failure():
    ok, errors = parse_and_validate("{not json}", SCHEMA)
    assert not ok
    assert "not valid JSON" in errors[0]


def test_boolean_is_not_integer():
    ok, errors = parse_and_validate('{"name": "X", "age": true}', SCHEMA)
    assert not ok
    assert any("expected type" in e for e in errors)


def test_float_is_not_integer():
    ok, errors = parse_and_validate('{"name": "X", "age": 3.5}', SCHEMA)
    assert not ok


def test_array_items_and_maxitems():
    ok, _ = parse_and_validate('{"name": "X", "age": 1, "tags": ["a", "b"]}', SCHEMA)
    assert ok
    ok2, errors = parse_and_validate('{"name": "X", "age": 1, "tags": ["a", "b", "c", "d"]}', SCHEMA)
    assert not ok2
    assert any("maxItems" in e for e in errors)
    ok3, errors3 = parse_and_validate('{"name": "X", "age": 1, "tags": [1, 2]}', SCHEMA)
    assert not ok3  # items must be strings


def test_validate_on_parsed_value():
    assert validate({"name": "Z", "age": 5}, SCHEMA) == []
    assert validate(42, {"type": "string"}) != []


def test_validity_rate():
    outputs = [
        '{"name": "Ada", "age": 36}',  # valid
        '{"name": "Bob", "age": -1}',  # invalid
        "{bad}",  # parse fail
        '{"name": "Cy", "age": 5}',  # valid
    ]
    assert validity_rate(outputs, SCHEMA) == pytest.approx(0.5)


def test_validity_report_splits_failures():
    outputs = [
        '{"name": "Ada", "age": 36}',  # valid
        '{"name": "Bob", "age": -1}',  # schema fail
        "{bad}",  # parse fail
    ]
    rep = validity_report(outputs, SCHEMA)
    assert rep["valid"] == 1
    assert rep["parse_failures"] == 1
    assert rep["schema_failures"] == 1
    assert rep["validity_rate"] == pytest.approx(1 / 3)


def test_empty_outputs():
    assert validity_rate([], SCHEMA) == 0.0
    rep = validity_report([], SCHEMA)
    assert rep["validity_rate"] == 0.0


def test_nested_object_validation():
    schema = {
        "type": "object",
        "properties": {"inner": {"type": "object", "required": ["k"], "properties": {"k": {"type": "integer"}}}},
    }
    ok, _ = parse_and_validate('{"inner": {"k": 1}}', schema)
    assert ok
    bad, errors = parse_and_validate('{"inner": {"k": "x"}}', schema)
    assert not bad
    assert any("inner.k" in e for e in errors)
