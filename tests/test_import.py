"""Smoke test: the package imports and exposes a version."""

import ai_eval_cookbook


def test_version_is_a_string():
    assert isinstance(ai_eval_cookbook.__version__, str)
    assert ai_eval_cookbook.__version__


def test_cli_list_runs():
    from ai_eval_cookbook.cli import main

    assert main(["list"]) == 0
