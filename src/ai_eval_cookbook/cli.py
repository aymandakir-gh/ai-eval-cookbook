"""Minimal command-line entry point for ai-eval-cookbook.

This is intentionally tiny: the value of the project is the importable recipe
modules, not a CLI. The ``aiec`` command simply reports the installed version
and lists the available recipe modules so users can discover them.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from typing import List, Optional

from . import __version__


def _list_recipe_modules() -> List[str]:
    """Return the importable recipe module names within the package."""
    import ai_eval_cookbook as pkg

    skip = {"cli"}
    names = []
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.ispkg or info.name in skip or info.name.startswith("_"):
            continue
        names.append(info.name)
    return sorted(names)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aiec",
        description="ai-eval-cookbook: offline LLM evaluation recipes.",
    )
    parser.add_argument(
        "--version", action="version", version=f"ai-eval-cookbook {__version__}"
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("list", help="List available recipe modules.")

    args = parser.parse_args(argv)

    if args.command == "list" or args.command is None:
        recipes = _list_recipe_modules()
        if recipes:
            print("Available recipes:")
            for name in recipes:
                print(f"  - {name}")
        else:
            print("No recipe modules found.")
        return 0

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
