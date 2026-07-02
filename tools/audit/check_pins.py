#!/usr/bin/env python3
"""Supply-chain pin verifier (Phase 3C).

Two guarantees, usable as a CI gate:

  1. Every runtime requirement in ``requirements.txt`` is EXACTLY pinned
     (``name==version``). Bare names, ``>=``/``<=``/``~=``/``!=``/``<``/``>``
     specifiers, ranges, and unpinned extras all fail.

  2. Each pinned version MATCHES what is installed in the interpreter running
     this script. A drift between the pin and the installed distribution fails.

This is deliberately conservative and stdlib-only so it can run anywhere the
plugin runs. It reads ``requirements.txt`` relative to the repo root (two levels
up from this file) unless ``--requirements`` is given.

Exit codes:
  0  all requirements exactly pinned AND installed versions match
  1  a pin is missing/loose, OR installed version differs, OR a dep is absent
  2  requirements file not found / unreadable

Usage:
  python3 tools/audit/check_pins.py
  python3 tools/audit/check_pins.py --requirements requirements.txt
  python3 tools/audit/check_pins.py --no-installed-check   # pin-shape only
"""
from __future__ import annotations

import argparse
import os
import re
import sys

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - Python < 3.8 not supported here
    import importlib_metadata  # type: ignore

# A fully-pinned line: distribution name, optional [extras], then EXACTLY '=='.
# Anything else (bare name, >=, ~=, <, >, !=, ===, no version) is a failure.
_PIN_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?P<extras>\[[^\]]*\])?"
    r"==(?P<version>[A-Za-z0-9][A-Za-z0-9._+!-]*)$"
)


def _canonical(name: str) -> str:
    """PEP 503 canonical distribution name (case/sep-insensitive)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _strip_inline_comment(line: str) -> str:
    # requirements allow trailing ` #comment`; drop it (not '#' mid-token).
    idx = line.find(" #")
    if idx != -1:
        line = line[:idx]
    return line.strip()


def parse_requirements(path: str):
    """Yield (raw_line, lineno) for each non-blank, non-comment requirement."""
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-"):
                # -r / -c / --hash style directives: not a plain pin. Skip the
                # option lines but flag include-of-other-files so a loose child
                # can't sneak past this gate unnoticed.
                yield (line, lineno, "directive")
                continue
            yield (_strip_inline_comment(line), lineno, "requirement")


def check(path: str, check_installed: bool = True):
    problems = []
    checked = 0

    for entry in parse_requirements(path):
        line, lineno, kind = entry
        if kind == "directive":
            problems.append(
                f"{path}:{lineno}: unsupported directive '{line}' -- this gate "
                f"only understands exact 'name==version' pins"
            )
            continue

        m = _PIN_RE.match(line)
        if not m:
            problems.append(
                f"{path}:{lineno}: NOT exactly pinned: '{line}' "
                f"(require 'name==version')"
            )
            continue

        checked += 1
        if not check_installed:
            continue

        name = m.group("name")
        pinned = m.group("version")
        try:
            installed = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            problems.append(
                f"{path}:{lineno}: '{name}' pinned to {pinned} but NOT installed "
                f"in this interpreter ({sys.executable})"
            )
            continue

        if _canonical(installed) and installed != pinned:
            problems.append(
                f"{path}:{lineno}: '{name}' pinned to {pinned} but installed "
                f"version is {installed} (drift)"
            )

    return checked, problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify requirements.txt is fully pinned.")
    default_req = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "requirements.txt",
    )
    parser.add_argument("--requirements", default=default_req,
                        help="Path to requirements file (default: repo requirements.txt)")
    parser.add_argument("--no-installed-check", action="store_true",
                        help="Only verify pin shape; skip comparison to installed versions")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.requirements):
        print(f"check_pins: requirements file not found: {args.requirements}", file=sys.stderr)
        return 2

    try:
        checked, problems = check(args.requirements, check_installed=not args.no_installed_check)
    except OSError as e:
        print(f"check_pins: cannot read {args.requirements}: {e}", file=sys.stderr)
        return 2

    if problems:
        print(f"check_pins: FAIL ({len(problems)} problem(s)) in {args.requirements}")
        for p in problems:
            print(f"  - {p}")
        return 1

    scope = "pin-shape" if args.no_installed_check else "pin-shape + installed-match"
    print(f"check_pins: OK -- {checked} requirement(s) fully pinned ({scope}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
