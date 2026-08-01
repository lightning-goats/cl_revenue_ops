#!/usr/bin/env python3
"""Supply-chain pin verifier (Phase 3C).

Three guarantees, usable as a CI gate:

  1. Every runtime requirement in ``requirements.txt`` is EXACTLY pinned
     (``name==version``). Bare names, ``>=``/``<=``/``~=``/``!=``/``<``/``>``
     specifiers, ranges, and unpinned extras all fail.

  2. Each pinned version MATCHES what is installed in the interpreter running
     this script. A drift between the pin and the installed distribution fails.

  3. (Audit 2026-08-01 wave2) Each requirements.txt pin also MATCHES the
     corresponding entry in ``requirements.lock`` — the hash-pinned closure
     used to drift silently because only requirements.txt was verified. A
     runtime pin missing from the lock, or pinned to a different version
     there, fails.

This is deliberately conservative and stdlib-only so it can run anywhere the
plugin runs. It reads ``requirements.txt`` relative to the repo root (two levels
up from this file) unless ``--requirements`` is given; the lock defaults to
``requirements.lock`` next to it (skipped only if the file does not exist, or
with ``--no-lock-check``).

Exit codes:
  0  all requirements exactly pinned AND installed versions match AND the
     lock agrees on every shared pin
  1  a pin is missing/loose, OR installed version differs, OR a dep is
     absent, OR the lock disagrees with requirements.txt
  2  requirements file not found / unreadable

Usage:
  python3 tools/audit/check_pins.py
  python3 tools/audit/check_pins.py --requirements requirements.txt
  python3 tools/audit/check_pins.py --no-installed-check   # pin-shape only
  python3 tools/audit/check_pins.py --no-lock-check        # skip lock diff
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


def parse_lock_pins(path: str):
    """Return {canonical_name: version} for every ``name==version`` line in a
    requirements.lock file. Lock lines carry ``--hash=`` options after the
    pin; only the leading requirement token is parsed. Malformed lines are
    ignored here — the lock's own installability is pip's job
    (``pip install --require-hashes``); this gate only diffs shared pins."""
    pins = {}
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            token = line.split()[0]
            m = _PIN_RE.match(token)
            if m:
                pins[_canonical(m.group("name"))] = m.group("version")
    return pins


def check(path: str, check_installed: bool = True, lock_path: str = None):
    problems = []
    checked = 0

    lock_pins = None
    if lock_path is not None:
        try:
            lock_pins = parse_lock_pins(lock_path)
        except OSError as e:
            problems.append(f"{lock_path}: cannot read lock file: {e}")

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
        name = m.group("name")
        pinned = m.group("version")

        # Guarantee 3: the hash-pinned lock must agree on every shared pin.
        if lock_pins is not None:
            lock_version = lock_pins.get(_canonical(name))
            if lock_version is None:
                problems.append(
                    f"{path}:{lineno}: '{name}' pinned to {pinned} but ABSENT "
                    f"from {lock_path} (the lock claims the full closure)"
                )
            elif lock_version != pinned:
                problems.append(
                    f"{path}:{lineno}: '{name}' pinned to {pinned} but "
                    f"{lock_path} pins {lock_version} (lock drift)"
                )

        if not check_installed:
            continue

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
    parser.add_argument("--lock", default=None,
                        help="Path to requirements.lock (default: requirements.lock "
                             "next to the requirements file, if it exists)")
    parser.add_argument("--no-installed-check", action="store_true",
                        help="Only verify pin shape; skip comparison to installed versions")
    parser.add_argument("--no-lock-check", action="store_true",
                        help="Skip the requirements.txt <-> requirements.lock shared-pin diff")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.requirements):
        print(f"check_pins: requirements file not found: {args.requirements}", file=sys.stderr)
        return 2

    lock_path = None
    if not args.no_lock_check:
        if args.lock is not None:
            lock_path = args.lock
            if not os.path.isfile(lock_path):
                print(f"check_pins: lock file not found: {lock_path}", file=sys.stderr)
                return 2
        else:
            candidate = os.path.join(
                os.path.dirname(os.path.abspath(args.requirements)),
                "requirements.lock",
            )
            if os.path.isfile(candidate):
                lock_path = candidate

    try:
        checked, problems = check(args.requirements,
                                  check_installed=not args.no_installed_check,
                                  lock_path=lock_path)
    except OSError as e:
        print(f"check_pins: cannot read {args.requirements}: {e}", file=sys.stderr)
        return 2

    if problems:
        print(f"check_pins: FAIL ({len(problems)} problem(s)) in {args.requirements}")
        for p in problems:
            print(f"  - {p}")
        return 1

    scope = "pin-shape" if args.no_installed_check else "pin-shape + installed-match"
    if lock_path is not None:
        scope += " + lock-agreement"
    print(f"check_pins: OK -- {checked} requirement(s) fully pinned ({scope}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
