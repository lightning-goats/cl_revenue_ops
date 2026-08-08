#!/usr/bin/env python3
"""Render post-removal and old-release rollback Core Lightning configs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import stat
import sys


RETIRED_PREFIXES = (
    "revenue-ops-boltz-",
    "revenue-ops-lnplus-",
    "revenue-ops-planner-",
    "revenue-ops-expansion-treasury-",
)
ROLLBACK_GATES = (
    ("revenue-ops-planner-enabled", "false"),
    ("revenue-ops-planner-dry-run", "true"),
    ("revenue-ops-planner-max-opens-per-cycle", "0"),
    ("revenue-ops-planner-execute-closes", "false"),
    ("revenue-ops-planner-max-closes-per-cycle", "0"),
    ("revenue-ops-boltz-enabled", "false"),
    ("revenue-ops-boltz-auto-cycle-enabled", "false"),
    ("revenue-ops-expansion-treasury-enabled", "false"),
    ("revenue-ops-lnplus-swaps-enabled", "false"),
    ("revenue-ops-lnplus-execute-applications", "false"),
)
ASSIGNMENT = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_-]*)\s*=")


class ConfigError(RuntimeError):
    pass


def _reject_symlink(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ConfigError(f"{label} must not be a symlink")


def _read_input(path: Path) -> bytes:
    _reject_symlink(path, "input")
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ConfigError("input does not exist") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ConfigError("input must be a regular file")
    return path.read_bytes()


def _option_name(line: str) -> str | None:
    stripped = line.lstrip()
    if not stripped or stripped.startswith("#"):
        return None
    match = ASSIGNMENT.match(line)
    if match:
        return match.group(1).lower()
    token = stripped.split(None, 1)[0].lower()
    if token.startswith(RETIRED_PREFIXES):
        raise ConfigError("malformed retired option assignment")
    return None


def render(source: bytes) -> tuple[bytes, bytes]:
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError("input is not UTF-8") from exc
    kept = []
    for line in text.splitlines(keepends=True):
        name = _option_name(line)
        if name is not None and name.startswith(RETIRED_PREFIXES):
            continue
        kept.append(line)
    active_text = "".join(kept)
    if active_text and not active_text.endswith(("\n", "\r")):
        active_text += "\n"
    rollback_text = active_text
    if rollback_text and not rollback_text.endswith("\n"):
        rollback_text += "\n"
    rollback_text += "\n# cl_revenue_ops liquidity-executor rollback safety gates\n"
    rollback_text += "".join(f"{key}={value}\n" for key, value in ROLLBACK_GATES)
    return active_text.encode("utf-8"), rollback_text.encode("utf-8")


def _exclusive_write(path: Path, payload: bytes) -> None:
    _reject_symlink(path, "output")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
    except Exception:
        os.close(fd)
        fd = -1
        try:
            path.unlink()
        except OSError:
            pass
        raise
    finally:
        if fd >= 0:
            os.close(fd)


def run(input_path: Path, active_output: Path, rollback_output: Path) -> None:
    source = _read_input(input_path)
    for output in (active_output, rollback_output):
        _reject_symlink(output, "output")
        if output.exists():
            raise ConfigError("output already exists")
    if active_output.absolute() == rollback_output.absolute():
        raise ConfigError("active and rollback outputs must differ")
    active, rollback = render(source)
    active_created = False
    try:
        _exclusive_write(active_output, active)
        active_created = True
        _exclusive_write(rollback_output, rollback)
    except Exception:
        if active_created:
            try:
                active_output.unlink()
            except OSError:
                pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--active-output", required=True, type=Path)
    parser.add_argument("--rollback-output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        run(args.input, args.active_output, args.rollback_output)
    except (ConfigError, OSError) as exc:
        print(f"config rendering failed: {exc}", file=sys.stderr)
        return 1
    print("active and rollback configs written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
