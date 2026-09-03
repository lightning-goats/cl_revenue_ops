#!/usr/bin/env python3
"""Seal and later verify a private Grand Prix holdout traffic seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import Any


SCHEMA = "polar-grand-prix-holdout-secret-v1"
DOMAIN = "cl_revenue_ops/polar-grand-prix/holdout/v1"
COMMITMENT_RE = re.compile(r"sha256:[0-9a-f]{64}")
MAX_BYTES = 4096


class HoldoutError(RuntimeError):
    """The holdout secret or commitment is unsafe or inconsistent."""


def commitment(seed: int, salt: str) -> str:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0:
        raise HoldoutError("seed must be a positive integer")
    if not isinstance(salt, str) or not re.fullmatch(r"[0-9a-f]{64}", salt):
        raise HoldoutError("salt must contain 32 random bytes in lowercase hex")
    payload = json.dumps(
        {"domain": DOMAIN, "salt": salt, "seed": seed},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _read_secret(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise HoldoutError(f"cannot open holdout secret: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise HoldoutError("holdout secret must be a regular file")
        if info.st_mode & 0o077:
            raise HoldoutError("holdout secret permissions must be 0600")
        raw = os.read(descriptor, MAX_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(raw) > MAX_BYTES:
        raise HoldoutError("holdout secret is oversized")
    try:
        value = json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HoldoutError("holdout secret is malformed") from exc
    if not isinstance(value, dict) or set(value) != {"schema", "seed", "salt", "commitment"}:
        raise HoldoutError("holdout secret fields are malformed")
    if value["schema"] != SCHEMA:
        raise HoldoutError("holdout secret schema is unsupported")
    expected = commitment(value["seed"], value["salt"])
    if value["commitment"] != expected:
        raise HoldoutError("holdout secret does not match its commitment")
    return value


def seal(path: Path) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    seed = secrets.randbelow(2**63 - 1) + 1
    salt = secrets.token_hex(32)
    digest = commitment(seed, salt)
    payload = json.dumps(
        {"schema": SCHEMA, "seed": seed, "salt": salt, "commitment": digest},
        indent=2,
        sort_keys=True,
    ).encode() + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise HoldoutError("refusing to replace an existing holdout secret") from exc
    except OSError as exc:
        raise HoldoutError(f"cannot create holdout secret: {exc}") from exc
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"schema": SCHEMA, "commitment": digest}


def reveal(path: Path, expected_commitment: str) -> dict[str, Any]:
    if not COMMITMENT_RE.fullmatch(expected_commitment):
        raise HoldoutError("expected commitment must be a complete SHA-256 digest")
    value = _read_secret(path)
    if not secrets.compare_digest(value["commitment"], expected_commitment):
        raise HoldoutError("holdout reveal does not match the frozen commitment")
    return {
        "schema": SCHEMA,
        "seed": value["seed"],
        "salt": value["salt"],
        "commitment": value["commitment"],
        "verified": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    seal_parser = sub.add_parser("seal")
    seal_parser.add_argument("--secret", type=Path, required=True)
    reveal_parser = sub.add_parser("reveal")
    reveal_parser.add_argument("--secret", type=Path, required=True)
    reveal_parser.add_argument("--expected-commitment", required=True)
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    try:
        result = (
            seal(args.secret)
            if args.command == "seal"
            else reveal(args.secret, args.expected_commitment)
        )
    except HoldoutError as exc:
        sys.stderr.write(f"holdout error: {exc}\n")
        return 2
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
