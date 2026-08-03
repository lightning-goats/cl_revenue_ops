#!/usr/bin/env python3
"""Read-only SQLite preflight for liquidity executor decommissioning.

External Boltz daemon/journal state is deliberately outside this report and must
be verified through the old runtime read-only status surface before shutdown.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import stat
import sys
import time
from urllib.parse import quote


TERMINAL_LNPLUS = frozenset({"ended", "failed", "withdrawn", "cancelled_remote"})
CONTRACT_LNPLUS = frozenset({"active"})
RETIRED_CATEGORIES = frozenset({"boltz", "lnplus", "channel_open", "channel_close"})
RETIRED_ID_PREFIXES = ("planner-", "lnplus-", "boltz-")


class UnsafeState(RuntimeError):
    pass


def _regular_nonsymlink(path: Path, *, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise UnsafeState(f"{label} does not exist") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise UnsafeState(f"{label} must be a regular non-symlink file")
    return info


def _validate_output_target(path: Path) -> None:
    if path.is_symlink():
        raise UnsafeState("output must not be a symlink")
    if not path.exists():
        return
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise UnsafeState("existing output is not a regular file")
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise UnsafeState("existing output is not owner-owned mode 0600")
    # Even a safe prior report is never overwritten: the new artifact uses
    # O_EXCL so concurrent/pre-existing evidence cannot be confused.
    raise UnsafeState("output already exists")


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row[1]) for row in rows}


def _required_table(conn: sqlite3.Connection, table: str, columns: set[str]) -> None:
    actual = _table_columns(conn, table)
    missing = columns - actual
    if missing:
        raise UnsafeState(f"required schema unavailable for {table}")


def _policy_has_no_close(conn: sqlite3.Connection, peer: str) -> bool:
    row = conn.execute(
        "SELECT tags FROM peer_policies WHERE peer_id = ?", (peer,)
    ).fetchone()
    if row is None:
        return False
    try:
        tags = json.loads(row[0] or "[]")
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(tags, list) and "no_close" in tags


def _inspect(conn: sqlite3.Connection) -> tuple[dict, list[dict]]:
    _required_table(
        conn, "lnplus_swaps",
        {"status", "ends_at", "outbound_peer", "incoming_peer"},
    )
    _required_table(
        conn, "spend_reservations",
        {"reservation_id", "category", "subcategory", "status"},
    )
    _required_table(conn, "peer_policies", {"peer_id", "tags"})

    unresolved = []
    contracts = []
    rows = conn.execute(
        "SELECT status, ends_at, outbound_peer, incoming_peer FROM lnplus_swaps"
    ).fetchall()
    now = int(time.time())
    for row in rows:
        status_value = row[0]
        status_name = str(status_value or "").strip().lower()
        if not status_name:
            raise UnsafeState("malformed LN+ row")
        if status_name in TERMINAL_LNPLUS:
            continue
        if status_name not in CONTRACT_LNPLUS:
            unresolved.append(status_name)
            continue
        expiry = row[1]
        try:
            expiry = int(expiry)
        except (TypeError, ValueError) as exc:
            raise UnsafeState("active LN+ contract has malformed expiry") from exc
        if expiry <= 0:
            raise UnsafeState("active LN+ contract has malformed expiry")
        for direction, peer in (("outbound", row[2]), ("incoming", row[3])):
            if peer is None or str(peer).strip() == "":
                continue
            peer = str(peer).strip()
            if not _policy_has_no_close(conn, peer):
                raise UnsafeState("active LN+ contract lacks generic no_close protection")
            contracts.append({
                "peer_id": peer,
                "direction": direction,
                "stored_expiry": expiry,
                "expired_at_preflight": expiry <= now,
                "tag_removal_owner": "operator",
            })
    if unresolved:
        raise UnsafeState("unresolved LN+ application or opening state exists")

    active = conn.execute(
        "SELECT reservation_id, category, subcategory FROM spend_reservations "
        "WHERE status = 'active'"
    ).fetchall()
    retired_active = []
    for reservation_id, category, subcategory in active:
        rid = str(reservation_id or "").strip().lower()
        cat = str(category or "").strip().lower()
        subcat = str(subcategory or "").strip().lower()
        if (
            cat in RETIRED_CATEGORIES
            or rid.startswith(RETIRED_ID_PREFIXES)
            or subcat == "lnplus_swap"
        ):
            retired_active.append(cat or "unknown")
    if retired_active:
        raise UnsafeState("active retired liquidity reservation exists")

    return {
        "safe": True,
        "unresolved_lnplus_rows": 0,
        "active_retired_reservations": 0,
        "active_contract_peer_tags_verified": len(contracts),
    }, contracts


def _exclusive_json(path: Path, payload: dict) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            fd = -1
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if fd >= 0:
            os.close(fd)


def run(db_path: Path, output: Path) -> dict:
    db_info = _regular_nonsymlink(db_path, label="database")
    _validate_output_target(output)
    uri = f"file:{quote(str(db_path.resolve()), safe='/')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.execute("PRAGMA query_only=ON")
        preconditions, contracts = _inspect(conn)
    finally:
        conn.close()
    payload = {
        "schema_version": 1,
        "generated_at": int(time.time()),
        "database": {
            "path": str(db_path.resolve()),
            "device": db_info.st_dev,
            "inode": db_info.st_ino,
            "size_bytes": db_info.st_size,
        },
        "preconditions": preconditions,
        "contracts": contracts,
    }
    _exclusive_json(output, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        run(args.db, args.output)
    except (UnsafeState, sqlite3.Error, OSError) as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return 1
    print("preflight passed; private report written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
