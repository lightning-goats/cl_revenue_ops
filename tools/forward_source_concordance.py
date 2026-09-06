#!/usr/bin/env python3
"""Read-only live/archive concordance, NOT historical source admission.

Run on the node: only aggregate results should leave it. No archive writes,
source-generation invention, migration, model training or economic actions.
An exact retained-view match does not establish absence of deleted history,
wallet continuity, channel-alias continuity, or causal/exposure completeness.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal, DecimalException
import json
from pathlib import Path
import re
import socket
import sqlite3
import time
from urllib.parse import quote

DAY = 86400
NS = 1_000_000_000
MAX_ROWS = 50_000
PAGE_SIZE = 500
MAX_PAGES = 101
MAX_REPLY_BYTES = 2 * 1024 * 1024
FIELDS = ("created_index", "updated_index", "in_channel", "in_htlc_id",
          "out_channel", "in_msat", "out_msat", "fee_msat",
          "received_time_ns", "resolved_time_ns")
TOTAL_FIELDS = ("settled_forward_count", "forwarded_in_msat", "forwarded_out_msat", "fee_msat")


class ConcordanceError(ValueError):
    """Sanitized refusal: never include raw RPC payloads or database paths."""


def _int(value, maximum=2**63 - 1):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ConcordanceError("invalid integer evidence")
    return value


def _ns(value):
    if isinstance(value, bool) or value is None:
        raise ConcordanceError("missing or invalid native time")
    try:
        parsed = Decimal(str(value)) * NS
        if not parsed.is_finite() or not 0 <= parsed <= 2**63-1 or parsed != parsed.to_integral_value():
            raise ConcordanceError("invalid native time precision")
        return _int(int(parsed))
    except (DecimalException, ValueError, OverflowError):
        raise ConcordanceError("invalid native time") from None


def _record(value, *, native=False):
    if not isinstance(value, dict):
        raise ConcordanceError("invalid settlement object")
    row = dict(value)
    if native:
        if row.get("status") != "settled":
            raise ConcordanceError("unexpected native settlement status")
        for name in ("received_time", "resolved_time"):
            row[name + "_ns"] = _ns(row.get(name))
        for name in ("in_msat", "out_msat", "fee_msat"):
            if isinstance(row.get(name), str) and len(row[name]) <= 24 and re.fullmatch(r"[0-9]+msat", row[name]):
                row[name] = int(row[name][:-4])
    try:
        result = {name: row[name] for name in FIELDS if name != "updated_index"}
        result["updated_index"] = row.get("updated_index")
        for name in ("created_index", "in_htlc_id"):
            _int(result[name], 2**64 - 1)
        if result["created_index"] == 0:
            raise ConcordanceError("missing created identity")
        if result["updated_index"] is not None:
            _int(result["updated_index"], 2**64 - 1)
            result["updated_index"] = result["updated_index"] or None
        for name in ("in_msat", "out_msat", "fee_msat", "received_time_ns", "resolved_time_ns"):
            _int(result[name])
        for name in ("in_channel", "out_channel"):
            channel = result[name]
            if not isinstance(channel, str) or not re.fullmatch(r"[0-9]{1,8}x[0-9]{1,8}x[0-9]{1,5}", channel):
                raise ConcordanceError("invalid channel label")
            if any(int(part) > bound for part, bound in zip(channel.split("x"), (2**24-1, 2**24-1, 2**16-1))):
                raise ConcordanceError("invalid channel range")
        if (not 0 < result["received_time_ns"] <= result["resolved_time_ns"]
                or result["out_msat"] == 0
                or result["in_msat"] - result["out_msat"] != result["fee_msat"]):
            raise ConcordanceError("inconsistent settlement payload")
        return result
    except (KeyError, TypeError):
        raise ConcordanceError("missing settlement evidence") from None


def _indexed(rows):
    records, native_ids, updates = {}, set(), set()
    for row in rows:
        identity = (row["in_channel"], row["in_htlc_id"])
        if row["created_index"] in records or identity in native_ids:
            raise ConcordanceError("duplicate settlement identity")
        if row["updated_index"] is not None:
            if row["updated_index"] in updates:
                raise ConcordanceError("duplicate settlement update index")
            updates.add(row["updated_index"])
        records[row["created_index"]] = row
        native_ids.add(identity)
    return records


def _totals(rows):
    totals = [0, 0, 0, 0]
    for row in rows:
        for index, value in enumerate((1, row["in_msat"], row["out_msat"], row["fee_msat"])):
            totals[index] = _int(totals[index] + value)
    return dict(zip(TOTAL_FIELDS, totals))


def _load_archive(database, start, end, now_ns, cursors):
    """One bounded SQLite snapshot; no Database constructor/schema/repair call."""
    conn = None
    try:
        path = Path(database).resolve(strict=True)
        conn = sqlite3.connect(f"file:{quote(str(path), safe='/')}?mode=ro", uri=True, timeout=1)
        conn.row_factory = sqlite3.Row
        sql_deadline = time.monotonic() + 5
        conn.set_progress_handler(lambda: int(time.monotonic() >= sql_deadline), 1000)
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        generations = [r[0] for r in conn.execute("""
            SELECT archive_generation FROM forward_archive_v1 WHERE received_time_ns >= ? AND received_time_ns < ?
            UNION SELECT archive_generation FROM forward_archive_coverage_v1 WHERE date_utc >= ? AND date_utc < ?
        """, (start*NS, end*NS, start, end))]
        if not generations or any(type(v) is not int or v != 1 for v in generations):
            raise ConcordanceError("missing or ambiguous archive generation")
        states = list(conn.execute("SELECT index_family,next_index,schema_version FROM forward_archive_sync_state_v1 WHERE archive_generation=1"))
        if len(states) != 2 or {r[0] for r in states} != {"created", "updated"}:
            raise ConcordanceError("missing archive cursor state")
        for family, next_index, version in states:
            if _int(version) != 1 or _int(next_index) > cursors[family] + 1:
                raise ConcordanceError("archive cursor exceeds live source or schema unsupported")
        raw = list(conn.execute("SELECT " + ",".join(FIELDS) + """ FROM forward_archive_v1
            WHERE archive_generation=1 AND status='settled' AND received_time_ns >= ? AND received_time_ns < ?
            ORDER BY created_index LIMIT ?""", (start*NS, end*NS, MAX_ROWS+1)))
        if len(raw) > MAX_ROWS:
            raise ConcordanceError("archive row budget exceeded")
        rows = [_record(dict(row)) for row in raw]
        by_day = defaultdict(list)
        for row in rows:
            if (row["created_index"] > cursors["created"] or (row["updated_index"] or 0) > cursors["updated"]
                    or row["resolved_time_ns"] > now_ns):
                raise ConcordanceError("archive evidence beyond live source view")
            by_day[row["received_time_ns"] // (DAY*NS) * DAY].append(row)
        coverage = list(conn.execute("SELECT * FROM forward_archive_coverage_v1 WHERE archive_generation=1 AND date_utc >= ? AND date_utc < ? ORDER BY date_utc", (start, end)))
        if len(coverage) != (end-start)//DAY:
            raise ConcordanceError("missing archive coverage day")
        daily_rows = list(conn.execute("SELECT * FROM forward_daily_channel_v1 WHERE archive_generation=1 AND date_utc >= ? AND date_utc < ? LIMIT ?", (start, end, MAX_ROWS+1)))
        if len(daily_rows) > MAX_ROWS:
            raise ConcordanceError("daily aggregate row budget exceeded")
        daily_totals = defaultdict(lambda: defaultdict(int))
        daily_keys = set()
        daily_fields = (*TOTAL_FIELDS, "sourced_forward_count", "sourced_volume_msat", "sourced_fee_msat")
        for row in daily_rows:
            day = _int(row["date_utc"])
            key = (day, row["channel_id"])
            if (day % DAY or not start <= day < end or key in daily_keys
                    or _int(row["schema_version"]) != 1):
                raise ConcordanceError("invalid daily channel identity")
            daily_keys.add(key)
            for key in daily_fields:
                daily_totals[row["date_utc"]][key] = _int(daily_totals[row["date_utc"]][key] + _int(row[key]))
        for day, coverage_row in zip(range(start, end, DAY), coverage):
            if (_int(coverage_row["date_utc"]) != day or _int(coverage_row["schema_version"]) != 1
                    or not (day+DAY)*NS <= _int(coverage_row["checked_at"]) <= now_ns
                    or coverage_row["reconciliation_status"] != "complete"
                    or json.loads(coverage_row["reasons_json"]) != []
                    or any(_int(coverage_row[key]) != 1 for key in ("created_sync_complete", "updated_sync_complete", "aggregate_complete"))):
                raise ConcordanceError("unqualified archive coverage")
            expected = _totals(by_day[day])
            expected["sourced_forward_count"] = expected["settled_forward_count"]
            for key, value in expected.items():
                if _int(coverage_row[key]) != value or daily_totals[day][key] != value:
                    raise ConcordanceError("raw/coverage/daily aggregate mismatch")
            if (daily_totals[day]["sourced_volume_msat"] != expected["forwarded_in_msat"]
                    or daily_totals[day]["sourced_fee_msat"] != expected["fee_msat"]):
                raise ConcordanceError("inbound aggregate mismatch")
        return _indexed(rows)
    except (OSError, sqlite3.Error, KeyError, IndexError, TypeError, json.JSONDecodeError):
        raise ConcordanceError("archive unavailable or malformed") from None
    finally:
        if conn is not None:
            conn.close()


def _identity(rpc):
    info = rpc("getinfo", {})
    if (not isinstance(info, dict) or not isinstance(info.get("id"), str)
            or not re.fullmatch(r"0[23][0-9a-f]{64}", info["id"])
            or info.get("network") not in ("bitcoin", "testnet", "testnet4", "regtest", "signet")):
        raise ConcordanceError("invalid live node identity")
    return info["id"], info["network"]


def _cursors(rpc):
    values = {}
    for family in ("created", "updated", "deleted"):
        response = rpc("wait", {"subsystem": "forwards", "indexname": family, "nextvalue": 0})
        if not isinstance(response, dict) or response.get("subsystem") != "forwards":
            raise ConcordanceError("invalid live cursor response")
        values[family] = _int(response.get(family), 2**64-1)
    return values


def _scan(rpc, family, cursors, start, end, now_ns):
    rows, request, read_count = [], 1, 0
    bound = cursors[family]
    for page_number in range(MAX_PAGES):
        response = rpc("listforwards", {"status": "settled", "index": family, "start": request, "limit": PAGE_SIZE})
        if not isinstance(response, dict) or not isinstance(response.get("forwards"), list):
            raise ConcordanceError("invalid native page")
        page = response["forwards"]
        if len(page) > PAGE_SIZE:
            raise ConcordanceError("native page exceeds limit")
        if not page:
            return _indexed(rows), read_count, page_number+1
        previous = request-1
        for payload in page:
            row = _record(payload, native=True)
            index = row[family + "_index"]
            if index is None or index <= previous:
                raise ConcordanceError("unordered or nonadvancing native page")
            previous = index
            if index > bound or row["created_index"] > cursors["created"] or (row["updated_index"] or 0) > cursors["updated"]:
                raise ConcordanceError("native source changed during scan")
            read_count += 1
            if read_count > MAX_ROWS:
                raise ConcordanceError("native row budget exceeded")
            if row["resolved_time_ns"] > now_ns:
                raise ConcordanceError("native outcome beyond observation bound")
            if start*NS <= row["received_time_ns"] < end*NS:
                rows.append(row)
        request = previous+1
    raise ConcordanceError("native page budget exceeded")


def check_concordance(database, rpc, start, end, *, now_ns=None):
    """Compare settled retained views, with no migration/admission authority.

    Caller RPC transport must have a per-call timeout. No retries or source
    mutations are attempted to manufacture a stable view on a busy node.
    """
    observed = time.time_ns() if now_ns is None else _int(now_ns)
    if (any(type(v) is not int or v <= 0 or v % DAY for v in (start, end))
            or not start < end <= observed//(DAY*NS)*DAY or end-start > 400*DAY):
        raise ConcordanceError("require closed UTC days within 400 days")
    deadline = time.monotonic() + 30

    def call(method, params):
        if time.monotonic() >= deadline:
            raise ConcordanceError("concordance time budget exceeded")
        try:
            response = rpc(method, params)
        except Exception:
            raise ConcordanceError("read-only native RPC unavailable") from None
        if time.monotonic() >= deadline:
            raise ConcordanceError("concordance time budget exceeded")
        return response

    identity, cursors = _identity(call), _cursors(call)
    archived = _load_archive(database, start, end, observed, cursors)
    created, created_count, created_pages = _scan(call, "created", cursors, start, end, observed)
    updated, updated_count, updated_pages = _scan(call, "updated", cursors, start, end, observed)
    if _cursors(call) != cursors or _identity(call) != identity:
        raise ConcordanceError("native source changed during scan")
    if updated != {k: v for k, v in created.items() if v["updated_index"] is not None}:
        raise ConcordanceError("created/updated native views disagree")
    common = archived.keys() & created.keys()
    conflicts = sum(archived[k] != created[k] for k in common)
    differing_fields = {field: sum(archived[k][field] != created[k][field] for k in common)
                        for field in FIELDS}
    time_differences = {}
    for field in ("received_time_ns", "resolved_time_ns"):
        deltas = [archived[k][field] - created[k][field] for k in common
                  if archived[k][field] != created[k][field]]
        time_differences[field] = {
            "different_events": len(deltas),
            "min_archive_minus_native_ns": min(deltas) if deltas else None,
            "max_archive_minus_native_ns": max(deltas) if deltas else None,
            # Diagnostic hypothesis only; never replace exact comparison with
            # a tolerance or admit the lossy representation as native truth.
            "binary_float_roundtrip_matches": sum(
                archived[k][field] == int(Decimal(str(float(Decimal(created[k][field])/NS)))*NS)
                for k in common if archived[k][field] != created[k][field]),
        }
    archive_only = archived.keys() - created.keys()
    native_only = created.keys() - archived.keys()
    # Check discrete temporal effects without claiming that unchanged order
    # proves every downstream model invariant or authorizes lossy timestamps.
    def timeline(records):
        return [(key, kind) for _, kind, key in sorted(
            (records[key][field], kind, key) for key in common
            for kind, field in enumerate(("received_time_ns", "resolved_time_ns")))]
    archived_timeline, native_timeline = timeline(archived), timeline(created)
    return {
        "schema_version": 1, "scope": "retained_settlement_concordance",
        "status": "match" if not (conflicts or archive_only or native_only) else "mismatch",
        "start": start, "end": end, "observed_at_ns": observed,
        "stable_cursor_observation": True, "cursors": cursors,
        "created_rows_scanned": created_count, "updated_rows_scanned": updated_count,
        "created_pages": created_pages, "updated_pages": updated_pages,
        "archive_totals": _totals(archived.values()), "native_totals": _totals(created.values()),
        "exact_matches": len(common)-conflicts, "conflicting_created_identities": conflicts,
        "differing_fields": differing_fields, "time_differences": time_differences,
        "received_utc_day_changes": sum(archived[k]["received_time_ns"]//(DAY*NS) != created[k]["received_time_ns"]//(DAY*NS) for k in common),
        "resolved_utc_day_changes": sum(archived[k]["resolved_time_ns"]//(DAY*NS) != created[k]["resolved_time_ns"]//(DAY*NS) for k in common),
        "event_time_order_position_changes": sum(a != b for a, b in zip(archived_timeline, native_timeline)),
        "archive_only_events": len(archive_only), "native_only_events": len(native_only),
        "coverage_days_checked": (end-start)//DAY,
        "historical_admission_eligible": False,
        "unverified": ["wallet_source_generation", "historical_channel_alias_continuity",
                       "history_deleted_before_observation", "causal_and_exposure_completeness"],
    }


class ReadOnlyUnixRpc:
    """Small bounded transport with no callable action-method escape hatch."""

    def __init__(self, path):
        self.path = str(path)

    def __call__(self, method, params):
        if not isinstance(method, str) or not isinstance(params, dict):
            raise ConcordanceError("RPC outside read-only concordance surface")
        allowed = (method == "getinfo" and params == {}) or (
            method == "wait" and set(params) == {"subsystem", "indexname", "nextvalue"}
            and params["subsystem"] == "forwards" and params["indexname"] in ("created", "updated", "deleted")
            and type(params["nextvalue"]) is int and params["nextvalue"] == 0) or (
            method == "listforwards" and set(params) == {"status", "index", "start", "limit"}
            and params["status"] == "settled" and params["index"] in ("created", "updated")
            and type(params["start"]) is int and 1 <= params["start"] <= 2**64-1
            and type(params["limit"]) is int and 1 <= params["limit"] <= PAGE_SIZE)
        if not allowed:
            raise ConcordanceError("RPC outside read-only concordance surface")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                deadline = time.monotonic() + 2
                connection.settimeout(2)
                connection.connect(self.path)
                connection.sendall(json.dumps({"jsonrpc": "2.0", "id": "concordance", "method": method, "params": params}).encode()+b"\n\n")
                data = bytearray()
                while b"\n\n" not in data:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ConcordanceError("native RPC timeout")
                    connection.settimeout(remaining)
                    chunk = connection.recv(65536)
                    if not chunk:
                        raise ConcordanceError("incomplete native RPC reply")
                    data.extend(chunk)
                    if len(data) > MAX_REPLY_BYTES:
                        raise ConcordanceError("native RPC reply budget exceeded")
                response = json.loads(bytes(data).split(b"\n\n", 1)[0], parse_float=Decimal)
                if (not isinstance(response, dict) or response.get("id") != "concordance"
                        or "error" in response or "result" not in response):
                    raise ConcordanceError("native RPC response refused")
                return response["result"]
        except (OSError, ValueError, KeyError, TypeError):
            raise ConcordanceError("read-only native RPC failed") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--rpc-file", required=True)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        result = check_concordance(args.database, ReadOnlyUnixRpc(args.rpc_file), args.start, args.end)
    except ConcordanceError as exc:
        print(json.dumps({"status": "refused", "reason": str(exc), "historical_admission_eligible": False}))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "match" else 1


if __name__ == "__main__":
    raise SystemExit(main())
