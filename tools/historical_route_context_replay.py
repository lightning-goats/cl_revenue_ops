#!/usr/bin/env python3
"""Read-only historical route-context research; never imported by runtime.

Predict the incoming adjacent channel CONDITIONAL on a settled forward, its
outgoing channel and amount. This does not estimate arrival demand, unseen
routes, counterfactual prices, rebalance returns or economic superiority.
Only aggregate scores leave this process. Raw history and channel labels stay
on the node. Parameters and temporal split must be frozen before evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sqlite3
import time
from urllib.parse import quote


DAY = 86400
NS = 1_000_000_000
MAX_ROWS = 50_000
HALF_LIFE_DAYS = 30
SHRINKAGE = 20
# Frozen research buckets in sats: small <50k, medium <250k, large >=250k.
AMOUNT_BOUNDARIES_MSAT = (50_000_000, 250_000_000)


class HistoryError(ValueError):
    pass


def validate_bounds(start: int, split: int, end: int) -> None:
    if any(type(value) is not int or value < 0 or value % DAY
           for value in (start, split, end)):
        raise HistoryError("bounds must be nonnegative UTC-midnight epochs")
    if not start < split < end or end - start > 400 * DAY:
        raise HistoryError("require start < split < end within 400 days")


def _validate_events(rows: list[dict], start: int, end: int) -> None:
    identities = set()
    for row in rows:
        integer_fields = ("archive_generation", "created_index", "in_msat",
                          "out_msat", "fee_msat", "received_time_ns", "resolved_time_ns")
        if any(type(row.get(key)) is not int or row[key] < 0 for key in integer_fields):
            raise HistoryError("malformed settled event")
        if (row["archive_generation"] != 1 or row["out_msat"] <= 0
                or row["in_msat"] - row["out_msat"] != row["fee_msat"]
                or not start * NS <= row["received_time_ns"] < end * NS
                or row["resolved_time_ns"] < row["received_time_ns"]
                or any(not isinstance(row.get(key), str) or not row[key]
                       for key in ("in_channel", "out_channel"))):
            raise HistoryError("inconsistent settled event")
        identity = (row["archive_generation"], row["created_index"])
        if identity in identities:
            raise HistoryError("duplicate canonical event identity")
        identities.add(identity)


def load_history(database: str, start: int, split: int, end: int,
                 *, now: int | None = None) -> list[dict]:
    """Bounded, snapshot-consistent, raw-to-coverage-checked read.

    A missing day is not fabricated zero exposure. Operational rollups are not
    mixed into this source. An archive/operational mismatch remains a separate
    diagnostic limitation, not something this reader repairs or waives.
    """
    validate_bounds(start, split, end)
    now = int(time.time()) if now is None else now
    if type(now) is not int or end > (now // DAY) * DAY:
        raise HistoryError("history must end before the current UTC day")
    path = Path(database).expanduser().resolve(strict=True)
    try:
        conn = sqlite3.connect(f"file:{quote(str(path), safe='/')}?mode=ro",
                               uri=True, timeout=1)
    except sqlite3.Error as exc:
        raise HistoryError("archive could not be opened read-only") from exc
    conn.row_factory = sqlite3.Row
    deadline = time.monotonic() + 10
    conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
    try:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        generations = {row[0] for row in conn.execute("""
            SELECT archive_generation FROM forward_archive_v1
            WHERE received_time_ns >= ? AND received_time_ns < ?
            UNION SELECT archive_generation FROM forward_archive_coverage_v1
            WHERE date_utc >= ? AND date_utc < ?
        """, (start * NS, end * NS, start, end))}
        if generations != {1}:
            raise HistoryError("missing or ambiguous archive generation")
        coverage = [dict(row) for row in conn.execute("""
            SELECT * FROM forward_archive_coverage_v1
            WHERE archive_generation = 1 AND date_utc >= ? AND date_utc < ?
            ORDER BY date_utc
        """, (start, end))]
        if len(coverage) != (end - start) // DAY:
            raise HistoryError("missing coverage day")
        for day, row in zip(range(start, end, DAY), coverage):
            if (row["date_utc"] != day or row["schema_version"] != 1
                    or type(row["checked_at"]) is not int
                    or row["checked_at"] < (day + DAY) * NS
                    or row["reconciliation_status"] != "complete"
                    or json.loads(row["reasons_json"]) != []
                    or any(type(row[key]) is not int or row[key] != 1 for key in (
                        "created_sync_complete", "updated_sync_complete", "aggregate_complete"))):
                raise HistoryError("unqualified coverage day")
        rows = [dict(row) for row in conn.execute("""
            SELECT archive_generation, created_index, in_channel, out_channel,
                   in_msat, out_msat, fee_msat, received_time_ns, resolved_time_ns
            FROM forward_archive_v1
            WHERE archive_generation = 1 AND status = 'settled'
              AND received_time_ns >= ? AND received_time_ns < ?
            ORDER BY received_time_ns, created_index LIMIT ?
        """, (start * NS, end * NS, MAX_ROWS + 1))]
        if len(rows) > MAX_ROWS:
            raise HistoryError("row budget exceeded; no partial training")
        _validate_events(rows, start, end)
        totals = defaultdict(Counter)
        for row in rows:
            day = row["received_time_ns"] // (DAY * NS) * DAY
            totals[day].update({"settled_forward_count": 1,
                                "forwarded_in_msat": row["in_msat"],
                                "forwarded_out_msat": row["out_msat"],
                                "fee_msat": row["fee_msat"], "sourced_forward_count": 1})
        for row in coverage:
            for key in ("settled_forward_count", "forwarded_in_msat",
                        "forwarded_out_msat", "fee_msat", "sourced_forward_count"):
                if type(row[key]) is not int or row[key] != totals[row["date_utc"]][key]:
                    raise HistoryError("raw archive/coverage mismatch")
        return rows
    except (sqlite3.Error, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise HistoryError("archive unavailable or malformed") from exc
    finally:
        conn.close()


def evaluate(rows: list[dict], start: int, split: int, end: int) -> dict:
    """Static training prefix and unseen suffix; no test-time fitting.

    Events crossing the training cutoff are withheld, not backdated into the
    prior. Events not yet settled by evaluation end are also withheld. This
    reconstructs availability at recorded settlement, not proof of historical
    notification receipt. Conditioning on settlement entails selection bias.
    """
    validate_bounds(start, split, end)
    if len(rows) > MAX_ROWS:
        raise HistoryError("row budget exceeded")
    _validate_events(rows, start, end)
    train = [row for row in rows if row["resolved_time_ns"] < split * NS]
    test = [row for row in rows if row["received_time_ns"] >= split * NS
            and row["resolved_time_ns"] < end * NS]
    result = {"schema_version": 1, "source": "canonical_archive_only",
              "scope": "incoming_channel_prediction_conditional_on_settled_forward",
              "start": start, "split": split, "end": end,
              "half_life_days": HALF_LIFE_DAYS, "shrinkage": SHRINKAGE,
              "amount_boundaries_msat": list(AMOUNT_BOUNDARIES_MSAT),
              "train_events": len(train), "test_events": len(test),
              "withheld_boundary_events": len(rows) - len(train) - len(test)}
    if not train or not test:
        return {**result, "status": "insufficient_evidence", "scores": None}
    pooled = Counter()
    outgoing = defaultdict(Counter)
    amount_context = defaultdict(Counter)
    bucket = lambda amount: sum(amount >= boundary for boundary in AMOUNT_BOUNDARIES_MSAT)
    for row in train:
        age_days = (split * NS - row["resolved_time_ns"]) / (DAY * NS)
        weight = 2 ** (-age_days / HALF_LIFE_DAYS)
        incoming, out = row["in_channel"], row["out_channel"]
        pooled[incoming] += weight
        outgoing[out][incoming] += weight
        amount_context[(out, bucket(row["out_msat"]))][incoming] += weight
    # One reserved unknown category; do not discover the vocabulary from test data.
    denominator = sum(pooled.values()) + len(pooled) + 1
    # Precompute totals: evaluation stays linear even with many sparse labels.
    outgoing_totals = {key: sum(counts.values()) for key, counts in outgoing.items()}
    context_totals = {key: sum(counts.values()) for key, counts in amount_context.items()}
    losses = Counter()
    unseen_incoming = 0
    unseen_outgoing = 0
    for row in test:
        incoming, out = row["in_channel"], row["out_channel"]
        unseen_incoming += incoming not in pooled
        unseen_outgoing += out not in outgoing
        global_p = (pooled.get(incoming, 0) + 1) / denominator
        out_counts = outgoing.get(out, {})
        out_p = (out_counts.get(incoming, 0) + SHRINKAGE * global_p) / (
            outgoing_totals.get(out, 0) + SHRINKAGE)
        context_key = (out, bucket(row["out_msat"]))
        context_counts = amount_context.get(context_key, {})
        context_p = (context_counts.get(incoming, 0) + SHRINKAGE * out_p) / (
            context_totals.get(context_key, 0) + SHRINKAGE)
        for name, p in (("pooled", global_p), ("outgoing", out_p),
                        ("outgoing_amount", context_p)):
            losses[name] -= math.log2(p)
    return {**result, "status": "evaluated", "weighted_train_events": sum(pooled.values()),
            "train_incoming_channels": len(pooled), "train_outgoing_channels": len(outgoing),
            "unseen_incoming_events": unseen_incoming, "unseen_outgoing_events": unseen_outgoing,
            "scores": {name: {"mean_log_loss_bits": loss / len(test)}
                       for name, loss in sorted(losses.items())}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--split", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    args = parser.parse_args()
    try:
        rows = load_history(args.database, args.start, args.split, args.end)
        result = evaluate(rows, args.start, args.split, args.end)
    except (HistoryError, OSError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "evaluated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
