#!/usr/bin/env python3
"""
Phase 7 performance profiler for cl_revenue_ops.

Reproducibly exercises the three cost centres identified in the deep-audit plan:

  (a) the heavy DB *read* queries (get_all_channels_full_pnl,
      get_total_routing_revenue, the budget / spend-ledger sums,
      get_recent_fee_changes, get_node_realized_fee_ppm_30d),
  (b) the profitability cycle (ChannelProfitabilityAnalyzer.analyze_all_channels),
  (c) the fee-adjustment cycle (FeeController.adjust_all_fees) and the
      rebalance planning cycle (RebalanceEngine.run_cycle) — best-effort,
      driven behind a synthetic mock plugin.

It seeds a synthetic SQLite DB to the production T0 row counts
(docs/audit/deep/prod-baseline-T0.md: ~92k fee_changes, realistic
forwards / rebalance_history / daily rollups) so the measured cumtime and
EXPLAIN QUERY PLAN reflect production scale.

Usage:
    python3 tools/perf/profile_cycles.py                 # synthetic DB, full run
    python3 tools/perf/profile_cycles.py --db PATH       # profile against a
                                                         # read-only copy of a
                                                         # real advisor.db
    python3 tools/perf/profile_cycles.py --repeats N     # timing repeats (default 20)
    python3 tools/perf/profile_cycles.py --markdown OUT  # write a markdown report

The script is READ-ONLY against any DB passed with --db (it opens a private
temp copy for seeding only when no --db is given).
"""

import argparse
import cProfile
import io
import os
import pstats
import random
import sqlite3
import sys
import tempfile
import time
from contextlib import contextmanager
from typing import Callable, Dict, List, Tuple
from unittest.mock import MagicMock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

# pyln.client is not installed in the profiling env; stub it exactly like the
# test conftest does so the modules import cleanly.
_mock_pyln = MagicMock()
_mock_pyln.Plugin = MagicMock
_mock_pyln.RpcError = Exception
sys.modules.setdefault("pyln", _mock_pyln)
sys.modules.setdefault("pyln.client", _mock_pyln)

from modules.database import Database  # noqa: E402

# Production T0 row counts (docs/audit/deep/prod-baseline-T0.md §2/§4).
T0_FEE_CHANGES = 92_410
T0_FORWARDS = 600
T0_REBALANCE_HISTORY = 477
T0_REBALANCE_COSTS = 500
T0_SPEND_EVENTS = 63
T0_FINANCIAL_SNAPSHOTS = 360
T0_PEER_CONNECTION_HISTORY = 2_706
N_CHANNELS = 36
DAY = 86_400


def _mock_plugin():
    p = MagicMock()
    p.log = MagicMock()
    return p


def _scids(n: int) -> List[str]:
    return [f"{800000 + i}x{i}x0" for i in range(n)]


def _peers(n: int) -> List[str]:
    return ["02" + f"{i:064x}"[:64] for i in range(n)]


def seed_synthetic_db(path: str, fee_changes: int = T0_FEE_CHANGES) -> None:
    """Seed a fresh DB to production T0 scale via the real schema.

    ``fee_changes`` can be lowered (e.g. by the benchmark-guard test) for a
    faster seed; the query plans are identical at any row count.
    """
    rnd = random.Random(1234)
    db = Database(path, _mock_plugin())
    db.initialize()
    conn = db._get_connection()
    now = int(time.time())
    scids = _scids(N_CHANNELS)
    peers = _peers(N_CHANNELS)
    n_fee_changes = int(fee_changes)

    # channel_states (36) + channel_costs
    for i, (scid, peer) in enumerate(zip(scids, peers)):
        conn.execute(
            "INSERT OR REPLACE INTO channel_states "
            "(channel_id, peer_id, state, flow_ratio, sats_in, sats_out, capacity, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (scid, peer, ["source", "sink", "balanced"][i % 3],
             rnd.random(), rnd.randint(0, 5_000_000), rnd.randint(0, 5_000_000),
             5_000_000, now),
        )
        conn.execute(
            "INSERT OR REPLACE INTO channel_costs "
            "(channel_id, peer_id, open_cost_sats, capacity_sats, opened_at) VALUES (?,?,?,?,?)",
            (scid, peer, rnd.randint(500, 5000), 5_000_000, now - rnd.randint(30, 200) * DAY),
        )

    # fee_changes (~92k) spread over 90 days across channels
    conn.execute("BEGIN")
    span = 90 * DAY
    rows = []
    for i in range(n_fee_changes):
        scid = scids[i % N_CHANNELS]
        peer = peers[i % N_CHANNELS]
        ts = now - rnd.randint(0, span)
        old_ppm = rnd.randint(0, 2000)
        new_ppm = max(0, old_ppm + rnd.randint(-200, 200))
        rows.append((scid, peer, old_ppm, new_ppm, "auto", 0, ts))
        if len(rows) >= 5000:
            conn.executemany(
                "INSERT INTO fee_changes "
                "(channel_id, peer_id, old_fee_ppm, new_fee_ppm, reason, manual, timestamp) "
                "VALUES (?,?,?,?,?,?,?)", rows)
            rows = []
    if rows:
        conn.executemany(
            "INSERT INTO fee_changes "
            "(channel_id, peer_id, old_fee_ppm, new_fee_ppm, reason, manual, timestamp) "
            "VALUES (?,?,?,?,?,?,?)", rows)
    conn.execute("COMMIT")

    # forwards (~600) over last 8 days
    conn.execute("BEGIN")
    for i in range(T0_FORWARDS):
        in_ch = scids[rnd.randrange(N_CHANNELS)]
        out_ch = scids[rnd.randrange(N_CHANNELS)]
        ts = now - rnd.randint(0, 8 * DAY)
        amt = rnd.randint(1000, 5_000_000)
        fee = max(1, amt // rnd.randint(1000, 100000))
        conn.execute(
            "INSERT INTO forwards "
            "(in_channel, out_channel, in_msat, out_msat, fee_msat, resolution_time, timestamp, resolved_time) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (in_ch, out_ch, amt * 1000, amt * 1000 - fee, fee, 0.5, ts, ts))
    conn.execute("COMMIT")

    # daily_forwarding_stats + inbound (36 channels x ~30 days)
    conn.execute("BEGIN")
    for d in range(30):
        date = ((now - d * DAY) // DAY) * DAY
        for scid in scids:
            if rnd.random() < 0.4:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO daily_forwarding_stats "
                "(channel_id, date, total_in_msat, total_out_msat, total_fee_msat, forward_count) "
                "VALUES (?,?,?,?,?,?)",
                (scid, date, rnd.randint(0, 10**9), rnd.randint(0, 10**9),
                 rnd.randint(0, 10**6), rnd.randint(0, 50)))
            conn.execute(
                "INSERT OR REPLACE INTO daily_forwarding_stats_inbound "
                "(channel_id, date, total_in_msat, total_fee_msat, forward_count) "
                "VALUES (?,?,?,?,?)",
                (scid, date, rnd.randint(0, 10**9), rnd.randint(0, 10**6), rnd.randint(0, 50)))
    conn.execute("COMMIT")

    # rebalance_costs (~500)
    conn.execute("BEGIN")
    for i in range(T0_REBALANCE_COSTS):
        scid = scids[rnd.randrange(N_CHANNELS)]
        peer = peers[rnd.randrange(N_CHANNELS)]
        ts = now - rnd.randint(0, 90 * DAY)
        cost = rnd.randint(1, 500)
        conn.execute(
            "INSERT INTO rebalance_costs "
            "(channel_id, peer_id, cost_sats, cost_msat, amount_sats, timestamp) VALUES (?,?,?,?,?,?)",
            (scid, peer, cost, cost * 1000, rnd.randint(10000, 2_000_000), ts))
    conn.execute("COMMIT")

    # rebalance_history (~477)
    conn.execute("BEGIN")
    for i in range(T0_REBALANCE_HISTORY):
        frm = scids[rnd.randrange(N_CHANNELS)]
        to = scids[rnd.randrange(N_CHANNELS)]
        ts = now - rnd.randint(0, 90 * DAY)
        status = rnd.choice(["success", "failed", "success", "success"])
        conn.execute(
            "INSERT INTO rebalance_history "
            "(from_channel, to_channel, amount_sats, max_fee_sats, actual_fee_sats, "
            " expected_profit_sats, actual_profit_sats, status, rebalance_type, timestamp) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (frm, to, rnd.randint(10000, 2_000_000), 500, rnd.randint(0, 400),
             rnd.randint(0, 1000), rnd.randint(-100, 1000), status, "normal", ts))
    conn.execute("COMMIT")

    # spend_events (~63)
    conn.execute("BEGIN")
    for i in range(T0_SPEND_EVENTS):
        ts = now - rnd.randint(0, 90 * DAY)
        conn.execute(
            "INSERT INTO spend_events "
            "(event_id, category, subcategory, amount_sats, timestamp, reference_id, channel_id, source) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (f"evt:{i}", rnd.choice(["rebalance", "boltz", "onchain"]), None,
             rnd.randint(10, 5000), ts, f"ref:{i}", scids[rnd.randrange(N_CHANNELS)], "test"))
    conn.execute("COMMIT")

    # financial_snapshots (~360)
    conn.execute("BEGIN")
    for i in range(T0_FINANCIAL_SNAPSHOTS):
        ts = now - i * DAY
        conn.execute(
            "INSERT OR REPLACE INTO financial_snapshots "
            "(timestamp, total_local_balance_sats, total_remote_balance_sats, total_onchain_sats, "
            " total_capacity_sats, total_revenue_accumulated_sats, total_rebalance_cost_accumulated_sats, "
            " channel_count) VALUES (?,?,?,?,?,?,?,?)",
            (ts, rnd.randint(0, 10**8), rnd.randint(0, 10**8), rnd.randint(0, 10**7),
             10**8, rnd.randint(0, 10**6), rnd.randint(0, 10**5), N_CHANNELS))
    conn.execute("COMMIT")

    # peer_connection_history (~2706) — best-effort (schema may vary)
    try:
        conn.execute("BEGIN")
        for i in range(T0_PEER_CONNECTION_HISTORY):
            peer = peers[rnd.randrange(N_CHANNELS)]
            ts = now - rnd.randint(0, 90 * DAY)
            conn.execute(
                "INSERT INTO peer_connection_history (peer_id, connected, timestamp) VALUES (?,?,?)",
                (peer, rnd.randint(0, 1), ts))
        conn.execute("COMMIT")
    except sqlite3.OperationalError:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass

    conn.execute("ANALYZE")
    conn.commit()


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_call(fn: Callable, repeats: int) -> Tuple[float, float, float]:
    """Return (best_ms, mean_ms, worst_ms) wall-clock over `repeats` calls."""
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return min(samples), sum(samples) / len(samples), max(samples)


def explain(conn: sqlite3.Connection, sql: str, params: tuple) -> List[str]:
    try:
        rows = conn.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()
    except sqlite3.OperationalError as e:
        return [f"(EXPLAIN failed: {e})"]
    out = []
    for r in rows:
        detail = r["detail"] if isinstance(r, sqlite3.Row) else r[-1]
        out.append(detail)
    return out


@contextmanager
def profiled(label: str, store: Dict[str, str]):
    pr = cProfile.Profile()
    pr.enable()
    try:
        yield
    finally:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(20)
        store[label] = s.getvalue()


# ---------------------------------------------------------------------------
# DB read-query benchmark set
# ---------------------------------------------------------------------------

def build_db_benchmarks(db: Database):
    now = int(time.time())
    since_30d = now - 30 * DAY
    since_24h = now - DAY
    conn = db._get_connection()

    benches = []  # (name, callable)

    benches.append(("get_all_channels_full_pnl(30)",
                    lambda: db.get_all_channels_full_pnl(30)))
    benches.append(("get_all_channels_full_pnl(90)",
                    lambda: db.get_all_channels_full_pnl(90)))
    benches.append(("get_total_routing_revenue(30d)",
                    lambda: db.get_total_routing_revenue(since_30d)))
    benches.append(("get_node_realized_fee_ppm_30d",
                    lambda: db.get_node_realized_fee_ppm_30d()))
    benches.append(("get_recent_fee_changes(limit=50)",
                    lambda: db.get_recent_fee_changes(limit=50)))
    benches.append(("get_recent_fee_changes(per-channel)",
                    lambda: db.get_recent_fee_changes(limit=50, channel_id=_scids(N_CHANNELS)[0])))
    benches.append(("get_daily_rebalance_spend(24h)",
                    lambda: db.get_daily_rebalance_spend(window_hours=24)))
    benches.append(("get_budget_status(30d)",
                    lambda: db.get_budget_status(since_30d)))
    benches.append(("get_spend_ledger_summary(30d)",
                    lambda: db.get_spend_ledger_summary(since_30d)))
    benches.append(("get_category_spend_sats(rebalance,30d)",
                    lambda: db.get_category_spend_sats("rebalance", since_timestamp=since_30d)))

    # Explicit EXPLAIN QUERY PLAN targets — the raw SQL of the scale-sensitive reads.
    explains = [
        ("forwards revenue-sum (get_total_routing_revenue)",
         "SELECT COALESCE(SUM(fee_msat),0) FROM forwards WHERE timestamp >= ?",
         (since_30d,)),
        ("daily_forwarding_stats date-range sum",
         "SELECT COALESCE(SUM(total_fee_msat),0) FROM daily_forwarding_stats WHERE date >= ? AND date < ?",
         (since_30d, now)),
        ("fee_changes ORDER BY timestamp DESC LIMIT",
         "SELECT * FROM fee_changes ORDER BY timestamp DESC LIMIT 50", ()),
        ("fee_changes per-channel recent",
         "SELECT * FROM fee_changes WHERE channel_id = ? ORDER BY timestamp DESC LIMIT 50",
         (_scids(N_CHANNELS)[0],)),
        ("forwards GROUP BY REPLACE(out_channel) (full_pnl exit)",
         "SELECT REPLACE(out_channel,':','x') AS c, SUM(fee_msat) FROM forwards WHERE timestamp >= ? GROUP BY REPLACE(out_channel,':','x')",
         (since_30d,)),
        ("rebalance_costs SUM by channel (full_pnl)",
         "SELECT REPLACE(channel_id,':','x') AS c, SUM(COALESCE(cost_msat,cost_sats*1000)) FROM rebalance_costs WHERE timestamp >= ? GROUP BY REPLACE(channel_id,':','x')",
         (since_30d,)),
        ("rebalance_costs timestamp SUM (budget)",
         "SELECT COALESCE(SUM(cost_sats),0) FROM rebalance_costs WHERE timestamp >= ?",
         (since_24h,)),
        ("spend_events timestamp SUM (budget)",
         "SELECT COALESCE(SUM(amount_sats),0) FROM spend_events WHERE timestamp >= ?",
         (since_30d,)),
    ]
    return benches, explains, conn


def run_db_profile(db: Database, repeats: int):
    benches, explains, conn = build_db_benchmarks(db)

    # Warm connection / caches once.
    for _name, fn in benches:
        try:
            fn()
        except Exception:
            pass

    timing_rows = []
    prof_store: Dict[str, str] = {}
    for name, fn in benches:
        try:
            best, mean, worst = time_call(fn, repeats)
            with profiled(name, prof_store):
                for _ in range(max(3, repeats // 4)):
                    fn()
            timing_rows.append({"name": name, "best_ms": best, "mean_ms": mean, "worst_ms": worst})
        except Exception as e:
            timing_rows.append({"name": name, "best_ms": -1, "mean_ms": -1,
                                "worst_ms": -1, "error": str(e)})

    explain_rows = []
    for name, sql, params in explains:
        plan = explain(conn, sql, params)
        explain_rows.append({"name": name, "plan": plan})

    return timing_rows, explain_rows, prof_store


# ---------------------------------------------------------------------------
# Cycle profiling (best-effort, behind synthetic mocks)
# ---------------------------------------------------------------------------

def run_cycle_profile(db: Database, repeats: int):
    """Best-effort profiling of profitability / fee / rebalance cycles.

    These paths are heavily RPC-coupled; where the synthetic driver cannot
    satisfy a path it is reported as SKIPPED with the reason, and the DB-read
    baseline above remains the authoritative signal.
    """
    from tools.perf.cycle_driver import build_stack  # local helper
    rows: List[dict] = []
    prof_store: Dict[str, str] = {}

    stack = build_stack(db)

    def _bench(label, fn):
        try:
            fn()  # warm
            best, mean, worst = time_call(fn, max(3, repeats // 4))
            with profiled(label, prof_store):
                fn()
            rows.append({"name": label, "best_ms": best, "mean_ms": mean, "worst_ms": worst})
        except Exception as e:
            rows.append({"name": label, "best_ms": -1, "mean_ms": -1, "worst_ms": -1,
                         "error": f"{type(e).__name__}: {e}"})

    if stack.get("profitability") is not None:
        _bench("profitability.analyze_all_channels(force=True)",
               lambda: stack["profitability"].analyze_all_channels(force=True))
    if stack.get("fee_controller") is not None:
        _bench("fee_controller.adjust_all_fees",
               lambda: stack["fee_controller"].adjust_all_fees())
    if stack.get("rebalance_engine") is not None:
        _bench("rebalance_engine.run_cycle",
               lambda: stack["rebalance_engine"].run_cycle())

    return rows, prof_store


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_timing_table(rows: List[dict]) -> str:
    out = ["| query / cycle | best ms | mean ms | worst ms |",
           "|---|---:|---:|---:|"]
    for r in rows:
        if r.get("error"):
            out.append(f"| {r['name']} | ERR | ERR | {r['error']} |")
        else:
            out.append(f"| {r['name']} | {r['best_ms']:.3f} | {r['mean_ms']:.3f} | {r['worst_ms']:.3f} |")
    return "\n".join(out)


def _fmt_explain(rows: List[dict]) -> str:
    out = []
    for r in rows:
        out.append(f"**{r['name']}**")
        for line in r["plan"]:
            out.append(f"  - `{line}`")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", help="path to a real advisor.db (opened read-only via a temp copy)")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--markdown", help="write a markdown report to this path")
    ap.add_argument("--no-cycles", action="store_true", help="skip cycle profiling")
    args = ap.parse_args()

    tmpdir = tempfile.mkdtemp(prefix="perf_cycles_")
    source_label = ""
    if args.db:
        import shutil
        dst = os.path.join(tmpdir, "copy.db")
        shutil.copy(args.db, dst)
        db_path = dst
        source_label = f"real DB copy of {args.db}"
        db = Database(db_path, _mock_plugin())
    else:
        db_path = os.path.join(tmpdir, "synthetic.db")
        print(f"Seeding synthetic DB at {db_path} (T0 scale, ~{T0_FEE_CHANGES} fee_changes)...")
        t0 = time.perf_counter()
        seed_synthetic_db(db_path)
        print(f"  seeded in {time.perf_counter()-t0:.1f}s "
              f"({os.path.getsize(db_path)/1e6:.1f} MB)")
        source_label = "synthetic DB seeded to T0 row counts"
        db = Database(db_path, _mock_plugin())

    print("Profiling DB read queries...")
    timing_rows, explain_rows, db_prof = run_db_profile(db, args.repeats)

    cycle_rows: List[dict] = []
    cycle_prof: Dict[str, str] = {}
    if not args.no_cycles:
        print("Profiling cycles (best-effort)...")
        try:
            cycle_rows, cycle_prof = run_cycle_profile(db, args.repeats)
        except Exception as e:
            print(f"  cycle profiling unavailable: {e}")

    print("\n=== DB read timings ===")
    print(_fmt_timing_table(timing_rows))
    if cycle_rows:
        print("\n=== Cycle timings ===")
        print(_fmt_timing_table(cycle_rows))

    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(f"<!-- generated by tools/perf/profile_cycles.py; source: {source_label} -->\n\n")
            f.write("## DB read-query timings\n\n")
            f.write(_fmt_timing_table(timing_rows) + "\n\n")
            if cycle_rows:
                f.write("## Cycle timings\n\n")
                f.write(_fmt_timing_table(cycle_rows) + "\n\n")
            f.write("## EXPLAIN QUERY PLAN (scale-sensitive reads)\n\n")
            f.write(_fmt_explain(explain_rows) + "\n\n")
            f.write("## cProfile cumtime (top 20) — DB reads\n\n")
            for label, txt in db_prof.items():
                f.write(f"### {label}\n\n```\n{txt}\n```\n\n")
            if cycle_prof:
                f.write("## cProfile cumtime (top 20) — cycles\n\n")
                for label, txt in cycle_prof.items():
                    f.write(f"### {label}\n\n```\n{txt}\n```\n\n")
        print(f"\nMarkdown report written to {args.markdown}")

    print("\n=== EXPLAIN QUERY PLAN ===")
    print(_fmt_explain(explain_rows))


if __name__ == "__main__":
    main()
