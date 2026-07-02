#!/usr/bin/env python3
"""Concurrency stress harness for cl_revenue_ops (deep-audit Phase 2B).

Drives the plugin's *real* modules under concurrent load to falsify the
"single-threaded timer" safety assumption (Phase-1 fact: daemon loops run
concurrently with pyln RPC handlers on the ThreadSafeRpcProxy worker pool).

What it does
------------
* Instantiates the real ``Database`` on a temp SQLite file in the SAME journal
  mode as production (WAL, ``isolation_level=None`` — the ``Database`` class
  enforces both in ``_get_connection``).
* Wires the real ``Config`` and (best-effort) the real fee_controller /
  rebalancer / flow_analyzer against a mock RPC so the actual daemon-loop
  bodies (``run_fee_adjustment`` / ``run_rebalance_check`` / ``run_flow_analysis``)
  execute. If a loop cannot be wired cleanly against the mock RPC it falls back
  to a DB-touching simulation that exercises the same tables (recorded honestly
  in the report).
* Runs those shortened timer loops in parallel with a firehose of money-path
  RPC calls from multiple threads:
    - ``revenue-spend-reserve`` / ``revenue-spend-release`` / ``revenue-spend-settle``
      (generic ledger, multiple categories)
    - direct rebalance-category reservations via ``Database.reserve_budget``
      (the OTHER budget table + OTHER lock — this is the P1-003 cross-category
      surface)
    - Boltz-swap reservations via the P4-014 atomic path
      (``category="boltz"`` reserve_spend against the live unified budget) so
      the budget invariant now covers the boltz leg of DD1's cross-category rail
    - chainswap reservations via the P4-023 atomic path (the SIXTH reserver;
      ``category="boltz"`` reserve_spend on the same boltz rail
      ``_open_swap_budget_reservation`` drives) so budget-never-exceeded now
      covers the 6th (chainswap) swap-create spend path
    - capacity-planner channel_open reservations via the P4-018 atomic path
      (``category="channel_open"`` reserve_spend against the live unified budget)
      so budget-never-exceeded now covers the FOURTH (planner) daemon spend path
    - defibrillation diagnostic-shock reservations via the P4-020 atomic path
      (``reserve_budget`` category=rebalance against the live unified budget,
      exactly the rail rebalancer.diagnostic_rebalance now reserves on) so the
      budget invariant covers the FIFTH (defibrillation) daemon spend path
    - ``revenue-config set`` reloads (config._lock / _version churn)

Asserted invariants (soak)
--------------------------
1. integrity      -- ``PRAGMA integrity_check`` == ok after the soak.
2. no-deadlock    -- a watchdog thread fails the run if global progress stalls
                     for longer than --watchdog seconds.
3. no-thread-exc  -- ``threading.excepthook`` captured no unhandled exception
                     escaping any worker/loop thread.
4. budget         -- total *active reserved* across ALL budget tables
                     (spend_reservations incl. boltz + budget_reservations)
                     never exceeds ``effective_budget_sats`` at any sampled
                     instant. This is a direct test of P1-003's / P4-014's
                     cross-category TOCTOU, now WITH a boltz reserver in the mix.

The budget invariant is EXPECTED to fail against the current tree (P1-003 is an
open finding). Pass ``--allow-known-budget-toctou`` to treat a budget violation
as a documented/known reproduction (harness still exits 0 as long as the
structural invariants hold); otherwise any invariant failure exits non-zero.

Usage
-----
    python3 tools/audit/stress_concurrency.py               # ~60s CI soak
    python3 tools/audit/stress_concurrency.py --soak 1800   # 30-min soak
    python3 tools/audit/stress_concurrency.py --json out.json --allow-known-budget-toctou
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import tempfile
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock

# Make the repo importable (tools/audit/ -> repo root) and reuse the test
# substrate's plugin loader (fakes pyln.client).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.plugin_test_utils import load_plugin_module  # noqa: E402


# ---------------------------------------------------------------------------
# Mock substrate
# ---------------------------------------------------------------------------
class _SafePluginMock:
    """Minimal ThreadSafePluginProxy stand-in: .log + .rpc, both thread-safe."""

    def __init__(self, rpc):
        self.rpc = rpc
        self._log_lock = threading.Lock()
        self.log_count = 0

    def log(self, message, level="info"):  # noqa: A003 - mirrors pyln API
        with self._log_lock:
            self.log_count += 1


def _make_mock_rpc():
    """Mock RPC returning production-shaped-but-empty payloads.

    Real loop bodies (adjust_all_fees / analyze_all_channels /
    find_rebalance_candidates) query several RPC methods; empty-but-typed
    returns let them run to completion without doing real work.
    """
    rpc = MagicMock()
    node_id = "02" + "a" * 64
    rpc.getinfo.return_value = {"id": node_id, "alias": "stress", "network": "regtest", "blockheight": 100}
    rpc.listchannels.return_value = {"channels": []}
    rpc.listpeerchannels.return_value = {"channels": []}
    rpc.listpeers.return_value = {"peers": []}
    rpc.listfunds.return_value = {"channels": [], "outputs": []}
    rpc.listforwards.return_value = {"forwards": []}
    rpc.listnodes.return_value = {"nodes": []}
    rpc.listhtlcs.return_value = {"htlcs": []}
    rpc.getchaininfo.return_value = {"blockcount": 100}
    rpc.feerates.return_value = {"perkb": {"opening": 1000}}
    rpc.bkpr_listincome.return_value = {"income_events": []}
    rpc.bkpr_listbalances.return_value = {"accounts": []}
    rpc.bkpr_channelsapy.return_value = {"channels_apy": []}
    rpc.listdatastore.return_value = {"datastore": []}
    rpc.datastore.return_value = {}
    rpc.deldatastore.return_value = {}
    return rpc


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
class StressHarness:
    def __init__(self, args):
        self.args = args
        self.rng_seed = args.seed
        self.budget_sats = args.budget

        self.stop = threading.Event()
        self.progress = 0
        self._progress_lock = threading.Lock()

        # Captured thread exceptions (from threading.excepthook + inline).
        self.thread_exceptions = []
        self._exc_lock = threading.Lock()

        # Budget invariant.
        self.budget_violations = []
        self.max_reserved_observed = 0
        self.max_overshoot = 0
        self._violation_captured = threading.Event()

        # Deadlock detection.
        self.deadlock = False
        self.last_progress_ts = time.monotonic()

        # Bounded op journal for interleaving capture on first violation.
        self.op_log = deque(maxlen=400)
        self._oplog_lock = threading.Lock()

        # Loop wiring status.
        self.loop_mode = {}  # name -> "real" | "sim" | "error:<msg>"

        self.mod = None
        self.db_path = None

    # -- progress / journaling ------------------------------------------
    def bump(self, n=1):
        with self._progress_lock:
            self.progress += n
            self.last_progress_ts = time.monotonic()

    def journal(self, thread, action, amount=None, category=None, result=None):
        with self._oplog_lock:
            self.op_log.append({
                "t": round(time.monotonic(), 6),
                "thread": thread,
                "action": action,
                "amount": amount,
                "category": category,
                "result": result,
            })

    def record_exception(self, where, exc):
        with self._exc_lock:
            self.thread_exceptions.append({
                "where": where,
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            })

    # -- setup ----------------------------------------------------------
    def setup(self):
        random.seed(self.rng_seed)
        mod = load_plugin_module()

        fd, self.db_path = tempfile.mkstemp(suffix="-stress.db")
        os.close(fd)

        rpc = _make_mock_rpc()
        safe_plugin = _SafePluginMock(rpc)

        Config = mod.Config
        Database = mod.Database

        cfg = Config(
            paused=False,
            daily_budget_sats=self.budget_sats,
            weekly_budget_sats=self.budget_sats * 7,
            min_fee_ppm=25,
            max_fee_ppm=1800,
            db_path=self.db_path,
            enable_reputation=True,
        )
        db = Database(self.db_path, safe_plugin)
        db.initialize()  # create the 33-table schema on the main thread

        # Wire module globals exactly like a subset of init() so the real RPC
        # handlers and loop bodies resolve their dependencies.
        mod.config = cfg
        mod.database = db
        mod.safe_plugin = safe_plugin
        mod.plugin = safe_plugin  # handlers ignore the passed plugin and use globals
        mod.boltz_manager = None
        mod.data_service = None
        mod.hive_hints = None
        mod.hive_router = None

        self.mod = mod
        self.safe_plugin = safe_plugin
        self.rpc = rpc

        # Confirm production journal mode / autocommit on the worker connection.
        conn = db._get_connection()
        jm = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        self.journal_mode = jm
        self.isolation_level = conn.isolation_level  # None == autocommit
        if str(jm).lower() != "wal":
            raise RuntimeError(f"expected WAL journal mode, got {jm!r}")
        if conn.isolation_level is not None:
            raise RuntimeError(f"expected autocommit (isolation_level=None), got {conn.isolation_level!r}")

        self._wire_loops(cfg, db, safe_plugin)

    def _wire_loops(self, cfg, db, safe_plugin):
        """Best-effort: construct real fee/rebalance/flow modules for the loops."""
        mod = self.mod
        # Flow analyzer.
        try:
            mod.flow_analyzer = mod.FlowAnalyzer(safe_plugin, cfg, db)
            self._probe_loop("flow", mod.run_flow_analysis)
        except Exception as exc:  # pragma: no cover - defensive wiring
            mod.flow_analyzer = None
            self.loop_mode["flow"] = f"error:{exc}"

        # Profitability + policy needed by fee_controller / rebalancer.
        try:
            pa = mod.ChannelProfitabilityAnalyzer(safe_plugin, cfg, db)
            pm = mod.PolicyManager(db, safe_plugin)
            mod.profitability_analyzer = pa
            mod.policy_manager = pm
            mod.fee_controller = mod.FeeController(safe_plugin, cfg, db, pm, pa)
            self._probe_loop("fee", mod.run_fee_adjustment)
        except Exception as exc:
            mod.fee_controller = None
            self.loop_mode["fee"] = f"error:{exc}"

        try:
            reb = mod.EVRebalancer(
                safe_plugin, cfg, db,
                getattr(mod, "policy_manager", None),
            )
            if hasattr(reb, "set_profitability_analyzer") and getattr(mod, "profitability_analyzer", None):
                reb.set_profitability_analyzer(mod.profitability_analyzer)
            # Faithful cross-category budget wiring (as init() does).
            reb.external_liquidity_cost_provider = mod._non_rebalance_liquidity_cost_components
            reb.global_budget_limit_provider = mod._total_cost_budget_limit_provider
            mod.rebalancer = reb
            self._probe_loop("rebalance", mod.run_rebalance_check)
        except Exception as exc:
            mod.rebalancer = None
            self.loop_mode["rebalance"] = f"error:{exc}"

    def _probe_loop(self, name, fn):
        """Run a loop body once; mark real if it survives, else sim fallback."""
        try:
            fn()
            self.loop_mode[name] = "real"
        except Exception as exc:
            self.loop_mode[name] = f"sim (probe raised {type(exc).__name__}: {exc})"

    # -- workers --------------------------------------------------------
    def _since_24h(self):
        return int(time.time()) - 24 * 3600

    def worker_generic_ledger(self, wid):
        """Firehose: revenue-spend-reserve in several categories + release/settle."""
        mod = self.mod
        rng = random.Random(self.rng_seed * 1000 + wid)
        categories = ["channel_open", "boltz", "misc_liquidity", "ledger"]
        counter = 0
        live = {}  # rid -> amount
        while not self.stop.is_set():
            counter += 1
            roll = rng.random()
            try:
                if roll < 0.7 or not live:
                    rid = f"gen-{wid}-{counter}"
                    amt = rng.randint(200, max(300, self.budget_sats // 3))
                    cat = rng.choice(categories)
                    res = mod.revenue_spend_reserve(
                        mod.plugin,
                        reservation_id=rid,
                        category=cat,
                        amount_sats=amt,
                    )
                    status = res.get("status") if isinstance(res, dict) else "?"
                    if status == "success":
                        live[rid] = amt
                    self.journal(f"gen{wid}", "spend_reserve", amt, cat, status)
                elif roll < 0.85:
                    rid = next(iter(live))
                    live.pop(rid, None)
                    res = mod.revenue_spend_release(mod.plugin, reservation_id=rid)
                    self.journal(f"gen{wid}", "spend_release", None, None,
                                 res.get("status") if isinstance(res, dict) else "?")
                else:
                    rid = next(iter(live))
                    live.pop(rid, None)
                    res = mod.revenue_spend_settle(mod.plugin, reservation_id=rid, record_event=False)
                    self.journal(f"gen{wid}", "spend_settle", None, None,
                                 res.get("status") if isinstance(res, dict) else "?")
            except Exception as exc:
                self.record_exception(f"generic_ledger[{wid}]", exc)
            self.bump()

    def worker_rebalance_reserve(self, wid):
        """Firehose: rebalance-category reservations via the OTHER budget table.

        Uses the same budget_limit provider init() wires
        (``_total_cost_budget_limit_provider`` -> memoized total-cost status),
        exactly reproducing the P1-003 cross-category reserve path.
        """
        mod = self.mod
        db = mod.database
        rng = random.Random(self.rng_seed * 2000 + wid)
        counter = 0
        live = deque()
        while not self.stop.is_set():
            counter += 1
            roll = rng.random()
            try:
                if roll < 0.75 or not live:
                    rid = f"reb-{wid}-{counter}"
                    amt = rng.randint(200, max(300, self.budget_sats // 3))
                    try:
                        prov = mod._total_cost_budget_limit_provider()
                        budget_limit = int(prov.get("effective_budget_sats", self.budget_sats) or 0)
                    except Exception:
                        budget_limit = self.budget_sats
                    ok, remaining = db.reserve_budget(
                        reservation_id=rid,
                        amount_sats=amt,
                        channel_id=f"{wid}x{counter}x0",
                        budget_limit=budget_limit,
                        since_timestamp=self._since_24h(),
                    )
                    if ok:
                        live.append(rid)
                    self.journal(f"reb{wid}", "reserve_budget", amt, "rebalance",
                                 f"ok={ok},limit={budget_limit}")
                elif roll < 0.9:
                    rid = live.popleft()
                    ok = db.release_budget_reservation(rid)
                    self.journal(f"reb{wid}", "release_budget", None, "rebalance", f"ok={ok}")
                else:
                    rid = live.popleft()
                    ok = db.mark_budget_spent(rid, 0)
                    self.journal(f"reb{wid}", "mark_spent", None, "rebalance", f"ok={ok}")
            except Exception as exc:
                self.record_exception(f"rebalance_reserve[{wid}]", exc)
            self.bump()

    def worker_boltz_reserve(self, wid):
        """Firehose: Boltz-swap reservations via the P4-014 atomic path.

        Mirrors modules.capex_budget.reserve_boltz_swap_budget — a
        category="boltz" reserve_spend against the LIVE unified budget inside
        BEGIN IMMEDIATE. This is the boltz leg of DD1's cross-category rail: a
        boltz reservation must be counted by (and count) the rebalance and
        generic reservations, so the budget-never-exceeded invariant now covers
        boltz too. reserve + release only (the settle-into-spend_event path is
        unit-tested separately) keeps the boltz commitment churning against the
        shared budget for the whole soak.
        """
        mod = self.mod
        db = mod.database
        rng = random.Random(self.rng_seed * 4000 + wid)
        counter = 0
        live = deque()
        while not self.stop.is_set():
            counter += 1
            roll = rng.random()
            try:
                if roll < 0.75 or not live:
                    rid = f"boltz-swap:{wid}-{counter}"
                    amt = rng.randint(200, max(300, self.budget_sats // 3))
                    try:
                        prov = mod._total_cost_budget_limit_provider()
                        eff = int(prov.get("effective_budget_sats", self.budget_sats) or 0)
                    except Exception:
                        eff = self.budget_sats
                    ok = db.reserve_spend(
                        reservation_id=rid,
                        amount_sats=amt,
                        category="boltz",
                        subcategory="swap_fee",
                        effective_budget_sats=eff,
                        since_timestamp=self._since_24h(),
                    )
                    if ok:
                        live.append(rid)
                    self.journal(f"boltz{wid}", "boltz_reserve", amt, "boltz", f"ok={ok},eff={eff}")
                else:
                    rid = live.popleft()
                    ok = db.release_spend_reservation(rid)
                    self.journal(f"boltz{wid}", "boltz_release", None, "boltz", f"ok={ok}")
            except Exception as exc:
                self.record_exception(f"boltz_reserve[{wid}]", exc)
            self.bump()

    def worker_chainswap_reserve(self, wid):
        """Firehose: chainswap reservations via the P4-023 atomic path.

        chainswap (createchainswap) is the 6th swap-create; after P4-023 it
        reserves its estimated swap fee through the SAME boltz rail as loop_in /
        loop_out — category="boltz" reserve_spend against the LIVE unified
        budget inside _reserve_budget_atomic's BEGIN IMMEDIATE (via
        capex_budget.reserve_boltz_swap_budget, the path
        boltz_manager._open_swap_budget_reservation drives). This is the SIXTH
        reserver on DD1's cross-category rail: a chainswap reservation must be
        counted by (and count) the rebalance, generic, boltz, planner, and
        defibrillation reservations, so budget-never-exceeded now covers all six
        autonomous spend paths. reserve + release churns the chainswap
        commitment against the shared budget.
        """
        mod = self.mod
        db = mod.database
        rng = random.Random(self.rng_seed * 7000 + wid)
        counter = 0
        live = deque()
        while not self.stop.is_set():
            counter += 1
            roll = rng.random()
            try:
                if roll < 0.75 or not live:
                    rid = f"boltz-swap:chain-{wid}-{counter}"
                    amt = rng.randint(200, max(300, self.budget_sats // 3))
                    try:
                        prov = mod._total_cost_budget_limit_provider()
                        eff = int(prov.get("effective_budget_sats", self.budget_sats) or 0)
                    except Exception:
                        eff = self.budget_sats
                    ok = db.reserve_spend(
                        reservation_id=rid,
                        amount_sats=amt,
                        category="boltz",
                        subcategory="swap_fee",
                        effective_budget_sats=eff,
                        since_timestamp=self._since_24h(),
                    )
                    if ok:
                        live.append(rid)
                    self.journal(f"chain{wid}", "chainswap_reserve", amt, "boltz",
                                 f"ok={ok},eff={eff}")
                else:
                    rid = live.popleft()
                    ok = db.release_spend_reservation(rid)
                    self.journal(f"chain{wid}", "chainswap_release", None, "boltz", f"ok={ok}")
            except Exception as exc:
                self.record_exception(f"chainswap_reserve[{wid}]", exc)
            self.bump()

    def worker_planner_open_reserve(self, wid):
        """Firehose: capacity-planner channel_open reservations via the P4-018
        atomic path.

        Mirrors modules.capacity_planner.CapacityPlanner._execute_open — a
        category="channel_open" reserve_spend against the LIVE unified budget
        (effective_budget_sats passed) inside BEGIN IMMEDIATE, then settle
        (mark spent + spend event) on "success" or release on "failure". This is
        the fourth (planner) leg of DD1's cross-category rail: a planner open
        reservation must be counted by (and count) the rebalance, generic, and
        boltz reservations, so budget-never-exceeded now covers ALL FOUR
        autonomous daemon spend paths.
        """
        mod = self.mod
        db = mod.database
        rng = random.Random(self.rng_seed * 5000 + wid)
        counter = 0
        live = deque()
        while not self.stop.is_set():
            counter += 1
            roll = rng.random()
            try:
                if roll < 0.7 or not live:
                    rid = f"planner-open-{wid}-{counter}"
                    amt = rng.randint(200, max(300, self.budget_sats // 3))
                    try:
                        prov = mod._total_cost_budget_limit_provider()
                        eff = int(prov.get("effective_budget_sats", self.budget_sats) or 0)
                    except Exception:
                        eff = self.budget_sats
                    ok = db.reserve_spend(
                        reservation_id=rid,
                        amount_sats=amt,
                        category="channel_open",
                        subcategory="automated",
                        effective_budget_sats=eff,
                        since_timestamp=self._since_24h(),
                    )
                    if ok:
                        live.append(rid)
                    self.journal(f"planner{wid}", "planner_open_reserve", amt,
                                 "channel_open", f"ok={ok},eff={eff}")
                elif roll < 0.85:
                    rid = live.popleft()
                    ok = db.mark_spend_reservation_spent(
                        reservation_id=rid, actual_spent_sats=None,
                        source="capacity_planner", record_event=True,
                    )
                    self.journal(f"planner{wid}", "planner_open_settle", None,
                                 "channel_open", f"ok={ok}")
                else:
                    rid = live.popleft()
                    ok = db.release_spend_reservation(rid)
                    self.journal(f"planner{wid}", "planner_open_release", None,
                                 "channel_open", f"ok={ok}")
            except Exception as exc:
                self.record_exception(f"planner_open_reserve[{wid}]", exc)
            self.bump()

    def worker_defib_reserve(self, wid):
        """Firehose: defibrillation diagnostic-shock reservations via the P4-020
        atomic path.

        Mirrors rebalancer.diagnostic_rebalance's active shock after P4-020: the
        shock reserves its D4 fee cap through the SAME rail as the auto cycle —
        db.reserve_budget (category=rebalance, budget_reservations) inside
        _reserve_budget_atomic's BEGIN IMMEDIATE against the live unified budget.
        This is the FIFTH (defibrillation) leg of DD1's cross-category rail: a
        diagnostic shock's reservation must be counted by (and count) the
        rebalance, generic, boltz, and planner reservations, so
        budget-never-exceeded now covers ALL FIVE autonomous daemon spend paths.
        The shock fee is small (D4-capped), so reserves stay bounded; reserve +
        mark_spent/release churns the shock commitment against the shared budget.
        """
        mod = self.mod
        db = mod.database
        rng = random.Random(self.rng_seed * 6000 + wid)
        counter = 0
        live = deque()
        while not self.stop.is_set():
            counter += 1
            roll = rng.random()
            try:
                if roll < 0.7 or not live:
                    rid = f"defib-{wid}-{counter}"
                    # D4 diagnostic fee cap envelope (bounded, 1/cycle in prod).
                    amt = rng.randint(50, 400)
                    try:
                        prov = mod._total_cost_budget_limit_provider()
                        budget_limit = int(prov.get("effective_budget_sats", self.budget_sats) or 0)
                    except Exception:
                        budget_limit = self.budget_sats
                    ok, remaining = db.reserve_budget(
                        reservation_id=rid,
                        amount_sats=amt,
                        channel_id=f"defib{wid}x{counter}x0",
                        budget_limit=budget_limit,
                        since_timestamp=self._since_24h(),
                    )
                    if ok:
                        live.append(rid)
                    self.journal(f"defib{wid}", "defib_reserve", amt, "rebalance",
                                 f"ok={ok},limit={budget_limit}")
                elif roll < 0.85:
                    rid = live.popleft()
                    ok = db.mark_budget_spent(rid, 0)
                    self.journal(f"defib{wid}", "defib_mark_spent", None, "rebalance", f"ok={ok}")
                else:
                    rid = live.popleft()
                    ok = db.release_budget_reservation(rid)
                    self.journal(f"defib{wid}", "defib_release", None, "rebalance", f"ok={ok}")
            except Exception as exc:
                self.record_exception(f"defib_reserve[{wid}]", exc)
            self.bump()

    def worker_config_reload(self, wid):
        """setconfig churn on non-budget public keys (config._lock / _version)."""
        mod = self.mod
        rng = random.Random(self.rng_seed * 3000 + wid)
        # daily_budget_sats is intentionally NOT reloaded so the budget invariant
        # threshold stays fixed and interpretable.
        knobs = [
            ("min_fee_ppm", lambda: str(rng.randint(1, 100))),
            ("max_fee_ppm", lambda: str(rng.randint(600, 2000))),
            ("paused", lambda: rng.choice(["true", "false"])),
        ]
        while not self.stop.is_set():
            key, valfn = rng.choice(knobs)
            try:
                res = mod.revenue_config(mod.plugin, action="set", key=key, value=valfn())
                self.journal(f"cfg{wid}", "config_set", None, key,
                             res.get("status") if isinstance(res, dict) else "?")
            except Exception as exc:
                self.record_exception(f"config_reload[{wid}]", exc)
            self.bump()
            time.sleep(0.002)

    def _sim_loop(self, name):
        """DB-touching fallback for a loop that could not be wired real."""
        db = self.mod.database
        cfg = self.mod.config
        try:
            if name == "flow":
                db.get_all_channel_states()
                if hasattr(db, "decay_reputation"):
                    db.decay_reputation(getattr(cfg, "reputation_decay", 0.99) or 0.99)
            elif name == "fee":
                db.get_recent_fee_changes(limit=10)
                cfg.snapshot()
                self.mod._total_cost_budget_status()
            elif name == "rebalance":
                db.get_daily_rebalance_spend(window_hours=24)
                db.cleanup_stale_reservations()
        except Exception as exc:
            self.record_exception(f"sim_loop[{name}]", exc)

    def worker_timer_loop(self, name, fn, interval):
        """Shortened timer loop: repeatedly invoke the (real or sim) loop body."""
        mode = self.loop_mode.get(name, "sim")
        use_real = mode == "real"
        while not self.stop.is_set():
            try:
                if use_real:
                    fn()
                else:
                    self._sim_loop(name)
            except Exception as exc:
                # Faithful to production: loop bodies re-raise; capture as the
                # thread-exception the daemon would suffer, then keep looping.
                self.record_exception(f"timer_loop[{name}]", exc)
            self.bump()
            time.sleep(interval)

    def worker_invariant_sampler(self):
        """Ground-truth budget check, independent of the (memoized) gate."""
        db = self.mod.database
        conn = db._get_connection()  # sampler's own thread-local connection
        while not self.stop.is_set():
            try:
                # Read BOTH budget tables in ONE SELECT so the sample is a single
                # consistent snapshot. Two separate autocommit SELECTs are a torn
                # read: they can capture spend_reservations from one instant and
                # budget_reservations from a later instant while reservations
                # churn, producing a phantom sum > budget even though the atomic
                # rail (BEGIN IMMEDIATE) keeps every COMMITTED instant <= budget.
                row = conn.execute(
                    "SELECT "
                    "(SELECT COALESCE(SUM(reserved_sats),0) FROM spend_reservations WHERE status='active'), "
                    "(SELECT COALESCE(SUM(reserved_sats),0) FROM budget_reservations WHERE status='active')"
                ).fetchone()
                gen = row[0]
                reb = row[1]
                total = int(gen or 0) + int(reb or 0)
                budget = int(getattr(self.mod.config, "daily_budget_sats", self.budget_sats) or 0)
                if total > self.max_reserved_observed:
                    self.max_reserved_observed = total
                if total > budget:
                    overshoot = total - budget
                    if overshoot > self.max_overshoot:
                        self.max_overshoot = overshoot
                    if not self._violation_captured.is_set():
                        with self._oplog_lock:
                            interleaving = list(self.op_log)
                        self.budget_violations.append({
                            "at_monotonic": round(time.monotonic(), 6),
                            "generic_reserved_sats": int(gen or 0),
                            "rebalance_reserved_sats": int(reb or 0),
                            "total_reserved_sats": total,
                            "effective_budget_sats": budget,
                            "overshoot_sats": overshoot,
                            "seed": self.rng_seed,
                            "interleaving_tail": interleaving[-40:],
                        })
                        self._violation_captured.set()
            except Exception as exc:
                self.record_exception("invariant_sampler", exc)
            self.bump()
            time.sleep(0.001)

    def worker_watchdog(self):
        timeout = self.args.watchdog
        while not self.stop.is_set():
            time.sleep(0.25)
            with self._progress_lock:
                stalled = time.monotonic() - self.last_progress_ts
            if stalled > timeout:
                self.deadlock = True
                self.stop.set()
                return

    # -- run ------------------------------------------------------------
    def run(self):
        self.setup()

        # threading.excepthook: capture anything that escapes a thread's own
        # try/except (there should be none, but this is the safety net the task
        # requires).
        orig_hook = threading.excepthook

        def hook(arg):
            with self._exc_lock:
                self.thread_exceptions.append({
                    "where": f"excepthook:{getattr(arg.thread, 'name', '?')}",
                    "type": arg.exc_type.__name__ if arg.exc_type else "?",
                    "message": str(arg.exc_value),
                    "traceback": "".join(
                        traceback.format_exception(arg.exc_type, arg.exc_value, arg.exc_traceback)
                    ),
                })
            if orig_hook:
                orig_hook(arg)

        threading.excepthook = hook

        threads = []

        def spawn(target, name, *a):
            t = threading.Thread(target=target, args=a, name=name, daemon=True)
            t.start()
            threads.append(t)

        n = self.args.threads
        for i in range(n):
            spawn(self.worker_generic_ledger, f"generic-{i}", i)
        for i in range(n):
            spawn(self.worker_rebalance_reserve, f"rebalance-{i}", i)
        for i in range(n):
            spawn(self.worker_boltz_reserve, f"boltz-{i}", i)
        for i in range(n):
            spawn(self.worker_chainswap_reserve, f"chainswap-{i}", i)
        for i in range(n):
            spawn(self.worker_planner_open_reserve, f"planner-{i}", i)
        for i in range(n):
            spawn(self.worker_defib_reserve, f"defib-{i}", i)
        for i in range(max(1, n // 2)):
            spawn(self.worker_config_reload, f"config-{i}", i)

        spawn(self.worker_timer_loop, "loop-fee", "fee", self.mod.run_fee_adjustment, 0.05)
        spawn(self.worker_timer_loop, "loop-rebalance", "rebalance", self.mod.run_rebalance_check, 0.05)
        spawn(self.worker_timer_loop, "loop-flow", "flow", self.mod.run_flow_analysis, 0.05)

        spawn(self.worker_invariant_sampler, "invariant-sampler")
        spawn(self.worker_watchdog, "watchdog")

        deadline = time.monotonic() + self.args.soak
        try:
            while time.monotonic() < deadline and not self.stop.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop.set()

        for t in threads:
            t.join(timeout=10)

        threading.excepthook = orig_hook

        alive = [t.name for t in threads if t.is_alive()]

        return self._finalize(alive)

    def _finalize(self, alive):
        # integrity check on a fresh connection.
        integrity = "unknown"
        try:
            ic = sqlite3.connect(self.db_path)
            integrity = ic.execute("PRAGMA integrity_check;").fetchone()[0]
            ic.close()
        except Exception as exc:
            integrity = f"error:{exc}"

        results = {
            "integrity": integrity == "ok",
            "no_deadlock": not self.deadlock,
            "no_thread_exception": len(self.thread_exceptions) == 0,
            "budget_never_exceeded": len(self.budget_violations) == 0,
        }
        report = {
            "config": {
                "soak_seconds": self.args.soak,
                "threads_per_pool": self.args.threads,
                "budget_sats": self.budget_sats,
                "seed": self.rng_seed,
                "watchdog_seconds": self.args.watchdog,
            },
            "substrate": {
                "journal_mode": getattr(self, "journal_mode", None),
                "isolation_level": getattr(self, "isolation_level", "unset"),
                "db_path": self.db_path,
                "loop_mode": self.loop_mode,
            },
            "progress_ops": self.progress,
            "threads_still_alive": alive,
            "integrity_check": integrity,
            "max_reserved_observed_sats": self.max_reserved_observed,
            "max_overshoot_sats": self.max_overshoot,
            "budget_violations": self.budget_violations,
            "thread_exceptions": self.thread_exceptions,
            "results": results,
        }

        # temp DB cleanup.
        try:
            for suffix in ("", "-wal", "-shm"):
                p = self.db_path + suffix
                if os.path.exists(p):
                    os.unlink(p)
        except OSError:
            pass

        return report


def build_arg_parser():
    p = argparse.ArgumentParser(description="cl_revenue_ops concurrency stress harness")
    p.add_argument("--soak", type=int, default=60, help="soak duration seconds (default 60; use 1800 for the 30-min run)")
    p.add_argument("--threads", type=int, default=6, help="worker threads per firehose pool")
    p.add_argument("--budget", type=int, default=10000, help="daily_budget_sats (the budget invariant threshold)")
    p.add_argument("--seed", type=int, default=1337, help="RNG seed for deterministic interleavings")
    p.add_argument("--watchdog", type=float, default=15.0, help="deadlock watchdog stall threshold seconds")
    p.add_argument("--json", type=str, default=None, help="write full JSON report to this path")
    p.add_argument("--allow-known-budget-toctou", action="store_true",
                   help="treat a budget-invariant violation as a documented known finding (P1-003) and still exit 0 if structural invariants hold")
    return p


def _print_report(report, allow_known_budget):
    r = report["results"]
    print("=" * 72)
    print("cl_revenue_ops concurrency stress harness")
    print("=" * 72)
    cfg = report["config"]
    sub = report["substrate"]
    print(f"soak={cfg['soak_seconds']}s threads/pool={cfg['threads_per_pool']} "
          f"budget={cfg['budget_sats']} seed={cfg['seed']}")
    print(f"journal_mode={sub['journal_mode']} isolation_level={sub['isolation_level']!r}")
    print(f"loop_mode={sub['loop_mode']}")
    print(f"progress_ops={report['progress_ops']} "
          f"max_reserved_observed={report['max_reserved_observed_sats']} "
          f"max_overshoot={report['max_overshoot_sats']}")
    print("-" * 72)

    def line(label, ok):
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")

    line("integrity_check == ok", r["integrity"])
    line("no deadlock (watchdog)", r["no_deadlock"])
    line("no unhandled thread exception", r["no_thread_exception"])
    line("budget never exceeded (P1-003)", r["budget_never_exceeded"])
    print("-" * 72)

    if report["budget_violations"]:
        v = report["budget_violations"][0]
        print("P1-003 REPRODUCED — cross-category budget overspend under stress:")
        print(f"  generic(spend_reservations)={v['generic_reserved_sats']} + "
              f"rebalance(budget_reservations)={v['rebalance_reserved_sats']} "
              f"= {v['total_reserved_sats']} > effective_budget={v['effective_budget_sats']} "
              f"(overshoot {v['overshoot_sats']} sats, seed {v['seed']})")
    else:
        print("P1-003 NOT reproduced in this run (budget invariant held) — latent-only.")

    if report["thread_exceptions"]:
        print(f"thread exceptions captured: {len(report['thread_exceptions'])}")
        print(f"  first: {report['thread_exceptions'][0]['where']}: "
              f"{report['thread_exceptions'][0]['type']}: {report['thread_exceptions'][0]['message']}")
    print("=" * 72)

    structural_ok = r["integrity"] and r["no_deadlock"] and r["no_thread_exception"]
    if allow_known_budget:
        return 0 if structural_ok else 1
    return 0 if all(r.values()) else 1


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    harness = StressHarness(args)
    report = harness.run()
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
    code = _print_report(report, args.allow_known_budget_toctou)
    return code


if __name__ == "__main__":
    sys.exit(main())
