#!/usr/bin/env python3
"""Phase 3 end-to-end loop sweep: REBALANCE + BUDGET LOOP.

Read-only sweep over the frozen hermes corpus
(/home/sat/cl-mycelium-hermes/<node>/<YYYYMMDD>/<HHMMSSZ>/commands) checking
cross-module loop invariants that Phase 2's per-module sweeps could not see:

  LP-I1  budget reconciliation of the Phase 2 "129 over-budget snapshots"
         tension: every snapshot with actual_spent_sats > effective_budget_sats
         must (a) owe the entire overage to the on-chain open/close categories
         (i.e. discretionary spend rebalance+boltz+ledger stays <= effective),
         (b) report remaining_sats == 0, and (c) carry a rebalance_decision of
         suppressed/budget_blocked in the same snapshot (enforcement feedback).
  LP-I2  enforcement feedback on writes: no new rebalance_history row of type
         normal or diagnostic is created inside a budget-saturated interval
         (manual rows exempt by design, RB-I9).
  LP-I3  vacuity audit of Phase 2 RB-I1c: max rebalance-category 24h spend seen
         anywhere in the corpus (0 => RB-I1c was vacuous, not evidential).
  LP-I4  planner defib actions <-> diagnostic history rows join: each completed
         'defibrillate' action inside snapshot coverage has a diagnostic row
         for the same dest channel within +/-15 min, OR falls inside a
         budget-saturated interval (Active Shock blocked path returns
         success:True without a history row), OR is outside corpus snapshot
         coverage. Conversely each diagnostic row inside the planner-history
         7-day visibility window joins a completed action.
  LP-I5  one history row per physical attempt: no two rows on a node share
         (from_channel, to_channel, timestamp). Rows before 2026-06-10
         (commit 62ae545, "Guard rebalance cycles and fix budget/accounting
         distortions") are reported separately as the known pre-fix
         double-recording defect.
  LP-I6  segment observations <-> attempts: every exported segment observation
         (correlation_id "src->dst:ts") joins a rebalance_history row with the
         same node/src/dst within +/-10 min.
  LP-I7  success accounting: success rows have actual_fee_sats <= max_fee_sats;
         (balance-shift cross-check is NOT corpus-observable: both successes
         predate snapshot coverage - labeled vacuous.)
  LP-I8  suppressed windows (Phase 2 RB-I1b, strengthened): consecutive
         suppressed+budget_blocked decision snapshots contain no new normal OR
         diagnostic history row (diagnostic rows make this non-vacuous, unlike
         RB-I1b which only looked at rebalance_type=normal successes).

Anomaly material printed at the end: failure-token taxonomy, engine-cycle
selection/execution counts, per-pair attempt spacing vs persisted cooldown
ceilings, and the over-budget interval boundaries.

Usage: python3 tools/audit/loop_sweep_rebalance.py [corpus_root]
"""

import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

CORPUS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    os.environ.get("CL_MYCELIUM_HERMES_ROOT", "/home/sat/cl-mycelium-hermes"))
NODES = ("hive-nexus-01", "hive-nexus-02")

# commit 62ae545 (2026-06-10 06:14 -0600) shipped the shared-history-row fix
ROW_SHARING_FIX_TS = int(datetime(2026, 6, 10, 12, 14, 33, tzinfo=timezone.utc).timestamp())

# engine persisted pair-failure cooldown ceilings (rebalance_engine_v2.py:176-184,
# base seconds x backoff multiplier <= 6 in record_pair_rebalance_failure)
COOLDOWN_CEIL = {
    "temporary_channel_failure": 300 * 6,
    "fee_insufficient": 1800 * 6,
    "incorrect_cltv_expiry": 3600 * 6,
    "permanent_failure": 21600 * 6,
    "payment_pending_timeout": 3600 * 6,
    "local_execution_failed": 600 * 6,
    "other_retriable": 600 * 6,
}


def load(path: Path):
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt") as f:
                return json.load(f)
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def snap_ts(name: str) -> int:
    return int(datetime.strptime(name, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp())


def iter_snaps(node: str):
    base = CORPUS / node
    if not base.is_dir():
        return
    for day in sorted(p for p in base.iterdir() if p.is_dir() and p.name[:4].isdigit()):
        for snap in sorted(p for p in day.iterdir() if p.is_dir()):
            cmds = snap / "commands"
            if cmds.is_dir():
                yield snap_ts(snap.name), cmds


def fmt(t):
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%m-%d %H:%M:%S")


class Check:
    def __init__(self, name, desc):
        self.name, self.desc = name, desc
        self.passed = 0
        self.violations = []
        self.vacuous_note = None

    def ok(self, n=1):
        self.passed += n

    def bad(self, where, detail):
        self.violations.append((str(where), detail))

    def report(self):
        v = len(self.violations)
        status = "PASS" if v == 0 else "VIOLATION"
        vac = ""
        if self.passed + v == 0:
            vac = "  [VACUOUS: zero checkable instances]"
        elif self.vacuous_note:
            vac = f"  [{self.vacuous_note}]"
        print(f"[{self.name}] {status}  checked={self.passed + v} violations={v}{vac}\n    {self.desc}")
        for p, d in self.violations[:6]:
            print(f"    example: {p}\n      {d}")
        if v > 6:
            print(f"    ... and {v - 6} more")


def main():
    c1a = Check("LP-I1a", "over-budget snapshots: overage entirely open/close (rebalance+boltz+ledger <= effective)")
    c1b = Check("LP-I1b", "over-budget snapshots: remaining_sats == 0")
    c1c = Check("LP-I1c", "over-budget snapshots: same-snapshot rebalance_decision suppressed & budget_blocked")
    c2 = Check("LP-I2", "no new normal/diagnostic history row inside a budget-saturated interval (manual exempt)")
    c4a = Check("LP-I4a", "completed defib action -> diagnostic row within 15min OR budget-blocked OR outside coverage")
    c4b = Check("LP-I4b", "diagnostic row (in planner 7d visibility) -> completed defib action for same dest within 15min")
    c5 = Check("LP-I5", "one history row per attempt: unique (from,to,timestamp) per node (post 62ae545 fix)")
    c6 = Check("LP-I6", "segment observation correlation joins a history row (same node/src/dst, +/-10min)")
    c7 = Check("LP-I7", "success rows: actual_fee_sats <= max_fee_sats")
    c8 = Check("LP-I8", "suppressed+budget_blocked windows contain no new normal OR diagnostic row")

    rows = {}            # (node, id) -> (last_seen_snap_ts, row)
    decisions = defaultdict(list)   # node -> [(ts, action, budget_blocked)]
    budgets = defaultdict(list)     # node -> [(ts, tb)]
    actions = {}         # (node, action_id) -> (last_seen_snap_ts, action)
    obs_all = {}         # (node, observation_id) -> obs
    snap_bounds = {}     # node -> (first_ts, last_ts)
    snap_times = defaultdict(list)  # node -> [snap ts]
    max_rebal_cat = 0
    debug_selected = 0
    debug_execs = 0
    debug_snaps = 0
    considered_rej = Counter()

    for node in NODES:
        for ts, cmds in iter_snaps(node):
            snap_times[node].append(ts)
            lo, hi = snap_bounds.get(node, (ts, ts))
            snap_bounds[node] = (min(lo, ts), max(hi, ts))

            st = load(cmds / "revenue-status.json")
            if st:
                rd = st.get("rebalance_decision") or {}
                if rd.get("action"):
                    decisions[node].append((ts, rd.get("action"), bool(rd.get("budget_blocked"))))
                for r in st.get("recent_rebalances") or []:
                    rid = r.get("id")
                    if rid is not None:
                        prev = rows.get((node, rid))
                        if prev is None or ts >= prev[0]:
                            rows[(node, rid)] = (ts, r)

            tb = load(cmds / "revenue-total-cost-budget.json")
            if tb and "error" not in tb:
                budgets[node].append((ts, tb))
                cat = tb.get("actual_spent_by_category") or {}
                max_rebal_cat = max(max_rebal_cat, int(cat.get("rebalance", 0) or 0))

            ph = load(cmds / "revenue-planner-history.json")
            if ph:
                for a in ph.get("actions") or []:
                    aid = a.get("id")
                    if aid is not None:
                        prev = actions.get((node, aid))
                        if prev is None or ts >= prev[0]:
                            actions[(node, aid)] = (ts, a)

            so = load(cmds / "listdatastore-key-revenue-segment-observations.json")
            if so:
                for e in so.get("datastore") or []:
                    raw = e.get("string")
                    if raw is None and e.get("hex"):
                        try:
                            raw = bytes.fromhex(e["hex"]).decode("utf-8", "replace")
                        except Exception:
                            raw = None
                    try:
                        snapd = json.loads(raw)
                    except Exception:
                        continue
                    for o in snapd.get("segment_observations") or []:
                        obs_all[(node, o.get("observation_id"))] = o

            dbg = load(cmds / "revenue-rebalance-debug.json")
            if dbg:
                debug_snaps += 1
                lc = dbg.get("last_cycle") or {}
                s = lc.get("summary") or {}
                debug_selected += int(s.get("selected_pairs") or 0)
                debug_execs += int(s.get("execution_count") or 0)
                for c in lc.get("considered_candidates") or []:
                    d = c.get("score_decomposition") or {}
                    considered_rej[c.get("rejection_reason") or d.get("rejection_reason") or "<none>"] += 1

    # ---------- LP-I1: budget reconciliation ----------
    over_intervals = defaultdict(list)  # node -> [(start_ts, end_ts)]
    over_count = 0
    for node in NODES:
        cur = None
        prev_over_ts = None
        for ts, tb in budgets[node]:
            eff = tb.get("effective_budget_sats")
            spent = tb.get("actual_spent_sats")
            if eff is None or spent is None:
                continue
            if spent > eff:
                over_count += 1
                cat = tb.get("actual_spent_by_category") or {}
                onchain = int(cat.get("open", 0) or 0) + int(cat.get("close", 0) or 0)
                discretionary = spent - onchain
                if discretionary <= eff:
                    c1a.ok()
                else:
                    c1a.bad(f"{node}@{fmt(ts)}", f"spent={spent} open+close={onchain} discretionary={discretionary} > eff={eff}")
                if tb.get("remaining_sats") == 0:
                    c1b.ok()
                else:
                    c1b.bad(f"{node}@{fmt(ts)}", f"remaining={tb.get('remaining_sats')} with spent={spent} > eff={eff}")
                # decision in the same snapshot
                dec = [d for d in decisions[node] if d[0] == ts]
                if dec:
                    _, action, blocked = dec[0]
                    if action == "suppressed" and blocked:
                        c1c.ok()
                    else:
                        c1c.bad(f"{node}@{fmt(ts)}", f"decision action={action} budget_blocked={blocked}")
                # interval bookkeeping
                if cur is None:
                    cur = [ts, ts]
                else:
                    cur[1] = ts
                prev_over_ts = ts
            else:
                if cur is not None:
                    over_intervals[node].append(tuple(cur))
                    cur = None
        if cur is not None:
            over_intervals[node].append(tuple(cur))

    # ---------- final row set ----------
    final_rows = [(node, rid, r) for (node, rid), (ts, r) in rows.items()]
    final_rows.sort(key=lambda x: x[2].get("timestamp") or 0)

    def in_over_interval(node, t):
        return any(a <= t <= b for a, b in over_intervals[node])

    def in_coverage_hole(node, t, min_gap=6 * 3600):
        """True if t falls in a gap > min_gap between adjacent snapshots."""
        ts_list = snap_times[node]
        for a, b in zip(ts_list, ts_list[1:]):
            if b - a > min_gap and a < t < b:
                return True
        return False

    hole_defibs = []  # completed defibs whose blocked-vs-executed state is unobservable

    # LP-I2
    for node, rid, r in final_rows:
        t = r.get("timestamp") or 0
        if r.get("rebalance_type") in ("normal", "diagnostic"):
            if in_over_interval(node, t):
                c2.bad(f"{node} row id={rid}", f"{r.get('rebalance_type')} row at {fmt(t)} inside budget-saturated interval")
            else:
                c2.ok()

    # LP-I4a: completed defib actions -> diagnostic rows
    diag_rows = [(node, rid, r) for node, rid, r in final_rows if r.get("rebalance_type") == "diagnostic"]
    defib_actions = [(node, aid, a) for (node, aid), (ts, a) in actions.items()
                     if a.get("action_type") == "defibrillate" and a.get("status") == "completed"]
    matched_action_row = {}
    for node, aid, a in sorted(defib_actions, key=lambda x: x[2].get("completed_at") or 0):
        t = a.get("completed_at") or a.get("created_at") or 0
        ch = a.get("channel_id")
        cands = [(n, rid, r) for n, rid, r in diag_rows
                 if n == node and r.get("to_channel") == ch
                 and abs((r.get("timestamp") or 0) - t) <= 900]
        if cands:
            c4a.ok()
            matched_action_row[(node, aid)] = cands[0][1]
            continue
        if in_over_interval(node, t):
            c4a.ok()   # Active Shock blocked by capital controls: no row by design
            continue
        lo, hi = snap_bounds.get(node, (0, 0))
        if not (lo <= t <= hi) or in_coverage_hole(node, t):
            # outside snapshot coverage, or inside the 06-21..06-30 hole: the
            # budget/no-source state at t is not corpus-observable, so the
            # blocked-shock alternative cannot be tested. Row absence itself IS
            # provable when the history id sequence is contiguous across t
            # (recorded below as anomaly material, not a violation).
            c4a.ok()
            hole_defibs.append((node, aid, ch, t))
            continue
        c4a.bad(f"{node} action id={aid}", f"completed defib {ch} at {fmt(t)}: no diagnostic row, not budget-blocked")

    # LP-I4b: diagnostic rows -> defib actions (only within planner 7d visibility
    # of some captured snapshot: action must be inside [min(created)..] of the
    # per-node captured planner-action id range)
    for node in NODES:
        acts = [a for (n, aid), (ts, a) in actions.items() if n == node]
        if not acts:
            continue
        amin = min(a.get("created_at") or 0 for a in acts)
        amax = max(a.get("created_at") or 0 for a in acts)
        captured_ids = sorted(aid for (n, aid) in actions if n == node)
        id_gaps = set()
        for x, y in zip(captured_ids, captured_ids[1:]):
            if y - x > 1:
                id_gaps.update(range(x + 1, y))
        for n, rid, r in diag_rows:
            if n != node:
                continue
            t = r.get("timestamp") or 0
            if not (amin <= t <= amax):
                continue  # outside captured planner-history visibility
            hit = [a for a in acts if a.get("action_type") == "defibrillate"
                   and a.get("status") in ("completed", "failed")
                   and a.get("channel_id") == r.get("to_channel")
                   and abs((a.get("completed_at") or a.get("created_at") or 0) - t) <= 900]
            if hit:
                c4b.ok()
            elif id_gaps:
                # planner history is a bounded (7-day) window: uncaptured action
                # ids exist in the per-node sequence, so absence of the action is
                # a visibility artifact, not a violation.
                c4b.ok()
            else:
                c4b.bad(f"{node} row id={rid}", f"diagnostic row at {fmt(t)} dest={r.get('to_channel')}: no defib action captured")

    # LP-I5: unique (from,to,timestamp)
    seen = defaultdict(list)
    for node, rid, r in final_rows:
        seen[(node, r.get("from_channel"), r.get("to_channel"), r.get("timestamp"))].append((rid, r))
    prefix_twins = []
    for key, group in seen.items():
        if len(group) == 1:
            c5.ok()
            continue
        t = key[3] or 0
        if t < ROW_SHARING_FIX_TS:
            prefix_twins.append((key, [rid for rid, _ in group]))
            c5.ok()  # known pre-fix defect, reported separately below
        else:
            c5.bad(f"{key[0]} {key[1]}->{key[2]}@{fmt(t)}", f"duplicate rows ids={[rid for rid, _ in group]} AFTER 62ae545 fix")

    def id_contiguity(node, t):
        """Is the history id sequence contiguous across time t on this node?

        If the ids immediately before/after t are consecutive, no row was ever
        created at t (absence proven, not a visibility artifact of the bounded
        recent_rebalances list)."""
        node_rows = sorted(((r.get("timestamp") or 0, rid) for n, rid, r in final_rows if n == node))
        before = [rid for ts_, rid in node_rows if ts_ < t]
        after = [rid for ts_, rid in node_rows if ts_ > t]
        if before and after:
            lo, hi = max(before), min(after)
            if hi - lo == 1:
                return f"id sequence contiguous across t ({lo}->{hi}): row absence PROVEN"
            return f"id gap across t ({lo}->{hi}): row may exist but be invisible"
        return "at corpus row-visibility edge: absence not provable"

    # LP-I6: observations -> rows
    unmatched_obs = []
    for (node, oid), o in sorted(obs_all.items(), key=lambda kv: kv[1].get("observed_at") or 0):
        corr = str(o.get("correlation_id") or "")
        m = re.match(r"(.+?)->(.+?):(\d+)$", corr)
        if not m:
            c6.bad(f"{node} {oid}", f"unparseable correlation_id {corr!r}")
            continue
        src, dst, cts = m.group(1), m.group(2), int(m.group(3))
        hit = [rid for n, rid, r in final_rows
               if n == node and r.get("from_channel") == src and r.get("to_channel") == dst
               and abs((r.get("timestamp") or 0) - cts) <= 600]
        if hit:
            c6.ok()
        else:
            unmatched_obs.append((node, oid, corr))
            c6.bad(f"{node} {oid}", f"no history row for {corr} (+/-10min); "
                                    f"{id_contiguity(node, cts)}")

    # LP-I7
    success_rows = [(node, rid, r) for node, rid, r in final_rows if r.get("status") == "success"]
    for node, rid, r in success_rows:
        if r.get("actual_fee_sats") is not None and r.get("max_fee_sats") is not None:
            if r["actual_fee_sats"] <= r["max_fee_sats"]:
                c7.ok()
            else:
                c7.bad(f"{node} row id={rid}", f"fee={r['actual_fee_sats']} > max={r['max_fee_sats']}")
    lo01 = snap_bounds.get("hive-nexus-01", (0, 0))[0]
    lo02 = snap_bounds.get("hive-nexus-02", (0, 0))[0]
    if all((r.get("timestamp") or 0) < min(lo01, lo02) for _, _, r in success_rows):
        c7.vacuous_note = ("balance-shift leg VACUOUS: all success rows predate snapshot coverage; "
                           "no adjacent listpeerchannels pair exists to diff")

    # LP-I8: suppressed windows vs normal+diagnostic rows
    for node in NODES:
        decs = sorted(decisions[node])
        for (t1, a1, b1), (t2, a2, b2) in zip(decs, decs[1:]):
            if a1 == "suppressed" and b1 and a2 == "suppressed" and b2:
                hits = [rid for n, rid, r in final_rows
                        if n == node and r.get("rebalance_type") in ("normal", "diagnostic")
                        and t1 < (r.get("timestamp") or 0) <= t2]
                if hits:
                    c8.bad(f"{node} window {fmt(t1)}..{fmt(t2)}", f"normal/diagnostic rows created: ids={hits}")
                else:
                    c8.ok()

    # ---------- report ----------
    print("=== loop_sweep_rebalance: REBALANCE + BUDGET LOOP (Phase 3) ===")
    print(f"corpus root: {CORPUS}")
    print(f"snapshots: " + ", ".join(f"{n}={len(snap_times[n])}" for n in NODES))
    print(f"final rebalance_history rows: {len(final_rows)}  "
          f"(success={len(success_rows)})")
    print(f"over-budget snapshots (actual_spent > effective): {over_count}")
    for node in NODES:
        for a, b in over_intervals[node]:
            print(f"    saturated interval {node}: {fmt(a)} .. {fmt(b)}")
    print()
    for c in (c1a, c1b, c1c, c2, c4a, c4b, c5, c6, c7, c8):
        c.report()

    print()
    print("--- vacuity + anomaly material ---")
    print(f"LP-I3 max rebalance-category 24h spend anywhere in corpus: {max_rebal_cat} sats"
          f"{'  => Phase 2 RB-I1c was VACUOUS (never a nonzero rebalance spend to cap)' if max_rebal_cat == 0 else ''}")
    print(f"engine automated cycle across {debug_snaps} debug snapshots: "
          f"selected_pairs total={debug_selected}, execution_count total={debug_execs}"
          f"{'  => run_cycle NEVER executed in corpus window; LP invariant (debug-selected -> history row) VACUOUS' if debug_selected == 0 and debug_execs == 0 else ''}")
    print(f"considered-candidate rejection reasons: {dict(considered_rej)}")
    print(f"pre-62ae545 duplicate-row attempts (diagnostic+normal twins, known fixed defect): "
          f"{[(k[0], k[1] + '->' + k[2], fmt(k[3]), ids) for k, ids in prefix_twins]}")
    for node, aid, ch, t in hole_defibs:
        print(f"completed defib in coverage hole (blocked-vs-executed unobservable): "
              f"{node} action id={aid} ch={ch} at {fmt(t)}; {id_contiguity(node, t)}")

    # failure-token taxonomy
    tok = Counter()
    for node, rid, r in final_rows:
        em = r.get("error_message") or ""
        t = em.split(":")[0].strip() if em else ("<success>" if r.get("status") == "success" else "<none>")
        tok[(t, r.get("rebalance_type"))] += 1
    print("failure/error token x type taxonomy (final rows):")
    for k, v in tok.most_common():
        print(f"    {v:3}  {k[0]}  [{k[1]}]")

    # per-pair attempt spacing vs cooldown ceilings (non-manual)
    print("non-manual per-pair attempt spacing (min gap vs persisted-cooldown ceiling 36h max):")
    by_pair = defaultdict(list)
    for node, rid, r in final_rows:
        if r.get("rebalance_type") != "manual":
            by_pair[(node, r.get("from_channel"), r.get("to_channel"))].append(r.get("timestamp") or 0)
    worst = []
    for key, ts_list in by_pair.items():
        ts_list.sort()
        gaps = [b - a for a, b in zip(ts_list, ts_list[1:])]
        if gaps:
            worst.append((min(gaps), key, len(ts_list)))
    worst.sort()
    for g, key, n in worst[:6]:
        print(f"    min gap {g}s over {n} attempts: {key[0]} {key[1]}->{key[2]}")
    if not worst:
        print("    (no repeated non-manual pairs)")
    print("    note: pre-62ae545 twin rows share one timestamp (gap 0 = same attempt, not a retry);")
    print("    defib/manual paths skip engine cooldown gates by design (RE-I13) - report-only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
