#!/usr/bin/env python3
"""Read-only hermes corpus sweep for modules/rebalancer.py invariants (RB-I*).

Phase 2 verification campaign. Walks /home/sat/cl-mycelium-hermes per-node
artifact dirs and checks every corpus-observable RB invariant, printing
per-invariant pass/violation counts with example paths on violation.

Sweepable here:
  RB-I1  - budget gating: (a) total-cost-budget internal consistency
           remaining == max(0, effective - spent - reserved);
           (b) suppressed windows (action=suppressed & budget_blocked at both
           endpoints) contain no automated (rebalance_type=normal) success row;
           (c) rebalance-category 24h spend never exceeds the effective budget.
  RB-I4/RB-I9 - accounting cross-check: ledger rebalance spent_24h vs the sum of
           visible success-row fees in the trailing 24h (reported as anomaly
           material, not a strict invariant: recent_rebalances is a bounded list).
  RB-I10 - diagnostic rows bounded: amount_sats <= 50_000, max_fee_sats <= 100.

Engine (RE-I*) checks over revenue-rebalance-debug.json last_cycle decompositions
and revenue-status recent_rebalances (added by Phase 2 verifier; the original
orphaned script was RB-only):
  RE-I2a - accepted/priced candidates: expected_fee_sats (route cost) never
           exceeds effective_budget_sats unless rejected as route_over_budget.
  RE-I2b - success rows: actual_fee_sats <= max_fee_sats.
  RE-I3  - below_hold_margin rejections only on positive-cost routes
           (zero-cost bypass); no selected/executed candidate carries
           final_score_sats < 0 with a positive route cost (margin >= 0).
  RE-I8  - recent_rebalances rows with status=pending_settlement carry a
           payment_hash.
  RE-I9  - last_cycle summary: selected_pairs <= 20 and execution_count <= 20
           (max_concurrent_jobs hard clamp).
  RE-I10 - p_success in [0.05, 0.99]; hive/metabolic/immune biases each in
           [0.85, 1.15].

Not corpus-observable (pure code properties, noted for the verification doc):
  RB-I2 (failure-mode semantics), RB-I3 (reservation ordering), RB-I5 (protected
  limit math), RB-I6 (row uniqueness vs DB), RB-I7 (always-[] return),
  RB-I8 (intent rejection), RB-I11 (internal signal), RB-I12 (liquidity-state
  datastore key is not captured by hermes); RE-I1/I4/I5/I6/I7/I11/I12/I13
  (internal engine mechanics: locks, reservations, futility windows, sweeps).
"""

import gzip
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

CORPUS = Path("/home/sat/cl-mycelium-hermes")
NODES = ["hive-nexus-01", "hive-nexus-02"]


def snap_ts(name: str) -> int:
    return int(datetime.strptime(name, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp())


def load(path: Path):
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt") as f:
                return json.load(f)
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def iter_snapshots(node: str):
    base = CORPUS / node
    for day in sorted(p for p in base.iterdir() if p.is_dir() and p.name[:4].isdigit()):
        for snap in sorted(p for p in day.iterdir() if p.is_dir()):
            cmds = snap / "commands"
            if cmds.is_dir():
                yield snap_ts(snap.name), cmds


class Check:
    def __init__(self, name, desc):
        self.name, self.desc = name, desc
        self.passed = 0
        self.violations = []  # (path, detail)

    def ok(self):
        self.passed += 1

    def bad(self, path, detail):
        self.violations.append((str(path), detail))

    def report(self):
        v = len(self.violations)
        status = "PASS" if v == 0 else "VIOLATION"
        print(f"[{self.name}] {status}  checked={self.passed + v} violations={v}  -- {self.desc}")
        for p, d in self.violations[:5]:
            print(f"    example: {p}\n      {d}")
        if v > 5:
            print(f"    ... and {v - 5} more")


def main():
    c_consist = Check("RB-I1a", "total-cost-budget: remaining == max(0, effective - spent - reserved)")
    c_suppress = Check("RB-I1b", "no automated success rebalance inside suppressed+budget_blocked windows")
    c_capped = Check("RB-I1c", "24h rebalance-category spend <= effective budget")
    c_diag = Check("RB-I10", "diagnostic rebalances bounded: amount<=50000 sats, max_fee<=100 sats")
    c_envelope = Check("RE-I2a", "priced candidates: route cost <= effective budget unless rejected route_over_budget")
    c_fee_cap = Check("RE-I2b", "success rows: actual_fee_sats <= max_fee_sats")
    c_hold = Check("RE-I3", "below_hold_margin only on positive-cost routes; no selected pair with final_score_sats<0 and cost>0")
    c_pending = Check("RE-I8", "pending_settlement rows carry payment_hash")
    c_conc = Check("RE-I9", "selected_pairs <= 20 and execution_count <= 20")
    c_bounds = Check("RE-I10", "p_success in [0.05,0.99]; hive/metabolic/immune biases in [0.85,1.15]")

    def check_decomposition(cand, cmds, selected: bool):
        """RE-I2a / RE-I3 / RE-I10 over one candidate's score_decomposition."""
        decomp = cand.get("score_decomposition") or {}
        inputs = decomp.get("inputs") or {}
        rejection = str(cand.get("rejection_reason") or decomp.get("rejection_reason") or "")
        fee = decomp.get("expected_fee_sats")
        budget = inputs.get("effective_budget_sats")
        pair_key = f"{cand.get('source_channel_id')}->{cand.get('dest_channel_id')}"
        # RE-I2a: a candidate that was NOT rejected for budget must be inside
        # the per-attempt envelope whenever both numbers were captured.
        if fee is not None and budget is not None:
            if fee > budget and "route_over_budget" not in rejection:
                c_envelope.bad(cmds, f"{pair_key}: expected_fee_sats={fee} > effective_budget_sats={budget} rejection={rejection!r}")
            else:
                c_envelope.ok()
        # RE-I3
        score = decomp.get("final_score_sats")
        if rejection == "below_hold_margin":
            if isinstance(fee, (int, float)) and fee <= 0:
                c_hold.bad(cmds, f"{pair_key}: below_hold_margin on zero-cost route (fee={fee})")
            else:
                c_hold.ok()
        if selected and isinstance(score, (int, float)) and isinstance(fee, (int, float)):
            if fee > 0 and score < 0:
                c_hold.bad(cmds, f"{pair_key}: SELECTED with final_score_sats={score} < 0, cost={fee}")
            else:
                c_hold.ok()
        # RE-I10
        p = decomp.get("p_success")
        if isinstance(p, (int, float)):
            if 0.05 - 1e-9 <= p <= 0.99 + 1e-9:
                c_bounds.ok()
            else:
                c_bounds.bad(cmds, f"{pair_key}: p_success={p}")
        for bias_key in ("hive_source_rebalance_bias", "hive_dest_rebalance_bias",
                         "metabolic_rebalance_bias", "immune_rebalance_bias"):
            b = cand.get(bias_key, inputs.get(bias_key))
            if isinstance(b, (int, float)):
                if 0.85 - 1e-9 <= b <= 1.15 + 1e-9:
                    c_bounds.ok()
                else:
                    c_bounds.bad(cmds, f"{pair_key}: {bias_key}={b}")

    # anomaly material
    ledger_mismatch = []   # (path, detail) ledger spent vs visible success fees
    decision_actions = defaultdict(int)
    statuses = defaultdict(int)
    fee_over_max_failed = 0

    for node in NODES:
        # row id -> last (snap_ts, row); ordered timeline of decisions & budgets
        rows_final = {}
        decisions = []   # (snap_ts, action, budget_blocked, path)
        budgets = []     # (snap_ts, effective, spent_rebal, spent_total, reserved, remaining, path)

        for ts, cmds in iter_snapshots(node):
            st = load(cmds / "revenue-status.json")
            if st:
                rd = st.get("rebalance_decision") or {}
                if rd.get("action"):
                    decisions.append((ts, rd.get("action"), bool(rd.get("budget_blocked")), cmds))
                    decision_actions[rd.get("action")] += 1
                for row in st.get("recent_rebalances") or []:
                    rid = row.get("id")
                    if rid is not None:
                        prev = rows_final.get((node, rid))
                        if prev is None or ts >= prev[0]:
                            rows_final[(node, rid)] = (ts, row, cmds)

            dbg = load(cmds / "revenue-rebalance-debug.json")
            if dbg:
                lc = dbg.get("last_cycle") or {}
                summary = lc.get("summary") or {}
                sel_n = int(summary.get("selected_pairs") or 0)
                ex_n = int(summary.get("execution_count") or 0)
                if sel_n <= 20 and ex_n <= 20:
                    c_conc.ok()
                else:
                    c_conc.bad(cmds, f"selected_pairs={sel_n} execution_count={ex_n}")
                selected_keys = set()
                for cand in lc.get("selected_candidates") or []:
                    selected_keys.add((cand.get("source_channel_id"), cand.get("dest_channel_id")))
                    check_decomposition(cand, cmds, selected=True)
                for cand in lc.get("considered_candidates") or []:
                    key = (cand.get("source_channel_id"), cand.get("dest_channel_id"))
                    if key not in selected_keys:
                        check_decomposition(cand, cmds, selected=False)

            tb = load(cmds / "revenue-total-cost-budget.json")
            if tb:
                eff = tb.get("effective_budget_sats")
                spent = tb.get("actual_spent_sats")
                res = tb.get("reserved_sats")
                rem = tb.get("remaining_sats")
                rebal = (tb.get("actual_spent_by_category") or {}).get("rebalance", 0)
                if None not in (eff, spent, res, rem):
                    budgets.append((ts, eff, rebal, spent, res, rem, cmds))
                    if rem == max(0, eff - spent - res):
                        c_consist.ok()
                    else:
                        c_consist.bad(cmds, f"remaining={rem} effective={eff} spent={spent} reserved={res}")
                    if rebal <= eff:
                        c_capped.ok()
                    else:
                        c_capped.bad(cmds, f"rebalance spent_24h={rebal} > effective_budget={eff}")

        # --- per-row checks on the final (deduplicated) row set
        success_rows = []
        for (n, rid), (ts, row, cmds) in sorted(rows_final.items(), key=lambda kv: kv[1][0]):
            statuses[row.get("status")] += 1
            if row.get("status") == "pending_settlement":
                if row.get("payment_hash"):
                    c_pending.ok()
                else:
                    c_pending.bad(cmds, f"row id={rid} pending_settlement without payment_hash")
            if (row.get("status") == "success"
                    and row.get("actual_fee_sats") is not None
                    and row.get("max_fee_sats") is not None):
                if row["actual_fee_sats"] <= row["max_fee_sats"]:
                    c_fee_cap.ok()
                else:
                    c_fee_cap.bad(cmds, f"row id={rid} actual_fee={row['actual_fee_sats']} > max_fee={row['max_fee_sats']} type={row.get('rebalance_type')}")
            if row.get("rebalance_type") == "diagnostic":
                amt = row.get("amount_sats") or 0
                mf = row.get("max_fee_sats")
                if amt <= 50_000 and (mf is None or mf <= 100):
                    c_diag.ok()
                else:
                    c_diag.bad(cmds, f"row id={rid} amount={amt} max_fee={mf}")
            if row.get("status") == "success":
                success_rows.append((row.get("timestamp") or 0, row, cmds))
            if (row.get("status") == "failed" and row.get("actual_fee_sats")
                    and row.get("max_fee_sats")
                    and row["actual_fee_sats"] > row["max_fee_sats"]):
                fee_over_max_failed += 1
        success_rows.sort()

        # --- RB-I1b suppressed windows
        decisions.sort()
        for (t1, a1, b1, p1), (t2, a2, b2, p2) in zip(decisions, decisions[1:]):
            if a1 == "suppressed" and b1 and a2 == "suppressed" and b2:
                hits = [r for (rt, r, _) in success_rows
                        if t1 < (r.get("timestamp") or 0) <= t2
                        and r.get("rebalance_type") == "normal"]
                if hits:
                    c_suppress.bad(p2, f"window {t1}..{t2}: automated success rows ids={[r['id'] for r in hits]}")
                else:
                    c_suppress.ok()

        # --- RB-I4/RB-I9 ledger cross-check (anomaly material)
        for ts, eff, rebal, spent, res, rem, cmds in budgets:
            visible = sum((r.get("actual_fee_sats") or 0)
                          for (rt, r, _) in success_rows
                          if ts - 86_400 <= (r.get("timestamp") or 0) <= ts)
            # ledger should be >= visible sum (rows list is bounded, so visible is a lower bound)
            if rebal + 2 < visible:
                ledger_mismatch.append((str(cmds), f"ledger rebalance spent_24h={rebal} < visible success fees={visible}"))

    print("=== sweep_rebalancer: corpus sweep results ===")
    for c in (c_consist, c_suppress, c_capped, c_diag,
              c_envelope, c_fee_cap, c_hold, c_pending, c_conc, c_bounds):
        c.report()
    print()
    print("Not corpus-observable: RB-I2 RB-I3 RB-I5 RB-I6 RB-I7 RB-I8 RB-I11 RB-I12; "
          "RE-I1 RE-I4 RE-I5 RE-I6 RE-I7 RE-I11 RE-I12 RE-I13")
    print()
    print("--- anomaly material ---")
    print(f"decision action distribution: {dict(decision_actions)}")
    print(f"final rebalance row status distribution: {dict(statuses)}")
    print(f"failed rows recording actual_fee_sats > max_fee_sats (executor rejection bookkeeping): {fee_over_max_failed}")
    print(f"ledger-vs-visible-success-fee mismatches (ledger < visible): {len(ledger_mismatch)}")
    for p, d in ledger_mismatch[:5]:
        print(f"    {p}\n      {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
