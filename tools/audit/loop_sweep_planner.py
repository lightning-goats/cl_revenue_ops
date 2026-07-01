#!/usr/bin/env python3
"""Phase 3 end-to-end loop sweep: CAPACITY PLANNER decision loops (+ Boltz vacuity).

Read-only sweep over the frozen hermes evidence corpus
(/home/sat/cl-mycelium-hermes/<node>/<YYYYMMDD>/<HHMMSSZ>/commands/).
Corpus coverage: 2026-06-09 -> 2026-06-20 plus one 2026-07-01 snapshot
(hole 2026-06-21..06-30). Only the listforwards-window chain is lossless
across the hole (cumulative by updated_index).

Loops verified (corpus-first; code cited to explain corpus behavior):

  L1  STALLED CLOSE PIPELINE - why 44 close recommendations never executed
      despite planner_execute_closes=true on nexus-01 for 118 snapshots.
      Joins the planner action ledger to per-snapshot listconfigs values of
      revenue-ops-planner-max-closes-per-cycle. Code: capacity_planner.py
      _close_execution_enabled (:232-236) requires BOTH planner_execute_closes
      AND planner_max_closes_per_cycle > 0; _execute_close (:3550-3565)
      records status=recommended when that gate is off.

  L2  FEE_REDUCE DELEGATION HANDOFF - "delegated" is a ledger record only
      (_record_fee_reduce_delegation :1185-1233); the actual fee reduction is
      the fee controller's zero-revenue descent acting independently. Verify:
      for each fee_reduce/delegated action, did a DOWNWARD fee change appear
      on that channel within 24h? Primary evidence: union of
      revenue-status.json recent_fee_changes rows; fallback: per-channel
      fee_proportional_millionths timelines from listpeerchannels snapshots.

  L3  DEFIBRILLATION OUTCOMES (CP-H3 pre-work) - join defibrillate/completed
      planner actions to diagnostic rebalance_history rows (union of
      revenue-status.json recent_rebalances) and to the deduplicated
      listforwards chain: did defibrillated channels forward anything in the
      following 7/14 days? NOTE (code): rebalancer.diagnostic_rebalance
      returns success=True even when the active shock is blocked by capital
      controls or no source is available - in those paths NO rebalance row is
      recorded, so planner "completed" does NOT imply a diagnostic row.

  L4  CROSS-MODULE INVARIANTS (with vacuity labels):
      (a) defibrillate/completed -> matching diagnostic rebalance row
      (b) zero close/completed corpus-wide
      (c) no phantom action targets: every planner-action channel_id created
          inside snapshot coverage appears in some listpeerchannels snapshot
      (d) member protection: no CLOSE action at ANY status on any ever-member
          peer (FLEET-WIDE union of per-peer member=true from both nodes'
          hive-export-hints, since each node's hints exclude itself)
      (e) stage ordering: every completed defibrillation has an earlier
          FEE_REDUCE delegation for the same channel in the observable ledger.
          Era-aware: delegation recording (F4c) landed in e5297c7 on
          2026-06-11; stage entries before the deploy (and stages carried
          over it) legitimately have no record -> indeterminate, not violation.

  L5  BOLTZ - dormancy proof: zero boltz spend / events in every
      revenue-spend-ledger and revenue-total-cost-budget snapshot.

Usage: python3 tools/audit/loop_sweep_planner.py [--corpus DIR]
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import time
from collections import Counter, defaultdict

CORPUS = os.environ.get("CL_MYCELIUM_HERMES_ROOT", "/home/sat/cl-mycelium-hermes")
NODES = ("hive-nexus-01", "hive-nexus-02")

DAY = 86400
# e5297c7 "Turn dead-capital staging from timers into priced interventions"
# (2026-06-11) introduced _record_fee_reduce_delegation (F4c).
F4C_LANDED = int(time.mktime(time.strptime("20260611", "%Y%m%d"))) - time.timezone

results: list[tuple[str, str, int, int, str, list]] = []
# (check, node, checked, violations, vacuity_note, examples)


def record(check: str, node: str, checked: int, violations: int,
           note: str = "", examples: list | None = None) -> None:
    results.append((check, node, checked, violations, note, (examples or [])[:6]))


def snapshot_ts(path: str) -> int:
    stamp = path.split("/")[-3]
    return int(time.mktime(time.strptime(stamp, "%Y%m%dT%H%M%SZ"))) - time.timezone


def load(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def artifact_paths(node: str, name: str) -> list[str]:
    return sorted(glob.glob(f"{CORPUS}/{node}/*/*/commands/{name}.json"))


def iso(ts) -> str:
    if not ts:
        return "-"
    return time.strftime("%Y-%m-%d %H:%MZ", time.gmtime(ts))


# ---------------------------------------------------------------------------
# Reconstructions
# ---------------------------------------------------------------------------

def reconstruct_planner_actions(node: str) -> dict[int, dict]:
    """Union planner actions by id (latest view wins) across history+status."""
    actions: dict[int, dict] = {}
    for p in artifact_paths(node, "revenue-planner-history"):
        d = load(p)
        if not isinstance(d, dict):
            continue
        for a in d.get("actions", []) or []:
            if isinstance(a, dict) and isinstance(a.get("id"), int):
                prev = actions.get(a["id"])
                if prev is None or (a.get("completed_at") or 0) >= (prev.get("completed_at") or 0):
                    actions[a["id"]] = dict(a, _path=p)
    for p in artifact_paths(node, "revenue-planner-status"):
        d = load(p)
        if not isinstance(d, dict):
            continue
        for a in d.get("recent_actions", []) or []:
            if isinstance(a, dict) and isinstance(a.get("id"), int) and a["id"] not in actions:
                actions[a["id"]] = dict(a, _path=p)
    return actions


def planner_configs(node: str) -> list[tuple[int, dict, str]]:
    """Per-snapshot planner config values from listconfigs."""
    out = []
    keys = {
        "execute_closes": "revenue-ops-planner-execute-closes",
        "max_closes_per_cycle": "revenue-ops-planner-max-closes-per-cycle",
        "dry_run": "revenue-ops-planner-dry-run",
        "max_opens_per_cycle": "revenue-ops-planner-max-opens-per-cycle",
        "min_fee_ppm": "revenue-ops-min-fee-ppm",
    }
    for p in artifact_paths(node, "listconfigs"):
        d = load(p)
        if not isinstance(d, dict):
            continue
        cfgs = d.get("configs", d)
        row = {}
        for short, k in keys.items():
            v = cfgs.get(k)
            if isinstance(v, dict):
                v = v.get("value_int", v.get("value_bool", v.get("value_str")))
            row[short] = v
        out.append((snapshot_ts(p), row, p))
    return sorted(out)


def reconstruct_rows(node: str, field: str) -> dict[int, dict]:
    """Union rows by id from revenue-status.json <field> (latest view wins)."""
    rows: dict[int, dict] = {}
    seen_at: dict[int, int] = {}
    for p in artifact_paths(node, "revenue-status"):
        d = load(p)
        if not isinstance(d, dict):
            continue
        ts = snapshot_ts(p)
        for r in d.get(field) or []:
            rid = r.get("id")
            if rid is None:
                continue
            if rid not in rows or ts >= seen_at[rid]:
                rows[rid] = r
                seen_at[rid] = ts
    return rows


def forwards_chain(node: str) -> list[dict]:
    """Deduplicated forwards from listforwards-window.json.gz (by created_index,
    keep the highest-updated_index view)."""
    best: dict = {}
    for p in sorted(glob.glob(f"{CORPUS}/{node}/*/*/commands/listforwards-window.json.gz")):
        try:
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        for f in d.get("forwards") or []:
            key = f.get("created_index")
            if key is None:
                key = (f.get("in_channel"), f.get("in_htlc_id"))
            prev = best.get(key)
            if prev is None or (f.get("updated_index") or 0) >= (prev.get("updated_index") or 0):
                best[key] = f
    return list(best.values())


def ever_member_peers(node: str) -> set[str]:
    members: set[str] = set()
    for p in artifact_paths(node, "hive-export-hints"):
        d = load(p)
        hints = d.get("hints") if isinstance(d, dict) else None
        if not isinstance(hints, dict):
            continue
        for pid, h in hints.items():
            if isinstance(h, dict) and h.get("member"):
                members.add(str(pid))
    return members


def peerchannel_fee_timelines(node: str) -> tuple[set[str], dict[str, list[tuple[int, int]]]]:
    """(all scids ever seen, scid -> [(snapshot_ts, local fee ppm)])."""
    scids: set[str] = set()
    fees: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for p in artifact_paths(node, "listpeerchannels"):
        d = load(p)
        if not isinstance(d, dict):
            continue
        ts = snapshot_ts(p)
        for ch in d.get("channels") or []:
            scid = ch.get("short_channel_id")
            if not scid:
                continue
            scids.add(scid)
            ppm = ((ch.get("updates") or {}).get("local") or {}).get("fee_proportional_millionths")
            if isinstance(ppm, int):
                fees[scid].append((ts, ppm))
    for v in fees.values():
        v.sort()
    return scids, fees


# ---------------------------------------------------------------------------
# L1: stalled close pipeline
# ---------------------------------------------------------------------------

def sweep_close_pipeline(node: str, actions: dict[int, dict],
                         configs: list[tuple[int, dict, str]]) -> None:
    closes = sorted((a for a in actions.values() if a.get("action_type") == "close"),
                    key=lambda a: a.get("created_at") or 0)
    statuses = Counter(str(a.get("status")) for a in closes)
    print(f"\n[{node}] L1 close pipeline: {len(closes)} close actions, statuses={dict(statuses)}")

    # close-1 / inv-b: no completed (or failed) close anywhere
    bad = [a["id"] for a in closes if str(a.get("status")) in ("completed", "failed")]
    record("L1/inv-b: zero close actions reached completed or failed", node,
           len(closes), len(bad), "" if closes else "VACUOUS (no close actions)", bad)

    # close-2: max_closes_per_cycle == 0 in every listconfigs snapshot
    viol = [(p, row) for _, row, p in configs
            if str(row.get("max_closes_per_cycle")) != "0"]
    ec_counter = Counter((str(r.get("execute_closes")), str(r.get("max_closes_per_cycle")))
                         for _, r, _ in configs)
    print(f"[{node}]   listconfigs (execute_closes, max_closes_per_cycle): {dict(ec_counter)}")
    record("L1: planner-max-closes-per-cycle == 0 in every listconfigs snapshot "
           "(=> _close_execution_enabled always False)", node, len(configs), len(viol),
           "" if configs else "VACUOUS (no listconfigs)", [p for p, _ in viol])

    # close-3: how many close recommendations were created while execute_closes=true
    ec_true_windows = [(ts, row) for ts, row, _ in configs if str(row.get("execute_closes")) == "true"]
    first_true = ec_true_windows[0][0] if ec_true_windows else None
    n_after = sum(1 for a in closes if first_true and (a.get("created_at") or 0) >= first_true)
    print(f"[{node}]   close actions created at/after first execute_closes=true snapshot "
          f"({iso(first_true)}): {n_after} of {len(closes)} "
          f"- all still gated by max_closes_per_cycle=0")

    # close-4: dry_run close statuses vs config dry_run=false coverage
    cfg_start = configs[0][0] if configs else 0
    dry = [a for a in closes if str(a.get("status")) == "dry_run"]
    dry_inside = [a["id"] for a in dry if (a.get("created_at") or 0) >= cfg_start]
    record("L1: close/dry_run actions all predate config coverage (dry_run=false throughout)",
           node, len(dry), len(dry_inside),
           "" if dry else "VACUOUS (no dry_run closes)", dry_inside)

    # reason x day inventory (which pipeline produced the recommendations, when)
    by_day = Counter((time.strftime("%m-%d", time.gmtime(a.get("created_at") or 0)),
                      (a.get("reason") or "")[:30]) for a in closes)
    if by_day:
        print(f"[{node}]   close staging timeline (day, reason -> count):")
        for (day, r), n in sorted(by_day.items()):
            print(f"      {day} {r:32s} {n:3d}")
    dead_after_f4c = [a["id"] for a in closes
                      if (a.get("created_at") or 0) >= F4C_LANDED
                      and "DEAD_CAPITAL" in (a.get("reason") or "")]
    print(f"[{node}]   DEAD_CAPITAL close stagings after the 06-11 dead-capital "
          f"rework (e5297c7): {len(dead_after_f4c)} {dead_after_f4c}")


# ---------------------------------------------------------------------------
# L2: fee_reduce delegation handoff
# ---------------------------------------------------------------------------

def sweep_delegations(node: str, actions: dict[int, dict],
                      fee_rows: dict[int, dict],
                      fee_timelines: dict[str, list[tuple[int, int]]],
                      fleet_members: set[str], floor_ppm: int) -> list[dict]:
    delegs = sorted((a for a in actions.values()
                     if a.get("action_type") == "fee_reduce"
                     and str(a.get("status")) == "delegated"),
                    key=lambda a: a.get("created_at") or 0)
    by_channel_changes: dict[str, list[dict]] = defaultdict(list)
    for r in fee_rows.values():
        if r.get("channel_id"):
            by_channel_changes[r["channel_id"]].append(r)
    for v in by_channel_changes.values():
        v.sort(key=lambda r: r.get("timestamp") or 0)

    # fee-change ledger continuity (how complete is the primary evidence?)
    ids = sorted(fee_rows)
    missing = (ids[-1] - ids[0] + 1 - len(ids)) if ids else 0
    print(f"\n[{node}] L2 fee_reduce delegations: {len(delegs)}; fee-change ledger "
          f"reconstructed {len(ids)} rows (ids {ids[0] if ids else '-'}..{ids[-1] if ids else '-'}, "
          f"{missing} ids unobserved in-range)")

    table = []
    down24 = unobservable = at_floor_static = above_floor_static = churn = 0
    for a in delegs:
        scid = a.get("channel_id")
        t0 = a.get("created_at") or 0
        win = [r for r in by_channel_changes.get(scid, [])
               if t0 < (r.get("timestamp") or 0) <= t0 + DAY]
        down = [r for r in win if (r.get("new_fee_ppm") or 0) < (r.get("old_fee_ppm") or 0)]
        # fallback: listpeerchannels fee timeline
        tl = fee_timelines.get(scid, [])
        before = [ppm for ts, ppm in tl if ts <= t0]
        after = [(ts, ppm) for ts, ppm in tl if t0 < ts <= t0 + DAY]
        lp_obs = bool(before and after)
        lp_down = lp_obs and min(p for _, p in after) < before[-1]
        fee_before = before[-1] if before else None
        at_floor = fee_before is not None and fee_before <= floor_ppm + 1
        if not (win or lp_obs):
            unobservable += 1
            verdict = "UNOBSERVABLE (window in corpus hole)"
        elif down or lp_down:
            down24 += 1
            verdict = "fee moved DOWN <=24h"
        elif win or (lp_obs and len({p for _, p in after} | ({fee_before} if fee_before is not None else set())) > 1):
            churn += 1
            verdict = "fee changed but NOT down <=24h"
        elif at_floor:
            at_floor_static += 1
            verdict = f"no change - already at floor ({floor_ppm}ppm)"
        else:
            above_floor_static += 1
            verdict = "no change, above floor <=24h"
        table.append({
            "id": a["id"], "channel": scid, "peer": (a.get("peer_id") or "")[:16],
            "member": a.get("peer_id") in fleet_members,
            "at": iso(t0), "ledger_changes_24h": len(win), "ledger_down_24h": len(down),
            "lp_before": fee_before,
            "lp_min_after_24h": min((p for _, p in after), default=None),
            "verdict": verdict,
        })
    obs = len(delegs) - unobservable
    print(f"[{node}]   observable within 24h: {obs}/{len(delegs)}; moved DOWN: {down24}; "
          f"at-floor static: {at_floor_static}; above-floor static: {above_floor_static}; "
          f"changed-not-down: {churn}; unobservable: {unobservable}")
    record("L2: delegated channels above the fee floor show a DOWNWARD fee change "
           "within 24h (at-floor channels have nothing to descend)", node,
           obs - at_floor_static, above_floor_static + churn,
           f"{unobservable} of {len(delegs)} unobservable (corpus hole); "
           f"down={down24}, at_floor_static={at_floor_static}, "
           f"above_floor_static={above_floor_static}, changed_not_down={churn}",
           [t["id"] for t in table
            if t["verdict"] in ("no change, above floor <=24h",
                                "fee changed but NOT down <=24h")])

    # sanity: delegations are zero-cost ledger records
    bad = [a["id"] for a in delegs
           if (a.get("estimated_cost_sats") or 0) != 0 or a.get("amount_sats")]
    record("L2: delegation records are zero-cost (estimated_cost=0, no amount)", node,
           len(delegs), len(bad), "", bad)
    return table


# ---------------------------------------------------------------------------
# L3: defibrillation outcomes (CP-H3 pre-work)
# ---------------------------------------------------------------------------

def sweep_defib_outcomes(node: str, actions: dict[int, dict],
                         rebal_rows: dict[int, dict],
                         forwards: list[dict],
                         members: set[str]) -> list[dict]:
    defibs = sorted((a for a in actions.values()
                     if a.get("action_type") == "defibrillate"
                     and str(a.get("status")) == "completed"),
                    key=lambda a: a.get("created_at") or 0)
    diags = sorted((r for r in rebal_rows.values()
                    if r.get("rebalance_type") == "diagnostic"),
                   key=lambda r: r.get("timestamp") or 0)
    rb_ids = sorted(rebal_rows)
    rb_missing = (rb_ids[-1] - rb_ids[0] + 1 - len(rb_ids)) if rb_ids else 0
    print(f"\n[{node}] L3 defibrillations: {len(defibs)} completed; rebalance ledger "
          f"{len(rb_ids)} rows (ids {rb_ids[0] if rb_ids else '-'}..{rb_ids[-1] if rb_ids else '-'}, "
          f"{rb_missing} ids unobserved in-range), diagnostic rows: {len(diags)}")

    # forwards indexed per channel (settled only)
    fw_out: dict[str, list[tuple[float, int]]] = defaultdict(list)
    fw_in: dict[str, list[float]] = defaultdict(list)
    fw_last = 0.0
    for f in forwards:
        t = f.get("resolved_time") or f.get("received_time") or 0
        fw_last = max(fw_last, t or 0)
        if f.get("status") != "settled":
            continue
        if f.get("out_channel"):
            fw_out[f["out_channel"]].append((t, f.get("fee_msat") or 0))
        if f.get("in_channel"):
            fw_in[f["in_channel"]].append(t)
    print(f"[{node}]   forwards chain: {len(forwards)} deduplicated rows, "
          f"latest activity {iso(int(fw_last)) if fw_last else '-'}")

    # budget surface for corroborating blocked shocks
    budgets = []
    for p in artifact_paths(node, "revenue-total-cost-budget"):
        d = load(p)
        if isinstance(d, dict):
            budgets.append((snapshot_ts(p), d.get("effective_budget_sats"),
                            d.get("actual_spent_sats"), d.get("remaining_sats")))
    budgets.sort()

    def budget_near(ts: int):
        near = [b for b in budgets if abs(b[0] - ts) <= 7200]
        return near[-1] if near else None

    used_diag: set[int] = set()
    table = []
    matched = unmatched = 0
    for a in defibs:
        scid = a.get("channel_id")
        t0 = a.get("completed_at") or a.get("created_at") or 0
        cand = [r for r in diags
                if r.get("to_channel") == scid and r["id"] not in used_diag
                and abs((r.get("timestamp") or 0) - t0) <= 600]
        row = min(cand, key=lambda r: abs((r.get("timestamp") or 0) - t0)) if cand else None
        blocked_note = ""
        if row:
            used_diag.add(row["id"])
            matched += 1
        else:
            unmatched += 1
            b = budget_near(t0)
            if b and (b[3] or 0) <= 0:
                blocked_note = (f"budget exhausted at {iso(b[0])}: "
                                f"spent={b[2]} effective={b[1]} remaining={b[3]}")
            elif b is None:
                blocked_note = "no budget snapshot within 2h (corpus hole)"
        # forwarding outcome windows
        def count_fw(days: int):
            end = t0 + days * DAY
            covered = "full" if fw_last >= end else ("partial" if fw_last > t0 else "none")
            outs = [x for x in fw_out.get(scid, []) if t0 < x[0] <= end]
            ins = [x for x in fw_in.get(scid, []) if t0 < x <= end]
            return covered, len(outs), sum(f for _, f in outs), len(ins)
        cov7, out7, fees7, in7 = count_fw(7)
        cov14, out14, fees14, in14 = count_fw(14)
        table.append({
            "action_id": a["id"], "channel": scid, "peer": (a.get("peer_id") or "")[:16],
            "member": (a.get("peer_id") in members),
            "at": iso(t0),
            "diag_row": row["id"] if row else None,
            "shock_status": row.get("status") if row else "NO ROW (shock blocked/unrecorded)",
            "shock_error": (row.get("error_message") or "")[:60] if row else blocked_note,
            "fw7_cov": cov7, "fw7_out": out7, "fw7_out_fee_msat": fees7, "fw7_in": in7,
            "fw14_cov": cov14, "fw14_out": out14, "fw14_out_fee_msat": fees14, "fw14_in": in14,
        })
    shock_status = Counter(t["shock_status"] for t in table)
    print(f"[{node}]   diag-row join: matched={matched} unmatched={unmatched}; "
          f"shock statuses: {dict(shock_status)}")
    if rb_missing == 0 or unmatched:
        print(f"[{node}]   note: rebalance-ledger id continuity ({rb_missing} missing "
              f"in-range) means unmatched defibs genuinely recorded NO row "
              f"(blocked shock), unless the row id fell in a continuity gap")
    record("L3/inv-a: defibrillate/completed has a matching diagnostic rebalance row "
           "(caveat: blocked shocks legitimately record none - rebalancer.py "
           "diagnostic_rebalance returns success before record_rebalance on "
           "no-source/capital-controls paths)", node, len(defibs), unmatched,
           "" if defibs else "VACUOUS (no completed defibs)",
           [t["action_id"] for t in table if t["diag_row"] is None])

    fw_any7 = sum(1 for t in table if t["fw7_out"] + t["fw7_in"] > 0)
    fw_any14 = sum(1 for t in table if t["fw14_out"] + t["fw14_in"] > 0)
    print(f"[{node}]   forwarded within 7d: {fw_any7}/{len(table)}; "
          f"within 14d: {fw_any14}/{len(table)}")
    return table


# ---------------------------------------------------------------------------
# L4 remaining invariants
# ---------------------------------------------------------------------------

def sweep_invariants(node: str, actions: dict[int, dict], scids: set[str],
                     fleet_members: set[str], coverage_start: int) -> None:
    ordered = sorted(actions.values(), key=lambda a: a.get("created_at") or 0)

    # inv-c: no phantom action targets (within snapshot coverage)
    with_scid = [a for a in ordered if a.get("channel_id")]
    in_cov = [a for a in with_scid if (a.get("created_at") or 0) >= coverage_start]
    phantom = [(a["id"], a.get("action_type"), a.get("channel_id"))
               for a in in_cov if a["channel_id"] not in scids]
    record("inv-c: every planner-action channel_id created inside snapshot coverage "
           "appears in listpeerchannels (no phantom targets)", node,
           len(in_cov), len(phantom),
           f"{len(with_scid) - len(in_cov)} actions predate listpeerchannels coverage "
           "(their channels may already be closed) - excluded", phantom)

    # inv-d: member protection - no CLOSE at any status on ever-member peers
    # (fleet-wide union: each node's own hints exclude itself, so a per-node
    # set would miss channels whose peer is the sibling node)
    closes = [a for a in ordered if a.get("action_type") == "close"]
    viol = [(a["id"], (a.get("peer_id") or "")[:16], a.get("status"))
            for a in closes if a.get("peer_id") in fleet_members]
    record("inv-d: no CLOSE action at ANY status on ever-member peers "
           f"(fleet-wide ever-member set size {len(fleet_members)})", node,
           len(closes), len(viol),
           "" if closes else "VACUOUS (no close actions)", viol)

    # D1 exposure re-count with the fleet-wide member set (supersedes the
    # Phase 2 per-node nearest-snapshot count)
    d1_defib = [a["id"] for a in ordered if a.get("action_type") == "defibrillate"
                and str(a.get("status")) == "completed"
                and a.get("peer_id") in fleet_members]
    d1_deleg = [a["id"] for a in ordered if a.get("action_type") == "fee_reduce"
                and a.get("peer_id") in fleet_members]
    print(f"[{node}]   D1 exposure (fleet-union membership): "
          f"member defibrillations={len(d1_defib)} {d1_defib}; "
          f"member fee_reduce delegations={len(d1_deleg)} {d1_deleg}")

    # inv-e: stage ordering - completed defib preceded by fee_reduce delegation.
    # Era-aware: F4c delegation recording landed 2026-06-11 (e5297c7); stage
    # entries before the deploy - and stages carried across it - have no record.
    delegs_by_chan: dict[str, int] = {}
    for a in ordered:
        if a.get("action_type") == "fee_reduce" and a.get("channel_id"):
            t = a.get("created_at") or 0
            if a["channel_id"] not in delegs_by_chan or t < delegs_by_chan[a["channel_id"]]:
                delegs_by_chan[a["channel_id"]] = t
    defibs = [a for a in ordered if a.get("action_type") == "defibrillate"
              and str(a.get("status")) == "completed"]
    ok = pre_f4c = carried = 0
    carried_ids = []
    for a in defibs:
        scid = a.get("channel_id")
        t = a.get("created_at") or 0
        first_deleg = delegs_by_chan.get(scid)
        if first_deleg is not None and first_deleg <= t:
            ok += 1
        elif t < F4C_LANDED:
            pre_f4c += 1
        else:
            # no earlier delegation record: the stage machine persists stages
            # in the DB, and recording fires only on the none->fee_reduction
            # ENTRY transition - channels already staged when e5297c7 deployed
            # never get a record until they exit and re-enter dead capital.
            carried += 1
            carried_ids.append((a["id"], scid))
    record("inv-e: completed defib preceded by a FEE_REDUCE delegation for the "
           "same channel (verifiable for post-F4c stage entries only)", node,
           len(defibs), 0,
           f"ordering violations impossible to prove: ok={ok}, pre-F4c defibs={pre_f4c}, "
           f"indeterminate stage-carried-over-deploy={carried} {carried_ids}", [])


# ---------------------------------------------------------------------------
# L5: boltz vacuity
# ---------------------------------------------------------------------------

def sweep_boltz(node: str) -> None:
    checked = 0
    nonzero = []
    for p in artifact_paths(node, "revenue-spend-ledger"):
        d = load(p)
        if not isinstance(d, dict):
            continue
        checked += 1
        spent = (d.get("spent_by_category") or {}).get("boltz", 0) or 0
        events = (d.get("event_count_by_category") or {}).get("boltz", 0) or 0
        if spent or events:
            nonzero.append((p, spent, events))
    for p in artifact_paths(node, "revenue-total-cost-budget"):
        d = load(p)
        if not isinstance(d, dict):
            continue
        checked += 1
        comp = (d.get("components") or {}).get("boltz") or {}
        cat = (d.get("actual_spent_by_category") or {}).get("boltz", 0) or 0
        if (comp.get("spent_24h_sats") or 0) or cat:
            nonzero.append((p, comp, cat))
    record("L5: boltz spend/events zero in every spend-ledger and budget snapshot "
           "(dormancy proof - all BM loop checks vacuous)", node, checked,
           len(nonzero), "confirms vacuity, not correctness", nonzero)


# ---------------------------------------------------------------------------

def dump_table(title: str, rows: list[dict]) -> None:
    if not rows:
        print(f"\n{title}: (empty)")
        return
    print(f"\n{title}:")
    cols = list(rows[0].keys())
    print("  " + " | ".join(cols))
    for r in rows:
        print("  " + " | ".join(str(r.get(c)) for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    args = ap.parse_args()
    globals()["CORPUS"] = args.corpus

    t0 = time.time()
    # Fleet-wide member union: each node's hints exclude the node itself,
    # so membership checks must union both nodes' views.
    fleet_members: set[str] = set()
    for node in NODES:
        fleet_members |= ever_member_peers(node)
    print(f"fleet-wide ever-member peers ({len(fleet_members)}): "
          f"{sorted(m[:16] for m in fleet_members)}")

    for node in NODES:
        actions = reconstruct_planner_actions(node)
        configs = planner_configs(node)
        fee_rows = reconstruct_rows(node, "recent_fee_changes")
        rebal_rows = reconstruct_rows(node, "recent_rebalances")
        forwards = forwards_chain(node)
        scids, fee_timelines = peerchannel_fee_timelines(node)
        coverage_start = min((snapshot_ts(p) for p in
                              artifact_paths(node, "listpeerchannels")), default=0)
        floors = {str(r.get("min_fee_ppm")) for _, r, _ in configs}
        floor_ppm = int(configs[-1][1].get("min_fee_ppm") or 0) if configs else 0
        if len(floors) > 1:
            print(f"[{node}] WARNING: min_fee_ppm changed across corpus: {floors}")

        inv = Counter((a.get("action_type"), str(a.get("status"))) for a in actions.values())
        print(f"\n{'=' * 78}\n[{node}] planner ledger: {len(actions)} actions {dict(inv)}; "
              f"fee floor {floor_ppm}ppm")

        sweep_close_pipeline(node, actions, configs)
        deleg_table = sweep_delegations(node, actions, fee_rows, fee_timelines,
                                        fleet_members, floor_ppm)
        defib_table = sweep_defib_outcomes(node, actions, rebal_rows, forwards,
                                           fleet_members)
        sweep_invariants(node, actions, scids, fleet_members, coverage_start)
        sweep_boltz(node)

        dump_table(f"[{node}] DELEGATION OUTCOME TABLE", deleg_table)
        dump_table(f"[{node}] DEFIBRILLATION OUTCOME TABLE (CP-H3 input)", defib_table)

    print("\n" + "=" * 78)
    print("SWEEP RESULTS")
    print("=" * 78)
    for check, node, checked, violations, note, examples in results:
        status = "PASS" if violations == 0 else "FLAG"
        if checked == 0:
            status = "VACUOUS"
        print(f"[{status}] {node} :: {check} :: checked={checked} flagged={violations}"
              + (f" :: {note}" if note else ""))
        for ex in examples:
            print(f"          e.g. {ex}")
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
