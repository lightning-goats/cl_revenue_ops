#!/usr/bin/env python3
"""Phase 2 corpus sweep: capacity_planner / boltz_manager / hive_hints invariants.

Read-only sweep over the hermes evidence corpus
(/home/sat/cl-mycelium-hermes/<node>/<YYYYMMDD>/<snapshot>Z/commands/*.json).

Checks every corpus-observable claim from the Tier-1 contracts:
  capacity_planner: CP-I1 (close execution gating / statuses), CP-I2 (dry-run),
    CP-I4 (defibrillations per cycle), CP-I5 + operator decision D1
    (hive-member close absence; member DEAD_CAPITAL defibrillation/fee_reduce
    episodes), CP-I14 (24h per-peer cooldown), candidate pool bound (<=32),
    production flag answer (planner_enabled / dry_run / execute_closes).
  boltz_manager: BM-H2 (rolling-24h boltz spend never exceeds the unified
    budget), boltz activity inventory (spend ledger / budget components).
  hive_hints: HH-I1 freshness consistency of the status surface, HH-I2/HH-I10
    stale-fallback field lists, HH-I3 bias bounds replayed through the real
    adapter against every producer payload, HH-I8 fleet fee prior bounds,
    m2 scope / policy wiring (always bounded_bias).

Prints per-invariant pass/violation counts with example paths on violation.

Usage: python3 tools/audit/sweep_planner_boltz_hints.py [--corpus DIR] [--replay-stride N]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter, defaultdict
from unittest.mock import MagicMock

# The adapter imports pyln.client; stub it (read-only sweep, no plugin).
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault("pyln", mock_pyln)
sys.modules.setdefault("pyln.client", mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.hive_hints import HiveHintAdapter  # noqa: E402

CORPUS = "/home/sat/cl-mycelium-hermes"
NODES = ("hive-nexus-01", "hive-nexus-02")

DEFIB_CLUSTER_SECONDS = 600      # actions created within this window = one cycle
DEFAULT_DEFIB_LIMIT = 1          # planner_max_defibrillations_per_cycle default
COOLDOWN_SECONDS = 24 * 3600

results: list[tuple[str, str, int, int, list]] = []  # (check, node, checked, violations, examples)


def record(check: str, node: str, checked: int, violations: int, examples: list) -> None:
    results.append((check, node, checked, violations, examples[:5]))


def snapshot_ts(path: str) -> int:
    """Parse the snapshot UTC timestamp from .../<YYYYMMDDTHHMMSSZ>/commands/x.json."""
    stamp = path.split("/")[-3]  # e.g. 20260620T151145Z
    return int(time.mktime(time.strptime(stamp, "%Y%m%dT%H%M%SZ"))) - time.timezone


def load(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def artifact_paths(node: str, name: str) -> list[str]:
    return sorted(glob.glob(f"{CORPUS}/{node}/*/*/commands/{name}.json"))


# ---------------------------------------------------------------------------
# Planner ledger reconstruction
# ---------------------------------------------------------------------------

def reconstruct_planner_actions(node: str) -> tuple[dict, list, list]:
    """Union planner actions by id across all revenue-planner-history snapshots.

    Returns (actions_by_id, status_flag_timeline, history_paths).
    status_flag_timeline: [(ts, enabled, dry_run, execute_closes, pool_size, path)]
    """
    actions: dict[int, dict] = {}
    flags = []
    paths = artifact_paths(node, "revenue-planner-history")
    for p in paths:
        d = load(p)
        if not isinstance(d, dict):
            continue
        for a in d.get("actions", []) or []:
            if isinstance(a, dict) and isinstance(a.get("id"), int):
                prev = actions.get(a["id"])
                # keep the latest view of each action (statuses can be updated)
                if prev is None or (a.get("completed_at") or 0) >= (prev.get("completed_at") or 0):
                    actions[a["id"]] = dict(a, _path=p)
    for p in artifact_paths(node, "revenue-planner-status"):
        d = load(p)
        if not isinstance(d, dict):
            continue
        flags.append((
            snapshot_ts(p),
            d.get("enabled"),
            d.get("dry_run"),
            d.get("execute_closes"),
            d.get("candidate_pool_size"),
            p,
        ))
        for a in d.get("recent_actions", []) or []:
            if isinstance(a, dict) and isinstance(a.get("id"), int) and a["id"] not in actions:
                actions[a["id"]] = dict(a, _path=p)
    return actions, sorted(flags), paths


def member_sets(node: str) -> list[tuple[int, frozenset, str]]:
    """Per-snapshot hive member peer-id sets from hive-export-hints payloads."""
    out = []
    for p in artifact_paths(node, "hive-export-hints"):
        d = load(p)
        if not isinstance(d, dict):
            continue
        hints = d.get("hints")
        if not isinstance(hints, dict):
            continue
        members = frozenset(
            str(pid) for pid, h in hints.items()
            if isinstance(h, dict) and bool(h.get("member", False))
        )
        out.append((snapshot_ts(p), members, p))
    return out


def members_at(ts: int, sets: list[tuple[int, frozenset, str]]) -> frozenset | None:
    """Member set from the nearest hints snapshot within 24h of ts."""
    best = None
    best_dt = None
    for sts, mem, _p in sets:
        dt = abs(sts - ts)
        if best_dt is None or dt < best_dt:
            best_dt, best = dt, mem
    if best is None or best_dt is None or best_dt > 86400:
        return None
    return best


def sweep_planner(node: str) -> None:
    actions, flags, _paths = reconstruct_planner_actions(node)
    msets = member_sets(node)
    ordered = sorted(actions.values(), key=lambda a: (a.get("created_at") or 0, a["id"]))

    # --- Production flag answer + flag consistency
    flag_counter = Counter((e, dr, xc) for _, e, dr, xc, _, _ in flags)
    print(f"\n[{node}] planner flags across {len(flags)} revenue-planner-status snapshots:")
    for (e, dr, xc), n in sorted(flag_counter.items()):
        print(f"    enabled={e} dry_run={dr} execute_closes={xc}: {n} snapshots")
    pool_max = max((f[4] for f in flags if isinstance(f[4], int)), default=None)

    # --- Status inventory
    inv = Counter((a.get("action_type"), a.get("status")) for a in ordered)
    print(f"[{node}] planner action ledger reconstructed: {len(ordered)} unique actions "
          f"(ids {ordered[0]['id']}-{ordered[-1]['id']})" if ordered else f"[{node}] no actions")
    for (at, st), n in sorted(inv.items()):
        print(f"    {at}/{st}: {n}")

    # --- id continuity (reconstruction completeness)
    if ordered:
        ids = sorted(a["id"] for a in ordered)
        gaps = [i for i in range(ids[0], ids[-1] + 1) if i not in set(ids)]
        record("ledger id continuity (reconstruction gaps)", node, ids[-1] - ids[0] + 1,
               len(gaps), gaps)

    # --- CP-I2: dry_run statuses only while dry_run flag set
    corpus_start = flags[0][0] if flags else 0
    dry_flag_ever = any(dr for _, _, dr, _, _, _ in flags)
    viol = [a for a in ordered
            if str(a.get("status")) == "dry_run" and (a.get("created_at") or 0) >= corpus_start
            and not dry_flag_ever]
    checked = sum(1 for a in ordered if (a.get("created_at") or 0) >= corpus_start)
    record("CP-I2 dry_run statuses only under dry_run flag", node, checked, len(viol),
           [a["id"] for a in viol])

    # --- CP-I4: defibrillations per cycle cluster
    defibs = [a for a in ordered if a.get("action_type") == "defibrillate"
              and str(a.get("status")) != "dry_run"]
    clusters: list[list[dict]] = []
    for a in sorted(defibs, key=lambda x: x.get("created_at") or 0):
        if clusters and (a.get("created_at") or 0) - (clusters[-1][-1].get("created_at") or 0) <= DEFIB_CLUSTER_SECONDS:
            clusters[-1].append(a)
        else:
            clusters.append([a])
    over = [c for c in clusters if len(c) > DEFAULT_DEFIB_LIMIT]
    record(f"CP-I4 defibrillations per cycle <= {DEFAULT_DEFIB_LIMIT}", node, len(clusters),
           len(over), [[a["id"] for a in c] for c in over])

    # --- CP-I5: no CLOSE on hive members; D1: member defib/fee_reduce episodes
    close_member_viol, member_defib, member_feereduce = [], [], []
    checked_member = 0
    for a in ordered:
        ts = a.get("created_at") or 0
        mem = members_at(ts, msets)
        if mem is None:
            continue
        checked_member += 1
        if a.get("peer_id") in mem:
            at = a.get("action_type")
            if at == "close":
                close_member_viol.append((a["id"], a.get("peer_id"), a.get("status"), a.get("_path")))
            elif at == "defibrillate":
                member_defib.append((a["id"], a.get("peer_id"), a.get("status"), a.get("channel_id"), a.get("_path")))
            elif at == "fee_reduce":
                member_feereduce.append((a["id"], a.get("peer_id"), a.get("status"), a.get("channel_id")))
    record("CP-I5 no CLOSE action on hive members", node, checked_member,
           len(close_member_viol), close_member_viol)
    record("D1 flag: hive-member DEFIBRILLATE episodes (permitted-as-implemented)", node,
           checked_member, len(member_defib), member_defib)
    record("D1 flag: hive-member FEE_REDUCE episodes (permitted-as-implemented)", node,
           checked_member, len(member_feereduce), member_feereduce)

    # --- CP-I1: completed closes only while execute_closes was on
    completed_closes = [a for a in ordered if a.get("action_type") == "close"
                        and str(a.get("status")) == "completed"]
    viol = []
    for a in completed_closes:
        ts = a.get("created_at") or 0
        if ts < corpus_start:
            continue
        # nearest flag snapshot
        nearest = min(flags, key=lambda f: abs(f[0] - ts), default=None)
        if nearest and nearest[3] is False:
            viol.append((a["id"], a.get("channel_id"), nearest[5]))
    record("CP-I1 completed closes only while execute_closes=true", node,
           len(completed_closes), len(viol), viol)

    # --- CP-I14: 24h per-peer cooldown on gated actions (open/close/defibrillate)
    GATED = {"open", "close", "defibrillate"}
    by_peer: dict[str, list[dict]] = defaultdict(list)
    for a in ordered:
        if a.get("peer_id"):
            by_peer[a["peer_id"]].append(a)
    cooldown_viol = []
    checked_pairs = 0
    for peer, acts in by_peer.items():
        acts = sorted(acts, key=lambda x: x.get("created_at") or 0)
        for i, later in enumerate(acts):
            if later.get("action_type") not in GATED:
                continue  # fee_reduce delegations are recorded without a cooldown check
            lts = later.get("created_at") or 0
            if lts < corpus_start:
                continue
            for earlier in acts[:i]:
                ets = earlier.get("created_at") or 0
                if str(earlier.get("status", "")).lower() in ("dry_run", "failed"):
                    continue
                checked_pairs += 1
                if 0 < lts - ets < COOLDOWN_SECONDS:
                    cooldown_viol.append((peer[:16], earlier["id"], later["id"], lts - ets))
    record("CP-I14 24h per-peer cooldown (open/close/defibrillate)", node,
           checked_pairs, len(cooldown_viol), cooldown_viol)

    # --- candidate pool bound (<=32)
    pool_viol = []
    npool = 0
    for p in artifact_paths(node, "revenue-planner-candidates"):
        d = load(p)
        if not isinstance(d, dict):
            continue
        cands = d.get("candidates")
        if isinstance(cands, list):
            npool += 1
            if len(cands) > 32:
                pool_viol.append((p, len(cands)))
    if pool_max is not None and pool_max > 32:
        pool_viol.append(("candidate_pool_size in status", pool_max))
    record("candidate pool size <= 32", node, npool, len(pool_viol), pool_viol)


# ---------------------------------------------------------------------------
# Boltz
# ---------------------------------------------------------------------------

def sweep_boltz(node: str) -> None:
    # BM-H2: rolling-24h boltz spend vs unified budget, every snapshot
    checked = 0
    viol = []
    boltz_spent_seen = Counter()
    for p in artifact_paths(node, "revenue-total-cost-budget"):
        d = load(p)
        if not isinstance(d, dict):
            continue
        checked += 1
        budget = d.get("effective_budget_sats")
        cat = d.get("actual_spent_by_category") or {}
        comp = (d.get("components") or {}).get("boltz") or {}
        boltz_cat = cat.get("boltz", 0) or 0
        boltz_comp = comp.get("spent_24h_sats", 0) or 0
        boltz_spent_seen[(boltz_cat, boltz_comp)] += 1
        if isinstance(budget, (int, float)):
            if boltz_cat > budget or boltz_comp > budget:
                viol.append((p, boltz_cat, boltz_comp, budget))
    record("BM-H2 rolling-24h boltz spend <= unified budget", node, checked, len(viol), viol)

    # Boltz activity inventory from spend ledger
    n = 0
    boltz_events = 0
    categories = Counter()
    for p in artifact_paths(node, "revenue-spend-ledger"):
        d = load(p)
        if not isinstance(d, dict):
            continue
        n += 1
        for k, v in (d.get("spent_by_category") or {}).items():
            categories[k] = max(categories[k], v or 0)
        boltz_events = max(boltz_events, (d.get("event_count_by_category") or {}).get("boltz", 0) or 0)
    print(f"[{node}] spend-ledger snapshots={n}; max 24h spent_by_category={dict(categories)}; "
          f"max boltz event count={boltz_events}")
    record("boltz swap activity present in corpus (informational)", node, n,
           0, [])


# ---------------------------------------------------------------------------
# Hive hints
# ---------------------------------------------------------------------------

BOUNDED_FIELDS = {"fee_bias", "rebalance_bias"}


def sweep_hints(node: str, replay_stride: int) -> None:
    # HH status surface: freshness consistency + fallback policy wiring
    checked = 0
    fresh_incons = []
    fallback_active = 0
    fallback_field_viol = []
    policy_viol = []
    scope_counter = Counter()
    ttl_counter = Counter()
    source_counter = Counter()
    fresh_counter = Counter()
    for p in artifact_paths(node, "revenue-hive-hints-status"):
        d = load(p)
        if not isinstance(d, dict):
            continue
        checked += 1
        fresh = d.get("snapshot_fresh")
        age = d.get("snapshot_age_seconds")
        ttl = d.get("effective_ttl_seconds")
        fresh_counter[bool(fresh)] += 1
        ttl_counter[ttl] += 1
        scope_counter[d.get("m2_scope")] += 1
        source_counter[d.get("snapshot_source")] += 1
        if fresh is True and isinstance(age, (int, float)) and isinstance(ttl, (int, float)):
            if not (-300 <= age <= ttl):
                fresh_incons.append((p, age, ttl))
        pol = d.get("stale_fallback_policy")
        if pol is not None and pol != "bounded_bias":
            policy_viol.append((p, pol))
        if d.get("stale_fallback_active") or d.get("stale_fallback"):
            fallback_active += 1
            allowed = set(d.get("stale_fallback_behavior_fields_allowed") or [])
            if allowed and allowed != BOUNDED_FIELDS:
                fallback_field_viol.append((p, sorted(allowed)))
    print(f"[{node}] hints-status snapshots={checked}; fresh={dict(fresh_counter)}; "
          f"ttl={dict(ttl_counter)}; m2_scope={dict(scope_counter)}; source={dict(source_counter)}; "
          f"stale_fallback_active={fallback_active}")
    record("HH-I1 status freshness consistency (fresh => age within [-300, ttl])",
           node, checked, len(fresh_incons), fresh_incons)
    record("HH-I2 stale-fallback allowed fields == {fee_bias, rebalance_bias}",
           node, fallback_active, len(fallback_field_viol), fallback_field_viol)
    record("HH policy wiring: stale_fallback_policy always bounded_bias",
           node, checked, len(policy_viol), policy_viol)

    # HH-I3 / HH-I8 replay: run every Nth producer payload through the real adapter
    paths = artifact_paths(node, "hive-export-hints")
    replayed = 0
    bias_viol = []
    prior_viol = []
    for p in paths[::max(1, replay_stride)]:
        d = load(p)
        if not isinstance(d, dict) or not isinstance(d.get("hints"), dict):
            continue
        adapter = HiveHintAdapter(MagicMock())
        snap = adapter._validate_and_normalize_snapshot(d)
        if snap is None:
            continue
        # Pin the snapshot as fresh regardless of wall-clock age: we are
        # replaying validation/bounding, not freshness.
        snap = dict(snap)
        snap["generated_at"] = int(time.time())
        snap["ttl_seconds"] = 900
        adapter._store_snapshot(snap, "datastore")
        replayed += 1
        for peer_id in list(d["hints"].keys()):
            fb = adapter.get_fee_bias(peer_id)
            rb = adapter.get_rebalance_bias(peer_id)
            cb = adapter.get_corridor_utilization_bias(peer_id)
            if not (0.9 <= fb <= 1.1):
                bias_viol.append((p, peer_id[:16], "fee_bias", fb))
            if not (0.85 <= rb <= 1.15):
                bias_viol.append((p, peer_id[:16], "rebalance_bias", rb))
            if not (0.9 <= cb <= 1.1):
                bias_viol.append((p, peer_id[:16], "corridor_utilization_bias", cb))
            prior = adapter.get_fleet_fee_prior(peer_id)
            if prior is not None and not (1 <= prior <= 10000):
                prior_viol.append((p, peer_id[:16], prior))
    record(f"HH-I3 replayed bias bounds over producer payloads (stride {replay_stride}, "
           f"{replayed} snapshots)", node, replayed, len(bias_viol), bias_viol)
    record("HH-I8 fleet fee prior in [1, 10000] or None (replay)", node, replayed,
           len(prior_viol), prior_viol)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--replay-stride", type=int, default=5)
    args = ap.parse_args()
    globals()["CORPUS"] = args.corpus

    t0 = time.time()
    for node in NODES:
        sweep_planner(node)
        sweep_boltz(node)
        sweep_hints(node, args.replay_stride)

    print("\n" + "=" * 78)
    print("SWEEP RESULTS")
    print("=" * 78)
    total_viol = 0
    for check, node, checked, violations, examples in results:
        status = "PASS" if violations == 0 else "VIOLATION"
        total_viol += violations
        print(f"[{status}] {node} :: {check} :: checked={checked} violations={violations}")
        if violations:
            for ex in examples:
                print(f"          e.g. {ex}")
    print(f"\nDone in {time.time() - t0:.1f}s; total flagged rows: {total_viol}")


if __name__ == "__main__":
    main()
