#!/usr/bin/env python3
"""Read-only corpus sweep for the Tier-2 data/budget verification campaign.

Modules under verification: database.py (DB-*), capex_budget.py (CB-*),
capital_efficiency.py (CE-*, metabolism-ledger seam), segment_observations.py
(SO-*), demand_flow.py (DF-*, not corpus-observable).

Sweeps every snapshot under /home/sat/cl-mycelium-hermes/<node>/<day>/<time>/commands
and prints per-invariant pass/violation counts with example paths on violation.

Usage: python3 tools/audit/sweep_data_budget.py [corpus_root]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

CORPUS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/sat/cl-mycelium-hermes")
NODES = ("hive-nexus-01", "hive-nexus-02")

SO_BUCKETS = {50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000}
SO_FAILURE_CLASSES = {"liquidity", "fee", "timeout", "unknown"}
CB_TIERS = {"proven", "active", "bootstrap", "fleet", "blocked"}
CB_TIER_PPM = {"proven": 2000, "active": 500, "bootstrap": 250, "fleet": 50, "blocked": 0}
CB_HIVE_MULTS = {1.0, 1.5, 2.0}

counts = defaultdict(lambda: [0, 0])   # check -> [pass, fail]
examples = defaultdict(list)           # check -> example paths


def tally(check, ok, path, detail=""):
    counts[check][0 if ok else 1] += 1
    if not ok and len(examples[check]) < 3:
        examples[check].append(f"{path} {detail}".strip())


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def check_spend_ledger(cdir, rel):
    p = cdir / "revenue-spend-ledger.json"
    d = load(p)
    if d is None or "error" in (d or {}):
        return
    spent = d.get("spent_24h_sats")
    resv = d.get("reserved_24h_sats")
    sbc = d.get("spent_by_category", {}) or {}
    rbc = d.get("reserved_by_category", {}) or {}
    tally("SL-ARITH-SPENT spent_24h==sum(spent_by_category)",
          spent == sum(sbc.values()), rel, f"spent={spent} sum={sum(sbc.values())}")
    tally("SL-ARITH-RESV reserved_24h==sum(reserved_by_category)",
          resv == sum(rbc.values()), rel, f"resv={resv} sum={sum(rbc.values())}")
    vals = [spent, resv] + list(sbc.values()) + list(rbc.values())
    tally("SL-NONNEG all ledger sums >= 0",
          all(isinstance(v, (int, float)) and v >= 0 for v in vals), rel)
    tally("SL-WINDOW window_hours == 24 (collector default)",
          d.get("window_hours") == 24, rel, f"wh={d.get('window_hours')}")
    # DB-7 (pre-drift): no coverage fields emitted in corpus era
    has_cov = ("covered_hours" in d) or ("coverage_hours" in d)
    tally("DB7-NO-COVERED-HOURS spend ledger lacks covered_hours (corpus era)",
          not has_cov, rel, f"keys={sorted(d.keys())}")


def check_total_cost_budget(cdir, rel):
    p = cdir / "revenue-total-cost-budget.json"
    d = load(p)
    if d is None or "error" in (d or {}):
        return
    eff = d.get("effective_budget_sats", 0)
    spent = d.get("actual_spent_sats", 0)
    resv = d.get("reserved_sats", 0)
    rem = d.get("remaining_sats", None)
    expected = max(0, eff - spent - resv)
    tally("TCB-REMAIN remaining==max(0,effective-spent-reserved)",
          rem == expected, rel, f"rem={rem} expected={expected}")
    asbc = d.get("actual_spent_by_category", {}) or {}
    # open/close may be excluded from generic totals; compare against
    # rebalance+boltz+ledger and against the full sum, count whichever matches.
    core = sum(int(asbc.get(k, 0) or 0) for k in ("rebalance", "boltz", "ledger"))
    full = sum(int(v or 0) for v in asbc.values())
    tally("TCB-CAT actual_spent equals category sum (core or full)",
          spent in (core, full), rel, f"spent={spent} core={core} full={full}")
    has_cov = ("covered_hours" in d) or ("coverage_hours" in d)
    tally("TCB-NO-COVERED-HOURS total-cost-budget lacks covered_hours (corpus era)",
          not has_cov, rel)


def check_capex_status(cdir, rel):
    p = cdir / "revenue-capex-status.json"
    d = load(p)
    if d is None or "error" in (d or {}):
        return
    channels = d.get("channels", {}) or {}
    env = int(d.get("global_envelope_sats", 0) or 0)
    expl = int(d.get("fleet_exploration_budget_sats", 0) or 0)
    tact = int(d.get("tactical_budget_sats", 0) or 0)
    ch_sum = sum(int(c.get("budget_sats", 0) or 0) for c in channels.values())
    total = ch_sum + expl + tact
    # Each sats figure is independently ceil-rounded from msat, so the sum of
    # parts may exceed the envelope by < (n_parts) sats. Allow that slack.
    slack = len(channels) + 2
    tally("CB1-ENVELOPE sum(budgets)+exploration+tactical <= envelope (+ceil slack)",
          total <= env + slack, rel, f"total={total} env={env} slack={slack}")
    for cid, c in channels.items():
        tier = c.get("tier")
        b = int(c.get("budget_sats", 0) or 0)
        tally("CB-TIER tier in {proven,active,bootstrap,fleet,blocked}",
              tier in CB_TIERS, rel, f"{cid} tier={tier}")
        tally("CB-TIERPPM tier_ppm matches tier",
              c.get("tier_ppm") == CB_TIER_PPM.get(tier, -1), rel,
              f"{cid} tier={tier} ppm={c.get('tier_ppm')}")
        if tier == "blocked":
            tally("CB5-BLOCKED-ZERO blocked tier has 0 budget", b == 0, rel,
                  f"{cid} budget={b}")
        if tier == "fleet":
            tally("CB5-FLEET-CAP fleet tier budget <= 200 sats", b <= 200, rel,
                  f"{cid} budget={b}")
        tally("CB6-HIVE-MULT hive_multiplier in {1.0,1.5,2.0}",
              float(c.get("hive_multiplier", 1.0)) in CB_HIVE_MULTS, rel,
              f"{cid} mult={c.get('hive_multiplier')}")
        tally("CB-NONNEG channel budget >= 0", b >= 0, rel, f"{cid} budget={b}")
    prio = d.get("allocated_by_priority_sats", {}) or {}
    if prio:
        defensive = sum(int(c.get("budget_sats", 0) or 0) for c in channels.values()
                        if c.get("priority_class") == "defensive")
        preservation = sum(int(c.get("budget_sats", 0) or 0) for c in channels.values()
                           if c.get("priority_class") == "preservation")
        ok = (abs(prio.get("defensive", 0) - defensive) <= slack
              and abs(prio.get("preservation", 0) - preservation) <= slack
              and prio.get("operational", -1) == tact
              and prio.get("growth", -1) == expl)
        tally("CB-PRIO allocated_by_priority consistent with channel budgets",
              ok, rel, f"prio={prio} def={defensive} pres={preservation}")
    tally("CB-PRIOCLASS priority_class in ordering set",
          d.get("priority_class") in {"defensive", "preservation", "operational", "growth"},
          rel, f"pc={d.get('priority_class')}")


def check_metabolism(cdir, rel):
    p = cdir / "hive-organism-status.json"
    d = load(p)
    if d is None:
        return
    ledger = (((d.get("goal_state") or {}).get("metabolism") or {}).get("ledger") or {})
    windows = ledger.get("windows") or {}
    if not windows:
        return
    intake = {w.get("energy_intake_msat") for w in windows.values()}
    reserves = {w.get("energy_reserves_msat") for w in windows.values()}
    burn = {w.get("metabolic_burn_msat") for w in windows.values()}
    dev = {w.get("developmental_expenditure_msat") for w in windows.values()}
    tally("ML-INTAKE-IDENT energy_intake identical across all windows (anomaly)",
          len(intake) > 1, rel, f"values={sorted(intake)}")
    tally("ML-RESERVE-IDENT energy_reserves identical across all windows (anomaly)",
          len(reserves) > 1, rel, f"values={sorted(reserves)}")
    tally("ML-BURN-IDENT burn identical across all windows (anomaly)",
          len(burn) > 1, rel, f"values={sorted(burn)}")
    tally("ML-DEV-IDENT development identical across all windows (anomaly)",
          len(dev) > 1, rel, f"values={sorted(dev)}")
    for name, w in windows.items():
        cov = w.get("coverage") or {}
        tally("ML-COVER coverage status==unknown & covered_hours==null (anomaly)",
              not (cov.get("status") == "unknown" and cov.get("covered_hours") is None),
              rel, f"{name} cov={cov}")


def check_segment_observations(cdir, rel):
    p = cdir / "listdatastore-key-revenue-segment-observations.json"
    d = load(p)
    if d is None:
        return
    entries = d.get("datastore") or []
    if not entries:
        return
    raw = entries[0].get("string")
    if raw is None:
        hexval = entries[0].get("hex")
        if hexval:
            try:
                raw = bytes.fromhex(hexval).decode("utf-8", "replace")
            except Exception:
                raw = None
    try:
        snap = json.loads(raw)
    except Exception:
        tally("SO-PARSE datastore value parses as JSON", False, rel)
        return
    tally("SO-PARSE datastore value parses as JSON", True, rel)
    tally("SO-SCHEMA-VERSION schema_version == 1", snap.get("schema_version") == 1, rel)
    tally("SO-OBSERVER observer_member_id non-empty",
          bool(str(snap.get("observer_member_id") or "").strip()), rel)
    obs = snap.get("segment_observations") or []
    gen = int(snap.get("generated_at") or 0)
    ttl = int(snap.get("ttl_seconds") or 0)
    tally("SO2-BOUND observation count <= 200", len(obs) <= 200, rel, f"n={len(obs)}")
    prev_by_counter = []
    for o in obs:
        ok_schema = (
            o.get("direction") in (0, 1)
            and o.get("amount_bucket_sats") in SO_BUCKETS
            and isinstance(o.get("confidence"), (int, float))
            and 0.0 <= float(o["confidence"]) <= 1.0
            and o.get("failure_class") in SO_FAILURE_CLASSES
            and o.get("outcome") == "failure"
            and bool(str(o.get("short_channel_id") or "").strip())
            and bool(str(o.get("observation_id") or "").strip())
        )
        tally("SO4-SCHEMA exported observation schema-valid", ok_schema, rel,
              f"obs={o.get('observation_id')}")
        oa = int(o.get("observed_at") or 0)
        tally("SO1-TTL observed_at within ttl of generated_at",
              0 < oa and (gen - oa) <= ttl, rel,
              f"obs={o.get('observation_id')} age={gen - oa} ttl={ttl}")
        oid = str(o.get("observation_id") or "")
        parts = oid.split("-")
        if len(parts) == 3 and parts[0] == "obs":
            try:
                prev_by_counter.append((int(parts[2]), int(parts[1])))
            except ValueError:
                pass
    # SO-5: within one snapshot, higher counter implies observed ts >= lower
    # counter's ts (per-process monotonic ids; restarts reset the counter,
    # duplicates across restarts are tallied separately).
    counters = [c for c, _ in prev_by_counter]
    if counters:
        dup = len(counters) != len(set(counters))
        tally("SO5-CTR-UNIQUE counters unique within snapshot", not dup, rel)
        if not dup:
            by_ctr = sorted(prev_by_counter)
            mono = all(by_ctr[i][1] <= by_ctr[i + 1][1] for i in range(len(by_ctr) - 1))
            tally("SO5-MONO id counter order matches timestamp order", mono, rel)
    # Export sorts newest-first (rebalance_engine pushes export_snapshot output)
    times = [int(o.get("observed_at") or 0) for o in obs]
    tally("SO-SORT snapshot sorted newest-first",
          all(times[i] >= times[i + 1] for i in range(len(times) - 1)), rel)


def main():
    snapshots = 0
    for node in NODES:
        base = CORPUS / node
        if not base.is_dir():
            continue
        for day in sorted(base.iterdir()):
            if not day.is_dir():
                continue
            for ts in sorted(day.iterdir()):
                cdir = ts / "commands"
                if not cdir.is_dir():
                    continue
                snapshots += 1
                rel = str(cdir.relative_to(CORPUS))
                check_spend_ledger(cdir, rel)
                check_total_cost_budget(cdir, rel)
                check_capex_status(cdir, rel)
                check_metabolism(cdir, rel)
                check_segment_observations(cdir, rel)

    print(f"corpus root: {CORPUS}")
    print(f"snapshots swept: {snapshots}\n")
    width = max(len(k) for k in counts) if counts else 0
    for check in sorted(counts):
        p, f = counts[check]
        status = "OK  " if f == 0 else "FAIL"
        print(f"{status} {check:<{width}}  pass={p:<6} fail={f}")
        for ex in examples[check]:
            print(f"       e.g. {ex}")


if __name__ == "__main__":
    main()
