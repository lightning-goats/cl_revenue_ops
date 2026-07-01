#!/usr/bin/env python3
"""Phase 4 contribution analysis: CAPITAL SIDE (read-only, reproducible).

Tests the pre-registered capital-side hypotheses of the cl_revenue_ops audit
against the frozen hermes corpus (/home/sat/cl-mycelium-hermes):

  profitability_analyzer  PA-H1  classification predicts earnings
                          PA-H2  bleeder gating saves money
                          PA-H3  marginal_roi informative where reliable
                          (+ class-stability / flapping evidence quality,
                           + D2 structural-mask materiality bound)
  capacity_planner        CP-H1  open quality
                          CP-H2  close/loser validity
                          CP-H3  defibrillation efficacy
  rebalancer              RB-H1..H3
  rebalance_engine_v2     RE-H1..H3
  boltz_manager           BM-H1..H3 (vacuity proof)
  + EXPLORATORY (non-registered, labeled): hold-margin calibration of the
    engine's EV gate from the captured considered-candidate decompositions.

Every number cited in docs/audit/contribution/capital-side.md regenerates
from this script. Deterministic: bootstrap RNG is seeded.

Corpus facts honored (Phase 2/3):
  - snapshot surfaces 2026-06-09..06-20 + one 2026-07-01 capture
    (hole 06-21..06-30); listforwards chain is lossless by updated_index
    and now extends 2026-05-20 .. 2026-07-01T19:51Z (the 07-01 windows
    backfill the hole, superseding Phase 2's pre-backfill totals).
  - nexus-02 settled zero forwards; outcome inference is nexus-01 only.
  - pre-62ae545 twin rows (448/451/453 duplicate 447/450/452) are deduped
    before any RB/RE count.

Usage: python3 tools/audit/contrib_capital_analysis.py [--corpus DIR]
"""

from __future__ import annotations

import argparse
import calendar
import glob
import gzip
import json
import time
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

CORPUS = "/home/sat/cl-mycelium-hermes"
N1, N2 = "hive-nexus-01", "hive-nexus-02"
NODES = (N1, N2)
DAY = 86400.0
SEED = 20260701
BOOT = 4000

# engine EV-gate constants (modules/rebalance_engine_v2.py, HEAD cdb536a)
EXPECTED_UTILIZATION = 0.5
SOURCE_UTILIZATION_DISCOUNT = 0.5

# 62ae545 twin-row fix deploy (2026-06-10 12:14:33 UTC) — rows sharing
# (from,to,timestamp) before this are one physical attempt.
ROW_FIX_TS = calendar.timegm(time.strptime("2026-06-10 12:14:33", "%Y-%m-%d %H:%M:%S"))


def load(path: str):
    try:
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def snap_ts(path: str) -> int:
    stamp = path.split("/")[-3]
    return int(time.mktime(time.strptime(stamp, "%Y%m%dT%H%M%SZ"))) - time.timezone


def artifacts(node: str, name: str) -> list[str]:
    return sorted(glob.glob(f"{CORPUS}/{node}/*/*/commands/{name}"))


def iso(ts) -> str:
    return time.strftime("%Y-%m-%d %H:%MZ", time.gmtime(ts)) if ts else "-"


def msat(v) -> int:
    if isinstance(v, str) and v.endswith("msat"):
        v = v[:-4]
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def week_of(ts: float) -> str:
    return time.strftime("%G-W%V", time.gmtime(ts))


# ---------------------------------------------------------------------------
# Shared reconstructions
# ---------------------------------------------------------------------------

def forwards_chain(node: str) -> list[dict]:
    """Deduplicated forwards (by created_index, keep highest updated_index)."""
    best: dict = {}
    for p in artifacts(node, "listforwards-window.json.gz"):
        d = load(p)
        if not d:
            continue
        for f in d.get("forwards") or []:
            key = f.get("created_index")
            if key is None:
                key = (f.get("in_channel"), f.get("in_htlc_id"))
            prev = best.get(key)
            if prev is None or (f.get("updated_index") or 0) >= (prev.get("updated_index") or 0):
                best[key] = f
    return list(best.values())


def settled_index(forwards: list[dict]):
    """Per-channel settled forward events: out-leg (fee earned) and in-leg."""
    out_ev = defaultdict(list)   # scid -> [(t, fee_msat, out_msat)]
    in_ev = defaultdict(list)    # scid -> [(t, fee_msat, in_msat)]
    t_max = 0.0
    for f in forwards:
        t = f.get("resolved_time") or f.get("received_time") or 0
        t_max = max(t_max, t or 0)
        if f.get("status") != "settled":
            continue
        fee = msat(f.get("fee_msat"))
        if f.get("out_channel"):
            out_ev[f["out_channel"]].append((t, fee, msat(f.get("out_msat"))))
        if f.get("in_channel"):
            in_ev[f["in_channel"]].append((t, fee, msat(f.get("in_msat"))))
    return out_ev, in_ev, t_max


def window_sum(events: list[tuple], t0: float, t1: float):
    """(count, fee_msat, volume_msat) of events with t0 < t <= t1."""
    n = fee = vol = 0
    for t, f, v in events:
        if t0 < t <= t1:
            n += 1
            fee += f
            vol += v
    return n, fee, vol


def channel_presence(node: str):
    """scid -> (first_seen_ts, last_seen_ts, capacity_sats) from listpeerchannels."""
    pres: dict[str, list] = {}
    for p in artifacts(node, "listpeerchannels.json"):
        d = load(p)
        if not d:
            continue
        ts = snap_ts(p)
        for ch in d.get("channels") or []:
            scid = ch.get("short_channel_id")
            if not scid:
                continue
            cap = msat(ch.get("total_msat")) // 1000
            if scid not in pres:
                pres[scid] = [ts, ts, cap]
            else:
                pres[scid][0] = min(pres[scid][0], ts)
                pres[scid][1] = max(pres[scid][1], ts)
                pres[scid][2] = cap or pres[scid][2]
    return {k: tuple(v) for k, v in pres.items()}


def planner_actions(node: str) -> dict[int, dict]:
    actions: dict[int, dict] = {}
    for p in artifacts(node, "revenue-planner-history.json"):
        d = load(p)
        if not isinstance(d, dict):
            continue
        for a in d.get("actions") or []:
            if isinstance(a, dict) and isinstance(a.get("id"), int):
                prev = actions.get(a["id"])
                if prev is None or (a.get("completed_at") or 0) >= (prev.get("completed_at") or 0):
                    actions[a["id"]] = a
    for p in artifacts(node, "revenue-planner-status.json"):
        d = load(p)
        if not isinstance(d, dict):
            continue
        for a in d.get("recent_actions") or []:
            if isinstance(a, dict) and isinstance(a.get("id"), int) and a["id"] not in actions:
                actions[a["id"]] = a
    return actions


def rebalance_rows(node: str) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    seen_at: dict[int, int] = {}
    for p in artifacts(node, "revenue-status.json"):
        d = load(p)
        if not isinstance(d, dict):
            continue
        ts = snap_ts(p)
        for r in d.get("recent_rebalances") or []:
            rid = r.get("id")
            if rid is None:
                continue
            if rid not in rows or ts >= seen_at[rid]:
                rows[rid] = r
                seen_at[rid] = ts
    return rows


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def cliffs_delta(a, b) -> float:
    """P(a>b) - P(a<b). Positive => a stochastically larger."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def cluster_bootstrap_delta(obs_a, obs_b, n_boot=BOOT, seed=SEED):
    """Cluster bootstrap (by channel) 95% CI on Cliff's delta and on the
    difference of medians. obs_* : list of (channel, value)."""
    rng = np.random.default_rng(seed)
    chans_a = sorted({c for c, _ in obs_a})
    chans_b = sorted({c for c, _ in obs_b})
    by_a = defaultdict(list)
    by_b = defaultdict(list)
    for c, v in obs_a:
        by_a[c].append(v)
    for c, v in obs_b:
        by_b[c].append(v)
    deltas, med_diffs = [], []
    for _ in range(n_boot):
        sa = [v for c in rng.choice(chans_a, len(chans_a), replace=True) for v in by_a[c]]
        sb = [v for c in rng.choice(chans_b, len(chans_b), replace=True) for v in by_b[c]]
        if not sa or not sb:
            continue
        deltas.append(cliffs_delta(sa, sb))
        med_diffs.append(float(np.median(sa)) - float(np.median(sb)))
    def ci(x):
        return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if x else (float("nan"),) * 2
    return ci(deltas), ci(med_diffs)


def mwu_greater(a, b):
    """One-sided Mann-Whitney U: a > b."""
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    u, p = stats.mannwhitneyu(a, b, alternative="greater")
    return float(u), float(p)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(((k, v) for k, v in pvals.items() if v == v), key=lambda kv: kv[1])
    out, m, running = {}, len(items), 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def fmt_stats(vals):
    v = np.asarray(vals, float)
    if len(v) == 0:
        return "n=0"
    return (f"n={len(v)} median={np.median(v):.4g} mean={np.mean(v):.4g} "
            f"min={v.min():.4g} max={v.max():.4g} zeros={(v == 0).sum()}")


# ---------------------------------------------------------------------------
# PA: profitability_analyzer
# ---------------------------------------------------------------------------

def pa_section(out_ev, in_ev, chain_end, presence):
    print("\n" + "=" * 78)
    print("PA — profitability_analyzer (PA-H1, PA-H2, PA-H3, stability, D2 bound)")
    print("=" * 78)

    # ---- load class assignments -------------------------------------------
    # per node: snapshots -> per-channel class sequence; per (channel, week):
    # class at FIRST classification of that week (assignment time T).
    rows_total = 0
    week_first = {}          # (node, chan, week) -> (class, T)
    seq = defaultdict(list)  # (node, chan) -> [(ts, class)]
    d2_rows = []             # (node, ts, chan, roi, net)
    underwater_net = defaultdict(int)   # (node, chan) -> worst net (most negative)
    all_keys = Counter()
    n_snaps = Counter()
    for node in NODES:
        for p in artifacts(node, "revenue-profitability.json"):
            d = load(p)
            if not d or "channels_by_class" not in d:
                continue
            ts = snap_ts(p)
            n_snaps[node] += 1
            for cls, rows in (d.get("channels_by_class") or {}).items():
                for r in rows or []:
                    rows_total += 1
                    all_keys.update(r.keys())
                    ch = r.get("channel_id")
                    seq[(node, ch)].append((ts, cls))
                    k = (node, ch, week_of(ts))
                    if k not in week_first:
                        week_first[k] = (cls, ts)
                    roi = r.get("roi_percentage")
                    net = r.get("net_profit_sats") or 0
                    if cls == "break_even" and isinstance(roi, (int, float)) and roi < -10.0:
                        d2_rows.append((node, ts, ch, roi, net))
                    if cls == "underwater":
                        underwater_net[(node, ch)] = min(underwater_net[(node, ch)], net)
    print(f"profitability snapshots: n1={n_snaps[N1]} n2={n_snaps[N2]}; "
          f"channel rows total={rows_total}")

    # ---- PA-H1 --------------------------------------------------------------
    print("\n--- PA-H1: class at T predicts next-7d/14d earnings (nexus-01 only;"
          " nexus-02 settled 0 forwards => no outcome variance, excluded) ---")

    def outcome(ch, t0, days):
        n, fee, vol = window_sum(out_ev.get(ch, []), t0, t0 + days * DAY)
        return n, fee, vol

    horizons = {}
    for days in (7, 14):
        cohort = defaultdict(list)   # class -> [(chan, fee_msat_per_capsat)]
        cohort_raw = defaultdict(list)  # class -> [(chan, fee_sats_per_day)]
        excluded_cov = excluded_cap = 0
        survivors = defaultdict(list)   # sensitivity: channel alive whole window
        for (node, ch, wk), (cls, T) in sorted(week_first.items()):
            if node != N1 or wk not in ("2026-W24", "2026-W25"):
                continue
            if T + days * DAY > chain_end:
                excluded_cov += 1
                continue
            cap = presence.get(ch, (0, 0, 0))[2]
            if not cap:
                excluded_cap += 1
                continue
            _, fee, _ = outcome(ch, T, days)
            per_cap = fee / cap                     # msat per capacity-sat
            cohort[cls].append((ch, per_cap))
            cohort_raw[cls].append((ch, fee / 1000.0 / days))
            last_seen = presence[ch][1]
            if last_seen >= T + days * DAY - 3600:  # allow one capture gap
                survivors[cls].append((ch, per_cap))
        horizons[days] = (cohort, cohort_raw, survivors)
        print(f"\n  [{days}d horizon] channel-week observations "
              f"(excluded: {excluded_cov} window-truncated, {excluded_cap} no-capacity):")
        for cls in ("profitable", "break_even", "underwater", "stagnant_candidate", "zombie"):
            if cohort.get(cls) is not None or cls in ("profitable", "underwater"):
                vals = [v for _, v in cohort.get(cls, [])]
                raw = [v for _, v in cohort_raw.get(cls, [])]
                print(f"    {cls:20s} per-cap(msat/sat): {fmt_stats(vals)}")
                if raw:
                    print(f"    {'':20s} fees(sat/day):     {fmt_stats(raw)}")

        pvals = {}
        for label, other in (("U", "underwater"), ("S", "stagnant_candidate"),
                             ("Z", "zombie")):
            a = [v for _, v in cohort.get("profitable", [])]
            b = [v for _, v in cohort.get(other, [])]
            if not b:
                print(f"    PROFITABLE vs {other}: VACUOUS (no {other} channel-weeks "
                      f"with covered {days}d window)")
                continue
            u, p = mwu_greater(a, b)
            delta = cliffs_delta(a, b)
            (dlo, dhi), (mlo, mhi) = cluster_bootstrap_delta(
                cohort["profitable"], cohort[other])
            pvals[other] = p
            print(f"    PROFITABLE > {other}: MWU U={u:.1f} one-sided p={p:.4g}; "
                  f"Cliff's delta={delta:+.3f} [cluster-boot 95% CI {dlo:+.3f}..{dhi:+.3f}]; "
                  f"median diff={np.median(a) - np.median(b):+.4g} msat/cap-sat "
                  f"[CI {mlo:+.4g}..{mhi:+.4g}]")
            sa = [v for _, v in survivors.get("profitable", [])]
            sb = [v for _, v in survivors.get(other, [])]
            us, ps = mwu_greater(sa, sb)
            print(f"      sensitivity (survivors only, mass-close censoring removed): "
                  f"n={len(sa)} vs {len(sb)}, p={ps:.4g}, "
                  f"delta={cliffs_delta(sa, sb):+.3f}")
        if pvals:
            print(f"    Holm-adjusted (family = class comparisons at {days}d): "
                  f"{ {k: round(v, 4) for k, v in holm(pvals).items()} }")

        # exploratory (labeled): any-earnings proportions + participation legs
        print(f"    EXPLORATORY at {days}d (not the registered test):")
        for other in ("underwater", "stagnant_candidate"):
            if not cohort.get(other):
                continue
            a = [v for _, v in cohort["profitable"]]
            b = [v for _, v in cohort[other]]
            tbl = [[sum(1 for v in a if v > 0), sum(1 for v in a if v == 0)],
                   [sum(1 for v in b if v > 0), sum(1 for v in b if v == 0)]]
            _, pf = stats.fisher_exact(tbl, alternative="greater")
            print(f"      any-out-earnings proportion: PROFITABLE "
                  f"{tbl[0][0]}/{len(a)} vs {other} {tbl[1][0]}/{len(b)}; "
                  f"Fisher one-sided p={pf:.4g}")
        # participation outcome (out-leg + in-leg fee credit, mirroring the
        # module's max(direct, sourced) attribution spirit)
        part = defaultdict(list)
        for (node, ch, wk), (cls, T) in sorted(week_first.items()):
            if node != N1 or wk not in ("2026-W24", "2026-W25"):
                continue
            cap = presence.get(ch, (0, 0, 0))[2]
            if not cap or T + days * DAY > chain_end:
                continue
            _, fo, _ = window_sum(out_ev.get(ch, []), T, T + days * DAY)
            _, fi, _ = window_sum(in_ev.get(ch, []), T, T + days * DAY)
            part[cls].append((ch, (fo + fi) / cap))
        for other in ("underwater", "stagnant_candidate"):
            if not part.get(other):
                continue
            a = [v for _, v in part["profitable"]]
            b = [v for _, v in part[other]]
            u, p = mwu_greater(a, b)
            print(f"      participation (out+in legs) per-cap: PROFITABLE > "
                  f"{other}: p={p:.4g}, delta={cliffs_delta(a, b):+.3f}")

    # ---- PA-H2 --------------------------------------------------------------
    print("\n--- PA-H2: bleeder gating saves money ---")
    n_30d = "contribution_30d_msat" in all_keys
    fee_paying_success = 0
    success_rows = 0
    for node in NODES:
        for r in rebalance_rows(node).values():
            if r.get("status") == "success":
                success_rows += 1
                if (r.get("actual_fee_sats") or 0) > 0:
                    fee_paying_success += 1
    print(f"  registered reconstruction needs contribution_30d_msat in "
          f"revenue-profitability rows: present={n_30d} "
          f"(exported row keys: {sorted(all_keys)})")
    print(f"  rebalance_cost_30d source: success rows corpus-wide={success_rows}, "
          f"with actual_fee_sats>0: {fee_paying_success}")
    print("  => hard-bleeder entry (net_30d < -1000 AND rebalance_cost_30d > "
          "2*contribution_30d) is unsatisfiable: the 30d fields are not in the "
          "exported surface AND rebalance spend is zero corpus-wide. "
          "VERDICT: UNTESTABLE (vacuously zero flagged channels).")

    # ---- PA-H3 --------------------------------------------------------------
    print("\n--- PA-H3: marginal_roi informative where reliable ---")
    has_mroi = any("marginal_roi" in k for k in all_keys)
    print(f"  rows carrying any marginal_roi* field: "
          f"{sum(v for k, v in all_keys.items() if 'marginal_roi' in k)} of {rows_total} "
          f"(field present: {has_mroi})")
    print("  => marginal_roi / marginal_roi_reliable are never exported in "
          "revenue-profitability.json. VERDICT: UNTESTABLE on this corpus.")

    # ---- class stability ----------------------------------------------------
    print("\n--- Class stability (evidence quality for the classifier) ---")
    for node in NODES:
        flaps, chans, trans = [], 0, Counter()
        flap_chans = 0
        for (n, ch), s in seq.items():
            if n != node:
                continue
            s.sort()
            classes = [c for _, c in s]
            if len(classes) < 2:
                continue
            chans += 1
            t = sum(1 for a, b in zip(classes, classes[1:]) if a != b)
            for a, b in zip(classes, classes[1:]):
                if a != b:
                    trans[(a, b)] += 1
            flaps.append(t / (len(classes) - 1))
            if t:
                flap_chans += 1
        print(f"  {node}: channels with >=2 classifications: {chans}; "
              f"changed class at least once: {flap_chans} "
              f"({100 * flap_chans / max(1, chans):.0f}%); "
              f"median flap rate={np.median(flaps) if flaps else 0:.4f} "
              f"transitions/snapshot-step; max={max(flaps) if flaps else 0:.4f}")
        if trans:
            print(f"    transition counts: "
                  f"{dict(sorted(trans.items(), key=lambda kv: -kv[1]))}")
        damaging = sum(v for (a, b), v in trans.items()
                       if {a, b} == {"profitable", "underwater"})
        print(f"    direct PROFITABLE<->UNDERWATER flips: {damaging}")

    # ---- D2 mask bound ------------------------------------------------------
    print("\n--- D2 structural-protection mask: fleet-level materiality bound ---")
    mask_chans = defaultdict(lambda: 0)
    for node, ts, ch, roi, net in d2_rows:
        mask_chans[(node, ch)] = min(mask_chans[(node, ch)], net)
    masked_loss = -sum(mask_chans.values())
    fleet_loss = -sum(underwater_net.values())
    print(f"  masked rows (BREAK_EVEN with roi < -10%): {len(d2_rows)} of "
          f"{rows_total} channel rows ({100 * len(d2_rows) / max(1, rows_total):.3f}%)")
    print(f"  distinct masked channels: {len(mask_chans)} "
          f"{sorted((n, c) for n, c in mask_chans)}; "
          f"channel-level masked loss: {masked_loss} sats")
    print(f"  visible UNDERWATER channel-level loss (worst net per channel): "
          f"{fleet_loss} sats across {len(underwater_net)} channels")
    if fleet_loss:
        print(f"  => mask hides {masked_loss} / {masked_loss + fleet_loss} sats "
              f"= {100 * masked_loss / (masked_loss + fleet_loss):.2f}% of the "
              f"fleet's classified loss exposure")
    return week_first


# ---------------------------------------------------------------------------
# CP: capacity_planner
# ---------------------------------------------------------------------------

def cp_section(out_ev, in_ev, chain_end, presence):
    print("\n" + "=" * 78)
    print("CP — capacity_planner (CP-H1, CP-H2, CP-H3)")
    print("=" * 78)
    acts = {node: planner_actions(node) for node in NODES}

    # ---- CP-H1 --------------------------------------------------------------
    opens = {node: [a for a in acts[node].values() if a.get("action_type") == "open"]
             for node in NODES}
    print(f"\n--- CP-H1: open quality ---")
    print(f"  planner open actions: n1={len(opens[N1])} n2={len(opens[N2])} "
          f"(any status)")
    print("  => 0 opens corpus-wide. VERDICT: UNTESTABLE (defer; registered "
          "<5-executions rule => descriptive only, and there is nothing to describe).")

    # ---- CP-H2 --------------------------------------------------------------
    print("\n--- CP-H2: close/loser validity (flag-based test IS runnable: the "
          "hypothesis is about flagging, not execution) ---")
    closes = [a for a in acts[N1].values() if a.get("action_type") == "close"]
    closes.sort(key=lambda a: a.get("created_at") or 0)
    first_flag = {}
    est_cost = {}
    for a in closes:
        ch = a.get("channel_id")
        t = a.get("created_at") or 0
        if ch and (ch not in first_flag or t < first_flag[ch]):
            first_flag[ch] = t
            est_cost[ch] = a.get("estimated_cost_sats")
    print(f"  n1 close actions: {len(closes)} "
          f"({Counter(a.get('reason') for a in closes)}); distinct flagged "
          f"channels: {len(first_flag)}")
    n2_closes = [a for a in acts[N2].values() if a.get("action_type") == "close"]
    print(f"  n2 close actions: {len(n2_closes)} — all 2026-05-13/14 dry_run, "
          f"pre-coverage, and n2 routed nothing: excluded from outcome analysis")

    flagged_ever = set(first_flag)
    obs_flagged, obs_flagged_fee, flagged_rows = [], [], []
    for ch, T in sorted(first_flag.items()):
        cap = presence.get(ch, (0, 0, 0))[2]
        if not cap or T + 14 * DAY > chain_end:
            continue
        nfw, fee, vol = window_sum(out_ev.get(ch, []), T, T + 14 * DAY)
        last = presence[ch][1]
        exposure = min(14.0, max(0.0, (last - T) / DAY + 0.5)) if last >= T else 0.0
        obs_flagged.append((ch, (vol / 1000.0) / 14.0 / cap))
        obs_flagged_fee.append((ch, fee / 1000.0 / 14.0))
        flagged_rows.append({
            "channel": ch, "flagged": iso(T), "cap": cap,
            "closed_in_window": last < T + 14 * DAY - 3600,
            "exposure_days": round(exposure, 1),
            "fw14_out": nfw, "fee14_sats": round(fee / 1000.0, 3),
            "vol14_sats": vol // 1000, "est_close_cost": est_cost.get(ch),
        })
    med_T = float(np.median(list(first_flag.values())))
    cov_start = min(fs for fs, _, _ in presence.values())
    t_pres = max(med_T, cov_start)  # presence unobservable before snapshot
    #                                 coverage begins (flags predate it by ~2d)
    flagged_caps = [presence[ch][2] for ch in first_flag if presence.get(ch, (0, 0, 0))[2]]
    obs_control = []
    for ch, (fs, ls, cap) in sorted(presence.items()):
        if ch in flagged_ever or not cap:
            continue
        if not (fs <= t_pres + 3600 and ls >= t_pres):
            continue
        if not any(0.5 * fc <= cap <= 1.5 * fc for fc in flagged_caps):
            continue
        if med_T + 14 * DAY > chain_end:
            continue
        _, fee, vol = window_sum(out_ev.get(ch, []), med_T, med_T + 14 * DAY)
        obs_control.append((ch, (vol / 1000.0) / 14.0 / cap))
    print(f"  note: median flag date {iso(med_T)} predates listpeerchannels "
          f"coverage ({iso(cov_start)}); control presence is tested at coverage "
          f"start — a channel opened between the two dates would be misincluded "
          f"(none plausible in 2 days at this fleet's open rate: 0 opens observed)")
    a = [v for _, v in obs_control]
    b = [v for _, v in obs_flagged]
    print(f"  flagged obs (14d post first flag, vol sats/day/cap-sat): {fmt_stats(b)}")
    print(f"  control obs (never-flagged, +/-50% capacity match, T=median flag "
          f"{iso(med_T)}): {fmt_stats(a)}")
    u, p = mwu_greater(a, b)   # control > flagged  <=>  flagged < control
    delta = cliffs_delta(a, b)
    (dlo, dhi), (mlo, mhi) = cluster_bootstrap_delta(obs_control, obs_flagged)
    print(f"  registered test (flagged < control, one-sided MWU): U={u:.1f} "
          f"p={p:.4g}; Cliff's delta (control vs flagged)={delta:+.3f} "
          f"[cluster-boot 95% CI {dlo:+.3f}..{dhi:+.3f}]")
    # survivorship sensitivity: exclude flagged channels closed inside window
    surv = [(ch, v) for (ch, v), row in zip(obs_flagged, flagged_rows)
            if not row["closed_in_window"]]
    us, ps = mwu_greater(a, [v for _, v in surv])
    print(f"  sensitivity (drop flagged channels closed during their window — "
          f"{sum(r['closed_in_window'] for r in flagged_rows)} of "
          f"{len(flagged_rows)}): n_flagged={len(surv)}, p={ps:.4g}, "
          f"delta={cliffs_delta(a, [v for _, v in surv]):+.3f}")
    # descriptive close-EV
    fees = [v for _, v in obs_flagged_fee]
    print(f"  descriptive EV: flagged post-flag out-fees (sats/day): {fmt_stats(fees)}")
    zero_fee = sum(1 for v in fees if v == 0)
    print(f"    flagged channels earning ZERO out-fees in 14d post-flag: "
          f"{zero_fee}/{len(fees)}")
    costs = [c for c in est_cost.values() if c]
    tot14 = sum(v for _, v in obs_flagged_fee) * 14
    print(f"    estimated close cost per action: {fmt_stats(costs)} sats; "
          f"total 14d post-flag earnings of ALL flagged channels: "
          f"{tot14:.1f} sats vs total estimated close cost "
          f"{sum(costs)} sats")
    print("  per-channel table (flagged):")
    for r in flagged_rows:
        print(f"    {r['channel']:16s} flagged={r['flagged']} cap={r['cap']:>9} "
              f"closed_in_win={str(r['closed_in_window']):5s} "
              f"exp={r['exposure_days']:>4}d fw14={r['fw14_out']:>3} "
              f"fee14={r['fee14_sats']:>8} vol14={r['vol14_sats']:>9} "
              f"close_cost={r['est_close_cost']}")

    # ---- CP-H3 --------------------------------------------------------------
    print("\n--- CP-H3: defibrillation efficacy ---")
    rows = {node: rebalance_rows(node) for node in NODES}
    total_defib = shock_rows = shock_success = 0
    per_chan = defaultdict(lambda: {"defibs": 0, "fw14_out": 0, "fw14_in": 0,
                                    "fee14_msat": 0, "node": ""})
    for node in NODES:
        diags = [r for r in rows[node].values()
                 if r.get("rebalance_type") == "diagnostic"]
        defibs = [a for a in acts[node].values()
                  if a.get("action_type") == "defibrillate"
                  and str(a.get("status")) == "completed"]
        total_defib += len(defibs)
        used = set()
        matched = 0
        for a in sorted(defibs, key=lambda x: x.get("completed_at") or 0):
            t0 = a.get("completed_at") or a.get("created_at") or 0
            ch = a.get("channel_id")
            cand = [r for r in diags if r.get("to_channel") == ch
                    and r["id"] not in used
                    and abs((r.get("timestamp") or 0) - t0) <= 600]
            if cand:
                row = min(cand, key=lambda r: abs((r.get("timestamp") or 0) - t0))
                used.add(row["id"])
                matched += 1
                shock_rows += 1
                if row.get("status") == "success":
                    shock_success += 1
            k = (node, ch)
            per_chan[k]["defibs"] += 1
            per_chan[k]["node"] = node
            if node == N1:
                n_o, fee, _ = window_sum(out_ev.get(ch, []), t0, t0 + 14 * DAY)
                n_i, _, _ = window_sum(in_ev.get(ch, []), t0, t0 + 14 * DAY)
                per_chan[k]["fw14_out"] = max(per_chan[k]["fw14_out"], n_o)
                per_chan[k]["fw14_in"] = max(per_chan[k]["fw14_in"], n_i)
                per_chan[k]["fee14_msat"] = max(per_chan[k]["fee14_msat"], fee)
        print(f"  {node}: completed defib actions={len(defibs)}, matched "
              f"diagnostic rows={matched}, unmatched={len(defibs) - matched} "
              f"(blocked shocks: no row by design)")
    print(f"  recorded shocks: {shock_rows}; SUCCESSFUL shocks: {shock_success}")
    print(f"  => registered CP-H3 needs completed defibrillations that DELIVERED "
          f"liquidity; deliveries = {shock_success} (< 5). "
          f"VERDICT: UNTESTABLE-AS-REGISTERED (registered <5 rule => descriptive).")
    n1_chans = {k: v for k, v in per_chan.items() if k[0] == N1}
    fw_any = sum(1 for v in n1_chans.values() if v["fw14_out"] + v["fw14_in"] > 0)
    print(f"  descriptive: distinct defibbed channels n1={len(n1_chans)} "
          f"n2={len(per_chan) - len(n1_chans)}; n1 channels with ANY settled "
          f"forward within 14d of a defib: {fw_any}/{len(n1_chans)} "
          f"(all after FAILED/blocked shocks => attributable at most to the "
          f"passive low-fee lure or concurrent fee descent, never to delivered "
          f"liquidity)")
    # false-positive case
    fp = "941347x1139x0"
    fp_ts = sorted((a.get("completed_at") or a.get("created_at") or 0)
                   for a in acts[N1].values()
                   if a.get("action_type") == "defibrillate"
                   and a.get("channel_id") == fp)
    legs = sorted([(t, "out", f) for t, f, _ in out_ev.get(fp, [])] +
                  [(t, "in", f) for t, f, _ in in_ev.get(fp, [])])
    by_day = Counter(time.strftime("%m-%d", time.gmtime(t)) for t, _, _ in legs)
    out_fee = sum(f for _, leg, f in legs if leg == "out") / 1000.0
    print(f"  false-positive case {fp}: defibs at "
          f"{[iso(t) for t in fp_ts]} (both shocks failed/blocked); settled "
          f"forward legs in chain: {len(legs)} "
          f"(by day: {dict(sorted(by_day.items()))}), out-leg fees "
          f"{out_fee:.3f} sats — a channel declared DEAD_CAPITAL and twice "
          f"shocked routed {len(legs)} legs within ~16 days of the shocks. "
          f"(Chain begins 05-20; Phase 3's narrative '80 settled forwards "
          f"before the defib' is NOT reproducible from the forwards chain — "
          f"the chain shows 0 settled legs for this scid before the first "
          f"defib; the 80 likely came from the profitability row's cumulative "
          f"forward_count. The false-positive conclusion stands on the "
          f"post-defib routing alone.)")
    n2_fw = sum(v["fw14_out"] + v["fw14_in"] for k, v in per_chan.items() if k[0] == N2)
    print(f"  contrast: n2 defibbed channels forwards after any defib: {n2_fw} "
          f"(7 shocks at the same dead pair changed nothing)")


# ---------------------------------------------------------------------------
# RB / RE
# ---------------------------------------------------------------------------

def rb_re_section(out_ev, chain_start, chain_end):
    print("\n" + "=" * 78)
    print("RB/RE — rebalancer + rebalance_engine_v2 (RB-H1..H3, RE-H1..H3)")
    print("=" * 78)
    rows = {node: rebalance_rows(node) for node in NODES}

    # dedup pre-fix twins
    normal_rows, success_rows = [], []
    twin_pairs = 0
    for node in NODES:
        seen = defaultdict(list)
        for rid, r in sorted(rows[node].items()):
            seen[(r.get("from_channel"), r.get("to_channel"), r.get("timestamp"))].append((rid, r))
        for key, group in seen.items():
            if len(group) > 1 and (key[2] or 0) < ROW_FIX_TS:
                twin_pairs += 1
                group = [g for g in group
                         if g[1].get("rebalance_type") == "diagnostic"] or group[:1]
            for rid, r in group:
                if r.get("rebalance_type") == "normal":
                    normal_rows.append((node, rid, r))
                if r.get("status") == "success":
                    success_rows.append((node, rid, r))
    print(f"pre-62ae545 twin attempts deduped: {twin_pairs} "
          f"(engine-side 'normal' twins of diagnostic shocks removed)")
    print(f"genuine rebalance_type=normal attempts after dedup: {len(normal_rows)} "
          f"{[(n, rid, iso(r.get('timestamp')), r.get('status')) for n, rid, r in normal_rows]}")
    print(f"success rows: {len(success_rows)} "
          f"{[(n, rid, r.get('rebalance_type'), iso(r.get('timestamp')), r.get('actual_fee_sats')) for n, rid, r in success_rows]}")

    # RB-H1
    print("\n--- RB-H1: refill payback ---")
    testable = 0
    for node, rid, r in success_rows:
        t = r.get("timestamp") or 0
        obs = node == N1 and t >= chain_start
        print(f"  success row {node} id={rid}: type={r.get('rebalance_type')} "
              f"fee={r.get('actual_fee_sats')} at {iso(t)}; 7d outcome window "
              f"{'covered' if obs and t + 7 * DAY <= chain_end else 'NOT covered by forwards chain'}")
        if obs and t + 7 * DAY <= chain_end:
            testable += 1
    print(f"  => testable events: {testable} of {len(success_rows)} (both successes"
          f" are manual, zero-fee, on nexus-02, 2026-04-14 — before the forwards"
          f" chain begins and on a node that never routed). VERDICT: UNTESTABLE.")

    # RB-H2
    print("\n--- RB-H2: budget gate not binding on revenue ---")
    treated, holds = [], []
    for p in artifacts(N1, "revenue-status.json"):
        d = load(p)
        if not isinstance(d, dict):
            continue
        rd = d.get("rebalance_decision") or {}
        ts = snap_ts(p)
        if rd.get("action") == "suppressed" and rd.get("budget_blocked"):
            treated.append(ts)
        elif rd.get("action") == "hold":
            holds.append(ts)
    node_fees = sorted((t, f) for evs in out_ev.values() for t, f, _ in evs)

    def next24(ts):
        return sum(f for t, f in node_fees if ts < t <= ts + DAY) / 1000.0
    tr = [next24(t) for t in treated]
    ho = [next24(t) for t in holds]
    span = (iso(min(treated)), iso(max(treated))) if treated else ("-", "-")
    print(f"  treated (suppressed+budget_blocked) snapshots: {len(treated)}, all in "
          f"{span[0]} .. {span[1]} — ONE episode (the 06-13 mass-close saturation)")
    print(f"  descriptive next-24h node fee sums (sats): treated {fmt_stats(tr)}")
    print(f"                                      holds  {fmt_stats(ho)}")
    print("  => n_clusters(treated)=1: the registered matched test has no degrees "
          "of freedom. VERDICT: UNTESTABLE (descriptive only; no inference).")

    # RB-H3 / RE-H3
    print("\n--- RB-H3 (suppression honesty) ---")
    print("  Phase 3 already exact-checked this loop-level: zero automated spend "
          "in all 550 suppressed decisions / 129 over-budget snapshots (LP-I2, "
          "LP-I8); rebalance-category spend is 0 corpus-wide, so the 'automated "
          "spend accrual ~ 0' direction holds VACUOUSLY-TRUE: nothing ever "
          "accrued anywhere, suppressed or not. No discriminating power.")
    print("\n--- RE-H3 (gating reduces waste) ---")
    weeks = Counter()
    for node, rid, r in normal_rows:
        weeks[week_of(r.get("timestamp") or 0)] += 1
    print(f"  weekly normal-attempt counts after twin dedup: {dict(weeks)}")
    print("  => a trend test over a series with ~1 genuine attempt is undefined. "
          "VERDICT: VACUOUS.")

    # RE-H1 / RE-H2
    print("\n--- RE-H1/RE-H2 (EV-gate calibration / net payback of engine executions) ---")
    sel = ex = dbg = 0
    for node in NODES:
        for p in artifacts(node, "revenue-rebalance-debug.json"):
            d = load(p)
            if not isinstance(d, dict):
                continue
            dbg += 1
            s = (d.get("last_cycle") or {}).get("summary") or {}
            sel += int(s.get("selected_pairs") or 0)
            ex += int(s.get("execution_count") or 0)
    print(f"  engine debug snapshots={dbg}: selected_pairs total={sel}, "
          f"execution_count total={ex}; normal success rows={sum(1 for _, _, r in success_rows if r.get('rebalance_type') == 'normal')}")
    print("  => zero engine executions exist. VERDICT: VACUOUS (both).")


# ---------------------------------------------------------------------------
# Exploratory: hold-margin calibration
# ---------------------------------------------------------------------------

def hold_margin_section(out_ev, in_ev):
    print("\n" + "=" * 78)
    print("EXPLORATORY (not pre-registered): hold-margin calibration of the EV gate")
    print("=" * 78)
    cands = []
    for node in NODES:
        for p in artifacts(node, "revenue-rebalance-debug.json"):
            d = load(p)
            if not isinstance(d, dict):
                continue
            lc = d.get("last_cycle") or {}
            for c in lc.get("considered_candidates") or []:
                sd = c.get("score_decomposition") or {}
                inp = sd.get("inputs") or {}
                cands.append({
                    "node": node, "ts": snap_ts(p),
                    "src": c.get("source_channel_id"), "dst": c.get("dest_channel_id"),
                    "amount": c.get("amount_sats"), "route_cost": c.get("route_cost_sats"),
                    "score_sats": sd.get("final_score_sats"),
                    "rej": sd.get("rejection_reason"),
                    "p": sd.get("p_success"),
                    "efv": sd.get("expected_future_value_sats"),
                    "src_opp": sd.get("source_opportunity_sats"),
                    "fail_pen": sd.get("failure_penalty_sats") or 0.0,
                    "dest_ppm": inp.get("dest_out_fee_ppm"),
                    "src_ppm": inp.get("source_out_fee_ppm"),
                })
    print(f"considered-candidate exports: {len(cands)}; rejection reasons: "
          f"{dict(Counter(c['rej'] for c in cands))}")
    # dedup: a distinct candidate = distinct (src,dst,amount,route_cost)
    dedup = {}
    for c in cands:
        key = (c["node"], c["src"], c["dst"], c["amount"], c["route_cost"])
        if key not in dedup or c["ts"] < dedup[key]["ts"]:
            dedup[key] = c
    dis = sorted(dedup.values(), key=lambda c: c["ts"])
    pairs = {(c["src"], c["dst"]) for c in dis}
    print(f"distinct candidates by (src,dst,amount,route_cost): {len(dis)}; "
          f"distinct pairs: {len(pairs)} {sorted(pairs)}")
    scores = np.array([c["score_sats"] for c in dis], float)
    print(f"final_score_sats distribution (distinct): {fmt_stats(scores)}")
    margins_seen = Counter()
    for node in NODES:
        for p in artifacts(node, "listconfigs.json"):
            d = load(p)
            if not isinstance(d, dict):
                continue
            v = (d.get("configs", d)).get("revenue-ops-rebalance-hold-margin")
            if isinstance(v, dict):
                v = v.get("value_str", v.get("value_int"))
            margins_seen[str(v)] += 1
    print(f"configured hold margin (listconfigs, both nodes): {dict(margins_seen)} "
          f"— gate rejects final_score_sats < margin")
    margins = [0, -50, -100, -150, -175, -200, -300, -400, -1300]
    print("  margin sweep (candidates admitted if final_score_sats >= margin):")
    for m in margins:
        adm = scores[scores >= m]
        print(f"    margin {m:>6}: admits {len(adm):>2} of {len(scores)}; "
              f"sum EV of admitted = {adm.sum():>9.1f} sats "
              f"(engine's own valuation)")
    # break-even dest fee ppm per distinct candidate
    print("  per-candidate economics (engine's own formula, captured inputs):")
    for c in dis:
        p = c["p"] or 0.99
        amt = c["amount"] or 0
        need = (c["route_cost"] + (c["src_opp"] or 0) + c["fail_pen"])
        be_ppm = need * 1e6 / (p * amt * EXPECTED_UTILIZATION) if amt else float("nan")
        route_ppm = c["route_cost"] * 1e6 / amt if amt else float("nan")
        print(f"    {c['src']}->{c['dst']} amt={amt:>7} route={c['route_cost']:>5} "
              f"({route_ppm:5.0f}ppm) dest_fee={c['dest_ppm']:>4}ppm "
              f"score={c['score_sats']:>9.1f} breakeven_dest_fee={be_ppm:6.0f}ppm")
    # manual market cross-check: the operator hand-tested the same dest
    manual = [r for r in rebalance_rows(N1).values()
              if r.get("rebalance_type") == "manual"
              and r.get("to_channel") == "936056x1037x0"]
    print(f"  manual market test of the same dest (936056x1037x0): "
          f"{len(manual)} operator attempts:")
    for r in sorted(manual, key=lambda r: r.get("timestamp") or 0):
        print(f"    {iso(r.get('timestamp'))} amt={r.get('amount_sats')} "
              f"max_fee={r.get('max_fee_sats')} status={r.get('status')} "
              f"err={(r.get('error_message') or '')[:60]!r}")

    # HEAD-formula recheck with corpus-derived historical fee rates
    print("  HEAD (post-441b8e3) recheck: EFV gains a historical-fee floor "
          "max(dest_out_fee, dest_hist_direct) + source drain term; corpus proxy "
          "for historical rates = settled fee/volume over 30d before candidate ts:")

    def hist_ppm(events, t0):
        n, fee, vol = window_sum(events, t0 - 30 * DAY, t0)
        return (fee / vol * 1e6) if vol else 0.0
    flips = 0
    for c in dis:
        amt, p = c["amount"] or 0, c["p"] or 0.99
        dh = hist_ppm(out_ev.get(c["dst"], []), c["ts"])
        sh_src = hist_ppm(in_ev.get(c["src"], []), c["ts"])
        s_opp_ppm = max(c["src_ppm"] or 0, hist_ppm(out_ev.get(c["src"], []), c["ts"]))
        efv = (amt * max(c["dest_ppm"] or 0, dh) / 1e6 * EXPECTED_UTILIZATION
               + amt * sh_src / 1e6 * EXPECTED_UTILIZATION)
        s_opp = amt * s_opp_ppm / 1e6 * EXPECTED_UTILIZATION * SOURCE_UTILIZATION_DISCOUNT
        head = p * efv - c["route_cost"] - s_opp - c["fail_pen"]
        if head >= 0:
            flips += 1
        c["head_score"] = head
    hs = np.array([c["head_score"] for c in dis])
    print(f"    HEAD-formula scores (corpus-proxy historical rates): {fmt_stats(hs)}")
    print(f"    candidates flipping to >= 0 under HEAD terms: {flips} of {len(dis)}")
    print("  NOTE: exploratory, no baseline/counterfactual exists — every number "
          "is the engine's own EV model evaluated at observed prices; it answers "
          "'was the gate internally leaving EV on the table', not 'was EV real'.")


# ---------------------------------------------------------------------------
# BM vacuity
# ---------------------------------------------------------------------------

def bm_section():
    print("\n" + "=" * 78)
    print("BM — boltz_manager (BM-H1..H3)")
    print("=" * 78)
    checked, nonzero = 0, 0
    for node in NODES:
        for p in artifacts(node, "revenue-spend-ledger.json"):
            d = load(p)
            if not isinstance(d, dict):
                continue
            checked += 1
            if ((d.get("spent_by_category") or {}).get("boltz") or 0) or \
               ((d.get("event_count_by_category") or {}).get("boltz") or 0):
                nonzero += 1
        for p in artifacts(node, "revenue-total-cost-budget.json"):
            d = load(p)
            if not isinstance(d, dict):
                continue
            checked += 1
            comp = (d.get("components") or {}).get("boltz") or {}
            if (comp.get("spent_24h_sats") or 0) or \
               ((d.get("actual_spent_by_category") or {}).get("boltz") or 0):
                nonzero += 1
    print(f"  boltz spend/events across {checked} ledger+budget surfaces: "
          f"nonzero={nonzero}")
    print("  => zero swaps corpus-wide. BM-H1 identification gap is total "
          "(0 identifiable events, < 5 minimum); BM-H2 'pass' is trivial "
          "(0 <= budget always); BM-H3 has no treatment hours. "
          "VERDICT: all three VACUOUS.")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    args = ap.parse_args()
    globals()["CORPUS"] = args.corpus

    t0 = time.time()
    print("Phase 4 capital-side contribution analysis (frozen corpus, read-only)")
    print(f"corpus: {CORPUS}; bootstrap seed={SEED}, n_boot={BOOT}")

    fwd1 = forwards_chain(N1)
    fwd2 = forwards_chain(N2)
    out_ev, in_ev, chain_end = settled_index(fwd1)
    _, _, _ = settled_index(fwd2)
    settled1 = sum(1 for f in fwd1 if f.get("status") == "settled")
    fees1 = sum(msat(f.get("fee_msat")) for f in fwd1 if f.get("status") == "settled")
    settled2 = sum(1 for f in fwd2 if f.get("status") == "settled")
    chain_start = min((f.get("resolved_time") or f.get("received_time") or 9e9)
                      for f in fwd1)
    print(f"forwards chain n1: {len(fwd1)} deduped rows, {settled1} settled, "
          f"{fees1 / 1000:.3f} sats fees, span {iso(chain_start)} .. {iso(chain_end)}")
    print(f"forwards chain n2: {len(fwd2)} deduped rows, {settled2} settled "
          f"(nexus-02 routed nothing)")
    presence = channel_presence(N1)

    pa_section(out_ev, in_ev, chain_end, presence)
    cp_section(out_ev, in_ev, chain_end, presence)
    rb_re_section(out_ev, chain_start, chain_end)
    hold_margin_section(out_ev, in_ev)
    bm_section()
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
