#!/usr/bin/env python3
"""Phase 4 contribution analysis: fee-side hypotheses + revenue decomposition.

Tests the PRE-REGISTERED fee-side hypotheses (fee_controller.md FC-H1..H3,
flow_analysis.md FA-H1..H3, policy_manager.md PM-H1..H3, hive_hints.md
HH-H1..H3) against the frozen hermes corpus, and builds the per-channel /
per-day revenue decomposition that anchors the Phase 4 campaign.

Read-only over the corpus and the repo. Deterministic (seeded bootstrap);
running it regenerates every number cited in
docs/audit/contribution/fee-side.md.

Data model (established Phase 0-3):
  - Settled-forward ground truth: the lossless deduplicated
    listforwards-window chain (dedup key: created_index, updated_index,
    in_channel, in_htlc_id, status), 2026-05-20 -> 2026-07-01 by
    updated_index. Fee revenue is attributed to the OUT channel (the channel
    whose advertised policy priced the forward; verified: fee_msat/out_msat
    == advertised ppm) at resolved_time (UTC).
  - Fee-change records: deduplicated rolling 10-row `recent_fee_changes`
    windows across all revenue-status snapshots. nexus-01 lost ~61% of ids
    to window overflow; every treatment-identification step below quantifies
    that loss. nexus-02 recovery is 99.8%.
  - Advertised fees / liquidity: hourly listpeerchannels snapshots
    (2026-06-09 -> 06-20 + one 07-01 capture; 10-day hole 06-21..06-30).
  - Flow labels: revenue-status channel_states (~5-min cadence).
  - Deploy epochs (corpus metadata code-stamp transitions, corroborating not
    authoritative): climb governor 9f8f219 first stamped 2026-06-12T13:13:43Z;
    zero-flow ratchet 071a5b3/245ac12 2026-06-15T19:22:38Z; flow-hysteresis F1
    (2df8d92) 2026-06-12T00:41:52Z; member zero-fee 8630ca6 in the corpus hole
    (earliest hive_member_zero_fee record 2026-06-27T22:27:01Z).

Sections (run all: no args; or --section decomp|fch1|fch2|fch3|e2|hh|pm|fa):
  decomp  Revenue decomposition + reconciliation vs plugin surfaces
  fch1    FC-H1 DTS repricing of stagnant channels vs matched holding controls
  fch2    FC-H2 climb governor overshoot / earnings before-after
  fch3    FC-H3 rebalance-floored channels (activation-population census)
  e2      EXPLORATORY: external-hand elasticity episode on 946890x2272x0
  hh      HH-H1..H3 (freshness contrast, fleet prior ITT, corridor-owner bias)
  pm      PM-H1..H3 (policy-transition population census)
  fa      FA-H1 label predictiveness, FA-H2 hysteresis, FA-H3 Kalman
          depletion replay (imports modules.flow_analysis for the exact
          filter arithmetic; observation pipeline reconstructed per
          flow_analysis.py:_compute_raw_kalman_observation /
          _calculate_confidence / _calculate_kalman_volatility with
          production cadence flow-interval=3600s, flow_window_days=7)
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

CORPUS = os.environ.get("CL_MYCELIUM_HERMES_ROOT", "/home/sat/cl-mycelium-hermes")
NODES = ("hive-nexus-01", "hive-nexus-02")
N1, N2 = NODES

SEED = 20260701
BOOT_N = 10000

# Deploy epochs (unix). Sources: corpus metadata.json code-stamp transitions
# (identical on both nodes; collector-host working tree -> corroborating).
GOV_DEPLOY = 1781270023        # 2026-06-12T13:13:43Z first 9f8f219 stamp
RATCHET_DEPLOY = 1781551358    # 2026-06-15T19:22:38Z first 071a5b3 stamp
F1_DEPLOY = 1781224912         # 2026-06-12T00:41:52Z first 2df8d92 stamp
EXTERNAL_CHANNEL = "946890x2272x0"   # three external fee writes + 06-15 opt-out
MASS_CLOSE_DAY = "2026-06-13"

DAY = 86400.0


def ts_of(stamp: str) -> int:
    return int(datetime.strptime(stamp, "%Y%m%dT%H%M%SZ")
               .replace(tzinfo=timezone.utc).timestamp())


def iso(ts) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def day_of(ts) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def fee_msat_int(v):
    if isinstance(v, str) and v.endswith("msat"):
        v = v[:-4]
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------
# Corpus loading
# --------------------------------------------------------------------------

class Forwards:
    """Lossless deduplicated listforwards chain for one node."""

    def __init__(self, node):
        self.node = node
        self.settled = []            # (resolved_ts, out_ch, in_ch, fee_msat, out_msat)
        self.all_status = defaultdict(int)
        seen = set()
        for path in sorted(glob.glob(os.path.join(
                CORPUS, node, "*", "*", "commands", "listforwards-window.json.gz"))):
            try:
                with gzip.open(path, "rt", encoding="utf-8") as fh:
                    payload = json.loads(fh.read(), strict=False)
            except Exception:
                continue
            for fwd in payload.get("forwards") or []:
                key = (fwd.get("created_index"), fwd.get("updated_index"),
                       fwd.get("in_channel"), fwd.get("in_htlc_id"),
                       fwd.get("status"))
                if key in seen:
                    continue
                seen.add(key)
                self.all_status[fwd.get("status")] += 1
                if fwd.get("status") == "settled":
                    self.settled.append((
                        float(fwd.get("resolved_time") or fwd.get("received_time") or 0),
                        fwd.get("out_channel"), fwd.get("in_channel"),
                        fee_msat_int(fwd.get("fee_msat")),
                        fee_msat_int(fwd.get("out_msat"))))
        self.settled.sort()
        self.total_fee_msat = sum(s[3] for s in self.settled)
        # per-out-channel time-sorted settled events
        self.by_out = defaultdict(list)
        for s in self.settled:
            self.by_out[s[1]].append(s)
        # per-channel daily fee (out attribution) and daily in/out volume sats
        self.daily_fee = defaultdict(lambda: defaultdict(float))   # ch -> day -> sats
        self.daily_out = defaultdict(lambda: defaultdict(float))   # ch -> day -> sats out
        self.daily_in = defaultdict(lambda: defaultdict(float))    # ch -> day -> sats in
        for ts, out_ch, in_ch, fee, out_msat in self.settled:
            d = day_of(ts)
            self.daily_fee[out_ch][d] += fee / 1000.0
            self.daily_out[out_ch][d] += out_msat / 1000.0
            self.daily_in[in_ch][d] += (out_msat + fee) / 1000.0

    def fees_in(self, ch, t0, t1):
        """Settled fee sats attributed to out-channel ch in (t0, t1]."""
        evs = self.by_out.get(ch, [])
        return sum(e[3] for e in evs if t0 < e[0] <= t1) / 1000.0

    def first_settled_after(self, ch, t0, t1):
        for e in self.by_out.get(ch, []):
            if t0 < e[0] <= t1:
                return e[0]
        return None

    def node_fees(self, t0, t1, exclude=None):
        return sum(e[3] for e in self.settled
                   if t0 < e[0] <= t1 and e[1] != exclude) / 1000.0


class Node:
    """Snapshot-derived surfaces for one node (loaded once)."""

    def __init__(self, node):
        self.node = node
        self.changes = {}          # id -> record
        self.snap_times = []       # revenue-status snapshot ts
        self.states = {}           # ts -> {chan: state}
        self.lpc = []              # (ts, {scid: dict(fee, base, spend, total, peer)})
        self.hints_status = []     # (ts, fresh, usable, stale_fallback)
        self.owner_roles = []      # (ts, frozenset(owner peer ids))
        self.members = []          # (ts, frozenset(member peer ids))
        self.rh = []               # (ts, lifetime dict)
        self.dashboards = []       # (ts, dict)
        self.spend_ledgers = []    # (ts, dict)
        self._load()

    def _load(self):
        for cmd_dir in sorted(glob.glob(os.path.join(
                CORPUS, self.node, "2*", "2*", "commands"))):
            ts = ts_of(os.path.basename(os.path.dirname(cmd_dir)))
            rs = load_json(os.path.join(cmd_dir, "revenue-status.json"))
            if rs is not None:
                self.snap_times.append(ts)
                self.states[ts] = {
                    st.get("channel_id"): (st.get("state") or "").lower()
                    for st in rs.get("channel_states") or []}
                for ch in rs.get("recent_fee_changes") or []:
                    cid = ch.get("id")
                    if cid is not None and cid not in self.changes:
                        self.changes[cid] = ch
            lpc = load_json(os.path.join(cmd_dir, "listpeerchannels.json"))
            if lpc is not None:
                chans = {}
                for c in lpc.get("channels") or []:
                    scid = c.get("short_channel_id")
                    if not scid or c.get("state") != "CHANNELD_NORMAL":
                        continue
                    fee = c.get("fee_proportional_millionths")
                    if fee is None:
                        fee = ((c.get("updates") or {}).get("local") or {}) \
                            .get("fee_proportional_millionths")
                    chans[scid] = {
                        "fee": fee,
                        "spend": fee_msat_int(c.get("spendable_msat")),
                        "total": fee_msat_int(c.get("total_msat")),
                        "peer": c.get("peer_id")}
                self.lpc.append((ts, chans))
            hs = load_json(os.path.join(cmd_dir, "revenue-hive-hints-status.json"))
            if hs is not None:
                self.hints_status.append((
                    ts, bool(hs.get("snapshot_fresh")), bool(hs.get("snapshot_usable")),
                    bool(hs.get("stale_fallback") or hs.get("stale_fallback_active"))))
            eh = load_json(os.path.join(cmd_dir, "hive-export-hints.json"))
            hints = (eh or {}).get("hints")
            if isinstance(hints, dict):
                self.owner_roles.append((ts, frozenset(
                    p for p, h in hints.items() if isinstance(h, dict)
                    and h.get("corridor_role") == "owner")))
                self.members.append((ts, frozenset(
                    p for p, h in hints.items()
                    if isinstance(h, dict) and h.get("member"))))
            rh = load_json(os.path.join(cmd_dir, "revenue-history.json"))
            if rh is not None and "lifetime_revenue_sats" in rh:
                self.rh.append((ts, rh))
            db = load_json(os.path.join(cmd_dir, "revenue-dashboard.json"))
            if db is not None:
                self.dashboards.append((ts, db))
            sl = load_json(os.path.join(cmd_dir, "revenue-spend-ledger.json"))
            if sl is not None:
                self.spend_ledgers.append((ts, sl))
        self.snap_times.sort()
        self.lpc.sort(key=lambda x: x[0])
        self.lpc_times = [t for t, _ in self.lpc]
        # recorded-change per-channel index + id gaps
        self.by_chan = defaultdict(list)
        for cid in sorted(self.changes):
            self.by_chan[self.changes[cid]["channel_id"]].append(self.changes[cid])
        ids = sorted(self.changes)
        self.gaps = [(self.changes[a]["timestamp"], self.changes[b]["timestamp"])
                     for a, b in zip(ids, ids[1:]) if b - a > 1]
        self.lost_ids = sum(b - a - 1 for a, b in zip(ids, ids[1:]) if b - a > 1)
        self.recovered_ids = len(ids)
        # first LPC appearance per scid (existence evidence)
        self.first_lpc = {}
        for t, chans in self.lpc:
            for scid in chans:
                self.first_lpc.setdefault(scid, t)

    def gap_overlaps(self, t0, t1):
        return any(g0 <= t1 and g1 >= t0 for g0, g1 in self.gaps)

    def lpc_at(self, ts, max_skew=7200):
        """Nearest LPC snapshot <= ts (within max_skew)."""
        i = bisect_right(self.lpc_times, ts) - 1
        if i >= 0 and ts - self.lpc_times[i] <= max_skew:
            return self.lpc[i]
        return None

    def lpc_near(self, ts, tol):
        """Nearest LPC snapshot within +-tol of ts."""
        i = bisect_left(self.lpc_times, ts)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self.lpc):
                d = abs(self.lpc_times[j] - ts)
                if d <= tol and (best is None or d < best[0]):
                    best = (d, self.lpc[j])
        return best[1] if best else None

    def fee_series(self, scid, t0, t1):
        """(ts, fee) points from LPC in [t0, t1]."""
        out = []
        i = bisect_left(self.lpc_times, t0)
        while i < len(self.lpc) and self.lpc_times[i] <= t1:
            c = self.lpc[i][1].get(scid)
            if c and c["fee"] is not None:
                out.append((self.lpc_times[i], c["fee"]))
            i += 1
        return out


_NODES_CACHE = {}
_FWD_CACHE = {}


def get_node(n) -> Node:
    if n not in _NODES_CACHE:
        _NODES_CACHE[n] = Node(n)
    return _NODES_CACHE[n]


def get_fwd(n) -> Forwards:
    if n not in _FWD_CACHE:
        _FWD_CACHE[n] = Forwards(n)
    return _FWD_CACHE[n]


# --------------------------------------------------------------------------
# Stats helpers
# --------------------------------------------------------------------------

def boot_ci(values, statfn=np.mean, n=BOOT_N, seed=SEED, alpha=0.05):
    """Percentile bootstrap CI for statfn(values)."""
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    boots = statfn(a[idx], axis=1)
    return (float(statfn(a)), float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


def boot_ratio_ci(after, before, n=BOOT_N, seed=SEED, alpha=0.05):
    """Bootstrap CI for mean(after)/mean(before) - 1 (unpaired, two-sample)."""
    a, b = np.asarray(after, float), np.asarray(before, float)
    if len(a) == 0 or len(b) == 0 or b.mean() == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    ra = a[rng.integers(0, len(a), size=(n, len(a)))].mean(axis=1)
    rb = b[rng.integers(0, len(b), size=(n, len(b)))].mean(axis=1)
    ok = rb != 0
    r = ra[ok] / rb[ok] - 1.0
    return (float(a.mean() / b.mean() - 1.0),
            float(np.percentile(r, 100 * alpha / 2)),
            float(np.percentile(r, 100 * (1 - alpha / 2))))


def mwu(x, y, alternative):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) == 0 or len(y) == 0:
        return float("nan"), float("nan")
    res = stats.mannwhitneyu(x, y, alternative=alternative)
    return float(res.statistic), float(res.pvalue)


def holm(pvals: dict):
    """Holm-Bonferroni adjusted p-values for the confirmatory family."""
    items = sorted((p, k) for k, p in pvals.items() if not math.isnan(p))
    m = len(items)
    out, running = {}, 0.0
    for i, (p, k) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


CONFIRMATORY_P = {}


# ==========================================================================
# Section 1 — revenue decomposition + reconciliation
# ==========================================================================

def sec_decomp():
    print("=" * 78)
    print("SECTION 1 — REVENUE DECOMPOSITION (anchor) + RECONCILIATION")
    print("=" * 78)
    for node in NODES:
        fwd = get_fwd(node)
        tot = fwd.total_fee_msat / 1000.0
        print(f"\n-- {node}: chain totals --")
        print(f"  forwards deduped: {sum(fwd.all_status.values())} "
              f"status={dict(fwd.all_status)}")
        print(f"  settled: {len(fwd.settled)}  settled fees: {tot:.3f} sats")
        if fwd.settled:
            print(f"  span: {iso(fwd.settled[0][0])} .. {iso(fwd.settled[-1][0])}")
        if not fwd.settled:
            continue

        # per-channel concentration
        per_ch = sorted(((sum(v.values()), ch) for ch, v in fwd.daily_fee.items()),
                        reverse=True)
        total = sum(x for x, _ in per_ch)
        print(f"\n  per-OUT-channel settled fees (n={len(per_ch)} earning channels):")
        cum = 0.0
        for i, (sats, ch) in enumerate(per_ch[:10]):
            cum += sats
            days = len(fwd.daily_fee[ch])
            n_f = len(fwd.by_out[ch])
            print(f"   {i+1:2d}. {ch:>16s} {sats:10.1f} sats "
                  f"({100*sats/total:5.1f}%, cum {100*cum/total:5.1f}%) "
                  f"{n_f} settled, {days} earning days")
        shares = np.array([x / total for x, _ in per_ch])
        hhi = float((shares ** 2).sum())
        print(f"  HHI (channel concentration): {hhi:.3f}  "
              f"top-1 {100*shares[0]:.1f}%  top-3 {100*shares[:3].sum():.1f}%  "
              f"top-5 {100*shares[:5].sum():.1f}%")

        # fee-band concentration (implied ppm = fee/out amount)
        bands = [(0, 50), (50, 100), (100, 150), (150, 250), (250, 500),
                 (500, 1000), (1000, 10**9)]
        band_fees = defaultdict(float)
        band_cnt = defaultdict(int)
        for ts, out_ch, in_ch, fee, out_msat in fwd.settled:
            ppm = 1e6 * fee / out_msat if out_msat else 0.0
            for lo, hi in bands:
                if lo <= ppm < hi:
                    band_fees[(lo, hi)] += fee / 1000.0
                    band_cnt[(lo, hi)] += 1
                    break
        print("  fee-band decomposition (implied ppm of each settled forward):")
        for lo, hi in bands:
            f = band_fees.get((lo, hi), 0.0)
            c = band_cnt.get((lo, hi), 0)
            hi_s = f"{hi}" if hi < 10**9 else "inf"
            print(f"    [{lo:>4d},{hi_s:>4s}) ppm: {f:9.1f} sats "
                  f"({100*f/total:5.1f}%)  {c:4d} forwards")

        # day profile
        daily = defaultdict(float)
        for ch, dd in fwd.daily_fee.items():
            for d, s in dd.items():
                daily[d] += s
        days_sorted = sorted(daily.items())
        vals = sorted(daily.values(), reverse=True)
        print(f"  day profile: {len(days_sorted)} earning days; "
              f"top day {vals[0]:.1f} sats ({100*vals[0]/total:.1f}%), "
              f"top-3 days {100*sum(vals[:3])/total:.1f}%, "
              f"top-7 days {100*sum(vals[:7])/total:.1f}%")
        print("  daily settled fees (sats):")
        for d, s in days_sorted:
            bar = "#" * min(60, int(s / 60))
            print(f"    {d} {s:9.1f} {bar}")

        # concentration cross-cut: which channel earns in which band
        print("  top-3 channels x fee bands (sats):")
        for sats, ch in per_ch[:3]:
            row = defaultdict(float)
            for e in fwd.by_out[ch]:
                ppm = 1e6 * e[3] / e[4] if e[4] else 0.0
                for lo, hi in bands:
                    if lo <= ppm < hi:
                        row[(lo, hi)] += e[3] / 1000.0
            cells = " ".join(f"[{lo}-{hi if hi<10**9 else 'inf'})={row.get((lo,hi),0):.0f}"
                             for lo, hi in bands if row.get((lo, hi), 0) >= 0.5)
            print(f"    {ch}: {cells}")

    # ---- reconciliation ----------------------------------------------------
    print("\n-- reconciliation vs plugin surfaces (2026-07-01 snapshot) --")
    for node in NODES:
        nd = get_node(node)
        fwd = get_fwd(node)
        if not nd.rh:
            continue
        (t0, rh0), (t1, rh1) = nd.rh[0], nd.rh[-1]
        chain = fwd.node_fees(t0, t1)
        chain_cnt = sum(1 for e in fwd.settled if t0 < e[0] <= t1)
        d_rev = rh1["lifetime_revenue_sats"] - rh0["lifetime_revenue_sats"]
        d_cnt = rh1["lifetime_forward_count"] - rh0["lifetime_forward_count"]
        print(f"  {node}: lifetime-counter delta {iso(t0)}..{iso(t1)}: "
              f"plugin {d_rev} sats / {d_cnt} fwds; chain {chain:.3f} sats / "
              f"{chain_cnt} fwds; residual {chain - d_rev:+.3f} sats / "
              f"{chain_cnt - d_cnt:+d} fwds")
        # localize the residual between adjacent revenue-history snapshots
        locs = []
        for (ta, ra), (tb, rb) in zip(nd.rh, nd.rh[1:]):
            pr = rb["lifetime_revenue_sats"] - ra["lifetime_revenue_sats"]
            pc = rb["lifetime_forward_count"] - ra["lifetime_forward_count"]
            cr = fwd.node_fees(ta, tb)
            cc = sum(1 for e in fwd.settled if ta < e[0] <= tb)
            if abs(cr - pr) >= 5 or cc != pc:
                locs.append((ta, tb, cr - pr, cc - pc))
        if locs:
            print(f"    residual localization (intervals with |fee diff|>=5 "
                  f"sats or count diff, n={len(locs)}):")
            for ta, tb, dr, dc in locs:
                print(f"      {iso(ta)}..{iso(tb)}: chain-plugin "
                      f"{dr:+.3f} sats / {dc:+d} fwds")
        if nd.dashboards:
            td, dash = nd.dashboards[-1]
            per = dash.get("period") or {}
            w = per.get("window_days", 30)
            chain_w = fwd.node_fees(td - w * DAY, td)
            cnt_w = sum(1 for e in fwd.settled if td - w * DAY < e[0] <= td)
            print(f"  {node}: dashboard {w}d @{iso(td)}: gross_revenue="
                  f"{per.get('gross_revenue_sats')} sats / "
                  f"{per.get('forward_count')} fwds; chain same window "
                  f"{chain_w:.3f} sats / {cnt_w} fwds; delta "
                  f"{per.get('gross_revenue_sats', 0) - chain_w:+.3f} sats / "
                  f"{per.get('forward_count', 0) - cnt_w:+d} fwds")


# ==========================================================================
# Section 2 — FC-H1
# ==========================================================================

def zero_rev_streak_days(fwd: Forwards, ch, event_ts, need=3):
    """True if the `need` full UTC days before event day had zero settled fees."""
    for k in range(1, need + 1):
        d = day_of(event_ts - k * DAY)
        if fwd.daily_fee.get(ch, {}).get(d, 0.0) > 0:
            return False
    return True


def channel_existed(nd: Node, fwd: Forwards, ch, t):
    """Existence evidence at time t: appeared in LPC by t, or routed by t."""
    fl = nd.first_lpc.get(ch)
    if fl is not None and fl <= t + 3600:
        # in LPC at/before t (LPC starts 06-09; treat presence in first LPC
        # snapshot as existence for earlier dates only if it also routed)
        if fl <= t:
            return True
    evs = fwd.by_out.get(ch, [])
    if evs and evs[0][0] <= t:
        return True
    for e in get_fwd(nd.node).settled:
        if e[2] == ch and e[0] <= t:
            return True
    return False


def fee_changed_lpc(nd: Node, scid, t0, t1):
    """Did the advertised fee change (hourly LPC) in [t0, t1]?
    Returns (changed, n_points)."""
    pts = nd.fee_series(scid, t0, t1)
    if len(pts) < 2:
        return None, len(pts)
    fees = {f for _, f in pts}
    return (len(fees) > 1), len(pts)


def sec_fch1():
    print("=" * 78)
    print("SECTION 2 — FC-H1: DTS repricing of stagnant channels vs holding")
    print("=" * 78)
    print("Registered: population = channels with >=3 consecutive zero-revenue")
    print("days whose fee the controller then lowered (dts_pid_sample, delta<0);")
    print("control = matched channel (same node, capacity decile, prior 7-day")
    print("revenue) whose fee did not change in the same window; outcome = fees/")
    print("day and forwards-resumption over the 7 days after; paired bootstrap.")

    rows = []          # (node, ch, t, treated_fpd, ctrl, ctrl_fpd, t_resume, c_resume)
    lost_treat = {n: 0 for n in NODES}
    skipped = defaultdict(int)
    for node in NODES:
        nd = get_node(node)
        print(f"  {node}: fee-change record recovery: {nd.recovered_ids} "
              f"recovered, {nd.lost_ids} lost to the 10-row window "
              f"({100*nd.lost_ids/max(1,nd.recovered_ids+nd.lost_ids):.1f}% "
              f"loss) across {len(nd.gaps)} id gaps")
    for node in NODES:
        nd, fwd = get_node(node), get_fwd(node)
        # capacity deciles from last LPC with full channel set
        caps = {}
        for t, chans in nd.lpc:
            for scid, c in chans.items():
                if c["total"]:
                    caps[scid] = c["total"]
        cap_vals = sorted(caps.values())

        def decile(scid):
            c = caps.get(scid)
            if c is None:
                return None
            return min(9, int(10 * bisect_right(cap_vals, c) / len(cap_vals)))

        # treatment events from recovered records
        events = []
        for cid in sorted(nd.changes):
            c = nd.changes[cid]
            if c.get("manual") or c.get("reason_code") != "dts_pid_sample":
                continue
            if c["new_fee_ppm"] >= c["old_fee_ppm"]:
                continue
            ch, t = c["channel_id"], c["timestamp"]
            if ch == EXTERNAL_CHANNEL:
                skipped[f"{node}: external-writer channel excluded"] += 1
                continue
            if not zero_rev_streak_days(fwd, ch, t, 3):
                continue
            if not channel_existed(nd, fwd, ch, t - 3 * DAY):
                skipped[f"{node}: existence during streak unverifiable"] += 1
                continue
            events.append((t, ch, c))
        # de-overlap: one event per channel per 7 days (keep first)
        events.sort()
        kept, last_t = [], {}
        for t, ch, c in events:
            if ch in last_t and t - last_t[ch] < 7 * DAY:
                skipped[f"{node}: overlapping event (same channel <7d)"] += 1
                continue
            last_t[ch] = t
            kept.append((t, ch, c))
        # outcome coverage end: settled chain end, or (zero-forward node)
        # the last snapshot instant — outcomes there are hard zeros
        n_lpc_end = nd.lpc_times[-1] if nd.lpc_times else 0
        chain_end = max(fwd.settled[-1][0] if fwd.settled else 0, n_lpc_end)

        for t, ch, c in kept:
            if t + 7 * DAY > chain_end:
                skipped[f"{node}: outcome window exceeds chain end"] += 1
                continue
            treated_fpd = fwd.fees_in(ch, t, t + 7 * DAY) / 7.0
            t_res = fwd.first_settled_after(ch, t, t + 7 * DAY)
            prior_rev = fwd.fees_in(ch, t - 7 * DAY, t)
            dec = decile(ch)
            # control pool
            lpc_now = nd.lpc_at(t)
            pool = []
            for scid in (lpc_now[1] if lpc_now else {}):
                if scid == ch or scid == EXTERNAL_CHANNEL:
                    continue
                changed, npts = fee_changed_lpc(nd, scid, t, min(t + 7 * DAY, n_lpc_end))
                if changed is None or changed:
                    continue
                d2 = decile(scid)
                if d2 is None or dec is None or abs(d2 - dec) > 1:
                    continue
                p2 = fwd.fees_in(scid, t - 7 * DAY, t)
                pool.append((abs(d2 - dec), abs(p2 - prior_rev), scid))
            if not pool:
                skipped[f"{node}: no eligible fee-constant matched control"] += 1
                continue
            pool.sort()
            _, _, ctrl = pool[0]
            ctrl_fpd = fwd.fees_in(ctrl, t, t + 7 * DAY) / 7.0
            c_res = fwd.first_settled_after(ctrl, t, t + 7 * DAY)
            unverified = t + 7 * DAY > n_lpc_end
            rows.append((node, ch, t, treated_fpd, ctrl, ctrl_fpd,
                         t_res, c_res, unverified))

        # record-loss bound: LPC-visible fee DROPS on >=3-zero-day channels
        # with no recovered record in the interval (probable lost treatments).
        # Deduplicated like the treatment set (one per channel per 7 days,
        # counting kept recovered events as occupying their 7d window).
        lost_last = dict(last_t)
        for (t0, f0), (t1, f1) in zip(nd.lpc, nd.lpc[1:]):
            for scid in sorted(set(f0) & set(f1)):
                a, b = f0[scid]["fee"], f1[scid]["fee"]
                if a is None or b is None or b >= a or scid == EXTERNAL_CHANNEL:
                    continue
                recs = [c for c in nd.by_chan.get(scid, [])
                        if t0 - 120 < c["timestamp"] <= t1 + 120]
                if recs:
                    continue
                if not zero_rev_streak_days(fwd, scid, t1, 3):
                    continue
                if not channel_existed(nd, fwd, scid, t1 - 3 * DAY):
                    continue
                if scid in lost_last and abs(t1 - lost_last[scid]) < 7 * DAY:
                    continue
                lost_last[scid] = t1
                lost_treat[node] += 1

    print(f"\n  treatment events kept: {len(rows)} "
          f"(n1={sum(1 for r in rows if r[0]==N1)}, "
          f"n2={sum(1 for r in rows if r[0]==N2)})")
    for k, v in sorted(skipped.items()):
        print(f"  skipped: {k}: {v}")
    print(f"  record-loss bound: LPC-visible unrecorded fee-drops on stagnant "
          f"channels, deduplicated one-per-channel-per-7d (probable lost "
          f"treatment events): n1={lost_treat[N1]}, n2={lost_treat[N2]}")

    if not rows:
        print("  UNTESTABLE: no treatment events with matched controls.")
        return

    for label, sel in (("all nodes (registered)", rows),
                       ("n1 only (n2 routed nothing)",
                        [r for r in rows if r[0] == N1]),
                       ("LPC-verified control windows only",
                        [r for r in rows if not r[8]])):
        if not sel:
            print(f"  [{label}] no rows")
            continue
        diffs = [r[3] - r[5] for r in sel]
        mean, lo, hi = boot_ci(diffs)
        med, mlo, mhi = boot_ci(diffs, statfn=np.median)
        t_resume = sum(1 for r in sel if r[6] is not None)
        c_resume = sum(1 for r in sel if r[7] is not None)
        n = len(sel)
        print(f"\n  [{label}] n pairs={n}")
        print(f"    treated fees/day mean={np.mean([r[3] for r in sel]):.3f} "
              f"ctrl={np.mean([r[5] for r in sel]):.3f}")
        print(f"    paired diff (treated-ctrl) fees/day: mean {mean:+.3f} "
              f"[95% CI {lo:+.3f}, {hi:+.3f}]; median {med:+.3f} "
              f"[{mlo:+.3f}, {mhi:+.3f}]")
        print(f"    resumption within 7d: treated {t_resume}/{n} "
              f"({100*t_resume/n:.0f}%), control {c_resume}/{n} "
              f"({100*c_resume/n:.0f}%)")
        resumed_pairs = [(r[6] - r[2], r[7] - r[2]) for r in sel
                         if r[6] is not None and r[7] is not None]
        if resumed_pairs:
            dt = [(a - b) / 3600 for a, b in resumed_pairs]
            print(f"    both-resumed pairs: {len(resumed_pairs)}; "
                  f"median resumption-time diff (treated-ctrl) "
                  f"{np.median(dt):+.1f} h")
        # exact paired sign test on non-zero fee diffs
        nz = [d for d in diffs if d != 0]
        if nz:
            k = sum(1 for d in nz if d > 0)
            sp = stats.binomtest(k, len(nz), 0.5, alternative="two-sided").pvalue
            print(f"    exact sign test on non-zero pairs: {k}/{len(nz)} "
                  f"treated>ctrl, p={sp:.3f}")
        # McNemar exact on resumption discordant pairs
        t_only = sum(1 for r in sel if r[6] is not None and r[7] is None)
        c_only = sum(1 for r in sel if r[6] is None and r[7] is not None)
        if t_only + c_only:
            mp = stats.binomtest(t_only, t_only + c_only, 0.5,
                                 alternative="two-sided").pvalue
            print(f"    McNemar exact on resumption (discordant {t_only} vs "
                  f"{c_only}): p={mp:.3f}")
        if label.startswith("all nodes"):
            # registered decision: 95% CI on 7d-after difference excluding zero
            CONFIRMATORY_P["FC-H1 fees/day 7d-after paired diff (sign test)"] = \
                stats.binomtest(sum(1 for d in nz if d > 0), len(nz), 0.5,
                                alternative="two-sided").pvalue \
                if nz else float("nan")
    print("\n  per-pair detail:")
    for node, ch, t, tf, ctrl, cf, tr, cr, unv in rows:
        print(f"    {node} {ch} @{iso(t)} treated {tf:7.3f} sat/d | "
              f"ctrl {ctrl} {cf:7.3f} sat/d | resume T={'%.1fh' % ((tr-t)/3600) if tr else '-':>7s} "
              f"C={'%.1fh' % ((cr-t)/3600) if cr else '-':>7s}"
              f"{' [ctrl window partly unverified]' if unv else ''}")


# ==========================================================================
# Section 3 — FC-H2
# ==========================================================================

def week_blocks(anchor, lo, hi):
    """7d blocks aligned to anchor covering [lo, hi]: list of (w0, w1, idx)."""
    out = []
    k = math.floor((lo - anchor) / (7 * DAY))
    while anchor + k * 7 * DAY < hi:
        w0 = anchor + k * 7 * DAY
        out.append((w0, w0 + 7 * DAY, k))
        k += 1
    return out


def earning_weighted_p90(fees_ppm_weights):
    """p90 of implied-ppm distribution weighted by fee earned."""
    if not fees_ppm_weights:
        return None
    arr = sorted(fees_ppm_weights)
    tot = sum(w for _, w in arr)
    if tot <= 0:
        return None
    cum = 0.0
    for ppm, w in arr:
        cum += w
        if cum >= 0.9 * tot:
            return ppm
    return arr[-1][0]


def sec_fch2():
    print("=" * 78)
    print("SECTION 3 — FC-H2: climb governor (9f8f219, deployed "
          f"{iso(GOV_DEPLOY)})")
    print("=" * 78)
    print("Registered: per channel-week (a) overshoot = max advertised fee /")
    print("earning-weighted p90 fee, (b) fees earned; before vs after 06-12;")
    print("MWU p<0.05 on (a); bootstrap CI on (b) lower bound excluding -20%.")
    print("NOTE the coverage asymmetry quantified below: advertised fees exist")
    print("only 06-09 onward, so the before-side of (a) is one partial week.")

    over_before, over_after, fees_before, fees_after = [], [], [], []
    fees_before_ch, fees_after_ch = defaultdict(list), defaultdict(list)
    over_rows = []
    surv_before, surv_after = [], []
    epoch_rows = defaultdict(list)   # descriptive 071a5b3 split
    for node in NODES:
        nd, fwd = get_node(node), get_fwd(node)
        if not fwd.settled:
            print(f"\n  {node}: no settled forwards; (a) undefined (no earning-"
                  "weighted p90 exists), (b) all-zero channel-weeks.")
            continue
        chain_lo = fwd.settled[0][0]
        chain_hi = fwd.settled[-1][0]
        # channels ever seen
        all_ch = set(nd.first_lpc) | set(fwd.by_out)
        all_ch.discard(None)
        # survivors: in LPC both before GOV_DEPLOY and after 06-14 (post mass close)
        pre_set = set()
        post_set = set()
        for t, chans in nd.lpc:
            if t < GOV_DEPLOY:
                pre_set |= set(chans)
            if t > GOV_DEPLOY + 2 * DAY:
                post_set |= set(chans)
        survivors = pre_set & post_set
        print(f"\n  {node}: channels pre={len(pre_set)} post={len(post_set)} "
              f"survivors={len(survivors)} (mass close 06-13 confound)")

        for ch in sorted(all_ch):
            evs = fwd.by_out.get(ch, [])
            for w0, w1, k in week_blocks(GOV_DEPLOY, chain_lo, chain_hi):
                # (b): fees earned per FULL chain-covered channel-week
                full_b = w0 >= chain_lo and w1 <= chain_hi
                # channel must exist during the week (LPC presence or routing)
                exists = (nd.first_lpc.get(ch, 1e18) < w1) or \
                         (evs and evs[0][0] < w1)
                if not exists:
                    continue
                wf = sum(e[3] for e in evs if w0 < e[0] <= w1) / 1000.0
                if full_b:
                    if w1 <= GOV_DEPLOY:
                        fees_before.append(wf)
                        fees_before_ch[ch].append(wf)
                        if ch in survivors:
                            surv_before.append(wf)
                    elif w0 >= GOV_DEPLOY:
                        fees_after.append(wf)
                        fees_after_ch[ch].append(wf)
                        if ch in survivors:
                            surv_after.append(wf)
                # (a): overshoot needs advertised coverage + earnings
                pts = nd.fee_series(ch, w0, w1)
                if len(pts) < 24:
                    continue
                ew = [(1e6 * e[3] / e[4], e[3] / 1000.0)
                      for e in evs if w0 < e[0] <= w1 and e[4]]
                p90 = earning_weighted_p90(ew)
                if not p90:
                    continue
                mx = max(f for _, f in pts)
                ratio = mx / p90
                row = (node, ch, k, iso(w0)[:10], len(pts), mx, p90, ratio, wf)
                over_rows.append(row)
                if w1 <= GOV_DEPLOY:
                    over_before.append(ratio)
                else:
                    over_after.append(ratio)
                # descriptive epoch split for 071a5b3
                if w1 <= GOV_DEPLOY:
                    epoch_rows["pre-governor"].append(ratio)
                elif w0 >= RATCHET_DEPLOY:
                    epoch_rows["post-ratchet"].append(ratio)
                else:
                    epoch_rows["governor-to-ratchet(mixed)"].append(ratio)

    print(f"\n  (a) overshoot channel-weeks: before n={len(over_before)}, "
          f"after n={len(over_after)}")
    for r in over_rows:
        print(f"    {r[0]} {r[1]:>16s} wk{r[2]:+d} ({r[3]}) lpc_pts={r[4]:3d} "
              f"max_adv={r[5]:5d} ew_p90={r[6]:7.1f} overshoot={r[7]:6.2f} "
              f"fees={r[8]:.1f}")
    if over_before and over_after:
        u, p = mwu(over_after, over_before, "less")
        print(f"    MWU one-sided (after < before): U={u:.0f} p={p:.4f}")
        CONFIRMATORY_P["FC-H2a overshoot decreases (MWU)"] = p
        mb, mlo, mhi = boot_ci(over_before, np.median)
        ma, alo, ahi = boot_ci(over_after, np.median)
        print(f"    median overshoot before {mb:.2f} [{mlo:.2f},{mhi:.2f}] "
              f"vs after {ma:.2f} [{alo:.2f},{ahi:.2f}]")
    else:
        print("    (a) UNDERPOWERED/UNTESTABLE as registered on this corpus: "
              "advertised-fee coverage begins 06-09, leaving at most one "
              "partial pre-deploy week.")
        CONFIRMATORY_P["FC-H2a overshoot decreases (MWU)"] = float("nan")

    print(f"\n  (b) fees per full channel-week: before n={len(fees_before)} "
          f"(3 weeks), after n={len(fees_after)} (2 weeks)")
    if fees_before and fees_after:
        eff, lo, hi = boot_ratio_ci(fees_after, fees_before)
        print(f"    mean fees/channel-week: before "
              f"{np.mean(fees_before):.2f}, after {np.mean(fees_after):.2f}; "
              f"change {100*eff:+.1f}% [95% CI {100*lo:+.1f}%, {100*hi:+.1f}%]"
              f" -> lower bound {'EXCLUDES' if lo > -0.20 else 'DOES NOT exclude'}"
              " a >20% decline")
        if surv_before and surv_after:
            eff2, lo2, hi2 = boot_ratio_ci(surv_after, surv_before)
            print(f"    survivors-only sensitivity: change {100*eff2:+.1f}% "
                  f"[{100*lo2:+.1f}%, {100*hi2:+.1f}%] "
                  f"(n={len(surv_before)}/{len(surv_after)})")
    print("\n  descriptive 071a5b3 (zero-flow ratchet, in-window 06-15 deploy) "
          "overshoot split:")
    for k in ("pre-governor", "governor-to-ratchet(mixed)", "post-ratchet"):
        v = epoch_rows.get(k, [])
        if v:
            print(f"    {k}: n={len(v)} median={np.median(v):.2f} "
                  f"mean={np.mean(v):.2f}")
        else:
            print(f"    {k}: n=0")


# ==========================================================================
# Section 4 — FC-H3
# ==========================================================================

def sec_fch3():
    print("=" * 78)
    print("SECTION 4 — FC-H3: rebalance-floored channels net-positive")
    print("=" * 78)
    print("Registered activation: >=4 realized rebalance cost samples in 30d")
    print("(spend ledger) on a non-sink/dormant channel with cost_ppm*1.20 over")
    print("the base floor. Census of the corpus spend ledger:")
    for node in NODES:
        nd = get_node(node)
        nonzero, total_rows = 0, 0
        cats = defaultdict(float)
        for ts, sl in nd.spend_ledgers:
            # sum any numeric spend fields found
            def walk(o, prefix=""):
                nonlocal nonzero, total_rows
                if isinstance(o, dict):
                    for k, v in o.items():
                        if isinstance(v, (int, float)) and \
                                ("spent" in k or "cost" in k or "sats" in k):
                            total_rows += 1
                            if v:
                                nonzero += 1
                                cats[f"{prefix}{k}"] += v
                        elif isinstance(v, (dict, list)):
                            walk(v, f"{prefix}{k}.")
                elif isinstance(o, list):
                    for x in o:
                        walk(x, prefix)
            walk(sl)
        print(f"  {node}: spend-ledger snapshots={len(nd.spend_ledgers)}, "
              f"numeric spend/cost fields scanned={total_rows}, non-zero={nonzero}")
        for k, v in sorted(cats.items()):
            print(f"      non-zero field {k}: sum={v}")
        # rebal cost floor reason tags (soft nudge, NOT activation - context)
        tagged = [c for c in nd.changes.values()
                  if "rebal_cost_floor" in (c.get("reason") or "")]
        print(f"    fee-change records mentioning rebal_cost_floor: "
              f"{len(tagged)} (soft-nudge tag, not hard-floor activation)")
        # dashboards rebalance cost
        if nd.dashboards:
            td, dash = nd.dashboards[-1]
            print(f"    dashboard 30d rebalance_cost_sats @{iso(td)}: "
                  f"{(dash.get('period') or {}).get('rebalance_cost_sats')}")
        if nd.rh:
            t0, rh0 = nd.rh[0]
            t1, rh1 = nd.rh[-1]
            print(f"    lifetime_rebalance_costs_sats: {rh0.get('lifetime_rebalance_costs_sats')} "
                  f"@{iso(t0)} -> {rh1.get('lifetime_rebalance_costs_sats')} @{iso(t1)} "
                  f"(delta = in-corpus realized rebalance spend)")
    print("\n  VERDICT inputs: activation population = channels with >=4 realized")
    print("  30d cost samples. With zero in-corpus rebalance spend, that")
    print("  population is empty -> FC-H3 is UNTESTABLE AS REGISTERED.")


# ==========================================================================
# Section 5 — E2 natural experiment (EXPLORATORY)
# ==========================================================================

def sec_e2():
    print("=" * 78)
    print("SECTION 5 — EXPLORATORY (non-confirmatory): external-hand elasticity")
    print(f"episode on {EXTERNAL_CHANNEL} (fee-loop.md E2/E2b)")
    print("=" * 78)
    node = N1
    nd, fwd = get_node(node), get_fwd(node)
    ch = EXTERNAL_CHANNEL
    evs = fwd.by_out.get(ch, [])
    tot = sum(e[3] for e in evs) / 1000.0
    print(f"  channel settled fees over full chain: {tot:.1f} sats "
          f"({len(evs)} forwards) = {100*tot/(get_fwd(node).total_fee_msat/1000):.1f}% "
          "of node revenue")

    # revenue by implied fee band for this channel
    bands = [(0, 150), (150, 300), (300, 10**9)]
    for lo, hi in bands:
        f = sum(e[3] for e in evs if e[4] and lo <= 1e6 * e[3] / e[4] < hi) / 1000.0
        c = sum(1 for e in evs if e[4] and lo <= 1e6 * e[3] / e[4] < hi)
        hs = f"{hi}" if hi < 10**9 else "inf"
        print(f"    settled at implied [{lo},{hs}) ppm: {f:9.1f} sats ({c} fwds)")

    # Price-exposure vs revenue over the LPC-covered span 06-09..06-20.
    # Exposure comes from the LPC backbone (hourly ground truth; the change
    # instant inside a diff interval is taken at the interval midpoint, so
    # per-boundary misallocation is bounded by half the LPC gap). Revenue is
    # priced per forward by its exact implied ppm (fee_msat/out_msat) - this
    # needs no timeline and is msat-exact. NOTE the 06-15T18:33 anomaly found
    # here: 10 MPP shards settled at exactly 150 ppm while LPC showed 2306 at
    # 18:16 and 250 at 19:35 - an interim external write (or grace-period
    # pricing) invisible at LPC granularity; revenue lands in [150,300) by
    # implied price, where it belongs economically.
    t_lo, t_hi = nd.lpc_times[0], ts_of("20260620T235900Z")
    pts = nd.fee_series(ch, t_lo, t_hi)
    changes = []
    for (ta, fa), (tb, fb) in zip(pts, pts[1:]):
        if fa != fb:
            changes.append(((ta + tb) / 2, fb, tb - ta))
    tl = [(pts[0][0], pts[0][1])] + [(t, f) for t, f, _ in changes]
    max_gap_h = max((g for _, _, g in changes), default=0) / 3600
    print(f"\n  LPC-backbone fee timeline 06-09..06-20: {len(tl)} price "
          f"segments; max change-interval {max_gap_h:.1f} h (midpoint "
          "attribution)")
    bands_f = [(0, 150), (150, 300), (300, 1000), (1000, 10**9)]

    def band_of(f):
        return next((f"[{lo},{hi if hi < 10**9 else 'inf'})"
                     for lo, hi in bands_f if lo <= f < hi), "?")

    hours_at = defaultdict(float)
    rev_at = defaultdict(float)
    cnt_at = defaultdict(int)
    for (ta, fa), (tb, _) in zip(tl, tl[1:] + [(t_hi, None)]):
        if fa is not None:
            hours_at[band_of(fa)] += (min(tb, t_hi) - ta) / 3600.0
    for e in fwd.by_out.get(ch, []):
        if t_lo < e[0] <= t_hi and e[4]:
            b = band_of(1e6 * e[3] / e[4])
            rev_at[b] += e[3] / 1000.0
            cnt_at[b] += 1
    print("  exposure (advertised, LPC backbone) vs revenue (implied ppm "
          "per forward), 06-09..06-20:")
    for lo, hi in bands_f:
        b = f"[{lo},{hi if hi < 10**9 else 'inf'})"
        h, r, c = hours_at.get(b, 0.0), rev_at.get(b, 0.0), cnt_at.get(b, 0)
        print(f"    {b:>12s} ppm: {h:7.1f} h exposure, {r:9.1f} sats, "
              f"{c:3d} fwds -> {(r / h if h else 0):8.2f} sats/h")

    # held-price contrast: LPC-confirmed 2306 span vs the held 250 span
    held0, held1 = ts_of("20260615T025321Z"), ts_of("20260615T181634Z")
    held_rev = fwd.fees_in(ch, held0, held1)
    print(f"\n  LPC-confirmed held-at-2306 span {iso(held0)}..{iso(held1)} "
          f"({(held1-held0)/3600:.1f} h): {held_rev:.1f} sats earned")
    reset = ts_of("20260615T193539Z")
    span = held1 - held0
    post_rev = fwd.fees_in(ch, reset, reset + span)
    print(f"  first same-length span held at 250 ({iso(reset)}..): "
          f"{post_rev:.1f} sats")
    wk250 = fwd.fees_in(ch, reset, ts_of("20260623T000000Z"))
    print(f"  full held-at-250 week 06-15T19:35..06-23T00:00: {wk250:.1f} sats "
          f"({wk250/((ts_of('20260623T000000Z')-reset)/3600):.2f} sats/h)")
    rest_b = fwd.node_fees(held0, held1, exclude=ch) / max(1e-9, span / 3600)
    rest_a = fwd.node_fees(reset, reset + span, exclude=ch) / \
        max(1e-9, span / 3600)
    print(f"  rest-of-node demand control (sats/h): {rest_b:.2f} during the "
          f"2306 span vs {rest_a:.2f} during the matched 250 span")

    # hole-period (06-20..07-01) revenue at the unobservable fee
    hole_rev = fwd.fees_in(ch, ts_of("20260621T000000Z"),
                           ts_of("20260701T000000Z"))
    print(f"\n  hole period 06-21..06-30 (fee unobservable; 07-01 records "
          f"imply 65-66 ppm by then): {hole_rev:.1f} sats earned")

    # daily table across the episode
    print("\n  daily settled fees on the channel (05-20..07-01):")
    for d in sorted(fwd.daily_fee.get(ch, {})):
        print(f"    {d}: {fwd.daily_fee[ch][d]:9.1f} sats")


# ==========================================================================
# Section 6 — HH hypotheses
# ==========================================================================

def sec_hh():
    print("=" * 78)
    print("SECTION 6 — HH-H1..H3 (hive hints)")
    print("=" * 78)
    for node in NODES:
        nd = get_node(node)
        n = len(nd.hints_status)
        fresh = sum(1 for _, f, _, _ in nd.hints_status if f)
        usable = sum(1 for _, _, u, _ in nd.hints_status if u)
        stale = sum(1 for _, _, _, s in nd.hints_status if s)
        print(f"  {node}: hints-status snapshots={n}: fresh={fresh} "
              f"({100*fresh/max(1,n):.2f}%), usable={usable}, "
              f"stale_fallback={stale}")
        notfresh = [(t, f, u, s) for t, f, u, s in nd.hints_status if not f]
        for t, f, u, s in notfresh[:10]:
            print(f"      not-fresh @{iso(t)} usable={u} stale_fallback={s}")
    print("  HH-H1 requires fresh vs not-fresh node-hour contrast.")

    # HH-H2: channels opened in-corpus
    print("\n  HH-H2: channels first appearing in LPC after corpus start:")
    for node in NODES:
        nd = get_node(node)
        t0 = nd.lpc_times[0]
        opened = [(t, s) for s, t in nd.first_lpc.items() if t > t0 + 3600]
        for t, s in sorted(opened):
            print(f"    {node} {s} first seen {iso(t)}")
        if not opened:
            print(f"    {node}: none")

    # HH-H3: corridor-owner peers vs channels
    print("\n  HH-H3: corridor_role=owner peers and their local channels:")
    for node in NODES:
        nd, fwd = get_node(node), get_fwd(node)
        # role held >= 24h: count snapshot presence
        owner_seen = defaultdict(int)
        for t, owners in nd.owner_roles:
            for p in owners:
                owner_seen[p] += 1
        # peer -> scids
        peer_scids = defaultdict(set)
        for t, chans in nd.lpc:
            for scid, c in chans.items():
                peer_scids[c["peer"]].add(scid)
        stable_owners = {p for p, cnt in owner_seen.items() if cnt >= 288}
        print(f"  {node}: owner-role peers seen={len(owner_seen)}, "
              f"held>=24h(288 snaps)={len(stable_owners)}"
              + (f"; max snapshots any peer held role="
                 f"{max(owner_seen.values())} "
                 f"(~{max(owner_seen.values())*5/60:.1f} h)"
                 if owner_seen else ""))
        owner_chans, other_chans = set(), set()
        for p, scids in peer_scids.items():
            (owner_chans if p in stable_owners else other_chans).update(scids)
        print(f"    local channels to stable owners: {sorted(owner_chans)}")
        if not owner_chans:
            print("    -> no local channels to owner peers; HH-H3 VACUOUS here")
            continue
        # daily forwarded sats (in+out) per capacity per channel
        caps = {}
        for t, chans in nd.lpc:
            for scid, c in chans.items():
                if c["total"]:
                    caps[scid] = c["total"] / 1000.0
        # restrict to LPC-covered days
        lpc_days = {day_of(t) for t in nd.lpc_times}

        def daily_vpc(chset):
            out = []
            for ch in chset:
                cap = caps.get(ch)
                if not cap:
                    continue
                for d in lpc_days:
                    v = fwd.daily_out.get(ch, {}).get(d, 0.0) + \
                        fwd.daily_in.get(ch, {}).get(d, 0.0)
                    out.append(v / cap)
            return out
        vo, vn = daily_vpc(owner_chans), daily_vpc(other_chans)
        if vo and vn:
            u, p = mwu(vo, vn, "greater")
            print(f"    daily volume/capacity: owner n={len(vo)} "
                  f"mean={np.mean(vo):.4f} median={np.median(vo):.4f}; "
                  f"non-owner n={len(vn)} mean={np.mean(vn):.4f} "
                  f"median={np.median(vn):.4f}")
            print(f"    MWU one-sided (owner > none): p={p:.4f}")
            frac_o = np.mean([v > 0 for v in vo])
            frac_n = np.mean([v > 0 for v in vn])
            print(f"    active-day fraction: owner {100*frac_o:.1f}% vs "
                  f"non-owner {100*frac_n:.1f}%")
            if node == N1:
                CONFIRMATORY_P["HH-H3 owner-corridor volume (MWU, n1)"] = p
        else:
            print("    insufficient data for MWU")


# ==========================================================================
# Section 7 — PM hypotheses
# ==========================================================================

def sec_pm():
    print("=" * 78)
    print("SECTION 7 — PM-H1..H3 (policy manager) population census")
    print("=" * 78)
    for node in NODES:
        nd = get_node(node)
        static = [c for c in nd.changes.values()
                  if c.get("reason_code") == "policy_static"]
        manual = [c for c in nd.changes.values() if c.get("manual")]
        codes = defaultdict(int)
        for c in nd.changes.values():
            codes[c.get("reason_code")] += 1
        print(f"  {node}: recovered fee-change records={len(nd.changes)} "
              f"reason_codes={dict(codes)}")
        print(f"    policy_static records: {len(static)}; manual records: "
              f"{len(manual)}")
    print("  No revenue-policy artifact exists in the corpus; no rebalance_mode")
    print("  transition, no PASSIVE/STATIC assignment, and no policy expiry is")
    print("  corpus-observable (Phase 2/3 concur). Descriptive PM-H2-adjacent")
    print("  observation (NOT the registered test): the one channel inferred to")
    print("  have left management (E2b) shows, during 06-16..06-20:")
    nd = get_node(N1)
    ch = EXTERNAL_CHANNEL
    t0, t1 = ts_of("20260616T000000Z"), ts_of("20260620T235900Z")
    recs = [c for c in nd.by_chan.get(ch, []) if t0 <= c["timestamp"] <= t1]
    changed, npts = fee_changed_lpc(nd, ch, t0, t1)
    print(f"    recorded automated changes: {len(recs)}; advertised-fee changes "
          f"(hourly LPC, {npts} points): {changed}")


# ==========================================================================
# Section 8 — FA hypotheses
# ==========================================================================

def sec_fa():
    print("=" * 78)
    print("SECTION 8 — FA-H1..H3 (flow analysis)")
    print("=" * 78)

    # ---- FA-H1: labels predictive of 24h outbound-ratio change -----------
    print("\n-- FA-H1: SOURCE vs BALANCED 24h outbound-ratio change --")
    deltas = {"source": [], "balanced": []}
    per_chan = {"source": defaultdict(list), "balanced": defaultdict(list)}
    for node in NODES:
        nd = get_node(node)
        st_times = sorted(nd.states)
        for t, chans in nd.lpc:
            # label from nearest revenue-status <= t (within 10 min)
            i = bisect_right(st_times, t) - 1
            if i < 0 or t - st_times[i] > 600:
                continue
            labels = nd.states[st_times[i]]
            fut = nd.lpc_near(t + DAY, 2 * 3600)
            if fut is None:
                continue
            for scid, c in chans.items():
                lab = labels.get(scid)
                if lab not in ("source", "balanced"):
                    continue
                if not c["total"]:
                    continue
                f = fut[1].get(scid)
                if not f or not f["total"]:
                    continue
                r0 = c["spend"] / c["total"]
                r1 = f["spend"] / f["total"]
                deltas[lab].append(r1 - r0)
                per_chan[lab][(node, scid)].append(r1 - r0)
    ns = {k: len(v) for k, v in deltas.items()}
    print(f"  channel-hours: source={ns['source']} balanced={ns['balanced']}")
    if ns["source"] and ns["balanced"]:
        u, p = mwu(deltas["source"], deltas["balanced"], "less")
        CONFIRMATORY_P["FA-H1 SOURCE declines more (MWU)"] = p
        ms, mb = np.median(deltas["source"]), np.median(deltas["balanced"])
        # bootstrap CI on median difference
        rng = np.random.default_rng(SEED)
        s = np.asarray(deltas["source"])
        b = np.asarray(deltas["balanced"])
        boots = [np.median(s[rng.integers(0, len(s), len(s))]) -
                 np.median(b[rng.integers(0, len(b), len(b))])
                 for _ in range(2000)]
        print(f"  median 24h delta: source {ms:+.4f} vs balanced {mb:+.4f}; "
              f"median diff {ms-mb:+.4f} "
              f"[95% CI {np.percentile(boots,2.5):+.4f}, "
              f"{np.percentile(boots,97.5):+.4f}]")
        print(f"  MWU one-sided (source < balanced): p={p:.2e}")
        # cluster sensitivity: per-channel means
        cs = [np.mean(v) for v in per_chan["source"].values()]
        cb = [np.mean(v) for v in per_chan["balanced"].values()]
        u2, p2 = mwu(cs, cb, "less")
        print(f"  cluster-robust sensitivity (per-channel means, "
              f"n={len(cs)}/{len(cb)}): p={p2:.4f} "
              f"(channel-hours are heavily autocorrelated; registered unit is "
              "channel-hours)")

    # ---- FA-H2: hysteresis flips before/after F1 --------------------------
    print(f"\n-- FA-H2: label churn before/after F1 deploy ({iso(F1_DEPLOY)}) --")
    for node in NODES:
        nd = get_node(node)
        st_times = sorted(nd.states)
        flips = {"before": 0, "after": 0}
        chan_days = {"before": set(), "after": set()}
        for t_prev, t_cur in zip(st_times, st_times[1:]):
            if t_cur - t_prev > 900:
                continue
            era = "before" if t_cur < F1_DEPLOY else "after"
            prev, cur = nd.states[t_prev], nd.states[t_cur]
            for ch in set(prev) & set(cur):
                chan_days[era].add((ch, day_of(t_cur)))
                if prev[ch] != cur[ch]:
                    flips[era] += 1
        nb, na = len(chan_days["before"]), len(chan_days["after"])
        print(f"  {node}: before: {flips['before']} flips / {nb} channel-days "
              f"= {flips['before']/max(1,nb):.3f}/cd; after: {flips['after']} "
              f"flips / {na} channel-days = {flips['after']/max(1,na):.3f}/cd")
        if nb and na and flips["before"] + flips["after"] > 0:
            # Poisson rate-ratio test (exact binomial conditioning)
            k1, k2 = flips["before"], flips["after"]
            res = stats.binomtest(k2, k1 + k2, na / (nb + na),
                                  alternative="less")
            print(f"    rate-ratio test (after < before): p={res.pvalue:.4f}")
            CONFIRMATORY_P[f"FA-H2a flips fall ({node})"] = res.pvalue
        elif nb and na:
            print("    zero flips in both eras -> rate test vacuous")
        # earnings guardrail (b): fees/day before vs after F1
        fwd = get_fwd(node)
        if fwd.settled:
            fb = [sum(fwd.daily_fee[ch].get(d, 0) for ch in fwd.daily_fee)
                  for d in sorted({day_of(e[0]) for e in fwd.settled})
                  if d < day_of(F1_DEPLOY)]
            fa_ = [sum(fwd.daily_fee[ch].get(d, 0) for ch in fwd.daily_fee)
                   for d in sorted({day_of(e[0]) for e in fwd.settled})
                   if d > day_of(F1_DEPLOY)]
            if fb and fa_:
                eff, lo, hi = boot_ratio_ci(fa_, fb)
                print(f"    (b) node fees/day: before mean {np.mean(fb):.1f} "
                      f"(n={len(fb)}d), after {np.mean(fa_):.1f} (n={len(fa_)}d); "
                      f"change {100*eff:+.1f}% [{100*lo:+.1f}%, {100*hi:+.1f}%] "
                      "(note: baseline days are pre-corpus-hole June + full May "
                      "-> heavy network-shift confounding; labels observable "
                      "only from 06-09 so the registered 'flips' baseline is "
                      "2.99 days)")

    # ---- FA-H3: Kalman depletion replay ------------------------------------
    print("\n-- FA-H3: depletion-forecast calibration (Kalman replay) --")
    from modules import flow_analysis as fam
    for node in NODES:
        nd, fwd = get_node(node), get_fwd(node)
        if not fwd.settled or not nd.lpc:
            print(f"  {node}: no forwards; every prediction would be 'no "
                  "depletion' -> VACUOUS")
            continue
        # per-channel net-flow events (settled only, matching plugin forwards
        # table): out_channel -> +out_msat (drain), in_channel -> -in_msat
        net_ev = defaultdict(list)
        for ts, out_ch, in_ch, fee, out_msat in fwd.settled:
            net_ev[out_ch].append((ts, out_msat))
            net_ev[in_ch].append((ts, -(out_msat + fee)))
        for ch in net_ev:
            net_ev[ch].sort()
        # capacities
        caps = {}
        for t, chans in nd.lpc:
            for scid, c in chans.items():
                if c["total"]:
                    caps[scid] = c["total"] / 1000.0
        # replay: hourly cadence (production flow-interval=3600) from chain
        # start; burn-in discarded per registered definition (first 5 days +
        # KALMAN_MIN_OBSERVATIONS enforced by the module's own convergence)
        t_start = fwd.settled[0][0]
        burn_end = t_start + 5 * DAY
        chans_all = set(caps) | set(net_ev)
        kfs = {ch: fam.KalmanFlowFilter() for ch in chans_all}
        last_t = {ch: None for ch in chans_all}
        # evaluation bookkeeping
        state_at = {}                 # (ch, lpc_ts) -> (ratio, velocity)
        lpc_set = set(nd.lpc_times)
        t = math.floor(t_start / 3600) * 3600 + 3600
        t_end = nd.lpc_times[-1]
        times = []
        while t <= t_end:
            times.append(t)
            t += 3600
        # merge in exact LPC times (they are ~hourly but not aligned)
        times = sorted(set(times) | lpc_set)
        for t in times:
            for ch in chans_all:
                cap = caps.get(ch)
                if not cap:
                    continue
                kf = kfs[ch]
                dt_h = 1.0 if last_t[ch] is None else \
                    min((t - last_t[ch]) / 3600.0, 168.0)
                last_t[ch] = t
                evs = net_ev.get(ch, [])
                i0 = bisect_left(evs, (t - DAY, -1e18))
                i1 = bisect_right(evs, (t, 1e18))
                recent = evs[i0:i1]
                net_sats = sum(m for _, m in recent) / 1000.0
                raw = max(-1.0, min(1.0, net_sats / cap))
                # volatility: daily buckets over flow_window_days=7
                buckets = []
                for k in range(7):
                    d0, d1 = t - (k + 1) * DAY, t - k * DAY
                    j0 = bisect_left(evs, (d0, -1e18))
                    j1 = bisect_right(evs, (d1, 1e18))
                    out_s = sum(m for _, m in evs[j0:j1] if m > 0) / 1000.0
                    in_s = -sum(m for _, m in evs[j0:j1] if m < 0) / 1000.0
                    buckets.append({"in": in_s, "out": out_s})
                if len(buckets) >= 3:
                    nets = [(b.get("out", 0) or 0) - (b.get("in", 0) or 0)
                            for b in buckets]
                    changes = [abs(nets[i] - nets[i-1])
                               for i in range(1, len(nets))]
                    mean_change = sum(changes) / len(changes)
                    mean_flow = sum(abs(x) for x in nets) / len(nets)
                    if mean_flow < 1000:
                        vol = 0.5
                    else:
                        vol = 0.5 + min(1.5, (mean_change / max(1, mean_flow)) * 3.0)
                else:
                    vol = 1.0
                kf.predict(dt_h, vol)
                if recent:
                    # confidence per _calculate_confidence at replay time
                    cnt = len(recent)
                    if cnt >= fam.MIN_FORWARDS_FOR_HIGH_CONFIDENCE:
                        cf = 1.0
                    else:
                        cf = fam.MIN_CONFIDENCE + (1.0 - fam.MIN_CONFIDENCE) * (
                            cnt / fam.MIN_FORWARDS_FOR_HIGH_CONFIDENCE)
                    days_since = (t - recent[-1][0]) / DAY
                    rec = math.pow(0.5, days_since /
                                   fam.CONFIDENCE_RECENCY_HALFLIFE_DAYS)
                    conf = max(fam.MIN_CONFIDENCE,
                               min(fam.MAX_CONFIDENCE, cf * rec))
                    kf.update(raw, conf)
                if kf._has_nan():
                    kf._reset_state()
                if t in lpc_set and t >= burn_end and \
                        kf.state.observation_count >= fam.KALMAN_MIN_OBSERVATIONS:
                    state_at[(ch, t)] = (kf.state.flow_ratio,
                                         kf.state.flow_velocity)
        # evaluate predictions at LPC times with 36h of LPC lookahead
        n_pred = n_hit = n_base = n_base_hit = n_already = 0
        episodes_pos = []
        last_pos = {}
        for ti, (tt, chans) in enumerate(nd.lpc):
            # need lookahead coverage: LPC points within (tt, tt+36h]
            look = [x for x in nd.lpc if tt < x[0] <= tt + 36 * 3600]
            if not look or look[-1][0] - tt < 30 * 3600:
                continue
            for scid, c in chans.items():
                st = state_at.get((scid, tt))
                if st is None or not c["total"]:
                    continue
                r0 = c["spend"] / c["total"]
                if r0 < 0.1:
                    n_already += 1
                    continue
                hours = fam.estimate_depletion_hours(
                    c["spend"] / 1000.0, c["total"] / 1000.0, st[0], st[1])
                hit = any((x[1].get(scid) or {}).get("total") and
                          (x[1][scid]["spend"] / x[1][scid]["total"]) < 0.1
                          for x in look)
                if hours is not None and hours <= 24:
                    n_pred += 1
                    n_hit += int(hit)
                    if scid not in last_pos or tt - last_pos[scid] > 36 * 3600:
                        episodes_pos.append((scid, tt, hit))
                    last_pos[scid] = tt
                else:
                    n_base += 1
                    n_base_hit += int(hit)
        print(f"  {node}: predictions evaluated (channel-hours, outbound ratio "
              f">=0.1 at prediction time): positive={n_pred} (hit {n_hit}), "
              f"negative={n_base} (hit {n_base_hit}); already-depleted "
              f"excluded={n_already}")
        if n_pred and n_base:
            p_pos = n_hit / n_pred
            p_neg = n_base_hit / n_base
            table = [[n_hit, n_pred - n_hit], [n_base_hit, n_base - n_base_hit]]
            fres = stats.fisher_exact(table, alternative="greater")
            print(f"    depletion within 36h: predicted {100*p_pos:.1f}% vs "
                  f"base {100*p_neg:.2f}% (ratio "
                  f"{(p_pos/max(p_neg,1e-9)):.1f}x, registered bar >=3x); "
                  f"Fisher one-sided p={fres.pvalue:.2e}")
            CONFIRMATORY_P[f"FA-H3 depletion calibration ({node})"] = fres.pvalue
            print(f"    de-duplicated positive episodes: {len(episodes_pos)} "
                  f"(hits {sum(1 for e in episodes_pos if e[2])})")
            for scid, tt, hit in episodes_pos[:12]:
                print(f"      {scid} @{iso(tt)} hit={hit}")
        elif n_pred == 0:
            print("    zero positive predictions -> calibration ratio "
                  "undefined; hypothesis INCONCLUSIVE/vacuous on this node")


# ==========================================================================

SECTIONS = {
    "decomp": sec_decomp, "fch1": sec_fch1, "fch2": sec_fch2,
    "fch3": sec_fch3, "e2": sec_e2, "hh": sec_hh, "pm": sec_pm, "fa": sec_fa,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", choices=list(SECTIONS) + ["all"], default="all")
    args = ap.parse_args()
    todo = list(SECTIONS) if args.section == "all" else [args.section]
    for s in todo:
        SECTIONS[s]()
        print()
    if len(todo) > 1 and CONFIRMATORY_P:
        print("=" * 78)
        print("MULTIPLE-COMPARISON SUMMARY (Holm-Bonferroni over the")
        print("confirmatory tests actually run)")
        print("=" * 78)
        adj = holm(CONFIRMATORY_P)
        for k in sorted(CONFIRMATORY_P):
            p = CONFIRMATORY_P[k]
            a = adj.get(k)
            print(f"  {k}: p={p if not math.isnan(p) else float('nan'):.4g}"
                  f"{'' if a is None else f'  holm-adj={a:.4g}'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
