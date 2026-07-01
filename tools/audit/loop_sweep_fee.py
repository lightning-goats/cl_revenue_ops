#!/usr/bin/env python3
"""Phase 3 end-to-end sweep: the FEE DECISION LOOP as a system.

Verifies, at snapshot granularity over the FULL frozen hermes corpus
(/home/sat/cl-mycelium-hermes/{hive-nexus-01,hive-nexus-02}), that the loop

    flow_analysis state -> fee_controller decision -> setchannel (gossip)
    -> recorded fee change -> forwarding outcome

is internally coherent across module boundaries. Phase 2 verified each module
in isolation; these are the CROSS-module invariants.

Data model
  - Recorded fee changes are recovered by deduplicating the rolling
    `recent_fee_changes` window (10 rows) across all revenue-status.json
    snapshots. The id sequence exposes losses: a cycle that changes more
    than 10 channels overflows the window between two 5-minute snapshots,
    so recovery is structurally INCOMPLETE on busy nodes. Every
    completeness-flavoured check therefore classifies misses as
    gap-explained (an unrecovered id interval overlaps the check window)
    vs unexplained.
  - Advertised fees come from hourly listpeerchannels.json snapshots
    (local fee_proportional_millionths / fee_base_msat).
  - Flow states come from revenue-status.json channel_states.
  - DTS internals from revenue-fee-debug.json (channel ids masked to
    prefix; joined on scid prefix).

Loop invariants
  LF-1  gossip->record completeness: every advertised-fee diff between
        consecutive listpeerchannels snapshots has a matching recorded
        change chain in that interval (last recorded new == observed new;
        first recorded old == observed old). Misses quantified and split
        into gap-explained vs unexplained (records are best-effort
        post-RPC, so this measures the real miss rate).
  LF-2  record->gossip agreement: the advertised fee at the first
        listpeerchannels snapshot after a recorded change equals that
        change's new_fee_ppm, unless superseded by a later change (recovered
        or gap-possible) before the snapshot.
  LF-2b record-chain continuity: for id-adjacent recovered records on the
        same channel, next.old_fee_ppm == prev.new_fee_ppm (gap-separated
        pairs reported separately).
  LF-3  pause suppression: no recorded change timestamp falls inside a
        paused-snapshot window. (VACUOUS on this corpus: 0 paused snapshots.)
  LF-4  hive-member zero-fee pinning (8630ca6): after the deploy epoch
        (earliest hive_member_zero_fee record, 2026-06-27T22:27:01Z),
        every channel to a member:true peer advertises 0 ppm / 0 base msat.
        (NEAR-VACUOUS: only the single 2026-07-01 snapshot per node is
        post-deploy.) Pre-deploy member fees reported as info; converse
        (zero-fee => member) checked corpus-wide.
  LF-5  bounds: recorded new fees and advertised fees within the
        snapshot-local [min_fee_ppm, max_fee_ppm] (member zero-fee exempt).
  LF-6  state handoff: the flow state embedded in a dts_pid_sample
        decision reason ("state=X") matches the channel_states row for that
        channel in the snapshots bracketing the decision timestamp
        (flow_analysis -> fee_controller handoff, measured at snapshot
        granularity; transitions between snapshots soften this to a rate).
        Also: reason-internal "flow=" (DTS input) vs "state=" consistency.
  LF-7  debug-surface coherence: revenue-fee-debug last_broadcast_fee_ppm
        == advertised fee in the same snapshot dir; fee-debug flow_state ==
        channel_states state in the same snapshot.
  LF-8  ratchet deployment + direction coherence (245ac12): guard-tagged
        changes (zero_flow_ratchet_guard / zero_flow_downshift) appear only
        after the code-stamp transition to 071a5b3 (2026-06-15T19:22Z), and
        ratchet_guard implies new <= old, downshift implies new < old.

Read-only over corpus and repo; writes nothing.
"""

import glob
import gzip
import json
import os
import re
import sys
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone

CORPUS = "/home/sat/cl-mycelium-hermes"
NODES = ("hive-nexus-01", "hive-nexus-02")

CYCLE_SECONDS = 1800          # fee_interval_seconds on both nodes (fee-debug config)
EDGE_SLACK = 120              # LPC capture vs change-timestamp slack
RATCHET_DEPLOY_STAMP = "20260615T192238Z"   # first snapshot stamped 071a5b3 (both nodes)

STATE_RE = re.compile(r"state=([a-z_]+)")
FLOW_RE = re.compile(r"flow=([a-z_]+)")
GUARD_RE = re.compile(r"guard=([a-z_]+)")


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def snap_ts(cmd_dir):
    """Capture time (unix, UTC) from the snapshot dirname."""
    stamp = os.path.basename(os.path.dirname(cmd_dir))
    return int(datetime.strptime(stamp, "%Y%m%dT%H%M%SZ")
               .replace(tzinfo=timezone.utc).timestamp())


def stamp_ts(stamp):
    return int(datetime.strptime(stamp, "%Y%m%dT%H%M%SZ")
               .replace(tzinfo=timezone.utc).timestamp())


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class NodeData:
    """All per-node corpus surfaces, loaded once."""

    def __init__(self, node):
        self.node = node
        self.changes = {}          # id -> change dict (deduped)
        self.first_seen = {}       # id -> cmd_dir where first recovered
        self.snap_times = []       # sorted revenue-status snapshot times
        self.states = {}           # snap_ts -> {channel_id: state}
        self.bounds = {}           # snap_ts -> (min_fee, max_fee)
        self.paused_snaps = []     # snap_ts of paused snapshots
        self.lpc = []              # sorted (snap_ts, {scid: (ppm, base_msat)}, cmd_dir)
        self.members = []          # sorted (snap_ts, frozenset(peer_ids), cmd_dir)
        self.scid_peer = {}        # scid -> peer_id (last seen)
        self.feedbg = []           # (snap_ts, cmd_dir, channels list)
        self._load()

    def _load(self):
        for cmd_dir in sorted(glob.glob(os.path.join(CORPUS, self.node,
                                                     "2*", "2*", "commands"))):
            ts = snap_ts(cmd_dir)
            rs = load_json(os.path.join(cmd_dir, "revenue-status.json"))
            if rs is not None:
                self.snap_times.append(ts)
                values = (rs.get("operator_controls") or {}).get("values") or {}
                self.bounds[ts] = (values.get("min_fee_ppm"),
                                   values.get("max_fee_ppm"))
                if values.get("paused"):
                    self.paused_snaps.append(ts)
                self.states[ts] = {
                    st.get("channel_id"): (st.get("state") or "").lower()
                    for st in rs.get("channel_states") or []
                }
                for ch in rs.get("recent_fee_changes") or []:
                    cid = ch.get("id")
                    if cid is not None and cid not in self.changes:
                        self.changes[cid] = ch
                        self.first_seen[cid] = cmd_dir

            lpc = load_json(os.path.join(cmd_dir, "listpeerchannels.json"))
            if lpc is not None:
                fees = {}
                for chan in lpc.get("channels") or []:
                    scid = chan.get("short_channel_id")
                    if chan.get("state") != "CHANNELD_NORMAL" or not scid:
                        continue
                    fee = chan.get("fee_proportional_millionths")
                    if fee is None:
                        fee = ((chan.get("updates") or {}).get("local") or {}) \
                            .get("fee_proportional_millionths")
                    base = chan.get("fee_base_msat")
                    if base is None:
                        base = ((chan.get("updates") or {}).get("local") or {}) \
                            .get("fee_base_msat")
                    fees[scid] = (fee, base)
                    self.scid_peer[scid] = chan.get("peer_id")
                self.lpc.append((ts, fees, cmd_dir))

            eh = load_json(os.path.join(cmd_dir, "hive-export-hints.json"))
            hints = (eh or {}).get("hints")
            if isinstance(hints, dict):
                self.members.append((ts, frozenset(
                    p for p, h in hints.items()
                    if isinstance(h, dict) and h.get("member")), cmd_dir))

            fdbg = load_json(os.path.join(cmd_dir, "revenue-fee-debug.json"))
            if fdbg is not None and isinstance(fdbg.get("channels"), list):
                self.feedbg.append((ts, cmd_dir, fdbg["channels"]))

        self.snap_times.sort()
        self.lpc.sort(key=lambda x: x[0])
        self.members.sort(key=lambda x: x[0])

        # Per-channel recorded-change index, id-ordered (== time-ordered).
        self.by_chan = defaultdict(list)
        for cid in sorted(self.changes):
            self.by_chan[self.changes[cid]["channel_id"]].append(self.changes[cid])

        # Unrecovered-id intervals: [ts(a), ts(b)] for each id gap (a, b).
        ids = sorted(self.changes)
        self.gaps = []
        for a, b in zip(ids, ids[1:]):
            if b - a > 1:
                self.gaps.append((self.changes[a]["timestamp"],
                                  self.changes[b]["timestamp"], b - a - 1))
        self.missed_ids = sum(g[2] for g in self.gaps)

    def gap_overlaps(self, t0, t1):
        """Does any unrecovered-id interval intersect [t0, t1]?"""
        return any(g0 <= t1 and g1 >= t0 for g0, g1, _ in self.gaps)

    def state_at(self, chan, ts, direction):
        """channel_states state for chan in the nearest snapshot <= / >= ts."""
        i = bisect_right(self.snap_times, ts)
        idxs = range(i - 1, -1, -1) if direction == "before" \
            else range(i, len(self.snap_times))
        for j in idxs:
            st = self.states.get(self.snap_times[j], {})
            if abs(self.snap_times[j] - ts) > 2 * CYCLE_SECONDS:
                return None
            if chan in st:
                return st[chan]
        return None


def main():
    ok = Counter()
    v = Counter()
    info = Counter()
    examples = defaultdict(list)

    def viol(key, example):
        v[key] += 1
        if len(examples[key]) < 6:
            examples[key].append(example)

    def note(key, example):
        info[key] += 1
        if len(examples[key]) < 6:
            examples[key].append(example)

    ratchet_deploy = stamp_ts(RATCHET_DEPLOY_STAMP)

    # Member zero-fee deploy epoch: earliest hive_member_zero_fee record.
    nodes = {n: NodeData(n) for n in NODES}
    member_deploy = min(
        (c["timestamp"] for nd in nodes.values() for c in nd.changes.values()
         if c.get("reason_code") == "hive_member_zero_fee"),
        default=None)

    for node, nd in nodes.items():
        info[f"changes_recovered_{node}"] = len(nd.changes)
        info[f"changes_lost_to_window_{node}"] = nd.missed_ids
        info[f"id_gaps_{node}"] = len(nd.gaps)
        info[f"lpc_snapshots_{node}"] = len(nd.lpc)
        info[f"rs_snapshots_{node}"] = len(nd.snap_times)

        # ---- LF-1: gossip diff -> recorded change ------------------------
        for (t0, f0, d0), (t1, f1, d1) in zip(nd.lpc, nd.lpc[1:]):
            for scid in set(f0) & set(f1):
                a, _ = f0[scid]
                b, _ = f1[scid]
                if a is None or b is None or a == b:
                    continue
                info[f"lf1_gossip_diffs_{node}"] += 1
                # Records near either boundary can land on the other side of
                # the LPC capture instant (record ts is post-RPC write time;
                # capture takes seconds), so gather with slack and accept any
                # contiguous link-consistent subchain a -> ... -> b.
                recs = [c for c in nd.by_chan.get(scid, [])
                        if t0 - EDGE_SLACK < c["timestamp"] <= t1 + EDGE_SLACK]
                where = (f"{node} {scid} {a}->{b} "
                         f"[{iso(t0)}..{iso(t1)}] recs={len(recs)}")
                if not recs:
                    if nd.gap_overlaps(t0 - EDGE_SLACK, t1 + EDGE_SLACK):
                        note("LF-1 miss, gap-explained (lost to 10-row window)",
                             where)
                    else:
                        viol("LF-1 gossip fee change with NO recovered record "
                             "(unexplained)", where)
                    continue
                subchain = None
                for i in range(len(recs)):
                    if recs[i]["old_fee_ppm"] != a:
                        continue
                    j = i
                    while j < len(recs) and (
                            j == i or recs[j]["old_fee_ppm"]
                            == recs[j - 1]["new_fee_ppm"]):
                        if recs[j]["new_fee_ppm"] == b:
                            subchain = recs[i:j + 1]
                            break
                        j += 1
                    if subchain:
                        break
                if subchain:
                    ok[f"LF-1 full chain match ({node})"] += 1
                elif recs[-1]["new_fee_ppm"] == b:
                    if nd.gap_overlaps(t0 - EDGE_SLACK, t1 + EDGE_SLACK):
                        note("LF-1 endpoint match, head lost to gap", where)
                    else:
                        viol("LF-1 chain head mismatch (old endpoint)",
                             f"{where} first_old={recs[0]['old_fee_ppm']}")
                else:
                    if nd.gap_overlaps(t0 - EDGE_SLACK, t1 + EDGE_SLACK):
                        note("LF-1 tail mismatch, gap-explained", where)
                    else:
                        viol("LF-1 chain tail mismatch (new endpoint)",
                             f"{where} last_new={recs[-1]['new_fee_ppm']}")

        # ---- LF-2: recorded change -> advertised fee ---------------------
        lpc_times = [t for t, _, _ in nd.lpc]
        for scid, recs in nd.by_chan.items():
            for i, c in enumerate(recs):
                cts = c["timestamp"]
                j = bisect_left(lpc_times, cts + 5)
                if j >= len(nd.lpc):
                    continue
                st, fees, sdir = nd.lpc[j]
                # superseded by a later recovered record before the snapshot?
                # (EDGE_SLACK: a record stamped seconds after the LPC capture
                # started can still be reflected in the capture)
                if i + 1 < len(recs) and \
                        recs[i + 1]["timestamp"] <= st + EDGE_SLACK:
                    continue
                if scid not in fees or fees[scid][0] is None:
                    info[f"lf2_channel_gone_before_snapshot_{node}"] += 1
                    continue
                adv = fees[scid][0]
                info[f"lf2_checked_{node}"] += 1
                if adv == c["new_fee_ppm"]:
                    ok[f"LF-2 advertised == last recorded new ({node})"] += 1
                elif nd.gap_overlaps(cts, st):
                    note("LF-2 mismatch, gap-explained (possible lost later "
                         "change)", f"{node} {scid} rec {c['old_fee_ppm']}->"
                         f"{c['new_fee_ppm']} @{iso(cts)} adv={adv} @{iso(st)}")
                else:
                    viol("LF-2 advertised fee != last recorded new",
                         f"{node} {scid} rec {c['old_fee_ppm']}->"
                         f"{c['new_fee_ppm']} @{iso(cts)} adv={adv} @{iso(st)}")

        # ---- LF-2b: record chain continuity ------------------------------
        for scid, recs in nd.by_chan.items():
            for p, q in zip(recs, recs[1:]):
                where = (f"{node} {scid} #{p['id']} ...->{p['new_fee_ppm']} "
                         f"then #{q['id']} {q['old_fee_ppm']}->...")
                if q["old_fee_ppm"] == p["new_fee_ppm"]:
                    ok[f"LF-2b chain continuity ({node})"] += 1
                elif any(p["id"] < m < q["id"]
                         for m in range(p["id"] + 1, q["id"])) and \
                        nd.gap_overlaps(p["timestamp"], q["timestamp"]):
                    note("LF-2b discontinuity, gap-explained", where)
                else:
                    viol("LF-2b record chain discontinuity (adjacent ids)",
                         where)

        # ---- LF-3: no change during pause --------------------------------
        info[f"lf3_paused_snapshots_{node}"] = len(nd.paused_snaps)
        for pts in nd.paused_snaps:
            for c in nd.changes.values():
                if abs(c["timestamp"] - pts) < 300 and not c.get("manual"):
                    viol("LF-3 fee change adjacent to paused snapshot",
                         f"{node} #{c['id']} @{iso(c['timestamp'])}")

        # ---- LF-4: member zero-fee pinning --------------------------------
        member_at = nd.members
        mtimes = [t for t, _, _ in member_at]
        for t, fees, cmd_dir in nd.lpc:
            k = bisect_right(mtimes, t) - 1
            if k < 0:
                continue
            mt, mset, _ = member_at[k]
            if t - mt > 2 * 3600:
                continue
            for scid, (fee, base) in fees.items():
                peer = nd.scid_peer.get(scid)
                is_member = peer in mset
                post = member_deploy is not None and t >= member_deploy
                if is_member:
                    if post:
                        info[f"lf4_member_channel_checks_post_deploy_{node}"] += 1
                        if fee == 0 and (base == 0 or base is None):
                            ok["LF-4 member channel pinned 0ppm/0base "
                               "post-deploy"] += 1
                        else:
                            viol("LF-4 member channel NOT pinned post-deploy",
                                 f"{node} {scid} fee={fee} base={base} @{iso(t)}")
                    else:
                        info[f"lf4_member_channel_obs_pre_deploy_{node}"] += 1
                if fee == 0 and not is_member:
                    viol("LF-4 zero-fee channel on non-member peer",
                         f"{node} {scid} peer={peer} @{iso(t)}")

        # ---- LF-5: bounds -------------------------------------------------
        for c in nd.changes.values():
            if c.get("manual"):
                continue
            new = c.get("new_fee_ppm")
            # bounds from nearest revenue-status snapshot before the change
            i = bisect_right(nd.snap_times, c["timestamp"]) - 1
            if i < 0 or new is None:
                continue
            mn, mx = nd.bounds.get(nd.snap_times[i], (None, None))
            if mn is None:
                continue
            if new == 0 and c.get("reason_code") == "hive_member_zero_fee":
                info["lf5_member_zero_exempt"] += 1
            elif mn <= new <= mx:
                ok["LF-5 recorded new fee within snapshot bounds"] += 1
            else:
                viol("LF-5 recorded fee outside configured bounds",
                     f"{node} #{c['id']} {c['old_fee_ppm']}->{new} "
                     f"bounds=[{mn},{mx}] code={c.get('reason_code')}")
        for t, fees, _ in nd.lpc:
            i = bisect_right(nd.snap_times, t) - 1
            mn, mx = nd.bounds.get(nd.snap_times[i], (None, None)) if i >= 0 \
                else (None, None)
            if mn is None:
                continue
            for scid, (fee, base) in fees.items():
                if fee is None:
                    continue
                if fee == 0:
                    info["lf5_advertised_zero_fee"] += 1   # member path (LF-4)
                elif mn <= fee <= mx:
                    ok["LF-5 advertised fee within snapshot bounds"] += 1
                else:
                    viol("LF-5 advertised fee outside bounds",
                         f"{node} {scid} fee={fee} bounds=[{mn},{mx}] @{iso(t)}")

        # ---- LF-6: state handoff ------------------------------------------
        for c in nd.changes.values():
            if c.get("reason_code") != "dts_pid_sample":
                continue
            reason = c.get("reason") or ""
            m = STATE_RE.search(reason)
            if not m:
                note("LF-6 dts change without state= token",
                     f"{node} #{c['id']}")
                continue
            decided = m.group(1)
            fm = FLOW_RE.search(reason)
            if fm and fm.group(1) != decided:
                note("LF-6 reason-internal flow= vs state= disagreement",
                     f"{node} #{c['id']} flow={fm.group(1)} state={decided}")
            before = nd.state_at(c["channel_id"], c["timestamp"], "before")
            after = nd.state_at(c["channel_id"], c["timestamp"], "after")
            info[f"lf6_checked_{node}"] += 1
            if decided == before:
                ok["LF-6 decision state == snapshot state (before)"] += 1
            elif decided == after:
                ok["LF-6 decision state == snapshot state (after only)"] += 1
            elif before is None and after is None:
                note("LF-6 no bracketing snapshot row", f"{node} #{c['id']}")
            else:
                viol("LF-6 decision state matches NEITHER bracketing snapshot",
                     f"{node} #{c['id']} {c['channel_id']} decided={decided} "
                     f"before={before} after={after} @{iso(c['timestamp'])}")

        # ---- LF-7: fee-debug coherence -------------------------------------
        lpc_by_dir = {d: (t, f) for t, f, d in nd.lpc}
        for t, cmd_dir, chans in nd.feedbg:
            lpc_here = lpc_by_dir.get(cmd_dir)
            st_here = nd.states.get(t)
            for dc in chans:
                prefix = (dc.get("channel_id") or "").rstrip(".")
                # join masked scid prefix to full scids
                full = None
                if lpc_here:
                    cands = [s for s in lpc_here[1] if s.startswith(prefix)]
                    if len(cands) == 1:
                        full = cands[0]
                    elif len(cands) > 1:
                        info["lf7_ambiguous_scid_prefix"] += 1
                lb = dc.get("last_broadcast_fee_ppm")
                if full is not None and lb is not None:
                    adv = lpc_here[1][full][0]
                    info[f"lf7_broadcast_checked_{node}"] += 1
                    if adv == lb:
                        ok["LF-7 last_broadcast_fee == advertised"] += 1
                    else:
                        # tolerate an in-flight change in this very snapshot
                        recs = [c for c in nd.by_chan.get(full, [])
                                if abs(c["timestamp"] - t) <= CYCLE_SECONDS]
                        if any(c["new_fee_ppm"] == adv or
                               c["old_fee_ppm"] == adv for c in recs) or \
                                nd.gap_overlaps(t - CYCLE_SECONDS, t):
                            note("LF-7 broadcast/advertised skew, "
                                 "in-flight-change or gap-explained",
                                 f"{node} {full} lb={lb} adv={adv} @{iso(t)}")
                        else:
                            viol("LF-7 last_broadcast_fee != advertised "
                                 "(no in-flight change)",
                                 f"{node} {full} lb={lb} adv={adv} @{iso(t)}")
                fs = dc.get("flow_state")
                if full and st_here and fs and full in st_here:
                    info[f"lf7_state_checked_{node}"] += 1
                    if fs == st_here[full]:
                        ok["LF-7 fee-debug flow_state == channel_states"] += 1
                    else:
                        viol("LF-7 fee-debug flow_state != channel_states "
                             "same snapshot",
                             f"{node} {full} debug={fs} states={st_here[full]}"
                             f" @{iso(t)}")

        # ---- LF-8: ratchet deploy + direction ------------------------------
        for c in nd.changes.values():
            g = GUARD_RE.search(c.get("reason") or "")
            if not g:
                continue
            guard = g.group(1).rstrip(")")
            old, new = c.get("old_fee_ppm"), c.get("new_fee_ppm")
            where = (f"{node} #{c['id']} {c['channel_id']} {old}->{new} "
                     f"guard={guard} @{iso(c['timestamp'])}")
            info[f"lf8_guard_{guard}_{node}"] += 1
            if c["timestamp"] < ratchet_deploy - CYCLE_SECONDS:
                viol("LF-8 guard-tagged change BEFORE ratchet deploy", where)
            elif new > old:
                # The guard returns max(floor_ppm, min(target, cap)) with
                # cap < current, so new > old is only reachable through the
                # floor arm: the effective floor (config min + rebalance-cost
                # floor) sits ABOVE the current fee and wins by design
                # ("hard floors are not negotiable"). Loop-coherent, but the
                # reason string still says guard=zero_flow_downshift while
                # applied=up — misleading telemetry, reported as info.
                note("LF-8 guard overridden by effective floor "
                     "(design; telemetry says downshift but fee rose)", where)
            elif guard == "zero_flow_downshift" and not new < old:
                viol("LF-8 downshift did not lower fee", where)
            else:
                ok["LF-8 guard-tagged change post-deploy, correct direction"] += 1

    # ---- report -----------------------------------------------------------
    print("=" * 78)
    print("PHASE 3 LOOP SWEEP — fee decision loop (flow -> decision -> gossip "
          "-> record)")
    print("=" * 78)
    if member_deploy:
        print(f"member zero-fee deploy epoch: {iso(member_deploy)} "
              f"(earliest hive_member_zero_fee record)")
    print(f"ratchet deploy stamp: {RATCHET_DEPLOY_STAMP} (code-stamp 071a5b3)")
    print("\n-- pass counts --")
    for k in sorted(ok):
        print(f"  {k}: {ok[k]}")
    print("\n-- violations --")
    if not v:
        print("  (none)")
    for k in sorted(v):
        print(f"  {k}: {v[k]}")
        for e in examples[k]:
            print(f"      e.g. {e}")
    print("\n-- informational / soft --")
    for k in sorted(info):
        print(f"  {k}: {info[k]}")
        if k in examples and not v.get(k):
            for e in examples[k][:3]:
                print(f"      e.g. {e}")
    return 1 if v else 0


if __name__ == "__main__":
    sys.exit(main())
