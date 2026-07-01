#!/usr/bin/env python3
"""Read-only hermes-corpus sweep for the Tier-1 fee stack verification (Phase 2).

Walks EVERY snapshot under /home/sat/cl-mycelium-hermes/{hive-nexus-01,hive-nexus-02}
and checks every corpus-observable instance of the fee_controller / flow_analysis /
policy_manager intent-contract invariants:

  FC-I1a  recent_fee_changes new_fee_ppm within [min_fee_ppm, max_fee_ppm]
          (non-manual; snapshot-local config; 0-ppm hive-member fees exempted+counted)
  FC-I1b  listpeerchannels advertised local fee within snapshot-local bounds
          (0-ppm exempted+counted, cross-checked against hive membership)
  FC-I2   fee_decision reason=="paused" only when operator_controls paused
  FC-I3   count of fee_decision reason=="adjustment_in_progress"
  FC-I9   recorded fee changes (excl. gossip_refresh) have new != old
  FC-I10  dts_pid_sample delta within per-cycle cap:
             wake:none -> max(ceil(0.5*old), 100);  wake:* -> max(ceil(0.2*old), 50)
          (active profile on both nodes, verified from operator_controls)
  ALPHA   dts_pid_sample delta >= alpha-guard min_change
             (old<100 -> 1; else max(5, ceil(3% of old)))  [soft: htlc/base-fee
             policy-change cycles legitimately bypass the guard]
  GOSSIP  dts_pid_sample delta <= 5% of old counted (soft: state-category-change
          broadcasts legitimately bypass the 5% gate)
  FA-I1   channel_states flow_ratio in [-1, 1]
  FA-I4   channel_states confidence in [0.1, 1.0]
  FA-I5   |velocity| <= 0.5 AND |velocity| <= 3*(|flow_ratio|+0.01)+eps;
          |kalman_velocity| <= 0.5/24 + eps
  FA-VOC  emitted state vocabulary; 'router' must never appear
  FA-I12  channel_states channel_ids subset of live CHANNELD_NORMAL scids
          (same snapshot dir; transient one-snapshot residue reported separately
           from persistent residue)

Prints per-invariant pass/violation counts with example paths on violation.
Purely read-only; writes nothing.
"""

import glob
import gzip
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

CORPUS = "/home/sat/cl-mycelium-hermes"
NODES = ("hive-nexus-01", "hive-nexus-02")

EPS = 1e-9
KALMAN_MAX_VELOCITY = 0.5 / 24.0

KNOWN_STATES = {
    "source", "sink", "balanced", "balanced_active", "dormant", "congested", "unknown",
}


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def snapshot_dirs(node):
    return sorted(glob.glob(os.path.join(CORPUS, node, "2*", "2*", "commands")))


WAKE_RE = re.compile(r"wake:([a-zA-Z_0-9]+)")


def main():
    v = Counter()          # violation counts
    ok = Counter()         # pass counts
    info = Counter()       # informational counts
    examples = defaultdict(list)

    def viol(key, example):
        v[key] += 1
        if len(examples[key]) < 5:
            examples[key].append(example)

    seen_changes = set()   # (node, change id) dedupe
    residue = defaultdict(lambda: [None, None, 0])  # (node,chid) -> [first,last,count]
    hive_member_cache = {}

    for node in NODES:
        for cmd_dir in snapshot_dirs(node):
            rs_path = os.path.join(cmd_dir, "revenue-status.json")
            rs = load_json(rs_path)
            if rs is None:
                continue
            values = (rs.get("operator_controls") or {}).get("values") or {}
            min_fee = values.get("min_fee_ppm")
            max_fee = values.get("max_fee_ppm")
            paused = bool(values.get("paused"))

            # ---- FC-I2 / FC-I3 -------------------------------------------
            fd = rs.get("fee_decision") or {}
            if fd.get("reason") == "paused":
                info["fc_i2_paused_decisions"] += 1
                if not paused:
                    viol("FC-I2 paused-decision while not paused", rs_path)
            if paused:
                info["fc_i2_paused_snapshots"] += 1
                if fd.get("action") != "suppressed":
                    # last decision may lag one cycle after a pause toggle;
                    # still record it — operator can adjudicate.
                    viol("FC-I2 paused but decision not suppressed", rs_path)
            if fd.get("reason") == "adjustment_in_progress":
                info["fc_i3_concurrency_suppressions"] += 1

            # ---- fee change record checks --------------------------------
            for ch in rs.get("recent_fee_changes") or []:
                key = (node, ch.get("id"))
                if key in seen_changes:
                    continue
                seen_changes.add(key)
                old = ch.get("old_fee_ppm")
                new = ch.get("new_fee_ppm")
                code = ch.get("reason_code")
                manual = bool(ch.get("manual"))
                reason = ch.get("reason") or ""
                where = f"{rs_path} change_id={ch.get('id')} {old}->{new} code={code}"
                if old is None or new is None:
                    info["fee_changes_missing_fields"] += 1
                    continue
                info[f"fee_changes_{node}"] += 1

                # FC-I1a: execution clamp on recorded changes
                if manual:
                    info["fee_changes_manual"] += 1
                elif new == 0:
                    info["fc_i1a_zero_fee_changes"] += 1  # hive-member fleet policy path
                elif min_fee is not None and (new < min_fee or new > max_fee):
                    viol("FC-I1a recorded fee outside configured bounds", where)
                else:
                    ok["FC-I1a"] += 1

                # FC-I9: idempotency — recorded changes actually change the fee
                if code != "gossip_refresh":
                    if new == old:
                        viol("FC-I9 recorded no-op fee change", where)
                    else:
                        ok["FC-I9"] += 1
                else:
                    info["gossip_refresh_changes"] += 1
                    if abs(new - old) > 1:
                        viol("FC-I9/gossip_refresh nudge larger than 1ppm", where)

                # FC-I10 + alpha guard + gossip gate (dts_pid_sample only)
                if code == "dts_pid_sample" and not manual and old > 0:
                    delta = abs(new - old)
                    m = WAKE_RE.search(reason)
                    wake = m.group(1) if m else None
                    if wake is None:
                        info["fc_i10_no_wake_field"] += 1
                    else:
                        if wake == "none":
                            cap = max(math.ceil(0.5 * old), 100)
                        else:
                            cap = max(math.ceil(0.2 * old), 50)
                        if delta > cap:
                            viol("FC-I10 delta above per-cycle cap", f"{where} wake={wake} cap={cap}")
                        else:
                            ok["FC-I10"] += 1
                    # Alpha guard (soft: policy-change cycles bypass)
                    min_change = 1 if old < 100 else max(5, -(-old * 3 // 100))
                    if delta < min_change:
                        info["alpha_guard_subthreshold_changes"] += 1
                        if len(examples["ALPHA subthreshold (soft)"]) < 5:
                            examples["ALPHA subthreshold (soft)"].append(where)
                    # Gossip gate (soft: state-category transitions bypass)
                    if delta <= 0.05 * old:
                        info["gossip_gate_sub5pct_changes"] += 1

            # ---- channel_states checks (FA) -------------------------------
            for st in rs.get("channel_states") or []:
                chid = st.get("channel_id")
                fr = st.get("flow_ratio")
                conf = st.get("confidence")
                vel = st.get("velocity")
                kvel = st.get("kalman_velocity")
                state = (st.get("state") or "").lower()
                where = f"{rs_path} {chid}"
                info["channel_state_rows"] += 1

                if fr is None or not (-1.0 - EPS <= fr <= 1.0 + EPS):
                    viol("FA-I1 flow_ratio out of [-1,1]", f"{where} flow_ratio={fr}")
                else:
                    ok["FA-I1"] += 1

                if conf is None or not (0.1 - EPS <= conf <= 1.0 + EPS):
                    viol("FA-I4 confidence out of [0.1,1.0]", f"{where} confidence={conf}")
                else:
                    ok["FA-I4"] += 1

                bad_vel = False
                if vel is None or abs(vel) > 0.5 + EPS:
                    bad_vel = True
                elif fr is not None and abs(vel) > 3.0 * (abs(fr) + 0.01) + 1e-6:
                    bad_vel = True
                if bad_vel:
                    viol("FA-I5 velocity out of bounds", f"{where} velocity={vel} flow_ratio={fr}")
                else:
                    ok["FA-I5 velocity"] += 1

                if kvel is None or abs(kvel) > KALMAN_MAX_VELOCITY + 1e-9:
                    viol("FA-I5 kalman_velocity out of bounds", f"{where} kalman_velocity={kvel}")
                else:
                    ok["FA-I5 kalman_velocity"] += 1

                info[f"state_vocab::{state}"] += 1
                if state == "router":
                    viol("FA-VOC 'router' emitted", where)
                elif state not in KNOWN_STATES:
                    viol("FA-VOC unknown state", f"{where} state={state}")

            # ---- listpeerchannels checks ----------------------------------
            lpc_path = os.path.join(cmd_dir, "listpeerchannels.json")
            lpc = load_json(lpc_path)
            if lpc is None:
                continue
            live = {}
            member_peers = None
            for chan in lpc.get("channels") or []:
                scid = chan.get("short_channel_id")
                if chan.get("state") == "CHANNELD_NORMAL" and scid:
                    live[scid] = chan
                    fee = chan.get("fee_proportional_millionths")
                    if fee is None:
                        upd = (chan.get("updates") or {}).get("local") or {}
                        fee = upd.get("fee_proportional_millionths")
                    if fee is None:
                        info["fc_i1b_missing_fee"] += 1
                        continue
                    info["fc_i1b_channels_checked"] += 1
                    if fee == 0:
                        info["fc_i1b_zero_fee_channels"] += 1
                        # cross-check hive membership lazily
                        if member_peers is None:
                            hh = load_json(os.path.join(cmd_dir, "revenue-hive-hints-status.json"))
                            member_peers = set()
                            if isinstance(hh, dict):
                                blob = json.dumps(hh)
                                # membership lists vary in shape; collect any pubkeys
                                member_peers = set(re.findall(r"0[23][0-9a-f]{64}", blob))
                        if chan.get("peer_id") not in (member_peers or set()):
                            info["fc_i1b_zero_fee_nonmember"] += 1
                            if len(examples["FC-I1b zero-fee on non-hive peer (info)"]) < 5:
                                examples["FC-I1b zero-fee on non-hive peer (info)"].append(
                                    f"{lpc_path} {scid} peer={chan.get('peer_id')}")
                    elif min_fee is not None and (fee < min_fee or fee > max_fee):
                        viol("FC-I1b advertised fee outside bounds", f"{lpc_path} {scid} fee={fee}")
                    else:
                        ok["FC-I1b"] += 1

            # FA-I12: residue — channel_states not in live set
            for st in rs.get("channel_states") or []:
                chid = st.get("channel_id")
                if chid and chid not in live:
                    r = residue[(node, chid)]
                    ts = cmd_dir
                    if r[0] is None:
                        r[0] = ts
                    r[1] = ts
                    r[2] += 1

    # ---- report ----------------------------------------------------------
    print("=" * 78)
    print("SWEEP RESULTS — fee stack (fee_controller / flow_analysis / policy_manager)")
    print("=" * 78)
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
    print("\n-- informational --")
    for k in sorted(info):
        print(f"  {k}: {info[k]}")
    for k in ("ALPHA subthreshold (soft)", "FC-I1b zero-fee on non-hive peer (info)"):
        if examples.get(k):
            print(f"  examples {k}:")
            for e in examples[k]:
                print(f"      {e}")

    # FA-I12 residue classification
    transient = {k: r for k, r in residue.items() if r[2] == 1}
    persistent = {k: r for k, r in residue.items() if r[2] > 1}
    print("\n-- FA-I12 channel_states residue (states for channels not live) --")
    print(f"  transient (1 snapshot): {len(transient)} channel(s)")
    print(f"  persistent (>1 snapshot): {len(persistent)} channel(s)")
    for (node, chid), r in sorted(persistent.items(), key=lambda kv: -kv[1][2])[:10]:
        print(f"      {node} {chid}: {r[2]} snapshots, first={r[0]}, last={r[1]}")

    return 1 if v else 0


if __name__ == "__main__":
    sys.exit(main())
