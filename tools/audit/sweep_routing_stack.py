#!/usr/bin/env python3
"""Read-only corpus sweep for the Tier-2 routing stack verification (Phase 2).

Modules covered: rebalance_executor (RX-*), rebalance_router_v2 (R2-*),
rebalance_router_v3 (R3-*), hive_router (HR-*), rebalance_hive_router (RHR-*),
rebalance_coordination_overlay (RCO-*).

These modules are weakly corpus-observable; this sweep checks every claim in
their contracts' "Observable surface" sections that IS visible in the hermes
corpus (/home/sat/cl-mycelium-hermes/hive-nexus-0*/<date>/<ts>/commands/):

  revenue-rebalance-debug.json (last_cycle candidates/skipped/executions):
    C1  route_cost_sats >= 0 on every considered/selected candidate   [R2-5]
    C2  route_summary hop amounts monotonically non-increasing        [R2-3/R3-6/RX-2 analog]
    C3  route_summary first hop channel == source_channel_id          [RHR-1 / source pinning]
    C4  route_summary last hop amount_msat == amount_sats * 1000      [pricing consistency]
    C5  ceil((first_hop - last_hop)/1000) == route_cost_sats (+/-1)   [R2-5 formula]
    C6  route_policy histogram (hive_only/hybrid/market_only)         [RHR-2 surface]
    C7  reason_code histogram (coordinated_rebalance presence)        [RCO surface]
    C8  skip-reason histogram, incl. coordination_* / lease / router
        error reasons                                                 [RCO-1/2/5/6, R3/RHR errors]
    C9  executions: attempts <= 3, fee_sats >= 0; when executions
        pair 1:1 with selected candidates, fee <= pair_budget         [budget gating downstream of R3-8/RHR-7]

  revenue-status.json (recent_rebalances, deduped by (node,id,status)):
    S1  successful rebalances: actual_fee_sats <= max_fee_sats        [engine/executor budget gate]
    S2  status / rebalance_type / reason_code histograms              [RCO surface: coordinated lineage]
    S3  error_message leading-token histogram (stable/translated
        route errors: no_route, unknown_layer, no_fleet_route, ...)   [R3-5/RHR-5 surface]

  listdatastore-key-revenue-segment-observations.json (hex-decoded):
    O1  router_kind histogram — expected ONLY "v3"                    [R3 surface; v2 retirement]
    O2  route_policy histogram; confidence in [0,1]                   [RHR/RCO surface]

  revenue-spend-ledger.json:
    L1  per-category spend aggregates are non-negative; rebalance
        category presence/magnitude                                   [spend surface]

NOT corpus-observable (no checks possible): askrene layer contents
(hive-fleet / revenue-local biases, exclude-layer lifecycle), reserve/unreserve
calls, RebalanceExecutor behavior (dead code), v2 price_pair output, HiveRouter
discover_route results.

Usage: python3 tools/audit/sweep_routing_stack.py [--root PATH] [--json]
Exit 0 when no violations, 1 otherwise. Strictly read-only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_ROOT = os.environ.get("CL_MYCELIUM_HERMES_ROOT", "/home/sat/cl-mycelium-hermes")
NODES = ("hive-nexus-01", "hive-nexus-02")

COORDINATION_SKIP_REASONS = {
    "coordination_unresolvable_endpoint",
    "coordination_unavailable",
    "not_designated_executor",
    "lease_conflict",
    "fleet_lease_held",
    "coordination_preempted",
}
ROUTER_ERROR_REASONS = {
    "no_route",
    "unknown_layer",
    "askrene_child_died",
    "no_fleet_route",
    "fleet_source_mismatch",
    "non_hive_intermediate",
    "fleet_invalid_amount",
    "path_loops_through_us",
    "route_pricing_failed",
}
MAX_EXAMPLES = 5


def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


class Sweep:
    def __init__(self) -> None:
        self.files_scanned = Counter()
        self.violations: dict[str, list[str]] = defaultdict(list)
        self.counts = Counter()
        self.route_policy_hist = Counter()
        self.reason_code_hist = Counter()
        self.skip_reason_hist = Counter()
        self.coordination_skips = Counter()
        self.router_error_reasons = Counter()
        self.exec_attempts_hist = Counter()
        self.exec_status = Counter()
        self.rebalance_status_hist = Counter()
        self.rebalance_type_hist = Counter()
        self.rebalance_reason_code_hist = Counter()
        self.error_token_hist = Counter()
        self.router_kind_hist = Counter()
        self.obs_route_policy_hist = Counter()
        self.spend_categories = Counter()
        self.seen_rebalance_ids: set = set()
        self.seen_observation_ids: set = set()

    def violate(self, check: str, where: str) -> None:
        bucket = self.violations[check]
        if len(bucket) < MAX_EXAMPLES:
            bucket.append(where)
        self.counts[f"violations:{check}"] += 1

    # ---------------- revenue-rebalance-debug.json ----------------

    def _check_candidate(self, cand: dict, where: str, kind: str) -> None:
        self.counts["candidates"] += 1
        cost = cand.get("route_cost_sats")
        amount_sats = cand.get("amount_sats")
        if isinstance(cost, (int, float)) and cost < 0:
            self.violate("C1_negative_route_cost", where)
        policy = cand.get("route_policy")
        if policy:
            self.route_policy_hist[str(policy)] += 1
        rc = cand.get("reason_code")
        if rc:
            self.reason_code_hist[str(rc)] += 1
        rej = cand.get("rejection_reason")
        if rej:
            self.skip_reason_hist[str(rej)] += 1
            if rej in ROUTER_ERROR_REASONS:
                self.router_error_reasons[str(rej)] += 1
        hops = cand.get("route_summary") or []
        if not isinstance(hops, list) or not hops:
            return
        amounts = [h.get("amount_msat") for h in hops if isinstance(h, dict)]
        if not all(isinstance(a, (int, float)) for a in amounts) or len(amounts) != len(hops):
            return
        self.counts["candidates_with_route"] += 1
        for prev, cur in zip(amounts, amounts[1:]):
            if cur > prev:
                self.violate("C2_amount_increases_along_route", where)
                break
        first_chan = hops[0].get("channel")
        src = cand.get("source_channel_id")
        if src and first_chan and first_chan != src:
            self.violate("C3_first_hop_not_source", where)
        if isinstance(amount_sats, (int, float)):
            if amounts[-1] != amount_sats * 1000:
                self.violate("C4_last_hop_not_delivery", where)
            if isinstance(cost, (int, float)):
                implied = math.ceil((amounts[0] - amounts[-1]) / 1000)
                if abs(implied - cost) > 1:
                    self.violate("C5_cost_mismatch", f"{where} (implied={implied} reported={cost})")

    def sweep_rebalance_debug(self, path: Path) -> None:
        data = load_json(path)
        if not isinstance(data, dict):
            return
        self.files_scanned["revenue-rebalance-debug.json"] += 1
        lc = data.get("last_cycle") or {}
        where = str(path)
        for kind in ("considered_candidates", "selected_candidates"):
            for cand in lc.get(kind) or []:
                if isinstance(cand, dict):
                    self._check_candidate(cand, where, kind)
        for skip in lc.get("skipped") or []:
            if not isinstance(skip, dict):
                continue
            reason = str(skip.get("reason") or "")
            if reason:
                self.skip_reason_hist[reason] += 1
                if reason in COORDINATION_SKIP_REASONS:
                    self.coordination_skips[reason] += 1
                if reason in ROUTER_ERROR_REASONS:
                    self.router_error_reasons[reason] += 1
        selected = lc.get("selected_candidates") or []
        executions = lc.get("executions") or []
        for idx, ex in enumerate(executions):
            if not isinstance(ex, dict):
                continue
            self.counts["executions"] += 1
            self.exec_status["success" if ex.get("success") else "failure"] += 1
            attempts = ex.get("attempts")
            if isinstance(attempts, int):
                self.exec_attempts_hist[attempts] += 1
                if attempts > 3:
                    self.violate("C9_attempts_gt_3", where)
            fee = ex.get("fee_sats")
            if isinstance(fee, (int, float)):
                if fee < 0:
                    self.violate("C9_negative_fee", where)
                # 1:1 pairing with selected candidates only when shapes agree
                if ex.get("success") and len(executions) == len(selected):
                    cand = selected[idx] if idx < len(selected) else None
                    budget = (cand or {}).get("pair_budget_sats")
                    if isinstance(budget, (int, float)) and fee > budget:
                        self.violate(
                            "C9_fee_over_pair_budget", f"{where} (fee={fee} budget={budget})"
                        )

    # ---------------- revenue-status.json ----------------

    def sweep_revenue_status(self, node: str, path: Path) -> None:
        data = load_json(path)
        if not isinstance(data, dict):
            return
        self.files_scanned["revenue-status.json"] += 1
        for entry in data.get("recent_rebalances") or []:
            if not isinstance(entry, dict):
                continue
            key = (node, entry.get("id"), entry.get("status"))
            if key in self.seen_rebalance_ids:
                continue
            self.seen_rebalance_ids.add(key)
            self.counts["rebalance_history_entries"] += 1
            status = str(entry.get("status") or "")
            self.rebalance_status_hist[status] += 1
            self.rebalance_type_hist[str(entry.get("rebalance_type") or "")] += 1
            rc = entry.get("reason_code")
            if rc:
                self.rebalance_reason_code_hist[str(rc)] += 1
            fee = entry.get("actual_fee_sats")
            cap = entry.get("max_fee_sats")
            if (
                status in ("success", "succeeded", "completed")
                and isinstance(fee, (int, float))
                and isinstance(cap, (int, float))
                and fee > cap
            ):
                self.violate("S1_fee_over_max_fee", f"{path} (id={entry.get('id')} fee={fee} max={cap})")
            err = entry.get("error_message")
            if err:
                token = str(err).split(":", 1)[0].strip()
                self.error_token_hist[token] += 1

    # ---------------- segment observations ----------------

    def sweep_segment_observations(self, path: Path) -> None:
        data = load_json(path)
        if not isinstance(data, dict):
            return
        self.files_scanned["segment-observations"] += 1
        for ds in data.get("datastore") or []:
            hexed = (ds or {}).get("hex")
            if not hexed:
                continue
            try:
                payload = json.loads(bytes.fromhex(hexed))
            except Exception:
                self.violate("O0_undecodable_segment_observations", str(path))
                continue
            for obs in payload.get("segment_observations") or []:
                if not isinstance(obs, dict):
                    continue
                oid = obs.get("observation_id")
                if oid in self.seen_observation_ids:
                    continue
                self.seen_observation_ids.add(oid)
                self.counts["segment_observations"] += 1
                rk = obs.get("router_kind")
                if rk is not None:
                    self.router_kind_hist[str(rk)] += 1
                    if rk != "v3":
                        self.violate("O1_router_kind_not_v3", f"{path} ({oid} router_kind={rk})")
                rp = obs.get("route_policy")
                if rp is not None:
                    self.obs_route_policy_hist[str(rp)] += 1
                conf = obs.get("confidence")
                if isinstance(conf, (int, float)) and not (0 <= conf <= 1):
                    self.violate("O2_confidence_out_of_range", f"{path} ({oid} conf={conf})")

    # ---------------- spend ledger ----------------

    def sweep_spend_ledger(self, path: Path) -> None:
        data = load_json(path)
        if not isinstance(data, dict):
            return
        self.files_scanned["revenue-spend-ledger.json"] += 1
        for field in ("spent_by_category", "reserved_by_category"):
            cats = data.get(field)
            if not isinstance(cats, dict):
                continue
            for cat, sats in cats.items():
                if isinstance(sats, (int, float)):
                    if sats < 0:
                        self.violate("L1_negative_spend", f"{path} ({field}.{cat}={sats})")
                    self.spend_categories[f"{field}.{cat}"] = max(
                        self.spend_categories.get(f"{field}.{cat}", 0), int(sats)
                    )

    # ---------------- report ----------------

    def report(self, as_json: bool) -> int:
        def top(counter: Counter, n: int = 15):
            return dict(counter.most_common(n))

        summary = {
            "files_scanned": dict(self.files_scanned),
            "counts": dict(self.counts),
            "route_policy_hist (debug candidates)": dict(self.route_policy_hist),
            "candidate_reason_codes": top(self.reason_code_hist),
            "skip_reasons_top": top(self.skip_reason_hist, 25),
            "coordination_skip_reasons": dict(self.coordination_skips),
            "router_error_reasons": dict(self.router_error_reasons),
            "execution_status": dict(self.exec_status),
            "execution_attempts_hist": {str(k): v for k, v in sorted(self.exec_attempts_hist.items())},
            "rebalance_history_status": dict(self.rebalance_status_hist),
            "rebalance_history_types": dict(self.rebalance_type_hist),
            "rebalance_history_reason_codes": top(self.rebalance_reason_code_hist),
            "rebalance_error_tokens_top": top(self.error_token_hist, 20),
            "segment_obs_router_kind": dict(self.router_kind_hist),
            "segment_obs_route_policy": dict(self.obs_route_policy_hist),
            "spend_category_max_observed": dict(self.spend_categories),
            "violations": {k: {"count": self.counts[f"violations:{k}"], "examples": v}
                           for k, v in sorted(self.violations.items())},
        }
        if as_json:
            print(json.dumps(summary, indent=2))
        else:
            for section, value in summary.items():
                print(f"== {section} ==")
                print(json.dumps(value, indent=2))
                print()
        total_violations = sum(
            v for k, v in self.counts.items() if k.startswith("violations:")
        )
        print(f"TOTAL VIOLATIONS: {total_violations}", file=sys.stderr)
        return 1 if total_violations else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--json", action="store_true", help="emit a single JSON summary")
    args = ap.parse_args()

    root = Path(args.root)
    sweep = Sweep()
    for node in NODES:
        node_root = root / node
        if not node_root.is_dir():
            continue
        for path in sorted(node_root.glob("*/*/commands/revenue-rebalance-debug.json")):
            sweep.sweep_rebalance_debug(path)
        for path in sorted(node_root.glob("*/*/commands/revenue-status.json")):
            sweep.sweep_revenue_status(node, path)
        for path in sorted(
            node_root.glob("*/*/commands/listdatastore-key-revenue-segment-observations.json")
        ):
            sweep.sweep_segment_observations(path)
        for path in sorted(node_root.glob("*/*/commands/revenue-spend-ledger.json")):
            sweep.sweep_spend_ledger(path)
    return sweep.report(args.json)


if __name__ == "__main__":
    sys.exit(main())
