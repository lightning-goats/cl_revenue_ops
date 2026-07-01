#!/usr/bin/env python3
"""Read-only hermes corpus sweep for modules/profitability_analyzer.py (PA-I*).

Phase 2 verification campaign. Walks /home/sat/cl-mycelium-hermes and checks
every PA invariant observable in revenue-profitability.json (the no-arg RPC
view: summary + channels_by_class rows) plus revenue-dashboard.json.

Sweepable here:
  PA-I9   - total_contribution_sats == max(fees_earned_sats,
            sourced_fee_contribution_sats) per channel row (ceil() is
            monotonic, so max-of-ceilings == ceiling-of-max; the sum would
            show as total > max component).
  PA-SUM  - summary class counts match channels_by_class list lengths and
            profitable/underwater/... partition all channels (internal
            consistency of the classification pass).
  PA-CLS  - classification vs roi_percentage boundaries:
              PROFITABLE  => roi > +5%  (widened floor 10%*0.5; synthetic
                             zero-cost earners carry roi=100%)
              UNDERWATER  => roi < -10% (threshold only widens to -15%)
            BREAK_EVEN rows with roi < -10% are counted separately as
            observed STRUCTURAL-PROTECTION UPGRADES (the D2 mask: hive
            member / corridor owner / centrality > 0.03 rewrites UNDERWATER
            to BREAK_EVEN). Rows in the widening band (-15%..-10% for
            underwater, +5%..+10% for profitable) are legal via Thompson
            threshold widening and are not violations.
  PA-I11w - (weak ceiling check) component sats fields are non-negative ints
            and total_contribution >= each component's implied floor.

Not corpus-observable (verified by unit tests / code reading instead):
  PA-I1 (zombie diagnostics gate inputs), PA-I2 (fee multiplier not in this
  surface), PA-I3 (priority multipliers), PA-I4 (open-cost validation),
  PA-I5/I6 (bleeder v2 verdicts are in-process only), PA-I7/I8 (30d fields
  not in this surface), PA-I10 (concurrency), PA-I11 exact msat rounding
  (msat values not exported here), PA-I12 (datastore key not captured).
"""

import gzip
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

CORPUS = Path(os.environ.get("CL_MYCELIUM_HERMES_ROOT", "/home/sat/cl-mycelium-hermes"))
NODES = ["hive-nexus-01", "hive-nexus-02"]

CLASS_KEYS = ["profitable", "break_even", "underwater", "stagnant_candidate", "zombie"]
SUMMARY_COUNT_KEYS = {
    "profitable": "profitable_count",
    "break_even": "break_even_count",
    "underwater": "underwater_count",
    "stagnant_candidate": "stagnant_candidate_count",
    "zombie": "zombie_count",
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


def iter_snapshots(node: str):
    base = CORPUS / node
    if not base.is_dir():
        return
    for day in sorted(p for p in base.iterdir() if p.is_dir() and p.name[:4].isdigit()):
        for snap in sorted(p for p in day.iterdir() if p.is_dir()):
            cmds = snap / "commands"
            if cmds.is_dir():
                yield cmds


class Check:
    def __init__(self, name, desc):
        self.name, self.desc = name, desc
        self.passed = 0
        self.violations = []

    def ok(self, n=1):
        self.passed += n

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
    c_max = Check("PA-I9", "total_contribution_sats == max(fees_earned_sats, sourced_fee_contribution_sats)")
    c_sum = Check("PA-SUM", "summary class counts == channels_by_class list lengths (+ total)")
    c_prof = Check("PA-CLS-P", "profitable rows: roi_percentage > +5% (widened floor)")
    c_under = Check("PA-CLS-U", "underwater rows: roi_percentage < -10%")
    c_nonneg = Check("PA-I11w", "sats fields non-negative; total >= each component")

    # anomaly / decision-support material
    upgrades = []            # break_even rows with roi < -10 (structural mask, D2)
    band_profitable = 0      # profitable rows with 5 < roi <= 10 (widening in effect)
    masked_loss_sats = 0     # sum of |net_profit| for observed upgrades
    upgrade_channels = defaultdict(int)
    class_totals = defaultdict(int)
    files = 0

    for node in NODES:
        for cmds in iter_snapshots(node):
            prof = load(cmds / "revenue-profitability.json")
            if not prof or "channels_by_class" not in prof:
                continue
            files += 1
            summary = prof.get("summary") or {}
            cbc = prof.get("channels_by_class") or {}

            ok_counts = True
            total = 0
            for cls in CLASS_KEYS:
                rows = cbc.get(cls) or []
                total += len(rows)
                want = summary.get(SUMMARY_COUNT_KEYS[cls])
                if want is not None and want != len(rows):
                    ok_counts = False
                    c_sum.bad(cmds, f"{cls}: summary says {want}, list has {len(rows)}")
            if summary.get("total_channels") is not None and summary["total_channels"] != total:
                ok_counts = False
                c_sum.bad(cmds, f"total_channels={summary['total_channels']} but lists hold {total}")
            if ok_counts:
                c_sum.ok()

            for cls in CLASS_KEYS:
                for row in cbc.get(cls) or []:
                    class_totals[cls] += 1
                    cid = row.get("channel_id", "?")
                    fees = row.get("fees_earned_sats")
                    sourced = row.get("sourced_fee_contribution_sats")
                    tot = row.get("total_contribution_sats")
                    roi = row.get("roi_percentage")

                    if None not in (fees, sourced, tot):
                        if tot == max(fees, sourced):
                            c_max.ok()
                        else:
                            c_max.bad(cmds, f"{cid}: total={tot} != max({fees},{sourced})")
                        if fees >= 0 and sourced >= 0 and tot >= fees and tot >= sourced:
                            c_nonneg.ok()
                        else:
                            c_nonneg.bad(cmds, f"{cid}: fees={fees} sourced={sourced} total={tot}")

                    if not isinstance(roi, (int, float)):
                        continue
                    if cls == "profitable":
                        if roi > 5.0 - 0.01:
                            c_prof.ok()
                            if roi <= 10.0:
                                band_profitable += 1
                        else:
                            c_prof.bad(cmds, f"{cid}: PROFITABLE with roi={roi}%")
                    elif cls == "underwater":
                        if roi < -10.0 + 0.01:
                            c_under.ok()
                        else:
                            c_under.bad(cmds, f"{cid}: UNDERWATER with roi={roi}%")
                    elif cls == "break_even" and roi < -10.0:
                        # Structural-protection upgrade (operator decision D2:
                        # ruled for removal; count occurrences + masked loss).
                        upgrades.append((str(cmds), cid, roi, row.get("net_profit_sats")))
                        upgrade_channels[cid] += 1
                        masked_loss_sats += abs(int(row.get("net_profit_sats") or 0))

    print("=== sweep_profitability: corpus sweep results ===")
    print(f"snapshots with revenue-profitability.json: {files}")
    for c in (c_max, c_sum, c_prof, c_under, c_nonneg):
        c.report()
    print()
    print("Not corpus-observable: PA-I1 PA-I2 PA-I3 PA-I4 PA-I5 PA-I6 PA-I7 PA-I8 "
          "PA-I10 PA-I11(exact) PA-I12")
    print()
    print("--- anomaly / decision-support material ---")
    print(f"class row totals across corpus: {dict(class_totals)}")
    print(f"structural-protection upgrades observed (break_even rows with roi < -10%): {len(upgrades)}")
    if upgrades:
        print(f"  distinct channels: {len(upgrade_channels)}; snapshot-weighted masked |net_profit| sum: {masked_loss_sats} sats")
        for p, cid, roi, net in upgrades[:5]:
            print(f"    {cid} roi={roi}% net={net} sats  <- {p}")
    print(f"profitable rows inside the widening band (5% < roi <= 10%): {band_profitable}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
