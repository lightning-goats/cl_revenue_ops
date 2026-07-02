#!/usr/bin/env python3
"""Per-module verification scorecard (Phase 5, re-scoped standalone per D3).

Re-executes the verification campaign's read-only invariant sweeps
(tools/audit/sweep_*.py, loop_sweep_*.py, check_hermes_forwards_chain.py)
against a corpus root and emits one PASS / KNOWN / WARN / INCONCLUSIVE /
ERROR verdict per module (plus the three decision loops and the corpus
forwards chain).

Statuses
  PASS          every mapped check clean, and at least one check was
                non-vacuous (actually exercised by corpus data).
  KNOWN         only allowlisted (documented, operator-accepted) findings
                fired -- see ALLOWLIST below for the doc references.
  WARN          at least one NON-allowlisted violation -- new signal;
                the violating check names are listed.
  INCONCLUSIVE  every mapped check was vacuous (zero checkable instances
                in the window) or the module has no corpus-observable
                checks at all. Data starvation, NOT a health verdict.
  ERROR         the sweep crashed or its output could not be parsed;
                never silently absent.

Telemetry caveat (docs/audit/decision-loops.md, defect 5): checks that read
the `recent_fee_changes` RPC echo (a rolling 10-row window that lost 61% of
nexus-01's change records in the frozen corpus) are labeled `[lossy-echo]`;
their pass counts are lower bounds, and a future collector should capture
the fee_changes DB table instead.

Usage:
  python3 tools/audit/scorecard.py [--root DIR] [--since YYYYMMDD]
                                   [--json FILE] [--keep-going]

  --root    corpus root (default /home/sat/cl-mycelium-hermes); propagated
            to every sweep via CL_MYCELIUM_HERMES_ROOT.
  --since   restrict to snapshot days >= YYYYMMDD (incremental scorecard).
            Implemented as a symlink shadow root, so the sweeps themselves
            are unchanged.
  --json    also write the full machine-readable scorecard to FILE.

Exit code: 0 all PASS/KNOWN/INCONCLUSIVE, 1 any WARN, 2 any ERROR.

Read-only over the corpus; writes nothing outside --json and a temp dir.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_ROOT = "/home/sat/cl-mycelium-hermes"
NODES = ("hive-nexus-01", "hive-nexus-02")
SWEEP_TIMEOUT = 1800

# ML-* metabolism-ledger checks read cl-hive's hive-organism-status.json, not
# any cl_revenue_ops module. They are reported under this cl-hive-owned
# pseudo-module so they never launder cl_revenue_ops's capital_efficiency
# verdict (P5-001). The trailing "(cl-hive)" makes the ownership explicit in
# the rendered scorecard.
METABOLISM_LEDGER_MODULE = "metabolism_ledger (cl-hive)"

# ---------------------------------------------------------------------------
# Check model
# ---------------------------------------------------------------------------


@dataclass
class Check:
    sweep: str
    check_id: str          # short invariant id, e.g. "FC-I1a", "LP-I6"
    name: str              # human line the sweep printed
    checked: int
    violations: int
    vacuous: bool = False
    vacuous_note: str = ""
    examples: list = field(default_factory=list)

    @property
    def effectively_vacuous(self) -> bool:
        return self.vacuous or self.checked == 0


# ---------------------------------------------------------------------------
# Allowlist: documented, operator-accepted findings. A check matching an
# entry reports KNOWN instead of WARN when it fires. Extend by appending;
# every entry MUST carry a doc reference.
# ---------------------------------------------------------------------------

ALLOWLIST = [
    {
        "sweep": "sweep_data_budget",
        "match": "ML-",
        "reason": "metabolism-ledger identical-windows / unknown-coverage "
                  "anomalies; defect owned by cl-hive (runtime.py:2702), "
                  "not cl_revenue_ops",
        "doc": "docs/audit/verification/capital_efficiency.md; "
               "docs/audit/phase2-summary.md (2247370 note)",
    },
    {
        "sweep": "sweep_data_budget",
        "match": "DB7-NO-COVERED-HOURS",
        "reason": "corpus-era assertion: commit 2247370 added covered_hours, "
                  "so post-drift snapshots (2026-07-01+) legitimately carry "
                  "the field; the hardcoded coverage_status='complete' is a "
                  "known drift finding on the follow-up list",
        "doc": "docs/audit/phase2-summary.md (contract drift, 2247370)",
    },
    {
        "sweep": "sweep_data_budget",
        "match": "TCB-NO-COVERED-HOURS",
        "reason": "same 2247370 covered_hours drift as DB7-NO-COVERED-HOURS",
        "doc": "docs/audit/phase2-summary.md (contract drift, 2247370)",
    },
    {
        "sweep": "loop_sweep_fee",
        "match": "LF-7",
        "reason": "single-channel unmanaged-desync episode (946890x2272x0, "
                  "06-15..07-01): external fee writer + silent management "
                  "opt-out; fee-debug staleness defect is on the follow-up "
                  "fix list",
        "doc": "docs/audit/decision-loops/fee-loop.md (E2b); "
               "docs/audit/decision-loops.md (defect 3)",
    },
    {
        "sweep": "loop_sweep_rebalance",
        "match": "LP-I6",
        "reason": "one pre-coverage observation (obs-1776182577-1, 04-14) "
                  "whose row absence is proven by id contiguity; hand-audited",
        "doc": "docs/audit/decision-loops/rebalance-loop.md (LP-I6)",
    },
    {
        "sweep": "loop_sweep_planner",
        "match": "L2:",
        "reason": "known delegation counter-cases: n1 id 947 (60->110->210 "
                  "up-ratchet, known pending-fix failure mode) and n2 ids "
                  "293/294 (member channels, static above floor)",
        "doc": "docs/audit/decision-loops/planner-boltz-loops.md "
               "(:119-124, :296)",
    },
    {
        "sweep": "loop_sweep_planner",
        "match": "L3/inv-a",
        "reason": "defibrillation placebo / status lie (ids 933/937/975): "
                  "blocked or hole-hidden shocks recorded as 'completed' "
                  "with no diagnostic row; follow-up fix list defect 1",
        "doc": "docs/audit/decision-loops.md (defect 1); "
               "docs/audit/decision-loops/planner-boltz-loops.md (:161-197)",
    },
    {
        "sweep": "sweep_planner_boltz_hints",
        "match": "D1 flag:",
        "reason": "hive-member DEFIBRILLATE/FEE_REDUCE episodes are "
                  "permitted-as-implemented pending the D1 follow-up "
                  "(remove member channels from the dead-capital pipeline)",
        "doc": "docs/audit/operator-decisions.md (D1)",
    },
    {
        "sweep": "sweep_planner_boltz_hints",
        "match": "ledger id continuity",
        "reason": "reconstruction gaps are a corpus-visibility artifact of "
                  "the bounded (7-day) planner-history RPC window, not "
                  "plugin behavior",
        "doc": "docs/audit/decision-loops.md (bounded-echo caveat); "
               "docs/audit/phase2-summary.md (evidence-quality lessons)",
    },
]


def allowlisted(sweep: str, check: Check):
    for entry in ALLOWLIST:
        if entry["sweep"] != sweep:
            continue
        m = entry["match"]
        if check.check_id.startswith(m) or check.name.startswith(m):
            return entry
    return None


# ---------------------------------------------------------------------------
# Telemetry caveats: checks fed by the lossy recent_fee_changes RPC echo.
# ---------------------------------------------------------------------------

LOSSY_ECHO_CHECKS = {
    ("sweep_fee_stack", "FC-I1a"), ("sweep_fee_stack", "FC-I9"),
    ("sweep_fee_stack", "FC-I10"),
    ("loop_sweep_fee", "LF-1"), ("loop_sweep_fee", "LF-2"),
    ("loop_sweep_fee", "LF-2b"), ("loop_sweep_fee", "LF-5"),
    ("loop_sweep_fee", "LF-6"), ("loop_sweep_fee", "LF-8"),
}
LOSSY_ECHO_NAME_PREFIXES = {("loop_sweep_planner", "L2: delegated channels")}
LOSSY_ECHO_NOTE = ("[lossy-echo] fed by the 10-row recent_fee_changes RPC "
                   "window (61% record loss observed); counts are lower "
                   "bounds -- read the fee_changes DB table for real audits")


def lossy_echo(sweep: str, check: Check) -> bool:
    if (sweep, check.check_id) in LOSSY_ECHO_CHECKS:
        return True
    return any(s == sweep and check.name.startswith(p)
               for s, p in LOSSY_ECHO_NAME_PREFIXES)


# Checks that pass over an all-zero activity surface: force-vacuous so the
# owning module reports INCONCLUSIVE rather than a hollow PASS.
VACUOUS_OVERRIDES = {
    ("sweep_planner_boltz_hints", "BM-H2"):
        "boltz dormant corpus-wide (Phase 3 L5 dormancy proof): "
        "spend<=budget checked against an all-zero surface",
    ("sweep_planner_boltz_hints", "boltz swap activity present in corpus"):
        "informational inventory only",
}


def vacuous_override(sweep: str, check: Check):
    for (s, prefix), reason in VACUOUS_OVERRIDES.items():
        if s == sweep and (check.check_id.startswith(prefix)
                           or check.name.startswith(prefix)):
            return reason
    return None


# ---------------------------------------------------------------------------
# Check -> module mapping
# ---------------------------------------------------------------------------

PREFIX_MODULES = [
    # (sweep filter or None, check-id prefix, module)
    (None, "FC-", "fee_controller"),
    (None, "FA-", "flow_analysis"),
    (None, "PM-", "policy_manager"),
    (None, "RB-", "rebalancer"),
    (None, "RE-", "rebalance_engine_v2"),
    (None, "PA-", "profitability_analyzer"),
    (None, "CP-", "capacity_planner"),
    (None, "BM-", "boltz_manager"),
    (None, "HH-", "hive_hints"),
    (None, "HH ", "hive_hints"),
    (None, "SL-", "database"),
    (None, "TCB-", "database"),
    (None, "DB", "database"),
    (None, "CB", "capex_budget"),
    # ML-* checks read cl-hive's hive-organism-status.json (metabolism ledger)
    # and are allowlisted as cl-hive-owned. They exercise NONE of
    # capital_efficiency's own code, so attributing them to capital_efficiency
    # would launder that module's verdict into a PASS/KNOWN it never earned.
    # Route them to a distinct cl-hive-owned pseudo-module instead; see P5-001.
    (None, "ML-", METABOLISM_LEDGER_MODULE),
    (None, "SO", "segment_observations"),
    (None, "DF-", "demand_flow"),
    (None, "LF-", "fee_loop"),
    (None, "LP-", "rebalance_loop"),
]

# modules that exist in the campaign but have no corpus-observable checks
NO_SWEEP_MODULES = {
    "policy_manager": "no corpus-observable PM-* checks (PM invariants are "
                      "code/test verdicts; open PM-I1/PM-I2/PM-I13 findings "
                      "are code defects, see docs/audit/phase2-summary.md)",
    "demand_flow": "no corpus-observable DF-* checks (classify path is "
                   "production-dead; Phase 2 code/test verdicts only)",
    "capital_efficiency": "no corpus-observable CE-* checks of its own; the "
                          "ML-* metabolism-ledger anomalies read cl-hive's "
                          "hive-organism-status.json and are reported under "
                          "'metabolism_ledger (cl-hive)', not here (P5-001). "
                          "A health verdict here would exercise none of this "
                          "module's code.",
}

MODULE_ORDER = [
    "fee_controller", "flow_analysis", "policy_manager",
    "rebalancer", "rebalance_engine_v2", "profitability_analyzer",
    "capacity_planner", "boltz_manager", "hive_hints",
    "database", "capex_budget", "capital_efficiency",
    METABOLISM_LEDGER_MODULE,
    "segment_observations", "demand_flow",
    "routing_stack",
    "fee_loop", "rebalance_loop", "planner_loop",
    "forwards_chain",
]


def module_for(sweep: str, check: Check) -> str:
    if sweep == "sweep_routing_stack":
        return "routing_stack"          # RX/R2/R3/HR/RHR/RCO/NX/RP2 surface
    if sweep == "loop_sweep_planner":
        return "planner_loop"
    if sweep == "check_hermes_forwards_chain":
        return "forwards_chain"
    if sweep == "sweep_planner_boltz_hints":
        n = check.name
        if n.startswith("BM-") or n.startswith("boltz"):
            return "boltz_manager"
        if n.startswith("HH"):
            return "hive_hints"
        return "capacity_planner"       # CP-*, D1 flag, pool, id continuity
    for sw, prefix, module in PREFIX_MODULES:
        if sw not in (None, sweep):
            continue
        if check.check_id.startswith(prefix):
            return module
    return f"UNMAPPED::{sweep}"


# ---------------------------------------------------------------------------
# Output parsers (per sweep). Each returns list[Check]; raising or returning
# [] marks the sweep's modules ERROR.
# ---------------------------------------------------------------------------

ID_RE = re.compile(r"^([A-Z]{2,6}-[A-Za-z0-9]+)\b")


def _check_id_of(name: str) -> str:
    m = ID_RE.match(name)
    if m:
        return m.group(1)
    head = name.split(":", 1)[0].strip()
    if 0 < len(head) <= 24:
        return head          # e.g. "L2", "L3/inv-a", "D1 flag", "inv-c"
    return name[:48]


def parse_sectioned(sweep: str, out: str) -> list[Check]:
    """sweep_fee_stack / loop_sweep_fee: '-- pass counts --' etc. sections."""
    section = None
    passes: dict[str, int] = defaultdict(int)
    viols: dict[str, int] = defaultdict(int)
    names: dict[str, set] = defaultdict(set)
    examples: dict[str, list] = defaultdict(list)
    last_viol_id = None
    info: dict[str, int] = {}
    residue = {}
    for line in out.splitlines():
        if line.startswith("-- pass counts"):
            section = "pass"
            continue
        if line.startswith("-- violations"):
            section = "viol"
            continue
        if line.startswith("-- informational"):
            section = "info"
            continue
        if line.startswith("-- FA-I12"):
            section = "residue"
            continue
        m = re.match(r"^  (\S.*): (\d+)$", line)
        if section == "pass" and m:
            cid = _check_id_of(m.group(1))
            passes[cid] += int(m.group(2))
            names[cid].add(m.group(1))
        elif section == "viol" and m:
            cid = _check_id_of(m.group(1))
            viols[cid] += int(m.group(2))
            names[cid].add(m.group(1))
            last_viol_id = cid
        elif section == "viol" and line.strip().startswith("e.g.") and last_viol_id:
            if len(examples[last_viol_id]) < 3:
                examples[last_viol_id].append(line.strip()[5:])
        elif section == "info" and m:
            try:
                info[m.group(1)] = int(m.group(2))
            except ValueError:
                pass
        elif section == "residue":
            rm = re.match(r"^  (transient|persistent) .*: (\d+) channel", line)
            if rm:
                residue[rm.group(1)] = int(rm.group(2))
    checks = []
    for cid in sorted(set(passes) | set(viols)):
        checks.append(Check(sweep, cid, "; ".join(sorted(names[cid])),
                            passes[cid] + viols[cid], viols[cid],
                            examples=examples[cid]))
    if sweep == "sweep_fee_stack" and residue:
        checks.append(Check(
            sweep, "FA-I12",
            "channel_states residue: persistent (>1 snapshot) channels",
            info.get("channel_state_rows", sum(residue.values())),
            residue.get("persistent", 0)))
    return checks


def parse_check_report(sweep: str, out: str) -> list[Check]:
    """sweep_rebalancer / sweep_profitability / loop_sweep_rebalance:
    '[NAME] PASS  checked=N violations=V ...' lines."""
    checks = []
    for line in out.splitlines():
        m = re.match(r"^\[([A-Za-z0-9/_-]+)\] (PASS|VIOLATION)\s+"
                     r"checked=(\d+) violations=(\d+)(.*)$", line)
        if not m:
            continue
        trailer = m.group(5)
        desc = ""
        dm = re.search(r"--\s*(.*)$", trailer)
        if dm:
            desc = dm.group(1)
        vac = "VACUOUS" in trailer
        note = ""
        nm = re.search(r"\[([^]]*VACUOUS[^]]*)\]", trailer)
        if nm:
            note = nm.group(1)
        checks.append(Check(sweep, m.group(1), desc or m.group(1),
                            int(m.group(3)), int(m.group(4)),
                            vacuous=vac and int(m.group(3)) == 0,
                            vacuous_note=note))
    return checks


def parse_data_budget(sweep: str, out: str) -> list[Check]:
    checks = []
    cur = None
    for line in out.splitlines():
        m = re.match(r"^(OK|FAIL)\s+(.+?)\s+pass=(\d+)\s+fail=(\d+)\s*$", line)
        if m:
            name = m.group(2).rstrip()
            cur = Check(sweep, name.split()[0], name,
                        int(m.group(3)) + int(m.group(4)), int(m.group(4)))
            checks.append(cur)
        elif cur and line.strip().startswith("e.g.") and len(cur.examples) < 3:
            cur.examples.append(line.strip()[5:])
    return checks


def parse_node_results(sweep: str, out: str) -> list[Check]:
    """sweep_planner_boltz_hints / loop_sweep_planner:
    '[STATUS] node :: check :: checked=N violations|flagged=V [:: note]'.
    Aggregated across nodes by check name."""
    agg: dict[str, Check] = {}
    cur = None
    for line in out.splitlines():
        m = re.match(r"^\[(PASS|VIOLATION|FLAG|VACUOUS)\] (\S+) :: (.+?) :: "
                     r"checked=(\d+) (?:violations|flagged)=(\d+)"
                     r"(?: :: (.*))?$", line)
        if m:
            name = m.group(3)
            c = agg.get(name)
            if c is None:
                c = agg[name] = Check(sweep, _check_id_of(name), name, 0, 0)
            c.checked += int(m.group(4))
            c.violations += int(m.group(5))
            if m.group(6) and m.group(6) not in c.vacuous_note:
                c.vacuous_note = (c.vacuous_note + " | " + m.group(6)).strip(" |")
            cur = c
        elif cur and line.strip().startswith("e.g.") and len(cur.examples) < 3:
            cur.examples.append(line.strip()[5:])
    return list(agg.values())


ROUTING_GROUPS = [
    ("C", "candidates", "route/candidate checks C1-C9 (R2/R3/RX/RHR/RCO)"),
    ("S", "rebalance_history_entries", "history checks S1-S3 (engine/RCO/R3 surface)"),
    ("O", "segment_observations", "segment-observation checks O0-O2 (R3/RHR/NX surface)"),
    ("L", None, "spend-ledger checks L1"),
]


def parse_routing_json(sweep: str, out: str) -> list[Check]:
    data = json.loads(out)
    counts = data.get("counts") or {}
    files = data.get("files_scanned") or {}
    violations = data.get("violations") or {}
    checks = []
    for prefix, count_key, desc in ROUTING_GROUPS:
        vio = 0
        ex = []
        for key, info in violations.items():
            if key.startswith(prefix):
                vio += int(info.get("count") or 0)
                ex.extend(info.get("examples") or [])
        if count_key:
            checked = int(counts.get(count_key) or 0)
        else:
            checked = int(files.get("revenue-spend-ledger.json") or 0)
        checks.append(Check(sweep, prefix + "*", desc, max(checked, vio),
                            vio, examples=ex[:3]))
    return checks


def parse_forwards_chain(sweep: str, out: str) -> list[Check]:
    checks = []
    node = None
    for line in out.splitlines():
        m = re.match(r"^== (\S+): (CONTIGUOUS|BROKEN) ==$", line)
        if m:
            node = m.group(1)
            continue
        m = re.match(r"^\s+windows=(\d+) unreadable=(\d+) gaps=(\d+) "
                     r"overlaps=(\d+)", line)
        if m and node:
            checks.append(Check(
                sweep, "CHAIN", f"{node} listforwards chain contiguity "
                f"(overlaps={m.group(4)} are retries, not losses)",
                int(m.group(1)), int(m.group(2)) + int(m.group(3))))
            node = None
    return checks


# ---------------------------------------------------------------------------
# Sweep registry
# ---------------------------------------------------------------------------

SWEEPS = [
    ("sweep_fee_stack", [], parse_sectioned),
    ("sweep_rebalancer", [], parse_check_report),
    ("sweep_profitability", [], parse_check_report),
    ("sweep_planner_boltz_hints", [], parse_node_results),
    ("sweep_data_budget", [], parse_data_budget),
    ("sweep_routing_stack", ["--json"], parse_routing_json),
    ("loop_sweep_fee", [], parse_sectioned),
    ("loop_sweep_rebalance", [], parse_check_report),
    ("loop_sweep_planner", [], parse_node_results),
    ("check_hermes_forwards_chain", [], parse_forwards_chain),
]

# Marker proving the sweep ran to completion. If a sweep yields zero parsed
# checks but its marker is present, the window simply had no data for it
# (modules go INCONCLUSIVE); a missing marker means broken output => ERROR.
SWEEP_RAN_MARKERS = {
    "sweep_fee_stack": "SWEEP RESULTS",
    "sweep_rebalancer": "corpus sweep results",
    "sweep_profitability": "corpus sweep results",
    "sweep_planner_boltz_hints": "SWEEP RESULTS",
    "sweep_data_budget": "snapshots swept:",
    "sweep_routing_stack": '"files_scanned"',
    "loop_sweep_fee": "PHASE 3 LOOP SWEEP",
    "loop_sweep_rebalance": "REBALANCE + BUDGET LOOP",
    "loop_sweep_planner": "SWEEP RESULTS",
    "check_hermes_forwards_chain": "forwards:",
}

_TRACEBACK_MARKER = "Traceback (most recent call last):"


def sweep_completed_cleanly(name: str, rc: int, out: str, err: str):
    """Distinguish a CLEAN sweep completion from a CRASH (P5-007).

    Several sweeps exit 1 to signal "violations found", so exit code alone
    cannot tell a legitimate finding from an uncaught exception that also
    exits 1 and leaves partial output. A sweep is treated as clean ONLY when
      * it exited 0 or 1 (not a timeout / launch failure / other signal),
      * it emitted its explicit end-of-run marker (proof it reached the
        results section rather than dying mid-run), AND
      * neither stdout nor stderr contains a Python traceback.
    Otherwise it is a crash and its modules must go ERROR, never PASS/KNOWN.

    Returns (clean: bool, reason: str). reason is "" when clean.
    """
    if rc not in (0, 1):
        detail = err.strip()[-400:]
        if rc == -1:
            return False, f"sweep timed out (exit={rc}){f'; {detail}' if detail else ''}"
        return False, f"sweep did not exit cleanly (exit={rc}){f'; stderr: {detail}' if detail else ''}"
    if _TRACEBACK_MARKER in out or _TRACEBACK_MARKER in err:
        detail = (err if _TRACEBACK_MARKER in err else out).strip()[-400:]
        return False, (f"sweep crashed mid-run (exit={rc}, traceback emitted); "
                       f"tail: {detail}")
    if SWEEP_RAN_MARKERS[name] not in out:
        detail = err.strip()[-300:]
        return False, ("sweep did not reach its clean-completion marker "
                       f"({SWEEP_RAN_MARKERS[name]!r} absent; exit={rc}), "
                       "treating as a crash"
                       + (f"; stderr: {detail}" if detail else ""))
    return True, ""


SWEEP_MODULES = {
    "sweep_fee_stack": ["fee_controller", "flow_analysis", "policy_manager"],
    "sweep_rebalancer": ["rebalancer", "rebalance_engine_v2"],
    "sweep_profitability": ["profitability_analyzer"],
    "sweep_planner_boltz_hints": ["capacity_planner", "boltz_manager",
                                  "hive_hints"],
    "sweep_data_budget": ["database", "capex_budget", "capital_efficiency",
                          METABOLISM_LEDGER_MODULE,
                          "segment_observations", "demand_flow"],
    "sweep_routing_stack": ["routing_stack"],
    "loop_sweep_fee": ["fee_loop"],
    "loop_sweep_rebalance": ["rebalance_loop"],
    "loop_sweep_planner": ["planner_loop"],
    "check_hermes_forwards_chain": ["forwards_chain"],
}


def run_sweep(name: str, extra_args: list, env: dict):
    """Returns (name, returncode, stdout, stderr)."""
    cmd = [sys.executable, str(HERE / f"{name}.py"), *extra_args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=str(REPO), env=env, timeout=SWEEP_TIMEOUT)
        return name, proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return name, -1, "", f"timeout after {SWEEP_TIMEOUT}s"
    except Exception as exc:  # noqa: BLE001
        return name, -2, "", f"launch failed: {exc}"


# ---------------------------------------------------------------------------
# Corpus window
# ---------------------------------------------------------------------------

def corpus_window(root: Path):
    days = []
    snapshots = 0
    per_node = {}
    for node in NODES:
        base = root / node
        n_snaps = 0
        node_days = []
        if base.is_dir():
            for day in sorted(base.iterdir()):
                if not day.is_dir() or not re.match(r"^\d{8}$", day.name):
                    continue
                day_snaps = sum(1 for s in day.iterdir()
                                if s.is_dir() and (s / "commands").is_dir())
                if day_snaps:
                    node_days.append(day.name)
                    n_snaps += day_snaps
        per_node[node] = {"days": len(node_days), "snapshots": n_snaps,
                          "first_day": node_days[0] if node_days else None,
                          "last_day": node_days[-1] if node_days else None}
        days.extend(node_days)
        snapshots += n_snaps
    return {
        "root": str(root),
        "first_day": min(days) if days else None,
        "last_day": max(days) if days else None,
        "distinct_days": len(set(days)),
        "snapshot_count": snapshots,
        "per_node": per_node,
    }


def build_since_shadow(root: Path, since: str) -> Path:
    """Symlink shadow root containing only day dirs >= since."""
    shadow = Path(tempfile.mkdtemp(prefix="scorecard-since-"))
    for node in NODES:
        (shadow / node).mkdir()
        base = root / node
        if not base.is_dir():
            continue
        for day in sorted(base.iterdir()):
            if day.is_dir() and re.match(r"^\d{8}$", day.name) \
                    and day.name >= since:
                (shadow / node / day.name).symlink_to(day.resolve())
    return shadow


# ---------------------------------------------------------------------------
# Scorecard assembly
# ---------------------------------------------------------------------------

def build_scorecard(root: Path, since: str | None):
    effective_root = root
    shadow = None
    if since:
        shadow = build_since_shadow(root, since)
        effective_root = shadow

    env = os.environ.copy()
    env["CL_MYCELIUM_HERMES_ROOT"] = str(effective_root)
    env.setdefault("PYTHONHASHSEED", "0")

    window = corpus_window(effective_root)
    window["root"] = str(root)          # display the real root, not the shadow
    window["since_filter"] = since

    sweep_runs = {}
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(run_sweep, name, args, env)
                       for name, args, _ in SWEEPS]
            for fut in futures:
                name, rc, out, err = fut.result()
                sweep_runs[name] = (rc, out, err)
    finally:
        if shadow is not None:
            shutil.rmtree(shadow, ignore_errors=True)

    modules: dict[str, dict] = {
        m: {"checks": [], "errors": []} for m in MODULE_ORDER
    }

    for name, args, parser in SWEEPS:
        rc, out, err = sweep_runs[name]
        checks: list[Check] = []
        parse_err = None
        clean, reason = sweep_completed_cleanly(name, rc, out, err)
        if clean:
            try:
                checks = parser(name, out)
            except Exception as exc:  # noqa: BLE001
                parse_err = f"parser raised: {exc}"
        else:
            # Crash / timeout / truncated output: never parse partial output
            # into a health verdict. The owning modules go ERROR.
            parse_err = reason
        if parse_err:
            for m in SWEEP_MODULES[name]:
                modules.setdefault(m, {"checks": [], "errors": []})
                modules[m]["errors"].append(f"{name}: {parse_err}")
            continue
        for check in checks:
            ov = vacuous_override(name, check)
            if ov and check.violations == 0:
                check.vacuous = True
                check.vacuous_note = ov
            m = module_for(name, check)
            modules.setdefault(m, {"checks": [], "errors": []})
            modules[m]["checks"].append(check)

    # per-module verdicts
    rows = []
    for m in MODULE_ORDER + sorted(k for k in modules if k not in MODULE_ORDER):
        data = modules[m]
        checks = data["checks"]
        new_viol, known_viol, caveats, vac_notes = [], [], [], []
        non_vacuous = 0
        for c in checks:
            if not c.effectively_vacuous:
                non_vacuous += 1
            elif c.vacuous_note:
                vac_notes.append(f"{c.check_id}: {c.vacuous_note}")
            if lossy_echo(c.sweep, c):
                caveats.append(c.check_id)
            if c.violations > 0:
                entry = allowlisted(c.sweep, c)
                if entry:
                    known_viol.append((c, entry))
                else:
                    new_viol.append(c)
        if data["errors"]:
            status = "ERROR"
        elif new_viol:
            status = "WARN"
        elif known_viol:
            status = "KNOWN"
        elif non_vacuous:
            status = "PASS"
        else:
            status = "INCONCLUSIVE"
        note_bits = []
        if m in NO_SWEEP_MODULES and not checks:
            note_bits.append(NO_SWEEP_MODULES[m])
        def _label(c: Check) -> str:
            return c.check_id if c.check_id == c.name else \
                f"{c.check_id} [{c.name[:64]}]" if c.check_id not in c.name \
                else c.name[:72]

        note_bits += [f"NEW: {_label(c)} ({c.violations}x)"
                      for c in new_viol]
        note_bits += [f"KNOWN: {_label(c)} ({c.violations}x) -> {e['doc']}"
                      for c, e in known_viol]
        note_bits += [f"vacuous: {v}" for v in vac_notes]
        if caveats:
            note_bits.append("lossy-echo checks: " + ",".join(sorted(set(caveats))))
        note_bits += data["errors"]
        rows.append({
            "module": m,
            "status": status,
            "checks_total": len(checks),
            "checks_non_vacuous": non_vacuous,
            "violations_new": sum(c.violations for c in new_viol),
            "violations_known": sum(c.violations for c, _ in known_viol),
            "notes": note_bits,
            "checks": [{
                "sweep": c.sweep, "id": c.check_id, "name": c.name,
                "checked": c.checked, "violations": c.violations,
                "vacuous": c.effectively_vacuous,
                "vacuous_note": c.vacuous_note,
                "allowlisted": bool(allowlisted(c.sweep, c)) and c.violations > 0,
                "lossy_echo": lossy_echo(c.sweep, c),
                "examples": c.examples,
            } for c in checks],
        })

    return {
        "window": window,
        "sweep_exit_codes": {n: sweep_runs[n][0] for n in sweep_runs},
        "modules": rows,
        "allowlist": ALLOWLIST,
        "telemetry_caveat": LOSSY_ECHO_NOTE,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(card: dict) -> str:
    w = card["window"]
    lines = []
    lines.append("=" * 100)
    lines.append("cl_revenue_ops VERIFICATION SCORECARD "
                 "(campaign Phase 5, standalone per operator decision D3)")
    lines.append("=" * 100)
    lines.append(f"corpus root: {w['root']}"
                 + (f"  (since-filter {w['since_filter']})"
                    if w.get("since_filter") else ""))
    lines.append(f"corpus window: {w['first_day']} .. {w['last_day']}  "
                 f"({w['distinct_days']} distinct days, "
                 f"{w['snapshot_count']} snapshots"
                 + "".join(f"; {n}={p['snapshots']}"
                           for n, p in w["per_node"].items()) + ")")
    if not w["snapshot_count"]:
        lines.append("!! ZERO snapshots in window -- everything below is "
                     "INCONCLUSIVE by construction")
    lines.append(f"telemetry caveat: {card['telemetry_caveat']}")
    lines.append("")
    hdr = (f"{'MODULE':<26} {'STATUS':<13} {'CHECKS':>7} {'NONVAC':>6} "
           f"{'NEWVIOL':>7} {'KNOWN':>6}  NOTES")
    lines.append(hdr)
    lines.append("-" * 100)
    for row in card["modules"]:
        first = (f"{row['module']:<26} {row['status']:<13} "
                 f"{row['checks_total']:>7} {row['checks_non_vacuous']:>6} "
                 f"{row['violations_new']:>7} {row['violations_known']:>6}  ")
        notes = row["notes"] or [""]
        lines.append(first + notes[0])
        for n in notes[1:]:
            lines.append(" " * len(first.rstrip()) + "   " + n)
    lines.append("-" * 100)
    counts = defaultdict(int)
    for row in card["modules"]:
        counts[row["status"]] += 1
    lines.append("summary: " + "  ".join(f"{k}={v}" for k, v in
                                         sorted(counts.items())))
    lines.append("INCONCLUSIVE = data starvation in this window, not module "
                 "health; KNOWN = documented+accepted findings only "
                 "(see allowlist in tools/audit/scorecard.py).")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=DEFAULT_ROOT, help="corpus root")
    ap.add_argument("--since", metavar="YYYYMMDD",
                    help="only snapshot days >= this (incremental scorecard)")
    ap.add_argument("--json", metavar="FILE",
                    help="also write machine-readable scorecard JSON")
    args = ap.parse_args()

    if args.since and not re.match(r"^\d{8}$", args.since):
        ap.error("--since must be YYYYMMDD")
    root = Path(args.root)
    if not root.is_dir():
        print(f"corpus root not found: {root}", file=sys.stderr)
        return 2

    card = build_scorecard(root, args.since)
    print(render(card))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(card, f, indent=2)
        print(f"\njson scorecard written: {args.json}")

    statuses = {r["status"] for r in card["modules"]}
    if "ERROR" in statuses:
        return 2
    if "WARN" in statuses:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
