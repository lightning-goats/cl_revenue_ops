# Full Plugin Audit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce a defensible, severity-ranked audit of `cl-revenue-ops` and `cl-hive` using current MCP read-only evidence plus code review, with explicit findings for bugs, logic flaws, safety issues, correctness drift, and test gaps.

**Architecture:** Use code as the authority for what should exist and MCP as the authority for what is live on `hive-nexus-01`. Separate read-only live verification from static review of mutating paths. Record each finding with a violated invariant, evidence source, code reference, runtime evidence when available, and existing test coverage.

**Tech Stack:** Python 3.10+, shell inventory scripts, MCP `hive` / `revenue` tools, Markdown reporting, `pytest` for focused verification when needed.

---

## Task 1: Build The RPC Inventory

**Files:**
- Create: `docs/audits/2026-03-27-plugin-rpc-matrix.md`
- Modify: `docs/plans/2026-03-27-full-plugin-audit.md`

**Step 1: Extract the code-defined RPC lists**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
import re
pat = re.compile(r'@plugin\.(?:async_method|method)\(\s*["\']([^"\']+)["\']')
for path in (
    Path('/home/sat/bin/cl_revenue_ops/cl-revenue-ops.py'),
    Path('/home/sat/bin/cl-hive/cl-hive.py'),
):
    methods = pat.findall(path.read_text())
    print(path.name, len(methods))
    for method in methods:
        print(method)
PY
```

Expected: one complete list for each plugin.

**Step 2: Map each RPC to handler, mutability, and probable subsystem**

Use the decorator inventory plus direct code reads to build a table:

- RPC name
- Handler name
- Read-only or mutating
- Subsystem
- Existing direct test file(s)

**Step 3: Save the initial matrix**

Write the initial inventory into `docs/audits/2026-03-27-plugin-rpc-matrix.md`.

**Step 4: Sanity-check handler coverage**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
for repo in ('/home/sat/bin/cl_revenue_ops/tests', '/home/sat/bin/cl-hive/tests'):
    p = Path(repo)
    files = sorted(str(x.relative_to(p.parent)) for x in p.rglob('test_*.py'))
    print(repo, len(files))
    for f in files:
        print(f)
PY
```

Expected: a direct list of available regression files to cross-reference against the matrix.

---

## Task 2: Build The Live Drift Matrix Through MCP

**Files:**
- Modify: `docs/audits/2026-03-27-plugin-rpc-matrix.md`

**Step 1: Group the live surfaces**

Audit these read-only groups via MCP:

- `hive-health`, `hive-status`, `hive-members`
- `revenue-status`, `revenue-fee-debug`, `revenue-rebalance-debug`
- `revenue-dashboard`, `revenue-spend-ledger`, `revenue-boltz-wallet`
- topology / positioning / rationalization summary surfaces
- planner and recommendation surfaces

**Step 2: Record live outcomes**

For each group, record:

- success / timeout / unavailable
- key returned schema
- mismatch from code/doc expectations
- whether the surface appears stale

**Step 3: Add contract-drift notes**

For every drift item, note whether the problem is:

- runtime timeout
- stale data
- removed dependency
- schema mismatch
- likely logic bug

**Step 4: Stop short of mutation**

Do not use live mutating RPCs. If a mutating path needs review, move it to static analysis in later tasks.

---

## Task 3: Audit `cl-revenue-ops` High-Risk Subsystems

**Files:**
- Create: `docs/audits/2026-03-27-full-plugin-audit-report.md`
- Modify: `docs/audits/2026-03-27-plugin-rpc-matrix.md`

**Step 1: Audit fee decisioning**

Read the fee controller, fee execution, and related debug/status paths.

Check:

- stale/sparse data handling
- clamps and bounds
- fail-open vs fail-closed behavior
- explainability matching actual execution

**Step 2: Audit rebalance decisioning**

Read candidate selection, EV filters, execution gating, and recent-rebalance/status surfaces.

Check:

- budget and hot-channel protections
- false-positive EV assumptions
- stale profitability dependence
- success classification vs actual liquidity movement

**Step 3: Audit unified budget and spend accounting**

Read total-cost budget, spend ledger, reservation, release, and settlement logic.

Check:

- double counting
- stale reservation leakage
- concurrent reservation races
- fail-closed behavior on DB errors

**Step 4: Audit Boltz flows**

Read the wallet, quote, recommendation, and cycle logic.

Check:

- reserve-target correctness
- pending-swap gating
- dry-run vs execute separation
- profitability guard correctness

**Step 5: Audit planner and policy writes**

Read planner open/close logic, recommendation-only behavior, and policy mutation paths.

Check:

- approval gates
- recommendation vs live execution guarantees
- unsafe defaults
- invalid assumptions about node state or fee environment

---

## Task 4: Audit `cl-hive` High-Risk Subsystems

**Files:**
- Modify: `docs/audits/2026-03-27-full-plugin-audit-report.md`
- Modify: `docs/audits/2026-03-27-plugin-rpc-matrix.md`

**Step 1: Audit membership and governance**

Read join/leave/member-state/governance handlers and supporting modules.

Check:

- invariant preservation
- stale membership view hazards
- unauthorized transition risk
- cleanup/idempotency behavior

**Step 2: Audit planner and recommendation logic**

Read topology, expansion, connectivity, rationalization, and recommendation logic.

Check:

- single-node vs fleet assumptions
- stale-data dependence
- recommendation safety
- invalid or contradictory recommendation generation

**Step 3: Audit fee coordination and routing intelligence**

Read corridor, coordination, routing-intelligence, and learned-signal paths.

Check:

- ownership drift
- stale coordination input
- hidden fail-open behavior
- explainability/reporting mismatch

**Step 4: Audit gossip and state synchronization**

Read gossip, bridge, and state-version logic.

Check:

- stale version handling
- re-sync behavior
- inconsistent member state risk
- partial failure cleanup

---

## Task 5: Run Cross-Cutting Safety Review

**Files:**
- Modify: `docs/audits/2026-03-27-full-plugin-audit-report.md`

**Step 1: Review failure modes**

Across both plugins, identify where exceptions, timeouts, or missing data:

- fail open
- silently degrade behavior
- silently change recommendation quality
- produce stale but plausible status outputs

**Step 2: Review concurrency and persistence**

Identify:

- shared mutable state without clear guards
- DB transaction boundaries
- stale cache invalidation issues
- background-loop races vs debug/manual RPC calls

**Step 3: Review dry-run and recommendation promises**

Verify that any method claiming to be:

- dry-run
- preview-only
- recommendation-only
- diagnostic-only

does not mutate state or trigger side effects unexpectedly.

---

## Task 6: Verify Suspicions With Focused Checks

**Files:**
- Modify: `docs/audits/2026-03-27-full-plugin-audit-report.md`

**Step 1: For each high-confidence issue, choose one verification path**

Allowed verification paths:

- focused code trace
- existing regression test
- narrow new repro test in the local repo
- read-only live MCP confirmation

**Step 2: Do not broaden the scope**

Only verify issues that materially change severity or remediation priority.

**Step 3: Mark residual uncertainty**

If an issue cannot be live-confirmed safely, mark it `code-only` and explain why.

---

## Task 7: Write The Final Audit Report

**Files:**
- Modify: `docs/audits/2026-03-27-full-plugin-audit-report.md`
- Modify: `docs/audits/2026-03-27-plugin-rpc-matrix.md`

**Step 1: Order findings by severity**

Use this order:

1. Critical safety or correctness bugs
2. High-severity logic flaws
3. Medium-severity drift and stale-data hazards
4. Low-severity issues and simplifications
5. Missing tests

**Step 2: Use a consistent finding template**

For each finding, include:

- title
- severity
- plugin / subsystem
- code reference
- evidence type
- description
- why it matters
- recommended fix
- missing test, if applicable

**Step 3: Add a short executive summary**

Include:

- top 5 risks
- top 5 fastest risk-reduction fixes
- confidence limits caused by unavailable live surfaces

---

## Task 8: Verification Before Completion

**Files:**
- Modify: `docs/audits/2026-03-27-full-plugin-audit-report.md`
- Modify: `docs/audits/2026-03-27-plugin-rpc-matrix.md`

**Step 1: Verify artifacts exist**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
for path in (
    Path('/home/sat/bin/cl_revenue_ops/docs/audits/2026-03-27-full-plugin-audit-report.md'),
    Path('/home/sat/bin/cl_revenue_ops/docs/audits/2026-03-27-plugin-rpc-matrix.md'),
):
    print(path, path.exists(), path.stat().st_size if path.exists() else 0)
PY
```

Expected: both files exist and are non-empty.

**Step 2: Verify any focused tests run cleanly**

Run only the focused `pytest` commands introduced during Task 6.

Expected: pass with no unresolved failures.

**Step 3: Verify the report matches the actual evidence**

Re-read:

- the live MCP outputs cited in the report
- the exact code references
- the severity ordering

Expected: every claim is backed by either live evidence, code evidence, or both.
