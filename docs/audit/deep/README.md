# Deep-Audit Artifacts (`docs/audit/deep/`)

This directory holds the coverage-accounting infrastructure for the deep audit
of `cl_revenue_ops` (the campaign described in the operator's plan). Unlike the
June–July 2026 *conformance* campaign, this is an **adversarial, accountable,
line-by-line** re-audit in which every prior audit is treated as untrusted and
every source line must be provably examined.

## Artifact layout

| File | Purpose |
| --- | --- |
| `coverage-manifest.md` | Generated. Every tracked SOURCE line divided into blob-pinned chunks with tier / owner / status columns. The coverage ledger's spine. |
| `findings-ledger.md` | One row per finding (P6). `file:line@blob` pins each finding to a chunk. |
| `attestations.md` | Structured "clean" attestations (P1). The only non-finding way a chunk becomes COVERED. |
| `README.md` | This file. |

Later phases add `deferred-ledger.md`, `concurrency-map.md`, `perf-baseline.md`,
`final-report.md`, etc. — see the plan.

## The coverage gate: `tools/audit/deep_manifest.py`

Read-only, deterministic, **stdlib + `git` subprocess only**. Three modes:

### 1. `generate` (default) — build the manifest

```
python3 tools/audit/deep_manifest.py
```

* Enumerates tracked SOURCE files via `git ls-files`, filtered to
  `cl-revenue-ops.py`, `modules/*.py`, `tools/**/*.py`, `scripts/**/*`.
  Excludes `tests/`, `docs/`, `config/`, `fixtures/`, `.worktrees/`, `.venv/`,
  `__pycache__/`, and `*.md`.
* Divides each file into chunks: **Tier 1 ≈ 400 lines/chunk**, **Tier 2/3 ≈ 700
  lines/chunk**.
* Emits per chunk: `chunk_id`, `file`, `line_start`, `line_end`, `tier`, git
  **blob hash** (the object id at HEAD, i.e. `git rev-parse HEAD:<path>`),
  `owner` (blank), `status` (`UNASSIGNED`).
* Writes `coverage-manifest.md`. Re-running on an unchanged tree is
  byte-identical (deterministic ordering, pure chunking function).

### 2. `--check` — drift detection (Phase 8)

```
python3 tools/audit/deep_manifest.py --check
```

Recomputes the current blob hash of every file in the manifest and flags any
chunk whose file blob changed since the manifest was written. Those chunks need
**re-attestation** (a mid-campaign fix may have invalidated an earlier clean
read). Exit 0 = clean, 1 = drift, 2 = manifest missing.

### 3. `--coverage` — report % COVERED

```
python3 tools/audit/deep_manifest.py --coverage
```

Reads `findings-ledger.md` + `attestations.md` and reports the percentage of
chunks COVERED, with a per-tier breakdown. A chunk is **COVERED** iff:

* **(a)** a findings-ledger row cites a `file:line@blob` inside it, **or**
* **(b)** an attestation block names its `chunk_id`.

`EXAMPLE`-marked rows/blocks are ignored, so the shipped templates report an
honest 0% (Tier 3's `EXAMPLE` attestation does not count). Exit 0 = 100%
covered, 1 = gaps remain, 2 = manifest missing.

## Tier assignment

Tier drives chunk size and read depth (Tier 1 is double-read per the plan).

* **Tier 1** (400-line chunks, ≈34.5k lines): `cl-revenue-ops.py`,
  `fee_controller.py`, `database.py`, `boltz_manager.py`,
  `rebalance_engine_v2.py`, `rebalancer.py`, `rebalance_native_executor_v2.py`,
  `rebalance_execution.py`, `rebalance_executor_v2.py`, `policy_manager.py`,
  `capex_budget.py`.
* **Tier 2** (700-line chunks, ≈16.3k lines): `capacity_planner`,
  `profitability_analyzer`, `hive_hints`, `flow_analysis`, `config`,
  `hive_router`, `rebalance_router_v3`, `rebalance_hive_router`,
  `rebalance_coordination_overlay`, `rebalance_router_v2`,
  `rebalance_planner_v2`, `rebalance_state_v2`, `rebalance_route_policy`.
* **Tier 3** (700-line chunks): everything else — the small remaining modules
  (`data_service`, `demand_flow`, `rebalance_audit_v2`, `capital_efficiency`,
  `segment_observations`, `utils`, `rebalance_types_v2`, `hive_runtime`,
  `__init__`) plus all of `tools/` and `scripts/`.

## Workflow

1. Regenerate the manifest at campaign start (already committed).
2. As auditors work, they either file findings (with `file:line@blob`) or write
   structured attestations for chunks they read clean.
3. `--coverage` gates progress toward 100%.
4. After every mid-campaign fix merge, `--check` reports which chunks drifted;
   those get re-attested.
5. At final HEAD (Phase 8), closure refuters re-run `--check` and `--coverage`
   to prove 100% coverage with no stale attestations.
