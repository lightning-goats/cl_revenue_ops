# Production Revenue Validation Automation Design

**Goal:** Automate daily production evidence collection and checkpoint drafting for the four-week validation window covering PRs #87, #88, and #89 on `lnnode` and `hive-nexus-02`.

**Status:** Approved design for implementation planning.

## Summary

This automation runs on the control host, not on either production node. It is intentionally read-only: collect evidence, evaluate rollout-watch conditions, and draft checkpoint reports, but never change node config, restart plugins, or trigger rollback actions.

The system is designed around one fixed daily run at `06:00 America/Denver`. Each run captures immutable raw snapshots, computes normalized trend metrics, evaluates red/yellow flags from the validation plan, and refreshes draft checkpoint artifacts for T+14 and T+28 once those windows are reached.

## Scope

Included:

- Daily raw evidence collection from `lnnode` and `hive-nexus-02`
- Trend tracking over the 28-day validation window
- Automated red-flag and yellow-flag checks
- Draft Markdown report generation for T+14 and T+28
- Persistent artifacts under `results/` and `docs/reports/`

Excluded:

- Any active experiments or parameter tuning
- Any automatic rollback or config changes
- Any restart or mutation of production nodes
- Any conclusion stronger than the underlying evidence supports

## Architecture

The automation is split into four repo-managed components plus one config file:

1. `tools/revenue_validation_collect.py`
   Collects raw evidence from both nodes and writes immutable daily snapshots.

2. `tools/revenue_validation_watch.py`
   Evaluates the rollback-watch checks from the validation plan using the latest collected evidence and recent log extracts.

3. `tools/revenue_validation_report.py`
   Generates or refreshes draft T+14 and T+28 Markdown reports from saved evidence only.

4. `tools/revenue_validation_daily.py`
   Orchestrates the daily pipeline in order: collect, watch, report.

5. `config/revenue_validation.yaml`
   Stores per-node transport commands, T0 timestamps, schedule assumptions, thresholds, and output locations.

The control host runs the daily pipeline via `systemd --user` timer units. Raw evidence, derived findings, and narrative reports are intentionally stored separately so later comparisons and audits can trace every conclusion back to saved source data.

## Scheduling

One user-level `systemd` timer runs daily at `06:00 America/Denver`.

Why one daily run:

- The validation goal is trend measurement over 14 and 28 days, not rapid incident response.
- Fixed wall-clock snapshots make per-day comparison cleaner than event-driven or ad hoc runs.
- A daily cadence keeps operational load low while still surfacing regressions and drift soon enough for this validation workflow.

The daily job does all of the following in one run:

- collect fresh snapshots from both nodes
- evaluate red/yellow flags from the latest evidence and recent logs
- append normalized trend records
- refresh checkpoint report drafts when the T+14 or T+28 date has been reached

## Output Layout

Raw evidence is written under:

`results/revenue-validation/YYYY-MM-DD/<node>/`

Expected files per node include:

- `revenue-dashboard-30.json`
- `revenue-report-summary.json`
- `revenue-profitability.json`
- `revenue-status.json`
- `revenue-config.json`
- `listforwards.json`
- `listpays.json`
- `listpeerchannels.json`
- `hive-members.json` when available
- `feerates.json`
- `debug-log-extract.log`

Derived artifacts are written under:

- `results/revenue-validation/trends/<node>.jsonl`
- `results/revenue-validation/watch/YYYY-MM-DD.json`
- `results/revenue-validation/manifests/YYYY-MM-DD.json`

Generated checkpoint drafts are written under:

- `docs/reports/YYYY-MM-DD-production-t14-findings.md`
- `docs/reports/YYYY-MM-DD-production-t28-findings.md`

## Validation Logic

Each node is tracked independently using its configured `T0`.

For every daily run, the automation:

1. Collects fresh raw evidence for the node.
2. Computes normalized daily metrics and appends one trend record.
3. Recomputes the rolling windows needed by the plan:
   - current snapshot
   - trailing 7-day rollback-watch window
   - trailing 14-day behavior window
   - trailing 28-day economic window
4. Evaluates red flags and yellow flags from the validation plan.
5. Assigns per-node checkpoint state:
   - `pre_t14`
   - `ready_t14`
   - `between_t14_t28`
   - `ready_t28`
   - `post_t28`

The implementation must preserve the plan’s interpretation constraints:

- no experiments
- no automatic rollback
- no on-the-fly tuning
- no manufactured conclusions from weak signal
- opener-aware profitability treatment where required
- amortized capex treatment rather than charging full open cost to a 28-day window

## Failure Handling

The pipeline is conservative by design:

- If one node fails, the other node is still collected and saved.
- The daily run writes a partial manifest describing which node failed and why.
- The overall service exits non-zero on any node failure or any detected red flag so `systemd` and journald expose the run as degraded.
- Scripts never mutate node state. They are read-only by contract.

This gives the operator a visible signal that something needs attention while still preserving whatever evidence was successfully collected.

## T0 Handling

`T0` must be explicit in config, one timestamp per node.

It is not inferred from snapshot file times because:

- the two nodes may not have been deployed at the exact same time
- later manual snapshots are not a reliable deployment boundary
- checkpoint drift would corrupt the T+14 and T+28 report windows

If the operator later corrects `T0`, the report generator must be able to rebuild checkpoint drafts from the saved raw evidence without recollecting historical data.

## Operator Surface

The system should be simple to operate:

- one config file for node definitions, T0, thresholds, and paths
- one daily service and one timer
- one manual command to run the pipeline immediately
- one clear location for raw data, trend data, watch findings, and generated reports

Manual operator actions remain outside the automation:

- reviewing red-flag findings
- deciding whether to pause, investigate, or rollback
- approving any T+14 or T+28 report conclusions

## Verification Strategy

Before leaving the timer scheduled, the implementation should support:

- dry-run execution on the control host
- a manual end-to-end collection run
- report generation from saved data only
- explicit verification that files land in the expected output tree
- explicit verification that the `systemd --user` service can be started manually

## Open Implementation Decisions

These are expected to be settled in the implementation plan:

- exact config schema in `config/revenue_validation.yaml`
- exact normalized trend record shape
- exact log extraction window and command transport wrappers
- exact `systemd --user` unit filenames and install instructions
- exact threshold encoding for the red/yellow checks

## Rationale

This design matches the validation plan’s intent:

- daily evidence collection for four weeks
- trend-focused rather than reactive monitoring
- reproducible checkpoint reports backed by saved artifacts
- no mutation of production state

It is intentionally boring. Reliability, traceability, and comparability matter more here than elegance.
