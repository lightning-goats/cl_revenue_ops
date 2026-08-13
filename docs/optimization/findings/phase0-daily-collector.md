# Phase 0.3 Daily Validation Collector Repair

## Hypothesis

The daily validation package can remain mechanically useful for Phase 0 measurement integrity only when it collects the retained read-only total-cost budget surface and the UTC-bounded reconciliation history, while no longer issuing the retired `hive-members` RPC.

## Exact collection evidence

For each requested `run_date` the collector writes these additional raw JSON artifacts:

- `revenue-budget.json` from `-k revenue-budget section=total_cost window_hours=24`.
- `revenue-econ-reconcile.json` from `-k revenue-econ-reconcile history_since=<UTC-midnight epoch> history_until=<next UTC-midnight epoch> history_limit=24`.

The reconciliation range is half-open: `[run_date 00:00:00 UTC, next day 00:00:00 UTC)`. It is built from timezone-aware UTC datetimes, uses only absolute epoch bounds, forces lightning-cli keyword mode with `-k` so history bounds cannot bind positionally to `apply`, and omits `apply` so the RPC remains dry-run. The collector no longer invokes `hive-members` and creates no new `hive-members.json` artifact.

## TDD evidence

The focused collector test was first run in RED state. It failed because `revenue-budget.json` and `revenue-econ-reconcile.json` did not exist, and malformed dates were validated only after RPC collection. After the minimal collector change, the focused tests passed.

A follow-up read-only review added RED tests for the exact `-k revenue-budget section=total_cost window_hours=24` collector command and the `total_cost` dispatcher section. The prior collector used bare `revenue-budget`, which reaches the capex telemetry refresh; the prior dispatcher also rejected `total_cost`. The tests turned GREEN only after `total_cost` was wired directly to the total-cost helper and the collector stopped issuing the bare command.

The test asserts the exact 2026-04-23 UTC bounds (`1776902400` through `1776988800`), the absence of `apply`, the absence of `hive-members`, both new evidence files, and zero RPC calls for a malformed `run_date`.

## Safety and compatibility

- The added RPCs are read-only; the budget collector selects `total_cost`, which does not compute capex allocations or push capex datastore telemetry, and reconciliation uses its default dry-run mode.
- No action RPC, direct CLN mutation, config change, fee/rebalance/policy/budget mutation, process change, Sling, Hive, mycelium, or fleet dependency was added.
- Existing collection failure semantics remain fail-closed: any failed collection still marks the node result as `error`. Phase 0.4 owns required-versus-optional classification and is intentionally not implemented here.
- Snapshot, trend, and default daily entrypoint behavior is unchanged. Existing report/watch consumers were not changed, so preserved historical `hive-members.json` evidence remains readable.
- Missing evidence is recorded as a collection error, never synthesized as zero.

## Limitations

The deployment target must include the branch-local `history_since`, `history_until`, and `history_limit` reconciliation arguments before this collector can query historical hourly evidence in production. This change does not classify collection completeness, evaluate reconciliation outcomes, or deploy anything.

## Recommendation

**CONTINUE SHADOW.** Collect the new raw evidence after the matching reconciliation history support is deployed, then proceed to the separate Phase 0.4 completeness-classification work.
