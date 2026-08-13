# Phase 0.4 Validator Failure Semantics

## Objective

Daily evidence must fail closed when evidence needed for completeness or
economic measurement is unavailable, without turning the loss of a redundant
diagnostic into a false RED watch condition.

## Evidence roles

| Role | Collected surfaces | Failure result |
| --- | --- | --- |
| `required_for_completeness` | `revenue-status`, `revenue-config get`, retained total-cost budget, UTC-bounded reconciliation history, uptime log extract | `incomplete` |
| `required_for_economic_metrics` | rolling dashboard, `revenue-profitability`, `listforwards`, `listpays`, `listpeerchannels` | `incomplete` |
| `optional_diagnostic` | report summary and current `feerates` | `collection_warning` when no required surface failed |

The exact forward, payment, channel, budget, reconciliation, configuration, and
uptime evidence remains required. The rolling dashboard is also required
because it is the source of checkpoint trend economics. The policy report
summary and current feerate snapshot remain optional diagnostics.

## Manifest statuses

| Status | Meaning | Collector exit | Watch behavior |
| --- | --- | ---: | --- |
| `complete` | every classified surface collected | 0 | evaluate normally |
| `collection_warning` | only optional diagnostics failed | 0 | evaluate normally; no synthetic RED |
| `incomplete` | one or more required surfaces failed | 1 | fail closed as collection RED |
| `collection_failure` | unexpected collector exception prevented a classified result | 1 | fail closed as collection RED |

Every per-surface error records its evidence role. Required and optional JSON
payloads are checked for their expected object, keys, and critical inner types;
transport success with empty or malformed data is still an error. Missing uptime
configuration is required-evidence loss rather than an implicit clean result.
Missing data is never synthesized as zero.

## TDD evidence

The focused tests were first run in RED state and produced five expected
failures: legacy `ok` status remained, optional failure became generic `error`,
required economic failure became generic `error`, an unexpected collector
exception escaped without a manifest, and watch converted an optional failure
to RED. After the initial implementation, all 16 focused collector/watch tests passed.
Independent review then reproduced two fail-open paths: empty required payloads
were treated as complete, and a missing optional dashboard created a null trend
that could produce a T28 `ship` decision. New RED tests covered both paths before
the corrections. The final focused collector, watch, and report suite contains
26 passing tests.

The regression cases prove:

- optional report-summary loss becomes `collection_warning` with an
  `optional_diagnostic` error;
- dashboard loss becomes required-economic `incomplete` and appends no trend;
- `listforwards` loss becomes `incomplete` with a
  `required_for_economic_metrics` error;
- empty, wrong-top-level, wrong-inner-type, and malformed list-record required
  payloads become `incomplete` without reaching watch;
- missing T28 net-profit evidence yields `investigate`, never `ship`;
- unexpected collection or trend-persistence exceptions become a durable
  `collection_failure` manifest;
- `collection_warning` does not create a `collection_failure` watch finding;
- `incomplete` remains fail closed;
- historical manifests using `status: ok` remain readable by watch.

## Safety and compatibility

- Collection commands are unchanged from Phase 0.3 and remain read-only.
- No action RPC, config write, fee/rebalance/policy/budget mutation, process
  change, production call, Sling, Hive, mycelium, or fleet dependency was
  introduced.
- Historical artifacts and the legacy `ok` watch input remain supported.
- Checkpoint trend rows are written only when both required dashboard and status
  evidence are structurally valid. T28 decisions require a numeric net-profit
  value. The daily pipeline still records watch RED and a nonzero overall
  result for required evidence loss.

## Limitation and recommendation

This phase classifies collection transport outcomes. It does not yet inspect
the content of budget, reconciliation, fee-intent, or uptime evidence to decide
whether a UTC day is countable; that is the remaining Phase 0 daily-completeness
work.

**CONTINUE SHADOW.** Deploy only with the Phase 0.1-0.3 prerequisite stack, then
prove the 72-hour reconstructability gate before enabling optimization work.
