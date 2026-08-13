# cl-revenue-ops

`cl_revenue_ops` is the independent local execution layer for Core Lightning routing economics. It owns fee control, budget-constrained rebalance decisioning/execution, profitability analysis, revenue reporting, and budget enforcement. It does not open or close channels and it does not execute swaps.

It is fully standalone: every decision runs on local evidence only (own forwards, gossip, node state). The former cl-mycelium/cl-hive fleet-hint integration was retired and removed in 2026-07 (see [docs/audit/HIVE_REMOVAL_PLAN.md](docs/audit/HIVE_REMOVAL_PLAN.md)).

## Product Architecture

```text
cl_revenue_ops decides and executes locally.
Core Lightning owns node runtime.
```

`cl_revenue_ops` is the local executor: it decides what is safe, applies local budgets and policy, and uses Core Lightning RPCs for execution when operator controls allow it.

## Economic Governance

Every retained economic mutation -- fee broadcasts and rebalances -- flows through one governed core:

```text
canonical EconomicSnapshot -> typed intents -> arbiter -> governor -> execution -> append-only ledger
```

- **Typed intents** carry EV, confidence, cost cap, and a snapshot reference.
- **Arbitration** rejects duplicate, stale, and conflicting fee or rebalance intents with stable reason codes.
- **Governance** applies pause, authority, staleness, policy, and atomic budget gates before execution.
- **Ledger** records governed decisions and spend for reporting and reconciliation.
- Runtime governance remains flag-gated through the public `revenue-config` controls.

The wire contracts are versioned and language-neutral: see [`schemas/`](schemas/), the conformance corpus under `tests/conformance/scenarios/`, and [ADR-001](docs/refactor/adr/ADR-001-dts-pid-fee-controller.md).

## What Operators Need To Know

- The plugin owns local fee execution, circular rebalance execution, profitability analysis, revenue reporting, and rebalance budget enforcement.
- It has no automatic channel-open, channel-close, Boltz, or LN+ execution path.
- Live rebalances use `RebalanceEngineV2` with native explicit-route execution and the `v3`/askrene router path.
- There is no Sling dependency.
- The main controls are `paused`, daily and weekly rebalance budgets, fee rails, `authority_level`, and `risk_profile`.
- `paused` is the kill switch for discretionary fee and rebalance execution. Read-only analysis and reporting remain available.
- Python fee authority is controlled through the dynamic `revenue-ops-fee-authority-enabled` option. Inspect it with `revenue-fee-authority-status` and use the [fee-authority handoff runbook](docs/runbooks/python-fee-authority-handoff.md) for a separately approved cutover.
- The primary operator surfaces are `revenue-status`, `revenue-fee-debug`, and `revenue-rebalance-debug`.
- The normal workflow is decision explainability first, knob tuning second.
- Auto fee bands are enabled by default; manual policy bands are fallback only.
- `revenue-policy list|get|find|changes` are diagnostic surfaces. Policy writes are explicit operator actions.
- The generic `no_close` policy remains stored for existing channels and external/manual workflows, but this plugin has no close executor.

## Rebalance Execution

- Route selection is pinned to `rebalance_router=v3`, which uses askrene path discovery.
- Live execution uses the explicit route priced by askrene and pays it with native Core Lightning RPCs.
- Failed route segments are recorded as local observations and exported through the `["revenue", "segment-observations"]` datastore key for read-only route evidence.
- No Sling plugin is required or used by the current execution path.

## Profitability Analysis

The profitability analyzer tracks channel economics using **millisatoshi-native accounting** — all revenue values are stored in msat internally and converted to sats only at reporting boundaries. This ensures sub-satoshi fees are never silently truncated to zero.

Key concepts:
- **Channel valuation** uses `max(exit_fees, sourced_fees)` — channels are credited for their most valuable role (routing exit traffic *or* sourcing inbound traffic).
- **Fleet revenue** sums exit fees only across all channels — no double-counting of inbound sourcing.
- **Inbound gateways** are classified from sourced traffic so reporting and rebalance decisions do not misclassify them from exit-side metrics alone.
- **Classification** uses total forward count (exit + sourced) for ZOMBIE, STAGNANT, and fleet member protection decisions.
- **Rebalance fee persistence** is msat-native: successful automatic and manual rebalances persist `actual_fee_msat` in `rebalance_history` and `cost_msat` in `rebalance_costs`, with sat fields derived once at write time for compatibility.
- **Conversion rules** are explicit at reporting boundaries: balances and capacity floor to sats, revenue ceils to sats (so sub-satoshi earnings stay visible), costs and budgets ceil to sats, and signed net deltas round toward zero so `-1msat` does not become a fabricated `-1sat` loss.

Use `revenue-profitability` to see per-channel analysis including sourced metrics, flow profiles, and total contribution.

### Profitability Snapshot Contract

`cl-revenue-ops` publishes the canonical profitability snapshot for external read-only consumers to CLN datastore key `["revenue", "profitability-summary"]`.

Payload shape:
- Top level: `timestamp`, `channels`
- Per channel: `channel_id`, `peer_id`, `class`, `roi_pct`, `days_open`, `role`, `fee_multiplier`
- Per channel msat fields: `fees_earned_msat`, `sourced_fee_contribution_msat`, `total_contribution_msat`, `volume_routed_msat`, `sourced_volume_msat`, `open_cost_msat`, `rebalance_cost_msat`, `net_pnl_msat`
- Per channel counters: `forward_count`, `sourced_forward_count`, `total_forward_count`

`revenue-profitability` remains available as an RPC surface, but the datastore snapshot is the canonical cross-plugin contract and is the path external consumers should prefer. See [docs/contracts/REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md](docs/contracts/REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md).

## Produced Telemetry Contracts

`cl_revenue_ops` publishes read-only telemetry for external consumers (e.g. monitoring/management tooling):

| Datastore key | Contract | Notes |
| --- | --- | --- |
| `["revenue","profitability-summary"]` | [REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md](docs/contracts/REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md) | msat-native profitability and channel role telemetry. |
| `["revenue","capex-summary"]` | [REVENUE_CAPEX_SUMMARY_CONTRACT.md](docs/contracts/REVENUE_CAPEX_SUMMARY_CONTRACT.md) | capital posture telemetry; cannot authorize spend. |
| `["revenue","segment-observations"]` | [REVENUE_SEGMENT_OBSERVATIONS_CONTRACT.md](docs/contracts/REVENUE_SEGMENT_OBSERVATIONS_CONTRACT.md) | route segment evidence; stale/malformed observations produce no penalty or score change. |

Consumers must treat stale, missing, or malformed payloads as unknown confidence, not zero value or an action command.

## Retired Liquidity Executors

Version 3.0.0 removes the CapacityPlanner, automatic channel opening and closing, planner defibrillation, Boltz swaps, and LN+ integration. These features increased code and external-API attack surface without producing operator value on this node; channel lifecycle and swap decisions remain manual operator work outside the plugin.

Historical database schemas and rows are intentionally preserved so old accounting and audit records remain readable. They are inert: historical planner, open/close, Boltz, and LN+ records cannot schedule work or authorize an action. The generic `no_close` policy remains available for existing channel metadata.

Use the [liquidity executor decommission runbook](docs/operations/LIQUIDITY_EXECUTOR_DECOMMISSION_RUNBOOK.md) for deployment, removed-surface checks, rollback gates, and evidence collection.

## Additional Runtime Subsystems

These retained subsystems are documented in [config/cl-revenue-ops.conf.full](config/cl-revenue-ops.conf.full):

- **Hot-channel protection** widens a profitable, fast-draining channel rebalance budget within its configured caps.
- **Growth budget** can add a bounded, profit-funded uplift to the base rebalance budget; it is disabled by default.
- **Dynamic htlcmax** adjusts advertised `htlc_max` from local flow and spendable-balance evidence with a gossip-churn deadband.
- **Fee replay capture** is default-off observational instrumentation for offline Rust parity work. It never starts a fee cycle.
- **Drain-bias / receivable-ratio** may bias fees on over-local channels; circular rebalancing remains the only liquidity execution mechanism.

## Architecture

```text
pair selection / fee decisions
    ↓
v3 router (askrene)
    ↓
RebalanceEngineV2
    ↓
native sendpay / waitsendpay
    ↓
Core Lightning
```

## Install

### Requirements

- Core Lightning `v24.11.1+` with askrene layer RPC support
- Python `3.10+`
- bookkeeper plugin: recommended for cleaner P&L and cost accounting

See [Core Lightning Compatibility](docs/CORE_LIGHTNING_COMPATIBILITY.md) for the
Polar-tested version floor and smoke checklist.

### License

`cl-revenue-ops` is released under the BSD 3-Clause License.

### Start The Plugin

```bash
cd ~/.lightning/plugins
git clone https://github.com/lightning-goats/cl_revenue_ops.git
cd cl_revenue_ops
pip install -r requirements.txt
chmod +x cl-revenue-ops.py
lightning-cli plugin start "$(pwd)/cl-revenue-ops.py"
```

### Upgrade

```bash
lightning-cli plugin stop cl-revenue-ops
git -C ~/.lightning/plugins/cl_revenue_ops pull
lightning-cli plugin start ~/.lightning/plugins/cl_revenue_ops/cl-revenue-ops.py
```

## Production Validation Automation

This repo includes a read-only daily validation pipeline for tracking the production effect of fee, capex, and rebalance behavior on `lnnode` (single production node since 2026-07-11). It preserved evidence for the closed refactor evaluation (`docs/refactor/phase0/production-evaluation-final.md`) and now supports the optimization measurement preflight.

- Before enabling a formal validation window, set each node's `evaluation.id`, `evaluation.version`, `evaluation.state`, `evaluation.formal_window_active`, and `evaluation.t0`; preflight identities must keep the formal-window flag false.
- The timer runs once per day at `06:00` in the control host's local timezone. The intended host timezone for this workflow is `America/Denver`.
- The pipeline is read-only by design: it collects evidence, evaluates red/yellow rollout-watch checks, and refreshes draft T+14/T+28 reports.

Install the user units on the control host:

```bash
mkdir -p ~/.config/systemd/user
cp tools/systemd/revenue-validation-daily.service ~/.config/systemd/user/
cp tools/systemd/revenue-validation-daily.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now revenue-validation-daily.timer
```

Trigger a manual run:

```bash
systemctl --user start revenue-validation-daily.service
systemctl --user status revenue-validation-daily.service --no-pager
```

Saved artifacts:

- Raw daily evidence: `results/revenue-validation/YYYY-MM-DD/<node>/`
- Daily manifest: `results/revenue-validation/manifests/YYYY-MM-DD.json`
- Daily watch findings: `results/revenue-validation/watch/YYYY-MM-DD.json`
- Trend rows: `results/revenue-validation/trends/<node>.jsonl`
- Draft checkpoint reports: `docs/reports/*-production-t14-findings.md` and `docs/reports/*-production-t28-findings.md`

## Day-1 Operator Workflow

1. Start the plugin and check `revenue-status`.
2. Set only the safety rails you actually want to constrain: `paused`, `daily_budget_sats`, `min_fee_ppm`, `max_fee_ppm`, `fee_profile`, and `authority_level` (default `capital`; dial DOWN to restrict what the node may do). `risk_profile` stays `custom` unless you explicitly adopt a bundle — run `revenue-profile-preview <name>` first to see the exact diff. (The `fee_market_boundary_*` controls and `rebalance_min_profit` are deprecated no-ops with an announced removal date of 2026-08-12; minimum-profit gating is enforced by the sats-EV gate and `rebalance_hold_margin`.)
3. Let the executor run.
4. Use `revenue-fee-debug` and `revenue-rebalance-debug` to understand holds, clamps, and actions before touching anything else.
Example runtime adjustments:

```bash
lightning-cli revenue-config get
lightning-cli revenue-config set fee_profile conservative
lightning-cli revenue-config set min_fee_ppm 75
lightning-cli revenue-config set daily_budget_sats 10000
```

## Primary RPCs

| Command | Use |
|---|---|
| `revenue-status` | Health, controls, and latest fee/rebalance decisions |
| `revenue-fee-debug` | Why a fee moved, held, or was clamped |
| `revenue-rebalance-debug` | Why a rebalance was selected, skipped, or blocked |
| `revenue-config get` | Inspect current runtime controls |
| `revenue-config set <key> <value>` | Change a supported runtime control |
| `revenue-profitability [channel_id]` | Per-channel profitability and flow analysis |
| `revenue-cycle <subsystem>` | Run `fees`, `rebalance`, `flow`, or `all` |
| `revenue-budget [section]` | Read total-cost, rebalance-capex, or spend-ledger state |
| `revenue-profile-preview [name]` | Preview a risk-profile change |
| `revenue-econ-snapshot` | Read the canonical economic snapshot |
| `revenue-econ-reconcile` | Read ledger/reservation reconciliation |
| `revenue-econ-cycle` | Run the deterministic shadow economic cycle |

The complete 36-method contract and mutation classification is in [the action RPC inventory](docs/audits/CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md).

### Retained compatibility aliases

The primary dispatchers are `revenue-cycle`, `revenue-budget`, and
`revenue-policy`. Retained standalone aliases cover fee/rebalance cycles,
analysis/wake, and total-cost/capex/spend-ledger reads. Removed peer-ban,
planner, Boltz, and LN+ names are not compatibility aliases and must return
method-not-found.

### revenue-config: actions and override precedence

`revenue-config` supports four actions: `get [key]` (public controls, or one key with its classification), `set <key> <value>`, `reset <key>`, and `list-mutable` (lists every public runtime key that `set`/`reset` will accept).

- `revenue-config set <key> <value>` writes a DB override for that key. Once set, the override **wins over `setconfig`/config-file changes to the same option on every dynamic-config refresh cycle** — the refresh loop explicitly skips any field with an active DB override rather than stomping it back to the file/`setconfig` value.
- `revenue-config reset <key>` removes the DB override, the escape hatch that lets `setconfig`/config-file values govern the field again (some fields apply immediately; others require a plugin restart to re-adopt the file default — the RPC response says which).
- `revenue-config list-mutable` returns the current set of public runtime keys; only keys in this list can be `set` or `reset` (all others return `"not a public runtime control"`).

## cl_revenue_ops standalone invariant

`cl_revenue_ops` is a fully independent local executor. The former cl-hive/cl-mycelium hint integration was removed entirely in 2026-07 (`docs/audit/HIVE_REMOVAL_PLAN.md`); `tests/test_architecture_guard.py` pins that no hive/fleet coordination code returns.

The read-only operator surfaces `revenue-status`, `revenue-fee-debug`, and `revenue-rebalance-debug` must keep returning JSON with no external plugins present, and read-only surfaces must never call fee, rebalance, spend, or CLN mutation RPCs. Removed planner, Boltz, and LN+ method names must remain absent.

## Public Contract Docs

- Contract index: [docs/contracts/README.md](docs/contracts/README.md)
- Standalone independence audit: [docs/audits/2026-05-19-standalone-independence-audit.md](docs/audits/2026-05-19-standalone-independence-audit.md)
- Hive/mycelium removal record: [docs/audit/HIVE_REMOVAL_PLAN.md](docs/audit/HIVE_REMOVAL_PLAN.md)
- Economic-core wire contracts: [schemas/](schemas/) + compatibility policy: [docs/refactor/phase0/contract-compatibility-policy.md](docs/refactor/phase0/contract-compatibility-policy.md)
- Fee-controller contract (ADR-001): [docs/refactor/adr/ADR-001-dts-pid-fee-controller.md](docs/refactor/adr/ADR-001-dts-pid-fee-controller.md)
- Refactor completion review: [docs/refactor/phase0/completion-review.md](docs/refactor/phase0/completion-review.md)
- Governance evidence report: [docs/refactor/phase0/governance-evidence-report.md](docs/refactor/phase0/governance-evidence-report.md)

## More Detail

- Minimal config example: [config/cl-revenue-ops.conf.minimal](config/cl-revenue-ops.conf.minimal)
- Full config example: [config/cl-revenue-ops.conf.full](config/cl-revenue-ops.conf.full)
- Note: the governance keys (`econ_*`, `authority_level`, `risk_profile`) are RUNTIME controls (`revenue-config set`), not config-file options — `revenue-config list-mutable` enumerates them.
