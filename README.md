# cl-revenue-ops

`cl_revenue_ops` is the independent local execution layer for Core Lightning routing economics. It owns fee control, rebalance decisioning/execution, planner/capex, profitability analysis, and budget enforcement. It watches channel economics, adjusts fees, and executes rebalancing while keeping the normal operator surface intentionally small.

It can consume bounded cl-mycelium hints, but those hints are optional enhancements, not dependencies. `cl_revenue_ops` must run safely when cl-mycelium or cl-hive is absent.

## Product Architecture

```text
cl-mycelium coordinates.
cl_revenue_ops executes.
Core Lightning owns node runtime.
```

`cl-mycelium` may produce bounded hints and read-only context. `cl_revenue_ops` remains the local executor: it decides what is safe, applies local budgets and policy, and uses Core Lightning RPCs for execution when operator controls allow it.

## What Operators Need To Know

- This is the executor. It owns local fee execution, rebalance execution, planner/capex decisions, profitability analysis, and budgets.
- `cl-mycelium` is the fleet coordination organism. `cl_revenue_ops` consumes its hints, but all spending decisions remain local and bounded by this plugin's controls.
- `cl_revenue_ops` runs safely without cl-mycelium or cl-hive. Missing, stale, malformed, or unavailable hints neutralize safely.
- Hints may bias local decisions only within bounded caps; they never override local budget, safety, or executor policy.
- Live rebalances execute through `RebalanceEngineV2` using native explicit-route execution; route discovery is pinned to the `v3`/askrene router path.
- There is no Sling dependency.
- The normal runtime controls are `paused`, `daily_budget_sats`, fee rails, fee market-boundary knobs, and planner execution caps.
- The primary operator surfaces are `revenue-status`, `revenue-fee-debug`, and `revenue-rebalance-debug`.
- The normal workflow is decision explainability first, knob tuning second.
- Auto fee bands are enabled by default. Manual policy bands are fallback only when an auto band is not yet available.
- `revenue-policy list|get|find|changes` are diagnostic surfaces. Write actions such as `set` and `delete` remain internal or debug workflows, not the normal operator path.
- Planner closes are recommendation-only by default.
- To allow live close RPCs, set `revenue-ops-planner-execute-closes=true` and `revenue-ops-planner-max-closes-per-cycle` to a positive value.
- Source-heavy drain: `revenue-ops-receivable-ratio-target`/`-floor` define the node-level inbound objective shown in `revenue-status.receivable`. The circular rebalancer always keeps first claim on internal redistribution; only the residual it cannot place (`revenue-rebalance-debug.drain_demand`) may earn the Boltz structural credit, capped by `revenue-ops-boltz-structural-budget-sats` per day (default 0 = off). `revenue-ops-drain-fee-discount-max` optionally biases fees down on stagnant over-local channels (default 0.0 = off).
- Rebalancing and Boltz swaps remain separate tools with separate budgets: rebalancing redistributes at routing-fee cost; loop-outs change the node's aggregate balance at swap+chain cost. Note: the structural envelope is enforced from recorded spend — a swap whose fee fails to record does not deplete it, so keep the envelope small relative to the unified daily budget.

## Rebalance Execution

- Route selection is pinned to `rebalance_router=v3`, which uses askrene and cl-mycelium-aware path discovery through the stable hive route layers.
- Live execution uses the explicit route priced by askrene and pays it with native Core Lightning RPCs.
- Failed route segments are recorded as local observations and exported through the `["revenue", "segment-observations"]` datastore key for read-only route evidence.
- No Sling plugin is required or used by the current execution path.

## Profitability Analysis

The profitability analyzer tracks channel economics using **millisatoshi-native accounting** — all revenue values are stored in msat internally and converted to sats only at reporting boundaries. This ensures sub-satoshi fees are never silently truncated to zero.

Key concepts:
- **Channel valuation** uses `max(exit_fees, sourced_fees)` — channels are credited for their most valuable role (routing exit traffic *or* sourcing inbound traffic).
- **Fleet revenue** sums exit fees only across all channels — no double-counting of inbound sourcing.
- **Inbound gateways** (channels that primarily source traffic for the fleet) receive enhanced protection from closure and are never misclassified as stagnant or zombie based on exit-side metrics alone.
- **Classification** uses total forward count (exit + sourced) for ZOMBIE, STAGNANT, and fleet member protection decisions.
- **Rebalance fee persistence** is msat-native: successful automatic, coordinated, manual, and diagnostic rebalances persist `actual_fee_msat` in `rebalance_history` and `cost_msat` in `rebalance_costs`, with sat fields derived once at write time for compatibility.
- **Conversion rules** are explicit at reporting boundaries: balances and capacity floor to sats, revenue ceils to sats (so sub-satoshi earnings stay visible), costs and budgets ceil to sats, and signed net deltas round toward zero so `-1msat` does not become a fabricated `-1sat` loss.

Use `revenue-profitability` to see per-channel analysis including sourced metrics, flow profiles, and total contribution.

### Profitability Snapshot Contract

`cl-revenue-ops` publishes the canonical profitability snapshot for `cl-mycelium` to CLN datastore key `["revenue", "profitability-summary"]`.

Payload shape:
- Top level: `timestamp`, `channels`
- Per channel: `channel_id`, `peer_id`, `class`, `roi_pct`, `days_open`, `role`, `fee_multiplier`
- Per channel msat fields: `fees_earned_msat`, `sourced_fee_contribution_msat`, `total_contribution_msat`, `volume_routed_msat`, `sourced_volume_msat`, `open_cost_msat`, `rebalance_cost_msat`, `net_pnl_msat`
- Per channel counters: `forward_count`, `sourced_forward_count`, `total_forward_count`

`revenue-profitability` remains available as an RPC surface, but the datastore snapshot is the canonical cross-plugin contract and is the path `cl-mycelium` should prefer. See [docs/contracts/REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md](docs/contracts/REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md).

## Produced Telemetry Contracts

`cl_revenue_ops` publishes read-only telemetry for cl-mycelium and other consumers:

| Datastore key | Contract | Notes |
| --- | --- | --- |
| `["revenue","profitability-summary"]` | [REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md](docs/contracts/REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md) | msat-native profitability and channel role telemetry. |
| `["revenue","capex-summary"]` | [REVENUE_CAPEX_SUMMARY_CONTRACT.md](docs/contracts/REVENUE_CAPEX_SUMMARY_CONTRACT.md) | capital posture telemetry; cannot authorize spend. |
| `["revenue","segment-observations"]` | [REVENUE_SEGMENT_OBSERVATIONS_CONTRACT.md](docs/contracts/REVENUE_SEGMENT_OBSERVATIONS_CONTRACT.md) | route segment evidence; stale/malformed observations produce no penalty or score change. |

Consumers must treat stale, missing, or malformed payloads as unknown confidence, not zero value or an action command.

## Channel Opening Intelligence

The capacity planner uses a multi-strategy candidate pipeline with portfolio-aware governance:

- **Portfolio balance governor** — hard gate at >95% local blocks outbound opens, constrained at 85-95% allows only sink-adjacent or dual-fund
- **Multi-strategy discovery** — winner (proven revenue), demand-flow (gossip heuristics), mycelium/hive hint contract (fleet intelligence), route-pair, graph, and neighbor strategies
- **Score normalization** — within-strategy 0-1 normalization with configurable weights; pool slot quotas prevent strategy monoculture
- **Demand-flow classifier** — classifies peers as source/sink/router using FlowMetrics aggregation and gossip heuristics (exchange, LSP, sink keyword matching)
- **Capital hurdle** — open EV subtracts a configurable annualized return hurdle (`planner_min_annual_roi_pct`, default 1%) so low-yield channel opens do not pass on tiny absolute-profit edges
- **Capital recycling** — evaluates underperformers for close-and-reopen when recycle EV exceeds threshold; coordinates with Boltz for on-chain fund management
- **Dual-fund support** — uses `fundchannel request_amt` when peers advertise `option_will_fund`

## Boltz Automation

- The in-plugin Boltz auto-cycle is treasury mode first when confirmed on-chain funds are below the configured reserve target.
- It maintains a standing on-chain reserve for reserve maintenance, and that reserve maintenance is independent of pending planner opens.
- When the reserve is healthy, it falls back to the existing balance cycle.

## Architecture

```text
pair selection / fee decisions
    ↓
configured router (v3 askrene+mycelium/hive layers or v2 local)
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

This repo includes a read-only daily validation pipeline for tracking the production effect of recent fee, capex, and rebalance changes across `lnnode` and `hive-nexus-02`.

- Edit `config/revenue_validation.yaml` and replace each node `t0` placeholder with the actual deploy timestamp before enabling the timer.
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
2. Set only the safety rails you actually want to constrain: `paused`, `daily_budget_sats`, `min_fee_ppm`, `max_fee_ppm`, `fee_profile`, and the `fee_market_boundary_*` controls.
3. Let the executor run.
4. Use `revenue-fee-debug` and `revenue-rebalance-debug` to understand holds, clamps, and actions before touching anything else.
Example runtime adjustments:

```bash
lightning-cli revenue-config get
lightning-cli revenue-config set fee_profile conservative
lightning-cli revenue-config set min_fee_ppm 75
lightning-cli revenue-config set fee_market_boundary_margin_ratio 0.03
lightning-cli revenue-config set daily_budget_sats 10000
```

## Primary RPCs

| Command | Use |
|---|---|
| `revenue-status` | Health, operator controls, and latest fee/rebalance decisions |
| `revenue-fee-debug` | Why a fee moved, held, or was clamped |
| `revenue-rebalance-debug` | Why a rebalance was selected, skipped, or blocked |
| `revenue-config get` | Inspect current runtime controls |
| `revenue-config set <key> <value>` | Change one of the supported runtime controls |
| `revenue-profitability [channel_id]` | Per-channel profitability with sourced metrics and flow profiles |
| `revenue-analyze` | Trigger immediate analysis |
| `revenue-wake-all` | Wake the background loops immediately |
| `revenue-hive-hints-status` | Diagnostic: cl-mycelium hint coverage and freshness |
| `revenue-planner-candidate-sources` | Diagnostic: candidate pipeline strategy breakdown |

## cl-mycelium Hints

`cl_revenue_ops` consumes `cl-mycelium` fleet hints only through `modules/hive_hints.py` (`HiveHintAdapter`). The adapter name, `hive-*` RPC names, and `["hive", "hints"]` datastore key remain the stable compatibility contract documented in [docs/contracts/HIVE_HINTS_CONTRACT.md](docs/contracts/HIVE_HINTS_CONTRACT.md).

- Transport order is datastore first: read CLN datastore key `["hive", "hints"]`, then fall back to `hive-export-hints` only if the datastore payload is missing, stale, or invalid.
- Missing, stale, malformed, or unavailable hints degrade to neutral local behavior; they do not bypass fee, rebalance, planner, budget, or policy safety rails.
- Once per fee cycle, `cl_revenue_ops` polls the hint snapshot and refreshes the shared `HiveRouter` compatibility layer (`hive-fleet` layer detection, fleet balance cache, route cache clear) so inbound-fee estimation and Boltz topology scoring see live fleet state instead of a startup-only snapshot.
- Rebalance candidates are classified before pricing as `hive_only`, `hybrid`, or `market_only`. `hive_only` uses the active cl-mycelium-aware route pricer with live `hive-*` and `revenue-*` askrene layers, `hybrid` compares that fleet-aware route against the configured market router, and `market_only` stays on the configured router only.
- Coordination hints now seed candidate generation before the active pair cap is applied. `rebalance_recommendations` / `rebalance_campaigns` can materialize coordinated pairs from peer IDs, local SCIDs, or route segments, and may steer policy via `route_policy`, `allow_market_fallback`, `prefer_hive_on_tie`, and `priority_score`.
- `route_segment_leases` are honored during that overlay stage: overlapping foreign leases suppress the candidate with an explicit `lease_conflict` audit reason, while our own leases are allowed through.
- Additional live hint consumers:
  - `fee_elasticity` slightly widens or narrows DTS exploration variance
  - `reputation_score` and `corridor_utilization_bias` modestly bias capacity-planner open scoring
  - `drain_direction` remains askrene/diagnostic only; the fee controller intentionally does not apply it directly
- `revenue-hive-hints-status` reports freshness and signal coverage for the currently cached cl-mycelium hint snapshot.
- Metabolism Level 2c: optional `metabolic_influence/v1` is consumed fresh-only and scope-valid as bounded scoring input: fee bias `[0.95, 1.05]`, rebalance bias `[0.85, 1.15]`, and planner/open bias `[0.85, 1.10]`. Immune/pathology Level 2c: optional `immune_influence/v1` is also consumed fresh-only and scope-valid as bounded scoring input with the same hard caps. Neither payload grants budget, peer-suppression, or execution authority, and neither proves Level 3 value.

### Hint Diagnostics

`revenue-hive-hints-status` is the primary full freshness diagnostic. The current diagnostic surface includes `diagnostics_version=standalone-hints-v1` and reports cache status, `cache_after_refresh`, `live_datastore`, `live_hive_export`, fallback state, and segment score counts.

`revenue-rebalance-debug.hive_hints` corroborates hint freshness. `revenue-fee-debug` is a lighter supporting surface for fee debugging, not the primary full freshness diagnostic.

## cl_revenue_ops standalone invariant

`cl_revenue_ops` remains a fully independent local executor when cl-hive or cl-mycelium is absent. Hint integration is confined to `modules/hive_hints.py`; missing datastore entries, unknown `hive-export-hints`, stale snapshots, malformed payloads, and disabled hint adapters must degrade to neutral hint lookups rather than crashing or changing budgets.

The read-only operator surfaces `revenue-status`, `revenue-fee-debug`, `revenue-rebalance-debug`, and `revenue-hive-hints-status` must keep returning JSON in standalone mode. Bad hints must not call fee, rebalance, planner, Boltz, or CLN mutation RPCs. Valid classic cl-hive hints, valid cl-mycelium M2-scoped hints, optional metabolic influence, and optional immune influence may bias local fee/rebalance/planner behavior only through bounded caps; they never override local budget, safety, or executor policy. M2 `all_hints` is not a production default for this plugin.

## Public Contract Docs

- Contract index: [docs/contracts/README.md](docs/contracts/README.md)
- Cross-repo doc reference audit: [docs/audits/CROSS_REPO_DOC_REFERENCE_AUDIT.md](docs/audits/CROSS_REPO_DOC_REFERENCE_AUDIT.md)
- Standalone independence audit: [docs/audits/2026-05-19-standalone-independence-audit.md](docs/audits/2026-05-19-standalone-independence-audit.md)
- Hint freshness diagnostics audit: [docs/audits/HIVE_HINT_FRESHNESS_DIAGNOSTICS_AUDIT.md](docs/audits/HIVE_HINT_FRESHNESS_DIAGNOSTICS_AUDIT.md)
- Cross-plugin contract audit: [docs/audits/CROSS_PLUGIN_CONTRACT_AUDIT.md](docs/audits/CROSS_PLUGIN_CONTRACT_AUDIT.md)

## More Detail

- Minimal config example: [config/cl-revenue-ops.conf.minimal](config/cl-revenue-ops.conf.minimal)
- Full config example: [config/cl-revenue-ops.conf.full](config/cl-revenue-ops.conf.full)
