# cl-revenue-ops

`cl-revenue-ops` is the local execution layer for routing profit and liquidity management on Core Lightning. It watches channel economics, adjusts fees, and executes rebalancing while keeping the normal operator surface intentionally small.

## What Operators Need To Know

- This is the executor. It owns local fee execution and rebalance execution.
- Live rebalances on this branch execute through `RebalanceEngineV2` and `sling`; route discovery can still use the configured `v3`/askrene or `v2` router path.
- The normal runtime controls are `paused`, `daily_budget_sats`, `min_fee_ppm`, and `max_fee_ppm`.
- The primary operator surfaces are `revenue-status`, `revenue-fee-debug`, and `revenue-rebalance-debug`.
- The normal workflow is decision explainability first, knob tuning second.
- Auto fee bands are enabled by default. Manual policy bands are fallback only when an auto band is not yet available.
- `sling` must be installed and loaded for live rebalances. If `sling-once` is unavailable, rebalance execution is skipped.
- `revenue-ops-sling-*` options are internal startup-hygiene overrides, not normal runtime knobs exposed through `revenue-config`.
- `revenue-policy list|get|find|changes` are diagnostic surfaces. Write actions such as `set` and `delete` remain internal or debug workflows, not the normal operator path.
- Planner closes are recommendation-only by default.
- To allow live close RPCs, set `revenue-ops-planner-execute-closes=true` and `revenue-ops-planner-max-closes-per-cycle` to a positive value.

## Sling Integration

- Route selection remains branch-specific: `rebalance_router=v3` keeps askrene and hive-layer path discovery, while `v2` keeps the legacy local router.
- Regardless of router choice, live execution is `sling-once` plus `sling-stats` observation; the older internal `sendpay` rebalance executor path is no longer used on this branch.
- At startup, `cl-revenue-ops` applies the subset of `sling-*` runtime hygiene it owns when those CLN config keys are exposed: stats retention, candidate age, max hops, and depletion guards.
- The mirrored `revenue-ops-sling-*` plugin options exist so operators can pin those defaults in `lightningd` config without making them part of the normal runtime control surface.

## Profitability Analysis

The profitability analyzer tracks channel economics using **millisatoshi-native accounting** — all revenue values are stored in msat internally and converted to sats only at reporting boundaries. This ensures sub-satoshi fees are never silently truncated to zero.

Key concepts:
- **Channel valuation** uses `max(exit_fees, sourced_fees)` — channels are credited for their most valuable role (routing exit traffic *or* sourcing inbound traffic).
- **Fleet revenue** sums exit fees only across all channels — no double-counting of inbound sourcing.
- **Inbound gateways** (channels that primarily source traffic for the fleet) receive enhanced protection from closure and are never misclassified as stagnant or zombie based on exit-side metrics alone.
- **Classification** uses total forward count (exit + sourced) for ZOMBIE, STAGNANT, and fleet member protection decisions.
- **Rebalance fee persistence** is msat-native: successful automatic, coordinated, manual, and diagnostic rebalances persist `actual_fee_msat` in `rebalance_history` and `cost_msat` in `rebalance_costs`, with sat fields derived once at write time for compatibility.
- **Conversion rules** are explicit at reporting boundaries: revenue and balances floor to sats, costs and budgets ceil to sats, and signed net deltas round toward zero so `-1msat` does not become a fabricated `-1sat` loss.

Use `revenue-profitability` to see per-channel analysis including sourced metrics, flow profiles, and total contribution.

### Profitability Snapshot Contract

`cl-revenue-ops` publishes the canonical profitability snapshot for `cl-hive` to CLN datastore key `["revenue", "profitability-summary"]`.

Payload shape:
- Top level: `timestamp`, `channels`
- Per channel: `channel_id`, `peer_id`, `class`, `roi_pct`, `days_open`, `role`, `fee_multiplier`
- Per channel msat fields: `fees_earned_msat`, `sourced_fee_contribution_msat`, `total_contribution_msat`, `volume_routed_msat`, `sourced_volume_msat`, `open_cost_msat`, `rebalance_cost_msat`, `net_pnl_msat`
- Per channel counters: `forward_count`, `sourced_forward_count`, `total_forward_count`

`revenue-profitability` remains available as an RPC surface, but the datastore snapshot is the canonical cross-plugin contract and is the path `cl-hive` should prefer.

## Channel Opening Intelligence

The capacity planner uses a multi-strategy candidate pipeline with portfolio-aware governance:

- **Portfolio balance governor** — hard gate at >95% local blocks outbound opens, constrained at 85-95% allows only sink-adjacent or dual-fund
- **Multi-strategy discovery** — winner (proven revenue), demand-flow (gossip heuristics), hive (fleet intelligence), route-pair, graph, and neighbor strategies
- **Score normalization** — within-strategy 0-1 normalization with configurable weights; pool slot quotas prevent strategy monoculture
- **Demand-flow classifier** — classifies peers as source/sink/router using FlowMetrics aggregation and gossip heuristics (exchange, LSP, sink keyword matching)
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
configured router (v3 askrene+hive or v2 local)
    ↓
RebalanceEngineV2
    ↓
sling-once / sling-stats
    ↓
Core Lightning
```

## Install

### Requirements

- Core Lightning `v23.05+`
- Python `3.10+`
- `sling` plugin: required for live rebalance execution
- bookkeeper plugin: recommended for cleaner P&L and cost accounting

### Start The Plugin

```bash
cd ~/.lightning/plugins
git clone https://github.com/lightning-goats/cl_revenue_ops.git
cd cl_revenue_ops
pip install -r requirements.txt
chmod +x cl-revenue-ops.py
# Ensure the sling plugin is already installed and loaded by lightningd.
lightning-cli plugin start "$(pwd)/cl-revenue-ops.py"
```

### Upgrade

```bash
lightning-cli plugin stop cl-revenue-ops
git -C ~/.lightning/plugins/cl_revenue_ops pull
lightning-cli plugin start ~/.lightning/plugins/cl_revenue_ops/cl-revenue-ops.py
```

## Day-1 Operator Workflow

1. Start the plugin and check `revenue-status`.
2. Set only the safety rails you actually want to constrain: `paused`, `daily_budget_sats`, `min_fee_ppm`, `max_fee_ppm`.
3. Let the executor run.
4. Use `revenue-fee-debug` and `revenue-rebalance-debug` to understand holds, clamps, and actions before touching anything else.
Example runtime adjustments:

```bash
lightning-cli revenue-config get
lightning-cli revenue-config set min_fee_ppm 75
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
| `revenue-hive-hints-status` | Diagnostic: hive hints coverage and freshness |
| `revenue-planner-candidate-sources` | Diagnostic: candidate pipeline strategy breakdown |

## Hive Hints

`cl_revenue_ops` consumes fleet hints only through `modules/hive_hints.py` (`HiveHintAdapter`).

- Transport order is datastore first: read CLN datastore key `["hive", "hints"]`, then fall back to `hive-export-hints` only if the datastore payload is missing, stale, or invalid.
- Missing or malformed per-peer hint entries degrade to neutral local behavior; they do not bypass fee, rebalance, planner, or policy safety rails.
- Once per fee cycle, `cl_revenue_ops` polls the hint snapshot and refreshes the shared `HiveRouter` (`hive-fleet` layer detection, fleet balance cache, route cache clear) so inbound-fee estimation and Boltz topology scoring see live fleet state instead of a startup-only snapshot.
- Rebalance candidates are classified before pricing as `hive_only`, `hybrid`, or `market_only`. `hive_only` uses the active hive-route pricer with live `hive-*` and `revenue-*` askrene layers, `hybrid` compares that fleet-aware route against the configured market router, and `market_only` stays on the configured router only.
- Coordination hints now seed candidate generation before the active pair cap is applied. `rebalance_recommendations` / `rebalance_campaigns` can materialize coordinated pairs from peer IDs, local SCIDs, or route segments, and may steer policy via `route_policy`, `allow_market_fallback`, `prefer_hive_on_tie`, and `priority_score`.
- `route_segment_leases` are honored during that overlay stage: overlapping foreign leases suppress the candidate with an explicit `lease_conflict` audit reason, while our own leases are allowed through.
- Additional live hint consumers:
  - `fee_elasticity` slightly widens or narrows DTS exploration variance
  - `reputation_score` and `corridor_utilization_bias` modestly bias capacity-planner open scoring
  - `drain_direction` remains askrene/diagnostic only; the fee controller intentionally does not apply it directly
- `revenue-hive-hints-status` reports freshness and signal coverage for the currently cached snapshot.

## More Detail

- Minimal config example: [config/cl-revenue-ops.conf.minimal](config/cl-revenue-ops.conf.minimal)
- Full config example: [config/cl-revenue-ops.conf.full](config/cl-revenue-ops.conf.full)
