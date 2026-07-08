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
- The normal runtime controls are `paused`, `daily_budget_sats`, fee rails, and planner execution caps. (`fee_market_boundary_*` and `rebalance_min_profit` are deprecated no-ops kept only for config-file compatibility — see Day-1 Operator Workflow.)
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

## Additional Runtime Subsystems

These subsystems are off by default and each has more knobs than fit in a table here — see [config/cl-revenue-ops.conf.full](config/cl-revenue-ops.conf.full) for every option, default, and comment.

- **Hot-channel protection** (`revenue-ops-hot-channel-protection-*`, 8 options) — gives fast-draining, high-profit channels a wider rebalance budget and shorter cooldown than normal channels get, so they don't starve mid-burst. Gated on minimum velocity and marginal ROI, capped by a fraction of the channel's own daily contribution, with an operator override-peer list to force protection regardless of the velocity/ROI gate.
- **Growth budget** (`revenue-ops-growth-budget-*`, 5 options) — an optional dynamic uplift on top of `daily_budget_sats`: a fraction of trailing net profit (`growth-budget-earned-fraction`) plus a smaller fleet-learning experiment fraction (`growth-budget-experiment-fraction`), bounded per-window by `growth-budget-max-extra-sats` and by a local hard ceiling (`growth-budget-hard-ceiling-sats`) that fleet hints cannot exceed. Disabled by default.
- **Dynamic htlcmax** (`revenue-ops-enable-dynamic-htlcmax` + 3 flow-class pct options) — scales each channel's advertised `htlc_max` by its flow classification (source/sink/balanced), tightest on sinks. As of the 2026-07 econ audit it is also **live-depletion-keyed**: regardless of flow class, `htlc_max` is additionally capped to a fraction of the channel's current spendable balance (clamped to a 10k-sat floor), so a channel that has drained to near-zero local balance stops advertising an `htlc_max` large enough to invite doomed HTLCs. A gossip-churn deadband limits how often the resulting change actually triggers a `setchannel` broadcast.
- **Expansion treasury** (`revenue-ops-expansion-treasury-*`, 7 options) — reverse-swaps excess Lightning balance from over-local channels to on-chain funds via Boltz, to build the on-chain reserve the capacity planner needs for new opens. Runs only when the confirmed on-chain reserve is below `expansion-treasury-onchain-target-sats` by at least `expansion-treasury-min-deficit-sats`; protected/hot channels are excluded from harvesting by default.
- **Drain-bias / receivable-ratio** (`revenue-ops-node-drain-bias-*`, `revenue-ops-receivable-ratio-*`, `revenue-ops-drain-fee-discount-max`, `revenue-ops-boltz-structural-budget-sats`) — the node-level inbound-liquidity objective described above under "What Operators Need To Know": biases fees down on stagnant over-local channels and can earn a capped Boltz structural credit for demand the circular rebalancer can't place internally. All off by default (0 / false).
- **Boltz** (`revenue-ops-boltz-*`, 17 options / 22 RPCs) — Lightning⇄on-chain swap integration behind the balance cycle and expansion treasury. On-chain capacity already committed to an in-flight LN+ swap open (`lnplus_reserved_sats`) is subtracted before Boltz's confirmed-on-chain-sats calculation, so LN+ is effectively a third consumer of the same on-chain reserve pool. Too large to table here; see [docs/audits/CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md](docs/audits/CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md) for the full RPC inventory and `config/cl-revenue-ops.conf.full` for every option.

## LN+ Liquidity Swaps

`cl_revenue_ops` can autonomously join [lightningnetwork.plus](https://lightningnetwork.plus) (LN+) liquidity swaps — a ring of nodes where each participant opens one channel to the next and receives one in return, so an outbound open buys an equal-capacity inbound channel. This is join-only: the plugin applies to swaps other operators posted, it never creates them.

- Each applicable swap is scored with the same EV machinery the capacity planner uses for regular opens (`_calculate_open_ev`, capex budget, ROI hurdle) plus an inbound-liquidity credit and a lockup haircut for the capital committed over the contract term. The swap only proceeds if its EV beats the best regular-open EV by `revenue-ops-lnplus-swap-preference-margin` — ties and near-ties favor the regular open, since a swap locks capital for the contract term and a regular open does not.
- **One swap in flight per node at a time.** No new application is submitted while a prior one is applied, opening, or awaiting completion (checked before every LN+ API call, so an outage or a tripped breaker can never queue up a second commitment).
- Applying to a filled swap slot is an irreversible commitment: once the last slot fills, a 48-hour clock starts to open the assigned channel. The obligations watcher (`revenue-ops-lnplus-watcher-interval`, default hourly) drives every step after application — connect, `fundchannel` (feerate escalates as the deadline approaches), `complete_application`, activation (tags the channel `no_close` so nothing else can close it mid-contract), and finally release + rating once the contract's `ends_at` passes.

**Fleet participation (2026-07-08 Revision 2 — operator-directed).** Swaps that include another fleet or hive node are allowed and welcome — they support intrafleet rebalancing, and only the own-node check remains (we still cannot join a swap we're already in). A fleet/hive participant (identity = `revenue-ops-lnplus-fleet-pubkeys` CSV union the live hive-membership check) is fully TRUSTED: it skips every LN+ reputation check — positive-ratings floor, negative-ratio ceiling, and the rank floor below — and counts as reliability 1.0 in the EV inbound credit. If every visible counterparty in a swap is a fleet/hive node, the swap's reliability is 1.0 outright with no Tor discount (trust overrides transport); a mixed ring still computes reliability from its non-fleet counterparties only.

**Rank floor (gold or better).** Every non-fleet participant must clear `revenue-ops-lnplus-min-peer-rank` (default `8`, LN+'s own 1-10 scale where higher is better and 8 = "Gold" in their docs). A missing or zero rank is treated as below the floor (fail-closed), not a pass. This is in addition to the existing positive-ratings and negative-ratio floors.

**Minimum participants.** Dual (2-party) swaps are rejected by default (`revenue-ops-lnplus-min-participants`, default `3`) — LN+'s smallest useful ring for this automation is a triangle. Among multiple qualifying swaps with equal EV, the smaller ring wins the tie-break (triangle beats square beats pentagon); EV still decides primarily.

**cl-mycelium hints (advisory, bounded).** When cl-hive/cl-mycelium is present, `cl_revenue_ops` consumes an optional `lnplus_swap_hints` section from the same hive-hints envelope the rest of the fleet-awareness machinery already reads. Each entry names a peer pubkey, an `action` (`prefer` / `avoid` / `allow_duplicate`), an `ev_multiplier` (clamped to `[0.8, 1.5]`, default `1.0`), and a `topology_gain` (clamped to `[0.0, 1.0]`). Hints only ever **bias** the EV of the assigned outbound peer (multiplicatively) and, for `allow_duplicate`, skip the duplicate-peer veto (a swap whose assigned outbound peer we already have a channel to is normally rejected) — they never bypass any other safety gate, and an `avoid` hint always dampens the EV to at most `×0.8`. The feature is fully functional without cl-mycelium: an absent adapter, missing section, or malformed entry is treated as fully neutral.

### Options

| Option | Default | Meaning |
|---|---|---|
| `revenue-ops-lnplus-swaps-enabled` | `true` | Master switch for LN+ automation. |
| `revenue-ops-lnplus-execute-applications` | `true` | `false` = recommendation-only; gates are still evaluated and logged, but no live `create_application` call is made. |
| `revenue-ops-lnplus-swap-preference-margin` | `0.2` | Fraction by which a regular open's EV must beat the best swap's EV to win the slot instead. |
| `revenue-ops-lnplus-max-duration-months` | `3` | Longest contract duration we'll apply to. |
| `revenue-ops-lnplus-min-peer-positive-ratings` | `5` | Minimum LN+ positive-rating floor for every non-fleet participant in the swap (one under-rated peer vetoes the whole swap). |
| `revenue-ops-lnplus-min-peer-rank` | `8` | Minimum LN+ rank (1-10, higher better; 8 = "Gold") for every non-fleet participant. Missing/zero rank fails closed. |
| `revenue-ops-lnplus-max-participants` | `4` | Maximum ring size we'll join. |
| `revenue-ops-lnplus-min-participants` | `3` | Minimum ring size we'll join (dual swaps rejected); among equal-EV qualifiers, fewer participants win the tie-break. |
| `revenue-ops-lnplus-apply-feerate-ceiling` | `5000` | No applications while the current opening feerate (perkw) exceeds this. |
| `revenue-ops-lnplus-pending-timeout-days` | `7` | Withdraw an application still stuck `pending` (unfilled) after this many days. |
| `revenue-ops-lnplus-inbound-credit-factor` | `0.5` | Damping applied to the inbound-liquidity EV credit (the value of the channel we receive) — conservative by default since inbound value is harder to realize than outbound. |
| `revenue-ops-lnplus-fleet-pubkeys` | `` (empty) | Comma-separated pubkeys treated as trusted fleet members — exempt from LN+ reputation checks (hive members are detected automatically without needing to be listed here). |
| `revenue-ops-lnplus-watcher-interval` | `3600` | Obligations watcher poll interval, in seconds. |

### RPCs

| Command | Use |
|---|---|
| `revenue-lnplus-status` | Circuit-breaker state, in-flight swap, and active/recently-ended contracts. |
| `revenue-lnplus-breaker-clear` | Operator acknowledgment that clears a tripped breaker so new applications can resume. |
| `revenue-lnplus-abandon <swap_id>` | Emergency abandon of an in-flight obligation — marks the local row failed and trips the breaker, since abandoning a commitment is a defection on our side and must never happen silently. |
| `revenue-lnplus-backfill` | Operator remedy that adopts pre-existing LN+ swaps (applied/opened/settled manually on the LN+ website, before or after this automation existed) into the local ledger. Idempotent — existing rows are never touched, so it is safe to run repeatedly. |

### Circuit breaker

The breaker is a one-strike mechanism: a missed 48-hour open deadline, or the local ledger diverging from what LN+ reports for an in-flight swap, trips it immediately and blocks all new applications. Obligations already in flight (an open in progress, an active contract) are still driven to completion — the breaker only stops new commitments, it does not abandon existing ones. Clearing it is always an explicit operator action via `revenue-lnplus-breaker-clear`; it never clears itself.

Note the distinction between disabling and abandoning: setting `revenue-ops-lnplus-swaps-enabled=false` stops new applications, but any swap already applied, opening, or active is still honored to completion (fundchannel executed, contract protected with `no_close`, rated and released at term end) — disabling is not the same as walking away from a commitment. Use `revenue-lnplus-abandon` only as a last resort, since it deliberately defects on an LN+ commitment and will draw a negative rating.

### Contract lifecycle and the 3-month cap

Swap contracts are capped at `revenue-ops-lnplus-max-duration-months` (default 3, configurable 1-3 (hard cap: never lock capital beyond a quarter)). While a contract is active the outbound channel is tagged `no_close` so the planner's capital-recycling logic cannot touch it. Once the contract's `ends_at` passes, the watcher rates the counterparty (positive if their channel to us is still open, negative — plus an ignore — if they defected), removes the `no_close` tag, and marks the swap `ended`. From that point the channel reverts to normal planner management like any other channel: it is eligible for fee optimization, rebalancing, and capital recycling on the same footing as a channel opened the regular way. A well-performing swap channel is expected to naturally stay open past the contract; the cap only bounds the *protected, locked* period.

## Architecture

```text
pair selection / fee decisions
    ↓
v3 router (askrene + mycelium/hive layers)
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
2. Set only the safety rails you actually want to constrain: `paused`, `daily_budget_sats`, `min_fee_ppm`, `max_fee_ppm`, and `fee_profile`. (The `fee_market_boundary_*` controls and `rebalance_min_profit` are deprecated no-ops kept only for config compatibility; they have no effect — minimum-profit gating is now enforced by the sats-EV gate and `rebalance_hold_margin`.)
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

See [docs/audits/CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md](docs/audits/CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md) for the full action/mutation RPC inventory across all subsystems (fees, rebalancing, planner, Boltz, LN+).

### revenue-config: actions and override precedence

`revenue-config` supports four actions: `get [key]` (public controls, or one key with its classification), `set <key> <value>`, `reset <key>`, and `list-mutable` (lists every public runtime key that `set`/`reset` will accept).

- `revenue-config set <key> <value>` writes a DB override for that key. Once set, the override **wins over `setconfig`/config-file changes to the same option on every dynamic-config refresh cycle** — the refresh loop explicitly skips any field with an active DB override rather than stomping it back to the file/`setconfig` value.
- `revenue-config reset <key>` removes the DB override, the escape hatch that lets `setconfig`/config-file values govern the field again (some fields apply immediately; others require a plugin restart to re-adopt the file default — the RPC response says which).
- `revenue-config list-mutable` returns the current set of public runtime keys; only keys in this list can be `set` or `reset` (all others return `"not a public runtime control"`).

## Zero-Fee Hive Corridor

**Operator strategy**: all hive-internal channels are zero fee (0 base msat / 0 ppm), public and announced. Third parties chaining our free internal hops into cheap end-to-end routes is *desired*, not a leak: the fleet is the cheap corridor, revenue is captured at the edge (non-hive) channels, and external flow transiting the mesh performs free intrafleet rebalancing as a side effect.

- The hive-member zero-fee gate (`FeeController._check_hive_member_fee` / `_hive_member_zero_fee_active`, `modules/fee_controller.py`) dominates the fee cycle for any peer with active hive membership: it forces 0 ppm and `base_fee_msat_override=0` with `enforce_limits=False`, ahead of DTS/PID pricing. An explicit operator STATIC policy still wins over the automatic gate (checked earlier in the cycle), and a `set-fee`/initial-fee call with `force=true` can pin a non-zero fee on a hive peer if an operator deliberately chooses to.
- `revenue-ops-fee-ppm-intra-fleet` (default `0`) and `revenue-ops-base-fee-intra-fleet` (default `0`) are the config-level expression of the same policy for any code path that falls back to configured base/ppm values rather than the hard gate.
- **Durability through hint staleness**: the zero-fee gate is normally contingent on live hive-hint freshness (`hive-fleet` datastore snapshot / `hive-export-hints`). To avoid repricing away from 0 ppm on every transient hint hiccup or cl-hive/cl-mycelium restart, the plugin persists each peer's last-confirmed hive membership (`hive_member_confirmations` table) and holds zero-fee for `revenue-ops-hive-zero-fee-stale-grace` seconds (default 604800 = 7 days) past the last confirmation before releasing the peer to normal dynamic repricing. Fresh hints that positively say a peer is *not* a member always win immediately -- the grace period only covers genuinely stale/unavailable membership data, never overrides a current "not a member" signal.
- **Corridor utilization instrumentation**: each settled forward is classified by hive membership of its in/out peers into `internal_transit` (hive→hive), `edge_in` (external→hive), `edge_out` (hive→external), or `external` (external→external), and aggregated daily (`corridor_flow_daily` table). `revenue-dashboard` surfaces a 7-day rollup under `mycelial_corridor` (counts, volume, fee split edge-vs-internal). This is a utilization/success metric only -- it carries no thresholds, warnings, or revocation logic.

## cl-mycelium Hints

`cl_revenue_ops` consumes `cl-mycelium` fleet hints only through `modules/hive_hints.py` (`HiveHintAdapter`). The adapter name, `hive-*` RPC names, and `["hive", "hints"]` datastore key remain the stable compatibility contract documented in [docs/contracts/HIVE_HINTS_CONTRACT.md](docs/contracts/HIVE_HINTS_CONTRACT.md).

- Transport order is datastore first: read CLN datastore key `["hive", "hints"]`, then fall back to `hive-export-hints` only if the datastore payload is missing, stale, or invalid.
- Missing, stale, malformed, or unavailable hints degrade to neutral local behavior; they do not bypass fee, rebalance, planner, budget, or policy safety rails.
- Once per fee cycle, `cl_revenue_ops` polls the hint snapshot and refreshes the shared `HiveRouter` compatibility layer (`hive-fleet` layer detection, fleet balance cache, route cache clear) so inbound-fee estimation and Boltz topology scoring see live fleet state instead of a startup-only snapshot.
- Rebalance candidates are classified before pricing as `hive_only`, `hybrid`, or `market_only`. `hive_only` uses the active cl-mycelium-aware route pricer with live `hive-*` and `revenue-*` askrene layers, `hybrid` compares that fleet-aware route against the configured market router, and `market_only` stays on the configured router only.
- Coordination hints now seed candidate generation before the active pair cap is applied. `rebalance_recommendations` / `rebalance_campaigns` can materialize coordinated pairs from peer IDs, local SCIDs, or route segments, and may steer policy via `route_policy`, `allow_market_fallback`, `prefer_hive_on_tie`, and `priority_score`.
- `route_segment_leases` are honored during that overlay stage: overlapping foreign leases suppress the candidate with an explicit `lease_conflict` audit reason, while our own leases are allowed through.
- Additional live hint consumers:
  - `fee_elasticity` arms the DTS exploration multiplier, hard-clamped to `[0.75, 2.0]` (`EXPLORATION_BOOST_MIN/MAX` in `modules/fee_controller.py`)
  - `fleet_fee_prior` / `optimal_fee_estimate_ppm` seed a fleet fee prior, clamped to `[1, 10000]` ppm (`MAX_FLEET_FEE_PRIOR_PPM` in `modules/hive_hints.py`); out-of-range values neutralize to no hint. These two are separate hive-influence channels with their own rails and are NOT the ±10% bounded fee-bias clamp
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
