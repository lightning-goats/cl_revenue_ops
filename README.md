# cl-revenue-ops

`cl_revenue_ops` is the independent local execution layer for Core Lightning routing economics. It owns fee control, rebalance decisioning/execution, planner/capex, profitability analysis, and budget enforcement. It watches channel economics, adjusts fees, and executes rebalancing while keeping the normal operator surface intentionally small.

It is fully standalone: every decision runs on local evidence only (own forwards, gossip, node state). The former cl-mycelium/cl-hive fleet-hint integration was retired and removed in 2026-07 (see [docs/audit/HIVE_REMOVAL_PLAN.md](docs/audit/HIVE_REMOVAL_PLAN.md)).

## Product Architecture

```text
cl_revenue_ops decides and executes locally.
Core Lightning owns node runtime.
```

`cl_revenue_ops` is the local executor: it decides what is safe, applies local budgets and policy, and uses Core Lightning RPCs for execution when operator controls allow it.

## Economic Governance

Every economic mutation — fee broadcasts (including `htlc_max`), rebalances, channel opens/closes, Boltz swaps, LN+ obligations — flows through one governed core:

```text
canonical EconomicSnapshot -> typed intents -> arbiter -> governor -> execution -> append-only ledger
```

- **Typed intents**: each action is an idempotent envelope carrying its EV, confidence, cost cap, and snapshot reference.
- **Arbiter**: duplicate, stale, and conflicting intents (close-vs-rebalance, duplicate opens, rebalance-vs-structural-swap, contradictory fee changes) are rejected with stable reason codes, both in batch and live via a shared registry.
- **Governor**: fail-closed authorization on every path — pause gate, `authority_level` gate, conflict registry, staleness, then atomic budget reservation. LN+ obligation fulfillment is exempt from pause/authority (an accepted swap is a debt) but never from authorization or the ledger.
- **Ledger**: an append-only event log (`econ_ledger.db`) with hourly self-reconciliation against the reservation store; replay reconstructs state.
- All of it is flag-gated (`econ_*` runtime keys, `revenue-config list-mutable` shows them) with instant per-capability rollback.

The wire contracts are versioned and language-neutral: see [`schemas/`](schemas/) (economic_snapshot, intent, ledger_event, ledger_projection, conformance_case), the conformance corpus under `tests/conformance/scenarios/` (40 scenario classes, validated by the standalone `tools/conformance/validate_fixtures.py`), and [ADR-001](docs/refactor/adr/ADR-001-dts-pid-fee-controller.md) for the fee-controller contract (DTS+PID is authoritative; rails -> rate-limit -> deadband -> cooldown).

## What Operators Need To Know

- This is the executor. It owns local fee execution, rebalance execution, planner/capex decisions, profitability analysis, and budgets.
- All spending decisions are local and bounded by this plugin's controls.
- Live rebalances execute through `RebalanceEngineV2` using native explicit-route execution; route discovery is pinned to the `v3`/askrene router path.
- There is no Sling dependency.
- The normal runtime controls are `paused`, `daily_budget_sats`, fee rails, `authority_level` (what the node MAY do: `observe` < `fees` < `liquidity` < `capital`), `risk_profile` (coherent economic-risk defaults: `preserve`/`conservative`/`balanced`/`growth`/`custom`; preview any change with `revenue-profile-preview` before activating), and planner execution caps. (`fee_market_boundary_*` and `rebalance_min_profit` are deprecated no-ops scheduled for removal after the announced 2026-08-12 compatibility window — see `docs/refactor/phase0/contract-compatibility-policy.md`.)
- The primary operator surfaces are `revenue-status`, `revenue-fee-debug`, and `revenue-rebalance-debug`.
- The normal workflow is decision explainability first, knob tuning second.
- Auto fee bands are enabled by default. Manual policy bands are fallback only when an auto band is not yet available.
- `revenue-policy list|get|find|changes` are diagnostic surfaces. Write actions such as `set` and `delete` remain internal or debug workflows, not the normal operator path.
- Planner closes are recommendation-only by default.
- To allow live close RPCs, set `revenue-ops-planner-execute-closes=true` and `revenue-ops-planner-max-closes-per-cycle` to a positive value.
- Source-heavy drain: `revenue-ops-receivable-ratio-target`/`-floor` define the node-level inbound objective shown in `revenue-status.receivable`. The circular rebalancer always keeps first claim on internal redistribution; only the residual it cannot place (`revenue-rebalance-debug.drain_demand`) may earn the Boltz structural credit, capped by `revenue-ops-boltz-structural-budget-sats` per day (default 0 = off). `revenue-ops-drain-fee-discount-max` optionally biases fees down on stagnant over-local channels (default 0.0 = off).
- Rebalancing and Boltz swaps remain separate tools with separate budgets: rebalancing redistributes at routing-fee cost; loop-outs change the node's aggregate balance at swap+chain cost. Note: the structural envelope is enforced from recorded spend — a swap whose fee fails to record does not deplete it, so keep the envelope small relative to the unified daily budget.

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
- **Inbound gateways** (channels that primarily source traffic for the fleet) receive enhanced protection from closure and are never misclassified as stagnant or zombie based on exit-side metrics alone.
- **Classification** uses total forward count (exit + sourced) for ZOMBIE, STAGNANT, and fleet member protection decisions.
- **Rebalance fee persistence** is msat-native: successful automatic, coordinated, manual, and diagnostic rebalances persist `actual_fee_msat` in `rebalance_history` and `cost_msat` in `rebalance_costs`, with sat fields derived once at write time for compatibility.
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

## Channel Opening Intelligence

The capacity planner uses a multi-strategy candidate pipeline with portfolio-aware governance:

- **Portfolio balance governor** — hard gate at >95% local blocks outbound opens, constrained at 85-95% allows only sink-adjacent or dual-fund
- **Multi-strategy discovery** — winner (proven revenue), demand-flow (gossip heuristics), route-pair, graph, and neighbor strategies
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
- **Growth budget** (`revenue-ops-growth-budget-*`, 5 options) — an optional dynamic uplift on top of `daily_budget_sats`: a fraction of trailing net profit (`growth-budget-earned-fraction`) plus a smaller experiment fraction (`growth-budget-experiment-fraction`), bounded per-window by `growth-budget-max-extra-sats` and by a local hard ceiling (`growth-budget-hard-ceiling-sats`). Disabled by default.
- **Dynamic htlcmax** (`revenue-ops-enable-dynamic-htlcmax` + 3 flow-class pct options) — scales each channel's advertised `htlc_max` by its flow classification (source/sink/balanced), tightest on sinks. As of the 2026-07 econ audit it is also **live-depletion-keyed**: regardless of flow class, `htlc_max` is additionally capped to a fraction of the channel's current spendable balance (clamped to a 10k-sat floor), so a channel that has drained to near-zero local balance stops advertising an `htlc_max` large enough to invite doomed HTLCs. A gossip-churn deadband limits how often the resulting change actually triggers a `setchannel` broadcast.
- **Expansion treasury** (`revenue-ops-expansion-treasury-*`, 7 options) — reverse-swaps excess Lightning balance from over-local channels to on-chain funds via Boltz, to build the on-chain reserve the capacity planner needs for new opens. Runs only when the confirmed on-chain reserve is below `expansion-treasury-onchain-target-sats` by at least `expansion-treasury-min-deficit-sats`; protected/hot channels are excluded from harvesting by default.
- **Drain-bias / receivable-ratio** (`revenue-ops-node-drain-bias-*`, `revenue-ops-receivable-ratio-*`, `revenue-ops-drain-fee-discount-max`, `revenue-ops-boltz-structural-budget-sats`) — the node-level inbound-liquidity objective described above under "What Operators Need To Know": biases fees down on stagnant over-local channels and can earn a capped Boltz structural credit for demand the circular rebalancer can't place internally. All off by default (0 / false).
- **Boltz** (`revenue-ops-boltz-*`, 17 options / 22 RPCs) — Lightning⇄on-chain swap integration behind the balance cycle and expansion treasury. On-chain capacity already committed to an in-flight LN+ swap open (`lnplus_reserved_sats`) is subtracted before Boltz's confirmed-on-chain-sats calculation, so LN+ is effectively a third consumer of the same on-chain reserve pool. Too large to table here; see [docs/audits/CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md](docs/audits/CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md) for the full RPC inventory and `config/cl-revenue-ops.conf.full` for every option.

## LN+ Liquidity Swaps

`cl_revenue_ops` can autonomously join [lightningnetwork.plus](https://lightningnetwork.plus) (LN+) liquidity swaps — a ring of nodes where each participant opens one channel to the next and receives one in return, so an outbound open buys an equal-capacity inbound channel. This is join-only: the plugin applies to swaps other operators posted, it never creates them.

- Each applicable swap is scored with the same EV machinery the capacity planner uses for regular opens (`_calculate_open_ev`, capex budget, ROI hurdle) plus an inbound-liquidity credit and a lockup haircut for the capital committed over the contract term. The swap only proceeds if its EV beats the best regular-open EV by `revenue-ops-lnplus-swap-preference-margin` — ties and near-ties favor the regular open, since a swap locks capital for the contract term and a regular open does not.
- **One swap in flight per node at a time.** No new application is submitted while a prior one is applied, opening, or awaiting completion (checked before every LN+ API call, so an outage or a tripped breaker can never queue up a second commitment).
- Applying to a filled swap slot is an irreversible commitment: once the last slot fills, a 48-hour clock starts to open the assigned channel. The obligations watcher (`revenue-ops-lnplus-watcher-interval`, default hourly) drives every step after application — connect, `fundchannel` (feerate escalates as the deadline approaches), `complete_application`, activation (tags the channel `no_close` so nothing else can close it mid-contract), and finally release + rating once the contract's `ends_at` passes.

**Rank floor (gold or better).** Every participant must clear `revenue-ops-lnplus-min-peer-rank` (default `8`, LN+'s own 1-10 scale where higher is better and 8 = "Gold" in their docs). A missing or zero rank is treated as below the floor (fail-closed), not a pass. This is in addition to the existing positive-ratings and negative-ratio floors.

**Minimum participants.** Dual (2-party) swaps are rejected by default (`revenue-ops-lnplus-min-participants`, default `3`) — LN+'s smallest useful ring for this automation is a triangle. Among multiple qualifying swaps with equal EV, the smaller ring wins the tie-break (triangle beats square beats pentagon); EV still decides primarily.

### Options

| Option | Default | Meaning |
|---|---|---|
| `revenue-ops-lnplus-swaps-enabled` | `true` | Master switch for LN+ automation. |
| `revenue-ops-lnplus-execute-applications` | `true` | `false` = recommendation-only; gates are still evaluated and logged, but no live `create_application` call is made. |
| `revenue-ops-lnplus-swap-preference-margin` | `0.2` | Fraction by which a regular open's EV must beat the best swap's EV to win the slot instead. |
| `revenue-ops-lnplus-max-duration-months` | `3` | Longest contract duration we'll apply to. |
| `revenue-ops-lnplus-min-peer-positive-ratings` | `5` | Minimum LN+ positive-rating floor for every participant in the swap (one under-rated peer vetoes the whole swap). |
| `revenue-ops-lnplus-min-peer-rank` | `8` | Minimum LN+ rank (1-10, higher better; 8 = "Gold") for every participant. Missing/zero rank fails closed. |
| `revenue-ops-lnplus-max-participants` | `4` | Maximum ring size we'll join. |
| `revenue-ops-lnplus-min-participants` | `3` | Minimum ring size we'll join (dual swaps rejected); among equal-EV qualifiers, fewer participants win the tie-break. |
| `revenue-ops-lnplus-apply-feerate-ceiling` | `5000` | No applications while the current opening feerate (perkw) exceeds this. |
| `revenue-ops-lnplus-pending-timeout-days` | `7` | Withdraw an application still stuck `pending` (unfilled) after this many days. |
| `revenue-ops-lnplus-inbound-credit-factor` | `0.5` | Damping applied to the inbound-liquidity EV credit (the value of the channel we receive) — conservative by default since inbound value is harder to realize than outbound. |
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

This repo includes a read-only daily validation pipeline for tracking the production effect of recent fee, capex, and rebalance changes on `lnnode` (single production node since 2026-07-11). It also feeds the running production economic evaluation (`docs/refactor/phase0/production-evaluation-spec.md`, window ending 2026-08-12).

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
| `revenue-status` | Health, operator controls, and latest fee/rebalance decisions |
| `revenue-fee-debug` | Why a fee moved, held, or was clamped |
| `revenue-rebalance-debug` | Why a rebalance was selected, skipped, or blocked |
| `revenue-config get` | Inspect current runtime controls |
| `revenue-config set <key> <value>` | Change one of the supported runtime controls |
| `revenue-profitability [channel_id]` | Per-channel profitability with sourced metrics and flow profiles |
| `revenue-analyze` | Trigger immediate analysis |
| `revenue-wake-all` | Wake the background loops immediately |
| `revenue-planner-candidate-sources` | Diagnostic: candidate pipeline strategy breakdown |
| `revenue-profile-preview [name]` | Diagnostic: risk-profile diff/comparison before activation |
| `revenue-econ-snapshot` | Diagnostic: the canonical economic snapshot (governance core) |
| `revenue-econ-reconcile` | Diagnostic: ledger-vs-reservations reconciliation state |
| `revenue-econ-cycle` | Diagnostic: deterministic shadow economic cycle |

See [docs/audits/CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md](docs/audits/CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md) for the full action/mutation RPC inventory across all subsystems (fees, rebalancing, planner, Boltz, LN+).

### revenue-config: actions and override precedence

`revenue-config` supports four actions: `get [key]` (public controls, or one key with its classification), `set <key> <value>`, `reset <key>`, and `list-mutable` (lists every public runtime key that `set`/`reset` will accept).

- `revenue-config set <key> <value>` writes a DB override for that key. Once set, the override **wins over `setconfig`/config-file changes to the same option on every dynamic-config refresh cycle** — the refresh loop explicitly skips any field with an active DB override rather than stomping it back to the file/`setconfig` value.
- `revenue-config reset <key>` removes the DB override, the escape hatch that lets `setconfig`/config-file values govern the field again (some fields apply immediately; others require a plugin restart to re-adopt the file default — the RPC response says which).
- `revenue-config list-mutable` returns the current set of public runtime keys; only keys in this list can be `set` or `reset` (all others return `"not a public runtime control"`).

## cl_revenue_ops standalone invariant

`cl_revenue_ops` is a fully independent local executor. The former cl-hive/cl-mycelium hint integration was removed entirely in 2026-07 (`docs/audit/HIVE_REMOVAL_PLAN.md`); `tests/test_architecture_guard.py` pins that no hive/fleet coordination code returns.

The read-only operator surfaces `revenue-status`, `revenue-fee-debug`, and `revenue-rebalance-debug` must keep returning JSON with no external plugins present, and read-only surfaces must never call fee, rebalance, planner, Boltz, or CLN mutation RPCs.

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
