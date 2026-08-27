# Polar test, bug-fix, and hardening program

## Objective

Turn the mixed-client smoke lab into a repeatable acceptance program for fee
setting, rebalancing, accounting, failure recovery, and reporting. Every phase
runs locally in Polar, uses one sequential network in the existing app, and
ends paused with a zero daily budget and restored peer policies.

## Safety invariants

- Only containers matching `polar-n<id>-revenue-node` may receive the plugin.
- A fresh deployment starts `dry_run=true`, `daily_budget_sats=0`, then
  persists `paused=true` before any experiment.
- Fee or rebalance action RPCs are permitted only for an explicit bounded
  phase. Every action has a before/after policy, balance, budget, reservation,
  history, and reconciliation readback.
- No production node, Sling, swap, channel-open/close plugin action, hive, or
  mycelium integration is in scope.
- Failed assertions stop the phase; cleanup restores pause, zero budget, and
  competitor policy before diagnosis continues.

## Phase 1 — reproducible fresh-node deployment

Status: **implemented and exercised on network 4**.

Use `tools/polar_plugin_deploy.py` in plan mode first, then `--apply` against a
fresh revenue container. The tool archives an exact commit, refuses non-Polar
or non-revenue targets, installs Python in the stock CLN image, creates an
isolated venv from `requirements.txt`, starts with dry-run/zero-budget rails,
persists pause, and emits source hashes plus Python/package versions.

Exit criteria:

- source commit and deployed hashes agree;
- plugin reports running, paused, dry-run, daily budget zero;
- no fee/rebalance history or active reservation exists;
- a second invocation refuses to overwrite the existing `/opt` deployment.

## Phase 2 — fee-setting acceptance

Status: **third 25-ppm replicate complete; dry-run mutation gate passed**.

For every candidate competitor fee, use a fresh network and separate LND/CLN
forward windows. Verify gossip before traffic, reset only the intended payer's
routing state, and record settled/failed totals, route attribution, volume,
earned msat, and balance drift. Run exactly one dry-run controller cycle before
any live fee cycle and require tracked broadcast state to equal live gossip.

Immediate execution target: create network 4 and complete the third independent
25-ppm run. If its assertions pass, 25 ppm has the minimum three-network count,
but it is still not preferred unless net revenue and payment reliability beat
the other candidates.

## Phase 3 — rebalance fail-closed matrix

Status: **zero-budget and pause gates passed; pause defect fixed and verified;
real Askrene below/at/above-ceiling and negative/positive-EV cases passed**.
The first live positive-EV attempt proved route-failure cleanup and cooldown;
the seeded retry proved alternate-route success, exact balance movement,
reservation cleanup, and durable-failure clearing. Pending-settlement and
restart cases remain.

For both payer families and real quoted routes, cover:

1. price below, at, and above the pair ceiling;
2. positive and negative sats-EV around the zero hold margin;
3. stale or missing profitability/flow/route evidence;
4. malformed RPC/datastore evidence;
5. first-route failure with alternate success;
6. pending settlement resolving success and failure;
7. plugin and node restart with active or recently resolved state.

Every non-execution must have an explicit reason and zero reservation/spend.
Every execution must have one reservation, actual fee within the ceiling,
atomic settlement, released remainder, fresh reporting, and clean reconcile.

## Phase 4 — automation and soak

Status: **deterministic MCP traffic driver hardened against ambiguous bridge
failures; long-running simulation soak remains**.

Run deterministic MCP bursts before and after longer Polar Simulation Designer
activity windows. Vary payment sizes, direction, cadence, and client family;
include HTLC-max pressure and opposing flow. Add a machine-readable scorecard
that rejects cache-dependent or one-client-only improvements.

## Phase 5 — compatibility and release gate

Status: **current bundled CLN campaign active; custom-image compatibility and
release soak remain**.

Repeat the smoke subset with a custom current-CLN image after the bundled CLN
25.12 campaign is stable. A change may advance only when focused tests and the
full Python 3.11 suite pass, deployed source hashes match, all Polar cleanup
rails read back correctly, and compact findings are committed.
