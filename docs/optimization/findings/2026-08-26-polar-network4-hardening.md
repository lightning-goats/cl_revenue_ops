# Polar network 4 hardening findings

## Scope and safety

The disposable `revenue-ops-mixed-client-r4` network used four Core Lightning
nodes, three LND nodes, one Bitcoin Core backend, and twelve public 2,000,000
sat channels. The revenue plugin stayed `dry_run=true`. Action windows used
EXIT-trap cleanup and ended with `paused=true`, `daily_budget_sats=0`, zero
rebalance spend, and zero reservations. No Sling, production, swap, or plugin
channel-open/close action was used.

## Reproducible deployment

`tools/polar_plugin_deploy.py` deployed exact commit
`8d3c460e3bf7c154616916c523e5dc76243c6bdc` into the stock Polar CLN 25.12
container with Python 3.11.2. It recorded the archive and selected source
hashes, started dry-run with a zero budget, persisted pause, and refused a
second invocation because `/opt/cl_revenue_ops` already existed.

The first live deployment found that the committed wrapper lives under
`tools/`, while the runtime expects it at the plugin root. The deployer now
copies that wrapper explicitly and transactionally stops/removes a partial
install on every install or post-install verification failure. Unit coverage
includes target-name confinement, startup rails, archive layout, and cleanup
after a failed safety readback.

## Third 25-ppm fee replicate

All four revenue-node policies were base 1 msat / 10 ppm. Both competitors
advertised base 1 msat / 25 ppm on every relevant edge before the test.

| payer | settled | revenue node | CLN competitor | LND competitor |
|---|---:|---:|---:|---:|
| LND | 10 | 10 | 0 | 0 |
| CLN | 10 | 3 | 6 | 1 |
| total | 20 | 13 (65%) | 6 (30%) | 1 (5%) |

The revenue node earned 6,513 msat in the two 50,000-sat forward windows.
Price advantage improved share but did not make CLN-origin route selection
deterministic. Fee evaluation therefore remains client-stratified; aggregate
share alone is not an acceptance metric.

One LND payment exposed a Polar MCP bridge defect: `pay_invoice` settled the
payment and then returned HTTP 500 because the UI's active-network state was
missing. The harness did not retry, and authoritative forwarding history
proved exactly one settlement before the remaining nine payments were sent.
The harness now classifies any failed payment response as
`unknown_do_not_retry`, writes a credential-free checkpoint, exits nonzero,
and requires client/router history reconciliation.

## Fee-controller gate

One temporarily unpaused dry-run fee cycle observed all new forwards. Two sink
channels were ready; the two source channels remained in their sparse-data
waiting state. The controller proposed a bounded one-ppm raise in debug state.
Readback proved all four advertised policies remained at 10 ppm,
`last_broadcast_fee_ppm` remained 10, pause was restored, and budget remained
zero. This also validates the fresh-channel broadcast-state initialization fix
from commit `8d3c460` against a real current-CLN node.

## Rebalance pause bug

The zero-budget cycle correctly suppressed before selection. An independent
test then set a temporary 1,000-sat budget while leaving `paused=true` and
found that the automatic cycle still entered dry-run candidate planning and
reported a non-safety `hold`. Execution-time governance would still have
blocked payment in live mode, but this violated the operator expectation that
pause stops the entire automatic decision cycle and performed unnecessary DB,
slot, snapshot, and route work.

`EVRebalancer.find_rebalance_candidates()` now exits immediately after its
thread-safe config snapshot when paused. It reports `suppressed / paused`,
marks the result as a safety block, and performs no cleanup, slot, capital,
planner, router, or executor work. The regression pins every one of those
never-call properties.

Live verification passed after a fresh exact deployment of commit
`a38ba9d87f624ec6ac5d5419b406503881f5f4be`. With `paused=true`,
`dry_run=true`, and a temporary 1,000-sat budget, the cycle returned
`suppressed / paused`, `safety_block=true`, zero considered/selected/executed
pairs, and no history. The cleanup trap restored the budget to zero. Final
readbacks found zero spend, zero reservations, no fee changes, no rebalances,
and no econ-ledger divergences. All twelve revenue/competitor gossip policies
were active at the restored base 1 msat / 10 ppm baseline.

## Live route-price and execution matrix

With a 5,000-sat temporary global budget, the candidate's destination capex
allocation produced a 100-sat pair budget and a 50-sat effective quote ceiling.
Changing only the disposable competitor policies produced real Askrene quotes:

| competitor ppm | quoted cost | effective ceiling | result |
|---:|---:|---:|---|
| 10 | 4 sats | 50 sats | held at −3.991026 sats EV |
| 900 | 48 sats | 50 sats | held at −47.991027 sats EV |
| 920 | 49 sats | 50 sats | held at −48.991028 sats EV |
| 940 | 50 sats | 50 sats | held at −49.991028 sats EV |
| 3,000 | no admissible route | 50 sats | Askrene excessive-cost/no-route hold |

All boundary cycles were dry-run, selected zero pairs, executed zero payments,
and restored pause and budget. This proves the ceiling is inclusive while the
sats-EV gate independently rejects fee-admissible but unprofitable routes.

After explicit operator approval for Polar-only spend, every non-revenue edge
was temporarily set to zero fee. The dry-run gate then selected a 49,996-sat
`270x1x0 -> 108x1x1` rebalance with a zero-sat quote and +0.046030 sats EV.
The plugin was restarted live, paused with zero budget, then allowed one
candidate under a temporary 5,000-sat budget. Askrene selected both competitor
families across the initial and alternate routes. Six bounded native attempts
failed with explicit `WIRE_TEMPORARY_CHANNEL_FAILURE` edges because the return
directions lacked liquidity. The history row failed terminally, the 100-sat
reservation was fully released, spend stayed zero, and the next cycle held on
the persisted pair cooldown. Two direct 100,000-sat Polar payments then seeded
the missing competitor return directions. SQLite recorded one
`temporary_channel_failure` with an exact five-minute cooldown; the test waited
for natural expiry rather than deleting or bypassing it.

The next one-candidate automatic cycle succeeded. Its first Askrene route hit
the now-known stale-liquidity edge, the executor excluded that direction, and
the alternate CLN-competitor route settled on attempt two for zero fee. Source
`270x1x0` fell from 1,825,000,201 to 1,775,004,201 msat and destination
`108x1x1` rose from 550,005,912 to 600,001,912 msat: exact opposing
49,996,000-msat deltas. The success row records the 100-sat ceiling, zero actual
fee, `ev_positive`, and destination post-local ratio 0.30. The reservation was
settled/released, the prior durable pair failure was cleared, spend remained
zero, and econ reconciliation found no divergence.

Cleanup restarted the plugin with `dry_run=true`, persisted pause and budget
zero, and restored all 24 directed policies to their exact pre-test settings:
the revenue/CLN/competitor sources at base 1 msat / 10 ppm and LND payer/sink
sources at their native base 1,000 msat / 1 ppm. Every edge was active.

## Remaining high-value work

1. Cover pending settlement, restart reconciliation, malformed
   evidence, and reservation cleanup.
2. Run longer client-stratified Polar Simulation Designer soaks, then restore
   competitor policies and require a clean final reconciliation.
3. Repeat the smoke matrix with the selected custom recent-CLN image.
