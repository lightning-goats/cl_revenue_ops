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
never-call properties. Live Polar verification against the fixed committed
revision is the next release-gate step.

## Remaining high-value work

1. Redeploy the fixed commit and repeat the positive-budget paused-cycle gate.
2. Exercise real Askrene prices below/at/above pair ceilings and positive/
   negative sats-EV without spending on rejected cases.
3. Cover route failure, pending settlement, restart reconciliation, malformed
   evidence, and reservation cleanup.
4. Run longer client-stratified Polar Simulation Designer soaks, then restore
   competitor policies and require a clean final reconciliation.
5. Repeat the smoke matrix with the selected custom recent-CLN image.
