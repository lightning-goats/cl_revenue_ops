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

The three original follow-ups were completed in the continuation below.

## Pending settlement and restart hardening

Code review found two coupled restart hazards. First, a malformed but
non-exceptional `listsendpays` payload was normalized to an empty payment list,
which terminally failed the row and released its reservation despite ambiguous
evidence. Second, when the initial history insert failed, the recovery row used
a new numeric id while the actual reservation remained under a synthetic id.
The stale sweep and reconciliation path assumed those ids were equal, so the
real hold could be released or stranded.

Settlement evidence now requires an object response, a list of object payment
entries, known statuses, and parseable nonnegative complete-payment amounts
with `amount_sent_msat >= amount_msat`. Malformed evidence logs a warning and
leaves the row and reservation untouched. `rebalance_history` now has an
additive nullable `reservation_id`; ordinary rows retain the legacy id fallback,
while recovered rows persist the synthetic link. Both legacy and unified stale
cleanup queries protect that explicit link.

Focused recovery, reservation, atomic-settlement, replay, and reconciliation
suites passed 99 tests; the broader rebalance/database/operator suite passed
251 tests. The full Python 3.11 suite at that commit passed 3,749 tests, with
five expected skips and two expected xfails.

Exact commit `9a79eac680d2020d93840d5902415c255b9d48c2` was deployed over the
existing network-4 database. Two ledger-backed reservations were linked to
five-hour-old pending rows: one used the already-complete 49,996-sat Polar
payment and one used a nonexistent hash. Both remained active across plugin
restart and the stale cleanup boundary. One bounded live cycle had no
executable candidates; its reconciliation sweep changed the rows to
`success/spent` at zero fee and `failed/released`, respectively. The final
econ reconcile reported zero divergences. The plugin was restarted dry-run,
paused, with budget zero.

## Mixed-client MCP soak

`tools/polar_soak_scorecard.py` adds read-only before/after router snapshots
and retained-module health checks. Its score rejects ambiguous payments,
one-client or one-direction coverage, fewer than three payment sizes,
incomplete router attribution, active reservations, dirty reconciliation, or
unsafe final controls.

The MCP driver settled all 60 payments: 30 per client family, with 15 forward
and 15 reverse payments per family. The nominal 5,000, 15,000, and 35,000-sat
matrix plus explicit return-liquidity buffers produced six observed amounts.

| payer family | revenue node | CLN competitor | LND competitor |
|---|---:|---:|---:|
| LND | 30 (100%) | 0 | 0 |
| CLN | 7 (23.3%) | 15 (50.0%) | 8 (26.7%) |

The revenue node earned 9,280 msat during the LND phase and 2,457 msat during
the CLN phase. The split at equal policies confirms that fee tuning must remain
client-stratified; aggregate route share would conceal materially different
client behavior. Fee, rebalance, profitability, budget, status, and economic
reconciliation surfaces all remained readable. Final state was paused,
dry-run, budget zero, zero active reservations, and zero divergences.

Polar v4's MCP catalog exposes invoice/payment and node/network operations but
does not expose Simulation Designer activity rules. The completed soak
therefore used deterministic MCP-issued Polar payments rather than UI-only
rules, preserving exact client, direction, size, cadence, and ambiguity
evidence.

## Recent Core Lightning compatibility

Polar's local catalog ends at CLN 25.12. The latest official stable release at
test time was Core Lightning v26.06.6, so a fresh official-image node was
attached temporarily to network 4 without cloning the funded revenue identity.
Exact plugin commit `9a79eac` started under Python 3.11 with dry-run, budget
zero, and persisted pause. `revenue-status`, config, fee debug, rebalance debug,
profitability, total-cost/capex budget, and econ reconcile all returned valid
empty-node results. A paused rebalance-cycle action returned
`suppressed/paused`, `safety_block=true`, and zero candidates/executions. The
temporary v26.06.6 container and source archive were removed afterward.

The higher-confidence continuation repeated that check with a funded identity
and exact commit `4c6ec87`. Two 1,000,000-sat payer channels and two
1,000,000-sat sink channels connected the temporary node to both client
families. With a temporary 1-ppm outbound policy, a three-size Polar MCP matrix
settled 60/60 payments. The v26 router carried 49 payments (30 to the LND sink
and 19 to the CLN sink), forwarded 885,000 sats, earned 934 msat, and recorded
no failed forward.

The plugin observed all four funded channels. A dry-run fee cycle evaluated two
ready sink channels and proposed 51 ppm from the temporary 1-ppm policy without
changing gossip; both source channels correctly waited for their own forwarding
sample. Paused fee and rebalance cycles both returned a safety-blocked
`suppressed/paused` result. Profitability classified two channels profitable
and two underwater, the total-cost surface attributed 321 sats to channel-open
cost, rebalance spend/reservations remained zero, and economic reconciliation
was clean. All module and safety readbacks passed again after a dynamic plugin
restart. The four channels were cooperatively closed and confirmed before the
temporary container, isolated volume, and archive were removed. The original
network remained traffic-ready with its target paused, dry-run, budget zero,
four base-1/10-ppm policies, and no reconciliation divergence.

## One-hour endurance and fee sweep

`tools/polar_endurance_campaign.py` ran a resumable 60-minute active-controller
campaign after a 600-payment, client-stratified fee sweep. Every one of the 600
sweep and 240 endurance payments settled. Across endurance, the revenue node
and LND competitor each forwarded 120 payments; the revenue node earned
116,062 msat and the LND competitor 22,120 msat. The 60 fee cycles produced 18
raise, two lower, and 40 interval-suppressed decisions, with 36 channel
adjustments. Every rebalance cycle considered current economics and held with
`below_hold_margin`; no candidate was selected, no payment executed, and no
reservation or rebalance spend was created. That is the expected economic
outcome, not missing module coverage.

| target ppm | LND route share / earned | CLN route share / earned |
|---:|---:|---:|
| 5 | 100% / 5,560 msat | 33.3% / 1,720 msat |
| 10 | 100% / 11,060 msat | 31.7% / 3,469 msat |
| 25 | 100% / 27,560 msat | 18.3% / 4,636 msat |
| 50 | 100% / 55,060 msat | 26.7% / 16,016 msat |
| 10 replicate | 100% / 11,060 msat | 0% / 0 msat |

Within this graph, 50 ppm maximized observed revenue without reducing payment
success. It is not a production recommendation: the CLN 10-ppm replicate moved
from 31.7% share to zero with no fee change, demonstrating route-memory and
liquidity variance. Promotion still requires fresh-network replication and
net-revenue comparison by client.

## Economic-floor defect found during endurance

One live cycle began from a temporary 10-ppm policy with a computed economic
floor above it. Normal DTS+PID blending produced 19 ppm and applied it, so the
gradual transition undercut the chain/rebalance-cost floor. The controller now
reapplies the computed hard floor and ceiling after normal blending/damping.
Sustained-congestion damping remains unchanged because its congestion target is
not the same economic rail. A focused regression starts at 10 ppm with an
80-ppm rebalance floor and requires the applied fee to remain at least 80 ppm.

The exact fix was hot-validated in endurance epoch 21: DTS selected 59 ppm,
PID produced 62, blending produced 33, and the final applied value was clamped
to the live 55-ppm economic floor. The broader scoped suite passed 178 tests;
the full Python 3.12.13 hash-pinned suite passed 3,773 tests with five expected
skips and two expected xfails.

## Runtime recovery and automation lessons

Docker had retained container state while their processes were gone. The
backend's block files survived but its index/chainstate did not, so an isolated
Bitcoin Core 30.0 reindex recovered height 317 before the original container
was restarted. A fresh block through Polar MCP restored LND tip freshness.
Docker IP reassignment also exposed persisted ephemeral peer addresses; every
LND edge was reconnected by pubkey and stable Docker DNS. The harness now
rejects traffic until all expected channel endpoints are active and all LND
nodes are chain-synced, and offers explicit preflight mining for an aged
regtest tip.

## Remaining optional endurance work

No correctness or compatibility blocker remains from this program. An
overnight or multi-day run can still improve confidence in long-horizon
controller convergence and rebalance opportunity arrival. It should use the
same resumable runner in bounded chunks; Polar still does not expose Simulation
Designer rules through MCP.
