# Polar mixed-client fee and rebalance smoke result

**Disposition:** useful local smoke evidence; **not a production tuning
recommendation** and not the three-fresh-network acceptance run.

## Run scope

The Polar v4.0.0 lab used Core Lightning 25.12 for the target and CLN client
nodes, LND 0.20.0-beta for the second client and routing competitor, twelve
public 2,000,000-sat channels, and Bitcoin Core 30 on regtest. Eclair was
excluded after its bundled 0.13.1 image reproducibly crashed in native
`libsecp256k1` during funding. The target ran `cl-revenue-ops` 3.0.0 with the
economic governor flags enabled. No production node, cl-hive, cl-mycelium,
Sling, swap, open, or close action was in scope.

## Measured outcomes

| Check | Outcome |
| --- | --- |
| Baseline traffic | 4/4 payments settled. |
| Competitors at 200 ppm | 10/10 settled; the target forwarded 7/10. |
| Competitors at 25 ppm | 10/10 settled; the target forwarded 6/10. |
| Fee crossover by client | With both competitors at 0 ppm, the LND lane sent 10/10 through the CLN competitor; the CLN lane split 1 target / 3 CLN competitor / 6 LND competitor. At 25 ppm, the LND lane sent 10/10 through the 14-ppm target while the CLN lane still split 2 target / 3 CLN competitor / 5 LND competitor. This is useful client/cache divergence evidence, not a fee-only attribution. |
| Governed fee cycle | Four policies changed: sink edges 10→14 ppm and payer edges 10→64/74 ppm; reason code `dts_pid_sample`; dynamic HTLC maxima were applied. |
| Paused and zero-budget gates | Both suppressed actions with zero rebalance execution and zero reservation. |
| Priced automatic rebalance | Initially held at −16.145 sats EV with a 15-sat quote and at −112.178 sats with a 112-sat quote. With zero-fee competitors it exposed a directional-activity bug: gross helpful-flow counters imposed an 8.302-sat penalty even though larger opposing flow made both legs net harmful. After fixing the planner to pass only net helpful activity, the same pair scored +5.425 sats before a route failure and +4.425 sats after its failure penalty, so it was selected without changing the zero-sat hold margin. |
| Failure path | A manual 300,000-sat, 4-sat-cap attempt hit `WIRE_TEMPORARY_CHANNEL_FAILURE` on a return edge with zero directional liquidity; it recorded a failure and spent zero. |
| Seeded execution | Eight competitor-routed 50,000-sat payments gave the failed return edge 400,000 sats local. The same manual rebalance then settled 300,000 sats for 3,301 msat, rendered as 4 sats. |
| Automatic failover and settlement | A first automatic 439,994-sat attempt correctly released its reservation after both visible return routes lacked directional liquidity. Ten bounded 50,000-sat LND-lane payments then seeded 500,000 sats through the CLN competitor. On retry, the LND route again failed, native routing excluded that edge, the CLN route settled 439,994 sats for 3,440 msat in two attempts, and the plugin paused immediately afterward. |
| Pending settlement — late success | A container-only fault injector raised a transport timeout immediately after real `sendpay` submission. The 400,000-sat CLN-competitor payment was parked as `pending_settlement` with its real payment hash; actual spend stayed at 318 sats while the full 400-sat pair budget remained reserved. `listsendpays` proved completion for 3,401 msat. The normal reconciliation sweep atomically marked success, booked 4 conservative sats, and released the 396-sat remainder. A second fixed-code run reproduced the same result at history row 10. |
| Pending settlement — confirmed failure | The same post-`sendpay` timeout was applied to a deliberately dry LND return route. The real payment first held a 400-sat reservation as `pending_settlement`; `listsendpays` then reported `failed`, target balances did not move, and reconciliation marked `payment_pending_resolved_failed`, released all 400 sats, and left actual spend unchanged at 326 sats. |
| Late-success cooldown bug | The first live late-success sweep exposed that it cleared only the in-memory pair failure, not the persisted `payment_pending_timeout` cooldown. After restart, an emergency 90/10 pair was incorrectly blocked for one hour even though the payment had succeeded. The sweep now clears the durable pair failure through the same database primitive as the synchronous-success path. A repeated late-success run followed by fresh depletion selected the pair immediately, proving the cooldown was gone. |
| Liquidity result | After the manual and automatic settlements, source `216x1x0` landed at 1,400,042 sats local and destination `108x1x1` at 599,999 sats: effectively the planner's 70/30 post-state. |
| Fresh accounting | Final budget state was 318 sats total = 310 open + 8 conservative whole-sat rebalance accounting, zero reservations, and 682/1,000 sats remaining. Exact aggregate rebalance cost was 6,741 msat, rendered as 7 sats by profitability/health. The five recorded jobs contained two successes; failed/governor-blocked paths spent zero. |
| Reconciliation | `revenue-econ-reconcile apply=false` found no divergences; the single governed fee cycle had complete intent evidence. Manual rebalances do not create an automated governor intent to match. |
| Reporting surface | `revenue-report summary` was only returning explicit peer-policy counts despite documenting node P&L, active channels, and warnings. The additive fix preserves `policies` and now returns canonical dashboard financials, the 30-day period, warnings, bleeder count, and live channel-state counts. Live readback reported four channels, 51 sats revenue, 7 sats exact rebalance opex, and 44 sats operating profit. |
| Regression suite | Focused rebalance suites: 519 passed. Planner plus operator/reporting regressions: 94 passed. Pending/atomic/engine suites after the late-success cooldown fix: 128 passed. Final full suite after all continuation changes: 3,720 passed, 5 environment-dependent skips, and 2 intentional expected failures. |

## Module disposition

- **Fee controller, gossip-visible application, mixed-client forwarding, flow
  archive, profitability refresh, capex tiers, route pricing, sats-EV gates,
  failure recording, native execution, msat settlement persistence, and
  restart recovery:** smoke-pass.
- **Automatic rebalance selection:** conservative and explainable in this run.
  Do not lower the zero-sat hold margin merely to make a lab rebalance fire.
  The planner now subtracts opposing activity before assigning the
  source-outbound/destination-inbound healing penalty; gross flow in the
  helpful direction must not mask a larger net-worsening flow. Validate
  directional return liquidity before every execution assertion.
- **Immediate budget reporting and runtime budget adoption:** the smoke run
  exposed and the follow-up patch fixed a reporting-cache bug. The first
  post-success `revenue-budget` read had returned a cached
  snapshot from before settlement (`rebalance=0`, remaining 690). A paused
  plugin restart rebuilt it as `rebalance=4`, remaining 686. Later,
  `revenue-config set daily_budget_sats 0` immediately changed the config RPC
  but `revenue-budget` continued to advertise the cached 1,000-sat envelope
  until another paused restart. The fix makes explicit operator budget reads
  force-fresh and invalidates telemetry snapshots only after committed spend,
  reservation, settlement, cleanup, or runtime-config mutations. The
  authorization provider was already force-fresh and the reservation rail was
  already atomic, so no budget-bypass evidence was found. In the paused live
  lab, 0→1,000→0-sat runtime changes were each visible in the immediately next
  budget read without a restart or TTL wait. Atomic settlement and rollback
  notification behavior are covered by regression tests.
- **Summary reporting:** the smoke sweep found that the legacy
  `revenue-report summary` implementation did not fulfil its documented
  financial/channel contract. It now composes the canonical dashboard result
  and current channel states while preserving the existing policy summary.
- **Pending-settlement reconciliation:** live fault injection now covers both
  terminal outcomes without synthetic database rows. Reservations remain held
  while the outcome is unknown, late success atomically records exact-msat
  cost and releases the remainder, and terminal failure releases without
  spend. Late success must also clear the persisted pending-timeout pair
  cooldown; a regression assertion and repeated live selection cover that
  contract.
- **Newest CLN:** this lab does not establish compatibility beyond Polar's
  bundled 25.12 image. Test current CLN in a separate custom-image lab; do not
  replace the known-good target image mid-run.

## Tuning implication

Keep `rebalance_hold_margin=0`, probability budget bonus off, and the automatic
budget gate unchanged. Expand the next run around the fee crossover (10, 14,
25, 50, 100, and 200 ppm), run forward and reverse demand in separate windows,
and pre-seed directional return liquidity before measuring automatic
execution. Preserve the automatic route-failover assertion: one dry return
edge must be excluded and a liquid mixed-client alternative must settle
without leaking its reservation. A setting can advance only after both client lanes pass three fresh
networks with fresh timestamps, zero orphan reservations, and net revenue after
actual rebalance cost.
