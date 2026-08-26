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
| Fresh-network repeatability | A second network was provisioned from zero with the same 12-channel topology. Its corrected deterministic matrix settled 8/8 payments: two rounds on each CLN/LND forward and reverse lane. Each forward seed was 45,000 sats and each return payment 20,000 sats, preserving the 20,000-sat channel reserve plus fee margin. Polar's fixed host ports require networks to run sequentially in the same app; the harness now refuses to create a fresh lab while another network is started, preventing a half-created error-state network. |
| Dry-run fee-state bug | A fresh-node dry-run fee cycle proposed 14/76/50/11 ppm while all four real policies correctly remained at 10 ppm, but the controller persisted the proposals as `last_broadcast_fee_ppm` and advanced applied-state timestamps/streaks. Dry-run results now carry an explicit proposal marker, retain the real broadcast state, keep the target pending, and do not consume an upward-probe/broadcast streak. In dry-run mode, the next evaluation also repairs and durably persists any prior positive tracked-versus-actual mismatch; live-mode desync tolerance is unchanged. The repaired live diagnostics converged all four channels back to the actual 10 ppm without a `setchannel` call. |
| Competitors at 200 ppm | 10/10 settled; the target forwarded 7/10. |
| Competitors at 25 ppm | 10/10 settled; the target forwarded 6/10. |
| Fee crossover by client | With both competitors at 0 ppm, the LND lane sent 10/10 through the CLN competitor; the CLN lane split 1 target / 3 CLN competitor / 6 LND competitor. At 25 ppm, the LND lane sent 10/10 through the 14-ppm target while the CLN lane still split 2 target / 3 CLN competitor / 5 LND competitor. This is useful client/cache divergence evidence, not a fee-only attribution. |
| Governed fee cycle | Four policies changed: sink edges 10→14 ppm and payer edges 10→64/74 ppm; reason code `dts_pid_sample`; dynamic HTLC maxima were applied. |
| Fresh governed fee crossover | On network 2, both competitors were raised from 10 to 400 ppm and their eight outgoing policies were verified in the target's gossip. A 10-payment mixed-client window settled 10/10 but added only two target forwards, showing strong payer route-cache effects despite the target's 10-ppm policy. After one governed cycle, the four target policies moved 10→14/79/29/11 ppm with complete `dts_pid_sample` history and fee-intent evidence, zero spend, and zero reservations. Both competitors were restored and gossip-verified at 10 ppm; the matching 10-payment window again settled 10/10 and added zero target forwards. The target was re-paused. This establishes fee sensitivity plus cache divergence, not a fee-only causal share model. |
| Fresh rebalance pricing hold | Network 2 reached approximately 99/95/3/0% target-channel local ratios. With the plugin paused and a temporary 1,000-sat evaluation budget, the engine priced a real 514,800-sat LND return route at 13 sats under a 515-sat pair budget. It correctly held at −23.154 sats EV despite `rebalance_hold_margin=0`: 0.259 sats expected refill value was outweighed by 13 sats route cost, 10.167 sats source opportunity cost, and 0.129 sats activity penalty. No execution or reservation occurred and the budget was restored to zero. This is evidence against lowering the hold margin merely to force activity. |
| Fresh-payer traffic continuation | LND mission control was reset and the CLN payer was restarted through Polar MCP without opening another Polar window. A 20-payment, 50,000-sat mixed-client window settled 20/20 and split 2 target / 14 CLN competitor / 4 LND competitor. A separate 400-ppm competitor window sent 8/8 LND payments through the target, while a single 190,000-sat reverse HTLC bypassed it because the depleted target edge advertised a roughly 55,000-sat dynamic `htlc_max`. Splitting the same demand did not dislodge the CLN payer from known competitor paths until the competitors were briefly stopped for setup. This confirms that mission-control probability and HTLC admission can dominate nominal fee order. |
| Automatic positive-control rebalance | Setup traffic was kept action-safe while the plugin remained paused and budget-zero. Exact directional readback rejected an initially underfunded return path, so six Polar-created invoices were explicitly paid through the CLN competitor to raise usable return liquidity to about 610,000 sats. With all return-path forwarding fees temporarily zero, a paused planning cycle selected exactly one 554,796-sat `270x1x0→162x1x0` candidate at +0.258443 sats EV and the governor blocked it with `PAUSED`. Exactly one unpaused governed cycle then reserved at most 555 sats, failed on the graph-visible but dry LND edge `306x1x1/1`, excluded it, and settled over the liquid CLN competitor on attempt two for 0 msat. Source/destination balances landed at 1,400,006/599,999 sats (approximately 70/30), the reservation returned to zero, budget readback showed one success and zero rebalance spend, and reconciliation was clean. The plugin was immediately re-paused, its budget restored to zero, and all test-path policies restored. |
| Successful-failover observation bug | The native executor correctly recorded the failed LND segment in memory, but the engine published `segment-observations` only when the final execution result failed. A successful alternate route therefore hid useful failure evidence from the datastore contract. Publication now also runs when a successful result carries `previous_failure` or `retry_excluded_channels`; plain successes still avoid an unnecessary write. The new successful-failover regression proves the failed segment is exported. |
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
| Regression suite | Focused rebalance suites: 519 passed. Planner plus operator/reporting regressions: 94 passed. Pending/atomic/engine suites after the late-success cooldown fix: 128 passed. Fee-controller family after the dry-run fix: 652 passed. Successful-failover telemetry family: 110 passed. Final exact-pinned Python 3.11 full suite after the observation fix: 3,727 passed, 5 environment-dependent skips, and 2 intentional expected failures. |

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
- **Successful route failover telemetry:** a failed first route remains useful
  evidence even when an alternate route settles. Publish the failure-derived
  segment observation on both terminal failure and successful failover; do not
  require the overall rebalance to fail before consumers can see the dry edge.
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
