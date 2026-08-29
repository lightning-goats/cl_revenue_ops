# CLBOSS tournament scorecard

Coverage: 36 replicas, 58 blocks, 2355 attempted / 2352 settled payments. Enhanced strict-schema blocks: 41; safety-eligible: 31.

| Comparable area | Revenue Ops | CLBOSS | Current leader |
|---|---:|---:|---|
| Routing volume (msat) | 13017724010 | 33990803657 | clboss |
| Forward count | 592 | 1621 | clboss |
| Gross routing fees (msat) | 1599917 | 428526 | revenue_ops |
| Rebalance cost (msat) | 18624 | 41554 | revenue_ops |
| Net routing profit (msat) | 1581293 | 386972 | revenue_ops |
| Gross yield (ppm) | 122.903 | 12.607 | revenue_ops |
| Volume share (%) | 27.692 | 72.308 | clboss |
| Mean worst imbalance (ppm; lower is better) | 760292.0 | 784240.3 | revenue_ops |

Formal verdict: **not ready**. It requires at least three fresh replicas and six enhanced cold/warm blocks per league per replica.

This table describes observed lab outcomes; it does not treat historical smoke blocks as decisive evidence.

## Safety-eligible results by market profile

Only enhanced blocks with no fallback traffic and no block-level or
contender-level safety violations contribute here.

| Profile / phase / scope | Revenue volume (msat) | CLBOSS volume (msat) | Revenue net (msat) | CLBOSS net (msat) | Current result |
|---|---:|---:|---:|---:|---|
| `realistic` / replicas 47-48 crossed cold / both | 2330000000 | 3870000000 | 340545 | 3870 | Revenue wins profit/balance; CLBOSS wins volume |
| `realistic` / replica 44 warm pressure / both | 645000000 | 15000000 | 91915 | 15 | Revenue wins routes, volume, profit, and ending balance |
| `realistic` / replica 44 cold pressure / both | 1145000000 | 1955000000 | 163790 | 1955 | Revenue wins profit/balance; CLBOSS wins cold volume |
| `realistic` / 100-payment crossed cold / both | 1790000000 | 1970000000 | 253755 | 1970 | Near-parity routes; Revenue wins profit and ending balance |
| `acquisition` / native positive-base retention / CLN | 90000000 | 160000000 | 767 | 1590 | CLBOSS wins combined volume, routes, and profit across two paid blocks |
| `acquisition` / mixed acquisition-to-retention transition / CLN | 120000000 | 5000000 | 628 | 5 | Revenue wins, but the in-window phase transition makes this diagnostic only |
| `acquisition` / native paid retention / CLN | 45000000 | 80000000 | 275 | 632 | Forward routes tie 5-5; CLBOSS wins weighted volume and profit |
| `acquisition` / paid retention / LND | 255000000 | 195000000 | 2225 | 1419 | Revenue wins volume and profit |
| `acquisition` / paid retention / CLN | 10000000 | 115000000 | 60 | 943 | CLBOSS wins volume and profit |
| `realistic` / forward pressure / both | 180000000 | 1370000000 | 27000 | 1370 | Revenue wins profit; CLBOSS wins volume/balance |
| `realistic` / forward pressure / CLN | 5000000 | 325000000 | 730 | 325 | Revenue wins profit; CLBOSS wins volume/balance |
| `realistic` / forward pressure / LND | 0 | 330000000 | 0 | 330 | CLBOSS wins volume, profit, and balance |
| `realistic` / 50-ppm treatment / LND | 0 | 330000000 | 0 | 330 | CLBOSS wins; ordinary floor cut buys no volume |
| `realistic` / crossed post-refill / LND | 570000000 | 470000000 | 66291 | 56400 | Revenue wins volume and linked net profit |
| `realistic` / crossed post-refill / CLN | 1140000000 | 940000000 | 155752 | 1645 | Revenue wins volume and linked net profit |
| `legacy_low_fee` / baseline / both | 5000000 | 445000000 | 75 | 4870 | CLBOSS wins volume and profit |

## Fee-market regimes

The tournament no longer treats the original 10-ppm startup policy as a
general market model. It now records one of two explicit profiles in every new
traffic block:

| Profile | Initial base / rate | Traffic amounts | Purpose |
|---|---:|---:|---|
| `acquisition` | 1 msat / 10 ppm | fixed 5k sat by default | Isolate low-price route acquisition and paid retention. |
| `realistic` | 500 msat / 150 ppm | deterministic 5k, 15k, 35k, 100k sat mix | Primary fee-setting, liquidity, and net-profit comparison. |

The realistic seed is a rounded, dated snapshot of the public announced graph,
not a claim that one fee fits every channel. On 2026-08-28, 1ML reported a
0.437-sat median base fee and a 150-ppm median rate; its 25th, 75th, and 95th
fee-rate percentiles were approximately 1, 633, and 2,863 ppm. Public graph
statistics omit private channels, so tournament conclusions must remain robust
across the full distribution rather than optimize to the median alone.

CLBOSS is intentionally not spend-capped in full-stack competition. It runs
its native xrebalancer with grant mode off and its fastest exposed rate of 120
attempts per hour. Revenue Ops retains its production budget enforcement; the
controlled rounds grant it a 1,000-sat rebalance allowance. The Polar-only
cadence is compressed to 15 seconds for Revenue Ops fee, flow, and rebalance
cycles. Production defaults remain unchanged.

## Controlled native rebalancing

These observations are separate from the aggregate traffic table because the
fixture payments create the starting liquidity state rather than scored route
demand. Both controllers start with matching selected-family lanes at
approximately 75% local on the source and 25% local on the destination, receive
equal 2M-sat neutral return paths, and resume simultaneously. No manual cycle
RPC is used.

| Replica / identity assignment | Revenue delivered / cost | CLBOSS delivered / cost | Safety | Result |
|---|---:|---:|---|---|
| 52 / Revenue B, CLBOSS A | 50,000 sats / 1.052 sats | 0 / 0 | no violations | Revenue completed a profitable native refill; CLBOSS did nothing |
| 53 / Revenue A, CLBOSS B | 50,000 sats / 1.052 sats | 0 / 0 | no violations | Crossed replication of the Revenue win |
| 54 / Revenue B, CLBOSS A | 50,000 sats / 1.052 sats | 0 / 0 | no violations | Revenue refill converted into a post-refill volume win |
| 55 / Revenue A, CLBOSS B | 50,000 sats / 1.052 sats | 0 / 0 | no violations | Crossed post-refill volume replication |
| 57 / Revenue A, CLBOSS B | 50,000 sats / 1.052 sats | 0 / 0 | no violations | Fixed-image post-refill volume win |
| 58 / Revenue B, CLBOSS A | 50,000 sats / 1.052 sats | 0 / 0 | no violations | Crossed fixed-image post-refill volume win |
| 60 / Revenue B, CLBOSS A | 50,000 sats / 1.052 sats | 0 / 0 | no violations | Exact 90-sat evidence-band validation over 180 seconds |
| 61 / Revenue A, CLBOSS B | 50,000 sats / 1.052 sats | 0 / 0 | no violations | Crossed exact-band validation |
| 62 / Revenue B, CLBOSS A / LND | 50,000 sats / 2.052 sats | 0 / 0 | no violations | Native LND-facing refill and post-refill win |
| 63 / Revenue A, CLBOSS B / LND | 50,000 sats / 2.052 sats | 0 / 0 | no violations | Crossed native LND-facing replication |
| 64 / Revenue B, CLBOSS A / repeated LND | first 50,000 sats / 2.052 sats; later 0 | later 155,000 sats / 1.311 sats | no violations | 10% emergency floor left a profitable 14.5% destination cooling while CLBOSS renewed |
| 65 / Revenue A, CLBOSS B / diagnostic | first 50,000 sats / 2.052 sats; later 0 | 0 | no violations | Discarded: plugin startup option still injected the old 10% default |
| 67 / Revenue A, CLBOSS B / equal pressure | 310,000 sats / 4.314 sats | 155,000 sats / 1.311 sats | no violations | 20% floor activated, but stale balance scheduled a duplicate Revenue refill |
| 68 / Revenue B, CLBOSS A / final equal pressure | 155,000 sats / 2.157 sats | 155,000 sats / 1.311 sats | no violations | Final image renewed once to exactly 30%; duplicate refill eliminated |

Across these ten clean observations Revenue delivered 500,000 sats for 12.520
sats while uncapped CLBOSS delivered zero. The refill moved each selected
Revenue destination from below the 30% threshold into the operating band;
untouched lanes in the other client family mean the contender-wide
worst-imbalance metric remains unsuitable for this fixture.

Repeated warm demand then exposed a renewal gap hidden by the original
single-epoch fixture. In replica 64, settled demand drained Revenue's profitable
LND destination from 30% to 14.5%. The 24-hour cooldown still blocked it because
the emergency floor was 10% and anchor drift was only 15.5 points; uncapped
CLBOSS restored 155,000 sats in the corresponding warm epoch. Revision
`43e006b` raised the configurable emergency refill floor to 20% while preserving
the ordinary cooldown, value, EV, and budget gates. Replica 65 proved that the
CLN plugin option still overrode the dataclass with 10%; revision `12a2baf`
aligned the actual startup surface and added a default-parity regression test.

An equal-pressure, noncompetitive functional lane then sent the same realistic
5k/15k/35k/100k mix through each held controller. Replica 67 proved the 20%
override activated, but Revenue sent two 155,000-sat refills because the next
15-second cycle still saw the stale pre-settlement balance. Revision `f201b22`
adds a 60-second post-success grace that suppresses only the emergency shortcut.
Final-image replica 68 started Revenue/CLBOSS destinations at 14.5%/9.5%; each
controller autonomously restored exactly 155,000 sats. Revenue spent 2.157 sats,
CLBOSS spent 1.311 sats, Revenue ended exactly at its 30% floor, and no forced
cycle, fallback, reservation leak, or safety violation occurred. This closes
the duplicate-refill bug but leaves CLBOSS ahead on equal-refill cost.

The same warm rounds identify the next fee-setting target. In corrected crossed
replicas 66-67, CLBOSS's natural 120-ppm LND destination carried every offered
155,000-sat competitive epoch while Revenue's roughly 133-ppm destination
carried none. Revenue remains the aggregate net-profit and yield leader, but it
cannot yet claim decisive fee-setting or route-share superiority.

The four safety-eligible post-refill demand blocks (54, 55, 57, and 58) then
routed 1.14B msat through Revenue versus 0.94B through CLBOSS. Revenue's routing
fees were 159.960 sats versus 1.645 sats. Charging the four linked 1.052-sat
refills to this phase leaves Revenue at 155.752 sats net versus 1.645 sats for
CLBOSS. Every block settled 15/15 payments with no fallback, and the 285M/235M
volume split repeated exactly across both identities and both product images.

The crossed LND-facing replicas 62-63 use the same exact 120-ppm evidence band
and multipart fixture payments pinned to one outgoing contender and the same
contender as last hop. Revenue again refilled 50,000 sats while CLBOSS did
nothing. Their two post-refill LND blocks settled 30/30 without fallback or
safety violations and repeated the 285M/235M split across identities. Revenue
earned 70,395 msat gross and 66,291 msat after both linked refills, versus
56,400 msat for CLBOSS. Across all six eligible post-refill blocks, linked net
is therefore 222,043 msat for Revenue versus 58,045 msat for CLBOSS.

Revenue Ops' bounded acquisition experiment remains default-off and may quote
0 ppm on only one capped episode. It now admits competitor observations from
1 through 10 ppm instead of requiring exactly 1 ppm; all duration, volume,
opportunity-cost, liquidity, and cooldown rails remain unchanged. After 50,000
acquired sats it may run a one-hour, 250,000-sat paid validation phase at
0 ppm plus a positive base fee one millisatoshi below the competitor's
proportional charge at the smallest acquired payment. If no positive strict
undercut exists, it exits. Both phases share the 25-sat opportunity-cost cap
and restore the exact captured base and proportional fees on exit.

## What the tournament has established

- Revenue Ops extracts more fee per routed sat, but CLBOSS wins far more routing volume. The main economic gap is conversion and retained demand, not fee arithmetic alone.
- Replicas 47-48 add a fresh crossed realistic repeat: Revenue earned 340,545 msat from 2.33B msat while CLBOSS earned 3,870 msat from 3.87B msat. Revenue finished materially better balanced in both runs (270,178 and 130,170 worst-imbalance ppm versus 970,000 for CLBOSS), with zero safety violations. CLBOSS still wins raw volume.
- Controlled replicas 52-53 establish a crossed native-rebalance win. Revenue completed one positive-EV 50,000-sat refill in each 90-second observation for 1.052 sats; uncapped CLBOSS completed none despite its 120/hour setting. Both runs were safety-clean. This is evidence for execution responsiveness and profitability discipline, not yet long-horizon profit superiority.
- Replicas 54-55 and fixed-image replicas 57-58 connect that refill to customer demand: Revenue won post-refill volume 285M to 235M msat in every run and linked net profit 155,752 to 1,645 msat across the four eligible blocks. The direct fixture paths were cooperatively closed and confirmed absent before scoring, so no payment bypassed the contenders.
- Crossed replicas 62-63 extend the same result to LND-facing liquidity. Revenue repeated the 50,000-sat native refill and exact 285M/235M post-refill volume win in both identities; aggregate linked net was 66,291 versus 56,400 msat with 30/30 settlements. This reverses the earlier unrepaired LND corridor loss under a causal liquidity fixture rather than a global fee-floor cut.
- Repeated warm replicas 64-68 changed the rebalance decision path twice: the default emergency floor is now 20% at both the dataclass and CLN option surfaces, and a 60-second settlement grace prevents stale-balance duplicate refills. Final-image replica 68 restored exactly 155,000 sats once, with clean safety and no forced cycle. CLBOSS still achieved the same refill for 0.846 sats less.
- Corrected crossed replicas 66-67 also show the current fee-setting loss directly: CLBOSS at 120 ppm won every offered LND demand payment against Revenue near 133 ppm. The next fee experiment must improve conversion without returning to globally unrealistic fee floors.
- The scorer now resolves every post-refill smoke block to its exact native observation, fails closed on missing or mismatched lineage, charges the linked rebalance cost, and publishes eligible single-family phase results. Historical aggregate profit no longer silently treats native refills as free.
- Replica 56 exposed an arbitrary early-channel capex cliff: a channel with four forwards, positive canonical contribution, and a profitable classification received zero budget because it had neither more than five forwards nor more than 100 sats contribution. Revision `4c26e11` now admits an early active tier funded only by the configured reinvestment share of realized 30-day contribution and capped by the existing bootstrap rail. Zero, absent, negative, malformed, and DB-degraded evidence still grants nothing.
- Replicas 60-61 validate `4c26e11` in the exact repaired band across identities. Equal 120-ppm fixture pricing produced approximately 90 sats of contribution, Revenue received 88 sats of combined allocation and completed the same 50,000-sat/1.052-sat refill, while CLBOSS completed none during each 180-second observation. Both observations were safety-clean.
- Diagnostic replicas 56, 59, and 60 demand blocks remain excluded: 56 and 60 each had one terminal failed payment, while 59 allowed a delayed CLBOSS circular payment to overlap scored traffic. The runner now freezes both controllers after the native observation and before retiring return paths, preventing later circular forwards from contaminating customer-demand attribution.
- The realistic 100-payment result now repeats across crossed identities. Replica 41 produced Revenue/CLBOSS volume of 1.79B/1.97B msat and fees of 253,755/1,970 msat; replica 44's eligible 80-payment cold plus 20-payment warm blocks produced the same 1.79B/1.97B volume and 255,705/1,970 fees. Combined, Revenue captured 92/200 routes, 90.9% of CLBOSS' volume, 129.3x its fees, and ended both crossed runs near 790k versus CLBOSS' 970k worst-imbalance ppm. Global fee cuts would sacrifice this replicated advantage; improvements must target missed lanes selectively.
- Replica 44's warm continuation exposed the sustainability difference: after the cold block, Revenue served 19/20 routes and 645M msat while CLBOSS served 1/20 and 15M. Revenue earned 91,915 versus 15 msat and remained less imbalanced. CLBOSS' xrebalancer was healthy but moved no liquidity.
- The crossed realistic CLN block repeated the profit result: Revenue earned 2.25x CLBOSS' fees from one of ten routes. On the LND corridor, however, Revenue won no routes at either 150 ppm or the 50-ppm safety floor while CLBOSS quoted 1 ppm. The 50-ppm cut therefore produced no conversion benefit.
- Native paid retention is mechanically verified but not a decisive strategy. In replica 39, Revenue Ops autonomously moved one CLN lane from 0 to the observed 1-ppm floor after exactly 50,000 acquired sats, persisted the phase, and restored its captured baseline during cleanup. The safety-eligible paid block split forward routes 5-5, while CLBOSS won weighted volume 80M to 45M msat and net fees 632 to 275 msat. Replica 40 then tested a strict positive-base undercut: Revenue moved from 0 to 4 msat + 0 ppm against 1 ppm, but across two paid blocks the treated lane split routes 8-12 and fees 32-110 msat; whole-contender profit was 767-1590 msat. The implementation and exact restoration are sound, but the economic hypothesis is not supported. Earlier manual retention won the LND block in replica 27 but lost the crossed CLN block in replica 34.
- A bounded 0-ppm acquisition quote can win a lane, but placement matters: observed lane share has ranged from 40% to 100% across client and peer identities.
- A 1-ppm tie did not acquire traffic in an earlier round. A zero-fee quote acquired 80% of the treated lane in replica 25 at an opportunity cost of 1.5 sats, then restored the captured 15-ppm baseline exactly.
- Autonomous rebalancing correctly refuses uneconomic routes below its contribution-margin hold. Product revision `9805b04` additionally prices one lower-ranked source fallback per selected destination, so an expensive first choice no longer suppresses a profitable alternative. It preserves the EV, budget, and one-pair-per-destination rails. Replicas 52-53 show that bounded fallback completing the same profitable route across crossed identities.
- Product revisions `3df9ad3` and `0aa7da8` fix the profitability-cache contradiction found in replicas 42 and 50. Any newly settled forward count can now trigger a backoff-protected canonical refresh, and Polar can genuinely run 15-second cycles instead of silently clamping them to 60 seconds. Only canonical profitability and capex output can create value or budget; production defaults are unchanged.
- After replica 44's warm block, Revenue selected and attempted a positive-EV 20,000-sat refill quoted at 2 sats with 98% estimated success. Four attempts failed on depleted lab return-path channels, spent zero, and left no reservation. The controlled fixture now supplies equal neutral return lanes after pressure and validates their balances before observation.
- Tournament preflight now pins the default image to the verified Revenue Ops revision and rejects a mismatched label before scored traffic; an unscored replica exposed the stale default tag.
- Replica 41's attempted warm replication stopped fail-closed after six settled payments when one CLN dispatch timed out and could not be reconciled immediately. A later read-only lookup found the invoice unpaid, but the partial block remains excluded rather than using hindsight to weaken the no-replay contract.
- Product revision `3df9ad3` and harness revision `5325003` add canonical profitability refresh on settled-flow contradiction, hardened timeout reconciliation, and the pinned contender image. Full tests passed before replicas 43-44 (3878 passed, 5 skipped, 2 xfailed).

## Active improvement loop

| Step | Evidence sought | Implementation or decision gate |
|---|---|---|
| Family attribution | CLN versus LND volume, fees, and forwards | Runner blocks map every contender SCID to a client family and fail closed on unmapped activity. |
| Automatic acquisition | Whether the default-off product selects and wins a natural lane | Enable the gate and wait for native fee cycles; never force a scored fee cycle. |
| Paid retention | Whether a positive base-fee undercut converts the 1-ppm tie into retained volume and profit | Keep experimental until positive lift repeats across crossed identities and both client families. |
| Retention curve | Whether any paid quote beats free acquisition on net profit under CLN route randomization | Measure multiple bounded price points with enough routes for confidence; optimize net contribution, not raw route count. |
| Liquidity pressure | Whether each controller restores depleted earning liquidity profitably | Run one-way traffic with equal spend caps; compare net fees, cost, and ending imbalance. |
| Controlled depletion | Whether each native controller repairs the same exact 75/25 liquidity state | Ten clean observations across CLN/LND and crossed identities; Revenue leads 10-0 while CLBOSS remains uncapped. |
| Reserved return lane | Whether a controller can complete a profitable circular refill after pressure | Equal post-pressure 2M-sat CLN/LND paths are removed and confirmed absent before demand scoring. |
| Post-refill demand | Whether repaired liquidity produces more routed volume and linked net profit | Six eligible CLN/LND blocks repeat Revenue's 285M/235M volume win; aggregate linked net is 222,043/58,045 msat. Extend to longer warm demand. |
| Warm renewal | Whether profitable outbound inventory renews inside the normal 24-hour cooldown without duplicate spend | `f201b22` passes one final equal-pressure epoch: exact 155k refill to 30%, one payment, clean safety. Cross the final image and extend to multiple epochs. |
| Fee conversion | Whether Revenue can beat CLBOSS's 120-ppm LND lane without a global low-fee policy | Replicas 66-67 lose every offered payment at ~133 ppm; design a bounded corridor-relative quote and score net contribution after refill cost. |
| CLN 26.06.7 compatibility | Whether both contenders and all read-only/action surfaces remain compatible with the 2026-08-28 security point release | Official source and Docker image are embargoed/unavailable; build an equal-runtime lane immediately when official artifacts appear. Do not treat 26.06.6 as the production recommendation. |
| Evidence freshness | Whether settled forwards become canonical value/budget evidence before a 15-minute cache TTL expires | Implemented through `0aa7da8`; keep the analyzer refresh canonical, read-only, and backoff protected. |
| Product change | Repeatable positive net lift across crossed identities and clients | Promote only treatments with replicated safety-eligible evidence. |

Regenerate the aggregate observation separately before reconciling the narrative table:

```bash
.venv/bin/python tools/polar_clboss_scorecard.py --format markdown
```
