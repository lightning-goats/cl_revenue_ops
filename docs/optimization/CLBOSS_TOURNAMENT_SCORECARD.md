# CLBOSS tournament scorecard

Coverage: 21 replicas, 40 blocks, 2033 attempted / 2032 settled payments. Enhanced strict-schema blocks: 23; safety-eligible: 16.

| Comparable area | Revenue Ops | CLBOSS | Current leader |
|---|---:|---:|---|
| Routing volume (msat) | 7892721354 | 26970803657 | clboss |
| Forward count | 485 | 1402 | clboss |
| Gross routing fees (msat) | 894132 | 185776 | revenue_ops |
| Rebalance cost (msat) | 0 | 0 | tie |
| Net routing profit (msat) | 894132 | 185776 | revenue_ops |
| Gross yield (ppm) | 113.286 | 6.888 | revenue_ops |
| Volume share (%) | 22.639 | 77.361 | clboss |
| Mean worst imbalance (ppm; lower is better) | 692414.8 | 688648.5 | clboss |

Formal verdict: **not ready**. It requires at least three fresh replicas and six enhanced cold/warm blocks per league per replica.

This table describes observed lab outcomes; it does not treat historical smoke blocks as decisive evidence.

## Safety-eligible results by market profile

Only enhanced blocks with no fallback traffic and no block-level or
contender-level safety violations contribute here.

| Profile / phase / scope | Revenue volume (msat) | CLBOSS volume (msat) | Revenue net (msat) | CLBOSS net (msat) | Current result |
|---|---:|---:|---:|---:|---|
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
- The realistic 100-payment result now repeats across crossed identities. Replica 41 produced Revenue/CLBOSS volume of 1.79B/1.97B msat and fees of 253,755/1,970 msat; replica 44's eligible 80-payment cold plus 20-payment warm blocks produced the same 1.79B/1.97B volume and 255,705/1,970 fees. Combined, Revenue captured 92/200 routes, 90.9% of CLBOSS' volume, 129.3x its fees, and ended both crossed runs near 790k versus CLBOSS' 970k worst-imbalance ppm. Global fee cuts would sacrifice this replicated advantage; improvements must target missed lanes selectively.
- Replica 44's warm continuation exposed the sustainability difference: after the cold block, Revenue served 19/20 routes and 645M msat while CLBOSS served 1/20 and 15M. Revenue earned 91,915 versus 15 msat and remained less imbalanced. CLBOSS' xrebalancer was healthy but moved no liquidity.
- The crossed realistic CLN block repeated the profit result: Revenue earned 2.25x CLBOSS' fees from one of ten routes. On the LND corridor, however, Revenue won no routes at either 150 ppm or the 50-ppm safety floor while CLBOSS quoted 1 ppm. The 50-ppm cut therefore produced no conversion benefit.
- Native paid retention is mechanically verified but not a decisive strategy. In replica 39, Revenue Ops autonomously moved one CLN lane from 0 to the observed 1-ppm floor after exactly 50,000 acquired sats, persisted the phase, and restored its captured baseline during cleanup. The safety-eligible paid block split forward routes 5-5, while CLBOSS won weighted volume 80M to 45M msat and net fees 632 to 275 msat. Replica 40 then tested a strict positive-base undercut: Revenue moved from 0 to 4 msat + 0 ppm against 1 ppm, but across two paid blocks the treated lane split routes 8-12 and fees 32-110 msat; whole-contender profit was 767-1590 msat. The implementation and exact restoration are sound, but the economic hypothesis is not supported. Earlier manual retention won the LND block in replica 27 but lost the crossed CLN block in replica 34.
- A bounded 0-ppm acquisition quote can win a lane, but placement matters: observed lane share has ranged from 40% to 100% across client and peer identities.
- A 1-ppm tie did not acquire traffic in an earlier round. A zero-fee quote acquired 80% of the treated lane in replica 25 at an opportunity cost of 1.5 sats, then restored the captured 15-ppm baseline exactly.
- Autonomous rebalancing correctly refused uneconomic routes below its contribution-margin hold. Forced route checks proved CLN 26 route compatibility and budget reconciliation, so weakening the margin rail is not justified by current evidence.
- Product revision `3df9ad3` fixes the profitability-cache contradiction found in replica 42. Recent settled flow can trigger one backoff-protected canonical refresh; only canonical profitability and capex output can create value or budget. In replica 43 this changed the native cycle from zero considered pairs/`dest_not_valuable` to two considered pairs and 550 sats of budget, while correctly holding a route whose expected value did not cover its costs. In replica 44 all four channels became canonically valuable and funded during the pressure run.
- After replica 44's warm block, Revenue selected and attempted a positive-EV 20,000-sat refill quoted at 2 sats with 98% estimated success. Four attempts failed on depleted lab return-path channels, spent zero, and left no reservation. CLBOSS attempted no refill. This is not evidence that the economic gate is wrong: one-way traffic had also exhausted every available circular return path. The next rebalance comparison needs a reserved neutral return lane that scored traffic cannot consume.
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
| Reserved return lane | Whether a controller can complete a profitable circular refill after scored one-way pressure | Provision a neutral circulation path that payment traffic cannot consume; score actual refill cost and post-refill income under equal caps. |
| Evidence freshness | Whether settled forwards become canonical value/budget evidence before a 15-minute cache TTL expires | Implemented in `3df9ad3`; keep the forced analyzer refresh canonical, read-only, and five-minute backoff protected. |
| Product change | Repeatable positive net lift across crossed identities and clients | Promote only treatments with replicated safety-eligible evidence. |

Regenerate the aggregate observation separately before reconciling the narrative table:

```bash
.venv/bin/python tools/polar_clboss_scorecard.py --format markdown
```
