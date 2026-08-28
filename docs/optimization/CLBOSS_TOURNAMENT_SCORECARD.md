# CLBOSS tournament scorecard

Coverage: 16 replicas, 30 blocks, 1490 attempted / 1490 settled payments. Enhanced strict-schema blocks: 13; safety-eligible: 9.

| Comparable area | Revenue Ops | CLBOSS | Current leader |
|---|---:|---:|---|
| Routing volume (msat) | 1997282244 | 14766242767 | clboss |
| Forward count | 286 | 1052 | clboss |
| Gross routing fees (msat) | 78495 | 108355 | clboss |
| Rebalance cost (msat) | 0 | 0 | tie |
| Net routing profit (msat) | 78495 | 108355 | clboss |
| Gross yield (ppm) | 39.301 | 7.338 | revenue_ops |
| Volume share (%) | 11.914 | 88.086 | clboss |
| Mean worst imbalance (ppm; lower is better) | 664180.9 | 590734.3 | clboss |

Formal verdict: **not ready**. It requires at least three fresh replicas and six enhanced cold/warm blocks per league per replica.

This table describes observed lab outcomes; it does not treat historical smoke blocks as decisive evidence.

## Safety-eligible results by market profile

Only enhanced blocks with no fallback traffic and no block-level or
contender-level safety violations contribute here.

| Profile / phase / scope | Revenue volume (msat) | CLBOSS volume (msat) | Revenue net (msat) | CLBOSS net (msat) | Current result |
|---|---:|---:|---:|---:|---|
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
- Under realistic one-way pressure, Revenue Ops earned 19.7x CLBOSS' net routing fees from 13.1% of the volume. Global fee cuts would sacrifice the strongest demonstrated advantage; improvements must target missed lanes selectively.
- The crossed realistic CLN block repeated the profit result: Revenue earned 2.25x CLBOSS' fees from one of ten routes. On the LND corridor, however, Revenue won no routes at either 150 ppm or the 50-ppm safety floor while CLBOSS quoted 1 ppm. The 50-ppm cut therefore produced no conversion benefit.
- Native paid retention is mechanically verified but not a decisive strategy. In replica 39, Revenue Ops autonomously moved one CLN lane from 0 to the observed 1-ppm floor after exactly 50,000 acquired sats, persisted the phase, and restored its captured baseline during cleanup. The safety-eligible paid block split forward routes 5-5, while CLBOSS won weighted volume 80M to 45M msat and net fees 632 to 275 msat. Earlier manual retention won the LND block in replica 27 but lost the crossed CLN block in replica 34.
- A bounded 0-ppm acquisition quote can win a lane, but placement matters: observed lane share has ranged from 40% to 100% across client and peer identities.
- A 1-ppm tie did not acquire traffic in an earlier round. A zero-fee quote acquired 80% of the treated lane in replica 25 at an opportunity cost of 1.5 sats, then restored the captured 15-ppm baseline exactly.
- Autonomous rebalancing correctly refused uneconomic routes below its contribution-margin hold. Forced route checks proved CLN 26 route compatibility and budget reconciliation, so weakening the margin rail is not justified by current evidence.
- Tournament preflight now pins the default image to the verified Revenue Ops revision and rejects a mismatched label before scored traffic; an unscored replica exposed the stale default tag.
- Product revision `9d5d8f7` and harness revision `ab38081` add restart-safe native paid retention and exact phase validation. Full tests passed before replica 39 (3856 passed, 5 skipped, 2 xfailed).

## Active improvement loop

| Step | Evidence sought | Implementation or decision gate |
|---|---|---|
| Family attribution | CLN versus LND volume, fees, and forwards | Runner blocks map every contender SCID to a client family and fail closed on unmapped activity. |
| Automatic acquisition | Whether the default-off product selects and wins a natural lane | Enable the gate and wait for native fee cycles; never force a scored fee cycle. |
| Paid retention | Whether a positive base-fee undercut converts the 1-ppm tie into retained volume and profit | Keep experimental until positive lift repeats across crossed identities and both client families. |
| Liquidity pressure | Whether each controller restores depleted earning liquidity profitably | Run one-way traffic with equal spend caps; compare net fees, cost, and ending imbalance. |
| Product change | Repeatable positive net lift across crossed identities and clients | Promote only treatments with replicated safety-eligible evidence. |

Regenerate the aggregate observation separately before reconciling the narrative table:

```bash
.venv/bin/python tools/polar_clboss_scorecard.py --format markdown
```
