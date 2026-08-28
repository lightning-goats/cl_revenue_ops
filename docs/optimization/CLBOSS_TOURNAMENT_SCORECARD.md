# CLBOSS tournament scorecard

Coverage: 13 replicas, 22 blocks, 1360 attempted / 1360 settled payments. Enhanced strict-schema blocks: 5; safety-eligible: 3.

| Comparable area | Revenue Ops | CLBOSS | Current leader |
|---|---:|---:|---|
| Routing volume (msat) | 1747343989 | 13401181022 | clboss |
| Forward count | 240 | 966 | clboss |
| Gross routing fees (msat) | 76587 | 104320 | clboss |
| Rebalance cost (msat) | 0 | 0 | tie |
| Net routing profit (msat) | 76587 | 104320 | clboss |
| Gross yield (ppm) | 43.831 | 7.784 | revenue_ops |
| Volume share (%) | 11.535 | 88.465 | clboss |
| Mean worst imbalance (ppm; lower is better) | 542064.8 | 501001.3 | clboss |

Formal verdict: **not ready**. It requires at least three fresh replicas and six enhanced cold/warm blocks per league per replica.

This table describes observed lab outcomes; it does not treat historical smoke blocks as decisive evidence.

## Safety-eligible results by market profile

Only enhanced blocks with no fallback traffic and no block-level or
contender-level safety violations contribute here.

| Profile / phase | Revenue volume (msat) | CLBOSS volume (msat) | Revenue net (msat) | CLBOSS net (msat) | Current result |
|---|---:|---:|---:|---:|---|
| `acquisition` / paid retention | 255000000 | 195000000 | 2225 | 1419 | Revenue wins volume and profit |
| `realistic` / forward pressure | 180000000 | 1370000000 | 27000 | 1370 | Revenue wins profit; CLBOSS wins volume/balance |
| `legacy_low_fee` / baseline | 5000000 | 445000000 | 75 | 4870 | CLBOSS wins volume and profit |

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
opportunity-cost, liquidity, and cooldown rails remain unchanged.

## What the tournament has established

- Revenue Ops extracts more fee per routed sat, but CLBOSS wins far more routing volume. The main economic gap is conversion and retained demand, not fee arithmetic alone.
- Under realistic one-way pressure, Revenue Ops earned 19.7x CLBOSS' net routing fees from 13.1% of the volume. Global fee cuts would sacrifice the strongest demonstrated advantage; improvements must target missed lanes selectively.
- In replica 27, automatic 0-ppm acquisition followed by 1-ppm paid retention kept all LND traffic and beat CLBOSS by 30.8% on volume and 56.8% on net routing fees in the clean warm block.
- A bounded 0-ppm acquisition quote can win a lane, but placement matters: observed lane share has ranged from 40% to 100% across client and peer identities.
- A 1-ppm tie did not acquire traffic in an earlier round. A zero-fee quote acquired 80% of the treated lane in replica 25 at an opportunity cost of 1.5 sats, then restored the captured 15-ppm baseline exactly.
- Autonomous rebalancing correctly refused uneconomic routes below its contribution-margin hold. Forced route checks proved CLN 26 route compatibility and budget reconciliation, so weakening the margin rail is not justified by current evidence.

## Active improvement loop

| Step | Evidence sought | Implementation or decision gate |
|---|---|---|
| Family attribution | CLN versus LND volume, fees, and forwards | New runner blocks map every contender SCID to a client family and fail closed on unmapped activity. |
| Automatic acquisition | Whether the default-off product selects and wins a natural lane | Enable the gate and wait for native fee cycles; never force a scored fee cycle. |
| Paid retention | Whether acquired flow remains at 1, 2, or 5 ppm | Restore the captured baseline, apply one bounded positive-fee treatment, then run a warm block. |
| Liquidity pressure | Whether each controller restores depleted earning liquidity profitably | Run one-way traffic with equal spend caps; compare net fees, cost, and ending imbalance. |
| Product change | Repeatable positive net lift across crossed identities and clients | Only promote a paid-retention ladder or rebalance-policy change after replicated evidence. |

Regenerate the observed table with:

```bash
.venv/bin/python tools/polar_clboss_scorecard.py --format markdown
```
