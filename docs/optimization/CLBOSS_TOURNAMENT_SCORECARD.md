# CLBOSS tournament scorecard

Coverage: 70 replicas, 228 blocks, 4329 attempted / 4326 settled payments. Enhanced strict-schema blocks: 211; safety-eligible: 192; diagnostic exclusions: 25.

| Comparable area | Revenue Ops | CLBOSS | Current leader |
|---|---:|---:|---|
| Routing volume (msat) | 25594581928 | 59152684890 | clboss |
| Forward count | 1316 | 2885 | clboss |
| Gross routing fees (msat) | 4005366 | 1697549 | revenue_ops |
| Rebalance cost (msat) | 145849 | 61924 | clboss |
| Net routing profit (msat) | 3859517 | 1635625 | revenue_ops |
| Gross yield (ppm) | 156.493 | 28.698 | revenue_ops |
| Volume share (%) | 30.201 | 69.799 | clboss |
| Mean worst imbalance (ppm; lower is better) | 871132.5 | 791924.4 | clboss |

Formal verdict: **Inconclusive** from frozen crossed series `cadence-666048e-20260829`. All common coverage, reliability, budget, and safety gates passed.

## Formal frozen-series result

This formal result controls tournament promotion; the larger historical aggregate below remains diagnostic. Frozen Revenue Ops revision: `666048e173fe66c7addcb55868f35d5f585d43e6`; image: `sha256:c55fc79d3802297cd932cdffd15f8ae9c11854440583aebfccebdba34d80784e`; replicas: replica-122, replica-123, replica-124.

| League | Revenue Ops normalized net | CLBOSS normalized net | Revenue margin | Paired 95% CI | Verdict |
|---|---:|---:|---:|---:|---|
| fee_only | 172590.21187 | 291239.132678 | -40.739% | [-284702.085751, 105360.560473] | Inconclusive |
| full_stack | 176563.712897 | 367987.982671 | -52.019% | [-834069.402934, 292972.468213] | Inconclusive |

Historical aggregate economic standing: **Revenue Ops leads the primary net-profit objective** at 2.36x CLBOSS net profit. Raw volume and forward count are diagnostics, not objectives; they matter only when the incremental traffic is profitable.

This table describes observed lab outcomes; it does not treat historical smoke blocks as decisive evidence.

## Current functional comparison

| Comparable functional area | Revenue Ops evidence | CLBOSS evidence | Current result |
|---|---|---|---|
| Fee setting | 3859517 msat net at 156.493 ppm yield | 1635625 msat net at 28.698 ppm yield | Inconclusive (formal) |
| Route acquisition / breadth | 25594581928 msat, 30.201% share | 59152684890 msat, 69.799% share | CLBOSS (diagnostic) |
| Rebalancing and post-refill conversion | 5960000000 msat / 1724570 msat linked net | 4890000000 msat / 1078223 msat linked net | Revenue Ops |
| Selective rebalance economics | 150/120 ppm: 0/2 replicas, 0 sats delivered / 0.000 sats cost; 800/120 ppm: 11/11 replicas, 549580 sats delivered / 79.166 sats cost | 150/120 ppm: 0/2 replicas, 0 sats delivered / 0.000 sats cost; 800/120 ppm: 0/11 replicas, 0 sats delivered / 0.000 sats cost | revenue_ops |
| Liquidity balance | Mean worst imbalance 871132.5 ppm | Mean worst imbalance 791924.4 ppm | clboss |
| Reliability | Strict safety-gated blocks only; shared traffic settled 4326/4329 payments | The same shared traffic and safety gate applies | Not attributable per controller |
| Channel open / close management | Intentionally absent from this standalone plugin | Disabled in the comparable harness | Not comparable |

## Controlled payer-refill economics

Safety-eligible native observations: 13/13. Destination/return fees are shown in ppm; CLBOSS is uncapped.

| Fee band | Controller | Executed replicas | Delivered (sats) | Cost (sats) |
|---|---|---:|---:|---:|
| 150/120 | revenue_ops | 0/2 | 0 | 0.000 |
| 150/120 | clboss | 0/2 | 0 | 0.000 |
| 800/120 | revenue_ops | 11/11 | 549580 | 79.166 |
| 800/120 | clboss | 0/11 | 0 | 0.000 |

## Eligible results by market profile

Only enhanced blocks with no block-level or contender-level safety violations appear below.

### acquisition

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 2195000000 | 805000000 |
| Net routing profit (msat) | 16736 | 6952 |
| Gross yield (ppm) | 7.625 | 8.636 |

### legacy_low_fee

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 5000000 | 445000000 |
| Net routing profit (msat) | 75 | 4870 |
| Gross yield (ppm) | 15.0 | 10.944 |

### realistic

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 19206619123 | 37632125772 |
| Net routing profit (msat) | 3498262 | 1297178 |
| Gross yield (ppm) | 188.103 | 34.864 |

## Eligible results by phase

This view isolates treatments and post-rebalance demand from historical baselines.

### automatic_acquisition

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 625000000 | 75000000 |
| Net routing profit (msat) | 3430 | 535 |
| Gross yield (ppm) | 5.488 | 7.133 |

### automatic_retention

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 1340000000 | 450000000 |
| Net routing profit (msat) | 11840 | 4432 |
| Gross yield (ppm) | 8.836 | 9.849 |

### baseline

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 13216619123 | 32827125772 |
| Net routing profit (msat) | 1772948 | 223118 |
| Gross yield (ppm) | 135.06 | 6.797 |

### manual_acquisition

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 0 | 330000000 |
| Net routing profit (msat) | 0 | 330 |
| Gross yield (ppm) | 0.0 | 1.0 |

### paid_retention

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 265000000 | 310000000 |
| Net routing profit (msat) | 2285 | 2362 |
| Gross yield (ppm) | 8.623 | 7.619 |

### post_rebalance_demand

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 5960000000 | 4890000000 |
| Net routing profit (msat) | 1724570 | 1078223 |
| Gross yield (ppm) | 306.549 | 223.527 |

## Eligible single-family results by phase

Single-family blocks charge their directly linked native rebalance cost to the same client-family phase.

### automatic_acquisition / cln

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 175000000 | 75000000 |
| Rebalance cost (msat) | 0 | 0 |
| Linked net profit (msat) | 918 | 535 |

### automatic_acquisition / lnd

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 450000000 | 0 |
| Rebalance cost (msat) | 0 | 0 |
| Linked net profit (msat) | 2512 | 0 |

### automatic_retention / cln

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 665000000 | 450000000 |
| Rebalance cost (msat) | 0 | 0 |
| Linked net profit (msat) | 6026 | 4432 |

### automatic_retention / lnd

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 675000000 | 0 |
| Rebalance cost (msat) | 0 | 0 |
| Linked net profit (msat) | 5814 | 0 |

### baseline / cln

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 1076619123 | 5553380877 |
| Rebalance cost (msat) | 0 | 0 |
| Linked net profit (msat) | 180647 | 65799 |

### baseline / lnd

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 3305000000 | 330000000 |
| Rebalance cost (msat) | 0 | 0 |
| Linked net profit (msat) | 403120 | 330 |

### manual_acquisition / lnd

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 0 | 330000000 |
| Rebalance cost (msat) | 0 | 0 |
| Linked net profit (msat) | 0 | 330 |

### paid_retention / cln

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 10000000 | 115000000 |
| Rebalance cost (msat) | 0 | 0 |
| Linked net profit (msat) | 60 | 943 |

### post_rebalance_demand / cln

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 2050000000 | 1270000000 |
| Rebalance cost (msat) | 32216 | 0 |
| Linked net profit (msat) | 777489 | 265645 |

### post_rebalance_demand / lnd

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 3910000000 | 3620000000 |
| Rebalance cost (msat) | 70249 | 14822 |
| Linked net profit (msat) | 947081 | 812578 |
