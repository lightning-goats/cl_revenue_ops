# CLBOSS tournament scorecard

Coverage: 104 replicas, 455 blocks, 9050 attempted / 9021 settled payments. Enhanced strict-schema blocks: 438; safety-eligible: 390; diagnostic exclusions: 31.

| Comparable area | Revenue Ops | CLBOSS | Current leader |
|---|---:|---:|---|
| Routing volume (msat) | 142944274159 | 116704466785 | revenue_ops |
| Forward count | 4307 | 4665 | clboss |
| Gross routing fees (msat) | 58946805 | 20784521 | revenue_ops |
| Rebalance cost (msat) | 197116 | 1062724 | revenue_ops |
| Net routing profit (msat) | 58749689 | 19721797 | revenue_ops |
| Gross yield (ppm) | 412.376 | 178.095 | revenue_ops |
| Volume share (%) | 55.053 | 44.947 | revenue_ops |
| Mean worst imbalance (ppm; lower is better) | 843821.4 | 844257.3 | revenue_ops |

Formal verdict: **Revenue Ops wins** from frozen crossed series `paid-retention-budget-393e353-20260830`. All common coverage, reliability, budget, and safety gates passed.

## Formal frozen-series result

This formal result controls tournament promotion; the larger historical aggregate below remains diagnostic. Frozen Revenue Ops revision: `393e3530294c450659e8d0b9b1c8c4e3eb1f00fd`; image: `sha256:73cb441c11d21b3f8d301ef4a94f79f43781653c64f928367fc8e0350a4183fd`; replicas: replica-133, replica-134, replica-135.

| League | Revenue Ops normalized net | CLBOSS normalized net | Revenue margin | Paired 95% CI | Verdict |
|---|---:|---:|---:|---:|---|
| fee_only | 912080.498524 | 109180.756236 | 735.386% | [490202.655978, 1106997.431066] | Revenue Ops wins |
| full_stack | 869742.852861 | 156005.064224 | 457.509% | [353292.419681, 1062826.381274] | Revenue Ops wins |

Historical aggregate economic standing: **Revenue Ops leads the primary net-profit objective** at 2.979x CLBOSS net profit. Raw volume and forward count are diagnostics, not objectives; they matter only when the incremental traffic is profitable.

This table describes observed lab outcomes; it does not treat historical smoke blocks as decisive evidence.

The final-revision crossed results, production-scale balance fix, and next hypotheses are recorded in the [selective displacement strategy](plans/2026-08-30-selective-displacement.md).

## Current functional comparison

| Comparable functional area | Revenue Ops evidence | CLBOSS evidence | Current result |
|---|---|---|---|
| Fee setting | 58749689 msat net at 412.376 ppm yield | 19721797 msat net at 178.095 ppm yield | Revenue Ops wins (formal) |
| Route acquisition / breadth | 142944274159 msat, 55.053% share | 116704466785 msat, 44.947% share | Revenue Ops (diagnostic) |
| Rebalancing and post-refill conversion | 5960000000 msat / 1724570 msat linked net | 4890000000 msat / 1078223 msat linked net | Revenue Ops |
| Selective rebalance economics | 150/120 ppm: 0/2 replicas, 0 sats delivered / 0.000 sats cost; 800/120 ppm: 11/11 replicas, 549580 sats delivered / 79.166 sats cost | 150/120 ppm: 0/2 replicas, 0 sats delivered / 0.000 sats cost; 800/120 ppm: 0/11 replicas, 0 sats delivered / 0.000 sats cost | revenue_ops |
| Liquidity balance | Mean worst imbalance 843821.4 ppm | Mean worst imbalance 844257.3 ppm | revenue_ops |
| Reliability | Strict safety-gated blocks only; shared traffic settled 9021/9050 payments | The same shared traffic and safety gate applies | Not attributable per controller |
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
| Routing volume (msat) | 131731169764 | 93132832589 |
| Net routing profit (msat) | 57611090 | 18990093 |
| Gross yield (ppm) | 438.597 | 214.808 |

## Eligible results by channel capacity

Capacity is matched between contenders inside every replica. Legacy artifacts without an explicit capacity remain separately labeled.

### 2,000,000 sats

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 12420000000 | 6459226358 |
| Net routing profit (msat) | 5091586 | 6463 |
| Gross yield (ppm) | 411.202 | 1.001 |
| Mean worst imbalance (ppm) | 758129.9 | 963048.4 |

### 5,000,000 sats

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 67735000000 | 38055000000 |
| Net routing profit (msat) | 37506113 | 18309799 |
| Gross yield (ppm) | 553.718 | 481.14 |
| Mean worst imbalance (ppm) | 933615.8 | 871357.4 |

### 20,000,000 sats

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 14852197870 | 900000000 |
| Net routing profit (msat) | 10150482 | -730334 |
| Gross yield (ppm) | 683.433 | 300.518 |
| Mean worst imbalance (ppm) | 952139.3 | 873954.7 |

### legacy_unspecified

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 38923971894 | 48968606231 |
| Net routing profit (msat) | 4879720 | 1415987 |
| Gross yield (ppm) | 129.226 | 29.219 |
| Mean worst imbalance (ppm) | 833933.6 | 840828.6 |

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
| Routing volume (msat) | 125741169764 | 88327832589 |
| Net routing profit (msat) | 55885776 | 17916033 |
| Gross yield (ppm) | 444.955 | 214.166 |

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
