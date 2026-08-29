# CLBOSS tournament scorecard

Coverage: 51 replicas, 88 blocks, 3209 attempted / 3206 settled payments. Enhanced strict-schema blocks: 71; safety-eligible: 56; diagnostic exclusions: 24.

| Comparable area | Revenue Ops | CLBOSS | Current leader |
|---|---:|---:|---|
| Routing volume (msat) | 22234581928 | 42303945739 | clboss |
| Forward count | 1173 | 1903 | clboss |
| Gross routing fees (msat) | 2609818 | 885570 | revenue_ops |
| Rebalance cost (msat) | 57732 | 61924 | revenue_ops |
| Net routing profit (msat) | 2552086 | 823646 | revenue_ops |
| Gross yield (ppm) | 117.377 | 20.934 | revenue_ops |
| Volume share (%) | 34.452 | 65.548 | clboss |
| Mean worst imbalance (ppm; lower is better) | 838557.7 | 856371.6 | revenue_ops |

Formal verdict: **not ready**. It requires at least 3 fresh replicas and 6 enhanced cold/warm blocks per league per replica.

Economic standing: **Revenue Ops leads the primary net-profit objective** at 3.099x CLBOSS net profit. Raw volume and forward count are diagnostics, not objectives; they matter only when the incremental traffic is profitable.

This table describes observed lab outcomes; it does not treat historical smoke blocks as decisive evidence.

## Current functional comparison

| Comparable functional area | Revenue Ops evidence | CLBOSS evidence | Current result |
|---|---|---|---|
| Fee setting | 2552086 msat net at 117.377 ppm yield | 823646 msat net at 20.934 ppm yield | Revenue Ops |
| Route acquisition / breadth | 22234581928 msat, 34.452% share | 42303945739 msat, 65.548% share | CLBOSS (diagnostic) |
| Rebalancing and post-refill conversion | 4305000000 msat / 553785 msat linked net | 4065000000 msat / 418223 msat linked net | Revenue Ops |
| Selective rebalance economics | 150/120 ppm: 0/2 replicas, 0 sats delivered / 0.000 sats cost; 800/120 ppm: 2/2 replicas, 100000 sats delivered / 14.004 sats cost | 150/120 ppm: 0/2 replicas, 0 sats delivered / 0.000 sats cost; 800/120 ppm: 0/2 replicas, 0 sats delivered / 0.000 sats cost | revenue_ops |
| Liquidity balance | Mean worst imbalance 838557.7 ppm | Mean worst imbalance 856371.6 ppm | revenue_ops |
| Reliability | Strict safety-gated blocks only; shared traffic settled 3206/3209 payments | The same shared traffic and safety gate applies | Not attributable per controller |
| Channel open / close management | Intentionally absent from this standalone plugin | Disabled in the comparable harness | Not comparable |

## Controlled payer-refill economics

Safety-eligible native observations: 4/4. Destination/return fees are shown in ppm; CLBOSS is uncapped.

| Fee band | Controller | Executed replicas | Delivered (sats) | Cost (sats) |
|---|---|---:|---:|---:|
| 150/120 | revenue_ops | 0/2 | 0 | 0.000 |
| 150/120 | clboss | 0/2 | 0 | 0.000 |
| 800/120 | revenue_ops | 2/2 | 100000 | 14.004 |
| 800/120 | clboss | 0/2 | 0 | 0.000 |

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
| Routing volume (msat) | 15961619123 | 21438380877 |
| Net routing profit (msat) | 2183306 | 497174 |
| Gross yield (ppm) | 139.56 | 23.882 |

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
| Routing volume (msat) | 11626619123 | 17458380877 |
| Net routing profit (msat) | 1628777 | 83114 |
| Gross yield (ppm) | 140.09 | 4.761 |

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
| Routing volume (msat) | 4305000000 | 4065000000 |
| Net routing profit (msat) | 553785 | 418223 |
| Gross yield (ppm) | 138.929 | 106.53 |

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
| Routing volume (msat) | 1140000000 | 940000000 |
| Rebalance cost (msat) | 4208 | 0 |
| Linked net profit (msat) | 155752 | 1645 |

### post_rebalance_demand / lnd

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 3165000000 | 3125000000 |
| Rebalance cost (msat) | 40097 | 14822 |
| Linked net profit (msat) | 398033 | 416578 |
