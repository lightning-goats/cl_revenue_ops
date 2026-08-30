# CLBOSS tournament scorecard

Coverage: 97 replicas, 448 blocks, 8154 attempted / 8125 settled payments. Enhanced strict-schema blocks: 431; safety-eligible: 383; diagnostic exclusions: 27.

| Comparable area | Revenue Ops | CLBOSS | Current leader |
|---|---:|---:|---|
| Routing volume (msat) | 111632076289 | 112137590162 | clboss |
| Forward count | 3539 | 4534 | clboss |
| Gross routing fees (msat) | 37293828 | 19462865 | revenue_ops |
| Rebalance cost (msat) | 193782 | 61924 | clboss |
| Net routing profit (msat) | 37100046 | 19400941 | revenue_ops |
| Gross yield (ppm) | 334.078 | 173.562 | revenue_ops |
| Volume share (%) | 49.887 | 50.113 | clboss |
| Mean worst imbalance (ppm; lower is better) | 842994.7 | 842945.5 | clboss |

Formal verdict: **Revenue Ops wins** from frozen crossed series `paid-retention-budget-393e353-20260830`. All common coverage, reliability, budget, and safety gates passed.

## Formal frozen-series result

This formal result controls tournament promotion; the larger historical aggregate below remains diagnostic. Frozen Revenue Ops revision: `393e3530294c450659e8d0b9b1c8c4e3eb1f00fd`; image: `sha256:73cb441c11d21b3f8d301ef4a94f79f43781653c64f928367fc8e0350a4183fd`; replicas: replica-133, replica-134, replica-135.

| League | Revenue Ops normalized net | CLBOSS normalized net | Revenue margin | Paired 95% CI | Verdict |
|---|---:|---:|---:|---:|---|
| fee_only | 912080.498524 | 109180.756236 | 735.386% | [490202.655978, 1106997.431066] | Revenue Ops wins |
| full_stack | 869742.852861 | 156005.064224 | 457.509% | [353292.419681, 1062826.381274] | Revenue Ops wins |

Historical aggregate economic standing: **Revenue Ops leads the primary net-profit objective** at 1.912x CLBOSS net profit. Raw volume and forward count are diagnostics, not objectives; they matter only when the incremental traffic is profitable.

This table describes observed lab outcomes; it does not treat historical smoke blocks as decisive evidence.

## Final-revision crossed validation

Revenue Ops revision `9c46aaf3dd67b936555fe714faba29cd29c96562`
was tested in both identity assignments against uncapped CLBOSS `8cb4e92` on
the equal-runtime CLN v26.06.7 image. Two attribution-contaminated multipart
blocks were explicitly excluded and replaced. All 10 accepted blocks settled
all 320 attempted payments with exact attribution, no fallback, no rebalance
spend, and no safety violation.

| Assignment | Revenue Ops net profit (msat) | CLBOSS net profit (msat) | Profit ratio | Revenue Ops volume (msat) | CLBOSS volume (msat) | Volume ratio | Blocks |
|---|---:|---:|---:|---:|---:|---:|---:|
| Revenue Ops identity B | 5,647,090 | 108,496 | 52.05x | 6,255,000,000 | 195,000,000 | 32.08x | 5-0 |
| Revenue Ops identity A | 4,018,330 | 867,465 | 4.63x | 5,290,000,000 | 1,160,000,000 | 4.56x | 5-0 |
| Combined | 9,665,420 | 975,961 | 9.90x | 11,545,000,000 | 1,355,000,000 | 8.52x | 10-0 |

The implementation and next selective-force hypotheses are recorded in the
[selective displacement strategy](plans/2026-08-30-selective-displacement.md).

## Current functional comparison

| Comparable functional area | Revenue Ops evidence | CLBOSS evidence | Current result |
|---|---|---|---|
| Fee setting | 37100046 msat net at 334.078 ppm yield | 19400941 msat net at 173.562 ppm yield | Revenue Ops wins (formal) |
| Route acquisition / breadth | 111632076289 msat, 49.887% share | 112137590162 msat, 50.113% share | CLBOSS (diagnostic) |
| Rebalancing and post-refill conversion | 5960000000 msat / 1724570 msat linked net | 4890000000 msat / 1078223 msat linked net | Revenue Ops |
| Selective rebalance economics | 150/120 ppm: 0/2 replicas, 0 sats delivered / 0.000 sats cost; 800/120 ppm: 11/11 replicas, 549580 sats delivered / 79.166 sats cost | 150/120 ppm: 0/2 replicas, 0 sats delivered / 0.000 sats cost; 800/120 ppm: 0/11 replicas, 0 sats delivered / 0.000 sats cost | revenue_ops |
| Liquidity balance | Mean worst imbalance 842994.7 ppm | Mean worst imbalance 842945.5 ppm | clboss |
| Reliability | Strict safety-gated blocks only; shared traffic settled 8125/8154 payments | The same shared traffic and safety gate applies | Not attributable per controller |
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
| Routing volume (msat) | 100418971894 | 88565955966 |
| Net routing profit (msat) | 35961447 | 18669237 |
| Gross yield (ppm) | 359.732 | 210.962 |

## Eligible results by channel capacity

Capacity is matched between contenders inside every replica. Legacy artifacts without an explicit capacity remain separately labeled.

### 2,000,000 sats

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 4500000000 | 4272349735 |
| Net routing profit (msat) | 361460 | 4275 |
| Gross yield (ppm) | 83.037 | 1.001 |
| Mean worst imbalance (ppm) | 739924.5 | 960587.5 |

### 5,000,000 sats

| Metric | Revenue Ops | CLBOSS |
|---|---:|---:|
| Routing volume (msat) | 59195000000 | 36575000000 |
| Net routing profit (msat) | 30737078 | 17260797 |
| Gross yield (ppm) | 519.251 | 471.929 |
| Mean worst imbalance (ppm) | 935969.6 | 869055.1 |

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
| Routing volume (msat) | 94428971894 | 83760955966 |
| Net routing profit (msat) | 34236133 | 17595177 |
| Gross yield (ppm) | 363.195 | 210.064 |

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
