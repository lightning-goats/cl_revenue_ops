# CLBOSS tournament scorecard

Coverage: 10 replicas, 17 blocks, 1000 attempted / 1000 settled payments. Enhanced strict-schema blocks: 0.

| Comparable area | Revenue Ops | CLBOSS | Current leader |
|---|---:|---:|---|
| Routing volume (msat) | 808589034 | 8289935977 | clboss |
| Forward count | 123 | 703 | clboss |
| Gross routing fees (msat) | 8216 | 48763 | clboss |
| Rebalance cost (msat) | 0 | 0 | tie |
| Net routing profit (msat) | 8216 | 48763 | clboss |
| Gross yield (ppm) | 10.161 | 5.882 | revenue_ops |
| Volume share (%) | 8.887 | 91.113 | clboss |
| Mean worst imbalance (ppm; lower is better) | 410322.8 | 381442.5 | clboss |

Formal verdict: **not ready**. It requires at least three fresh replicas and six enhanced cold/warm blocks per league per replica.

This table describes observed lab outcomes; it does not treat historical smoke blocks as decisive evidence.

## What the tournament has established

- Revenue Ops extracts more fee per routed sat, but CLBOSS wins far more routing volume. The main economic gap is conversion and retained demand, not fee arithmetic alone.
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
