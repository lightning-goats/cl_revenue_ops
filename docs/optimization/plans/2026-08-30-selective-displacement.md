# Selective displacement strategy

## Decision

Revenue Ops should apply competitive force only where local evidence predicts
positive incremental net profit. The objective is not maximum routed volume. It
is to preserve proven high-yield traffic, take profitable traffic that would
otherwise pay a competing route, and withdraw the treatment when its marginal
profit turns negative.

The first implementation prices a channel at one ppm below a route-specific
competitor floor after Revenue Ops has directly observed a successful
acquisition route. The normal inventory adjustment may contribute no more than
50 ppm of credit, and the proven target is pinned only when it remains inside
that bound. Missing, stale, malformed, or failed evidence is neutral.

## Final-revision CLBOSS validation

Revision `9c46aaf3dd67b936555fe714faba29cd29c96562` ran on the equal-runtime
CLN v26.06.7 image against uncapped CLBOSS `8cb4e92`. Each assignment used five
accepted eight-round blocks with mixed CLN/LND traffic. Two blocks with
unattributed multipart volume were excluded and replaced.

| Assignment | Revenue Ops profit (msat) | CLBOSS profit (msat) | Profit ratio | Revenue Ops volume (msat) | CLBOSS volume (msat) | Volume ratio | Block wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| Revenue Ops identity B (replica 172) | 5,647,090 | 108,496 | 52.05x | 6,255,000,000 | 195,000,000 | 32.08x | 5-0 |
| Revenue Ops identity A (replica 173) | 4,018,330 | 867,465 | 4.63x | 5,290,000,000 | 1,160,000,000 | 4.56x | 5-0 |
| Combined | 9,665,420 | 975,961 | 9.90x | 11,545,000,000 | 1,355,000,000 | 8.52x | 10-0 |

All accepted blocks settled all 320 attempted payments with exact contender
volume attribution, no fallback, no rebalance spend, and no safety violation.
The result supports selective displacement as a profit strategy, but it does
not prove that every additional unit of volume is profitable.

## Expansion backlog

Implement and tournament-test these independently, in priority order. Promote
a treatment only when crossed replicas show positive incremental net profit
after all execution costs, without degrading retained profitable traffic.

| Priority | Selective use of force | Treatment | Required evidence and rollback |
|---:|---|---|---|
| 1 | Temporal fee displacement | Undercut only during observed demand windows, then restore the profitable baseline. | Compare displaced fees, incremental net profit, and post-restoration retention; restore immediately after negative marginal profit or expired evidence. |
| 2 | Payment-size price shaping | Choose base-fee/ppm pairs that undercut the competitor only for profitable payment-size bands. | Require local size-bucket forwarding history and route-specific competitor cost; fall back to the existing fee model for sparse or malformed buckets. |
| 3 | Selective rebalance bids | Buy liquidity only where forecast displaced revenue plus inventory value exceeds the complete route cost. | Gate on conservative payback, settled outcome, and budget reservation; stop after negative realized payback or repeated no-route/failure outcomes. |
| 4 | Rebalance budget concentration | Rank liquidity deficits by expected incremental profit rather than imbalance alone. | Track cost, delivered liquidity, subsequent routed fees, and opportunity cost against an untreated/control channel. |
| 5 | Channel-role coordination | Pair cheap, saturated egress acquisition with scarce, high-yield ingress protection instead of repricing channels independently. | Require evidence for both channel roles; roll back the pair if either leg loses net profit or worsens availability. |
| 6 | Selective HTLC envelope | Adjust HTLC minima/maxima only when a profitable size band is being lost and the policy will not block retained traffic. | First add policy simulation and read-only reporting; do not mutate HTLC limits until malformed-input, sparse-data, and traffic-loss rollback tests pass. |

Every experiment must report competitor-fee displacement, incremental net
profit, inventory change, persistence after treatment restoration, and the
counterfactual or control used. Volume is a supporting metric. Profit after
rebalance and other execution costs remains the promotion objective.

## Deferred xrebalance application

After the CLBOSS tournament is closed, brainstorm and freeze a standalone
xrebalance comparison protocol before executing it. Test both adaptation of
xrebalance's route search/scheduling/retry behavior into the native rebalancer
and the alternative of using xrebalance as an execution backend under Revenue
Ops' economic policy. Revenue Ops must retain budget ownership, settlement
safety, and the selective-profit gate in either design.
