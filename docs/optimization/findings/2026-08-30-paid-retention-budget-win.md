# Paid-retention evidence budget: formal CLBOSS win

## Frozen treatment

Revenue Ops revision `393e3530294c450659e8d0b9b1c8c4e3eb1f00fd`
ran from image
`sha256:73cb441c11d21b3f8d301ef4a94f79f43781653c64f928367fc8e0350a4183fd`
against uncapped CLBOSS on fresh crossed replicas 133, 134, and 135. The image
contains Core Lightning v26.06.7. Each replica contributed six fee-only and
six full-stack realistic blocks, including one cold and five warm blocks per
league.

All 288 scored payments settled. No fallback route, safety violation, or
rebalance cost was recorded. CLBOSS remained uncapped; Revenue Ops received
the tournament's 1,000-sat native-rebalance allowance but did not spend it.

## Result

| League | Revenue Ops normalized net | CLBOSS normalized net | Revenue margin | Paired 95% CI | Verdict |
|---|---:|---:|---:|---:|---|
| Fee-only | 912080.498524 | 109180.756236 | +735.386% | [490202.655978, 1106997.431066] | Revenue Ops wins |
| Full-stack | 869742.852861 | 156005.064224 | +457.509% | [353292.419681, 1062826.381274] | Revenue Ops wins |

Revenue Ops won gross fee capture in both client families. In fee-only it
earned 58,686 versus 7,025 msat and routed 1.880B versus 0.460B msat. In
full-stack it earned 56,220 versus 10,084 msat while routing 1.055B versus
1.285B msat. The latter is the expected profit/volume distinction: CLBOSS can
carry somewhat more traffic without creating more profit.

The formal scorer returns `revenue_ops_wins`; every coverage, reliability,
budget, family, and safety promotion gate passes, and both paired confidence
intervals are strictly positive.

## Diagnosis and treatment

The preceding revision `bcd0f51` proved two distinct low-fee markets and
converted both to a positive `1 msat + 0 ppm` quote. It improved fee-only
normalized net by 83.0%, but full-stack native fee cycles restored the old
150-ppm baseline after roughly 170,000 routed sats. The free acquisition phase
and paid validation phase shared the same 25-sat modeled opportunity-cost cap;
the controller therefore treated revenue that had never cleared at 150 ppm as
if it were being forgone. LND then selected CLBOSS's 1-ppm route and Revenue
Ops captured no LND fee in full-stack.

The winning treatment keeps free acquisition unchanged at one hour, 250,000
routed sats, and 25 sats of modeled opportunity cost. Only a lane that has
already proved demand and converted to a positive quote receives a separate
paid-validation budget: one hour, 1,000,000 additional routed sats, and 250
sats of modeled opportunity cost. Congestion, a 70% outbound-liquidity stop,
competitor drift, and malformed evidence still exit fail closed, and at most
two distinct peer markets may be active.

Pilot replica 132 verified that both CLN and LND sink policies remained exactly
`1 msat + 0 ppm` through native full-stack fee cycles. Revenue Ops then won
all six league/replica earnings comparisons in the formal triplicate,
including the formerly failing crossed identity-B assignment.

## Residual risk and next round

This is decisive evidence for the frozen balanced realistic workload, not a
claim that the smallest positive quote is optimal for every production market.
The low quote is a bounded, peer-local validation policy rather than the
global fee floor. Production observations with different competitor fees,
payment sizes, or liquidity demand can reach the time, volume, opportunity,
liquidity, congestion, or evidence-drift exits sooner.

Before changing another default, repeat the frozen treatment with a new
traffic seed. After the CLBOSS confirmation closes, run the deferred
xrebalance-versus-Revenue-Ops tournament and decide from net profit, delivered
liquidity, cost, and post-refill earnings whether to adapt xrebalance behavior
or use xrebalance as the execution engine.
