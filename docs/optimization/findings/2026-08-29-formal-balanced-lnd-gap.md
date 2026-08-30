# Formal balanced-traffic LND gap

## Frozen baseline

The first promotion-grade tournament series used Revenue Ops revision
`35d77d6f3b17b0a2c398333452e95f20a17afaca` and one pinned CLN v26.06.7 image
across fresh replicas 113, 116, and 117. Revenue Ops ran on identity A, B, and A;
uncapped CLBOSS ran on the opposite identity. Each replica contributed six
fee-only and six full-stack realistic blocks, with an independently cold first
block and warm repeats in each league.

Before scored selection, eight exact-path payments proved both controllers,
both CLN/LND clients, and both directions. These payments are recorded as
unscored. Across the 36 scored blocks, all 288 payments settled, no fallback
route was used, and every reliability, budget, attribution, and safety gate
passed.

## Formal result

| League | Revenue Ops normalized net | CLBOSS normalized net | Revenue margin | Paired 95% CI | Verdict |
|---|---:|---:|---:|---:|---|
| Fee-only | 164438.613600 | 267498.156286 | -38.527% | [-260285.124258, 103860.296520] | Inconclusive |
| Full-stack | 121596.016526 | 281690.309763 | -56.833% | [-271949.111762, -26442.205879] | CLBOSS wins |

Revenue Ops won CLN fee capture in fee-only, 11,043 versus 6,948 msat, and was
near parity in full-stack, 8,209 versus 8,001 msat. The decisive regression was
LND: Revenue Ops captured zero scored LND fees while CLBOSS captured 11,016
msat in each league. Balanced demand triggered no native rebalance by either
controller, so rebalance economics remain supported by the separate controlled
payer-refill fixtures rather than this series.

The larger historical smoke aggregate still favors Revenue Ops net profit, but
it cannot override this frozen result. The living scorecard therefore uses the
formal series for promotion and labels the historical aggregate diagnostic.

## Diagnosis

Live policy readback on replica 117 showed Revenue Ops still quoting about 150
ppm on both LND lanes while CLBOSS had moved to roughly 1--24 ppm and acquired
the balanced traffic. The lanes had no scored forward evidence before route
selection. The controller's explicit 15-second tournament cadence was active
in both leagues, but the active fee profile retained a 15-minute time window;
the full 162-second sequence could therefore finish without a zero-flow LND
reprice. This is a controller-response gap, not a route-readiness, settlement,
or fallback artifact.

## Bounded treatment

For an explicitly configured fast fee cadence, the minimum observation window
now becomes the smaller of the profile window and three configured cycles,
with a two-minute floor. Thus the 15-second fake-sat cadence gets a two-minute
window. The production-default 1,800-second fee cadence still gets the existing
15-minute active-profile window. Missing, malformed, non-finite, zero, or
negative cadence input fails closed to the profile window.

The next tournament must use a newly pinned product image and the same three
fresh crossed replicas, cold/warm split, traffic schedule, score gates, and
uncapped CLBOSS configuration. Promotion requires a greater-than-10% normalized
net advantage with a positive paired interval, no client-family fee regression
beyond 5%, and no reliability, budget, attribution, or safety regression.
