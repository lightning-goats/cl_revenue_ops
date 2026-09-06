# Default-off joint-credit rebalance gate

## Disposition

This is an executable conservative baseline for the
[paired-credit counterexample](2026-09-06-pair-credit-overlap.md), not a
production-qualified repair, a learned joint-value controller, or proof of
higher earnings. It changes Revenue Ops only. No competitor, traffic,
topology, payer state, scoring rule, cadence, fee ceiling or production
configuration was changed. No native tournament was run for this candidate.

The startup option `revenue-ops-rebalance-value-model` accepts `legacy_sum`
(unchanged default) or `joint_lower_bound`. It cannot be changed through the
runtime configuration RPC or a persisted override. Invalid startup values
are rejected. Existing manually constructed configuration snapshots retain
the legacy default. Risk profiles classify the selector as `advanced_expert`,
never bundle it, and preserve its startup value under every profile.

## What changes

For the incumbent destination and source fee credits D and S, the candidate
records the possible joint-credit interval `[max(D, S), D + S]` and uses its
lower endpoint in the priced-pair EV gate. Unknown overlap remains explicit;
it is not inferred from historical incoming-event share. Nonfinite, negative,
boolean or otherwise malformed bounds are refused. Invalid joint credit
cannot pass the existing free-route exception.

This interval assumes nonnegative credits on the same future evidence basis.
It is **not a statistical confidence interval or guaranteed realized-profit
bound**: the incumbent marginal forecasts and causal effects remain
unqualified. Replacing addition with a maximum cannot fix inaccurate
marginals. The activity penalty retains its original summed-credit cap so
this candidate cannot offset lower benefits by silently shrinking a cost.
Probability, opportunity cost, failure penalty, budget and fee rules stay
unchanged. Valid zero-fee routes retain the incumbent gate bypass; zero fees
do not establish zero opportunity cost, which remains a separate limitation.

Actual priced selection, using the same synthetic pair within the 1,200-ppm
rail, demonstrates the intended difference:

| Route fee, sats | Legacy margin, sats | Candidate margin, sats | Candidate selection |
| --- | ---: | ---: | --- |
| 105 | +1.4 | -8.5 | Rejected |
| 90 | +16.4 | +6.5 | Accepted |

These are modeled margins, not observed earnings. If the two future benefits
are genuinely disjoint, the +1.4-sat move could be worthwhile; this baseline
still rejects it. The test preserves that lost-opportunity counterexample
instead of asserting universal superiority.

## Replay and verification boundary

Candidate scores are explicitly tagged `v3-joint-lower-bound`. The independent
replay implementation recomputes the maximum from primitive score inputs
without importing the live bounds helper. Existing `v2-sats-ev` records keep
their original summed interpretation; unknown versions fail closed. Old
readers do not support the new score version. The wire envelope schema stays
at v0 because its model-version field already permits nonempty strings.

A sealed real-planner fixture with synthetic priced evidence passes the CLI
replay with the candidate tag. Relabeling that score as legacy yields an EV
mismatch. This verifies the reader and arithmetic contract, not router
execution, causal validity, or replay of the rejected-pair universe. The
reader checks final selected scores and takes the recorded activity penalty
as input; it does not independently validate the optional joint-credit
diagnostic object. Malformed upstream pair fields are not comprehensively
sanitized by this patch; bounds refusal is not a general input-cleanup claim.

The first full isolated suite reported 4,760 passed, two failed, five skipped
and two existing expected failures in 175.85 seconds. Both failures were
omissions in this candidate: the new Config field was missing from the risk
classification and compatibility catalog. Added its expert-only
classification and default catalog entry without weakening either coverage
test. An additional regression verifies that every risk profile preserves
both possible startup modes. The candidate, risk-profile and full replay-CLI
group then passed 69 tests in 1.25 seconds. Earlier focused configuration,
economics, EV, wire and RPC-inventory tests passed 364 tests.

The final full isolated suite passed **4,763 tests**, with five skips and two
existing expected failures, in 169.29 seconds. This includes 31 candidate
tests and the architecture/RPC guards. Skips are four opt-in live-router
tests and unavailable optional `pyln.testing`; expected failures are existing
staged-removal tests. All tests use mocks or local fixtures; no live test was
enabled. `git diff --check` passed. These software results do not satisfy the
native economic promotion gates.

## Relationship to historical learning

Nine months of production history can seed useful models, but retaining all
old evidence at fixed weight did not help the earlier online-count ablation.
The [adaptive-history experiment](2026-09-06-adaptive-history-ablation.md)
improved incoming-context prediction in the three inspected periods; that is
not evidence of learned incremental fee income or rebalance profitability.
This gate deliberately does not load that research model into production.

Historical bootstrap still needs validated source/generation continuity,
receipt-identified accounting migration, delayed-outcome handling, and a
warm-start-versus-cold economic ablation. A future joint-value challenger must
predict fee-weighted incremental income on a common horizon, count overlap
once, represent unknown causal effects, and recover disjoint benefits only
when evidence supports them. Compare it with this conservative baseline and
the unchanged incumbent under the frozen native tournament protocol. A run
with no relevant rebalance decisions cannot qualify this gate. Do not alter
the environment to produce favorable results; retain losses and insufficient
activity as such. All original native-competitor, retention, net-yield,
replication and holdout promotion gates remain required.

## Operator handoff

Files: joint-value helper, rebalance engine, configuration and startup option,
full example configuration, risk classification and compatibility catalog,
independent EV replay and wire reader, candidate tests, and this evidence note.
No Sling, coordinator or Archon DID was added.
No action RPC, production read/write, database migration or deployment was
performed in this change. Production remains on its existing configuration;
this candidate must remain disabled until qualified. No historical model or
native-receipt admission is activated by selecting this mode. Principal risks
are conservative underinvestment, inaccurate marginal forecasts, the free-
route exception, and unproven economic performance.
