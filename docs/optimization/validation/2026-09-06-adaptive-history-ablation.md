# Adaptive historical influence: frozen development experiment

## Mechanism and source review

Follows the [online count ablation](2026-09-06-online-history-ablation.md), which
rejected naive historical-count retention in every tested period/context.
Keep those losses and the original static losses; do not rename them as wins.

The research candidate borrows normalized likelihood weighting and cross-expert
weight sharing from Fixed Share, reviewed in section 2, equations 3b and 5 of
Adamskiy et al., [A Closer Look at Adaptive Regret](https://jmlr.org/papers/volume17/13-533/13-533.pdf).
For two experts the update is `posterior = w*p_w / (w*p_w + (1-w)*p_c)`, followed
by `w_next = alpha + (1-2*alpha)*posterior`. Sharing lets a previously losing
expert recover. This is an established mechanism, not a novelty claim.

Section 2 and algorithm 1 of Joulani et al.,
[Online Learning under Delayed Feedback](https://proceedings.mlr.press/v28/joulani13.pdf)
separate predictions from timestamped outcomes that can arrive out of order.
Their BOLD construction uses free/busy learner instances; this candidate does
**not** implement BOLD or claim its theorem. It uses one chronological gate per
context and retains each original forecast until settlement. Neither the
asynchronous update order nor our additional influence cap is covered by the
synchronous Fixed Share guarantee. No theorem is asserted for selected Lightning
settlements, price response or net earnings.

Our bounded variant caps the applied historical weight at one half after each
update. Thus `p_mix >= 0.5*p_cold`, so its per-event log loss exceeds the cold
predictor's by at most one bit. This direct algebraic bound is **not** a guarantee
of average improvement or an economic safety bound. The cap can also discard
useful historical influence; test that tradeoff instead of assuming it helps.

## Frozen card, before score inspection

- Same January 4 prefix and revealed half-open UTC suffixes June 7–July 7,
  July 7–August 6 and August 6–September 5. Same raw-to-coverage-checked,
  read-only canonical reader. The missing January 3 is not repaired or hidden.
- Keep both base experts exactly as published: 30-day decay, shrinkage 20,
  pooled/outgoing/outgoing-plus-amount contexts, 50k/250k-sat amount buckets,
  shared prefix alphabet with settlement-available expansion.
- Five arms for every context: cold counts; warm counts; fixed 50/50 blend;
  adaptive uncapped blend; adaptive capped blend. Initial warm weight 0.5,
  share `alpha=0.01` per available forecast outcome, capped weight maximum 0.5.
  The share is a frozen hypothesis (not fitted to production scores).
- Store forecast probabilities at received time. Use them for gate likelihood
  updates only after the event's settlement time is strictly earlier than the
  next prediction. Equal-time outcomes cannot affect that instant's predictions.
  Never recompute a past forecast using a model already trained on its answer.
- Late prefix settlements update both base experts but not gates: they had no
  issued gate forecast. Unresolved-at-end outcomes never train. Native archive
  duplicate identities are refused. This is not runtime source continuity or
  exactly-once model persistence qualification.
- Primary hypothesis: the capped adaptive outgoing-plus-amount predictor has
  lower mean loss than cold outgoing-plus-amount in **all three** periods.
  Report every arm/context regardless of outcome; no post-score parameter
  selection. Other contexts, uncapped gating and fixed blending are diagnostic
  ablations. All periods remain development evidence, never sealed confirmation.
- Report loss, daily paired differences, actual historical weights/ranges,
  pending forecasts, gate/base updates and unknown/boundary counts. Pending
  state is bounded by the 50,000-row input ceiling, not a production memory SLA.
- No code installation, raw/model export, fee/rebalance action, environment or
  competitor change. Settlement availability remains approximate and target
  selection remains conditioned on eventual settlement. No demand, causal fee
  response, inventory valuation or economic superiority follows from this score.

## Results and verification

All three runs completed read-only on production revision
`294e649783d0aadc1df40fe035d4acd39e1ca35e`, without installation, raw/model export or action RPCs.
The unchanged base expert scores matched `evaluate_online` exactly in the same
source snapshots. Each read passed the existing raw-to-closed-day-coverage checks.
This is not live wallet-generation continuity or operational accounting repair.

Source SHA-256:

- `historical_adaptive_context_replay.py`: `310831432d4f877427b8a93fe98087d2003c0f01ec57660e0cf0db9ff3242dc9`.
- `historical_online_context_replay.py`: `c4fc7367d8c0fd18ab9a93ad779529d477d472f7e853cb8a8c6ce6da7434043b`.
- `historical_route_context_replay.py`: `23310d9fc6a95dc8d9dc2a1868b7ad6834406371c1f9c02a106fdd968d826fb2`.

Primary outgoing-plus-amount mean log loss (bits, lower is better):

| Suffix (UTC) | Events | Cold | Warm | Fixed half | Adaptive | Adaptive capped | Capped improvement over cold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| June 7–July 7 | 1700 | 2.473384 | 2.726933 | 2.524382 | 2.435411 | 2.433352 | 1.619% |
| July 7–August 6 | 1918 | 2.821017 | 3.386285 | 2.942745 | 2.789845 | 2.780097 | 1.451% |
| August 6–September 5 | 2109 | 2.578349 | 3.163494 | 2.711143 | 2.573441 | 2.560396 | 0.696% |

All-context mean-loss differences from cold (negative is better):

| Suffix | Context | Fixed half − cold | Adaptive − cold | Capped − cold |
| --- | --- | ---: | ---: | ---: |
| June 7–July 7 | pooled | 0.136432 | -0.037598 | -0.024935 |
| June 7–July 7 | outgoing | 0.059324 | -0.034323 | -0.030608 |
| June 7–July 7 | outgoing_amount | 0.050998 | -0.037973 | -0.040032 |
| July 7–August 6 | pooled | 0.202410 | -0.027746 | -0.028130 |
| July 7–August 6 | outgoing | 0.134009 | -0.025899 | -0.031810 |
| July 7–August 6 | outgoing_amount | 0.121728 | -0.031172 | -0.040920 |
| August 6–September 5 | pooled | 0.137279 | -0.047784 | -0.044696 |
| August 6–September 5 | outgoing | 0.128610 | -0.009095 | -0.018504 |
| August 6–September 5 | outgoing_amount | 0.132794 | -0.004908 | -0.017953 |

The predeclared primary development hypothesis passed all three periods.
Both adaptive arms beat cold in all nine context/period comparisons; fixed
blending lost all nine. The cap helped outgoing-plus-amount in every period,
but did **not** improve every other context versus uncapped gating. It is
not universally superior. The original warm/cold losses remain unchanged.

Capped outgoing-plus-amount mean historical weights were 0.153255, 0.123767,
and 0.142060. Their observed maxima were exactly 0.5, minima approximately
0.01049, 0.01001 and 0.01001. Uncapped means were 0.234550/0.202568/0.190750,
with maxima above 0.989 in each period. These are actual prediction weights,
not unapplied proposals.

Capped primary daily losses beat cold on only 14/30, 13/30 and 14/29 scored
days. Mean improvement is not a claim of a typical-day win or statistical
significance; gains can concentrate in particular events/days. Unknown labels
were 7/16/9; late-prefix and unresolved-at-end counts were zero in these runs.
Synthetic tests exercise those missing cases. Maximum pending forecasts were
12/7/7. Gate and cold updates before the last prediction were 1,699/1,917/2,108;
warm updates were 8,550/10,468/12,577. One final forecast remained pending in
each run and could not train its own prediction.

The production process, including all three reader/evaluator/base-parity runs,
took 0.808 seconds and reported maximum RSS 38,264 KiB. A separate local
50,000-event synthetic check scored 24,224 events in 0.553 seconds with maximum
RSS 42,032 KiB and one pending forecast at peak. That synthetic check has 64
incoming/32 outgoing labels and short settlements; it is not worst-case
cardinality, concurrency or pending-queue qualification.

## Qualification boundary and next action

This is the first positive development result in this historical-mixture
sequence, not confirmation of a production learner. A reasonable next step
is a frozen confirmation and a downstream policy ablation using this estimator
only where local evidence identifies a decision-relevant quantity. A more
accurate incoming mixture cannot by itself value inventory, estimate avoided
demand, identify a fee elasticity or attribute rebalance returns.

Do not activate it as a fee multiplier. First resolve source/model/cursor
admission and specify how predicted mixture changes a feasible local decision,
including neutral fallback, uncertainty, realistic fee rails and independent
budget enforcement. Qualification still requires unchanged native competitors,
topology, traffic, payer history, cadence and scorer plus original delivery,
retention, economic, replication and sealed-holdout gates. No predictive score
substitutes for those requirements.

Focused historical suites: **82 passed** (33 new adaptive tests). Full isolated
suite with `CLN_INTEGRATION=0`: **4,714 passed, five skipped, two existing
expected failures**, 191.96 seconds. Skips were four opt-in live-router cases
and unavailable optional `pyln.testing`; expected failures were the existing
staged-removal cases. No thresholds changed. This successful run does not
diagnose or erase the capture timing failure recorded in the prior ablation.
Tests include equation/cap checks, recovery after
a regime change, tiny/malformed likelihoods, no mutation on invalid feedback,
exact base-expert parity, saved forecasts through out-of-order settlements,
equal-time withholding, late-prefix exclusion from gate fitting, duplicate
and row-budget refusal, read-only source preservation and aggregate-only output.

Files changed: offline adaptive evaluator, tests and this evidence note.
No runtime or production schema/configuration changes; compatibility is unchanged.
No Sling, external coordinator, Archon DID or action RPCs. No competitor or
test-environment changes. Remaining risks include availability approximation,
settlement selection, stale/source-ambiguous history, unqualified persistence,
small revealed-period gains and no demonstrated economic advantage.
