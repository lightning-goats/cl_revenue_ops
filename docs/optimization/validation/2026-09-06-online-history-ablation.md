# Historical-count warm-start with continuous adaptation

## Frozen development card

Freeze before production score inspection, 2026-09-06. This follows the losing
[static context experiment](2026-09-05-historical-learning-preflight.md), without
changing or reclassifying its results. No tournament setting or competitor changes.

- Same canonical-only reader, January 4 prefix and three revealed suffixes:
  June 7–July 7, July 7–August 6, August 6–September 5 (UTC, half-open).
- Same 30-day exponential half-life, shrinkage 20 and 50k/250k-sat buckets.
  Compare pooled, outgoing and outgoing/amount predictors in two arms: retained
  historical counts plus online updates versus zero initial counts plus the
  identical online updates. No tuning from these results.
- Both arms share the prefix-known label alphabet and expand it only as
  settlements become available. This isolates historical **counts**, not all
  historical side information. A different alphabet per arm would let a cold
  model score everything as probability-one UNKNOWN, making losses incomparable.
- Score at received time before any outcome with an equal/later settlement
  timestamp can update the model. Late prefix settlements update both arms,
  once, after availability. Never train on outcomes unresolved at evaluation end.
  Use native archive created identities to reject duplicate rows. This does
  not qualify wallet-generation continuity or runtime native/legacy merging.
- Report paired per-event log losses and per-day differences, unknown labels,
  prefix/test/late/unresolved counts. Negative warm-minus-cold means lower loss.
  These periods are revealed development evidence, not sealed confirmation.
- Predict incoming adjacency conditional on an eventually settled event and
  its outgoing channel/amount. Selection bias remains. Do not infer absent
  demand, causal fee response, payer routes, inventory or rebalance profitability.
  Settlement time is an availability approximation, not receipt-time proof.
- Reuse the read-only, snapshot-consistent, raw-to-daily-coverage reader with
  its 50k row/10-second SQL limits. Supply code on SSH stdin; no installation,
  model/raw export, action RPC, schema write or production configuration change.

Implementation: `tools/historical_online_context_replay.py`. Lazy exponential
weights and cached totals avoid a whole-model decay pass for every event.
Sparse contexts shrink toward their simpler parent. Old history is allowed to
hurt; it is not discarded on observing a loss. This is a research-only learner,
not production historical bootstrap or fee authority.

## Results and verification

All three frozen runs completed read-only on production revision
`294e649783d0aadc1df40fe035d4acd39e1ca35e`. No code was installed and no raw
history or fitted model was exported. Reader SHA-256:
`23310d9fc6a95dc8d9dc2a1868b7ad6834406371c1f9c02a106fdd968d826fb2`.
Online evaluator SHA-256:
`c4fc7367d8c0fd18ab9a93ad779529d477d472f7e853cb8a8c6ce6da7434043b`.
The combined process took 0.497 seconds, with reported maximum RSS 38,368 KiB.
This is one measurement, not a runtime resource guarantee. Each source read
passed the existing closed-day raw-to-coverage checks; this does not prove
live wallet-generation continuity or repair the operational accounting gap.

Mean log loss in bits; lower is better:

| Suffix (UTC) | Bootstrap / scored | Cold pooled | Warm pooled | Cold outgoing | Warm outgoing | Cold outgoing + amount | Warm outgoing + amount |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| June 7–July 7 | 6,851 / 1,700 | 3.700013 | 4.073386 | 2.641693 | 2.871565 | 2.473384 | 2.726933 |
| July 7–August 6 | 8,551 / 1,918 | 4.116897 | 4.815085 | 2.953943 | 3.470914 | 2.821017 | 3.386285 |
| August 6–September 5 | 10,469 / 2,109 | 4.026451 | 4.532649 | 2.765981 | 3.278334 | 2.578349 | 3.163494 |

Warm-minus-cold loss was positive for all nine comparisons: pooled
0.373373/0.698188/0.506199; outgoing 0.229872/0.516971/0.512353;
outgoing-plus-amount 0.253549/0.565268/0.585145 bits per event.
Historical counts therefore hurt all three predictors in all three suffixes
under this frozen scheme. No winning period or context was selected afterward.

There were 30/30/29 days with scored events. Warm beat cold on only 3/2/4
days for pooled, 4/1/0 for outgoing, and 4/0/0 for outgoing-plus-amount. Days
without scored events are not assigned zero loss. Shared unknown incoming
labels occurred 7/16/9 times; these are first-unavailable-label counts under
online expansion, **not** comparable to static training-vocabulary unknown
counts. Late-prefix and unresolved-at-end counts were zero in these runs;
synthetic tests exercise both. Before the last prediction, cold had consumed
1,699/1,917/2,108 outcomes and warm 8,550/10,468/12,577, exactly the same online
updates plus the respective bootstrap counts. The last predicted event cannot
train its own prediction.

## Consequence for the improvement program

This rejects naive retention of historical counts at the chosen decay rate,
not the requirement to learn from history. Online context beat online pooling
in both arms and all periods, unlike the earlier static comparison. However,
the online alphabet and update schedule differ from the static experiment;
its absolute scores are not an isolated estimate of the benefit of adaptation.
The valid ablations are within this common-alphabet online card.

Next test a predeclared evidence-adaptive combination of historical and recent
experts, including a historical-influence cap and delayed-outcome-safe updates.
Any weight/gate must be learned from prior available predictive outcomes, not
chosen with this event's answer or retrospective selection of a winning window.
Compare against the cold online reference and simple fixed decay before adding
fee authority. Preserve these losses. Revealed development periods can guide a
new hypothesis but cannot become fresh confirmation.

Primary-source starting points for that next mechanism are Adamskiy et al.,
[A Closer Look at Adaptive Regret](https://jmlr.org/papers/v17/13-533.html)
(2016), whose abstract describes Fixed Share and interval-local comparison,
and Joulani et al.,
[Online Learning under Delayed Feedback](https://proceedings.mlr.press/v28/joulani13.html)
(2013), which studies how outcome delay changes online-learning guarantees.
This turn inspected their publication pages/abstracts, not full proofs. They
motivate expert recovery after regime changes and explicit delayed feedback;
they do not establish Lightning economic performance or a guarantee for our
selected/censored historical observations. Review full mechanisms before
claiming their theoretical properties for a successor implementation.

The broader path still needs source-aware durable model/cursor admission,
actual fee exposure/demand/value learning, and a demonstrated downstream action
benefit. Predicting a settled incoming mixture is not yet predicting marginal
inventory value, rebalance returns, optimal fees or competitor superiority.

## Verification and safety

Focused historical reader and online replay suite: **49 passed**. Initial test
invocations found no system `pytest` executable and then a test-helper import
error; using the existing project environment and correcting the test import
resolved collection. No evaluator or frozen parameter changed after scores.
Full isolated suite with `CLN_INTEGRATION=0`: **4,680 passed, one failed, five
skipped, two existing expected failures**, 191.85 seconds. The failure was
`test_disable_is_bounded_when_filesystem_lock_is_stuck`: elapsed 114.6 ms
exceeded its unchanged 100 ms bound. Its source and capture-manager module
match the parent commit. The exact test plus the full capture, both historical,
architecture and RPC-inventory suites then passed **148 tests** in 2.41 seconds.
The full-suite failure remains unresolved; the focused rerun does not erase
it or prove a scheduler-only cause. No timing threshold was relaxed.
Skips were four opt-in live-router tests and unavailable optional `pyln.testing`;
the two expected failures are pre-existing staged-removal cases.

A separate local synthetic 50,000-event limit check scored 24,224 suffix events
in 0.423 seconds, reporting maximum RSS 42,252 KiB. This checks the bounded
research computation with 64 incoming/32 outgoing labels, not worst-case
cardinality, production concurrency or an economic simulation.

Files changed: new offline online evaluator, its tests and this evidence note.
No runtime module or production schema/configuration changed. No Sling,
external coordinator, Archon DID, action RPC, competitor or test-environment
changes. Production compatibility is unchanged: this research tool is not
imported by the plugin. Follow-up risks include settlement-selection bias,
approximate outcome availability, source continuity, missing January 3
coverage, unqualified model persistence and no economic confirmation.
