# Historical learning: production coverage and route-context preflight

## Scope and production boundary

Read-only checks on 2026-09-05 confirmed production at
`aa79eba64eac474d56920a80cdb4782e25f7a522`, with the Revenue Ops plugin active.
No plugin restart, configuration change, action RPC, database migration or raw
history export occurred. Queries used read-only SQLite, low process priority
and 10–20-second process deadlines. Results below are aggregate observations,
not a guarantee of continued current runtime state.

## Retained history found

| Source | Retained rows | Earliest observed UTC | Latest observed UTC | Observed days |
| --- | ---: | --- | --- | ---: |
| Operational forwards | 937 | 2026-08-28 | 2026-09-05 | 9 |
| Outbound daily rollups | 1,172 | 2026-03-04 | 2026-08-26 | 175 |
| Inbound daily rollups | 1,117 | 2026-03-04 | 2026-08-26 | 175 |
| Fee changes | 32,426 | 2026-06-07 | 2026-09-05 | 91 |
| Rebalance history | 349 | 2026-06-09 | 2026-09-05 | 60 |
| Rebalance cost rows | 1 | 2026-03-19 | 2026-03-19 | 1 |
| Financial snapshots | 528 | 2026-03-18 | 2026-09-05 | 168 |
| Canonical forward archive | 194,518 | 2026-01-02 | 2026-09-05 | 245 |

Rows and observed days are not independent samples or proof of uninterrupted
exposure. Cost-table count alone does not describe total rebalance expenditure;
history, spend accounting and other canonical records must be reconciled.
The archive has 162,463 failed, 19,433 local-failed and 12,622 settled records.
All settled records have adjacent channel pairs and exact in/out/fee amounts;
the first settled event is January 7. A bounded CLN `listforwards` read confirmed
created index 1 dates to January 2. This is substantial retained raw history,
but does not substantiate nine complete months. Forty-six persisted fee states
contain valid JSON; syntax does not prove their rewards or calibration correct.

## Integrity findings: retained is not automatically qualified

Archive coverage reports 245 complete closed days through September 4 and an
incomplete September 5 (`day_not_closed`, created/updated sync incomplete).
Both source cursors were complete to their last observed index with no error.
However, the full January 2–September 5 half-open interval has 246 days: January
3 has no coverage row. Do not manufacture that day as zero demand.

The existing archive verifier failed the full range with `coverage_incomplete`
and `archive_operational_mismatch`, despite no malformed records and bounded
query plans. It found 12,578 settled archive events versus 10,237 operational
events. January and February have 2,119 archive settlements and no operational
rollups. Additional differences persist after operational collection starts.

For the overlapping March 4–September 5 interval, the verifier confirmed
`coverage_complete=true` and bounded plans, but still failed overlap:
10,437 archive versus 10,237 operational events, with 114,378,285 versus
113,130,016 fee msat. `legacy_projection_equal=false`; do not claim known
legacy deduplication fully explains this discrepancy. No historical totals
were rewritten to make these checks pass. This is not a quantified loss of
actual collected fees or proof of a current notification defect.

Historical research can separately test a single internally verified canonical
source. It must retain the operational mismatch as a limitation, never mix the
sources or treat that research as passing the failed overlap gate. Operational
learning requires the explicit architecture review in the integrated plan.

## Frozen exploratory route-context comparison

Before inspecting prediction scores, freeze this research card:

- Source: canonical settled archive only, generation 1. Start January 4 after
  the missing January 3 coverage; that original gap remains a recorded failure.
  Check each requested day's closed coverage and reconcile raw counts and
  integer-msat totals against it in one read transaction. Never use rollups to
  invent missing event pairs or historical inventory.
- Compare pooled incoming-channel frequencies, outgoing-conditioned frequencies,
  and outgoing-plus-amount-conditioned frequencies. Use hierarchical shrinkage
  20, 30-day evidence half-life and fixed 50k/250k-sat amount boundaries. These
  are hypotheses, not calibrated production settings or a novelty claim.
- Three chronological suffixes: June 7–July 7, July 7–August 6, August 6–September
  5. For each, train only on the January 4 prefix ending at that suffix's start.
  No fit uses its test suffix; all three results remain development evidence,
  not fresh tournament holdouts. No hyperparameter selection from these scores.
- Training events must settle before the split. Test events must arrive after
  the split and settle before the end. Withhold boundary-crossing outcomes;
  do not backdate settlement into the prior. Settlement time approximates when
  the node could know an outcome, not proof of historical notification receipt.
- Predict the incoming adjacent channel conditional on an already settled
  forward, its outgoing channel and amount. Measure per-event mean log loss in
  bits. Keep training vocabulary separate from test labels; reserve one unknown
  category. Report unknown-context rates and sample counts with every score.
- This does not predict demand that avoided the node, overall arrival intensity,
  failure probabilities, full routes, price elasticity or economic returns.
  It does not compare deployed controllers. Conditioning on settlement creates
  selection bias that must be addressed before applying it to decisions.
- Standalone, standard-library-only research script; no runtime import or API
  dependency, 50,000-row ceiling, ten-second query budget, read-only database
  opening and explicit failure on missing/malformed/inconsistent evidence.
  Only aggregate scores leave production; no serialized channel model export.

The downstream question is whether incoming-mixture context merits a bounded
shared estimator for inventory value and fee decisions. A good prediction score
alone cannot show that using it raises net fees. A losing richer model is also
useful evidence and must not be hidden by changing its buckets or evaluation.

## Results: richer context rejected in this static replay

All three runs completed on production without installing the script or
changing runtime state. Script SHA-256:
`ed76cd3a6bc012fda8e0072dc41dba0ea430ebb5de33b92122c9aa37494e9b34`.
The final implementation caches context totals to keep evaluation linear;
SHA-256 `23310d9fc6a95dc8d9dc2a1868b7ad6834406371c1f9c02a106fdd968d826fb2`
reproduced all three complete score outputs exactly. No parameters changed.
The script was supplied on SSH stdin, not written into the production plugin.
Every requested day's canonical raw-to-coverage checks passed. This does not
resolve the separate archive/operational mismatch above.

| Unseen suffix (UTC, half-open) | Training events | Test events | Pooled loss (bits) | Outgoing loss | Outgoing + amount loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| June 7–July 7 | 6,851 | 1,700 | 5.383651 | 5.385318 | 5.762961 |
| July 7–August 6 | 8,551 | 1,918 | 6.365317 | 6.913383 | 7.607972 |
| August 6–September 5 | 10,469 | 2,109 | 5.857263 | 6.459572 | 7.040573 |

Lower loss is better. Outgoing context did not beat pooling in any period;
amount conditioning worsened it further in all three. Retain this negative
result rather than promoting richer context because it sounds appropriate.
No boundary-crossing events were withheld in these particular runs; synthetic
tests verify that such events are withheld when present.

Incoming labels unseen in training accounted for 349/1,700, 605/1,918 and
457/2,109 test events (20.5%, 31.5%, 21.7%). Outgoing labels unseen in training
accounted for 281, 285 and 624 events respectively. Unseen does not necessarily
mean newly opened: it can also mean no settled training observations on that
channel. Scores group unknown incoming labels into one reserved category, not
an inferred identity. Weighted training counts were approximately 1,784.80,
2,260.75 and 2,452.46; these are summed decay weights, not independent sample
counts or an estimate of causal evidence strength.

Consequences for the improvement loop:

- Historical initialization is feasible from the canonical source at checked
  granularity, but a static context prior is not validated for promotion.
- Preserve simpler pooling as the predictive reference. Next hypotheses must
  address unknown-context probability, sparse evidence, contextual overconfidence
  and adaptation to changing mixtures. These results do not isolate which
  mechanism caused the loss; use explicit component tests.
- Test evidence-gated contextual use and online adaptation before translating
  mixture predictions into liquidity values or advertised fees. Preserve
  settlement-time availability and source identity in any historical-to-online
  transition. No model update may consume overlapping operational/archive data
  twice or bypass spending/fee authority.
- These revealed periods are development evidence. Any tuned successor needs
  separate temporal confirmation and unchanged native tournament qualification;
  predictive improvements alone are not an economic or competitor victory.

## Verification and remaining boundaries

The new tests cover read-only opening with URI-safe filenames, unchanged source
bytes, missing/invalid coverage, raw/coverage mismatch, malformed events, source
generation, missing databases, row budgets, current-day exclusion, strict time
bounds, late settlement, unseen-context backoff, duplicate identities, and
synthetic cases where context helps or hurts future predictions. No CLN client
or runtime controller is imported. No Sling dependency or action RPC is added.
The only new code is an offline research tool and its tests; production
compatibility and active model state are unchanged. This does not implement
historical bootstrap in the plugin or correct its current fee reward proxy.

The combined research, archive, archive-sync, verifier, architecture and RPC
inventory suite passed 216 tests in 1.25 seconds in the working tree.
The same 216 tests passed in 1.47 seconds from an isolated staged-source copy,
excluding all pre-existing experimental pricing and xrebalance changes.
