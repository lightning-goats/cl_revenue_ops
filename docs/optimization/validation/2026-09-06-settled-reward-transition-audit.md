# Settled-reward production transition: read-only evidence

## Decision

Production remains on `294e649`; this work does not deploy the settled-reward
candidate or activate historical learning. It narrows the next maintenance
release check to the actual saved-model transition, rather than treating either
a green unit suite or mostly quiet current windows as sufficient evidence.
Maintenance qualification remains separate from proof of competitive superiority;
none of the native competitor, retention, replication or holdout gates change.

## Diagnostic contract

`tools/settled_reward_transition_audit.py` accepts only read-only
`listpeerchannels` calls. It reads the operational database through SQLite
`mode=ro`, `query_only`, a single read transaction and a time budget. It reads at
most 1,000 strategy rows and refuses model blobs over 1 MiB. The socket transport
has a two-second deadline and 4 MiB response budget. No controller is executed,
model rewritten, plugin initialized or CLN mutation requested. Only aggregates
leave the node; channel identities, individual forwards and saved models do not.

For each active channel with saved state, the tool compares the incumbent's
volume-times-current-quote reward with actual operational settled fees. It pins
exclusive received-time cursor semantics, integer-sat volume rounding, bootstrap
duration and the incumbent's separate sleeping-path denominator clamp. Malformed
or future-dated matching rows are unknown, not zero. Aggregate integer overflow,
missing schema, malformed models and resource exhaustion refuse the report.
Active quote maps must match before and after the database read. This is not an
atomic CLN/database view and cannot detect a quote changing and changing back.

It compares only the existing stability, volatility and sleeping-wake predicates,
not complete controller decisions. Inventory, time gates, demand normalization,
contextual sampling, acquisition, congestion, price rails and action execution
are not replayed. A changed predicate is diagnostic, not necessarily an action;
an unchanged one is not proof of a safe full-controller transition.

Saved-model inspection follows the runtime's nested-first `fee_state` contract,
with the legacy flat fallback. Missing model data is counted explicitly.
Conflicting nested/row cycle scalars become unknown windows: the audit does not
invent a single cursor when the two runtime loaders would see different state.

## Production observation

Both the startup option and read-only effective config query reported a
1,800-second fee interval. No setting was changed. In the corrected observation
at Unix time `1788718966`:

- 46 active channels had 46 matching strategy rows; no missing model or unknown
  observation window was encountered, and none needed bootstrap fallback.
- 44 windows had no volume; two contained four forwards earning **15,887 msat**.
  These are different per-channel current windows, not a daily node earnings
  report or an estimate of foregone revenue.
- One window's reward arithmetic differed. The maximum absolute rate difference
  divided by `max(1 sat/hour, incumbent rate)` was `0.00028465970443106793`.
  No stability, volatility or sleeping-wake predicate changed in this snapshot.
- No current-window proportional-fee shortfall was observed. This does not
  establish policy exposure or clear the separate attribution problem.
- The saved models contained **8,845 observations**, **107 positive-revenue
  observations**, **579 contextual entries**, and **40 positive-rate references**.
  All 46 models lacked a reward-source marker. Counts describe persisted
  content, not a claim that every old observation is valid or causally labeled.
  The observation list can include the incumbent's synthetic zero-probe tuples;
  its length must not be presented as a count of independent paid experiments.

An initial diagnostic mistakenly inspected only the legacy flat model field and
reported zero stored observations. That model-count result is invalid. An
independent nested-first on-node read returned the counts above. Four added
regressions failed before correcting nested precedence, missing-model reporting,
cycle disagreement and malformed nested models; all pass after correction.
The corrected full diagnostic reproduced the independent counts. Earlier and
later current-window totals differ because production continued operating.

## Release consequence and remaining work

The current-window arithmetic difference was small, but the sample contains only
four forwards. Production also has substantial existing learned state. There is
no justification here to reset it, rescale every old reward, enable yield-aware
mode, or deploy all of development `main`. The old tuples do not provide the
complete price/exposure and observation-window provenance needed to reconstruct
every historical reward exactly.

Next, exercise exact incumbent/candidate controller code on isolated copies of
the saved state with matched clocks/entropy and captured read-only inputs,
including positive traffic, sleep/wake, malformed/unavailable evidence and
restart. Keep private data on the node. Assess model compatibility, bounded
policy changes and source-only rollback without erasing later accounting.
Re-run the native economic comparison for changed decision behavior; do not
inherit an improvement claim from this audit or from the earlier short losing
diagnostics. The historical learner's separate runtime-admission work remains
open, including the preserved unfinished forecast-retirement implementation.

## Verification and compatibility

The final focused audit, settled-reward, architecture and RPC inventory group
passed **126 tests in 1.35 seconds**. Tests use local SQLite fixtures and mocked
RPC only, including exact database-file preservation, read-only method rejection,
transport framing/budgets, malformed/missing evidence, current quote drift,
source-format precedence and agreement with actual controller rate/threshold
helpers. The initial flat-only version's full-suite result is superseded. The
final isolated full suite passed **5,001 tests**, with five skips and two existing
expected failures, in **184.24 seconds**. Four skips are opt-in live-router tests;
the fifth is unavailable optional `pyln.testing`. No live tests were enabled.

Tool SHA-256: `28c246b196a11eb389fea68f1940f163fa39271174ecafeaf05e2500e01b263b`.

Files changed: diagnostic tool, regression tests, this report and research-plan
link. No runtime source, schema, dependency, fee rail, competitor, topology,
traffic, payer, timing, scorer or production setting changed. No Sling, external
coordinator or Archon DID was added. No production action RPC was triggered;
production checks were read-only and installed no files. The full competitive
goal remains unachieved.
