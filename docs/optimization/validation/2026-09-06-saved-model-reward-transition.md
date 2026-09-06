# Saved-model compatibility at the settled-reward boundary

## Outcome

The actual saved production models passed a bounded, no-action DTS component
probe of the `d16f223` reward correction. At Unix time `1788720057`, all **46**
active channels had usable strategy/model rows. Across **32 common-entropy draws
per channel**, the **1,472 paired proposals** had **zero ppm differences** and
identical bounded samples before/after model serialization. One positive-rate
reference changed under the corrected reward; no zero-revenue streak changed.
Only two current windows earned positive revenue, and only one window's reward
arithmetic differed. No source model or production policy was changed.

This clears a specific saved-model compatibility check; it is **not** a complete
fee-controller replay, production rollout, or proof of increased net yield.
The production checkout remains `294e649`. Full fee-path integration, operational
release preflight and the native economic evidence are separate requirements.
The four native competitors and original tournament gates remain in scope.

## Exact code and preserved state

The definitions of `GaussianThompsonState`, `PIDState` and `ChannelFeeState`
are byte-identical between production `294e649`, settled-reward candidate
`d16f223`, and the audit's base `a34d202`. Their combined decorated-class source
digest is:

`ccce4f1568c0e712e490162e2a22f8e0b646b5767e0521aa615ebecb1cbf7cc8`

Production's complete controller source was independently pinned to:

`2520360c1a2ed8cf899153a2c6c53d5903b8d94dcb771507835587086b314097`

The production/candidate `utils.py` and `fee_cycle_capture.py` Git blobs also
match exactly. The on-node launcher checked their complete file SHA-256 values
before imports. The probe compiles the unchanged class AST in a private module
namespace, retaining the original future-annotations semantics. Only its
decision clock and Gaussian draw provider are substituted; the running plugin,
imported controller module globals and process-wide random state are untouched.
Common standard-normal draws are keyed by seed, semantic label and per-label
ordinal, so a branch's unrelated entropy calls cannot shift later draws.

The real nested-first fee-state extractor and real loaders consume each saved
row. All legacy row scalars are supplied, including sleep/broadcast/cursor
fields, rather than an abbreviated model-only row. Copies retain PID and
contextual state, observations, positive references and existing model
normalizations. No old reward is relabeled and no historical posterior is reset.

For each arm, serialization/deserialization must be idempotent. The incumbent
loader must read the candidate-updated model format, and paired samples must
match across restart while remaining within the tested **100–1,200 ppm** rails.
Effective production min/max config reads supplied those rails; the fee
interval remained 1,800 seconds. This is a model-format source-rollback check,
not a rehearsal of restoring a database with newer accounting arrivals.

## Scope of the stimulus

The read-only operational window is `(last_update, now]`, or the bounded
fee-interval bootstrap when the cursor is zero. No prior historical window is
substituted into today's state. Malformed, future-dated or unavailable matching
rows are unknown; failed queries/source pins/models refuse the report. Quote
maps must agree before and after the database view, although this cannot make
CLN and SQLite atomic or detect a price changing and changing back.

The two component stimuli are the incumbent's current-price proxy and exact
operational `fee_msat` rate, with the candidate's real-duration bootstrap
denominator. Both use the same current fee label, saved context and one common
0.98 discount step. The zero-proportional-fee learning guard is retained. Five
invalid reward forms must leave the whole copied learning state unchanged.

These are **undemand-adjusted model stimuli**, not alleged actual full-pipeline
observations. The probe does not reconstruct policy exposure or dynamic context,
apply Kalman demand normalization/profile selection, determine whether a window
would close, enter/leave sleep, execute PID inventory adjustment, or run the
governor and execution stages. Its proposals are DTS samples, not applied fees.
Most current windows are quiet. Zero changed samples here do not establish
equivalence under other traffic, or economic superiority.

## Refusal discovered during qualification

The initial source check used `ast.dump` as a class fingerprint. It agreed across
revisions on the laptop but refused production before processing any models:
the laptop used Python **3.12.13**, production **3.13.3**, and their AST dump
representations differed despite the verified full controller file hash.
That was a diagnostic fingerprint defect, not evidence of changed production
model code. Three attempted runs retained that refusal; the pin was not bypassed.

The correction uses AST only to locate class boundaries, then hashes exact
decorated source text, including comments/whitespace. Source/dependency pins
remain mandatory. A regression makes `ast.dump` unavailable and verifies the
portable pin; another changes a model constant and requires a different digest.
The corrected production run matched all pins and completed successfully.

## Verification and next release work

The focused model probe, earlier transition audit, settled-reward, DTS/PI,
architecture and RPC group passed **281 tests in 4.75 seconds**, including
32 new tests. They cover neutral/absent/malformed evidence, nested state,
serialization, incumbent readback, preserved PID/sleep scalars, common entropy,
quote drift, deadline/resource/source refusal, read-only database-file
preservation and aggregate-only output. All tests use fixtures or mocked RPC;
no live action RPC is enabled. The final isolated full suite passed **5,033
tests**, with five skips and two existing expected failures, in **176.74
seconds**. Four skips are opt-in live-router tests; one is unavailable optional
`pyln.testing`. No live test flag was enabled.

Tool SHA-256:
`1dc9301558a2feccca049bbeaecc27ae2a8844a568e312507da1e9c30f19f80c`.

Next, finish the narrow exact-candidate release assessment: combine the retained
native runtime evidence and clean-source regressions with full fee-path state
transition checks, then verified backup/source-only rollback and operational
preflight before deployment. Do not require every maintenance correction to
prove the full competitive objective, but do not present this component probe
as coverage of omitted fee-path behavior. Historical-model runtime admission
and the unfinished forecast-retirement work remain separate from this release.

Files changed: model-probe tool, its tests, this report and the fee research-plan
link. No runtime module, schema, dependency, fee setting or production file was
changed. The on-node probe ran from stdin in a separate existing interpreter
with bytecode writes disabled; saved models stayed in memory on the node and
only aggregates were exported. No Sling, Archon DID, external coordinator,
competitor policy, topology, traffic, payer, cadence or scorer change. The full
competitive goal remains unachieved.
