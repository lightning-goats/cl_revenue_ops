# Phase 0.5 Versioned Evaluation Identity

## Problem

The validation config and trend stream still used the unversioned historical
timestamp `2026-04-23T16:31:01Z`. That timestamp did not identify the closed
2026-07-13 through 2026-08-12 evaluation and could not safely identify the new
optimization measurement initiative. Consequently, old trend rows could be
mistaken for evidence belonging to a later checkpoint.

## Decision

The active validation config now declares this identity:

```yaml
evaluation:
  id: optimization-phase0-measurement-preflight-v1
  version: 1
  state: preflight
  formal_window_active: false
  t0: "2026-08-13T00:00:00Z"
```

This is a measurement-preflight identity, not the activation record for the
successor economic evaluation. `formal_window_active: false` is deliberate.
The ≥25-day window remains inactive until every precondition in
`production-validation-spec-v2.md` is met and committed to `baseline.md`.

## Evidence provenance

- Every new trend row stores evaluation ID, version, state, formal-window flag,
  and T0.
- Every node manifest stores the complete evaluation identity.
- Watch output carries the identity and rejects a manifest from a different
  explicit identity as required-evidence loss. Checkpoint report readers also
  reject missing or mismatched explicit watch identities as RED evidence loss.
- Checkpoint trend selection filters on both evaluation ID and version.
- Explicit evaluation checkpoint filenames include the ID and version, so an
  older T+14/T+28 document cannot suppress a new identity's report.
- Formal T+14/T+28 generation is disabled unless every configured node shares
  an explicit `active` identity with `formal_window_active: true`; preflight
  evidence cannot accidentally produce a formal checkpoint decision.
- Historical unversioned rows and manifests remain readable when a legacy
  config is explicitly used; they are not selected for the new identity.

## Validation rules

An explicit identity fails closed unless:

- `id` uses the collision-free filename-safe form `[A-Za-z0-9][A-Za-z0-9._-]*`;
- `version` is a positive integer;
- `state` is `preflight`, `active`, or `closed`;
- `formal_window_active` is boolean and is true only for `active`;
- `t0` is timezone-aware and has a literal UTC offset.

The collector validates identity before issuing any RPC. Invalid identity
therefore creates collection failure rather than mixing evidence into an
ambiguous stream.

## TDD and verification evidence

RED tests first showed that no identity parser existed, manifests/trends did
not preserve an identity, watch accepted a mismatched manifest, report
selection could see unversioned history, and report filenames were unscoped.
The focused identity/collector/watch/report suite then passed after the minimal
implementation.

The repository config test pins the preflight identity and proves the stale
April timestamp is no longer the current config T0. A later-dated historical
unversioned row is also excluded in favor of the matching versioned row. The
final focused validation, reconciliation, architecture, and RPC suite passed
`95` tests. The full functional suite passed `2,995` tests with `5` expected
skips and `2` expected staged-removal xfails; only the known shared-environment
dependency-pin checks were excluded.

## Safety and compatibility

- No production RPC was called and no runtime setting was changed.
- The identity is explicitly preflight and cannot claim formal activation.
- Existing collection commands remain read-only.
- No action RPC, Sling, Hive, mycelium, fleet, or external coordinator
  dependency was added.
- Old artifacts are preserved byte-for-byte; selection is identity-aware rather
  than a rewrite of history.

## Recommendation

**CONTINUE SHADOW.** Deploy the full Phase 0 measurement stack under this
preflight identity, then gather the separately required 72 consecutive hours
of reconstructable evidence. Create a new `active` identity and activation
record only after the full v2 preconditions are satisfied.
