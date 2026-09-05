# Invalid fee-learning inputs must remain unknown

## Demonstrated defect and correction

The global Thompson update replaced invalid revenue (including NaN, infinity
and negative values) with zero, and invalid/nonpositive exposure with one hour.
Thus missing evidence could increment the zero-revenue streak, add downward
pseudo-observations, and change the next fee sample. The contextual update had
no equivalent input guard: invalid fees/rates could change means, precision,
counts and related contexts, or crash on malformed types.

The new regression suite failed 63 cases before the correction, with five
passing controls. It demonstrates the learning defect with synthetic inputs;
it does not establish its frequency or an earnings loss in production.

Both update entry points now reject absent, boolean, nonnumeric, nonfinite,
negative and unrepresentably large integer measurements before any mutation
or decision-clock capture. Exposure must be positive. Empty/malformed context
keys also leave the contextual model unchanged. Valid zero revenue with valid
exposure remains an observation: no demand is different from unknown demand.
No valid-input learning formula, fee rail, exploration policy, authority,
configuration option or persisted state format changes.

Tests compare complete model state, including related contexts, zero streaks,
positive references and observations. Repeated unknown inputs cannot create
false descent evidence; the next fee sample is unchanged under identical
clock and entropy. Round-trip persistence retains the neutral result.

Two older tests needed semantically explicit updates: negative duration is no
longer expected to create an invented hour, and secondary-context initialization
is tested with a genuine zero-revenue observation rather than an artificial
negative reward chosen to cancel its precision increment.

## Qualification and production boundary

The combined neutrality, controller regression, DTS/PI, fee-guard, convergence,
capture/replay, architecture and RPC inventory suite passed 372 tests in 5.82
seconds. All calls are local tests; no live action RPCs or Sling dependency.
Pre-existing v30 pricing and xrebalance work remains excluded.

This is a maintenance correctness candidate, not proof of better competitive
earnings. Exact settled-fee reward, mixed-policy attribution, late settlement,
historical bootstrap and the broader competitive program remain unfinished.
It does not repair already-poisoned persisted observations or silently reset
production models. Full clean-release tests and live preflight are required
before using the standing production-deployment authorization.

## Exact clean-release validation

Commit `294e649783d0aadc1df40fe035d4acd39e1ca35e` passed the complete suite from
an isolated checkout with Git history: 4,316 passed, five skipped and two
expected failures in 164.05 seconds. Four live-router tests were deliberately
disabled; the other skip lacked optional `pyln.testing`. Expected failures
remain the existing staged-removal tests. No test gate was relaxed.

Relative to production `aa79eba`, runtime changes are confined to
`modules/fee_controller.py`, `modules/database.py` and the new
`modules/fee_policy_evidence.py`. This includes the previously verified bounded
settled-event reader and observational fee-execution evidence. The database
adds one nullable `fee_changes.execution_evidence` column; legacy rows remain
unknown. No dependency, option, fee ceiling or model-state format changes.
Neither the historical research models nor the uncommitted v30 pricing and
xrebalance work is activated by this release.

The expected runtime SHA-256 hashes are:

```text
7726b379fc77b320eca3aeb81b2ec7c793e957cd240e1afe0d113f2604c6cc1f  modules/database.py
2520360c1a2ed8cf899153a2c6c53d5903b8d94dcb771507835587086b314097  modules/fee_controller.py
0bb7d11c20e7a2bd9a5cd0158f2e16cc6b6a997c272ed8c7a0880404bd00f05c  modules/fee_policy_evidence.py
```

## Production rollout

The operator-authorized rollout deployed the exact tested `294e649` source.
Preflight verified source cleanliness, healthy loops, no active rebalance jobs
and the existing operating controls using their actual RPC response schema.
Missing/null fields were never interpreted as safe values.

A consistent database backup passed `PRAGMA quick_check`; source and configuration
were preserved privately. Only Revenue Ops was stopped, fast-forwarded and
restarted. Runtime file hashes matched and the nullable audit column was present.
Operating controls were unchanged. Source-only recovery was prepared but not
needed; never overwrite newer settled accounting merely to roll back code.

Independent checks waited for all staggered startup loops, confirming ordinary
fee, rebalance and financial-snapshot heartbeats, completed startup work and no
stalled loops. A normal fee update produced structured requested/reported policy
evidence. This is an evidence-path smoke check, not resolved price attribution
or validation of model learning.

The only operator-issued production action RPCs were plugin stop/start. No manual
fee/rebalance cycle, channel operation or fund transfer was issued. No Sling or
external coordinator was introduced. Temporary test/deployment files were
removed. Node-specific operating details and recovery locations are retained
privately, not in this public report. The public v3.0.1 tag is unchanged and is
not this production revision.

Continue monitoring ordinary behavior and earnings; this rollout has no
measured economic uplift claim. The full competitive-improvement objective,
including native competitors and realistic held-out qualification, remains
active and unproven.
