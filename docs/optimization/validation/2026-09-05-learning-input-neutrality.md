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
