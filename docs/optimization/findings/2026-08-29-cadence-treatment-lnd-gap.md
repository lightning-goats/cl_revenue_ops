# Fast-cadence treatment: LND acquisition remains the blocker

> Resolved by the separated paid-retention evidence budget. See
> `2026-08-30-paid-retention-budget-win.md` for the frozen formal win.

## Frozen treatment

Revenue Ops revision `666048e173fe66c7addcb55868f35d5f585d43e6` ran from
image `sha256:c55fc79d3802297cd932cdffd15f8ae9c11854440583aebfccebdba34d80784e`
against uncapped CLBOSS on fresh crossed replicas 122, 123, and 124. The
treatment shortened the effective observation window only when operators
explicitly configure a fast fee cadence; the production-default cadence keeps
the existing 15-minute active-profile window.

Each replica contributed six fee-only and six full-stack realistic blocks,
including one independently cold block and five warm repeats per league. All
288 scored payments settled, no fallback route was used, and all coverage,
budget, attribution, reliability, and safety gates passed.

## Result

| League | Revenue Ops normalized net | CLBOSS normalized net | Revenue margin | Paired 95% CI | Verdict |
|---|---:|---:|---:|---:|---|
| Fee-only | 172590.211870 | 291239.132678 | -40.739% | [-284702.085751, 105360.560473] | Inconclusive |
| Full-stack | 176563.712897 | 367987.982671 | -52.019% | [-834069.402934, 292972.468213] | Inconclusive |

The treatment moved the formal overall verdict from `clboss_wins` to
`inconclusive`, but it is not promotable. Revenue Ops was effectively tied on
CLN in fee-only (11,390 versus 11,378 msat) and led CLN in full-stack (23,745
versus 12,113 msat). It again captured zero LND fees while CLBOSS captured
7,842 msat in fee-only and 12,182 msat in full-stack.

Revenue Ops spent 12,088 msat on native rebalancing in full-stack. The scorer
attributes only opponent-forward excess bounded by the native delivered and
sent amounts; an external return path is also valid. The result therefore does
not hide those costs or misclassify the resulting forward volume as ordinary
CLBOSS demand.

## Next diagnosis

The remaining formal regression is selective LND route acquisition rather than
global fee responsiveness. The built-in bounded acquisition experiment must be
traced before another fee rail is changed. In particular, exact-path readiness
forwards are real forwards and may make every fresh lane fail the experiment's
24-hour no-forward eligibility test before scored demand begins. The next
round must distinguish that fixture interaction from missing or unsuitable
small-graph competitor-floor evidence, then change only the failing mechanism.

Promotion still requires a greater-than-10% normalized net advantage with a
positive paired interval, no client-family fee regression beyond 5%, and no
reliability, budget, attribution, or safety regression.
