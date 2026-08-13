# Optimization Program Baseline

## Historical evaluation disposition

The original production evaluation is closed and immutable:

| Field | Value |
| --- | --- |
| Window | 2026-07-13 00:00:00 through 2026-08-12 23:59:59 UTC |
| Formal verdict | **YELLOW** |
| Counted days | 0 / 31 |
| Cause | Required hourly clean reconciliation evidence was not durable |
| Production SHA at final freeze | `5a45a91753556ce096291e03a9417519b92e8144` |
| Runtime version | `3.0.0` |

The verdict is not retroactively changed by later telemetry improvements. The
successor validation defined in `production-validation-spec-v2.md` is a new
evaluation with its own identity and boundary.

## Frozen historical anchors

These values remain the comparison anchors for interpreting the closed refactor
window. They are not automatically the baseline for a future optimization
activation.

| Metric | Frozen value |
| --- | ---: |
| Gross routing revenue | 22,703 sats / 30 days |
| Gross routing revenue/day | 756.77 sats |
| Net profit | 17,755 sats / 30 days |
| Net profit/day | 591.83 sats |
| Opex | 4,948 sats |
| Forward volume | 250,475,546 sats |
| Forward count | 2,372 |
| Fee changes | 353 / day |
| Total lightning value | 187,276,439 sats |
| On-chain reserve | 13,255,364 sats |

Historical classification anchors:

```text
profitable:   25
break-even:    2
underwater:    9
stagnant:      4
zombie:        0
bleeders:      0
```

## Closed-window observation

The closed window produced the following useful but non-countable observations:

| Metric | Value |
| --- | ---: |
| Gross routing revenue | 19,994 sats |
| Baseline-compatible net | 19,606 sats |
| Baseline-compatible net/day over 31 UTC days | 632.45 sats |
| Ratio to frozen baseline net/day | 106.9% |
| Rebalance costs | 0 sats |
| Automatic rebalance rows selected | 207 |
| Automatic rebalance attempts | 108 |
| Automatic rebalance successes | 0 |
| Local-budget skips | 99 |
| Temporary-channel failures | 102 |

The final report is authoritative for the accounting classifications, evidence
inventory, caveats, and channel-level analysis.

## Successor baseline state

No successor validation window is active. The configured
`optimization-phase0-measurement-preflight-v1` identity begins at
`2026-08-13 00:00:00 UTC`, has state `preflight`, and explicitly sets
`formal_window_active=false`. It identifies measurement-hardening evidence only;
it is not a successor-window activation record.

state: preflight
formal_window_active: false
72-hour durable-evidence gate: not started

The forward-archive preflight correction is documented in the [design](../plans/2026-08-13-forward-archive-preflight-corrections-design.md),
[implementation plan](../plans/2026-08-13-forward-archive-preflight-corrections.md),
and [measurement-hardening finding](../findings/phase0-measurement-hardening.md).

Before a successor window can start, an operator must commit an activation
record to this file containing:

- evaluation identity;
- exact inclusive UTC start and end rules;
- production Git SHA and runtime version;
- CLN version and node identity;
- complete public runtime configuration and override version;
- capital/channel snapshot;
- the exact preceding baseline interval;
- frozen per-day economic anchors derived from durable evidence;
- the configuration and algorithm regime identifier.

The activation record must be written before any successor-window observations
are evaluated. History must not be re-based after the window starts.

## Source evidence

- `docs/refactor/phase0/production-evaluation-final.md`
- `docs/refactor/phase0/production-evaluation-interim-2026-07-13.md`
- `docs/refactor/phase0/production-evaluation-spec.md`
