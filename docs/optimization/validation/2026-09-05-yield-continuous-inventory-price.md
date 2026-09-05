# Continuous inventory reservation price (v36)

## Preregistered clean-room algorithm card

- Source/license: native CLBOSS `8cb4e9215eba58b049375f234f5f073d0c7fc622`,
  `Boss/Mod/FeeModderByBalance.cpp` and MIT project COPYING, verified locally.
  Its capacity-binned exponential balance multiplier motivates a continuous
  market-relative curve; no implementation is copied. See the
  [v35 diagnostic](2026-09-05-yield-reserve-fee-stability.md) for the observed
  1,158-versus-203-ppm quotes on similarly balanced relay channels.
  The frozen image and inspected research checkout both use
  [ksedgwic/clboss at that commit](https://github.com/ksedgwic/clboss/blob/8cb4e9215eba58b049375f234f5f073d0c7fc622/Boss/Mod/FeeModderByBalance.cpp).
- Baseline: restore the exact v35 candidate (including its admission reserve
  and fee-deadband separation), then change only Revenue Ops's compressed-cap
  reservation anchor. V35 passed retention but lost fees in both assignments;
  it remains rejected and is not promoted by reuse in a new candidate.
- Independent specification: with valid positive frontier `F <= B <= C`,
  where `B` is the broad local gossip anchor and `C` the configured ceiling,
  retain the existing market-close requirement `C <= 10*B`. At outbound
  inventory ratio `r`, interpolate geometrically between `(0,C)`, `(0.5,B)`
  and `(0.75,F)`. At or above 0.75 retain `F`. Thus ordinary balanced inventory
  prices at its broad market, scarce inventory rises toward the operator cap,
  and saturated inventory retains the existing cheap coverage anchor. Round
  to integer ppm and retain all downstream hard rails and fee-change gates.
- Enhancement over the motivating algorithm: endpoint calibration uses our
  observed market/frontier and operator ceiling rather than a fixed multiplier
  followed by clipping. It removes our flat ceiling throughout 0--75% inventory
  and preserves continuity at both joins, while incumbent modes are unchanged.
- Neutrality: missing, malformed, nonfinite, nonintegral fee evidence, invalid
  ratio, contradictory anchors or a non-market-close ceiling return the prior
  anchor unchanged. No RPC, clock, dependency, schema or new configuration.
- Hypothesis: retain viable prices on ordinary relay paths as they drain,
  without discarding scarce-lane value; recover useful paid routing enough
  to increase net revenue. This is not established by the final-state quotes.
- Comparator: unmodified native CLBOSS, same frozen v4 topology/public traffic,
  same runtime/controller timing and 1,200-ppm Revenue ceiling. Fresh unused
  crossed r234 (Revenue B) and r235 (Revenue A), 240 payments each. Source must
  be frozen before traffic and unchanged between assignments.
- Promotion: every unchanged revenue, retention, delivery, attribution, safety,
  coverage and bootstrap gate; any promising diagnostic still needs the
  production-incumbent comparison and fresh held-out validation.
- Rollback: preserve all results and exact source; reject on failed gates.
  Never change competitors, payer beliefs, environment or scoring to rescue it.

## Risks and production boundary

Lowering an anchor can surrender revenue without gaining traffic. Public
gossip quotes are imperfect substitutes, and geometric interpolation is a
policy hypothesis, not a learned demand curve. The inherited flow reserve
can still restrict native path allocation. The existing deadbands mitigate
fee churn but cannot guarantee stale-policy failures disappear.

This is lab-only. Production stays on the verified `aa79eba` deployment in
`undercut` mode; public v3.0.1 and the 1,200-ppm production ceiling are unchanged.
Only Revenue Ops source may change, and unrelated operator files remain
excluded from the four-source experimental image.

Post-traffic diagnostics additionally read Revenue Ops's fee-change history
through SQLite `mode=ro` and the LND payer's mission control through `querymc`.
These observations occur only after the scored traffic; they do not reset
beliefs, change routes, or become Revenue Ops decision inputs. Any unavailable
optional mission-control query is retained as a missing diagnostic, not hidden.

## Frozen candidate and pre-native verification

The full targeted suite passed 527 checks, including price-curve endpoints,
monotonicity, join continuity, neutral malformed evidence, admission/fee
composition, batched capture, architecture and read-only RPC surfaces. The
earlier focused run passed 374. No pricing threshold or acceptance gate was
changed in response to a native result; neither assignment has run yet.

- Four-source digest:
  `sha256:c91779e18b9c3b93850a010965e5a4f4c54e3787f8aa2e3e37c7c0d6e208ee5b`.
- Image `cl-revenue-ops-grand-prix:yield-aware-v36`:
  `sha256:f6ce03f6a6044837ce6dd7efd560e01959ce0d1fc4d77a8d662142c57218b222`.
- Unchanged base:
  `sha256:dd7e5fa57f07df6ae8c488ad570216c5e9a7fec1a10fad5b06eb2e02ed41deba`.
- Exact patch: `results/polar-grand-prix/yield-aware-v36-vs-ec3cebd.patch`,
  SHA-256 `c386b52539b12d4fbad2bd080ceba0d7d256aece2294e09c610120f1e78cd6db`.

Before launch, there were no v36/r234/r235 states or remaining tournament
containers. Only Revenue Ops sources are replaced in the candidate image;
competitors, protocol, topology and runtime flags are unchanged.

## Native outcome: rejected

Both assignments settled all 240 payments and completed scoped Docker cleanup.
The continuous curve did not recover sufficient fees or preserve every cell.

| Assignment | Revenue Ops fees (msat) | CLBOSS fees (msat) | Revenue volume (msat) | CLBOSS volume (msat) |
| --- | ---: | ---: | ---: | ---: |
| r234, Revenue B | 8,845,369 | 23,706,244 | 23,922,745,320 | 5,386,085,760 |
| r235, Revenue A | 8,541,757 | 35,587,085 | 23,019,957,320 | 6,288,741,760 |
| Paired diagnostic | 17,387,126 | 59,293,329 | 46,942,702,640 | 11,674,827,520 |

Delivery, per-payment attribution, frozen-protocol and safety gates passed.
Cell retention, positive economic bootstrap and required replica coverage did
not; every score reports `insufficient_evidence`. Minimum retention is zero:
the `shock_fault|lnd|medium` cell carries CLBOSS volume and no Revenue Ops
volume in both assignments. Coverage is incomplete by design for a diagnostic
pair, not waived. Economic and retention failures independently reject v36.

Revenue Ops recorded three local failures in r234 (all 4108), and five in r235
(four 4108, one 4103). These are diagnostic observations, not a causal
explanation for the full revenue gap. The lower price curve also earned much
less than v35 in its separate pair; that cross-run comparison is descriptive,
not an isolated estimate of the curve's treatment effect.

Read-only fee history confirms that DTS proposals can be replaced by the
yield market target, and some settled forwards pay a previous policy despite
a newer quote. The [integrated controller research plan](../plans/2026-09-05-fee-controller-research-loop.md)
records an exact example and the next audit. Do not infer that the current
quote represents the fee that generated an observation, or that another
inventory multiplier alone will correct feedback attribution.

## Rollback and handoff

After terminal completion, the archived v36 patch was verified byte-for-byte
against the seven candidate files before inversion. The reserve, continuous
curve, and added tests were removed, preserving the pre-existing uncommitted
v30 changes (59 fee-controller lines and 37 competition-test lines). Unrelated
operator database/xrebalance changes were untouched. The rollback suite passed
471 tests, covering admission, fee execution/composition, competition modes,
capture, architecture and RPC surfaces; log: `v36-rollback-tests.log`.

Individual and paired scorecards, state, native forwards, policy history and
post-run diagnostics remain under `results/polar-grand-prix/` with v36/r234/r235
names. Docker readback found no containers, networks or volumes for networks
1788644720 and 1788645200. The owned temporary launch script was removed;
immutable images and exact evidence remain intentionally reusable/auditable.
No production action RPC, deployment, ceiling change or release occurred.
