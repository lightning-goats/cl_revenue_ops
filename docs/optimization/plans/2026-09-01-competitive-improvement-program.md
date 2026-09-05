# Competitive-improvement program

**Status:** active local-only improvement program. The operator approved
regtest mutations and fake-sat action RPCs for this program. The tested v9
source was deployed to production at `994ffb3` on 2026-09-03, but its new mode
remains default-off: production still uses `undercut` with the existing
1,200-ppm maximum. No live-fund action RPC was used for the deployment.

The machine-readable protocol is frozen with holdout commitment
`sha256:28f431477a4fcbb431a6b4e23485cab1bbfd8536e191def91f1c5c4639b1f4cf`.
Its private seed and salt were stored mode 0600 outside tracked source; the
commitment was verified before the sealed v9 holdout was scored.

## Objective

### Native competitor behavior (operator correction, 2026-09-05)

Only Revenue Ops may be adjusted for competitiveness. Competitors must retain
their native decision algorithms, price outputs, learning, and update cadence.
Do not clip competitor ppm/base fees, freeze their fee management, or alter
their configuration to improve Revenue Ops's ranking. High native quotes are
evidence to explain, not outputs the harness may correct.

The `clboss_bounded` experiment violated this requirement: it froze CLBOSS fee
management after warm-up and clipped its selected policies while Revenue Ops
continued adapting. Replicas 216--221 are excluded from competitive evidence;
replica 222 was stopped and cleaned up following the operator correction. The
bounded runner mode and temporary launch script were removed. Archived raw
artifacts are retained for audit only; the scorer rejects this competitor ID.
Previously generated bounded score JSON files are superseded by this exclusion.
No bounded result supports deployment, promotion, or product-superiority claims.

Calibration changes must be justified by production observations and frozen
before candidate comparison. Do not tune the topology or traffic to suppress
a competitor's strengths or to rescue a failing cell. Realistic constraints
apply to the environment and Revenue Ops experiments; native competitor
responses remain observable outcomes.

Source-derived models remain research aids. Their results cannot replace
evidence against the actual product or establish superiority over that product.
Fee-only tests with rebalance/open/close execution disabled establish only
fee-policy behavior; full-product claims require the corresponding native
functions under matched capital and workload conditions.

The uncommitted v30 Revenue Ops candidate is not production-qualified. It still
requires comparison with native competitors and the production incumbent under
realistic conditions, followed by fresh held-out validation. The retained scorer
hardening prevents pooling runs with different Revenue Ops fee ceilings,
dynamic HTLC settings, or update intervals.

Establish a production-calibrated, highly competitive local Docker market in
which Revenue Ops can be measured and improved against reproducible competitor
controllers. The first admission pair is Revenue Ops versus CLBOSS. Later arms
must pin an exact product revision and complete controller configuration. When
the product runtime cannot be made symmetric with the CLN contender topology,
only a source-derived clean-room model may be admitted, and every result must
remain explicitly scoped as algorithm- or workflow-equivalent.

The primary objective is settled capital-normalized incremental net profit. A
candidate must also retain at least 95% of the best controller's useful settled
volume in every predeclared client/capacity cell, unless a separately frozen
exception defines a stronger economic margin. Any safety, accounting,
attribution, or reconciliation failure rejects its block.

## Calibration and topology

Production supplies only an aggregate private calibration manifest: channel
count, capacity and local-balance distributions, peer-degree/age buckets,
policy and forwarding distributions, concentration, route lengths, failure
codes, and rebalance costs. Node IDs, peers, SCIDs, exact balances, invoices,
payment hashes, and raw forward records are forbidden.

The local graph has 24--32 CLN/LND nodes. Each contender receives a crossed,
equivalent production-shaped portfolio. Six or more lateral competitors and
four specialist corridors ensure that important corridors have at least two
alternative paths; no controller owns a captive route. Eclair remains excluded
until a pinned, reproducible Docker arm is qualified.

Traffic is natural-pathfinding, not forced-path, and combines retail,
merchant-directional, exchange-burst, competitive-displacement, and
shock/fault classes. Amounts are heavy-tailed; arrivals contain diurnal and
burst components. Public development seeds are deterministic and recorded;
the holdout seed is sealed by commitment before a contender revision freezes.

## Improvement rule

Competitor research creates a clean-room algorithm card: source/license,
observable behavior, independent specification, omitted variable, Revenue Ops
enhancement hypothesis, retained safety invariants, comparator arm, promotion
measure, and rollback. Never copy competitor implementation code or add its
dependency merely to reproduce behavior.

Every card is evaluated in three arms: Revenue Ops incumbent,
competitor-equivalent baseline, and Revenue Ops enhanced. Promotion requires
that the enhanced arm beat *both* alternatives in public-seed ablation and
sealed-holdout crossed replicas, with the existing exact attribution,
settled-cost, reservation, and read-only-surface gates. Until then it stays
default-off.

## Controller-start fairness

Controller startup order must not allocate the first routes. CLBOSS publishes
its initial market-derived fee surface on an approximately one-minute cadence,
while Revenue Ops can cycle every 15 seconds. After both controllers pass their
safety readbacks, the harness therefore enforces a 75-second no-traffic warm-up
and records anonymous policy snapshots for both identities. A block is rejected
if either contender has fewer than all 16 channels active after warm-up.

## Verified tournament results

Yield-aware v9 uses a capacity-weighted live market anchor, bounded
capacity/inventory pricing, and route-evidence selective displacement. It is
experimental and default-off. On topology v4 and public seed 20260901, six
fresh replicas (r37--r42, three per crossed assignment) settled 1,443/1,443
payments with every protocol, attribution, delivery, safety, cell-retention,
and nested-bootstrap gate passing. Revenue Ops earned 315,254,700 msat versus
CLBOSS's 150,053,101 msat (2.10x) and carried 150.813B versus 25.041B msat
(6.02x). The frozen scorecard verdict is `revenue_ops_wins`.

The sealed holdout was revealed only after that image froze. Six additional
fresh crossed replicas (r43--r48) again passed every gate: Revenue Ops earned
266,693,073 msat versus 40,980,970 msat (6.51x) and carried 144.351B versus
11.775B msat (12.26x). Its frozen verdict is also `revenue_ops_wins`.

### Production-cap price discovery

The live production configuration remains unchanged at a 1,200-ppm maximum.
A matched r49 replay showed that this ceiling made a fee win mathematically
impossible despite a 6.04x volume advantage: Revenue Ops earned 4,274,982
msat versus 30,799,832 msat. A conservative repeat-paid-demand probe improved
r51 revenue to 15,025,471 msat but still could not overcome CLBOSS's
55,383,318 msat under the same ceiling.

A Docker-only mainline-v9 cap sweep then tested whether a modest production
increase could close that gap safely. At 3,000 ppm, r64 carried 40.136B msat
versus CLBOSS's 9.675B but earned only 49,956,041 versus 65,286,275 msat; only
233/240 payments settled, so both the revenue objective and delivery gate
failed. At 5,000 ppm, two fresh Revenue-B replicas independently left native
CLN multipart payments unresolved beyond xpay's five-second retry horizon and
the harness's 120-second RPC bound (r60 payment 171 and r62 payment 143). The
r60 timeout advance eventually forced two payer channels on-chain. A 10,000
ppm repeat, r66, produced the same unresolved-payment failure at payment 206.

These results are non-monotonic with the successful 50,000-ppm v9 tournament
because fee surfaces change native route and multipart composition; the high
ceiling result cannot be interpolated into a safe production setting. No cap
increase or yield-aware activation is approved from this sweep. Production
remains in the healthy default `undercut` mode with its 1,200-ppm maximum.
The runner now marks an unresolved replica `public_traffic_invalid` as soon as
timeout handling puts a payer channel on-chain, preventing a damaged graph
from being resumed or scored.

V11 tested a faster repeat-paid-demand probe. At a 5,000-ppm rail in r53,
Revenue Ops retained 1.86x the volume and 3.07x the route count but earned only
80% of CLBOSS revenue. At a 10,000-ppm rail in crossed r54, it won all three
metrics: 62,087,765 versus 51,207,394 msat revenue (1.21x), 15.685B versus
13.624B msat volume (1.15x), and 166 versus 74 routes (2.24x).

The opposite assignment r55 decisively rejected that apparent improvement.
Revenue Ops earned only 21,341,807 msat versus CLBOSS's 96,086,452 msat
(0.22x) while carrying 13.336B versus 16.608B msat (0.80x), despite winning
route count 158 to 91. All 240 payments still settled and the attribution,
delivery, and safety checks passed. The paired total was only 0.57x competitor
revenue. The unqualified v11 probe was therefore removed from the deployable
tree. Three forwards are evidence that a quote clears, but not evidence that
doubling it improves marginal revenue.

V12 tested paired price windows. An upward move was admissible only after a
minimum-volume baseline and could continue only when the adjacent window
preserved settled volume and improved fee revenue. A losing probe rolled back
and entered cooldown. Missing, malformed, sparse, or identity-asymmetric
evidence remained neutral. This combined the useful slow-ratchet/stock-out
distinction found in LN Operator with Revenue Ops' market, inventory,
profitability, and global safety rails; no competitor implementation code was
imported.

The experiment was rejected. Initial Docker replica r58 (Revenue Ops on
identity B) earned 83,697,637 msat versus CLBOSS's 42,958,964 msat and carried
38.759B versus 11.046B msat, but only 230/240 payments settled, failing the
delivery gate. Its logs also exposed an accepted-price oscillation. The
corrected v12.1 implementation chained an accepted step directly from its
adjacent observation window and held a verified market-ceiling price for
revalidation. Focused tests passed, and the live trace demonstrated the
intended accept, chain, reject, and rollback lifecycle.

Corrected crossed replica r59 (Revenue Ops on identity A) nevertheless lost
decisively: Revenue Ops earned 83,296,865 msat versus CLBOSS's 136,418,678
msat, despite carrying 28.212B versus 24.449B msat and settling all 240
payments. The cell-retention gate also failed. The result shows that locally
validated marginal price increases can still cede more valuable network-wide
traffic to a competitor than their sampled channel revenue captures. V12 and
v12.1 were removed from the deployable tree; neither was deployed or enabled
in production.

### Docker-only execution

Polar was an orchestration convenience, not a protocol requirement. It was
retired after Docker-only crossed replicas r56 and r57 reproduced the v9 result
with the same pinned bitcoind, CLN, LND, contender image, topology manifest,
natural-path traffic, and scorecard. Revenue Ops earned 99,908,922 msat versus
CLBOSS's 52,687,632 msat (1.90x), carried 51.477B versus 7.141B msat (7.21x),
and handled 430 versus 50 forwards across the pair. All 480 payments settled;
the safety, delivery, per-payment attribution, and cell-retention gates passed
in both identity assignments. Docker retained 92.8% of the comparable Polar
pair's Revenue Ops fees and 105.0% of its Revenue Ops volume.

Docker is now the only maintained orchestration backend. Completed-run
containers, networks, and volumes are removed by exact scoped labels/names;
the r56 and r57 cleanup audits found zero residual resources. Scored JSON
evidence is retained. Reuse is permitted only when an immutable snapshot/reset
proves fresh channel balances, gossip, clock, wallet, and controller state.

The historical `polar-grand-prix-*-v1` JSON schema identifiers are deliberately
retained so pre-migration replicas remain scoreable; they do not indicate a
runtime Polar dependency. Fresh run artifacts live under `results/grand-prix/`.

Build the immutable equal-runtime base once, then use the checkpointed runner
for each crossed replica:

```bash
docker build -f tools/grand-prix/base.Dockerfile \
  --build-arg REVENUE_COMMIT=2987608d525075ec974ace6e17ee93986c1d0ba5 \
  -t cl-revenue-ops-grand-prix-base:2987608 .

python3 tools/grand_prix_manifest.py build-topology \
  tests/fixtures/competitive_improvement/calibration.v1.json \
  --public-seed 20260901 \
  --output results/grand-prix/topology-public.json

python3 tools/grand_prix_runner.py plan \
  --topology results/grand-prix/topology-public.json
```

Every mutating phase requires `--apply`. Run `stop-lab --apply` after scoring;
it removes only resources bearing the Grand Prix label and preserves the JSON
state and score. A stopped Docker state is scoreable only when its last durable
event proves that exact cleanup completed.

### Clean-room LN Operator and Torq expansion arms

The non-runtime expansion layer is now executable through
`tools/equivalent_competitor_controller.py` and the immutable
`tools/grand-prix/equivalent-controllers.v1.json` configuration. It imports no
competitor implementation code and has no dispatch surface of its own. The
Docker runner applies its policy intents only behind the existing explicit
`--apply` gate.

- `ln_operator` reproduces the documented default 25--750 ppm, `k=8`
  inventory sigmoid with a two-hour cadence, zero initial market multiplier,
  no refill floor, and rebalancing disabled for the comparable fee-only
  league. It is an `algorithm_equivalent` fee-policy response model, not an
  LN Operator runtime or rebalance comparison.
- `torq` freezes a deliberately aggressive operator workflow: a 25--2000 ppm,
  `k=10` inventory sigmoid, 50,000-sat balance-change triggers, and 25 ppm / 10%
  deadband. Torq publishes workflow mechanics rather than a canonical fee
  algorithm, so this is a `workflow_equivalent` strongest-plausible Torq-style
  configuration, not a Torq product runtime or canonical Torq strategy.

Admission smoke replicas r67 and r68 used the v9 contender image and the public
Docker topology. Both settled 24/24 payments. Revenue Ops captured 24/24
contender forwards in each: 954,200 versus 0 msat against the LN Operator model
and 1,650,386 versus 0 msat against the Torq workflow. The Torq arm completed
its initial policy application without a CLN failure; no later refresh was due
because its managed lane carried no smoke traffic. These results prove
executable integration and attribution only; they are not formal
superiority evidence. Formal claims still require three replicas per crossed
assignment, nested-bootstrap and cell-retention gates, followed by the sealed
holdout. Direct-product claims remain prohibited until exact product runtimes
and immutable operator configurations are admitted.

The LN Operator algorithm-equivalent public block (r69--r74) subsequently
passed every protocol, attribution, delivery, safety, bootstrap, and
cell-retention gate. Revenue Ops earned 94,533,286 msat versus 16,050,376 msat
(5.89x) and carried 149.726B versus 26.122B msat (5.73x). This is strong public
algorithm-equivalent evidence, but no sealed holdout was run and it remains
explicitly narrower than a claim against the LN Operator product runtime.

The Torq workflow-equivalent arm exposed a stricter per-cell retention problem:
v16 won aggregate economics but retained only 68.3% of the competitor's best
cell volume. V17 lowered the already admission-protected scarce-inventory
market-relief boundary from 10% to 5%; the existing 85% per-HTLC executable
liquidity limit and balance-change wake remain the safety rails. The formal
public block (r96, r103, r119--r122) passed every gate and earned 138,038,262
msat versus 57,545,814 msat (2.40x), while carrying 139.142B versus 36.710B
msat (3.79x). Minimum crossed-cell retention was 100%.

Only after freezing that candidate, the sealed Torq holdout block (r123--r125,
r127, r128, r130) also passed every gate and was promotion-eligible. Revenue
Ops earned 123,093,237 msat versus 26,169,295 msat (4.70x), carried 139.132B
versus 16.990B msat (8.19x), and recorded 1,278 versus 170 forwards. Minimum
cell retention was again 100%, and the nested bootstrap result was p=1.0 with
a 95% revenue-difference interval of [11,096,581, 18,434,620] msat. Replica
r126 was excluded before traffic after its competitor-side contender stopped
during readiness, so it contributes no outcomes to the frozen block.

V17 is therefore promoted over the strongest-plausible clean-room Torq-style
workflow comparator for the local Docker protocol. It does not establish that
Torq itself ships or would select that policy. The production code change is
safe to deploy dormant, but activating yield-aware pricing in production is
still prohibited: all qualifying blocks used the predeclared 50,000-ppm
tournament ceiling, while the separate 1,200--10,000-ppm production-cap sweep
did not establish a safe profitable activation point. Production remains
`undercut` with `max_fee_ppm=1200`.

### LNDg admission and next loop

LNDg is next in the tournament queue at the operator's request. Admission is
pinned to the signed `v1.11.0` tag,
`0fe400029240fc59431b56b6ce47e24b764396b1`; the similarly named branch points
at a different later commit and must not be substituted. Its source is MIT
licensed. LNDg is LND-native while Revenue Ops is CLN-native, so an actual
runtime match would introduce a node-implementation confounder in the current
crossed topology. Series 1 will therefore begin with an algorithm-equivalent
fee-only arm derived from `af.py`, not a claim against the whole LNDg product.

The frozen model must reproduce LNDg's peer-aggregated seven-day flow and
assisted-revenue signals, 24-hour per-channel cooldown, 15%/95% liquidity
zones, failed-liquidity HTLC threshold, 5-ppm rounding, and 0--2,500 ppm
default rails. Rebalancing stays off in the fee-only league. A later separate
rebalance league may model LNDg's profitable-outflow refill selection,
inbound target, source threshold, cooldown, and the lower of its absolute
max-fee rail and 65% target-channel revenue-rate budget.

The execution sequence is the same as the prior arms: independent algorithm
card and source fixtures, malformed/absent-data neutrality tests, crossed
public ablation, Revenue improvement only when evidence identifies a gap,
three replicas per assignment, then a previously committed sealed holdout.
Promotion and production activation remain separate decisions.

Admission smoke replica r131 executed the clean-room LNDg model on the public
Docker topology with Revenue Ops v17. The model changed 12 of 16 eligible
channels, with resulting targets from 25 to 285 ppm; this verifies that the
pinned response is executable rather than a paper specification. All 24 smoke
payments settled. Revenue Ops carried 2.424M sats and earned 567.444 sats while
the LNDg lane carried no contender traffic. This is deliberately not formal
superiority evidence: the fresh lab supplied no seven-day forwarding or
eligible-liquidity-failure history, and a short smoke must not silently
compress LNDg's source 24-hour cooldown. A formal history-aware arm must freeze
its simulated clock and evidence window before the crossed public block.

Example controller selection:

```bash
python3 tools/grand_prix_runner.py start-controllers \
  --topology results/grand-prix/topology-public.json \
  --state results/grand-prix/runner-state-r1.json \
  --competitor-controller ln_operator \
  --revenue-market-mode yield_aware \
  --revenue-max-fee-ppm 50000 --apply
```

## Initial research queue

1. Multi-source/destination allocation: improve minimum-cost delivery with
   conservative post-refill conversion and source opportunity cost.
2. Partial delivery/replanning: improve bounded fragmentation with per-part
   durable reservations and marginal-EV stopping.
3. Failure learning: improve route constraints with evidence confidence,
   expiry, and restart persistence rather than assuming one failure is truth.
4. Fee response: improve hysteresis with route-specific clearing price,
   inventory horizon, and profitable selective displacement.

Validate the machine-readable contract locally:

```bash
python3 tools/competitive_improvement_protocol.py \
  tests/fixtures/competitive_improvement/grand-prix.v1.json
```

This contract complements the Grand Prix scorer and does not replace its frozen
legacy-compatible evidence schema. The xrebalance Series 1 protocol remains
separately frozen and read-only.
