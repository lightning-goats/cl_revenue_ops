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

Establish a production-calibrated, highly competitive local Docker market in
which Revenue Ops can be measured and improved against reproducible competitor
controllers. The first admission pair is Revenue Ops versus CLBOSS. Torq and
LN Operator are admitted only after their exact versions and complete
controller configurations can be frozen and reproduced.

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

V12 must use paired price windows. An upward move is admissible only after a
minimum-volume baseline, and may continue only when the next window preserves
settled volume and improves fee revenue (or when explicit insufficient-balance
drops prove stocked-out demand). A losing probe rolls back and enters a
cooldown. Missing, malformed, sparse, or identity-asymmetric evidence remains
neutral. This combines the useful slow-ratchet/stock-out distinction found in
LN Operator with Revenue Ops' stronger market, inventory, profitability, and
global safety rails; no competitor implementation code is imported.

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

Reproducible Torq and LN Operator executable arms remain required before any
claim against those products; their clean-room strongest-plausible algorithm
cards are frozen in `competitor-research.v1.json` without importing their code.

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
