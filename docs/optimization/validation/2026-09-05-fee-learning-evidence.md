# Fee learning: exact evidence before controller changes

## Verified problem and scope

The controller currently estimates window revenue from outbound volume times
the current proportional fee. A real-SQLite regression reproduces the v36
example: 250,000 sats settled at 775 ppm earns 193,750 msat, whereas valuing
that volume at the newer 856-ppm quote produces 214,000 msat. This is an
accounting-input discrepancy, not a measured production earnings loss.

The operational `forwards` table is populated at settlement but stores received
time as its `timestamp`. A second regression inserts a settlement after an
observation cursor advanced beyond its received time. The existing
`get_volume_since` query returns no volume for that record. Another case shows
that integer-second timestamp boundaries have the same problem for a later
insert in the cursor's second. These tests demonstrate missing evidence; they
do not claim the entire posterior/action path is corrected yet.

## Implemented evidence primitive

`Database.get_fee_learning_events` reads at most 1,000 operational settled
records per page, ordered by their local ingestion ID. It uses one SQLite
statement to freeze a watermark and read a page, retaining exact integer-msat
fees/amounts, incoming/outgoing channels and received/resolved timestamps.
Subsequent pages can use the same watermark so new arrivals belong to a later
batch. SQLite's sequence keeps pruning and duplicate-insert gaps from resetting
the watermark. No cursor, schema, RPC, fee, or model is changed by a read.

Invalid/ahead cursors and database errors are explicit failures, not fabricated
zero rewards. Raw malformed evidence is preserved for consumer validation,
not silently coerced to a plausible fee. Repeated reads are repeatable while
retained rows remain unchanged; callers must atomically persist consumed IDs
and their learned state before claiming exactly-once learning. Pruning during
a batch or restoring a database needs explicit recovery; this reader does not
prove that all historical rows are present or recover deleted evidence.

The new API remains unused by runtime decisions. It is a bounded low-latency
evidence building block, not a second archive, new controller, or production
fix. Operational IDs and timestamps are not canonical CLN identities. The
existing `ForwardArchiveStore` already preserves CLN created/updated indexes,
nanosecond timestamps and failure outcomes. Its reconciliation role, current
15-minute synchronization interval and ADR-002's evidence-only boundary must
be respected when designing the shared learning path.

## Next integration requirements

Follow the [integrated learning plan](../plans/2026-09-05-fee-controller-research-loop.md).
Record actual fee/base-fee exposure and pre-action inventory/context; handle
accepted older fees, ambiguous mixed windows, missing evidence and late
settlement without assigning them to today's price. Existing `fee_changes`
legacy rows alone lack structured base-fee history and cannot reconstruct every
exposure. Link observations and model updates to stable action identities.

Evaluate route-pair/contextual learning with sparse-data pooling and calibrated
uncertainty, then shared demand/reliability/liquidity-value estimates across
fees, rebalances and budgets. Never treat a rebalance settlement as proof of
profit or give learned predictions authority over spend limits. Every action
must have a path to an evidence-backed future decision or an explicit reason
why no learning update was warranted.

## Verification and production boundary

The new API test failed before implementation because the method did not exist.
The combined real-SQLite, database, forward-hot-path, maintenance, deduplication,
architecture and RPC suite passes 125 tests. Cases include old-policy earnings,
base/subsat fees, late/same-second settlement, bounded paging, frozen watermark,
restart, duplicate ingestion, pruning, malformed input and database errors.
A query-only SQLite test verifies exactly one statement, no writes, and no RPCs.
The same 125 tests also passed from an isolated copy of the staged source,
excluding all pre-existing uncommitted database and pricing changes.

Only the narrow database method, its tests and learning-plan/evidence documents
belonged to that first change. Pre-existing database/xrebalance work and unqualified
v30 pricing remain separate. No new dependency or migration; no Sling or action
RPC; no production deployment. No native tournament is claimed for this
non-policy change. Economic qualification still requires the full unchanged
competitive-improvement program after learning is actually integrated.

## Follow-on: structured fee execution evidence

The existing `fee_changes` table now accepts a nullable `execution_evidence`
JSON field. The migration is additive and idempotent; legacy rows remain NULL.
The fee executor snapshots available prior fee/base fee and local inventory
before its already-authorized RPC. After success it records the requested
policy separately from uniquely matching CLN-reported fee/base-fee/HTLC values,
using the existing acknowledgement clock and fee-change row identity. Units
are explicit, malformed amounts stay unknown, and JSON is finite and bounded
to 16 KiB. No extra CLN request or decision-clock call is introduced.

Missing or ambiguous response channels never turn requested prices into
reported facts. Acknowledgement is not proof of network-wide gossip exposure;
every new record starts with attribution pending. Context is an available local
observation, not a claim of fresh authoritative liquidity. Dry runs, denied
authority and failed RPCs do not produce applied-action records. Evidence
construction failures preserve the legacy audit attempt; bookkeeping failures
remain warnings after the fee was successfully applied. No learned decision,
reward, observation cursor or authority changes in this step.

The new pure-helper, mocked-executor and real-SQLite tests cover malformed and
missing evidence, snapshot isolation, differing requested/reported prices,
failed/denied/dry-run actions, bookkeeping failures, legacy migration,
reinitialization and invalid/oversized payload rejection. All action calls in
tests use mocks. This is not a production deployment or an earnings correction.
Historical learning remains required: use retained events and aggregates only
at their supported granularity, validate chronologically, and reconcile
historical/online overlap before updating a model. See the integrated plan's
historical-bootstrap requirements; production coverage has not been inspected
as part of this change.

Compatibility: existing fee-history readers return an additive nullable JSON
string field; consumers must tolerate the additional key. Existing writer call
signatures remain valid. The field follows the existing 90-day fee-audit cleanup,
so long-term learning will need durable model checkpoints and an explicit
evidence-retention decision rather than assuming nine months of fee exposure.
Production upgrade/rollback and full-release qualification remain outstanding.

Verification: 349 targeted tests passed in the working tree (121.49 seconds)
and an isolated staged-source copy (112.29 seconds), excluding the pre-existing
xrebalance and experimental pricing changes. The suite covers the evidence
helper, fee execution, settled-event reader, database, fee authority, governed
fees, capture/replay, RPC inventory, architecture and maintenance. No live
action RPCs, Sling dependency, production changes or native benchmark claims.
