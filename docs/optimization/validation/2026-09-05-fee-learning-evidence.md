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
rows alone lack structured base-fee history and cannot reconstruct every
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
belong to this change. Pre-existing database/xrebalance work and unqualified
v30 pricing remain separate. No new dependency or migration; no Sling or action
RPC; no production deployment. No native tournament is claimed for this
non-policy change. Economic qualification still requires the full unchanged
competitive-improvement program after learning is actually integrated.
