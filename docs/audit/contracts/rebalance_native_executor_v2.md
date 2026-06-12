# Intent Contract: modules/rebalance_native_executor_v2.py

Tier 2 — medium treatment. Audited 2026-06-12 against commit 9f8f219.

## Purpose

`NativeRouteExecutor` (modules/rebalance_native_executor_v2.py:16) executes one
already-priced circular rebalance route via raw Core Lightning RPCs: it validates the
route hop-by-hop, creates a self-invoice, calls `sendpay`/`waitsendpay` on that exact
route, and translates outcomes into an `ExecutionResult`. It does no route discovery or
pricing — router v3 supplies the sendpay-ready route; this module's job is faithful
execution, honest fee accounting (planned vs. actual), failure attribution (which
channel/direction broke), and refusing to treat unresolved payments as failures.

## Inputs / Outputs

- **Constructed by**: `RebalanceEngine._make_executor` (modules/rebalance_engine_v2.py:2260–2268),
  with the engine's cached node id injected to skip per-cycle `getinfo`.
  `modules/rebalance_executor_v2.py:11` re-exports it as `RebalanceExecutor` (legacy alias).
- **Inputs**: `execute(route, amount_sats, source_channel_id, dest_channel_id,
  max_fee_sats, observation_store, observation_context)` (:377); plugin RPC proxy
  (`ThreadSafeRpcProxy` with `timeout=` kwarg fallback, :36–48); optional
  `SegmentObservationStore` for failure observations.
- **CLN RPCs called**: `getinfo` (:61), `invoice` (:413), `sendpay` (:441),
  `waitsendpay` (:442, 60 s timeout), `delpay`/`delinvoice` cleanup (:365–375).
- **Outputs**: `ExecutionResult` (from modules/rebalance_execution.py) carrying
  success/fee_msat/fee_ppm, `payment_pending`, `excluded_channels` (directed
  `scid/dir` exclusions for retry, :156–194), and `failure_data` with failure class and
  route summary. Side effect: failure observations written to the segment observation
  store (:208–275).

## Invariants

- **NX-1** No route executes unless validation passes: first hop channel ==
  `source_channel_id`, final hop channel == `dest_channel_id`, final hop id == our node,
  final amount == `amount_sats*1000`, hop amounts non-increasing, every hop carries
  `id/channel/amount_msat/delay` (`_validate_route`, :277–332).
- **NX-2** The fee budget is a hard ceiling: planned fee = first-hop msat minus delivered
  msat; if it exceeds `max_fee_sats*1000` the route is rejected pre-send with
  `native_route_over_budget` (:323–331, :403).
- **NX-3** Unresolved payments are never reported as failures: `waitsendpay` code 200,
  proxy `RPCTimeoutError`, or `waitsendpay_status=pending` set `payment_pending=True`,
  keep invoice and payment records, emit no exclusions, and surface the planned fee so
  the engine holds the budget for the reconciliation sweep (`_is_payment_unresolved`
  :342–363, handler :474–494).
- **NX-4** Failed (terminal) payments are cleaned up: `delpay(status=failed)` +
  `delinvoice(status=unpaid)` are attempted on every terminal failure that raises after
  the invoice RPC (`_cleanup_failed_payment` :365–375, called only from the exception
  handler at :508) — never on pending ones. Gap: the two malformed-invoice-response
  early returns (`invoice` not a dict / missing `payment_hash`, :422–428) return without
  cleanup, so a server-side invoice created under that label could linger unpaid until
  its 300 s expiry. Validation failures (:402–407) precede invoice creation and need no
  cleanup.
- **NX-5** Failure-observation confidence is attribution-scaled: channel+direction known
  → 0.85; channel only → both directions at 0.425; no attribution → 0.85/n over the
  middle-hop suspect set with a 0.2 floor (constants :205–206, logic :208–257).
- **NX-6** *Inferred* retry exclusions never name our own pinned hops: when CLN does not
  identify the erring channel, only middle hops (`route[1:-1]`) become exclusions, and
  only for `liquidity`/`fee` failure classes (`_exclude_from_failure` :156–194). When CLN
  *does* name `erring_channel`, that channel is excluded verbatim (:162–167) — which can
  be our own source or dest hop if the failure was local; the middle-hop restriction
  applies only to the no-attribution fallback.
- **NX-7** Reported success fees are actuals, not estimates: on success `fee_msat` is
  derived from `waitsendpay`'s `amount_sent_msat` minus delivered amount (falling back to
  the route's first hop), and `fee_ppm` from that (:453–465).

## Revenue role

Indirect but close to the money: every successful execution spends real sats (routing
fees) to position liquidity where forwards earn. NX-2 and NX-3 are the two claims that
keep realized rebalance cost from exceeding what the planner authorized.

## Observable surface

Not directly observable as its own artifact. Its behavior shows up in:
`revenue-spend-ledger.json` / rebalance cost tables (fees it reports get recorded by the
engine), the `["revenue","segment-observations"]` datastore key (failure observations it
writes via the store, exported at modules/rebalance_engine_v2.py:2970–2998), and
`rebalance_history` rows with `status='pending_settlement'` for NX-3 cases.

## Uncertainties

- `INVOICE_EXPIRY_SEC=300` vs `SENDPAY_TIMEOUT_SEC=60`: a payment pending past invoice
  expiry relies entirely on the engine's reconciliation sweep; behavior if the invoice
  expires while the HTLC is in flight is untested here.
- `stable_failure_reason` (:67–80) maps error strings for coordination-overlay reporting;
  whether its taxonomy matches what hive peers expect was not verified.
- `_failure_class` keys on substring matching of CLN error text (:127–136); new CLN
  failcode spellings would silently degrade to `unknown` (lower observation value, not
  incorrect).
- Duplicate-label collision: labels are `rebal-native-<ms>-<scid>`; two executions for
  the same dest within the same millisecond would collide (practically impossible under
  the engine's single-flight lock, but unenforced here).
