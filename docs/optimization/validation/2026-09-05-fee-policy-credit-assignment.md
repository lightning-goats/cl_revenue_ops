# Fee-policy mechanics before historical credit assignment

## Result and scope

Added a pure, offline model of the pinned CLN fee and HTLC-range checks. It is
not imported by the runtime controller, does not train a model, and does not
correct the current volume-times-latest-ppm reward. It establishes mechanics
that a historical or online reward correction must respect. No production,
competitor, traffic, timing, scoring or Docker configuration changed.

The historical-learning requirement remains open: retained earnings can inform
realized profitability without establishing which advertised price caused a
payment. Historical bootstrap must not convert missing policy exposure into
confident fee-response training labels.

## Pinned mechanism

Sources are CLN `v26.06.7`, commit
`9f7baf66e1e6b421c0c81a3c3f7c307f8e78a911`:

- [`set_channel_config`](https://github.com/ElementsProject/lightning/blob/9f7baf66e1e6b421c0c81a3c3f7c307f8e78a911/lightningd/peer_control.c#L3611)
  keeps one previous policy. A requested base/ppm/minimum increase or maximum
  decrease replaces that slot and resets its expiry. This decision precedes
  peer-minimum/capacity clamping, so effective readback alone is insufficient.
  A permissive request preserves the previous slot and deadline.
- [Forwarding checks](https://github.com/ElementsProject/lightning/blob/9f7baf66e1e6b421c0c81a3c3f7c307f8e78a911/lightningd/peer_htlcs.c#L858)
  try current fees first and permit previous fees strictly before expiry. Fee
  and HTLC-range checks are independent; they can pass different policy slots.
  CLTV, balance and other routing checks still apply separately.
- [Fee arithmetic](https://github.com/ElementsProject/lightning/blob/9f7baf66e1e6b421c0c81a3c3f7c307f8e78a911/common/amount.c#L608)
  uses integer msat: base plus the floored proportional component.
- The [`setchannel` reference](https://docs.corelightning.org/reference/setchannel)
  documents the default 600-second delay, overpayment acceptance and loss of
  grace on a CLN-daemon restart. Restarting only Revenue Ops does not establish
  a daemon restart. No enforcement delay or restart was changed for this work.

Consequently, an HTLC-limit change can replace the previous fee slot even if
the advertised fee is unchanged. Conversely, a payment exactly matching an old
price can satisfy a lower current price. Neither an RPC acknowledgment nor a
matching paid fee proves the payer's policy knowledge or causal price exposure.

## Retained native-run diagnostic

Read-only comparison of the existing `v36-r234` and `v36-r235` Revenue
`listforwards.json` and `fee-history.json` exports:

| Run | Settled local forwards | Compared | Ambiguous | Actual fee msat | Latest-ppm-only proxy msat | Below / equal / above proxy |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| r234 | 199 | 139 | 60 | 3,561,350 | 3,639,480 | 7 / 130 / 2 |
| r235 | 197 | 150 | 47 | 4,957,379 | 4,985,569 | 7 / 141 / 2 |

Method: for each settled local forward, select the latest fee-history second
at or before its received-time second for its outgoing channel. Exclude the
forward if its received second contains that change, or the latest second has
multiple records. No forward lacked a prior quote. Sum exact observed fee msat
and `floor(out_msat * latest_ppm / 1000000)` over the remaining records.
Only whole-second comparisons are used; no nanosecond attribution is claimed.

These are subsets of local forwarding records, not tournament-wide payment
counts or recomputed tournament scores. Old exports lack complete base-fee,
requested/effective HTLC-limit, monotonic transition and daemon-instance
history. The comparison is therefore descriptive: it neither reconstructs the
full enforcement timeline nor identifies the exact reward consumed by DTS.
Unknown base fees must not be silently treated as known zero for price learning.

One concrete amount example is 250,000 sat earning 193,750 msat (775 ppm), while
a latest 856-ppm-only calculation gives 214,000 msat. The offline model shows
how the old-fee grace slot can permit this; it does not prove that this was the
unique explanation for every retained mismatch. The earlier tournament losses
remain unchanged and are not attributed solely to this discrepancy.

## Model contract and verification

`modules/fee_acceptance_model.py` requires known requested fields, effective
readback, and a complete ordered timeline within one CLN instance. Omitted
request fields are distinct from unknown requests; inconsistent inputs fail
explicitly. An unknown previous slot produces an unknown check when needed,
not fabricated rejection or zero revenue. Integer amounts are restricted to
the local SQLite evidence domain, not CLN's full unsigned amount range.

Tests cover single-slot replacement, each restrictive field, clamping before/
after the grace decision, omitted fields, overpayment, strict expiry, independent
fee/range checks, explicit daemon restart, unknown history, base fees, rounding,
malformed inputs and backward time. All operations are immutable and have no
RPC, wall clock, database, network or model-state dependency. This is a source-
reviewed unit model, not a native-node integration conformance test.

Verification command:

```sh
.venv/bin/python -m pytest -q tests/test_fee_acceptance_model.py tests/test_historical_route_context_replay.py tests/test_architecture_guard.py tests/test_rpc_surface_inventory.py
```

Result: 111 passed in the working tree and 111 passed in an isolated staged-
source copy excluding all unrelated changes. No Sling dependency or action RPC
was added or invoked.
No runtime imports, RPC surface, schema, active learned state or production
configuration changed. Existing unrelated pricing and xrebalance work is kept
outside this change.

## Next integration gates

1. Replace proxy accounting with actual settled fees without assigning them to
   the newest price. Preserve late outcomes and atomic evidence/model cursors.
2. Capture enough request/readback/instance/ordering evidence to identify known
   policy-check intervals, or mark the interval unknown. A fixed ten-minute
   exclusion after only fee changes is insufficient. Even a known enforcement
   interval does not establish the payer's perceived price or demand elasticity.
3. Separate realized-earnings learning, predictive context and causal price
   response. Use old history only at supported granularity; do not invent
   historical inventory, route alternatives or missing exposure.
4. Qualify historical warm starts against recent-only and cold starts using
   chronological validation and unchanged native tournaments. Keep raw
   production records local and production control unchanged until qualified.
