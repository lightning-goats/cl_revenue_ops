# Core Lightning Compatibility

## Minimum Supported Version

Minimum supported Core Lightning: `v24.11.1+`.

`cl-revenue-ops` uses the askrene v3 route path. `v24.08.1` contains `getroutes` and
`askrene-listlayers`, but it does not expose the full askrene layer lifecycle
used by the route layer integration. In Polar, `askrene-create-layer` returns
`Unknown command` on `v24.08.1`, so `v24.08.1` is not a valid runtime floor for
this project.

## Polar Compatibility Record

This matrix should be rerun before opening or updating the `lightningd/plugins`
listing PR. Do not claim support for a Core Lightning version that has not passed
the smoke checklist below.

| Core Lightning | Result | Evidence |
| --- | --- | --- |
| `v24.08.1` | Not supported | `getroutes` and `askrene-listlayers` exist, but `askrene-create-layer` is missing. |
| `v24.11.1` | Pass | askrene create/update/bias/remove layer lifecycle passes; plugin starts with `pyln-client==24.11.1`; `revenue-status`, `revenue-fee-debug`, `revenue-rebalance-debug`, and `revenue-hive-hints-status` respond. |
| `v25.02.2` | Pass | askrene layer creation passes; plugin starts with documented dependencies; `revenue-status`, `revenue-rebalance-debug`, and fresh live `hive-export-hints` consumption respond. |
| `v25.05` | Pass | askrene layer creation passes; plugin starts with documented dependencies; `revenue-status`, `revenue-rebalance-debug`, and fresh live `hive-export-hints` consumption respond. |
| `v25.09.3` | Pass | askrene layer creation/removal passes; plugin starts after adding a Python runtime to the Polar image; `revenue-status` and `revenue-rebalance-debug` respond. |
| `v25.12` | Pass | askrene layer creation passes; plugin starts after adding a Python runtime to the Polar image; `revenue-status`, `revenue-rebalance-debug`, and fresh live `hive-export-hints` consumption respond. |
| `v26.06.6` | Pass, superseded for production | Exact revision `4c6ec87` ran on a fresh official image with four funded public channels. A mixed CLN/LND Polar matrix settled 60/60 payments and the node forwarded 49 with no failed forwards. Fee, rebalance, profitability, budget, and reconciliation surfaces passed before and after plugin restart. The later `v26.06.7` security release means this is retained as compatibility evidence, not a production recommendation. |
| [`v26.06.7`](https://github.com/ElementsProject/lightning/releases/tag/v26.06.7) | Mandatory lane pending official artifacts | Core Lightning published the embargoed security point release on 2026-08-28. Equal-runtime source and Docker tournament artifacts were not yet publicly available during the 2026-08-28 rounds. Run the complete smoke and crossed CLBOSS lane as soon as official artifacts are available; do not infer compatibility from `v26.06.6`. |

## Real-Channel Askrene Smoke

A three-node `v25.12` Polar topology (`alice -> bob -> carol`) was also tested
with real public channels. The smoke created an askrene layer, applied
`askrene-update-channel` and `askrene-bias-channel` to real
`short_channel_id_dir` values, and confirmed `getroutes` returned the expected
two-hop path through the temporary layer.

Observed path:

```text
132x1x0/0 -> 120x1x0/1
```

Some Polar CLN images do not include Python by default. For those images, install
a Python runtime before applying the plugin dependency smoke. This does not
change the Core Lightning compatibility result; it only affects the test image's
ability to run Python plugins.

## Smoke Checklist

For each tested CLN version:

1. Start a one-node Polar network with `bitcoind` and a CLN node for the target version.
2. Verify `getinfo` reports the expected CLN version.
3. Verify askrene layer lifecycle RPCs:
   - `askrene-create-layer`
   - `askrene-update-channel`
   - `askrene-bias-channel`
   - `askrene-remove-layer`
4. Install the documented Python dependencies into the test runtime.
5. Copy the current `cl-revenue-ops` plugin and modules into the Polar node.
6. Start the plugin dynamically.
7. Verify at least:
   - `revenue-status`
   - `revenue-fee-debug`
   - `revenue-rebalance-debug summary_only=true`
8. For the newest supported release, attach a fresh identity to the mixed-client
   Polar graph and require funded traffic through both a CLN and LND sink.
   Verify source hashes, dry-run policy immutability, paused rebalance
   suppression, profitability/budget/economic reconciliation, and plugin
   restart before removing the temporary node.

The minimum supported CLN version is the oldest version that passes this full
checklist, not merely the oldest version with `getroutes`.
