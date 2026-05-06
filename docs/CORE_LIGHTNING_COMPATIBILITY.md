# Core Lightning Compatibility

## Minimum Supported Version

Minimum supported Core Lightning: `v24.11.1+`.

`cl-revenue-ops` uses the askrene v3 route path with optional mycelium/hive
route-layer coordination. `v24.08.1` contains `getroutes` and
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
| `v25.09.3` | Pass | askrene layer creation/removal passes; plugin starts after adding a Python runtime to the Polar image; `revenue-status` and `revenue-rebalance-debug` respond. |

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
   - `revenue-hive-hints-status`

The minimum supported CLN version is the oldest version that passes this full
checklist, not merely the oldest version with `getroutes`.
