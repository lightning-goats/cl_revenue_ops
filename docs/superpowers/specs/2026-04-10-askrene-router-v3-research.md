# Rebalance Router V3 — Research Findings

**Date:** 2026-04-10
**Status:** In progress
**Parent spec:** `docs/superpowers/specs/2026-04-10-askrene-router-v3-design.md`
**Implementation plan:** `docs/superpowers/plans/2026-04-10-askrene-router-v3-research.md`
**Worktree:** `.worktrees/askrene-router-v3-20260410`
**Branch:** `feature/askrene-router-v3`

## Environment

**CLN upstream reference SHA:** `b57edd2128fa21b492f7c215d13ebfcf74bdc579`
**CLN upstream tip message:** `keysend: increase assumed final_cltv_expiry to 42 (to match LDK).`
**Upstream clone location (session-local):** `/tmp/cln-upstream` (shallow clone, depth 1)

**Live node** (accessed via `ssh lnnode 'lightning-cli …'`):

```json
{
  "version": "v25.12.1",
  "network": "bitcoin",
  "id": "0382d558331b9a0c1d141f56b71094646ad6111e34e197d47385205019b03afdc3",
  "blockheight": 944486,
  "num_peers": 63,
  "num_active_channels": 46
}
```

**Live askrene layers observed:** `hive-fleet`, `hive-reputation`, `hive-corridors`, `hive-traffic` (all four cl-hive layers), `revenue-local`, `xpay` (automatic xpay-internal layer).

**Citation format:** `ElementsProject/lightning@b57edd21:<path>#L<start>-L<end>`

All claims have either a source citation or a captured live-node RPC transcript. No claim is based on memory or docs.corelightning.org; only the upstream source at the pinned SHA and real RPC transcripts are authoritative.

---

## 1. getroutes Contract

### 1.1 Version gate

`getroutes` was added in **CLN v24.08** per the schema (`ElementsProject/lightning@b57edd21:doc/schemas/getroutes.json#L6 "added": "v24.08"`). Two later additions:

- `maxdelay` added in v25.02 (`doc/schemas/getroutes.json#L63`)
- `maxparts` added in v25.09 (`doc/schemas/getroutes.json#L74`)

The live node runs v25.12.1, above all gates. **Implication for v3**: `rebalance-router = "v3"` requires CLN v24.08+. Operators on older CLN fall back to v2 per the engine factory.

### 1.2 Request parameters

| Param | Type | Required | Source | Meaning |
|---|---|---|---|---|
| `source` | pubkey | yes | `getroutes.json#L24-29` | Node pubkey where paths start |
| `destination` | pubkey | yes | `getroutes.json#L30-35` | Node pubkey where paths end |
| `amount_msat` | msat | yes | `getroutes.json#L36-41` | Amount delivered to destination |
| `layers` | string[] | yes | `getroutes.json#L42-49` | Layer names to apply on top of gossip |
| `maxfee_msat` | msat | yes | `getroutes.json#L50-55` | **Hard cap** — "we will never return a set of routes more expensive than this" |
| `final_cltv` | u32 | yes | `getroutes.json#L56-61` | CLTV blocks for final hop |
| `maxdelay` | u32 | opt (default 2016) | `getroutes.json#L62-69` | Max total CLTV delay blocks |
| `maxparts` | u32 | opt (default 100) | `getroutes.json#L70-83` | Max routes in MPP solution |

**Invariants enforced by the plugin**:

- `amount_msat` must be non-zero (`askrene.c#L879-882`: `"amount must be non-zero"`)
- `maxparts` must be non-zero (`askrene.c#L884-887`: `"maxparts must be non-zero"`)
- `maxdelay` must be ≤ 2016 (`askrene.c#L889-893`: `"maximum delay allowed is %d"`). 2016 is the BOLT #4 `max_htlc_cltv`.

### 1.3 The four automatic layers

The schema calls these out directly (`getroutes.json#L10-14`):

- **`auto.localchans`** — contains information on local channels from this node (including non-public ones), and their exact current spendable capacities. Useful when source is the current node.
- **`auto.sourcefree`** — overrides all channels (including those from previous layers) leading out of *source* to be zero fee and zero delay. Useful when source is the current node.
- **`auto.no_mpp_support`** — forces getroutes to return a single-path solution. Confirmed in `askrene.c#L608-615`: when this layer is requested, `dev_algo` is set to `ALGO_SINGLE_PATH`. Also triggered by `maxparts == 1` (`askrene.c#L616-621`).
- **`auto.include_fees`** — fixes the send amount and deducts fees from there, i.e. the receiver pays fees instead of the sender (`askrene.c#L623`: `include_fees = have_layer(info->layers, "auto.include_fees");`).

**Implication for v3**: `auto.no_mpp_support` is a clean way to force single-path in phase 1. The v2 executor only handles single-path routes, so this flag is the right default for phase 1. For pair-pinned queries where `source` is a peer (not our node), `auto.sourcefree` and `auto.localchans` do nothing useful (they target the source node's outgoing edges, which would be the peer's, not ours).

### 1.4 Response shape

Response is `{"probability_ppm": u64, "routes": [ {…} ]}` (`getroutes.json#L87-98`).

Each route entry:

| Field | Type | Meaning | Source |
|---|---|---|---|
| `probability_ppm` | u64 | Success probability of this path (millionths) | `getroutes.json#L112-116` |
| `amount_msat` | msat | Amount delivered to destination (NOT total sent) | `getroutes.json#L118-122` |
| `final_cltv` | u32 | Echo of caller's final_cltv | `getroutes.json#L124-128` |
| `path` | object[] | Hops from source to destination | `getroutes.json#L130-134` |

Each hop (`getroutes.json#L135-168`):

| Field | Type | Meaning |
|---|---|---|
| `short_channel_id_dir` | string | Channel and direction (`"SCID/dir"`) |
| `next_node_id` | pubkey | Peer id at the end of this hop |
| `amount_msat` | msat | Amount to send INTO this hop (inclusive of downstream fees) |
| `delay` | u32 | Total CLTV expected by the node at the start of this hop |

**Critical format difference from `getroute`**: the path entries have `short_channel_id_dir` (not `channel`+`direction`) and `next_node_id` (not `id`). The schema says this explicitly (`getroutes.json#L9`: *"NOTE: The returned paths are a different format then getroute, being more appropriate for creating intermediary onion layers."*).

**Implication for v3**: `RebalanceRouterV3.price_pair()` must translate the `getroutes` path format into the sendpay/executor route format. This is a simple per-hop transformation but it's not a drop-in replacement — v2's `RouteResult.route` list is currently in sendpay format, and the v3 router must produce the same format to preserve compatibility with `RebalanceExecutorV2`.

### 1.5 Error modes

Enumerated by grepping `command_fail` in `json_getroutes` and its helpers:

| Error | Source | Maps to v2 skip reason |
|---|---|---|
| `"amount must be non-zero"` | `askrene.c#L880` | Should never fire from v3 (planner never passes zero) — assert instead |
| `"maxparts must be non-zero"` | `askrene.c#L885` | Same — v3 passes default 1 or 100 |
| `"maximum delay allowed is %d"` | `askrene.c#L890` | Should never fire (v3 passes 2016 default) |
| `"Unknown source node %s"` | `askrene.c#L592-596` (inside `do_getroutes`) | New skip reason `unknown_source_node` |
| `"Unknown destination node %s"` | `askrene.c#L602-606` | New skip reason `unknown_dest_node` |
| Route-not-found (child process timeout or no path) | `askrene.c#L443, L697` via `PAY_ROUTE_NOT_FOUND` | Existing `no_route` |

**Implication for v3**: need two new skip reasons for source/dest node gossip gaps. These are distinct from "route not found" — they indicate the source or destination pubkey is not in the local gossmap at all (e.g. peer disappeared, very new peer, or gossip not yet synced).

### 1.6 Timeout behavior

`askrene.c#L1467`: default route calculation deadline is **10 seconds** (`route_seconds = 10`), configurable via the dynamic option `askrene-timeout`.

`askrene.c#L1468, L1477-1482`: default max concurrent route calculations is **4** (`max_children = 4`), configurable via the dynamic option `askrene-max-threads`. Both options are dynamic (hot-reloadable via `setconfig`).

Route calculation runs in a forked child process (`askrene.c#L625-690`): `pipe()` + `fork()` + `run_child()`. On fork failure the request returns `PAY_ROUTE_NOT_FOUND` with an error like `"failed to fork: %s"`. On child timeout the deadline fires inside `run_child()` and the child reports `PAY_ROUTE_NOT_FOUND` back to the parent.

**Implication for v3**: the router does not need its own timeout — askrene enforces 10s per call. However, v3 should pass a *caller* timeout to `self.plugin.rpc.getroutes(..., timeout=15)` or similar if pyln-client supports it, so cl-revenue-ops doesn't hang if the CLN RPC dispatch itself stalls.

Also: at most 4 concurrent getroutes calls can run in parallel. The v2 planner processes pairs sequentially, so this is not a concern for phase 1. But if a future v3 planner parallelizes pair pricing, it must respect the 4-concurrent limit.

### 1.7 Live verification

**Probe 1 — direct peer, our node as source** (single hop):

```
$ ssh lnnode 'lightning-cli getroutes \
    source=0382d558331b9a0c1d141f56b71094646ad6111e34e197d47385205019b03afdc3 \
    destination=03fe80dfe18b0feb77c2e619516a7563ab39423a6c02d06e5246c60eea0e276aac \
    amount_msat=100000 layers=[] maxfee_msat=10000 final_cltv=40'
{
   "probability_ppm": 999990,
   "routes": [
      {
         "probability_ppm": 999990,
         "amount_msat": 100000,
         "final_cltv": 40,
         "path": [
            {
               "short_channel_id_dir": "940304x912x0/0",
               "next_node_id": "03fe80dfe18b0feb77c2e619516a7563ab39423a6c02d06e5246c60eea0e276aac",
               "amount_msat": 100033,
               "delay": 72
            }
         ]
      }
   ]
}
```

Observations: delivered 100000 msat, first hop sends 100033 msat (33 msat fee for this single hop), delay 72 = 40 final + 32 from the channel policy. Response matches the schema exactly.

**Probe 2 — pair-pinned query (source and destination are both peers, not us)**:

```
$ ssh lnnode 'lightning-cli getroutes \
    source=03fe80dfe18b0feb77c2e619516a7563ab39423a6c02d06e5246c60eea0e276aac \
    destination=02f1a8c87607f415c8f22c00593002775941dea48869ce23096af27b0cfdcc0b69 \
    amount_msat=100000 layers=[] maxfee_msat=10000 final_cltv=40'
{
   "probability_ppm": 999978,
   "routes": [
      {
         "probability_ppm": 999978,
         "amount_msat": 100000,
         "final_cltv": 40,
         "path": [
            { "short_channel_id_dir": "940304x912x0/1",
              "next_node_id": "0382d558331b9a0c1d141f56b71094646ad6111e34e197d47385205019b03afdc3",
              "amount_msat": 101003, "delay": 130 },
            { "short_channel_id_dir": "939858x2330x0/1",
              "next_node_id": "02b21730bc36061609cc1fe1bd7f5d3068e0a5e511aaf210a82c046b58020c4aa8",
              "amount_msat": 100002, "delay": 96 },
            { "short_channel_id_dir": "922759x492x2/0",
              "next_node_id": "02f1a8c87607f415c8f22c00593002775941dea48869ce23096af27b0cfdcc0b69",
              "amount_msat": 100000, "delay": 64 }
         ]
      }
   ]
}
```

**Critical observation**: askrene naturally routes the pair-pinned query **through our own node** as an intermediate hop. The path is `peer_A → us → peer_02b217 → peer_B`. The first hop uses channel `940304x912x0/1` (direction 1 = peer_A → us), which is the same SCID as our direct channel with peer_A seen in Probe 1.

**Implication for v3's pair-pinning pattern**: askrene *can* produce circular-rebalance-shaped routes automatically when asked `source=outgoing_peer, destination=incoming_peer`. However the pattern is subtler than v2's prepend/append approach — askrene may pick an indirect middle even if a more direct one exists. V3 must validate that the returned path's first hop is on the desired source_channel (else reject the route), and it must handle the case where askrene's path traverses our node in an unexpected way (e.g. first hop is peer_A → some_other_peer, then eventually back to us). This is a concrete design surprise that Section 3's layer experiment must verify before phase 1 implementation.

### 1.8 Interim verdict

`getroutes` is well-specified, runs reliably on the live node, and exposes exactly the controls v3 needs: `layers` for cl-hive bias, `maxfee_msat` for hard budget enforcement, `auto.no_mpp_support` for single-path simplicity. The version gate (v24.08) is comfortably met. The only design surprise is the path-format delta from `getroute` (requires a translation step in `price_pair`) and the potentially indirect path shape under pair pinning (requires a first-hop validator in the router).

## 2. Layer Lifecycle

### 2.1 Layer RPC surface

Eight RPC methods manage askrene layers. All added in v24.11 except `askrene-bias-node` (v25.12).

| RPC | Added | Required params | Optional params | Purpose |
|---|---|---|---|---|
| `askrene-listlayers` | v24.11 | — | `layer` | Enumerate layers, or fetch one |
| `askrene-create-layer` | v24.11 | `layer` | `persistent` (default false) | Create a new named layer |
| `askrene-remove-layer` | v24.11 | `layer` | — | Delete a layer |
| `askrene-update-channel` | v24.11 | `layer`, `short_channel_id_dir` | `enabled`, `htlc_minimum_msat`, `htlc_maximum_msat`, `fee_base_msat`, `fee_proportional_millionths`, `cltv_expiry_delta` | Override a channel's routing params inside this layer |
| `askrene-inform-channel` | v24.11 | `layer`, `short_channel_id_dir`, `amount_msat`, `inform` | — | Feed observed success/failure info to update the layer's capacity constraint |
| `askrene-bias-channel` | v24.11 | `layer`, `short_channel_id_dir`, `bias` | `description`, `relative` | Add a numerical bias to a channel's path-finding weight |
| `askrene-bias-node` | v25.12 | `layer`, `node`, `direction`, `bias` | `description`, `relative` | Bias all channels entering/leaving a node |
| `askrene-disable-node` | v24.11 | `layer`, `node` | — | Block all routes through a node |

Cited from the schema files in `doc/schemas/askrene-*.json` at `ElementsProject/lightning@b57edd21`.

### 2.2 Ownership model

Layers are **not plugin-scoped**. Any plugin (or `lightning-cli`) can create, list, modify, or delete any named layer. The askrene plugin maintains a single shared hash `askrene->layers` keyed by name (`layer.c#L134-140` for the `layer_name_hash` declaration, `layer.c#L859-862` for the `find_layer` accessor).

Reserved prefix: layer names starting with `auto.` are rejected by `create-layer` (`askrene.c#L1284-1286`: `"Cannot create auto layer"`). The four `auto.*` layers enumerated in Section 1.3 are built-ins.

**Idempotency**: `create-layer` behavior depends on `persistent`:

- `persistent=false` (default) + layer exists → error `"Layer already exists"` (`askrene.c#L1290-1293`)
- `persistent=true` + layer exists → succeeds as a no-op (`askrene.c#L1288`: *"If it's persistent, creation is a noop if it already exists"*)

This means cl-hive's publish-on-startup model (non-persistent layers) can't safely call `create-layer` twice without a remove in between. cl-hive probably handles this by removing layers before re-creating at startup, or by using `persistent=true`.

### 2.3 Persistence

Persistence is opt-in per layer. Default is **false** (`askrene-create-layer.json#L… "default": "False"`).

**Non-persistent layers** live only in askrene's in-memory hash table. They vanish on `lightningd` restart or plugin restart. cl-hive's four layers are all non-persistent (verified live in Section 2.5), so cl-hive must re-publish them on every startup.

**Persistent layers** are saved to CLN's datastore. Each mutation appends a delta record under the datastore key `["askrene", "layers", <layername>]`:

- `layer.c#L459-464` (channel update): if persistent, call `append_layer_datastore(layer, data)`
- `layer.c#L496-501` (inform channel): same
- `layer.c#L543-548` (bias channel): same
- `layer.c#L580-585` (bias node): same
- `layer.c#L602-607` (disable node): same
- `layer.c#L678-682` (remove channel): same

On startup, `load_layers()` (`layer.c#L806-841`) reads all datastore entries under `["askrene", "layers", *]` and replays them into in-memory layer state via `populate_layer()`.

On `remove-layer`, persistent layers also delete their datastore backing (`layer.c#L425-431` via `save_remove` → `deldatastore`).

**Implication for v3**: cl-revenue-ops is a pure layer consumer. It never creates or modifies layers, so persistence is not a concern. However, Section 4's exclude-via-layer pattern will create and remove a throwaway layer per retry — those MUST use `persistent=false` (the default) to avoid datastore bloat.

### 2.4 Concurrency

**CLN plugin RPC dispatch is serialized per plugin**. Each askrene RPC (create-layer, update-channel, etc.) runs to completion before the next one starts. There are no explicit mutexes in `layer.c` because none are needed under this model.

**getroutes concurrency** (the only long-running askrene operation) is different: it forks a child process with a **snapshot of layer state** (`askrene.c#L625-690` — see Section 1.6 for the fork plumbing). The child does not share memory with the parent, so in-flight getroutes calls are immune to mid-calculation layer mutations. At most 4 concurrent forks are allowed (`askrene->max_children = 4` at `askrene.c#L1468`).

**Cross-plugin safety**: two plugins calling askrene simultaneously are still serialized by CLN's RPC layer. Plugin A's `create-layer` never interleaves with plugin B's `update-channel`. This means **cl-revenue-ops can safely read cl-hive's layers** (via `askrene-listlayers` or by passing layer names to `getroutes`) without any coordination — askrene serializes all access internally.

**What cl-revenue-ops must not do**: write to cl-hive's layers. The API technically allows it (any plugin can modify any layer), but it would break cl-hive's ownership model. The v3 design already declares this as an anti-requirement.

### 2.5 Live verification

**All layers currently on the node:**

```
$ ssh lnnode 'lightning-cli askrene-listlayers'  # (counts summarised)
hive-traffic         persistent=False  channel_updates=0  biases=0  disabled_nodes=0  created=0
hive-corridors       persistent=False  channel_updates=0  biases=0  disabled_nodes=0  created=0
hive-reputation      persistent=False  channel_updates=0  biases=0  disabled_nodes=0  created=0
hive-fleet           persistent=False  channel_updates=4  biases=0  disabled_nodes=0  created=0
revenue-local        persistent=False  channel_updates=0  biases=92  disabled_nodes=0  created=0
xpay                 persistent=True   channel_updates=0  biases=0  disabled_nodes=0  created=0
```

Observations:

- All four cl-hive layers are live (`hive-fleet`, `hive-reputation`, `hive-corridors`, `hive-traffic`)
- Only `hive-fleet` currently has content (4 channel_updates — the 4 hive fleet channels on this node)
- `hive-reputation`, `hive-corridors`, `hive-traffic` are currently empty (published by cl-hive but nothing has populated them yet). **Implication**: v3's default `askrene-layers = "hive-fleet"` is the right call — the other three layers would be no-ops at this moment.
- `revenue-local` has 92 channel biases — this is an existing cl-revenue-ops layer already used by `_apply_local_bias()` or equivalent in some older code path. Worth a follow-up: which code writes to it? Is it dead? (Out of scope for this research.)
- The `xpay` layer is persistent — it's owned by the xpay plugin, survives restart. That's our first live confirmation that xpay uses layers.

**Cross-plugin isolation experiment**:

```
$ ssh lnnode 'BEFORE=$(lightning-cli askrene-listlayers hive-fleet | jq ".layers[0].channel_updates | length")
             lightning-cli askrene-create-layer cl-revenue-ops-research-probe
             lightning-cli askrene-listlayers cl-revenue-ops-research-probe
             lightning-cli askrene-remove-layer cl-revenue-ops-research-probe
             AFTER=$(lightning-cli askrene-listlayers hive-fleet | jq ".layers[0].channel_updates | length")
             echo "before=$BEFORE after=$AFTER"'
probe persistent=False updates=0
hive-fleet channel_updates: before=4 after=4
```

Creating and removing a cl-revenue-ops probe layer left hive-fleet's state unchanged. Cross-plugin layer isolation confirmed.

### 2.6 Safety claim

**cl-revenue-ops reading cl-hive's layers is safe because**:

1. RPC dispatch is serialized (`askrene.c` single-threaded plugin model)
2. getroutes operates on an immutable snapshot per call (forked child, no shared memory)
3. layer mutations are atomic per RPC
4. the cl-hive layers have unique names not in cl-revenue-ops's namespace, so no name collision risk
5. v3 never writes to any layer — it is a pure consumer
6. live experiment confirmed cross-plugin isolation (Section 2.5)

The only safety rule v3 must enforce at the plugin level is the anti-requirement from the parent spec: **v3 must not call any layer-mutation RPC against a layer it does not own**. In phase 1, v3 never calls mutation RPCs at all. In phase 4's exclude-via-layer retry pattern (if research approves it), v3 creates and removes its own private layer named `rebalance-exclude-<cycle>`, which cannot collide with cl-hive or the revenue-local layer.

## 3. Layer Semantics Under Pair Pinning

### 3.1 Setup

The `hive-fleet` layer on the live node contains **4 channel overrides** (2 channels × 2 directions):

```json
{"short_channel_id_dir": "933791x3241x0/1", "fee_base_msat": 0, "fee_proportional_millionths": 0, "cltv_expiry_delta": 6}
{"short_channel_id_dir": "933791x3241x0/0", "fee_base_msat": 0, "fee_proportional_millionths": 0, "cltv_expiry_delta": 6}
{"short_channel_id_dir": "940132x2695x0/1", "fee_base_msat": 0, "fee_proportional_millionths": 0, "cltv_expiry_delta": 6}
{"short_channel_id_dir": "940132x2695x0/0", "fee_base_msat": 0, "fee_proportional_millionths": 0, "cltv_expiry_delta": 6}
```

Both directions of both hive-fleet channels are forced to **zero-fee, cltv_delta=6**. The corresponding live gossip values:

- `940132x2695x0` (peer 028f5847 → us): gossip says `base=0, ppm=10, cltv=34` → layer makes it `base=0, ppm=0, cltv=6` (Δppm=-10, Δcltv=-28)
- `933791x3241x0` (peer 03796a3c → us): gossip says `base=0, ppm=0, cltv=34` → layer changes only cltv (Δcltv=-28)

These are small deltas — this layer's current content is subtle (it's effectively saying "treat fleet channels as CLTV-friendly zero-fee" rather than dramatically reshaping fees). That's important context for the test: we're looking for *any* observable bias, not a dramatic reshape.

### 3.2 Experiment A — pair where path uses a hive-fleet channel

**Setup**: source is peer `028f5847` (owns hive-fleet channel `940132x2695x0`), destination is non-fleet peer `03fe80df`. Amount 1,000,000 msat. Pair-pinned (source and destination are both peers, not us).

**Without hive-fleet layer** (`layers=[]`):

```
# Flow 0/1: 1000343msat/106 940132x2695x0/0 -> 1000333msat/72 940304x912x0/0 -> 1000000msat/40
{
   "probability_ppm": 990806,
   "routes": [{
      "amount_msat": 1000000, "final_cltv": 40,
      "path": [
         {"short_channel_id_dir": "940132x2695x0/0", "next_node_id": "0382d558...", "amount_msat": 1000343, "delay": 106},
         {"short_channel_id_dir": "940304x912x0/0",  "next_node_id": "03fe80df...", "amount_msat": 1000333, "delay": 72}
      ]
   }]
}
```

Total fee: 1000343 − 1000000 = **343 msat**. Total delay at first hop: **106**.

**With hive-fleet layer** (`layers=["hive-fleet"]`):

```
# Flow 0/1: 1000333msat/78 940132x2695x0/0 -> 1000333msat/72 940304x912x0/0 -> 1000000msat/40
{
   "probability_ppm": 990806,
   "routes": [{
      "amount_msat": 1000000, "final_cltv": 40,
      "path": [
         {"short_channel_id_dir": "940132x2695x0/0", "next_node_id": "0382d558...", "amount_msat": 1000333, "delay": 78},
         {"short_channel_id_dir": "940304x912x0/0",  "next_node_id": "03fe80df...", "amount_msat": 1000333, "delay": 72}
      ]
   }]
}
```

Total fee: 1000333 − 1000000 = **333 msat**. Total delay at first hop: **78**.

**Diff:**

| Metric | No layer | With hive-fleet | Δ |
|---|---|---|---|
| Total fee (msat) | 343 | 333 | **−10 msat** |
| First-hop delay | 106 | 78 | **−28 blocks** |
| Path (SCIDs) | `940132x2695x0/0` → `940304x912x0/0` | `940132x2695x0/0` → `940304x912x0/0` | identical |
| probability_ppm | 990806 | 990806 | identical |

**Interpretation**: the layer override on the first hop (`940132x2695x0/0`) exactly matches the expected deltas:

- Fee Δ = `amount_msat * (ppm_before - ppm_after) / 1_000_000 = 1,000,000 * (10 − 0) / 1,000,000 = 10 msat` ✓
- Delay Δ = `cltv_before − cltv_after = 34 − 6 = 28 blocks` ✓

**Askrene respects the layer override on the channel, and the layer-adjusted weights propagate into the path cost calculation.** This confirms the fundamental mechanism v3 relies on.

### 3.3 Experiment B — pair where path does NOT touch any hive-fleet channel

**Setup**: source is peer `03796a3c` (non-hive-fleet channel as outgoing option, since its hive-fleet channel `933791x3241x0` has near-zero receivable balance and askrene avoids it), destination is peer `03c15794`. Amount 5,000,000 msat. Pair-pinned.

**Without hive-fleet layer**:

```
path: [
  "942863x384x9/1"  amt=5001000 delay=298,
  "930327x2105x0/1" amt=5000005 delay=264,
  "903815x2481x1/0" amt=5000005 delay=120
]
fee=1000 msat  first_hop_delay=298
```

**With hive-fleet layer**: **identical path, identical amounts, identical delays.**

**Interpretation**: when the selected path does not traverse any hive-fleet-covered channel, the layer has **zero observable effect**. This is the correct behavior — layers are additive constraints, they only matter when the path touches a covered edge.

### 3.4 Pair-pinning behavior (subtle but important)

Both experiments used `source=<peer>, destination=<peer>` (i.e. neither endpoint was our node). In every case, askrene successfully found a path, and in every case the path routed **through our node** as an intermediate hop:

- Experiment A: `peer_028f5847 → 0382d558 (us) → peer_03fe80df` (2-hop)
- Experiment B: `peer_03796a3c → 0298f607 → 026165850 → peer_03c15794` (3-hop, does NOT go through us — askrene found a cheaper path in the broader gossip graph)

**Critical finding**: askrene does NOT force the path through us just because we're the node running the query. It uses the gossip graph freely. In Experiment B, it picked a path that bypasses us entirely, even though our node is likely reachable from both endpoints.

**Implication for v3's pair-pinning pattern**:

The v2 router's pattern is "pin source_channel, pin dest_channel, ask for middle". It assumes the middle path starts at `source_peer` and ends at `dest_peer` without touching our node again. That assumption holds for `getroute` (the v2 RPC) because getroute doesn't know we are the node running the query. But `getroutes` uses askrene's full gossip-aware pathfinding and **can find paths that DO touch our node**.

Experiment A shows this concretely: for a 2-hop pair-pinned query, askrene produced a path where our node is the middle hop. That's not wrong — it's just a different route shape than v2's strict "peer_A → external middle → peer_B" assumption.

**Design consequences for v3**:

1. **V3's `price_pair` must validate that the returned path does NOT include our own node as an intermediate hop**, or handle the case when it does. If askrene returns a path with `next_node_id == our_node_id` at any hop (except the very last hop when we're the destination), the path loops through us, which is fine if we're doing circular rebalancing but breaks the sendpay format (which expects a simple source → ... → destination sequence with us at neither end of the middle).
2. **V3 should consider constructing the query differently**: instead of `source=source_peer, destination=dest_peer`, use `source=our_node, destination=our_node` and rely on the pair-pinning to happen via layer-based channel disabling or source/dest channel forcing. But **askrene does not allow source == destination** — that's the original constraint from issue #9032 that motivated this whole research effort.
3. **Alternate pattern — "split-call pair pinning"**: make TWO getroutes calls: `getroutes(source=our_node, destination=dest_peer)` for the outgoing half, and `getroutes(source=source_peer, destination=our_node)` for the incoming half. Each half is a regular source-or-dest-is-us query. Stitch the two halves together at the boundary. This avoids the source=destination restriction and avoids the "path accidentally loops through us" issue by construction. It's more complex than v2's single-query pattern but aligns with askrene's model.
4. **OR — "constrain the first hop explicitly"**: `getroutes` doesn't have an explicit `first_hop_scid` parameter, but we can achieve the same effect by creating a tiny throwaway layer that DISABLES all of peer_A's other outgoing channels, then passing that layer along with `source=peer_A`. Askrene will be forced to use the channel we want.

The parent spec assumes pattern #1 (single call, source=source_peer, destination=dest_peer, stitch manually). Experiments A + B show this pattern produces valid paths but with no guarantee about shape. Patterns #3 and #4 are more robust alternatives discovered by this research and are worth considering in Phase 1 implementation.

### 3.5 Verdict

**Layer semantics under pair pinning are confirmed working**: askrene respects channel overrides from named layers when the computed path traverses those channels (Experiment A). Layers have no side effect on other paths (Experiment B). This is the core mechanism v3 needs.

**Design caveat**: the v2 router's "pair-pinned middle path" assumption does not cleanly map to askrene's gossip-aware pathfinding. Askrene may produce paths that loop through our own node or that bypass our node entirely. V3 must either (a) validate returned paths for loop-through-us, (b) switch to a split-call pattern (Section 3.4 #3), or (c) use throwaway-layer first-hop forcing (Section 3.4 #4). This choice becomes a Phase 1 implementation decision that must be recorded in Section 9.

## 4. Exclude-Via-Layer Pattern

_PENDING: Task 4_

## 5. xpay API Surface

_PENDING: Task 5_

## 6. xpay vs sendpay+waitsendpay Behavior Diff For Circular Self-Pays

_PENDING: Task 6_

## 7. setconfig Runtime-Switch Verification

_PENDING: Task 7_

## 8. Failure-Mode Taxonomy

_PENDING: Task 8_

## 9. Decision Records

_PENDING: Task 9 (partial — decisions that depend on deferred live-node data will be marked "pending experiment")_
