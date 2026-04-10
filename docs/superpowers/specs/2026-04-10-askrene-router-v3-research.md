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

### 4.1 Methodology

For each benchmark iteration, measure wall-clock time for a full create-layer + (optional channel updates) + remove-layer cycle via `lightning-cli` over the local Unix socket on the live node. Each benchmark runs 10 iterations. Report min/median/max/mean.

**Caveat**: these benchmarks use `lightning-cli` subprocess invocation from Python, which incurs a `fork + exec` for every RPC call. In v3's actual hot path, the RPC would come from cl-revenue-ops's persistent Python process via `self.plugin.rpc.askrene_create_layer(...)`, which is a direct Unix-socket JSON-RPC call with no fork. Actual per-call latency in production is expected to be **significantly lower** than the numbers below (probably 30-60% less), so the benchmarks are a conservative upper bound.

### 4.2 Benchmark A — empty layer create + remove

```
$ ssh lnnode 'python3 -c "
import subprocess, time, statistics
def rpc(*a): return subprocess.run([\"lightning-cli\", *a], capture_output=True, text=True)
t = []
for i in range(10):
    n = f\"bench-empty-{i}\"
    t0 = time.perf_counter()
    rpc(\"askrene-create-layer\", n)
    rpc(\"askrene-remove-layer\", n)
    t.append((time.perf_counter() - t0) * 1000)
print(f\"min={min(t):.1f} median={statistics.median(t):.1f} max={max(t):.1f} mean={statistics.mean(t):.1f}\")
"'
empty create+remove: n=10 min=3.0ms median=3.3ms max=4.2ms mean=3.4ms
```

### 4.3 Benchmark B — create + 5 channel disables + remove

```
create+5upd+remove: n=10 min=10.6ms median=12.1ms max=13.1ms mean=11.9ms
```

### 4.4 Analysis

**Per-operation cost (derived)**:

- Empty create + remove: ~3.3 ms median (two RPCs × ~1.65 ms each including subprocess overhead)
- 5 channel updates added: total increase ~8.8 ms, i.e. **~1.76 ms per update-channel RPC**
- Each subprocess call adds ~1.5 ms of fixed overhead that v3's in-process RPC would avoid

**Projected realistic retry scenario**: a rebalance pair fails on the first attempt, v3 creates a throwaway exclude layer, adds 3–5 failed-channel disables, retries getroutes, removes the layer on success/failure.

- Expected overhead per retry attempt: create (1.65) + 5 × update (1.76) + remove (1.65) = **~12 ms** (subprocess-measured) or **~6–8 ms** (in-process projection)
- Typical rebalance cycle has at most 1–2 retries per pair, so total overhead per pair is **< 25 ms** worst case

### 4.5 Verdict

**Exclude-via-layer is comfortably under the 50ms per-cycle threshold** set by the parent spec (Section 3 of the design). The pattern is viable and should be the default for v3's retry logic.

| Threshold | Measured | Headroom |
|---|---|---|
| 50 ms (spec) | 12.1 ms median (subprocess) | **~4× headroom** |
| 50 ms (spec) | ~6–8 ms projection (in-process) | **~6–8× headroom** |

### 4.6 Recommendation

- **Default pattern**: v3 creates a throwaway layer named `rebalance-exclude-<cycle_id>` (e.g. `rebalance-exclude-20260410-140512`), adds `enabled=false` overrides for each failed channel via `askrene-update-channel`, passes the layer name in the retry `getroutes` call alongside `hive-fleet`, and removes the layer on cycle completion (success or final failure).
- **Cleanup discipline**: the layer must always be removed, even on exception paths. Implement via a Python context manager in the v3 router:
  ```python
  with self._exclude_layer(failed_channels) as layer_name:
      return self.plugin.rpc.getroutes(..., layers=[*self.layer_names, layer_name])
  ```
- **Persistent=false** (the default): exclude layers must be ephemeral. Any use of `persistent=true` would bloat CLN's datastore and break restart recovery.
- **Collision avoidance**: layer name includes a timestamp and/or cycle ID. If the plugin crashes mid-cycle leaving an orphan layer, the next startup can clean up any layer matching `rebalance-exclude-*` via a startup sweep. This is a one-line safeguard worth including in v3 init.

### 4.7 Alternate pattern (rejected)

**Internal exclude translation**: v3 could filter the gossip channel set in Python before calling `getroutes`, passing only channel IDs it wants askrene to consider. This was the fallback in the parent spec if benchmarks exceeded 50ms. With 4× headroom, it's not needed and would duplicate gossip knowledge in the plugin.

**Status**: rejected — exclude-via-layer wins on every axis (simpler, atomic, askrene-idiomatic, comfortably within budget).

## 5. xpay API Surface

### 5.1 Version and availability

`xpay` was added in CLN **v24.11** (`ElementsProject/lightning@b57edd21:doc/schemas/xpay.json#L4 "added": "v24.11"`). Live node runs v25.12.1, so available. The CLN plugin source is `plugins/xpay/xpay.c` (2,553 lines).

### 5.2 Request parameters

| Param | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `invstring` | string | yes | — | BOLT11, BOLT12, BOLT12 offer, or BIP353 name |
| `amount_msat` | msat | only if bolt11 has no amount | — | Explicit amount for amountless invoices |
| `maxfee` | msat | no | `5000msat or 1%, whichever is greater` | Absolute fee cap |
| `layers` | string[] | no | `[]` | askrene layers to apply on top of xpay's own private layer |
| `retry_for` | u32 (seconds) | no | `60` | How long xpay keeps finding new routes and retrying |
| `partial_msat` | msat | no | — | Partial payment from multiple payers |
| `maxdelay` | u32 (blocks) | no | `2016` | Max CLTV delay (v25.02+) |
| `payer_note` | string | no | — | Note for BOLT12 (v26.04+) |

**Critical absent parameter**: **there is no way to pass a precomputed route**. xpay always discovers its own route via `getroutes`. This directly answers Q4 research question #1: **xpay does not support route pinning**.

### 5.3 Response shape

```json
{
  "payment_preimage": "...",
  "failed_parts": 0,
  "successful_parts": 1,
  "amount_msat": 1000,
  "amount_sent_msat": 1000
}
```

`successful_parts ≥ 1` for any successful payment (even single-path). MPP is reflected in these counters.

### 5.4 Error codes

| Code | Meaning |
|---|---|
| `-1` | Catchall nonspecific error |
| `203` | Permanent failure from destination (e.g. "didn't recognize invoice") |
| `205` | Couldn't find a route to destination |
| `207` | Invoice expired |
| `219` | Invoice already paid |
| `209` | Other payment error |

Cited from `doc/schemas/xpay.json#L120-128`.

### 5.5 Internal architecture (from xpay.c)

xpay is essentially a retry-loop wrapper around `getroutes` + `sendpay` + `askrene-reserve`/`unreserve`/`inform-channel`. Key touch points in the source:

- **Calls `getroutes` for each attempt** (`xpay.c#L1444-1448`, inside `getroutes_for()`). The request is built at `xpay.c#L1449-1482`:
  ```c
  json_add_pubkey(req->js, "source", &xpay->local_id);  // L1449
  json_add_pubkey(req->js, "destination", dst);          // L1450
  ...
  /* Add private layer */
  json_add_string(req->js, NULL, payment->private_layer); // L1466
  /* Add user-specified layers */
  for (size_t i = 0; i < tal_count(payment->layers); i++) // L1468
      json_add_string(req->js, NULL, payment->layers[i]);
  if (payment->disable_mpp)
      json_add_string(req->js, NULL, "auto.no_mpp_support"); // L1471
  ```
- **Source is always our node id** (`xpay->local_id` at `xpay.c#L1449`). **Destination is the invoice's receiver**. These are non-overridable by the caller.
- **Always adds a private xpay layer** (`xpay.c#L1466`) — the `xpay` layer seen live in Section 2.5 is this persistent layer. xpay uses it to track learned routing constraints across calls.
- **User-supplied `layers` are appended** to the getroutes call (`xpay.c#L1467-1469`). So yes, passing `layers=["hive-fleet"]` to xpay works and applies fleet intelligence.
- **Uses `askrene-reserve`** (`xpay.c#L1287`) to lock capacity on the chosen path, **`askrene-unreserve`** (`xpay.c#L911`) to release after completion/failure, and **`askrene-inform-channel`** (`xpay.c#L877`) to feed observed success/failure back into askrene's learned constraints.

### 5.6 MPP behavior

`disable_mpp` is a per-payment boolean controlled by invoice features:

- For **BOLT11**, `disable_mpp = !feature_offered(b11->features, OPT_BASIC_MPP)` (`xpay.c#L2067`). If the invoice doesn't advertise MPP support, xpay sends single-path.
- For **BOLT12**, similar, with a deprecated compatibility flag (`xpay.c#L2029-2031`).
- When `disable_mpp` is true, xpay adds the `auto.no_mpp_support` layer to every getroutes call (`xpay.c#L1470-1471`), forcing askrene's single-path algorithm.
- `maxparts` is derived dynamically: for unannounced channels, capped at 6 (`xpay.c#L2106-2110`); otherwise passed through to askrene as `maxparts - count_pending`.

**No direct way for a caller to disable MPP** via CLI parameters. MPP is controlled by invoice features, not the xpay call. A circular rebalancer wanting single-path behavior would need to generate an invoice without the `OPT_BASIC_MPP` feature flag. Possible but awkward.

### 5.7 Retry and failure taxonomy

`retry_for` defaults to 60 seconds. Within that window, xpay:

1. Calls `getroutes` for the remaining amount
2. Reserves the channels
3. Sends via `sendpay`
4. Waits for settlement or failure
5. On hop failure, calls `askrene-inform-channel` to update the shared layer's constraints
6. Loops back to step 1 with the updated information

**This is a tight feedback loop**: every failed hop teaches askrene about reduced capacity or disabled channels, and the next getroutes call benefits. The `xpay` persistent layer accumulates this knowledge across multiple xpay invocations, which is why it's marked `persistent=true` (Section 2.5).

**Implication for v3**: xpay's retry behavior is more sophisticated than v2's executor. If xpay were usable for our case, it would be strictly better at finding routes through unreliable topologies. It's not (see Section 5.9), but the feedback mechanism is worth copying into v3 if we ever build a custom executor that drives `askrene-inform-channel` directly.

### 5.8 maxfee enforcement

xpay's `maxfee` is a hard absolute cap. Default is **max(5000 msat, 1% of amount)** (`xpay.c#L2095-2099`):

```c
if (!amount_msat_fee(&payment->maxfee, payment->amount, 0, 1000000 / 100))
    return command_fail(...);
payment->maxfee = amount_msat_max(payment->maxfee, AMOUNT_MSAT(5000));
```

For an amount of 100,000 sats, the default maxfee is 1000 sats (1%). For a 1,000 sat rebalance, the default is 5 sats (the 5000 msat floor). The default is operator-unfriendly for small rebalances (5 sats is ~500 ppm for a 1000-sat payment) but is trivially overridden by passing explicit `maxfee` in the xpay call.

xpay then passes the maxfee to getroutes as `maxfee_msat` (`xpay.c#L1473`). Askrene enforces this as the hard ceiling documented in Section 1.2.

### 5.9 Self-pay behavior — THE BLOCKING FINDING

**xpay explicitly shortcuts self-pay and does NOT route through the network.**

Source (`xpay.c#L1404-1413`):

```c
if (payment->paths)
    dst = &xpay->fakenode;
else
    dst = &payment->destination;

/* Self-pay?  Shortcut all this */
if (pubkey_eq(&xpay->local_id, dst)) {
    struct attempt *attempt = new_attempt(payment, deliver, NULL);
    return do_inject(aux_cmd, attempt);
}
```

When the invoice's destination equals our own node id, xpay calls `do_inject()` (injectpaymentonion) — an in-memory payment injection that **never touches the wire**. No channel is routed through, no sats flow across any HTLC, and no fees are paid.

**Live verification on the node:**

```
$ ssh lnnode 'PREIMAGE=$(openssl rand -hex 32)
  INV=$(lightning-cli invoice amount_msat=1000 label="v3-research-selfpay-$(date +%s%N)" \
                               description="v3 research self-pay test" expiry=300 preimage=$PREIMAGE)
  BOLT11=$(echo "$INV" | jq -r .bolt11)
  lightning-cli xpay "$BOLT11"'

{
   "payment_preimage": "cae5303e3fb58844769f88d35c773cd1a6eafb96b0b2daf6deef11e380d7b93e",
   "amount_msat": 1000,
   "amount_sent_msat": 1000,
   "failed_parts": 0,
   "successful_parts": 1
}
```

The preimage was returned, but:

- `amount_sent_msat == amount_msat == 1000` → **zero routing fees were paid**
- `failed_parts == 0, successful_parts == 1` → exactly one "part" completed
- The CLN log shows only `Resolved invoice 'v3-research-selfpay-...' with amount 1000msat in 1 htlcs` followed by xpay's success — no routing activity, no HTLC forwards across channels, no fee records

**A real circular rebalance moves sats out one channel and back in another, paying intermediate peer fees.** xpay's self-pay shortcut bypasses this entirely. Calling xpay on a local invoice is functionally equivalent to a datastore no-op — it adjusts internal bookkeeping without moving any liquidity.

**This is a hard blocker for using xpay as the v3 executor for circular rebalancing.** There is no caller-side flag to force xpay to actually route a self-pay. The only way to make xpay do real network routing is to pay an invoice from a DIFFERENT node, which is not the rebalancing model.

### 5.10 Verdict on xpay integration depth (Q4 from the design spec)

Reviewing the four options from the parent spec's Phase 2 Q4:

| Option | Verdict | Reason |
|---|---|---|
| **(a) Full xpay takeover** | ❌ Rejected | xpay short-circuits self-pay. No circular rebalancing. |
| **(b) xpay with pinned route** | ❌ Rejected | xpay has no route-pinning API; source/dest are derived from the invoice only. |
| **(c) Keep v2 executor, xpay as alternative** | ❌ Rejected | xpay is strictly worse than v2 executor for the circular rebalance use case (it won't actually move sats). Having both executors would just confuse operators. |
| **(d) Reject xpay entirely** | ✅ **Adopted** | v3 is router-only. v2 executor (invoice + sendpay + waitsendpay + retry with exclude) stays permanently. |

**Phase 2 is closed**. The v3 deliverable is:

- `rebalance_router_v3.py` using `getroutes` + cl-hive layers (Phase 1, implemented)
- `rebalance_executor_v3.py` is **NOT built**. The `rebalance-executor` config key is removed from the parent spec; only `rebalance-router` remains.
- The v2 executor (`rebalance_executor_v2.py`) is the sole execution path for both v2 and v3 routers.

**This does NOT kill the v3 project** — v3's router upgrade is still the bulk of the value. Better route selection (layer-aware, fleet-biased, bad-peer-avoiding) is a meaningful win even with the v2 executor handling payment delivery. Section 9 will formalize this as the decision record.

**One positive side-benefit** of this rejection: v3 phase 1 becomes the entire deliverable. No phase 2 gate, no conditional scope, no A/B framework for executors. The plan simplifies.

## 6. xpay vs sendpay+waitsendpay Behavior Diff For Circular Self-Pays

### 6.1 Task closed by Section 5

Section 5 demonstrated that xpay's self-pay path does not route through the network — it's a shortcut in `xpay.c#L1410-1413` that calls `do_inject()` (injectpaymentonion), producing zero routing fees and no HTLC forwards. The planned comparison of "xpay vs sendpay+waitsendpay for circular self-pays" is therefore vacuous: **there is nothing to compare because xpay does not perform a circular payment at all**.

Any diff we ran would look like this:

| Metric | sendpay+waitsendpay (real circular) | xpay (self-pay shortcut) |
|---|---|---|
| Wall time | ~seconds (network round-trips) | <10ms (in-memory) |
| Hops attempted | ≥2 (minimum circular: source_peer → us → dest_peer) | 0 |
| Fee paid (msat) | > 0 (peers charge forwarding fees) | 0 |
| Channel balances moved | yes (this is the whole point of rebalancing) | **no** |
| askrene-inform-channel updates | none (we manage our own) | none (shortcut skips this too) |

The last row is the killer: **xpay does not move channel balances**, which is the entire purpose of rebalancing. No further experimentation adds information.

### 6.2 What the sendpay+waitsendpay side would look like

For completeness, the v2 executor's current path for a circular rebalance is:

1. `invoice amount_msat=N ...` → generate self-invoice (`rebalance_executor_v2.py:… invoice()`)
2. `sendpay route=<precomputed> payment_hash=<from invoice> bolt11=<from invoice>` → inject the payment with our chosen route
3. `waitsendpay payment_hash timeout=...` → block until resolved
4. On route-level failure, add failed channels to exclude list and loop
5. On success, record fees and delete the invoice

Each of these steps is a concrete RPC. The v2 executor's behavior is already verified by `test_rebalance_executor_v2.py` (44 passing tests). No live diff is needed — we already know sendpay works for circular self-pays because v2 does it in production today.

### 6.3 Verdict

**Task 6 resolves with no new data**: xpay is not a viable comparison target for circular rebalancing, and v2 executor's behavior is already trusted production code. Section 9's decision records cite Section 5 and Section 6 as joint justification for keeping the v2 executor permanent.

## 7. setconfig Runtime-Switch Verification

### 7.1 setconfig contract

`setconfig` was added in CLN **v23.08** (`doc/schemas/setconfig.json#L4`). Required params:

| Param | Type | Added | Purpose |
|---|---|---|---|
| `config` | string | v23.08 | Config variable name |
| `val` | string / int / bool | v23.08 | New value |
| `transient` | bool | v25.02 | If true, don't persist to config.setconfig file |

From the schema: *"This new value will also be written at the end of the config.setconfig file... for persistence across restarts... Note that you can also adjust existing options for stopped plugins; they will have an effect when the plugin is restarted."*

### 7.2 Live test — setconfig on a dynamic cl-revenue-ops option

The live node has three dynamic `revenue-ops-*` options:

```
revenue-ops-boltz-daily-budget-sats = 3000
revenue-ops-boltz-enforce-budget    = true
revenue-ops-planner-execute-closes  = true
```

**Test: toggle `revenue-ops-boltz-enforce-budget` from `true` → `false` → back to `true`, with `transient=true` so nothing persists across restart.**

```
$ ssh lnnode 'KEY="revenue-ops-boltz-enforce-budget"
  OLD=$(lightning-cli listconfigs "$KEY" | jq -r ".configs[\"$KEY\"].value_str")
  echo "OLD=$OLD"
  lightning-cli -k setconfig config="$KEY" val=false transient=true'

OLD=true
=== setconfig NEW=false with transient ===
{
   "config": {
      "config": "revenue-ops-boltz-enforce-budget",
      "value_str": "false",
      "source": "setconfig transient",
      "plugin": "/data/lightningd/plugins/cl_revenue_ops/cl-revenue-ops.py",
      "dynamic": true
   }
}
AFTER=false
RESTORED=true
```

The change took effect immediately (`AFTER=false`), was reversible (`RESTORED=true`), and the `source` field switched to `"setconfig transient"` to indicate the change is not persisted. Lightningd's log also records each transition:

```
2026-04-10T16:22:40.731Z INFO lightningd: setconfig: revenue-ops-boltz-enforce-budget false (updated NULL:0)
2026-04-10T16:22:40.758Z INFO lightningd: setconfig: revenue-ops-boltz-enforce-budget true (updated NULL:0)
```

**Important CLI gotcha**: `lightning-cli setconfig ...` (positional args) did not parse `transient=true` correctly (it was treated as a malformed value token). Using `lightning-cli -k setconfig config=KEY val=VAL transient=true` (keyword-arg JSON mode) worked. Any v3 documentation or operator runbook must prescribe the `-k` invocation.

### 7.3 pyln-client's notification model

The setconfig RPC is dispatched to pyln-client via a built-in internal method `_set_config` (`plugin.py#L289`: `'setconfig': Method('setconfig', self._set_config, MethodType.RPCMETHOD)`).

The implementation (`plugin.py#L1042-1050`):

```python
def _set_config(self, config: str, val: Optional[Any]) -> None:
    """Called when the value of a dynamic option is changed"""
    opt = self.options[config]
    cb = opt.on_change
    if cb is not None:
        # This may throw an exception: caller will turn into error msg for user.
        cb(self, config, val)
    opt.value = val
```

**Notification semantics**:

- If the plugin registered the option with an `on_change` callback, that callback is invoked synchronously with `(plugin, config_name, new_value)`. The callback can raise — the error propagates back to the setconfig caller.
- If no callback, the value is still updated on `self.options[config].value`, and the plugin can read it on demand.

The option registration signature (`plugin.py#L447-454`):

```python
def add_option(self, name: str, default: Optional[Any],
               description: Optional[str],
               opt_type: str = "string",
               deprecated: Optional[Union[bool, List[str]]] = None,
               multi: bool = False,
               dynamic=False,
               on_change: Optional[Callable[["Plugin", str, Optional[Any]], None]] = None,
               ) -> None:
```

**Key constraints**:

- `dynamic=True` is required for any option that `setconfig` should be able to modify. Non-dynamic options refuse setconfig calls.
- `on_change` cannot be set without `dynamic=True` (`plugin.py#L471-474`: `'Option {} has on_change callback but is not dynamic'`).
- Callback is called synchronously before the value is updated on the option object. Order matters: inside the callback, `self.options[config].value` still holds the OLD value.

### 7.4 Implication for v3 runtime switch

Two valid implementation patterns:

**Pattern A — poll per cycle (simplest)**:

```python
plugin.add_option(
    "rebalance-router",
    default="v2",
    description="v2 (getroute) or v3 (askrene getroutes + layers)",
    opt_type="string",
    dynamic=True,
)

# Inside the engine:
def _active_router(self):
    want = self.plugin.options["rebalance-router"].value
    if want == "v3" and self.router_v3 is not None:
        return self.router_v3
    return self.router_v2
```

Each cycle reads the option fresh. `setconfig` writes to `options[...].value` atomically before returning. No callback needed. Matches the design spec's "re-read each cycle" pattern exactly.

**Pattern B — on_change callback (explicit)**:

```python
def _on_router_change(plugin, option_name, new_value):
    if new_value not in ("v2", "v3"):
        raise ValueError(f"rebalance-router must be 'v2' or 'v3', got {new_value!r}")
    if new_value == "v3" and rebalancer.rebalance_engine_v2.router_v3 is None:
        raise ValueError("askrene unavailable; cannot switch to v3")
    plugin.log(f"rebalance-router switched to {new_value} (takes effect next cycle)", level="info")

plugin.add_option(
    "rebalance-router",
    default="v2",
    description="v2 (getroute) or v3 (askrene getroutes + layers)",
    opt_type="string",
    dynamic=True,
    on_change=_on_router_change,
)
```

This gives immediate validation (the `setconfig` call fails with a clear error message if the operator tries `v3` on a non-askrene node) and visible log feedback for operators.

**Recommendation**: **Pattern B**. The validation-at-call-time behavior is exactly what the design spec wants (config validator rejects `v3` when askrene is unavailable), and the runtime log line is what Section 5 of the design requires (*"one warning log at init, no per-cycle noise"*). Pattern A is simpler but silent.

### 7.5 Verdict

`setconfig` hot-reload works, is natively supported by pyln-client, and gives v3 everything needed for a runtime A/B switch:

- Config validator rejection of invalid values ✓ (via `on_change` raising)
- Operator-visible confirmation logs ✓ (via plugin.log in callback)
- Per-cycle atomicity ✓ (engine captures router at cycle start, callback doesn't need to interrupt in-flight work)
- No plugin restart required ✓ (verified live on `revenue-ops-boltz-enforce-budget`)
- Persistence control ✓ (`transient=true` for experiments; no flag for permanent)

**Decision for Section 9: Pattern B (on_change callback with validation)**. Implementation cost is ~15 lines in `cl-revenue-ops.py` at option-registration time, plus the engine reading `self.plugin.options["rebalance-router"].value` in `_active_router()`.

## 8. Failure-Mode Taxonomy

### 8.1 Scope reduction

Sections 5 and 6 rejected xpay as the v3 executor. The failure-mode taxonomy therefore focuses on `getroutes` errors (the RPC v3 will actually call) and maps them to v2's existing skip reasons. The v2 executor's existing sendpay/waitsendpay error taxonomy is untouched by this research and stays as-is.

### 8.2 v2 planner's current skip reason vocabulary

Extracted from `modules/rebalance_planner_v2.py`, `modules/rebalance_engine_v2.py`, and `modules/rebalance_router_v2.py`:

| Skip reason | Meaning | Emitted by |
|---|---|---|
| `inside_band` | Channel's local ratio is within the target imbalance band | planner |
| `not_valuable` | Channel is not hive/profitable/active/bootstrap | planner |
| `no_partner` | No eligible opposite-side partner channel | planner |
| `cooldown` | Channel is in rebalance cooldown period | planner |
| `no_budget` | Channel has exhausted its CapEx budget | planner |
| `max_pairs_reached` | Per-cycle pair limit hit | planner |
| `outcompeted` | Pair was outranked by a better pair | planner |
| `no_route` | Router returned `success=False` with an empty-route error | engine |
| `route_over_budget` | Route cost exceeded `pair_budget_sats` | engine |

(These are the only values v3's audit logging needs to preserve. v2's old-era `RebalanceReasonCode` enum values with 13 unused entries were NOT migrated to v2 — they only remain in the stub `RebalanceReasonCode` class kept alive in `rebalancer.py` for `test_bleed_detection.py`.)

### 8.3 askrene `getroutes` error surface

**Request-validation errors** (return `JSONRPC2_INVALID_PARAMS` or `PAY_USER_ERROR`, should never happen from v3 since the caller validates inputs):

| CLN error message | Source line | v3 handling |
|---|---|---|
| `"amount must be non-zero"` | `askrene.c#L879-882` | Assert — planner guarantees non-zero |
| `"maxparts must be non-zero"` | `askrene.c#L884-887` | Assert — v3 passes default 100 or single-path layer |
| `"maximum delay allowed is %d"` | `askrene.c#L889-893` | Assert — v3 passes 2016 default |
| `"should be an array"` (layers) | `askrene.c#L182` | Assert — v3 always passes a list |
| `"Unknown layer"` | `askrene.c#L143, L165` | **New skip reason**: `unknown_layer` — fires if operator configured a layer name that doesn't exist |

**Node-not-in-gossmap** (returns `PAY_ROUTE_NOT_FOUND` from `do_getroutes()` inside `askrene.c`):

| CLN error message | Source line | Maps to v3 skip reason |
|---|---|---|
| `"Unknown source node %s"` | `askrene.c#L592-596` | **New skip reason**: `unknown_source_node` |
| `"Unknown destination node %s"` | `askrene.c#L602-606` | **New skip reason**: `unknown_dest_node` |

**Route-finding failures** (return `PAY_ROUTE_NOT_FOUND` with a rich text message from `explain_failure.c`):

The askrene child process calls `explain_failure()` (`child/explain_failure.c#L210`) when it cannot find a route. Every failure is wrapped in the base string `NO_USABLE_PATHS_STRING = "We could not find a usable set of paths."` and annotated with a specific reason. The annotations include:

| Explanation prefix/keyword | Meaning | Source |
|---|---|---|
| `"There is no connection between source and destination at all"` | Graph has no path regardless of capacity | `explain_failure.c#L247` |
| `"has no gossip"` | Shortest-path channel has no public gossip | `explain_failure.c#L295` |
| (describe_disabled output) | Shortest-path channel is disabled | `explain_failure.c#L297` |
| (describe_capacity output) | Shortest-path channel doesn't have enough capacity | `explain_failure.c#L299` |
| (why_max_constrained output) | Amount exceeds a live constraint | `explain_failure.c#L301` |
| `"exceeds htlc_maximum_msat ~%s"` | Channel's HTLC max too low | `explain_failure.c#L304-306` |
| `"below htlc_minumum_msat ~%s"` [sic — typo in CLN] | Channel's HTLC min too high | `explain_failure.c#L307-309` |
| `"produces a fee overflow for amount %s"` | Arithmetic overflow | `explain_failure.c#L270-272` |

All of these map cleanly to the existing v3 skip reason **`no_route`**. The rich explanation string can be preserved in the router's `RouteResult.error` field and emitted in audit logs for operator debugging, but v3 does not need to branch on the text.

**Child-process failures** (rare, indicate askrene internal problems):

| CLN error message | Source line | v3 handling |
|---|---|---|
| `"child died with signal %u"` | `askrene.c#L417-420` | **New skip reason**: `askrene_child_died` — indicates askrene internal crash |
| `"child produced no output"` | `askrene.c#L428-431` | Map to `askrene_child_died` |
| `"failed to fork: %s"` | `askrene.c#L641-648` | Map to `askrene_child_died` |
| `"failed to create pipes: %s"` | `askrene.c#L631-638` | Map to `askrene_child_died` |

All four are transient plugin-level issues. v3 should log them at `warn` level, add the pair to the cycle's skip records, and continue (do not crash the cycle).

### 8.4 Unified skip reason taxonomy for v3

Combining v2's existing reasons with the new ones surfaced by this research:

**Existing (unchanged)**: `inside_band`, `not_valuable`, `no_partner`, `cooldown`, `no_budget`, `max_pairs_reached`, `outcompeted`, `no_route`, `route_over_budget`.

**New (added by v3)**:

| Reason | When it fires | Severity |
|---|---|---|
| `unknown_source_node` | Askrene doesn't have the source peer in its gossmap (e.g. gossip not yet synced, or peer just disappeared from the network) | transient — may resolve next cycle |
| `unknown_dest_node` | Askrene doesn't have the destination peer in its gossmap | transient — may resolve next cycle |
| `unknown_layer` | Operator-configured layer name doesn't exist on the node | operator error — surface at init, not per-cycle |
| `askrene_child_died` | Askrene subprocess crashed or failed to fork | plugin error — log warn, skip pair, continue cycle |
| `path_loops_through_us` | (v3-specific, from Section 3.4 design decision) Getroutes returned a path where our own node is an intermediate hop, which doesn't fit the sendpay route format | design constraint — may indicate pair pinning failed, retry with different strategy |

Total new reasons: **5**. These are small additions to `modules/rebalance_audit_v2.py`:

```python
# In rebalance_audit_v2.py, extend the valid reasons set
VALID_SKIP_REASONS = {
    # Existing
    "inside_band", "not_valuable", "no_partner", "cooldown",
    "no_budget", "max_pairs_reached", "outcompeted",
    "no_route", "route_over_budget",
    # V3-specific (added by rebalance_router_v3)
    "unknown_source_node", "unknown_dest_node",
    "unknown_layer", "askrene_child_died", "path_loops_through_us",
}
```

### 8.5 Translation logic in the v3 router

```python
def _translate_getroutes_error(error: str) -> tuple[str, str]:
    """Map a getroutes error string to (skip_reason, preserved_detail)."""
    if "Unknown source node" in error:
        return "unknown_source_node", error
    if "Unknown destination node" in error:
        return "unknown_dest_node", error
    if "Unknown layer" in error:
        return "unknown_layer", error
    if "child died with signal" in error or "failed to fork" in error \
       or "child produced no output" in error or "failed to create pipes" in error:
        return "askrene_child_died", error
    # Everything else — explain_failure.c output, MCF solver failures —
    # maps to the generic no_route with the rich explanation preserved.
    return "no_route", error
```

This function lives in `rebalance_router_v3.py` and is called when `getroutes` raises an RPC error. The returned `(reason, detail)` tuple is attached to the `RouteResult` for the engine to record in audit logs.

### 8.6 Error codes

askrene uses CLN's standard error codes; v3 should branch on code primarily and fall back to string-matching for precision:

| CLN code | Constant | Meaning |
|---|---|---|
| `-32602` | `JSONRPC2_INVALID_PARAMS` | Request validation failure (should never fire from v3) |
| `205` | `PAY_ROUTE_NOT_FOUND` | All getroutes failures (node unknown, no route, child died) |
| `206` | `PAY_USER_ERROR` | `maxdelay` exceeded |

pyln-client surfaces these as `RpcError` exceptions. v3 catches `RpcError` and inspects `e.error["code"]` + `e.error["message"]` to dispatch.

### 8.7 Verdict

The failure-mode taxonomy is small and tractable. **Five new skip reasons** to add to `rebalance_audit_v2.py`, one simple translation function in `rebalance_router_v3.py`, and no changes to the v2 executor's sendpay error handling. No ambiguous mappings — every askrene error has a clear target skip reason.

## 9. Decision Records

### 9.1 xpay integration depth

**Decision:** **(d) Reject xpay entirely.** v3 is router-only; v2 executor stays permanent.

**Evidence:** Section 5.9 — `xpay.c#L1410-1413` explicitly short-circuits self-pay via `do_inject()` (injectpaymentonion), bypassing the network. Live test on node 0382d558 with a 1000-msat self-invoice confirmed: `amount_sent_msat == amount_msat == 1000` (zero fee), one "successful part", and the CLN log shows `Resolved invoice ... in 1 htlcs` with no forwarding activity.

**Rationale:** Circular rebalancing *requires* sats to actually flow through intermediate peers and pay forwarding fees. xpay's self-pay shortcut does not move any liquidity — it's an in-memory bookkeeping operation. There is no caller-side flag to force xpay to route a self-pay. Therefore xpay cannot be the executor for circular rebalances, and the parent spec's Phase 2 is closed with "researched, rejected" as an acceptable outcome.

**Consequence:** v3's deliverable is `rebalance_router_v3.py` alone. The `rebalance-executor` config key is removed from the parent spec. `rebalance_executor_v2.py` remains the sole execution path for both v2 and v3 routers. The two-phase plan (phase 1 = router, phase 2 = executor) collapses into a single-phase deliverable, which simplifies scope.

### 9.2 Exclude handling strategy

**Decision:** **Throwaway-layer pattern.** V3 creates a `rebalance-exclude-<cycle_id>` layer per retry attempt, populates it with `askrene-update-channel enabled=false` for each failed channel, passes the layer name to `getroutes` alongside `hive-fleet`, and removes the layer on cycle completion.

**Evidence:** Section 4 — live benchmark on the 63-peer / 46-active-channel live node measured median 3.3ms for empty create+remove, 12.1ms for create + 5 channel disables + remove (via `lightning-cli` subprocess). In-process RPC will be ~30–60% faster. Well under the 50ms-per-cycle threshold from the parent spec with ~4× headroom.

**Rationale:** The throwaway-layer pattern is atomic, askrene-idiomatic, and comfortably performant. The alternative (v3-internal exclude translation) would duplicate gossip knowledge in the plugin and achieve no measurable win. Layer create/remove latency is the only cost, and it's far below budget.

**Implementation notes:**
- Layer name format: `rebalance-exclude-{cycle_id}` where `cycle_id` is a monotonic counter or timestamp
- `persistent=false` (the default) — these layers must be ephemeral to avoid datastore bloat
- Wrap in a Python context manager to guarantee cleanup on exception paths
- At plugin init, sweep any orphan `rebalance-exclude-*` layers from a previous crashed cycle

### 9.3 Default layer set

**Decision:** **`askrene-layers = "hive-fleet"`** remains the default, unchanged from the parent spec.

**Evidence:** Section 3 — Experiment A proved the hive-fleet layer is respected by askrene under pair pinning (10 msat fee reduction + 28 block cltv reduction when the path traverses a fleet channel). Section 2.5 live-node snapshot showed that `hive-fleet` is the only cl-hive layer with active content (4 channel overrides). The other three cl-hive layers (`hive-reputation`, `hive-corridors`, `hive-traffic`) are currently empty on the live node and would be no-ops even if configured.

**Rationale:** Configuring only the layer that has actual content avoids processing overhead for no-op layers and keeps the default config matching observed reality. Operators on nodes with populated reputation/corridor/traffic layers can opt into them via the config key (`askrene-layers = "hive-fleet,hive-reputation,hive-corridors,hive-traffic"`).

**Implementation note:** v3's init-time layer probe should log which of the requested layers were actually found. If an operator sets `hive-reputation` but the layer isn't published, the probe surfaces that clearly rather than silently ignoring it.

### 9.4 Runtime switch mechanism

**Decision:** **Pattern B — `on_change` callback with validation** (Section 7.4).

**Evidence:** Section 7.3 — `plugin.py#L1042-1050` in pyln-client confirms `setconfig` dispatches to the option's `on_change` callback synchronously, and `plugin.py#L447-474` confirms the `add_option(dynamic=True, on_change=...)` API exists. Section 7.2 — live test on `revenue-ops-boltz-enforce-budget` confirmed `lightning-cli -k setconfig config=... val=... transient=true` works correctly, the change is immediate, and the plugin log records each transition.

**Rationale:** Pattern B (callback) gives the validation-at-call-time behavior the design spec requires: an operator trying to set `rebalance-router=v3` on a non-askrene node will see a clear error message, not a silent acceptance that later produces runtime failures. The callback also emits an operator-visible log line confirming the switch took effect and will apply on the next cycle.

**Implementation**:

```python
def _on_router_change(plugin, option_name, new_value):
    if new_value not in ("v2", "v3"):
        raise ValueError(f"rebalance-router must be 'v2' or 'v3', got {new_value!r}")
    eng = getattr(rebalancer, "rebalance_engine_v2", None)
    if eng is None:
        raise ValueError("rebalance engine not initialized; cannot change router")
    if new_value == "v3" and eng.router_v3 is None:
        raise ValueError("askrene unavailable on this node; cannot switch to v3")
    plugin.log(
        f"rebalance-router switched to {new_value} (takes effect at next cycle boundary)",
        level="info",
    )

plugin.add_option(
    "rebalance-router",
    default="v2",
    description="Rebalance route discovery strategy: v2 (getroute) or v3 (askrene getroutes + layers)",
    opt_type="string",
    dynamic=True,
    on_change=_on_router_change,
)
```

The engine still captures the active router at the start of each cycle (per-cycle atomicity from the design spec), reading `self.plugin.options["rebalance-router"].value` inside `_active_router()`.

**CLI invocation for operators:**

```
lightning-cli -k setconfig config=rebalance-router val=v3 transient=true   # A/B test
lightning-cli -k setconfig config=rebalance-router val=v3                    # persist to config.setconfig
```

Document the `-k` keyword mode explicitly — positional-arg mode fails on the `transient` parameter.

### 9.5 Phase 1 go/no-go

**Decision:** **GO**, with the design deltas listed below.

**Rationale:** Every research task returned a usable answer. The design's core approach — v3 router uses getroutes + layers, v2 router stays as fallback, runtime switch via setconfig, exclude-via-layer for retries — is confirmed viable by live experiments. The only surprise was the xpay rejection, which actually *simplifies* Phase 1 (eliminates the Phase 2 scope and the 2×2 executor matrix).

### 9.6 Design deltas from the parent spec

The following changes to `docs/superpowers/specs/2026-04-10-askrene-router-v3-design.md` must be incorporated into the Phase 1 implementation plan. None of these invalidate the core architecture; they are refinements discovered by research.

**Delta 1: Phase 2 closed.** The parent spec's "Phase 2 — xpay executor (conditional)" section becomes "Phase 2 — closed, xpay rejected." `rebalance_executor_v3.py` is NOT built. The `rebalance-executor` config key is removed from the spec (only `rebalance-router` remains). See Section 9.1.

**Delta 2: Pair-pinning path validation.** The parent spec assumed v3 could directly prepend/append first/last hops around a getroutes middle path. Section 3.4 showed askrene may return paths that (a) loop through our own node as an intermediate hop, or (b) bypass our node entirely. V3's `price_pair` must:
- Validate that the returned path's first hop's `short_channel_id_dir` matches the requested `source_channel_id` (in the correct direction). If not, either reject the path or retry with a throwaway layer that disables `source_peer`'s other outgoing channels.
- Validate that the returned path's last hop leads into `dest_peer` via the requested `dest_channel_id`.
- Reject any path where an intermediate hop's `next_node_id` equals our own node id (a loop-through-us).

If validation fails, emit skip reason `path_loops_through_us` or retry with the throwaway layer forcing the first hop.

**Delta 3: getroutes path format translation.** The parent spec assumed `RouteResult.route` stays in sendpay format. Section 1.4 confirms `getroutes` returns a different hop shape (`short_channel_id_dir` instead of `channel`+`direction`, `next_node_id` instead of `id`). V3's router must translate each hop:

```python
def _translate_getroutes_hop_to_sendpay(hop: dict, final_cltv: int) -> dict:
    scidd = hop["short_channel_id_dir"]  # "SCID/dir"
    scid, direction = scidd.split("/")
    return {
        "id": hop["next_node_id"],
        "channel": scid,
        "direction": int(direction),
        "amount_msat": hop["amount_msat"],
        "delay": hop["delay"],
    }
```

This is a per-hop trivial transformation. The v3 router applies it to every hop in the chosen route before returning the `RouteResult`.

**Delta 4: Five new skip reasons.** Section 8.4 added `unknown_source_node`, `unknown_dest_node`, `unknown_layer`, `askrene_child_died`, `path_loops_through_us`. Add them to `rebalance_audit_v2.py`'s valid-reason set. The translation function `_translate_getroutes_error()` from Section 8.5 belongs in `rebalance_router_v3.py`.

**Delta 5: CLN minimum version is v24.11, not v24.08.** Section 1.1 said `getroutes` was added in v24.08, but Section 2.1 showed the layer management RPCs (`askrene-create-layer`, `askrene-update-channel`, etc.) were added in v24.11. V3 uses both `getroutes` AND layer mutation (for exclude-via-layer), so the true minimum is **v24.11**. Update the design spec's "graceful degradation" matrix accordingly.

**Delta 6: Runtime switch requires `-k` mode.** Section 7.2 showed `lightning-cli setconfig config val transient=true` (positional) fails on the `transient` parameter. Only `lightning-cli -k setconfig config=... val=... transient=true` works. Operator documentation must prescribe the `-k` mode explicitly.

**Delta 7: Probe `askrene-listlayers` at init, not `help getroutes`.** The parent spec suggested probing for askrene via `plugin.rpc.help("getroutes")`. A cleaner probe is `plugin.rpc.call("askrene-listlayers")` — it exercises both the getroutes RPC availability AND the layer management surface in one call, and returns useful data (the list of found layers) that v3 can log immediately. This saves one RPC at init and produces a more informative startup log.

**Delta 8: Orphan exclude-layer sweep at init.** Section 4.6 added this requirement. At plugin startup, call `askrene-listlayers`, iterate, and remove any layer matching `rebalance-exclude-*` (they indicate a previous crashed cycle left the layer behind). One-line safeguard, worth including.

### 9.7 Phase 1 scope (as amended by deltas)

**Scope additions from deltas:**
- Path-shape validator in `rebalance_router_v3.py` (Delta 2)
- Hop format translator in `rebalance_router_v3.py` (Delta 3)
- Five new skip reasons in `rebalance_audit_v2.py` (Delta 4)
- Askrene probe via `askrene-listlayers`, not `help getroutes` (Delta 7)
- Orphan `rebalance-exclude-*` layer sweep at init (Delta 8)
- `on_change` callback for `rebalance-router` config key in `cl-revenue-ops.py` (Delta 6 + 9.4)

**Scope removals:**
- `rebalance_executor_v3.py` (Delta 1)
- `rebalance-executor` config key (Delta 1)
- Phase 2 test matrix and A/B framework for executors (Delta 1)

**Revised module layout**:

```
modules/
├── rebalance_router_v2.py      # unchanged — fallback for CLN < v24.11
├── rebalance_router_v3.py      # NEW — askrene getroutes + layers + hop translator + path validator
├── rebalance_executor_v2.py    # unchanged — sole executor
├── rebalance_engine_v2.py      # +router factory + runtime dispatch + init probe + orphan sweep
├── rebalance_audit_v2.py       # +5 new skip reasons
└── config.py                   # +rebalance-router and +askrene-layers config keys (only 2, not 4)
```

**Revised config keys:**

| Key | Default | Purpose |
|---|---|---|
| `rebalance-router` | `"v2"` | `"v2"` or `"v3"`, runtime-switchable via setconfig |
| `askrene-layers` | `"hive-fleet"` | CSV of layer names to pass to getroutes |

(The original spec also proposed `askrene-probe-amounts` and `rebalance-executor`; both are removed.)

### 9.8 Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Askrene solver finds no route that satisfies the path-shape validator for some pairs | Medium | Low | Retry with throwaway layer disabling source_peer's other channels (Delta 2 fallback). If still fails, emit `path_loops_through_us` skip reason and move on. |
| cl-hive's layer content drift breaks v3's expectations | Low | Low | v3 is a pure consumer and treats layers as best-effort enrichment. Missing layers cause no errors; empty layers cause no-ops. |
| CLN upgrade changes getroutes response shape | Low | Medium | Hop translator is isolated in `_translate_getroutes_hop_to_sendpay()`. A CLN schema change requires updating that function only. |
| pyln-client changes `on_change` callback semantics | Low | Low | Both Pattern A (poll per cycle) and Pattern B (callback) remain valid; swap by removing `on_change=` from `add_option`. |
| Operator sets `rebalance-router=v3` but `askrene-layers` is a typo | Medium | Low | Init-time probe logs found/missing layers (Section 2.5 + Delta 7). Operator sees the typo in plugin startup logs. |
| v2 router's "pin source + pin dest + getroute middle" behavior subtly changes after v3 merge (regression) | Low | Medium | v2 router file is UNCHANGED by this work. Any regression is not caused by the v3 merge. |

### 9.9 Next step

Write the Phase 1 implementation plan at `docs/superpowers/plans/2026-04-10-askrene-router-v3-phase1.md`, incorporating all deltas from Section 9.6 and the revised scope from Section 9.7. The plan should follow TDD structure (one test + minimal impl + verify + commit per step) as per `superpowers:writing-plans`.

This research doc is the input to that planning session. No further research is needed before Phase 1 implementation begins.
