# Askrene Router V3 — Phase 0 Research Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md` with nine evidence-backed sections that unblock the Phase 1 implementation plan for the askrene-based rebalance router v3.

**Architecture:** Evidence-first research. Every behavioral claim has a receipt: either a CLN upstream source citation (`ElementsProject/lightning@<sha>:<path>#L<start>-L<end>`) or a captured live-node RPC transcript. No guessing, no memory, no hallucinated field names. The research doc is the sole deliverable — no v3 router code is written in this plan.

**Tech Stack:**
- `gh api repos/ElementsProject/lightning/contents/<path>` for upstream source reads
- `lightning-cli` (or `plugin.rpc` if running inside the plugin) for live-node experiments
- Plain markdown for the research doc
- Git for incremental commits (one per research section)

---

## Reference Spec

This plan implements Phase 0 of the design at:

`docs/superpowers/specs/2026-04-10-askrene-router-v3-design.md` (commit `f352dbf` on branch `feature/askrene-router-v3`)

Specifically the "Research Phase & Reference Sources" section, which enumerates the nine deliverable sections and the CLN upstream files to cite.

## Parallelism Notes

The following research tasks have no dependencies on each other and can be dispatched to parallel subagents if desired:

- Task 1 (getroutes contract)
- Task 2 (layer lifecycle)
- Task 5 (xpay API surface)
- Task 7 (setconfig runtime-switch)

These must run sequentially after their dependencies:

- Task 3 (layer semantics under pair pinning) ← depends on Task 1 + Task 2
- Task 4 (exclude-via-layer pattern) ← depends on Task 2
- Task 6 (xpay vs sendpay diff) ← depends on Task 5
- Task 8 (failure-mode taxonomy) ← depends on Task 1 + Task 5
- Task 9 (decision records) ← depends on Tasks 1–8

If executing serially, run in the order listed below.

## Environment Expectations

Every task assumes:

- Worktree: `/home/sat/bin/cl_revenue_ops/.worktrees/askrene-router-v3-20260410`
- Branch: `feature/askrene-router-v3`
- Live CLN node accessible via `~/.lightning/bitcoin/lightning-rpc` (the sat's own node)
- `gh` CLI authenticated to github.com for `gh api` calls against `ElementsProject/lightning`
- CLN version 24.11+ for every experiment (fall back to documenting "version gated" explicitly if the sat's node is older)

At the start of every task that touches the live node: capture `lightning-cli --version` and `lightning-cli getinfo | jq '{version, network, id}'` in the research doc's environment section for reproducibility.

---

## Task 0: Scaffold The Research Doc

**Files:**
- Create: `docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md`

- [ ] **Step 1: Capture CLN upstream master SHA for citations**

Run:

```bash
gh api repos/ElementsProject/lightning/commits/master --jq '.sha' > /tmp/cln-master-sha.txt
cat /tmp/cln-master-sha.txt
```

Expected: a 40-character commit SHA. Write it down.

- [ ] **Step 2: Capture live-node environment metadata**

Run:

```bash
lightning-cli --version
lightning-cli getinfo | python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps({"version": d.get("version"), "network": d.get("network"), "id": d.get("id"), "blockheight": d.get("blockheight")}, indent=2))'
```

Save the output verbatim — it will be pasted into the research doc's environment section.

- [ ] **Step 3: Write the research doc scaffold**

Create `docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md` with exactly this content:

````markdown
# Rebalance Router V3 — Research Findings

**Date:** 2026-04-10
**Status:** In progress
**Parent spec:** `docs/superpowers/specs/2026-04-10-askrene-router-v3-design.md`
**Worktree:** `.worktrees/askrene-router-v3-20260410`
**Branch:** `feature/askrene-router-v3`

## Environment

**CLN upstream reference SHA:** `<paste SHA from Step 1>`

**Live node:**

```json
<paste lightning-cli getinfo redaction from Step 2>
```

**Citation format:** `ElementsProject/lightning@<sha>:<path>#L<start>-L<end>`

---

## 1. getroutes Contract

_TODO: Task 1_

## 2. Layer Lifecycle

_TODO: Task 2_

## 3. Layer Semantics Under Pair Pinning

_TODO: Task 3_

## 4. Exclude-Via-Layer Pattern

_TODO: Task 4_

## 5. xpay API Surface

_TODO: Task 5_

## 6. xpay vs sendpay+waitsendpay Behavior Diff For Circular Self-Pays

_TODO: Task 6_

## 7. setconfig Runtime-Switch Verification

_TODO: Task 7_

## 8. Failure-Mode Taxonomy

_TODO: Task 8_

## 9. Decision Records

_TODO: Task 9_
````

Replace the two `<paste ...>` markers with the actual values captured in steps 1 and 2.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md
git commit -m "docs(research): scaffold v3 router research doc"
```

Note: the `_TODO: Task N` markers are intentional scaffolding that is replaced in later tasks. The final self-review in Task 10 verifies all of them are gone.

---

## Task 1: Document The getroutes Contract

**Files:**
- Modify: `docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md` (Section 1)

- [ ] **Step 1: Read the getroutes schema from CLN upstream**

Run:

```bash
gh api repos/ElementsProject/lightning/contents/doc/schemas/getroutes.json --jq '.content' | base64 -d > /tmp/getroutes-schema.json
python3 -m json.tool /tmp/getroutes-schema.json | head -120
```

Expected: a valid JSON schema document describing the `getroutes` RPC method.

- [ ] **Step 2: Read the askrene plugin implementation for error paths**

Run:

```bash
gh api repos/ElementsProject/lightning/contents/plugins/askrene --jq '.[].name'
```

Expected: list of files in `plugins/askrene/`. Identify files likely to contain the getroutes entry point (e.g. `askrene.c`, `flow.c`).

Then fetch each candidate and grep for `"getroutes"`:

```bash
for f in askrene.c flow.c mcf.c reserve.c layer.c; do
  gh api "repos/ElementsProject/lightning/contents/plugins/askrene/$f" --jq '.content' 2>/dev/null | base64 -d > "/tmp/askrene-$f"
done
grep -n '"getroutes"' /tmp/askrene-*.c
```

Record the file and line number of the getroutes command registration.

- [ ] **Step 3: Verify getroutes against the live node**

Run a minimum-viable call to confirm the local CLN supports it and see the real shape:

```bash
lightning-cli help getroutes
```

Then run a tiny exploratory call (pick any two well-known peer IDs from `listnodes`):

```bash
PEER_A=$(lightning-cli listnodes | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["nodes"][0]["nodeid"])')
PEER_B=$(lightning-cli listnodes | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["nodes"][1]["nodeid"])')
lightning-cli getroutes source=$PEER_A destination=$PEER_B amount_msat=1000000 layers=[] maxfee_msat=100000 final_cltv=40 2>&1 | head -80
```

If the call returns an error like "unknown method", note it — the node is not on CLN 24.11+ and the research section must document the version gate.

- [ ] **Step 4: Write Section 1 of the research doc**

Replace the `## 1. getroutes Contract` section's `_TODO: Task 1_` placeholder with content covering:

1. **Request parameters.** Table of every parameter: name, type, required/optional, meaning. Each row cites the schema file with line numbers.
2. **Response shape.** Top-level keys, shape of `routes` array, shape of each hop, units (msat vs sat). Citations from schema.
3. **Error modes.** Every error the plugin can return (from the .c grep). Each with a citation.
4. **Timeout behavior.** Default timeout, how to override, what happens on timeout.
5. **Version gate.** Confirmed minimum CLN version supporting this call. Live-node version from environment section.

Every factual claim in this section gets a citation in the form `ElementsProject/lightning@<sha>:doc/schemas/getroutes.json#L<start>-L<end>` or `ElementsProject/lightning@<sha>:plugins/askrene/<file>#L<start>-L<end>`. Live-node transcripts are inlined as fenced code blocks.

- [ ] **Step 5: Sanity check — every claim has a citation**

Run:

```bash
grep -c "ElementsProject/lightning@" docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md
```

Expected: at least 5 citations in Section 1 alone. If fewer, go back and cite each claim.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md
git commit -m "docs(research): section 1 — getroutes contract"
```

---

## Task 2: Document Layer Lifecycle

**Files:**
- Modify: `docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md` (Section 2)

- [ ] **Step 1: Read the five layer RPC schemas**

Run:

```bash
for f in askrene-listlayers askrene-create-layer askrene-remove-layer askrene-update-channel askrene-inform-channel askrene-bias-channel askrene-bias-node; do
  gh api "repos/ElementsProject/lightning/contents/doc/schemas/$f.json" --jq '.content' 2>/dev/null | base64 -d > "/tmp/$f.json"
  echo "--- $f.json ---"
  python3 -m json.tool "/tmp/$f.json" 2>/dev/null | head -40
done
```

Expected: seven JSON schema documents. Record request/response shape and parameter meanings.

- [ ] **Step 2: Read the layer implementation for persistence and concurrency**

Run:

```bash
gh api "repos/ElementsProject/lightning/contents/plugins/askrene/layer.c" --jq '.content' | base64 -d > /tmp/askrene-layer.c
wc -l /tmp/askrene-layer.c
grep -n 'persist\|disk\|save\|load\|restart' /tmp/askrene-layer.c
grep -n 'mutex\|lock\|concurrent' /tmp/askrene-layer.c
```

Record any findings about persistence and concurrency.

- [ ] **Step 3: Verify layers against the live node**

Run:

```bash
lightning-cli askrene-listlayers
```

Expected: a JSON response listing all current layers (cl-hive's layers should appear if cl-hive is running on this node). Capture the output verbatim.

Then test read-only access from cl-revenue-ops's perspective by listing a single layer:

```bash
lightning-cli askrene-listlayers hive-fleet 2>&1 | head -40
```

- [ ] **Step 4: Test safe concurrent-read semantics experimentally**

Create a throwaway layer from cl-revenue-ops's process namespace, inspect it, delete it:

```bash
lightning-cli askrene-create-layer cl-revenue-ops-research-probe
lightning-cli askrene-listlayers cl-revenue-ops-research-probe
lightning-cli askrene-remove-layer cl-revenue-ops-research-probe
```

Confirm nothing in cl-hive's layers was affected:

```bash
lightning-cli askrene-listlayers hive-fleet | python3 -c 'import sys,json; d=json.load(sys.stdin); print("channels:", len(d.get("layers",[{}])[0].get("channel_updates",[])))'
```

Record the channel counts before and after. Any change = a concurrency red flag.

- [ ] **Step 5: Write Section 2 of the research doc**

Replace `_TODO: Task 2_` with content covering:

1. **Lifecycle API.** Each of the seven RPC methods: purpose, request params, response, citations.
2. **Ownership model.** Is a layer scoped to a plugin, a user, or globally? Can plugin A write to plugin B's layer? Cited from source.
3. **Persistence.** Are layers saved across `lightningd` restarts? (From source grep in Step 2.) If not, document that cl-hive re-publishes them on every plugin startup.
4. **Multi-plugin concurrency.** Can two plugins read the same layer simultaneously? Write simultaneously? Cited from source grep in Step 2 and verified by the experiment in Step 4.
5. **Safety claim.** Explicit statement: "cl-revenue-ops reading cl-hive's layers is safe because [cited reason]." If the source doesn't support the claim, document what extra safety measure v3 must take.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md
git commit -m "docs(research): section 2 — layer lifecycle"
```

---

## Task 3: Layer Semantics Under Pair Pinning (Live Experiment)

**Files:**
- Modify: `docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md` (Section 3)

**Depends on:** Tasks 1 and 2 must be complete.

- [ ] **Step 1: Pick a pair of peers that share a middle path**

From the sat's node, identify two peers `A` and `B` that:
- Are not directly connected
- Have a known routing path between them
- At least one hop on that path is also a hive-fleet channel (visible in `askrene-listlayers hive-fleet`)

Use this exploration:

```bash
lightning-cli listpeerchannels | python3 -c '
import sys, json
d = json.load(sys.stdin)
for ch in d["channels"]:
    peer = ch.get("peer_id", "")[:16]
    scid = ch.get("short_channel_id", "?")
    state = ch.get("state", "?")
    print(f"{peer}... {scid} {state}")
' | head -20
```

Pick two peers manually and record their pubkeys and SCIDs.

- [ ] **Step 2: Get an unbiased route (no layers)**

Run:

```bash
PEER_A=<paste from Step 1>
PEER_B=<paste from Step 1>
lightning-cli getroutes source=$PEER_A destination=$PEER_B amount_msat=1000000 layers=[] maxfee_msat=100000 final_cltv=40 > /tmp/route-no-layers.json
python3 -m json.tool /tmp/route-no-layers.json
```

Capture the full response.

- [ ] **Step 3: Get a fleet-biased route (hive-fleet layer)**

Run:

```bash
lightning-cli getroutes source=$PEER_A destination=$PEER_B amount_msat=1000000 layers='["hive-fleet"]' maxfee_msat=100000 final_cltv=40 > /tmp/route-with-fleet.json
python3 -m json.tool /tmp/route-with-fleet.json
```

Capture the full response.

- [ ] **Step 4: Diff the two routes**

Run:

```bash
diff /tmp/route-no-layers.json /tmp/route-with-fleet.json
```

Record whether the two routes are identical or different. Two possible outcomes:

- **Identical**: the fleet layer had no observable effect on this pair. This is a *negative result* and must be documented — try a second pair before concluding.
- **Different**: the fleet layer influenced middle-hop selection. Record exactly which hops changed.

If the first pair returns identical routes, repeat Steps 1-4 with a different pair until either (a) you observe a layer-driven difference, or (b) you've tried three pairs and can conclude "fleet layer had no observable effect on this node's current topology."

- [ ] **Step 5: Write Section 3 of the research doc**

Replace `_TODO: Task 3_` with content covering:

1. **Experimental setup.** Which peers were tested, which pair ultimately showed the effect (if any), SCIDs, amounts.
2. **Full RPC transcripts.** Both the unlayered and fleet-layered `getroutes` responses as fenced code blocks.
3. **Diff analysis.** Which hops changed and why (if the difference is attributable to a specific fleet channel).
4. **Conclusion.** One of:
   - "Askrene respects the `hive-fleet` layer's channel constraints under pair pinning." (with evidence)
   - "Askrene did not observably change routes when the `hive-fleet` layer was applied in the tested pairs." (with negative evidence and implications for v3 design)

If the conclusion is negative, flag it as a blocker for Phase 1 and surface it explicitly in Task 9 (Decision Records).

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md
git commit -m "docs(research): section 3 — layer semantics under pair pinning"
```

---

## Task 4: Exclude-Via-Layer Pattern Cost Measurement

**Files:**
- Modify: `docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md` (Section 4)

**Depends on:** Task 2 must be complete.

- [ ] **Step 1: Benchmark create+remove-layer latency**

Run 10 create/remove cycles and measure wall time:

```bash
python3 - <<'PY'
import subprocess, time, statistics

def rpc(*args):
    r = subprocess.run(["lightning-cli", *args], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr

times = []
for i in range(10):
    name = f"cl-revenue-ops-bench-{i}"
    t0 = time.perf_counter()
    rpc("askrene-create-layer", name)
    rpc("askrene-remove-layer", name)
    times.append((time.perf_counter() - t0) * 1000)  # ms

print(f"create+remove (empty layer): min={min(times):.1f}ms median={statistics.median(times):.1f}ms max={max(times):.1f}ms")
PY
```

Record the result.

- [ ] **Step 2: Benchmark create + 5 channel updates + remove**

Run:

```bash
python3 - <<'PY'
import subprocess, time, statistics, json

def rpc(*args):
    r = subprocess.run(["lightning-cli", *args], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr

# Get 5 real SCIDs for realistic update calls
r = subprocess.run(["lightning-cli", "listchannels"], capture_output=True, text=True)
channels = json.loads(r.stdout)["channels"][:5]
scid_dirs = [f'{c["short_channel_id"]}/{c["direction"]}' for c in channels]

times = []
for i in range(10):
    name = f"cl-revenue-ops-bench-{i}"
    t0 = time.perf_counter()
    rpc("askrene-create-layer", name)
    for sd in scid_dirs:
        rpc("askrene-update-channel", f"layer={name}", f"short_channel_id_dir={sd}", "enabled=false")
    rpc("askrene-remove-layer", name)
    times.append((time.perf_counter() - t0) * 1000)

print(f"create+5updates+remove: min={min(times):.1f}ms median={statistics.median(times):.1f}ms max={max(times):.1f}ms")
PY
```

Record the result.

- [ ] **Step 3: Compare against the 50ms spec threshold**

The parent spec's Section 3 says "If research measures layer create/remove cost >50ms per cycle, fall back to v3-internal exclude translation."

Compare the median result from Step 2 against 50ms. Record a clear pass/fail:

- **≤50ms median**: exclude-via-layer is the preferred pattern for v3
- **>50ms median**: exclude-via-layer is rejected; v3 must translate excludes internally before calling getroutes

- [ ] **Step 4: Write Section 4 of the research doc**

Replace `_TODO: Task 4_` with content covering:

1. **Methodology.** Exactly how the benchmarks were run, sample size, hardware notes.
2. **Results.** The two benchmark outputs verbatim.
3. **Conclusion.** Pass/fail against the 50ms threshold with a one-sentence recommendation for v3's exclude strategy.
4. **Alternative pattern (if rejected).** Brief description of how v3-internal exclude translation would work: filter channels from `listchannels` output before calling getroutes, or use `getroutes`'s built-in exclude mechanism if one exists (cite from Task 1 findings).

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md
git commit -m "docs(research): section 4 — exclude-via-layer cost measurement"
```

---

## Task 5: Document The xpay API Surface

**Files:**
- Modify: `docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md` (Section 5)

- [ ] **Step 1: Read the xpay schema**

Run:

```bash
gh api repos/ElementsProject/lightning/contents/doc/schemas/xpay.json --jq '.content' | base64 -d > /tmp/xpay-schema.json
python3 -m json.tool /tmp/xpay-schema.json | head -200
```

Record every request parameter and response field.

- [ ] **Step 2: Read the xpay plugin implementation**

Run:

```bash
gh api repos/ElementsProject/lightning/contents/plugins/xpay.c --jq '.content' | base64 -d > /tmp/xpay.c
wc -l /tmp/xpay.c
```

Grep for the key questions (each grep is a step in building understanding):

```bash
grep -n 'route\|routehint\|precompute' /tmp/xpay.c | head -20
grep -n 'layer\|askrene\|getroutes' /tmp/xpay.c | head -20
grep -n 'retry\|exclude\|failed' /tmp/xpay.c | head -20
grep -n 'mpp\|multipart\|split\|partid' /tmp/xpay.c | head -20
grep -n 'maxfee\|maxdelay\|max_fee' /tmp/xpay.c | head -20
grep -n 'bolt11\|invoice\|self_pay' /tmp/xpay.c | head -20
```

For each grep, follow up by reading the surrounding context in `/tmp/xpay.c` (e.g. `sed -n '100,150p' /tmp/xpay.c`) and recording source line refs.

- [ ] **Step 3: Verify xpay exists on the live node**

Run:

```bash
lightning-cli help xpay 2>&1 | head -40
```

If it returns "unknown command", the sat's CLN is too old. Record that fact — it changes the urgency of Phase 2 (no point planning for xpay if the node can't run it).

- [ ] **Step 4: Write Section 5 of the research doc**

Replace `_TODO: Task 5_` with content covering, one subsection per question from the parent spec:

1. **Route-pinning capability.** Can xpay accept a precomputed route? Cited answer with source lines.
2. **Layer support.** Does xpay automatically use askrene layers? Or must they be passed explicitly? Cited.
3. **Retry semantics.** Does xpay retry internally with excludes on hop failure? What failure modes are retryable vs. terminal? Cited failure taxonomy.
4. **MPP behavior.** Does xpay auto-split? Is it configurable? Can MPP be disabled? Cited.
5. **maxfee enforcement.** Hard stop or soft budget? Cited.
6. **Self-pay / circular pay.** Does xpay support source=destination (paying our own invoice)? Cited or "unknown, must test in Task 6."

Each subsection ends with a verdict sentence: "(a) xpay full takeover viable", "(b) xpay with pinned route viable", or "(c) xpay unsuitable". These verdicts feed into Task 9's decision record.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md
git commit -m "docs(research): section 5 — xpay API surface"
```

---

## Task 6: xpay vs sendpay+waitsendpay Behavior Diff

**Files:**
- Modify: `docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md` (Section 6)

**Depends on:** Task 5 must be complete. Skip this task if Task 5 determined xpay is not available on the live node.

- [ ] **Step 1: Pick a safe circular test pair**

Pick two of our channels where:
- The source channel has ≥ 10,000 sats outbound
- The dest channel has ≥ 10,000 sats inbound
- Both peers are known-online (recent activity)

Record the two SCIDs and peer pubkeys.

- [ ] **Step 2: Build the sendpay baseline route**

Use the existing v2 router logic mentally: construct the 3-hop circular route (source_peer → ... → dest_peer → us). Use the v2 helper functions by calling them directly:

```bash
cd /home/sat/bin/cl_revenue_ops/.worktrees/askrene-router-v3-20260410
python3 - <<'PY'
import sys, json
sys.path.insert(0, ".")
# TODO: import the v2 router and run price_pair for the chosen pair
# For this research step, we want a concrete captured route.
PY
```

Or, if scripting through Python is inconvenient, construct the route by hand using `lightning-cli listpeerchannels` + `lightning-cli getroute` for the middle hop. Capture the full route as JSON.

- [ ] **Step 3: Run a 1000-sat sendpay circular rebalance**

With the route from Step 2, generate an invoice and send:

```bash
PREIMAGE=$(openssl rand -hex 32)
HASH=$(python3 -c "import hashlib; print(hashlib.sha256(bytes.fromhex('$PREIMAGE')).hexdigest())")
lightning-cli invoice amount_msat=1000000 label="v3-research-sendpay-$(date +%s)" description="v3 research sendpay test" expiry=600 preimage=$PREIMAGE > /tmp/invoice.json
BOLT11=$(python3 -c 'import json; print(json.load(open("/tmp/invoice.json"))["bolt11"])')
PAYMENT_HASH=$(python3 -c 'import json; print(json.load(open("/tmp/invoice.json"))["payment_hash"])')

# Time and capture
time lightning-cli sendpay route="$(cat /tmp/route.json)" payment_hash=$PAYMENT_HASH bolt11=$BOLT11 2>&1 | tee /tmp/sendpay-result.json
time lightning-cli waitsendpay payment_hash=$PAYMENT_HASH timeout=60 2>&1 | tee /tmp/waitsendpay-result.json
```

Capture: total wall time, number of hops, any errors, whether retries happened.

Clean up:

```bash
lightning-cli delinvoice label="v3-research-sendpay-..." status=paid 2>/dev/null || true
```

- [ ] **Step 4: Run a 1000-sat xpay circular rebalance**

Generate a fresh invoice for xpay:

```bash
PREIMAGE=$(openssl rand -hex 32)
lightning-cli invoice amount_msat=1000000 label="v3-research-xpay-$(date +%s)" description="v3 research xpay test" expiry=600 preimage=$PREIMAGE > /tmp/xpay-invoice.json
BOLT11=$(python3 -c 'import json; print(json.load(open("/tmp/xpay-invoice.json"))["bolt11"])')

time lightning-cli xpay "$BOLT11" 2>&1 | tee /tmp/xpay-result.json
```

Capture: total wall time, any "retry" / "split" output, final status.

Clean up:

```bash
lightning-cli delinvoice label="v3-research-xpay-..." status=paid 2>/dev/null || true
```

- [ ] **Step 5: Write Section 6 of the research doc**

Replace `_TODO: Task 6_` with content covering:

1. **Test pair.** SCIDs, peers, amounts.
2. **sendpay transcript.** Full RPC output from Step 3, timing data.
3. **xpay transcript.** Full RPC output from Step 4, timing data.
4. **Side-by-side comparison table.**

| Metric | sendpay+waitsendpay | xpay |
|---|---|---|
| Total wall time (ms) | ... | ... |
| Hops attempted | ... | ... |
| Retries | ... | ... |
| Final status | ... | ... |
| Fees paid (msat) | ... | ... |
| Audit-log noise (lines written) | ... | ... |

5. **Observations.** Anything surprising: xpay's route choice vs. v2 router's, MPP behavior, error messages.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md
git commit -m "docs(research): section 6 — xpay vs sendpay behavior diff"
```

---

## Task 7: setconfig Runtime-Switch Verification

**Files:**
- Modify: `docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md` (Section 7)

- [ ] **Step 1: Read the setconfig schema**

Run:

```bash
gh api repos/ElementsProject/lightning/contents/doc/schemas/setconfig.json --jq '.content' | base64 -d > /tmp/setconfig.json
python3 -m json.tool /tmp/setconfig.json | head -60
```

Record the request params and response shape.

- [ ] **Step 2: Find a hot-reloadable config key on cl-revenue-ops to test against**

Run:

```bash
lightning-cli listconfigs 2>&1 | python3 -c '
import sys, json
d = json.load(sys.stdin)
for k, v in d.get("configs", {}).items():
    if "revenue-ops" in k and isinstance(v, dict) and v.get("dynamic"):
        print(k, "=", v.get("value_str") or v.get("value_int"))
' | head -20
```

Pick one dynamic key that can be safely toggled (e.g. a log level or a boolean flag). Record its name and current value.

- [ ] **Step 3: Test setconfig on the chosen key**

Run (substitute KEY/VALUE with the choice from Step 2):

```bash
KEY=<choice>
OLD=$(lightning-cli listconfigs $KEY | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["configs"][sys.argv[1]]["value_str"])' $KEY)
NEW=<safe new value>

lightning-cli setconfig $KEY $NEW
lightning-cli listconfigs $KEY
# Inspect whether pyln-client picks up the change (check cl-revenue-ops logs for a config-change event)
tail -20 ~/.lightning/bitcoin/cl-revenue-ops.log 2>/dev/null || true

# Restore
lightning-cli setconfig $KEY $OLD
```

Record the full output of each step.

- [ ] **Step 4: Read pyln-client's config-change handling**

Run:

```bash
python3 -c 'import pyln.client, inspect; print(inspect.getfile(pyln.client))' 2>/dev/null
```

If pyln-client is installed, find the file and grep for `setconfig\|on_set_config\|config_change\|reconfigure`:

```bash
PYLN_CLIENT=$(python3 -c 'import pyln.client, os; print(os.path.dirname(inspect.getfile(pyln.client)))' 2>/dev/null || echo "")
[ -n "$PYLN_CLIENT" ] && grep -rn "setconfig\|on_set_config" "$PYLN_CLIENT"
```

Record whether pyln-client automatically notifies the plugin on setconfig, or if the plugin must poll.

- [ ] **Step 5: Write Section 7 of the research doc**

Replace `_TODO: Task 7_` with content covering:

1. **setconfig API.** Schema-cited request/response.
2. **Dynamic vs. static keys.** How CLN marks a key as runtime-changeable. Cite the schema field.
3. **Live test transcript.** Full RPC output from Step 3.
4. **pyln-client notification model.** Auto-notify vs. poll, with source citations from Step 4.
5. **Implication for v3.** One of:
   - "v3's `rebalance-router` config key can be declared `dynamic=True` and pyln-client will auto-push changes to the plugin" (best case)
   - "v3's `rebalance-router` must be re-read each cycle from `self.plugin.options` because pyln-client does not auto-notify" (fallback — matches the design's "re-read each cycle" pattern)

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md
git commit -m "docs(research): section 7 — setconfig runtime switch"
```

---

## Task 8: Failure-Mode Taxonomy

**Files:**
- Modify: `docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md` (Section 8)

**Depends on:** Tasks 1 and 5 must be complete.

- [ ] **Step 1: Extract every error string from askrene plugin source**

Run:

```bash
grep -rn 'command_fail\|PLUGIN_ERROR\|error.*"\|RPC_ERROR' /tmp/askrene-*.c | head -40
```

Record each error string with its source line.

- [ ] **Step 2: Extract every error string from xpay plugin source**

```bash
grep -n 'command_fail\|PLUGIN_ERROR\|error.*"\|RPC_ERROR' /tmp/xpay.c | head -40
```

Record each error string with its source line.

- [ ] **Step 3: List v2 router's existing skip reasons**

Read the v2 audit module and collect the skip reason strings:

```bash
grep -n 'reason=' modules/rebalance_audit_v2.py modules/rebalance_planner_v2.py modules/rebalance_router_v2.py
```

- [ ] **Step 4: Build the mapping table**

For every error collected in Steps 1-2, decide which v2 skip reason it maps to, or flag it as "new reason needed." Fill a table:

| Error (source) | Source line | Maps to skip reason | Notes |
|---|---|---|---|
| `"no route found"` | `askrene/flow.c:123` | `no_route` | existing |
| `"route cost exceeds maxfee_msat"` | `askrene/flow.c:145` | `route_over_budget` | existing |
| `"xpay: all partial payments failed"` | `xpay.c:890` | ??? | **new reason: `payment_all_parts_failed`** |
| ... | ... | ... | ... |

- [ ] **Step 5: Write Section 8 of the research doc**

Replace `_TODO: Task 8_` with content covering:

1. **Methodology.** How errors were collected (greps + source reading).
2. **Full error taxonomy table.** Every error with source line citation and mapped skip reason.
3. **New skip reasons needed.** Enumerated list with justification for each. These become new entries in `rebalance_audit_v2.py` during Phase 1.
4. **Ambiguity notes.** Any error whose mapping was guessed rather than obvious — flag for Phase 1 implementation review.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md
git commit -m "docs(research): section 8 — failure-mode taxonomy"
```

---

## Task 9: Decision Records

**Files:**
- Modify: `docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md` (Section 9)

**Depends on:** All previous research tasks must be complete.

- [ ] **Step 1: Decision — xpay integration depth**

Review the findings from Tasks 5 and 6. Based on:
- Whether xpay supports pinned routes
- Whether xpay handles self-pays correctly
- How its behavior diffs from sendpay on the real node

Pick ONE of:
- **(a) Full xpay takeover** — replace v2 executor entirely
- **(b) xpay with pinned route** — keep pair pinning, gain MPP
- **(c) Keep v2 executor, make v3 executor an alternative** — both coexist
- **(d) Reject xpay** — v3 is router-only, v2 executor stays permanently

Write the decision as one paragraph with cited evidence from Tasks 5-6 as justification.

- [ ] **Step 2: Decision — exclude handling**

Review Task 4. Based on the 50ms benchmark:

Pick ONE of:
- **Exclude-via-layer** — create throwaway layer per retry
- **Exclude-internal** — filter channels in v3 before calling getroutes

Write the decision as one paragraph with benchmark numbers.

- [ ] **Step 3: Decision — layer default set**

Review Task 3. Based on whether the `hive-fleet` layer actually influenced routes:

Pick ONE of:
- Keep spec default `askrene-layers = "hive-fleet"` (layer proved effective)
- Change default to `askrene-layers = "hive-fleet,hive-reputation"` (fleet alone insufficient, reputation adds value)
- Change default to `askrene-layers = ""` (layers had no observable effect, standalone mode is the default)

Write the decision with cited diff evidence from Task 3.

- [ ] **Step 4: Decision — runtime switch mechanism**

Review Task 7. Based on pyln-client's notification model:

Pick ONE of:
- **Auto-notify** — declare `rebalance-router` as `dynamic=True` and let pyln-client push changes
- **Poll per cycle** — engine re-reads `self.config.rebalance_router` at the start of each cycle

Write the decision with citation.

- [ ] **Step 5: Phase 1 go/no-go**

Look at all decisions above. If any of them forced a design compromise severe enough that Phase 1 no longer makes sense (e.g. Task 3 showed layers have zero effect AND Task 6 showed xpay is broken for self-pays), declare "Phase 1 no-go" and document the reason.

Otherwise declare "Phase 1 go" and note any design deltas from the parent spec that Phase 1 must incorporate.

- [ ] **Step 6: Write Section 9 of the research doc**

Replace `_TODO: Task 9_` with a structured decision log:

````markdown
## 9. Decision Records

### 9.1 xpay integration depth
**Decision:** <a|b|c|d>
**Evidence:** <citation>
**Rationale:** <one paragraph>

### 9.2 Exclude handling
**Decision:** <via-layer|internal>
**Evidence:** <benchmark>
**Rationale:** <one paragraph>

### 9.3 Default layer set
**Decision:** <hive-fleet|hive-fleet,hive-reputation|empty>
**Evidence:** <citation from Task 3>
**Rationale:** <one paragraph>

### 9.4 Runtime switch mechanism
**Decision:** <auto-notify|poll>
**Evidence:** <citation>
**Rationale:** <one paragraph>

### 9.5 Phase 1 go/no-go
**Decision:** <go|no-go>
**Rationale:** <one paragraph>
**Design deltas:** <list or "none">
````

- [ ] **Step 7: Commit**

```bash
git add docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md
git commit -m "docs(research): section 9 — decision records"
```

---

## Task 10: Self-Review And Finalize

**Files:**
- Modify: `docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md` (status header)

- [ ] **Step 1: Placeholder scan**

Run:

```bash
grep -n "TODO\|TBD\|_TODO: Task\|<paste\|<choice\|<citation\|<one paragraph\|<list\|<benchmark\|<a|b|c|d>\|\.\.\. |" docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md
```

Expected: zero matches (except inside fenced code blocks that are intentional examples). Any match is a placeholder that was never filled in — go back to the relevant task and fix it.

- [ ] **Step 2: Citation coverage sanity check**

Run:

```bash
grep -c "ElementsProject/lightning@" docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md
```

Expected: at least 20 citations across the whole doc. If fewer, specific claims lack evidence — go find them.

- [ ] **Step 3: Internal consistency check**

Read the doc top to bottom once. Check:
- Every conclusion in Section 9 is supported by evidence in Sections 1-8
- Terminology is consistent (always "getroutes" not "askrene-getroutes", always "hive-fleet" not "fleet-layer")
- The environment section at the top matches the CLN version actually used in experiments

If anything is inconsistent, fix inline.

- [ ] **Step 4: Flip the status header**

Change the doc header from:

```markdown
**Status:** In progress
```

to:

```markdown
**Status:** Complete — ready for Phase 1 plan
```

Only flip this if Section 9's Phase 1 go/no-go is "go." If it's "no-go," the status becomes "Complete — Phase 1 rejected" and this plan ends here; no Phase 1 plan is written.

- [ ] **Step 5: Final commit**

```bash
git add docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md
git commit -m "docs(research): finalize v3 router research, ready for phase 1"
```

- [ ] **Step 6: Report completion**

Output a summary:

```
Research phase complete.
Commits on feature/askrene-router-v3:
<git log main..HEAD --oneline>

Key decisions:
- xpay depth: <a|b|c|d>
- Exclude handling: <via-layer|internal>
- Default layers: <...>
- Runtime switch: <auto-notify|poll>
- Phase 1: <go|no-go>

Next step: user reviews research doc and approves writing the Phase 1 implementation plan.
```

---

## Self-Review

**Spec coverage check:** Does this plan produce all nine research sections enumerated in the parent spec?

- Section 1 (getroutes contract) → Task 1 ✓
- Section 2 (layer lifecycle) → Task 2 ✓
- Section 3 (layer semantics under pair pinning) → Task 3 ✓
- Section 4 (exclude-via-layer pattern) → Task 4 ✓
- Section 5 (xpay API surface) → Task 5 ✓
- Section 6 (xpay vs sendpay diff) → Task 6 ✓
- Section 7 (setconfig runtime switch) → Task 7 ✓
- Section 8 (failure-mode taxonomy) → Task 8 ✓
- Section 9 (decision records) → Task 9 ✓

Plus scaffolding (Task 0) and final review (Task 10). All covered.

**Placeholder scan:** This plan contains intentional placeholders like `<paste SHA from Step 1>` that are to be filled in during execution — those are meta-instructions for the executor, not plan gaps. The plan itself has no unfilled sections.

**Methodology discipline:** Every behavioral claim in every task has a required citation or transcript. Task 10's self-review enforces this at the doc level.

**YAGNI check:** No task produces code. No task implements optimizations. The plan ends the moment the research doc is complete — no scope creep into Phase 1.
