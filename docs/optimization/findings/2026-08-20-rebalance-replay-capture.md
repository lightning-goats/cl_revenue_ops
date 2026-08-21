# Rebalance replay capture — Phase 1A finding

**Date:** 2026-08-20
**Scope:** Phase 1A deterministic rebalance-planner evidence
**Status:** **IMPLEMENTED, NOT DEPLOYED**
**Activation:** **SHADOW ACTIVATION NOT AUTHORIZED**
**Program gate:** **PHASE 1 GATE NOT YET MET**

## Disposition

This increment adds default-off, local, observational capture and an offline planner replay reader. It does not enable capture, change a selected pair, change route pricing, reserve budget, execute a rebalance, or authorize an optimizer. No production deployment, runtime enablement, restart, action RPC, or live RPC occurred while preparing this finding.

Phase 0's 72-hour durable-evidence gate remains the prerequisite for any reviewed production shadow rollout. The current Phase 1A result is therefore an implementation and test result only; it is not evidence of an economic improvement, optimization, deployment, or activation.

## Reviewed implementation evidence

The final-review fix wave reviewed the complete range from `b6e6c9a5437405fae2c09169847d507acab13b94` through parent `c94d432a39bfd9f42b4bb323ff940ef6dbb9380f` and the implementation/test changes committed with this finding update. Key exact earlier commits are:

- `583bd0657a890e153f1bacee900bae50a40c094f` — approved Phase 1A design;
- `17575ec0c6890df2ca093c2e7c5e798fc7c829c2` — implementation plan;
- `25f996e97cdedb4b4986f54080a6dd604e52cb8a` (initial envelope), `c188ce4b207491fd1a517d06e3f58a06c893fcb3` (validation hardening), `3d23e2243860c16f7cfc91827dcff13519a54941` (schema alignment), `1877c04494b4e1f0212e99d7d436282cf6de3875` (binary64 encoding), `94950b6e1e2345426cfbc33b76b57d1758e47273` (reserved float tags), and `79b61069eb7b74703ffd3d68f17d3033796e4089` (integer normalization);
- `3ca0324382e9aafe04557bda17abd95df794ba10` — complete planner-funnel retention;
- `d0c8d7fbab35001dc178b0c057ad69dff537b0f2` through `7630b75077629b9da85b0fbb9e038170fb535b75` — bounded capture and Task 3 lifecycle hardening; and
- `503db0eda7c2d88da416dccef5e957b0c6798b4a` and `cec0309a55b00b669f20c1ead8b68fc60c828981` — standalone replay and final hardening; and
- `81237f2ccbd8c12d92784b09dca9727b9da41730` and `c94d432a39bfd9f42b4bb323ff940ef6dbb9380f` — evidence finding and clarification preceding this final-review fix wave.

The v0 envelope captures the normalized pre-cycle `StateSnapshot`, the six planner configuration fields, producer identity/version/timestamps, the full generated cheap-pair universe and rank, planner selection/rejection metadata, final selected-pair observations (including the existing route summary and post-price evidence), pair-linked execution outcomes, completeness counters, terminal stage, and a SHA-256 integrity seal. It intentionally does **not** capture payment secrets, invoices, raw RPC payloads, a full plugin-config dump, historical gossip, route alternatives, amount ladders, or a re-executable Askrene session.

Bounds are explicit: 1,024 snapshot channels; 4,096 generated and 4,096 planner-selected pair references; 64 final pairs and pair outcomes; 128 skips; a 100,000-node handoff graph; a two-slot non-blocking writer queue; 32 MiB per sealed envelope; and retention of at most 32 owned files / 256 MiB. Runtime validation and JSON Schema now enforce the same six collection limits and the same integer domains. Error text and structured failure evidence are bounded; outcome evidence is reduced to the strict primitive-only allowlist before graph traversal, deepcopy, or async handoff, so payment secrets, hashes, invoices, and raw RPC graphs cannot enter the queue-owned object graph. Symlink output and unsafe replay input types are rejected.

### Manifest durability limitation

Lifecycle calls enqueue attempted, drop, queued terminal-stage, and terminal-result manifest mutations without waiting for filesystem I/O; disk state may therefore lag the newest in-memory or pending lifecycle state. Newer snapshots coalesce per run, and a fixed 64-pending-run cap applies backpressure rather than silently accepting a run whose terminal manifest cannot be retained. Consumers must verify that the terminal manifest is durable before treating a capture as recorded. An abrupt process death can still lose an enqueued, not-yet-durable manifest correction; this is observational evidence loss, not an authorization or execution signal.

### Narrow replay guarantee

For an eligible, sealed, complete v0 envelope, the offline reader reconstructs only the captured normalized snapshot and planner configuration, reruns the pure `RebalancePlanner`, and byte-compares the generated universe and planner-selected list under the documented binary64 representation. It does not replay historical route pricing, probability, post-price EV, policy, governor decisions, reservations, executor work, or outcomes. In particular, a future gossip graph cannot recreate the historical Askrene quote.

The reader accepts one local regular file, verifies schema and digest, rejects ineligible/truncated/count-incomplete evidence, and offers only an optional formatting flag. Architecture and clean-process regressions confirm it does not import the plugin entrypoint, an RPC client, router/executor code, database code, or an action surface.

## RED and GREEN evidence

Task 3's review probes were RED before their corresponding lifecycle, ownership, bounded-copy, and publication corrections. Task 4 began RED with:

```text
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_rebalance_replay.py -q
11 failed: tools/rebalance_replay.py did not yet exist.
```

The original Task 5 GREEN runs were executed at `cec0309`:

```text
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest -q \
  tests/test_rebalance_cycle_replay_wire.py \
  tests/test_rebalance_cycle_capture.py \
  tests/test_rebalance_cycle_capture_config.py \
  tests/test_rebalance_replay.py \
  tests/test_rebalance_planner_v2.py \
  tests/test_rebalance_engine_v2.py \
  tests/test_rebalance_orchestrator_v2.py \
  tests/test_rebalance_policy_gate.py \
  tests/test_operator_surface.py \
  tests/test_rpc_surface_inventory.py \
  tests/test_architecture_guard.py
404 passed in 3.69s

/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest -q --ignore=tests/test_supply_chain_pins.py
3469 passed, 5 skipped, 2 xfailed in 54.18s
```

The final whole-branch findings were then reproduced RED before production edits:

```text
engine ownership/bookkeeping regressions: 3 failed, 87 deselected
capture sanitation/lifecycle regressions: 2 failed, 70 deselected
wire integer/boundary regressions: 31 failed, 155 passed, 77 deselected
```

After the integrated fixes and the retention-race correction exposed by the first focused run, final GREEN evidence is:

```text
exact focused Task 5 matrix: 575 passed in 10.78s
full functional suite excluding only tests/test_supply_chain_pins.py:
3640 passed, 5 skipped, 2 xfailed in 63.12s

installed-environment pin suite:
1 failed, 18 passed in 0.40s
```

The focused command is exactly the Task 5 matrix in `.superpowers/sdd/task-5-brief.md`: replay wire, capture, capture config, replay, planner, engine, orchestrator, policy gate, operator/RPC surface, and architecture guards. The skips require unavailable live/`pyln.testing` infrastructure. The two expected failures are separately staged compatibility-removal checks; they do not concern replay capture.

Static and boundary evidence:

```text
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m py_compile \
  modules/rebalance_cycle_replay_wire.py modules/rebalance_cycle_capture.py \
  modules/rebalance_types_v2.py modules/rebalance_planner_v2.py \
  modules/rebalance_engine_v2.py tools/rebalance_replay.py cl-revenue-ops.py
exit 0

rg -n -i '(^|[^[:alpha:]])(sling|hive|mycelium|fleet)([^[:alpha:]]|$)' \
  modules/rebalance_cycle_capture.py modules/rebalance_cycle_replay_wire.py \
  tools/rebalance_replay.py
exit 1 (no matches)

rg -n -i 'sendpay|waitsendpay|lightning|plugin|pyln|\.rpc|rpc\.call|revenue-rebalance|revenue-fee|execute|apply' \
  tools/rebalance_replay.py
exit 1 (no matches)

tests/test_architecture_guard.py: included in the 404-pass focused run
```

The prescribed `pyflakes` command reports four pre-existing unused-import diagnostics in `modules/rebalance_engine_v2.py`: `RouteDecision`, `RoutePolicy`, `_drain_score`, and `_refill_urgency`. The same imports exist at the merge base; the capture increment introduced none. After removing the two branch-owned trailing blank EOF lines in the Task 1 plan/design, the prescribed whole-range `git diff --check $(git merge-base main HEAD)..HEAD` exits 0.

The separately run installed-environment pin test is intentionally recorded without concealment:

```text
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest -q tests/test_supply_chain_pins.py
1 failed, 18 passed in 0.41s
```

The only failure is known development-environment drift: required versus installed `pyln-client` `25.12.1` / `26.4`, `PyYAML` `6.0.1` / `6.0.3`, and `numpy` `1.26.4` / `2.4.2`. It is not a functional-suite bypass and must be resolved in a separately pinned environment before a release gate relies on that test.

## No-behavior and safety review

The complete merge-base range was reviewed for selected-list/order, route-call, executor-call, capture-thread, filesystem/retention, secret/payload, malformed evidence, replay dependency, schema/config/RPC, and standalone-architecture regressions. The five final-review findings were fixed together and no Critical or Important finding remained.

Planner trace fields are assigned after the existing score sort and greedy selection. `selected` continues to be the only list passed to existing pricing/execution; `generated` is observational. The review and golden tests establish unchanged selected output. New direct capture-enabled/disabled engine regressions prove identical route pricing, budget reservation, executor, authoritative worker bookkeeping, and idempotent backstop calls even when an execution result rejects deepcopy. A separate deterministic ownership race proves that an outer capture attempt returning no identity cannot let inner planning acquire a second identity after capture becomes enabled.

On the cycle path, authoritative result bookkeeping now precedes fallible observational detachment; a detachment failure retains returned status/result authority and is contained. Bounded manager evidence detachment occurs only after the engine cycle lock is released. Outcome allowlisting occurs before generic graph traversal/deepcopy. Projection, validation, sealing, serialization, manifest I/O, fsync, and retention are daemon-owned. A full handoff queue drops the observation before copying it; capture exceptions remain contained. Direct and publication-triggered retention rotation share one re-entrant filesystem lock, preserving the 32-file bound during concurrent manifest publication.

The following are archival Task 3 in-memory measurements, not a preserved benchmark harness or repeatable gate: representative 16-pair/32-channel preparation measured 1.746–2.195 ms and 41,751 serialized bytes. The recorded maximum fixture (4,096 generated pairs, 1,024 channels, 64 final pairs, 128 skips, 64 outcomes) prepared in 142.640 ms and serialized to 2,820,095 bytes. Its daemon projection took 30.989 ms and sealing/serialization 220.578 ms; those daemon costs are absent from the cycle thread. These archival laboratory measurements are not production latency evidence.

## Follow-ups and gate

- Implement a separately reviewed recorded-price EV replay slice; v0 retains price evidence but cannot recreate or evaluate historical Askrene pricing.
- Capture the complete pre-engine/orchestrator suppression funnel before claiming whole-cycle replay coverage.
- Gather shadow evidence for alternate amount candidates/amount ladders; no alternate-amount regret claim is supported now.
- After the Phase 0 gate passes, require a separate operator-approved, bounded-latency production shadow proposal before enabling capture. That proposal still must not give any optimizer authority.
- Resolve the local `pyln-client`/`PyYAML`/`numpy` pin drift in a release-like environment; the four inherited pyflakes diagnostics remain outside this documentation slice.

## Activation recommendation

**IMPLEMENTED, NOT DEPLOYED. SHADOW ACTIVATION NOT AUTHORIZED. PHASE 1 GATE NOT YET MET.** Keep capture default-off and replay offline. No economic, routing, fee, budget, or execution change is recommended by this finding.
