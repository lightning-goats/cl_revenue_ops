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

The reviewed range is `b6e6c9a5437405fae2c09169847d507acab13b94..cec0309a55b00b669f20c1ead8b68fc60c828981`. Key exact commits are:

- `583bd0657a890e153f1bacee900bae50a40c094f` — approved Phase 1A design;
- `17575ec0c6890df2ca093c2e7c5e798fc7c829c2` — implementation plan;
- `25f996e97cdedb4b4986f54080a6dd604e52cb8a` through `79b61069eb7b74703ffd3d68f17d3033796e4089` — v0 sealed wire corrections;
- `3ca0324382e9aafe04557bda17abd95df794ba10` — complete planner-funnel retention;
- `d0c8d7fbab35001dc178b0c057ad69dff537b0f2` through `7630b75077629b9da85b0fbb9e038170fb535b75` — bounded capture and Task 3 lifecycle hardening; and
- `503db0eda7c2d88da416dccef5e957b0c6798b4a` and `cec0309a55b00b669f20c1ead8b68fc60c828981` — standalone replay and final hardening.

The v0 envelope captures the normalized pre-cycle `StateSnapshot`, the six planner configuration fields, producer identity/version/timestamps, the full generated cheap-pair universe and rank, planner selection/rejection metadata, final selected-pair observations (including the existing route summary and post-price evidence), pair-linked execution outcomes, completeness counters, terminal stage, and a SHA-256 integrity seal. It intentionally does **not** capture payment secrets, invoices, raw RPC payloads, a full plugin-config dump, historical gossip, route alternatives, amount ladders, or a re-executable Askrene session.

Bounds are explicit: 1,024 snapshot channels; 4,096 generated pairs; 64 final pairs and pair outcomes; 128 skips; a 100,000-node handoff graph; a two-slot non-blocking writer queue; 32 MiB per sealed envelope; and retention of at most 32 owned files / 256 MiB. Error text and structured failure evidence are bounded; only an allowlisted failure projection and the existing bounded route summary cross the capture boundary. Symlink output and unsafe replay input types are rejected.

### Narrow replay guarantee

For an eligible, sealed, complete v0 envelope, the offline reader reconstructs only the captured normalized snapshot and planner configuration, reruns the pure `RebalancePlanner`, and byte-compares the generated universe and planner-selected list under the documented binary64 representation. It does not replay historical route pricing, probability, post-price EV, policy, governor decisions, reservations, executor work, or outcomes. In particular, a future gossip graph cannot recreate the historical Askrene quote.

The reader accepts one local regular file, verifies schema and digest, rejects ineligible/truncated/count-incomplete evidence, and offers only an optional formatting flag. Architecture and clean-process regressions confirm it does not import the plugin entrypoint, an RPC client, router/executor code, database code, or an action surface.

## RED and GREEN evidence

Task 3's review probes were RED before their corresponding lifecycle, ownership, bounded-copy, and publication corrections. Task 4 began RED with:

```text
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_rebalance_replay.py -q
11 failed: tools/rebalance_replay.py did not yet exist.
```

The current Task 5 GREEN runs were executed at `cec0309`:

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

The focused command is exactly the Task 5 matrix in `.superpowers/sdd/task-5-brief.md`: replay wire, capture, capture config, replay, planner, engine, orchestrator, policy gate, operator/RPC surface, and architecture guards. The skips require unavailable live/`pyln.testing` infrastructure. The two expected failures are separately staged compatibility-removal checks; they do not concern replay capture.

Static and boundary evidence:

```text
python -m py_compile [the seven Task 5 runtime files]: passed
tests/test_architecture_guard.py: included in the 404-pass focused run
direct scan of the three new runtime files for Sling/Hive/mycelium/fleet: no matches
direct scan of tools/rebalance_replay.py for action/RPC/plugin terms: no matches
```

The prescribed `pyflakes` command reports three pre-existing unused imports in `modules/rebalance_engine_v2.py` (`RouteDecision`, `RoutePolicy`, and the shared `_drain_score`/`_refill_urgency` import line). The same imports exist at the merge base; the capture increment introduced none. The prescribed `git diff --check b6e6c9a..HEAD` is also non-green solely for blank EOF lines in the pre-existing Task 1 plan/design files: `docs/optimization/plans/2026-08-20-rebalance-replay-capture.md:574` and `...-design.md:241`. Neither issue is changed by this finding.

The separately run installed-environment pin test is intentionally recorded without concealment:

```text
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest -q tests/test_supply_chain_pins.py
1 failed, 18 passed in 0.41s
```

The only failure is known development-environment drift: required versus installed `pyln-client` `25.12.1` / `26.4`, `PyYAML` `6.0.1` / `6.0.3`, and `numpy` `1.26.4` / `2.4.2`. It is not a functional-suite bypass and must be resolved in a separately pinned environment before a release gate relies on that test.

## No-behavior and safety review

The complete merge-base range was reviewed for selected-list/order, route-call, executor-call, capture-thread, filesystem/retention, secret/payload, malformed evidence, replay dependency, schema/config/RPC, and standalone-architecture regressions. No Critical or Important finding remained.

Planner trace fields are assigned after the existing score sort and greedy selection. `selected` continues to be the only list passed to existing pricing/execution; `generated` is observational. Dedicated regressions cover unchanged golden selected output, disabled-path inertness, unchanged route, reserve, and executor call counts when capture fails or drops, pair-linked concurrent outcomes, malformed/neutral data, and action-option rejection.

On the cycle path, bounded evidence detachment occurs only after the engine cycle lock is released. Projection, validation, sealing, serialization, manifest I/O, fsync, and retention are daemon-owned. A full handoff queue drops the observation before copying it; capture exceptions are contained and leave the pre-existing cycle result/exception identity authoritative.

Task 3's in-memory host benchmark measured representative 16-pair/32-channel preparation at 1.746–2.195 ms and 41,751 serialized bytes. The maximum fixture (4,096 generated pairs, 1,024 channels, 64 final pairs, 128 skips, 64 outcomes) prepared in 142.640 ms and serialized to 2,820,095 bytes. Its daemon projection took 30.989 ms and sealing/serialization 220.578 ms; those daemon costs are absent from the cycle thread. These are laboratory preparation measurements, not production latency evidence.

## Follow-ups and gate

- Implement a separately reviewed recorded-price EV replay slice; v0 retains price evidence but cannot recreate or evaluate historical Askrene pricing.
- Capture the complete pre-engine/orchestrator suppression funnel before claiming whole-cycle replay coverage.
- Gather shadow evidence for alternate amount candidates/amount ladders; no alternate-amount regret claim is supported now.
- After the Phase 0 gate passes, require a separate operator-approved, bounded-latency production shadow proposal before enabling capture. That proposal still must not give any optimizer authority.
- Resolve the local `pyln-client`/`PyYAML`/`numpy` pin drift in a release-like environment, and separately clean the inherited pyflakes/diff-check hygiene findings if their owners choose to do so.

## Activation recommendation

**IMPLEMENTED, NOT DEPLOYED. SHADOW ACTIVATION NOT AUTHORIZED. PHASE 1 GATE NOT YET MET.** Keep capture default-off and replay offline. No economic, routing, fee, budget, or execution change is recommended by this finding.
