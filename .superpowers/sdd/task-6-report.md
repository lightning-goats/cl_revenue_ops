# Task 6 report: complete local verification and independent review gate

## Status

**DONE**

Local functional, static, scope, independent review, and fresh copied-snapshot
verification gates are complete. The separately isolated supply-chain pin test
still reports three shared-environment version drifts as a nonfunctional
environment concern; it is not hidden or treated as a functional failure. No
deployment or Task 7 production action was performed.

Review range: `11c1a8d...0ea57ea` before this report commit.

## Task 2 reviewer-gap closure

Commit:

- `0ea57ea test: close forward archive recovery review gaps`

Files changed in that test-only commit:

- `tests/test_forward_archive.py`
- `tests/test_forward_archive_sync.py`

No production file changed. The added store test parameterizes all five
existing incomplete-coverage predicates:

- `created_sync_complete != 1`
- `updated_sync_complete != 1`
- `aggregate_complete != 1`
- `reconciliation_status != 'complete'`
- `reasons_json != '[]'`

The existing created- and updated-backlog tests now wrap
`closed_days_needing_rebuild()` and `rebuild_days()` and directly assert that
neither method is called during a partial cycle.

### Test-first and mutation evidence

The seven new/strengthened cases first passed against the already-correct
implementation:

```text
.......                                                                  [100%]
7 passed in 0.21s
```

Because these were coverage gaps rather than an implementation defect,
controlled temporary mutations were used to prove regression sensitivity
without retaining any production edit:

1. Removing the five incomplete-row predicates produced the expected five
   assertion failures; every case returned `()` instead of `(1699920000,)`:

   ```text
   FFFFF                                                                    [100%]
   5 failed in 0.18s
   ```

2. Temporarily invoking discovery from both partial-return branches produced
   the expected direct call failures:

   ```text
   FF                                                                       [100%]
   Expected 'mock' to not have been called. Called 1 times.
   Calls: [call(0)].
   2 failed in 0.22s
   ```

Both mutations were restored immediately. `git diff` then showed no production
file change. The complete affected-file suite and static checks were green
before the test commit:

```text
61 passed in 0.46s
pyflakes tests/test_forward_archive.py tests/test_forward_archive_sync.py
# exit 0, no output
git diff --check
# exit 0, no output
```

## Task 6 verification evidence

The isolated worktree has no `.venv`; commands used the repository shared
interpreter `/home/sat/bin/cl_revenue_ops/.venv/bin/python` (Python 3.12.3), as
already recorded for Task 2.

### Syntax and static checks

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m py_compile \
  cl-revenue-ops.py \
  modules/forward_archive.py \
  modules/forward_archive_sync.py \
  tools/audit/verify_forward_archive.py
pyflakes \
  modules/forward_archive.py \
  modules/forward_archive_sync.py \
  tools/audit/verify_forward_archive.py \
  tests/test_forward_archive.py \
  tests/test_forward_archive_sync.py \
  tests/test_verify_forward_archive.py
```

Result: exit 0, no output from either command.

### Complete focused regression suite

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest -q \
  tests/test_forward_archive.py \
  tests/test_forward_archive_sync.py \
  tests/test_verify_forward_archive.py \
  tests/test_operator_surface.py \
  tests/test_perf_regression_guard.py \
  tests/test_persistence_inventory.py \
  tests/test_architecture_guard.py
```

Result:

```text
167 passed in 1.71s
```

Specific contracts confirmed by this suite:

- the shaped empty source remains valid and does not call `listforwards` when
  both cursor states are already beyond the sampled live maximum;
- empty pages before a live maximum, malformed records, malformed live-max
  payloads, cursor regressions, unsupported schema, and malformed verifier
  identity fail closed;
- current-day coverage retains `day_not_closed` and cannot become complete;
- `test_sync_rpc_allowlist_is_wait_and_listforwards_only` restricts sync calls
  to `wait` and `listforwards`.

### Full functional suite

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest -q \
  --deselect=tests/test_supply_chain_pins.py::test_requirements_txt_matches_installed_environment
```

Result:

```text
3122 passed, 5 skipped, 1 deselected, 2 xfailed in 48.10s
```

The five skips were the absent `pyln.testing` integration dependency plus four
`CLN_INTEGRATION`-gated live router tests. The two xfails are the already-pinned
staged removal-readiness cases.

### Separately isolated supply-chain pin test

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest -q \
  tests/test_supply_chain_pins.py::test_requirements_txt_matches_installed_environment
```

Result:

```text
1 failed in 0.14s
```

The repository pin checker reported exactly three shared-environment drifts:

```text
pyln-client pinned 25.12.1, installed 26.4
PyYAML pinned 6.0.1, installed 6.0.3
numpy pinned 1.26.4, installed 2.4.2
```

This is an environment-pin mismatch, not a functional-suite failure; it was
run and reported separately as required.

## Production snapshot verifier

The initial permitted `scp` attempt was rejected by the host escalation
reviewer, and no snapshot bytes were transferred locally. The supervisor then
resolved the gate without data transfer: the reviewed verifier was streamed
over SSH into transient `python3 -` execution on `lnnode` and opened the
existing copied snapshot at
`/tmp/revenue_ops-cf0cf49-20260813T2158Z.db` with SQLite `mode=ro`.

Exact half-open UTC bounds:

```text
history_since=1783900800
history_until=1786579200
```

Fresh result:

```text
exit_code=1
reasons=["coverage_incomplete"]
overlap_status=legacy_dedup_explained
legacy_projection_equal=true
legacy_identity_consistent=true
legacy_loss_consistent=true
query_plan_bounded=true
warnings=["legacy_operational_dedup_loss"]

canonical.settled_forward_count=1592
canonical.fee_msat=20264370
legacy_projected_archive.settled_forward_count=1559
legacy_projected_archive.fee_msat=19993272
operational.settled_forward_count=1559
operational.fee_msat=19993272

legacy_dedup_loss.settled_forward_count=33
legacy_dedup_loss.fee_msat=271098
legacy_dedup_loss.forwarded_in_msat=1530608765
legacy_dedup_loss.forwarded_out_msat=1530337667
```

Exit 1 is explained solely by `coverage_incomplete`, which is the expected
state of this intentionally pre-fix two-row coverage snapshot. The exact
legacy projection matches operational count and fees, all identity/loss checks
are true, every required query plan remains bounded, and the canonical delta is
reported only as `legacy_operational_dedup_loss`.

## Diff, scope, and coordinator checks

Commands:

```bash
git diff 11c1a8d...HEAD --check
git diff 11c1a8d...HEAD --stat
git status --short --branch
rg -n "sling|cl_hive|mycelium|fleet" \
  modules/forward_archive.py \
  modules/forward_archive_sync.py \
  tools/audit/verify_forward_archive.py \
  cl-revenue-ops.py
git diff 11c1a8d...HEAD -- \
  cl-revenue-ops.py modules/forward_archive.py \
  modules/forward_archive_sync.py tools/audit/verify_forward_archive.py \
  | rg -n '^\+.*(sling|cl_hive|mycelium|fleet)'
```

Results before this report was written:

- `git diff --check`: exit 0, no output.
- worktree: clean on `codex/forward-archive-preflight-fix`.
- exact range: 15 paths, 2,170 insertions, 37 deletions.
- the four runtime files contain only pre-existing historical/inert fleet or
  retired-integration terminology in `cl-revenue-ops.py`.
- no added line in the exact-range runtime diff contains Sling, Hive,
  mycelium, fleet, or coordinator integration.
- functional file-map scope matches the plan. The two meta-artifact paths
  outside that map are the implementation plan itself (the first commit after
  `11c1a8d`) and the required Task 4 report; neither affects runtime.

Exact-range files before this report:

- `.superpowers/sdd/task-4-report.md`
- `cl-revenue-ops.py`
- `docs/optimization/README.md`
- `docs/optimization/adr/ADR-002-canonical-forward-archive.md`
- `docs/optimization/findings/phase0-measurement-hardening.md`
- `docs/optimization/plans/2026-08-13-forward-archive-preflight-corrections.md`
- `docs/optimization/validation/baseline.md`
- `docs/refactor/phase0/production-evaluation-final.md`
- `modules/forward_archive.py`
- `modules/forward_archive_sync.py`
- `tests/test_forward_archive.py`
- `tests/test_forward_archive_sync.py`
- `tests/test_operator_surface.py`
- `tests/test_verify_forward_archive.py`
- `tools/audit/verify_forward_archive.py`

## Independent code-review verdict

Verdict for `11c1a8d...0ea57ea`: **APPROVED for local implementation; no
Critical or Important findings. Deployment remains gated on the separately
authorized Task 7 production workflow.**

Review categories:

- **False-clean and missing-data behavior:** incomplete recovery selects
  missing coverage plus every incomplete predicate; malformed legacy-key fields
  produce `malformed_settled_record` and `unexplained`; incomplete coverage and
  invalid JSON fail closed. The new mutation-tested coverage closes both Task 2
  reviewer gaps.
- **Bounded SQLite work:** recovery is restricted to the indexed half-open
  400-day raw window, orders at most 401 discovered days, and errors rather than
  silently truncating above the 400-day contract. Rebuild work is bounded to
  the resulting date set. Verifier query-plan tests require the archive,
  operational raw, rollup, inbound, and legacy-projection indexes.
- **Production-table mutation:** recovery only replaces
  `forward_daily_channel_v1` and refreshes `forward_archive_coverage_v1`.
  It does not write the legacy `forwards` table. The offline verifier resolves
  the copied path and opens SQLite with `mode=ro`; its byte-preservation test is
  green.
- **Economic decisions:** archive references outside its store/synchronizer are
  limited to construction, its daemon, and the bounded read-only history RPC.
  No fee, profitability, flow, budget, routing, or rebalance decision path reads
  archive evidence.
- **Exact legacy key and resolution time:** projection groups on the seven-field
  operational identity `(in_channel, out_channel, in_msat, out_msat, fee_msat,
  integer received timestamp, integer resolved timestamp)`. Archive nanoseconds
  are converted to integer seconds, and resolution time matches the legacy
  `resolved_time` unique-index field. Single-field mismatches and null settled
  identity fields fail closed.
- **Current-day completion:** recovery discovery uses `[current_day - 400 days,
  current_day)`, so it excludes the current day. Any current-day touched record
  passed to coverage refresh receives `day_not_closed`, preventing a complete
  status.
- **RPC surface and action reachability:** synchronizer code calls only `wait`
  and `listforwards`; the focused allowlist and architecture tests pass. No
  action or mutation RPC was added or invoked.
- **No coordinator dependency:** no new Sling, Hive, mycelium, fleet, or
  external-coordinator import, call, or decision path appears in the exact
  range.

## Required safety report

- **Files changed by this Task 6 worker:**
  `tests/test_forward_archive.py`, `tests/test_forward_archive_sync.py`, and
  this report. Production code was not changed.
- **Tests run:** focused reviewer-gap cases (7 pass); mutation checks (5 and 2
  expected failures); affected-file suite (61 pass); focused Task 6 suite (167
  pass); functional suite (3,122 pass, 5 skip, 2 xfail, 1 deselected); isolated
  pin test (1 environment-drift failure).
- **No-Sling confirmation:** no Sling dependency or newly added retired
  coordinator reference; architecture guard passes.
- **No action RPCs triggered:** no CLN RPC was called by this worker. Tests used
  in-memory/temporary SQLite and mocks. The initial explicitly permitted
  read-only snapshot copy was rejected before copying any bytes. The supervisor
  completed fresh verification by streaming verifier code to transient
  `python3 -`; it opened only the existing copied snapshot with `mode=ro` and
  did not call a CLN RPC.
- **Production compatibility:** Python syntax/static checks are clean; all
  functional tests pass; schema stays additive/version 1; recovery writes only
  archive-owned aggregate/coverage tables; verifier remains offline/read-only.
  Fresh verification against the named copied production snapshot confirms
  exact explained legacy deduplication, consistent identity/loss semantics, and
  bounded query plans; incomplete coverage is expected for the pre-fix snapshot.
- **Follow-up risks:** reconcile the three shared-environment pin drifts
  separately. Task 7 production approval/deploy/post-fix verification remains
  unauthorized and unperformed.
