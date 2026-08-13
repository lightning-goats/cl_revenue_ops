# Task 4 report: strict legacy-key archive projection

## Status

Implemented and verified the copied-database verifier's strict legacy projection.
The verifier now distinguishes literal canonical equality, an exactly explained
legacy unique-key dedup loss, and every unexplained residual or inconsistent
delta.

## Files changed

- `tools/audit/verify_forward_archive.py`
- `tests/test_verify_forward_archive.py`
- `.superpowers/sdd/task-4-report.md`

No other project files were changed.

## Root cause and implementation

The previous verifier compared only canonical and operational aggregate totals.
It therefore could neither explain rows necessarily collapsed by the legacy
unique index nor detect channel/time identity drift when aggregate amounts still
matched.

The verifier now:

- projects settled archive rows through the exact legacy key: inbound channel,
  outbound channel, inbound msat, outbound msat, fee msat, integer received
  timestamp, and integer resolved timestamp;
- bounds archive projection by generation, status, and the requested half-open
  nanosecond window;
- validates retained operational raw keys against projected archive keys in the
  same half-open second window;
- publishes canonical, projected, and operational totals plus the canonical to
  operational delta;
- preserves `overlap_equal` as literal canonical aggregate equality;
- publishes `legacy_projection_equal`, `legacy_loss_consistent`, and
  `overlap_status` (`equal`, `legacy_dedup_explained`, or `unexplained`);
- emits `legacy_operational_dedup_loss` only in `warnings` for an exact explained
  loss, while every residual, identity mismatch, or negative delta adds
  `archive_operational_mismatch` to `reasons`.

## Schema and test-fixture changes

The required archive schema now includes `in_channel`, `out_channel`, and
`resolved_time_ns`. The required operational `forwards` schema now includes
`resolved_time`. Fixture DDL and all fixture inserts use explicit column lists.

Regression coverage includes exact dedup acceptance; independent mutation of
all seven identity fields; one-msat fee excess; operational count excess;
matching projected count with a residual amount mismatch; required-column
failure; complete coverage; query-plan bounds; and source-byte immutability.

## TDD evidence

The brief's relative command was unavailable because this isolated worktree has
no `.venv` entry:

```text
$ .venv/bin/python -m pytest tests/test_verify_forward_archive.py -q
/bin/bash: line 1: .venv/bin/python: No such file or directory
exit 127
```

The repository virtual environment was then invoked by absolute path.

RED, after test-only changes:

```text
$ /home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_verify_forward_archive.py -q
FFFFFFFFF.FFF..F. [100%]
13 failed, 4 passed in 0.55s
```

Failures were the expected missing result keys, missing overlap status, and
missing `resolved_time` schema contract.

GREEN, after the minimal verifier implementation:

```text
$ /home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_verify_forward_archive.py -q
................. [100%]
17 passed in 0.46s
```

Additional verification:

```text
pyflakes tools/audit/verify_forward_archive.py tests/test_verify_forward_archive.py  # exit 0
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m py_compile tools/audit/verify_forward_archive.py tests/test_verify_forward_archive.py  # exit 0
git diff --check  # exit 0
```

## Query-plan and read-only proof

The copied fixture produced `query_plan_bounded: true`. SQLite reported:

```text
SEARCH forward_archive_v1 USING INDEX idx_forward_archive_v1_status_received (archive_generation=? AND status=? AND received_time_ns>? AND received_time_ns<?)
SEARCH forwards USING INDEX idx_forwards_time (timestamp>? AND timestamp<?)
snapshot_bytes_unchanged: true
```

The verifier still opens only a resolved copied SQLite path with `mode=ro`.
`test_verifier_opens_sqlite_read_only` compares source bytes before and after the
full verifier call. No production database or live RPC was accessed.

## Compatibility and safety

- Output changes are additive; `schema_version` remains 1 and `overlap_equal`
  retains its prior literal meaning.
- A copied database missing the migrated `forwards.resolved_time` column now
  fails closed with `VerificationError` rather than weakening identity.
- No Sling references were introduced.
- No action RPC or direct CLN RPC appears in the changed verifier or tests, and
  no action RPC was triggered.
- Production behavior is unchanged; this is an offline verifier only.

## Commit

Commit message: `fix: reconcile archive with legacy forward identity`.
The final commit hash is reported in the supervisor handoff because a commit
cannot embed its own hash in its tracked report.

## Follow-up risks and concerns

Historical daily rollups do not retain per-forward identity columns. Exact-key
membership can therefore be checked directly for retained raw operational rows;
the rolled component remains independently guarded by exact projected count and
msat totals. Any mismatch in that residual fails closed.
