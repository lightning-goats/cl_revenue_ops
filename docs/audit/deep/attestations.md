# Deep-Audit Attestations

Per P1, a chunk is COVERED without a finding only by a **structured
attestation**. A bare "clean" is rejected by the refuter (P2). Each attestation
block MUST contain, for its chunk:

1. **chunk_id** — must match a row in `coverage-manifest.md`.
2. A **3-5 line control-flow / state-write summary** — what the code in the
   chunk does, and specifically what external state it writes (DB rows, files,
   RPC calls, in-memory shared caches). Pure/no-write chunks say so explicitly.
3. The **invariant checklist applied** — tick which invariants you actively
   checked (msat/sat rounding direction, sign/off-by-one, budget/cap arithmetic,
   None/missing handling, SQL/transaction boundaries, AGENTS.md never-call
   compliance, concurrency/shared-state). Mark N/A where a class cannot apply.
4. The **single most-suspicious line + why it is acceptable** — the one line you
   would flag if forced to pick, and the concrete argument for why it is safe.

## Format

Use a level-3 heading whose first token is the exact `chunk_id`, then the four
required fields as a bullet list. The coverage tool reads the `chunk_id:` field
(preferred) or the heading token. Any block containing the word `EXAMPLE` is
ignored by `deep_manifest.py --coverage`, so the worked example below never
counts toward real coverage.

```
### <chunk_id>
- chunk_id: <chunk_id>
- summary: <3-5 lines of control flow / state writes>
- invariants: [rounding: OK, sign: OK, caps: N/A, none-handling: OK,
  sql: N/A, never-call: OK, concurrency: N/A]
- most_suspicious: L<NNN> — <the line> — <why acceptable>
- auditor: <name/agent>  date: <YYYY-MM-DD>
```

---

## EXAMPLE (worked, non-counting)

### EXAMPLE modules/utils.py#1
- chunk_id: modules/utils.py#1
- summary: Pure helper module (lines 1-114). `normalize_scid` maps ':'→'x';
  `parse_msat` coerces heterogeneous msat representations to int; the
  `base_to_sats_*` family converts between msat base units and sats with an
  explicit rounding direction; module tail defines constants and backward-compat
  aliases. **No I/O, no DB, no RPC, no shared mutable state** — only a
  module-level logger used for debug messages.
- invariants: [rounding: OK — `base_to_sats_ceil` uses `-(-base // 1000)` so
  fees/budgets round UP (never undercharge); `base_to_sats_floor` rounds DOWN for
  spendable balances; `base_delta_to_sats_toward_zero` handles the signed case;
  matches README rounding contract], [sign: OK — negative deltas explicitly
  handled at L93-95], [caps: N/A], [none-handling: OK — `parse_msat(None)`→0,
  `normalize_scid(None)`→""], [sql: N/A], [never-call: OK — no RPC], [concurrency:
  N/A — stateless, thread-safe by construction]
- most_suspicious: L42 `if isinstance(msat_val, (int, float))` — a bool would
  match `int` and coerce (True→1), but L39-41 intercepts bool first and returns
  0 (the "U-1 FIX"), so the ordering makes this acceptable.
- auditor: EXAMPLE  date: 2026-07-01

---

## Attestations

<!-- Real attestation blocks go below this line. -->
