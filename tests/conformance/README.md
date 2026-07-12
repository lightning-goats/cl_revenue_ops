# Cross-language conformance corpus (refactor Phase 0 layout, J5)

Layout per scenario (refactor.md lines 700–709):

    scenarios/<scenario-name>/
      snapshot.json                # required from Phase 1
      config.json                  # resolved cycle config
      cycle-context.json           # injected clock/seed
      expected-intents.json        # added as Workstream B lands
      expected-arbitration.json    # Workstream C
      expected-authorizations.json # Workstream D
      expected-projections.json    # Workstream E/I

Rules:

- Every payload declares `schema_name` + `schema_version` and validates
  against `schemas/` via `tools/conformance/validate_fixtures.py`
  (standalone; no plugin imports) — run in CI from Phase 1 onward.
- No live credentials, tokens, or unsanitized production identifiers.
- Comparison contract (Phase 1+): exact for integers, enums, ordering,
  reason codes, lifecycle, authorization outcomes; human-readable text
  and `_diag` fields excluded.
- Phase 0 ships only the layout + one smoke scenario; production-derived
  scenarios are captured during Phase 1 golden-parity work.
