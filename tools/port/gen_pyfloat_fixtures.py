#!/usr/bin/env python3
"""Generate CPython `repr(float)` parity fixtures for the Rust port.

Run from the repo root (~/bin/cl_revenue_ops-port):

    python3 tools/port/gen_pyfloat_fixtures.py <output.json>

The output feeds `crates/revops-econ/src/pyfloat.rs`'s `py_repr` tests in
the Rust port (cl-revenue-ops-r), committed there as
`fixtures/pyfloat.json`. Python's builtin `repr()` is the source of truth
for every expected string below: nothing here is hand-computed.

Context: `modules/econ_cycle.py`'s `_rebalance_intent` puts
`round(float(score), 6)` into a shadow-cycle `Explanation` component — the
ONE place a float legitimately reaches `CycleResult.to_wire()` /
`.canonical()`. Rust's `revops_core::canonical_json` rejects all
float-typed numbers fail-closed (by design, for idempotency-key safety), so
`CycleResult::canonical()` in the Rust port uses a local canonical writer
that renders float leaves via `pyfloat::py_repr` instead. This fixture pins
that formatter against real CPython float repr semantics.

Coverage (per the Task 8 brief in
docs/superpowers/plans/2026-07-16-phase2-econ-core.md):
- integrals (both signs, including zero/negative zero)
- halves and other short fractional-decimal values
- shortest-round-trip cases (0.1, 0.3, 1/3, ...) that need many digits
- exponent-notation boundaries around 1e16 (upper) and 1e-4 (lower)
- negatives across all of the above
- round(score, 6)-shaped values (what `_rebalance_intent` actually
  produces): both hand-picked and a deterministic pseudo-random batch of
  plausible economic scores, rounded exactly the way the module does.
"""
import json
import random
import sys


def cases():
    values = []

    def add(v):
        values.append(float(v))

    # --- integrals ---
    for v in (0.0, -0.0, 1.0, -1.0, 5.0, -5.0, 100.0, -100.0, 1_000_000.0,
              2.0, 10.0, 42.0):
        add(v)

    # --- halves / short fractional decimals ---
    for v in (0.5, -0.5, 1.5, 2.5, -2.5, 3.5, 100.5, -100.5, 0.25, 0.125,
              1000000.5):
        add(v)

    # --- shortest-round-trip cases (need every significant digit) ---
    for v in (0.1, -0.1, 0.3, -0.3, 1.0 / 3.0, -(1.0 / 3.0), 2.0 / 3.0,
              0.2, 0.7, 1.0 / 7.0, 0.1 + 0.2, 0.30000000000000004):
        add(v)

    # --- exponent boundaries: upper (1e16) ---
    for v in (1e15, 1e16, 9.999e15, 9_999_999_999_999_998.0,
              1.0000000000000002e16, 1e17, -1e16, -9.999e15, 1.0000001e16):
        add(v)

    # --- exponent boundaries: lower (1e-4) ---
    for v in (1e-4, 9.99e-5, 1e-5, 9.9999e-5, -1e-4, -9.99e-5, -1e-5,
              1.0001e-4):
        add(v)

    # --- other magnitude spot-checks (well past both boundaries) ---
    for v in (1e21, 1e100, 1e-100, 1e300, 1e-300, 1.234e20, -1.234e20,
              1.7976931348623157e308, 5e-324):
        add(v)

    # --- negatives generally (beyond the ones already interleaved above) ---
    for v in (-123.456, -0.001, -3.14159265358979):
        add(v)

    # --- round(score, 6)-shaped values: what _rebalance_intent actually
    # writes into the "score" explanation component. Hand-picked ties and
    # near-ties (decimal-looking .5 boundaries that are not exact binary
    # ties) plus a deterministic pseudo-random batch of plausible economic
    # scores (roughly [-1.0, 1.0], as `score`/`final_score_sats`-derived
    # ratios tend to be), all put through the SAME round(x, 6) the module
    # uses.
    hand_picked = (0.1234565, 0.29812345, 2.6755555, 100.0000005,
                   -0.123456789, 0.298, 0.5, 2.675, 1.005, 0.000001,
                   -0.000001, 0.9999995, -0.9999995)
    for v in hand_picked:
        add(round(float(v), 6))

    rng = random.Random(20260716)  # fixed seed: reproducible fixture
    for _ in range(60):
        raw = rng.uniform(-1.0, 1.0) * (10 ** rng.randint(-3, 6))
        add(round(float(raw), 6))

    # De-duplicate while preserving first-seen order (repr is a pure
    # function of value, so duplicates would just be redundant assertions).
    seen = set()
    unique = []
    for v in values:
        key = v.hex()  # exact bit-pattern key, safe for -0.0 vs 0.0 too
        if key not in seen:
            seen.add(key)
            unique.append(v)
    return unique


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <output.json>", file=sys.stderr)
        sys.exit(1)

    out = {
        "repr_cases": [{"value": v, "repr": repr(v)} for v in cases()],
    }
    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {sys.argv[1]} ({len(out['repr_cases'])} cases)")


if __name__ == "__main__":
    main()
