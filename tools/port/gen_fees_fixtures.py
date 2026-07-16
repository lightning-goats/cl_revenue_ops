#!/usr/bin/env python3
"""Generate fee-controller parity fixtures for the Rust port (Phase 4).

Run from the repo root (~/bin/cl_revenue_ops-port, branch `port`):

    python3 tools/port/gen_fees_fixtures.py pyrand <outdir>
    python3 tools/port/gen_fees_fixtures.py mat3 <outdir>

This is the ONE generator every Phase-4 task extends: each suite is a
subcommand writing JSON under <outdir> (the Rust repo commits the output
under `fixtures/fees/<suite>/`). Later tasks add subcommands (thompson,
pid, rails, ...) — do not fork a second generator.

Parity rules (Global Constraints, 2026-07-17 phase4 plan):
- Every float, input or expected, is serialized as CPython `repr(f)`
  strings. The Rust side parses inputs with `str::parse::<f64>()` (exact
  for shortest-round-trip reprs) and compares outputs via
  `revops_econ::pyfloat::py_repr(actual) == expected` — bit-for-bit, no
  epsilon.
- Every suite pins NOW = 1_752_400_000 (same pin as the conformance
  corpus). Suites whose Python paths read `time.time()` monkeypatch it.
- The real module code is the oracle: `modules.fee_controller` static
  methods and CPython's own `random.Random` are called directly; nothing
  here reimplements an algorithm.
"""
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.fee_controller import GaussianThompsonState  # noqa: E402

NOW = 1_752_400_000


def _r(f):
    """CPython repr string for a float (the cross-language wire format)."""
    return repr(float(f))


def _rmat(m):
    return None if m is None else [[_r(x) for x in row] for row in m]


def _rvec(v):
    return [_r(x) for x in v]


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# pyrand: CPython random.Random(seed) stream — random(), gauss(), and the
# gauss_next cached-pair semantics across interleaved call patterns.
# ---------------------------------------------------------------------------

# Seeds per the Task 1 plan step. The last one is
# CycleContext(seed=0).derive_seed("fee-sample") — recomputed here from the
# real module rather than trusted from the plan text (which had a stale
# value); see the assertion below.
def _derive_seed_fee_sample() -> int:
    from modules.cycle_context import CycleContext
    from modules.econ_types import UnixTime
    c = CycleContext(cycle_id="c", cycle_time=UnixTime(NOW), seed=0,
                     snapshot_id="s")
    return c.derive_seed("fee-sample")


# Interleaving patterns pinning the gauss_next cache. Each entry is a list
# of calls; "random" -> r.random(), "gauss" -> r.gauss(0.0, 1.0),
# ["gauss", mu, sigma] -> r.gauss(mu, sigma). Fresh Random per pattern.
INTERLEAVE_PATTERNS = [
    # The plan-mandated pattern: cache filled by first gauss, consumed by
    # second, then a random() call in between the next fill/consume.
    ["random", "gauss", "gauss", "random", "gauss"],
    # Cache-first: gauss opens the stream; random() calls interleave while
    # a cached second value is pending, proving random() does NOT clear it.
    ["gauss", "random", "gauss", "gauss", "random", "random", "gauss",
     "gauss", "gauss"],
    # Non-standard mu/sigma with a pending cache: the cached z is raw
    # (mu + z*sigma applied at consume time, with the CONSUMING call's
    # mu/sigma) — exactly CPython's semantics.
    [["gauss", 200.0, 110.0], ["gauss", 0.0, 1.0], "random",
     ["gauss", 200.0, 110.0], ["gauss", 50.0, 25.0]],
]


def gen_pyrand(outdir: Path) -> None:
    fee_sample_seed = _derive_seed_fee_sample()
    seeds = [0, 1, 42, 2**31 - 1, fee_sample_seed]

    entries = []
    for seed in seeds:
        rnd = random.Random(seed)
        randoms = [_r(rnd.random()) for _ in range(16)]
        rnd = random.Random(seed)
        gausses = [_r(rnd.gauss(0.0, 1.0)) for _ in range(16)]
        interleaved = []
        for pattern in INTERLEAVE_PATTERNS:
            rnd = random.Random(seed)
            calls = []
            values = []
            for call in pattern:
                if call == "random":
                    calls.append({"op": "random"})
                    values.append(_r(rnd.random()))
                elif call == "gauss":
                    calls.append({"op": "gauss", "mu": _r(0.0),
                                  "sigma": _r(1.0)})
                    values.append(_r(rnd.gauss(0.0, 1.0)))
                else:
                    _, mu, sigma = call
                    calls.append({"op": "gauss", "mu": _r(mu),
                                  "sigma": _r(sigma)})
                    values.append(_r(rnd.gauss(mu, sigma)))
            interleaved.append({"calls": calls, "values": values})
        entries.append({
            "seed": seed,
            "random": randoms,
            "gauss": gausses,
            "interleaved": interleaved,
        })

    _write(outdir / "sequences.json", {
        "now": NOW,
        "fee_sample_seed": fee_sample_seed,
        "seeds": entries,
    })


# ---------------------------------------------------------------------------
# mat3: fee_controller.py GaussianThompsonState._mat3_det / _mat3_invert /
# _mat3_vec_mul / _cholesky3 (fee_controller.py lines 468-528).
# ---------------------------------------------------------------------------

def _rand_mat(rnd, lo=-10.0, hi=10.0):
    return [[rnd.uniform(lo, hi) for _ in range(3)] for _ in range(3)]


def _rand_pd(rnd, jitter=0.5):
    """Random positive-definite matrix: B*B^T + jitter*I."""
    b = _rand_mat(rnd, -3.0, 3.0)
    m = [[sum(b[i][k] * b[j][k] for k in range(3)) for j in range(3)]
         for i in range(3)]
    for i in range(3):
        m[i][i] += jitter
    return m


def _posterior_precision_matrices():
    """Run the REAL _recompute_posterior_core on 3 seeded observation sets
    and dump the resulting posterior_precision (Lambda_n)."""
    orig_time = time.time
    time.time = lambda: float(NOW)
    try:
        mats = []
        for seed in (1, 2, 3):
            rnd = random.Random(seed)
            state = GaussianThompsonState()
            obs = []
            for k in range(12):
                fee = rnd.uniform(50.0, 800.0)
                revenue_rate = max(0.0, rnd.gauss(300.0, 120.0))
                weight = rnd.uniform(0.5, 2.0)
                ts = NOW - rnd.randint(0, 96) * 3600
                obs.append((fee, revenue_rate, weight, ts, "all"))
            state.observations = obs
            state._recompute_posterior_core()
            # Sanity: the polynomial branch must have run (fixture intent).
            assert state._last_fee_max > state._last_fee_min, seed
            mats.append((f"posterior_precision_seed{seed}",
                         [row[:] for row in state.posterior_precision]))
        return mats
    finally:
        time.time = orig_time


def _mat3_case_matrices():
    """Named input matrices shared across the det/invert/cholesky suites."""
    cases = [
        ("identity", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        # DTS default prior precision (fee_controller.py line 405-407).
        ("dts_default_prior",
         [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]),
        ("diag_2", [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]),
        # Exactly singular (rank 2: row3 = row1 + row2).
        ("singular_rank2",
         [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [5.0, 7.0, 9.0]]),
        ("zero", [[0.0] * 3, [0.0] * 3, [0.0] * 3]),
        # Relative-tolerance branch pins. max_elem = 100 => tol =
        # 1e-10 * 100^3 = 1e-4. det = 1e4 * x.
        # x = 1e-9 => det = 1e-5 < tol  => None (relative rejects what an
        #                                  ABSOLUTE 1e-10 would accept)
        # x = 1e-7 => det = 1e-3 > tol  => Some
        ("near_singular_below_rel_tol",
         [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1e-9]]),
        ("near_singular_above_rel_tol",
         [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1e-7]]),
        # Same pin at max_elem <= 1 (tol floor = 1e-10 exactly).
        ("small_below_abs_tol",
         [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1e-11]]),
        ("small_above_abs_tol",
         [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1e-9]]),
        # Non-PD (negative pivot) and indefinite-symmetric: invertible but
        # cholesky3 -> None.
        ("non_pd_diag",
         [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
        ("non_pd_late_pivot",
         [[4.0, 2.0, 0.6], [2.0, 1.0000000001, 0.5], [0.6, 0.5, 3.0]]),
        # Asymmetric, well-conditioned.
        ("asymmetric",
         [[3.0, -1.0, 2.0], [0.5, 4.0, -2.5], [1.0, 0.0, 5.0]]),
        # Values shaped like real precision-matrix magnitudes.
        ("precision_shaped",
         [[12.345, 6.789, 3.21], [6.789, 8.9, 4.56], [3.21, 4.56, 7.89]]),
    ]
    cases.extend(_posterior_precision_matrices())
    rnd = random.Random(20260717)
    for i in range(6):
        cases.append((f"random_{i}", _rand_mat(rnd)))
    for i in range(4):
        cases.append((f"random_pd_{i}", _rand_pd(rnd)))
    return cases


def gen_mat3(outdir: Path) -> None:
    mats = _mat3_case_matrices()

    det_cases = []
    invert_cases = []
    cholesky_cases = []
    for name, m in mats:
        det_cases.append({
            "name": name, "m": _rmat(m),
            "expected": _r(GaussianThompsonState._mat3_det(m)),
        })
        invert_cases.append({
            "name": name, "m": _rmat(m),
            "expected": _rmat(GaussianThompsonState._mat3_invert(m)),
        })
        cholesky_cases.append({
            "name": name, "m": _rmat(m),
            "expected": _rmat(GaussianThompsonState._cholesky3(m)),
        })

    matvec_cases = []
    vecs = [
        ("unit_x", [1.0, 0.0, 0.0]),
        ("ones", [1.0, 1.0, 1.0]),
        # phi(f) = [f^2, f, 1] shapes from the regression accumulation.
        ("phi_half", [0.25, 0.5, 1.0]),
        ("phi_norm", [0.6889000000000001, 0.83, 1.0]),
        ("mixed", [-2.5, 300.75, 1e-6]),
    ]
    rnd = random.Random(20260718)
    for i in range(4):
        vecs.append((f"random_v{i}", [rnd.uniform(-50.0, 50.0)
                                      for _ in range(3)]))
    for name, m in mats:
        for vname, v in vecs:
            matvec_cases.append({
                "name": f"{name}__{vname}", "m": _rmat(m), "v": _rvec(v),
                "expected": _rvec(GaussianThompsonState._mat3_vec_mul(m, v)),
            })

    _write(outdir / "det.json", {"now": NOW, "cases": det_cases})
    _write(outdir / "invert.json", {"now": NOW, "cases": invert_cases})
    _write(outdir / "cholesky.json", {"now": NOW, "cases": cholesky_cases})
    _write(outdir / "matvec.json", {"now": NOW, "cases": matvec_cases})


SUITES = {
    "pyrand": gen_pyrand,
    "mat3": gen_mat3,
}


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in SUITES:
        names = "|".join(SUITES)
        print(f"usage: {sys.argv[0]} {{{names}}} <outdir>", file=sys.stderr)
        sys.exit(1)
    SUITES[sys.argv[1]](Path(sys.argv[2]))


if __name__ == "__main__":
    main()
