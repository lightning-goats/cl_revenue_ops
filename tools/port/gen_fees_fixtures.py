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


# ---------------------------------------------------------------------------
# posterior / discount: GaussianThompsonState._recompute_posterior_core /
# _recompute_posterior_legacy / apply_dts_discount (Phase 4 Task 2) + the
# read-only helpers _positive_revenue_mass / _earning_region_fee /
# _effective_positive_rate_ref. The real dataclass methods are the oracle.
# ---------------------------------------------------------------------------

def _dump_obs(obs):
    """[fee_repr, rev_repr, weight_repr, ts, time_bucket, flag?]."""
    out = [_r(obs[0]), _r(obs[1]), _r(obs[2]), obs[3], obs[4]]
    if len(obs) >= 6:
        out.append(obs[5])
    return out


def _posterior_expected(state):
    return {
        "posterior_mean": _r(state.posterior_mean),
        "posterior_std": _r(state.posterior_std),
        "posterior_coeffs": _rvec(state.posterior_coeffs),
        "posterior_precision": _rmat(state.posterior_precision),
        "noise_variance": _r(state.noise_variance),
        "charged_fee_mean": _r(state.charged_fee_mean),
        "last_fee_min": _r(state._last_fee_min),
        "last_fee_max": _r(state._last_fee_max),
    }


def _run_recompute_core(state):
    orig_time = time.time
    time.time = lambda: float(NOW)
    try:
        state._recompute_posterior_core()
    finally:
        time.time = orig_time


def gen_posterior_scenario(name, *, prior_mean_fee=200, prior_std_fee=100,
                            noise_variance=1000.0, zero_revenue_streak=0,
                            zero_run_start_fee=0.0, zero_run_start_ts=0,
                            prior_coeffs=None, prior_precision=None,
                            observations):
    state = GaussianThompsonState()
    state.prior_mean_fee = prior_mean_fee
    state.prior_std_fee = prior_std_fee
    state.noise_variance = noise_variance
    state.zero_revenue_streak = zero_revenue_streak
    state.zero_run_start_fee = zero_run_start_fee
    state.zero_run_start_ts = zero_run_start_ts
    if prior_coeffs is not None:
        state._prior_coeffs = list(prior_coeffs)
    if prior_precision is not None:
        state._prior_precision = [row[:] for row in prior_precision]
    state.observations = list(observations)

    _run_recompute_core(state)

    return {
        "name": name,
        "now": NOW,
        "input": {
            "prior_mean_fee": _r(float(prior_mean_fee)),
            "prior_std_fee": _r(float(prior_std_fee)),
            "prior_coeffs": _rvec(state._prior_coeffs),
            "prior_precision": _rmat(state._prior_precision),
            "noise_variance": _r(noise_variance),
            "zero_revenue_streak": zero_revenue_streak,
            "zero_run_start_fee": _r(float(zero_run_start_fee)),
            "zero_run_start_ts": zero_run_start_ts,
            "observations": [_dump_obs(o) for o in observations],
        },
        "expected": _posterior_expected(state),
    }


def gen_posterior(outdir: Path) -> None:
    scenarios = []

    # 1. Empty observations -> prior reset, charged_fee_mean forced to 0.0.
    scenarios.append(gen_posterior_scenario(
        "empty_observations", prior_mean_fee=250, prior_std_fee=80,
        observations=[]))

    # 2. Every observation excluded by the < 1e-6 weight cutoff (ancient
    # timestamps): empty weighted_obs AND empty anchor_pool -> falls to
    # legacy on an explicit empty list -> prior fallback (total_weight==0).
    ancient = NOW - 400 * 24 * 3600  # ~400 days: decay well below 1e-6.
    scenarios.append(gen_posterior_scenario(
        "all_weight_cutoff_excluded",
        observations=[
            (500.0, 300.0, 1.0, ancient, "all"),
            (520.0, 280.0, 1.0, ancient - 3600, "all"),
        ]))

    # 3. Exactly 3 observations -> minimal polynomial fit (3-parameter fit
    # with zero residual degrees of freedom, sw - 3.0 floored to 1.0).
    scenarios.append(gen_posterior_scenario(
        "exactly_3_obs_minimal_fit",
        observations=[
            (100.0, 50.0, 1.0, NOW - 3600, "all"),
            (300.0, 220.0, 1.0, NOW - 7200, "all"),
            (600.0, 180.0, 1.0, NOW - 1800, "all"),
        ]))

    # 4. 200-observation full buffer (MAX_OBSERVATIONS), deterministic
    # pseudo-random spread of fee/revenue/weight/age.
    rnd = random.Random(90210)
    obs200 = []
    for _ in range(200):
        fee = rnd.uniform(50.0, 900.0)
        rev = max(0.0, rnd.gauss(200.0, 90.0))
        w = rnd.uniform(0.3, 1.5)
        ts = NOW - rnd.randint(0, 160) * 3600
        obs200.append((fee, rev, w, ts, "all"))
    scenarios.append(gen_posterior_scenario(
        "two_hundred_obs_full_buffer", observations=obs200))

    # 5. All-zero-revenue anchor, NOT a streak override: anchor over
    # charged+probed fees (no earning history branch, recency-weighted).
    scenarios.append(gen_posterior_scenario(
        "zero_revenue_anchor_no_streak_override",
        zero_revenue_streak=2, zero_run_start_fee=400.0,
        zero_run_start_ts=NOW - 4 * 3600,
        observations=[
            (400.0, 0.0, 1.0, NOW - 3600, "all"),
            (380.0, 0.0, 1.0, NOW - 7200, "all"),
            (360.0, 0.0, 0.8, NOW - 10800, "all"),
        ]))

    # 6. Probe-heavy anchor pool: zero-probe pseudo-observations mixed with
    # zero-revenue real observations. anchor_pool includes the probes
    # (shifting the anchor mean); weighted_obs (and the fit) excludes them.
    scenarios.append(gen_posterior_scenario(
        "probe_heavy_anchor_pool",
        zero_revenue_streak=5, zero_run_start_fee=400.0,
        zero_run_start_ts=NOW - 5 * 3600,
        observations=[
            (400.0, 0.0, 1.0, NOW - 3600, "all"),
            (360.0, 0.0, 1.0, NOW - 5400, "all", "zero_probe"),
            (324.0, 0.0, 1.0, NOW - 7200, "all", "zero_probe"),
            (291.6, 0.0, 1.0, NOW - 9000, "all", "zero_probe"),
        ]))

    # 7. Streak override WITH earning history elsewhere -> earning_anchor
    # branch (anchors on the revenue-mass-weighted fee, not the dead run).
    scenarios.append(gen_posterior_scenario(
        "streak_override_with_earning_history",
        zero_revenue_streak=30, zero_run_start_fee=900.0,
        zero_run_start_ts=NOW - 30 * 3600,
        observations=[
            (200.0, 500.0, 1.0, NOW - 200 * 3600, "all"),
            (220.0, 480.0, 1.0, NOW - 190 * 3600, "all"),
            (210.0, 510.0, 1.0, NOW - 180 * 3600, "all"),
            (900.0, 0.0, 1.0, NOW - 3600, "all"),
            (900.0, 0.0, 1.0, NOW - 7200, "all"),
        ]))

    # 8. Streak override, NO earning history anywhere -> anchor_all with
    # the ts >= zero_run_start_ts filter (an older pre-run fee excluded).
    scenarios.append(gen_posterior_scenario(
        "streak_override_no_earning_history",
        zero_revenue_streak=30, zero_run_start_fee=900.0,
        zero_run_start_ts=NOW - 10 * 3600,
        observations=[
            (900.0, 0.0, 1.0, NOW - 3600, "all"),
            (850.0, 0.0, 1.0, NOW - 7200, "all"),
            (800.0, 0.0, 1.0, NOW - 9000, "all"),
            (100.0, 0.0, 1.0, NOW - 50 * 3600, "all"),
        ]))

    # 9. Narrow fee range (< 5 ppm) -> legacy fallback via the range guard.
    scenarios.append(gen_posterior_scenario(
        "narrow_range_to_legacy",
        observations=[
            (500.0, 120.0, 1.0, NOW - 3600, "all"),
            (502.0, 130.0, 1.0, NOW - 7200, "all"),
            (503.5, 110.0, 1.0, NOW - 10800, "all"),
        ]))

    # 9b. fee_range exactly at the 5.0 boundary -> guard is strict `< 5.0`,
    # so this must NOT fall to legacy (proves the boundary is inclusive of
    # the polynomial path).
    scenarios.append(gen_posterior_scenario(
        "fee_range_exactly_at_boundary",
        observations=[
            (500.0, 100.0, 1.0, NOW - 3600, "all"),
            (505.0, 150.0, 1.0, NOW - 7200, "all"),
            (502.5, 120.0, 1.0, NOW - 10800, "all"),
        ]))

    # 10. Singular Ln -> legacy: zero out the fixed prior precision and
    # collapse the observations onto only 2 distinct fee values so the phi
    # vectors span < 3 dimensions (Vandermonde-rank-deficient).
    scenarios.append(gen_posterior_scenario(
        "singular_fit_to_legacy",
        prior_precision=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        observations=[
            (100.0, 50.0, 1.0, NOW - 3600, "all"),
            (100.0, 55.0, 1.0, NOW - 7200, "all"),
            (300.0, 90.0, 1.0, NOW - 10800, "all"),
            (300.0, 95.0, 1.0, NOW - 14400, "all"),
        ]))

    # 11. Concave fit (a < -1e-8), un-clamped optimum. Verified against the
    # real oracle: posterior_coeffs a=-4.456... (< -1e-8), f_star=-0.297 in
    # normalized units — inside [-0.5, 1.5], so the clamp is a no-op here.
    concave_unclamped_obs = [
        (100.0, 400.0, 1.0, NOW - 1 * 3600, "all"),
        (110.0, 398.0, 1.0, NOW - 2 * 3600, "all"),
        (500.0, 10.0, 1.0, NOW - 3 * 3600, "all"),
    ]
    scenarios.append(gen_posterior_scenario(
        "concave_fit_unclamped", observations=concave_unclamped_obs))

    # 12. Concave fit clamped at the extrapolation boundary: a low
    # noise_variance (heavier data weighting relative to the fixed prior)
    # pushes the fitted peak past the tested range. Verified against the
    # real oracle: a=-86.9... (< -1e-8), raw f_star=-1.506 -> clamped to
    # -0.5 -> posterior_mean = -0.5*300 + 100 = -50.0.
    scenarios.append(gen_posterior_scenario(
        "concave_fit_clamped_low", noise_variance=10.0,
        observations=[
            (100.0, 500.0, 1.0, NOW - 1 * 3600, "all"),
            (200.0, 260.0, 1.0, NOW - 2 * 3600, "all"),
            (300.0, 140.0, 1.0, NOW - 3 * 3600, "all"),
            (400.0, 60.0, 1.0, NOW - 4 * 3600, "all"),
        ]))

    # 13. Non-concave (a >= -1e-8): monotonically increasing revenue picks
    # the bucket-LCB path.
    scenarios.append(gen_posterior_scenario(
        "non_concave_bucket_lcb",
        observations=[
            (100.0, 50.0, 1.0, NOW - 1 * 3600, "all"),
            (150.0, 80.0, 1.0, NOW - 2 * 3600, "all"),
            (200.0, 120.0, 1.0, NOW - 3 * 3600, "all"),
            (260.0, 150.0, 1.0, NOW - 4 * 3600, "all"),
            (320.0, 210.0, 1.0, NOW - 5 * 3600, "all"),
            (400.0, 260.0, 1.0, NOW - 6 * 3600, "all"),
        ]))

    # 14. Non-concave with a lone high-variance whale window inside one
    # bucket: proves the LCB (not the raw mean) drives bucket selection.
    scenarios.append(gen_posterior_scenario(
        "non_concave_whale_lcb_guard",
        observations=[
            (100.0, 60.0, 1.0, NOW - 1 * 3600, "all"),
            (100.0, 55.0, 1.0, NOW - 2 * 3600, "all"),
            (100.0, 65.0, 1.0, NOW - 3 * 3600, "all"),
            (400.0, 20.0, 1.0, NOW - 4 * 3600, "all"),
            (400.0, 5000.0, 1.0, NOW - 5 * 3600, "all"),
        ]))

    # 15. Mixed-age decay weighting across a wide age spread.
    scenarios.append(gen_posterior_scenario(
        "mixed_age_decay_weighting",
        observations=[
            (150.0, 100.0, 1.0, NOW - 1 * 3600, "all"),
            (250.0, 130.0, 1.0, NOW - 48 * 3600, "all"),
            (350.0, 90.0, 0.5, NOW - 120 * 3600, "all"),
        ]))

    # 16. Custom (non-default) fixed prior coefficients/precision feeding a
    # normal polynomial fit — proves the FIXED prior (not stored posterior)
    # anchors the accumulation.
    scenarios.append(gen_posterior_scenario(
        "custom_fixed_prior",
        prior_coeffs=[0.1, 0.9, 5.0],
        prior_precision=[[0.02, 0.0, 0.0], [0.0, 0.015, 0.0],
                          [0.0, 0.0, 0.03]],
        observations=[
            (120.0, 70.0, 1.0, NOW - 3600, "all"),
            (240.0, 150.0, 1.0, NOW - 7200, "all"),
            (360.0, 180.0, 1.0, NOW - 10800, "all"),
            (480.0, 140.0, 1.0, NOW - 14400, "all"),
        ]))

    # 17. High existing noise_variance feeding sigma2 = max(10, noise_variance).
    scenarios.append(gen_posterior_scenario(
        "high_prior_noise_variance", noise_variance=50000.0,
        observations=concave_unclamped_obs))

    # 18. Congestion-flagged observations are real market windows and stay
    # in the fit (unlike zero-probes).
    scenarios.append(gen_posterior_scenario(
        "congestion_flagged_observations_included",
        observations=[
            (150.0, 90.0, 1.0, NOW - 3600, "all"),
            (300.0, 200.0, 1.0, NOW - 7200, "all", "congestion"),
            (450.0, 160.0, 1.0, NOW - 10800, "all"),
        ]))

    # 19. Legacy path reached with a NON-empty weighted_obs (< 3 points):
    # 2 real observations after excluding one zero-probe.
    scenarios.append(gen_posterior_scenario(
        "legacy_with_two_real_observations",
        observations=[
            (500.0, 120.0, 1.0, NOW - 3600, "all"),
            (620.0, 95.0, 1.0, NOW - 7200, "all"),
            (700.0, 0.0, 1.0, NOW - 10800, "all", "zero_probe"),
        ]))

    # 20. Legacy path with total_weight <= 0.1 (tiny surviving weights) ->
    # prior fallback branch inside the legacy function itself.
    scenarios.append(gen_posterior_scenario(
        "legacy_tiny_total_weight",
        prior_mean_fee=300, prior_std_fee=120,
        observations=[
            (500.0, 120.0, 1.0, NOW - 30 * 24 * 3600, "all"),
            (520.0, 95.0, 1.0, NOW - 30 * 24 * 3600 - 60, "all"),
        ]))

    _write(outdir / "recompute.json", {"now": NOW, "cases": scenarios})

    # -----------------------------------------------------------------
    # helpers.json: direct pins for _positive_revenue_mass /
    # _earning_region_fee / _effective_positive_rate_ref, including the
    # whale-window winsorization branch (>=4 masses, cap at 3x median).
    # -----------------------------------------------------------------
    mass_cases = []

    def _mass_case(name, observations):
        state = GaussianThompsonState()
        state.observations = list(observations)
        masses = state._positive_revenue_mass(NOW)
        earning = state._earning_region_fee(NOW)
        mass_cases.append({
            "name": name,
            "now": NOW,
            "observations": [_dump_obs(o) for o in observations],
            "positive_revenue_mass": [[_r(f), _r(m)] for f, m in masses],
            "earning_region_fee": _r(earning) if earning is not None else None,
        })

    _mass_case("basic_excludes_zero_and_congestion", [
        (100.0, 50.0, 1.0, NOW - 3600, "all"),
        (200.0, 0.0, 1.0, NOW - 7200, "all"),
        (300.0, 80.0, 1.0, NOW - 10800, "all", "congestion"),
    ])
    _mass_case("earning_region_fee_none", [
        (100.0, 0.0, 1.0, NOW - 3600, "all"),
    ])
    _mass_case("whale_window_winsorized", [
        (100.0, 50.0, 1.0, NOW - 3600, "all"),
        (150.0, 60.0, 1.0, NOW - 3600, "all"),
        (200.0, 55.0, 1.0, NOW - 3600, "all"),
        (250.0, 5000.0, 1.0, NOW - 3600, "all"),
    ])
    _mass_case("below_min_weight_excluded", [
        (100.0, 1e-9, 1e-9, NOW - 3600, "all"),
        (200.0, 50.0, 1.0, NOW - 7200, "all"),
    ])
    _mass_case("decayed_across_multiple_ages", [
        (100.0, 50.0, 1.0, NOW - 24 * 3600, "all"),
        (200.0, 80.0, 1.0, NOW - 168 * 3600, "all"),
        (300.0, 120.0, 1.0, NOW - 336 * 3600, "all"),
    ])

    rate_cases = []
    for name, ref, ref_ts, now in [
        ("zero_ref", 0.0, 0, NOW),
        ("negative_ts_guard", 500.0, -1, NOW),
        ("fresh", 500.0, NOW - 3600, NOW),
        ("one_week_decay", 500.0, NOW - 168 * 3600, NOW),
        ("two_week_decay", 500.0, NOW - 336 * 3600, NOW),
    ]:
        state = GaussianThompsonState()
        state.positive_rate_ref = ref
        state.positive_rate_ref_ts = ref_ts
        rate_cases.append({
            "name": name,
            "positive_rate_ref": _r(ref),
            "positive_rate_ref_ts": ref_ts,
            "now": now,
            "expected": _r(state._effective_positive_rate_ref(now)),
        })

    _write(outdir / "helpers.json", {
        "now": NOW,
        "positive_revenue_mass_cases": mass_cases,
        "effective_positive_rate_ref_cases": rate_cases,
    })


def gen_discount(outdir: Path) -> None:
    """apply_dts_discount order-of-operations proof: pin every field after
    EACH step of (recompute, discount, discount again, recompute)."""

    def snapshot(state):
        return {
            "posterior_mean": _r(state.posterior_mean),
            "posterior_std": _r(state.posterior_std),
            "posterior_coeffs": _rvec(state.posterior_coeffs),
            "posterior_precision": _rmat(state.posterior_precision),
            "noise_variance": _r(state.noise_variance),
            "observation_weights": [_r(o[2]) for o in state.observations],
        }

    def run_sequence(name, observations, gammas):
        state = GaussianThompsonState()
        state.observations = list(observations)
        orig_time = time.time
        time.time = lambda: float(NOW)
        steps = []
        try:
            state._recompute_posterior_core()
            steps.append({"op": "recompute", "state": snapshot(state)})
            for gamma in gammas:
                state.apply_dts_discount(gamma)
                steps.append({
                    "op": "discount", "gamma": _r(gamma),
                    "state": snapshot(state),
                })
            state._recompute_posterior_core()
            steps.append({"op": "recompute", "state": snapshot(state)})
        finally:
            time.time = orig_time
        return {
            "name": name,
            "now": NOW,
            "observations": [_dump_obs(o) for o in observations],
            "steps": steps,
        }

    base_obs = [
        (100.0, 50.0, 1.0, NOW - 1 * 3600, "all"),
        (200.0, 140.0, 1.0, NOW - 2 * 3600, "all"),
        (300.0, 190.0, 1.0, NOW - 3 * 3600, "all"),
        # Already at/under DISCOUNT_WEIGHT_FLOOR (0.05): must never rise.
        (400.0, 160.0, 0.05, NOW - 4 * 3600, "all"),
        (500.0, 90.0, 0.02, NOW - 5 * 3600, "all"),
    ]

    sequences = [
        run_sequence("standard_gamma_098_twice", base_obs, [0.98, 0.98]),
        run_sequence("sparse_gamma_0992_twice", base_obs, [0.992, 0.992]),
        run_sequence("fast_gamma_095_twice", base_obs, [0.95, 0.95]),
        run_sequence("mixed_gammas", base_obs, [0.98, 0.992, 0.95]),
        # gamma outside (0,1): apply_dts_discount is a documented no-op.
        run_sequence("gamma_noop_guard", base_obs, [1.0, 0.0, -0.5]),
    ]

    _write(outdir / "sequences.json", {"now": NOW, "sequences": sequences})


SUITES = {
    "pyrand": gen_pyrand,
    "mat3": gen_mat3,
    "posterior": gen_posterior,
    "discount": gen_discount,
}


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in SUITES:
        names = "|".join(SUITES)
        print(f"usage: {sys.argv[0]} {{{names}}} <outdir>", file=sys.stderr)
        sys.exit(1)
    SUITES[sys.argv[1]](Path(sys.argv[2]))


if __name__ == "__main__":
    main()
