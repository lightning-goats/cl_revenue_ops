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
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.config import Config  # noqa: E402
from modules.fee_controller import (  # noqa: E402
    FeeController,
    GaussianThompsonState,
    PIDState,
)

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
# rails: FEE_PROFILES tables + the pure rail stages (Phase 4 Task 5) —
# _resolve_fee_profile, _get_fee_step_cap, _get_target_blend_ratio,
# _blend_fee_target, _get_exploration_fee_target, _zero_flow_streak_thresholds,
# _apply_zero_flow_ratchet_guard, _kalman_demand_factor,
# _exploration_std_threshold. The real FeeController methods are the oracle;
# nothing here reimplements the algorithm.
# ---------------------------------------------------------------------------

def _controller() -> FeeController:
    return FeeController(MagicMock(), MagicMock(spec=Config), MagicMock())


def _profile_settings_to_dict(settings) -> dict:
    d = settings.to_dict()
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = _r(v)
        else:
            out[k] = v
    return out


def gen_rails(outdir: Path) -> None:
    fc = _controller()

    # --- profiles.json: both FEE_PROFILES tables + name-resolution fallback.
    profile_cases = []
    for raw_name in ("active", "ACTIVE", "Active", "conservative",
                     "CONSERVATIVE", "bogus", "", "active "):
        cfg = SimpleNamespace(fee_profile=raw_name)
        resolved_name, settings = fc._resolve_fee_profile(cfg)
        profile_cases.append({
            "input_name": raw_name,
            "resolved_name": resolved_name,
            "settings": _profile_settings_to_dict(settings),
        })
    _write(outdir / "profiles.json", {"now": NOW, "cases": profile_cases})

    # --- step_cap.json: fee_step_cap boundary + ceil-rounding vectors.
    step_cap_cases = []
    for current, woke, profile_name in [
        (1, False, "active"),
        (1, True, "active"),
        (0, False, "active"),
        (-5, False, "active"),
        (100, False, "conservative"),   # exact half-integer boundary (25.0)
        (101, False, "conservative"),   # 25.25 -> ceil 26
        (201, False, "conservative"),   # 50.25 -> ceil 51
        (200, False, "conservative"),   # 50.0 exact
        (1000, False, "active"),        # scaled (500) >> min_delta (100)
        (1000, True, "active"),         # wake ratio 0.20 -> 200
        (5, False, "conservative"),     # scaled below min_delta -> min wins
        (3, True, "conservative"),      # wake: ceil(3*0.10)=1 -> min 10 wins
    ]:
        cfg = SimpleNamespace(fee_profile=profile_name)
        cap = fc._get_fee_step_cap(current, woke, cfg=cfg)
        step_cap_cases.append({
            "current_fee_ppm": current, "woke_from_sleep": woke,
            "profile_name": profile_name, "expected": cap,
        })
    _write(outdir / "step_cap.json", {"now": NOW, "cases": step_cap_cases})

    # --- blend.json: target_blend_ratio band edges + blend_fee_target
    # (incl. the +-1 minimum-step rule both directions).
    blend_ratio_cases = []
    for std in (300.0, 200.0, 199.999999, 150.0, 100.0, 99.999999,
                75.0, 50.0, 49.999999, 10.0, 0.0):
        for woke in (False, True):
            for profile_name in ("active", "conservative"):
                cfg = SimpleNamespace(fee_profile=profile_name)
                ratio = fc._get_target_blend_ratio(
                    woke_from_sleep=woke, sparse_data_conservative=False,
                    posterior_std=std, cfg=cfg)
                blend_ratio_cases.append({
                    "posterior_std": _r(std), "woke_from_sleep": woke,
                    "profile_name": profile_name, "expected": _r(ratio),
                })
    _write(outdir / "blend_ratio.json", {"now": NOW, "cases": blend_ratio_cases})

    blend_target_cases = []
    for (current, bounded_target, woke, sparse, std, profile_name) in [
        (1000, 1003, False, False, 100.0, "active"),   # requested=3, ratio .30 -> round(0.9)=1
        (1000, 997, False, False, 100.0, "active"),    # requested=-3 -> round(-0.9)=-1
        (1000, 1001, False, False, 100.0, "active"),   # requested=1 -> round(0.3)=0 -> +-1 min-step (+1)
        (1000, 999, False, False, 100.0, "active"),    # requested=-1 -> round(-0.3)=0 -> min-step (-1)
        (1000, 1000, False, False, 100.0, "active"),   # requested=0 -> stays 0, no min-step
        (500, 1500, True, False, 250.0, "active"),     # woke caps ratio to wake_target_blend_ratio
        (500, 1500, False, True, 30.0, "conservative"),  # sparse flag plumbed to diag only
        (2000, 100, False, False, 40.0, "active"),     # tight posterior, large cut
    ]:
        cfg = SimpleNamespace(fee_profile=profile_name)
        blended, diag = fc._blend_fee_target(
            current, bounded_target, woke, sparse, posterior_std=std, cfg=cfg)
        blend_target_cases.append({
            "current_fee_ppm": current, "bounded_target_ppm": bounded_target,
            "woke_from_sleep": woke, "sparse_data_conservative": sparse,
            "posterior_std": _r(std), "profile_name": profile_name,
            "expected_blended_target_ppm": blended,
            "expected_diag": {
                "blend_ratio": _r(diag["blend_ratio"]),
                "blended_delta_ppm": diag["blended_delta_ppm"],
                "sparse_data_conservative": diag["sparse_data_conservative"],
            },
        })
    _write(outdir / "blend_target.json", {"now": NOW, "cases": blend_target_cases})

    # --- exploration.json: floor-pin, sparse halving, ceil/headroom/discount
    # interplay (8+ vectors).
    exploration_cases = []
    for (current, floor_ppm, cfg_min, sparse, effective_min) in [
        (50, 100, 10, False, None),      # current <= floor -> exploration_floor
        (100, 100, 10, False, None),     # current == floor (<=) -> floor
        (1000, 100, 10, False, None),    # headroom candidate wins under discount ceiling
        (1000, 100, 10, True, None),     # sparse halving of discount + wider headroom ratio
        (110, 100, 10, False, None),     # near floor: discounted ceiling clamps to floor
        (300, 100, 10, False, None),     # discounted ceiling binds above floor
        (1000, 50, 10, False, 200),      # effective_min_fee_ppm overrides cfg floor
        (10, 4, 1, False, None),         # small integers exercise ceil rounding
        (10, 4, 1, True, None),          # small integers, sparse
        (750, 300, 10, False, None),     # mid-range, non-sparse
    ]:
        cfg = SimpleNamespace(min_fee_ppm=cfg_min)
        target = fc._get_exploration_fee_target(
            current, floor_ppm, cfg, sparse, effective_min_fee_ppm=effective_min)
        exploration_cases.append({
            "current_fee_ppm": current, "floor_ppm": floor_ppm,
            "cfg_min_fee_ppm": cfg_min, "sparse_data_conservative": sparse,
            "effective_min_fee_ppm": effective_min, "expected": target,
        })
    _write(outdir / "exploration.json", {"now": NOW, "cases": exploration_cases})

    # --- zero_flow_thresholds.json: cadence scaling (gap-cap, downshift>=guard).
    zft_cases = []
    for gap, cycle in [
        (0.0, 0.5), (-5.0, 0.5), (48.0, 0.5), (float("nan"), 0.5),
        (48.0, 0.0), (48.0, float("nan")), (1000.0, 0.5), (1.0, 1.0),
        (168.0, 0.5), (200.0, 0.5),
    ]:
        guard, downshift = fc._zero_flow_streak_thresholds(gap, cycle)
        zft_cases.append({
            "gap_ema_hours": _r(gap), "cycle_hours": _r(cycle),
            "expected_guard": guard, "expected_downshift": downshift,
        })
    _write(outdir / "zero_flow_thresholds.json", {"now": NOW, "cases": zft_cases})

    # --- zero_flow_guard.json: all three tags, hold-vs-downshift interval
    # arithmetic (thresh / thresh+11 / thresh+12 @ interval 12), trickle
    # reclassification, floor-override raise, custom guard/downshift streaks.
    zfg_cases = []

    def _guard_case(label, **kwargs):
        applied, tag = fc._apply_zero_flow_ratchet_guard(**kwargs)
        zfg_cases.append({
            "label": label,
            "inputs": {
                "current_fee": kwargs["current_fee"],
                "target_fee": kwargs["target_fee"],
                "min_fee": kwargs["min_fee"],
                "zero_revenue_streak": kwargs["zero_revenue_streak"],
                "forwards_since_update": kwargs["forwards_since_update"],
                "revenue_rate": _r(kwargs["revenue_rate"]),
                "supported_fee_ceiling": (
                    _r(kwargs["supported_fee_ceiling"])
                    if kwargs.get("supported_fee_ceiling") is not None else None),
                "earning_anchor_ppm": (
                    _r(kwargs["earning_anchor_ppm"])
                    if kwargs.get("earning_anchor_ppm") is not None else None),
                "guard_streak": kwargs.get("guard_streak"),
                "downshift_streak": kwargs.get("downshift_streak"),
                "rate_is_meaningful": kwargs.get("rate_is_meaningful"),
            },
            "expected_fee": applied,
            "expected_tag": tag,
        })

    _guard_case("passthrough_nonzero_rate", current_fee=500, target_fee=800,
                min_fee=10, zero_revenue_streak=20, forwards_since_update=0,
                revenue_rate=1.5)
    _guard_case("passthrough_nonzero_forwards", current_fee=500, target_fee=800,
                min_fee=10, zero_revenue_streak=20, forwards_since_update=3,
                revenue_rate=0.0)
    _guard_case("passthrough_streak_below_guard", current_fee=500,
                target_fee=800, min_fee=10, zero_revenue_streak=2,
                forwards_since_update=0, revenue_rate=0.0)
    _guard_case("hold_at_guard_thresh_lower_target", current_fee=500,
                target_fee=300, min_fee=10, zero_revenue_streak=8,
                forwards_since_update=0, revenue_rate=0.0)
    _guard_case("hold_at_guard_thresh_higher_target_blocked",
                current_fee=500, target_fee=900, min_fee=10,
                zero_revenue_streak=8, forwards_since_update=0,
                revenue_rate=0.0)
    _guard_case("floor_override_raise", current_fee=50, target_fee=40,
                min_fee=100, zero_revenue_streak=8, forwards_since_update=0,
                revenue_rate=0.0)
    _guard_case("downshift_step_at_thresh", current_fee=1000, target_fee=1200,
                min_fee=10, zero_revenue_streak=24, forwards_since_update=0,
                revenue_rate=0.0)
    _guard_case("hold_at_thresh_plus_11", current_fee=1000, target_fee=1200,
                min_fee=10, zero_revenue_streak=35, forwards_since_update=0,
                revenue_rate=0.0)
    _guard_case("downshift_step_at_thresh_plus_12", current_fee=1000,
                target_fee=1200, min_fee=10, zero_revenue_streak=36,
                forwards_since_update=0, revenue_rate=0.0)
    _guard_case("downshift_capped_by_supported_ceiling", current_fee=1000,
                target_fee=1200, min_fee=10, zero_revenue_streak=24,
                forwards_since_update=0, revenue_rate=0.0,
                supported_fee_ceiling=700.0)
    _guard_case("downshift_soft_anchor_floor_absorbs_step", current_fee=1000,
                target_fee=100, min_fee=10, zero_revenue_streak=24,
                forwards_since_update=0, revenue_rate=0.0,
                earning_anchor_ppm=1900.0)
    _guard_case("downshift_anchor_floor_override_raise", current_fee=500,
                target_fee=100, min_fee=10, zero_revenue_streak=24,
                forwards_since_update=0, revenue_rate=0.0,
                earning_anchor_ppm=1900.0)
    _guard_case("trickle_reclassified_as_silence", current_fee=500,
                target_fee=300, min_fee=10, zero_revenue_streak=8,
                forwards_since_update=2, revenue_rate=0.4,
                rate_is_meaningful=False)
    _guard_case("meaningful_rate_not_reclassified", current_fee=500,
                target_fee=300, min_fee=10, zero_revenue_streak=8,
                forwards_since_update=2, revenue_rate=0.4,
                rate_is_meaningful=True)
    _guard_case("custom_guard_streak_used", current_fee=500, target_fee=300,
                min_fee=10, zero_revenue_streak=4, forwards_since_update=0,
                revenue_rate=0.0, guard_streak=4)
    _guard_case("falsy_zero_guard_streak_falls_back_to_default",
                current_fee=500, target_fee=300, min_fee=10,
                zero_revenue_streak=8, forwards_since_update=0,
                revenue_rate=0.0, guard_streak=0)
    _write(outdir / "zero_flow_guard.json", {"now": NOW, "cases": zfg_cases})

    # --- kalman.json: _kalman_demand_factor curve (10 points).
    kalman_cases = []
    for ed in (0.0, 0.01, 0.05, 0.25, 0.4999999, 0.5, 0.75, 1.0, 1.5, 10.0):
        kalman_cases.append({
            "expected_demand": _r(ed),
            "expected": _r(fc._kalman_demand_factor(ed)),
        })
    _write(outdir / "kalman.json", {"now": NOW, "cases": kalman_cases})

    # --- std_threshold.json: _exploration_std_threshold curve (10 points).
    std_threshold_cases = []
    for fee in (0, 1, -50, 500, 2499, 2500, 2501, 5000, 25000, 100000):
        std_threshold_cases.append({
            "current_fee_ppm": fee,
            "expected": _r(fc._exploration_std_threshold(fee)),
        })
    _write(outdir / "std_threshold.json",
           {"now": NOW, "cases": std_threshold_cases})

    # --- unroutable.json: _is_unroutable_zero_window.
    unroutable_cases = []
    for rate, spendable in [
        (0.0, 24999.0), (0.0, 25000.0), (0.0, 25001.0),
        (1.0, 100.0), (-1.0, 0.0), (0.0, 0.0),
    ]:
        unroutable_cases.append({
            "revenue_rate": _r(rate), "spendable_sats": _r(spendable),
            "expected": fc._is_unroutable_zero_window(rate, spendable),
        })
    _write(outdir / "unroutable.json", {"now": NOW, "cases": unroutable_cases})


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


# ---------------------------------------------------------------------------
# pid: PIDState.calculate_multiplier (fee_controller.py:1977-2020), 12
# sequences of calls pinning (multiplier, ewma_error, integral_error) per
# step as repr strings. Phase 4 Task 3.
# ---------------------------------------------------------------------------

def _run_pid_sequence(steps):
    """steps: list of dicts with now/ratio/capacity/flow_state. Returns the
    per-step results list, running the REAL PIDState.calculate_multiplier
    under a monkeypatched time.time() (the method reads time.time()
    internally rather than taking `now` as an argument — clock injection is
    the Rust port's job, per Phase 4 Global Constraints, but the Python
    oracle itself is unchanged)."""
    state = PIDState()
    orig_time = time.time
    results = []
    try:
        for step in steps:
            time.time = lambda t=step["now"]: float(t)
            mult = state.calculate_multiplier(
                step["ratio"], step["capacity"], step["flow_state"]
            )
            results.append({
                "now": step["now"],
                "ratio": _r(step["ratio"]),
                "capacity_sats": step["capacity"],
                "flow_state": step["flow_state"],
                "multiplier": _r(mult),
                "ewma_error": _r(state.ewma_error),
                "integral_error": _r(state.integral_error),
            })
    finally:
        time.time = orig_time
    return results


def gen_pid(outdir: Path) -> None:
    sequences = []

    def seq(name, steps):
        sequences.append({"name": name, "steps": _run_pid_sequence(steps)})

    # 1: flow_state="source" (target 0.7), capacity floor-clamped to 1,
    # varying dt across steps including the first-call dt=0.
    seq("source_capacity_floor", [
        {"now": NOW, "ratio": 0.95, "capacity": 1, "flow_state": "source"},
        {"now": NOW + 3600, "ratio": 0.85, "capacity": 1, "flow_state": "source"},
        {"now": NOW + 3600 * 6, "ratio": 0.6, "capacity": 1, "flow_state": "source"},
    ])

    # 2: flow_state="sink" (target 0.3), capacity 1e6, includes a long dt
    # (48h) that should saturate the integral clamp.
    seq("sink_capacity_1e6_integral_saturation", [
        {"now": NOW, "ratio": 0.9, "capacity": 1_000_000, "flow_state": "sink"},
        {"now": NOW + 3600 * 48, "ratio": 0.95, "capacity": 1_000_000, "flow_state": "sink"},
        {"now": NOW + 3600 * 96, "ratio": 0.97, "capacity": 1_000_000, "flow_state": "sink"},
    ])

    # 3: flow_state="balanced_active", capacity 5e6.
    seq("balanced_active_capacity_5e6", [
        {"now": NOW, "ratio": 0.55, "capacity": 5_000_000, "flow_state": "balanced_active"},
        {"now": NOW + 1800, "ratio": 0.45, "capacity": 5_000_000, "flow_state": "balanced_active"},
    ])

    # 4: flow_state="dormant" (target 0.5, F6), capacity 5e8.
    seq("dormant_capacity_5e8", [
        {"now": NOW, "ratio": 0.5, "capacity": 500_000_000, "flow_state": "dormant"},
        {"now": NOW + 3600 * 12, "ratio": 0.5, "capacity": 500_000_000, "flow_state": "dormant"},
    ])

    # 5: flow_state="congested", capacity floor-clamped, ratio edges 0/1.
    seq("congested_ratio_edges", [
        {"now": NOW, "ratio": 0.0, "capacity": 1, "flow_state": "congested"},
        {"now": NOW + 3600, "ratio": 1.0, "capacity": 1, "flow_state": "congested"},
    ])

    # 6: flow_state="unknown" (explicitly listed key, still 0.5), capacity 1e6.
    seq("flow_state_unknown_key", [
        {"now": NOW, "ratio": 0.3, "capacity": 1_000_000, "flow_state": "unknown"},
        {"now": NOW + 3600 * 3, "ratio": 0.4, "capacity": 1_000_000, "flow_state": "unknown"},
    ])

    # 7: flow_state NOT in the dict at all ("router" — reserved vocabulary
    # the classifier does not emit yet) — exercises the `.get(x, 0.5)`
    # fallback, capacity 5e6.
    seq("flow_state_not_in_dict_router", [
        {"now": NOW, "ratio": 0.6, "capacity": 5_000_000, "flow_state": "router"},
        {"now": NOW + 3600 * 2, "ratio": 0.6, "capacity": 5_000_000, "flow_state": "totally_unknown_state"},
    ])

    # 8: NaN ratio — non-finite input replaced by the target ratio before
    # the raw-error computation, both on the fresh (dt=0) call and a
    # subsequent dt>0 call.
    seq("nan_ratio_guard", [
        {"now": NOW, "ratio": float("nan"), "capacity": 500_000_000, "flow_state": "balanced"},
        {"now": NOW + 3600, "ratio": float("nan"), "capacity": 500_000_000, "flow_state": "balanced"},
        {"now": NOW + 3600 * 2, "ratio": 0.5, "capacity": 500_000_000, "flow_state": "balanced"},
    ])

    # 9: very large dt (1000h) — integral term should still clamp to
    # ±integral_clamp (3.0) rather than overflow.
    seq("very_large_dt_integral_clamp", [
        {"now": NOW, "ratio": 0.99, "capacity": 1, "flow_state": "sink"},
        {"now": NOW + 3600 * 1000, "ratio": 0.99, "capacity": 1, "flow_state": "sink"},
    ])

    # 10: overshoot ratios (negative and >1) drive the multiplier to its
    # hard 0.5/2.0 clamp bounds.
    seq("overshoot_ratio_hits_multiplier_clamp", [
        {"now": NOW, "ratio": -0.5, "capacity": 1_000_000, "flow_state": "balanced"},
        {"now": NOW + 3600, "ratio": 1.5, "capacity": 1_000_000, "flow_state": "balanced"},
    ])

    # 11: flow_state AND capacity both change across steps of the SAME
    # persistent PidState (a channel migrating between flow-classifier
    # regimes over successive cycles).
    seq("flow_state_and_capacity_change_across_steps", [
        {"now": NOW, "ratio": 0.8, "capacity": 1_000_000, "flow_state": "source"},
        {"now": NOW + 3600 * 4, "ratio": 0.4, "capacity": 5_000_000, "flow_state": "sink"},
        {"now": NOW + 3600 * 8, "ratio": 0.5, "capacity": 500_000_000, "flow_state": "balanced"},
    ])

    # 12: the exact conformance scenario 12 case
    # (tests/conformance/scenarios/12-dts-pid-components/case.json) — a
    # single fresh-state (dt=0) call. `round(pid_multiplier, 12) ==
    # 1.026338203439` and `round(pid_ewma_error, 12) == 0.09` per that
    # scenario's `expected` block (generate_scenarios.py rounds to 12
    # digits; the fixture here pins the FULL, unrounded repr, and this
    # assertion is a sanity cross-check that this is the same code path).
    seq("conformance_scenario_12_fresh_state", [
        {"now": NOW, "ratio": 0.2, "capacity": 5_000_000, "flow_state": "balanced"},
    ])

    assert len(sequences) == 12, len(sequences)
    last_step = sequences[-1]["steps"][-1]
    last_mult = float(last_step["multiplier"])
    last_ewma = float(last_step["ewma_error"])
    assert round(last_mult, 12) == 1.026338203439, last_mult
    assert round(last_ewma, 12) == 0.09, last_ewma

    _write(outdir / "sequences.json", {"now": NOW, "sequences": sequences})


# ---------------------------------------------------------------------------
# state_dict: GaussianThompsonState.to_dict/from_dict
# (fee_controller.py:1721-1940), byte-identical round trips through the
# REAL Python json.dumps (default separators (", ", ": "), ensure_ascii).
# Phase 4 Task 3.
# ---------------------------------------------------------------------------

def _state_dict_case_with_unknown(name, base_d, extra_keys):
    """Like `_state_dict_case`, but `extra_keys` are top-level keys Python's
    own `to_dict()` doesn't know about and therefore silently drops. This
    port's contract (Task 3 brief) is to survive them instead, re-emitted
    after all known keys in first-seen order — so `expected` is Python's
    real `from_dict(d).to_dict()` output (the known-fields truth) with
    `extra_keys` manually re-appended in that position, i.e. the CONTRACT
    truth, not the as-is (lossy) Python behavior."""
    d = dict(base_d)
    d.update(extra_keys)
    state = GaussianThompsonState.from_dict(d)
    to_dict = state.to_dict()
    for k, v in extra_keys.items():
        to_dict[k] = v
    return {
        "name": name,
        "blob": json.dumps(d),
        "expected": json.dumps(to_dict),
    }


def _state_dict_case(name, d):
    """`d`: a plain dict fed to GaussianThompsonState.from_dict. Records the
    raw input bytes (`blob`, via json.dumps(d) — NOT the generator's own
    indent=1 wrapper format) and the Python truth for a from_dict/to_dict
    round trip (`expected`, via json.dumps(state.to_dict())) — both using
    Python's DEFAULT json.dumps separators (", ", ": "), the format real
    v2_state_json blobs are written with."""
    state = GaussianThompsonState.from_dict(d)
    return {
        "name": name,
        "blob": json.dumps(d),
        "expected": json.dumps(state.to_dict()),
    }


def gen_state_dict(outdir: Path) -> None:
    cases = []

    # 1: CURRENT-format round trip — a state built via the REAL running
    # code paths (update_posterior, update_contextual, record_posterior_nudge),
    # so `d` is exactly what a live channel's to_dict() would produce
    # (observations with a congestion-flagged 6-tuple, a populated
    # contextual_posteriors 4-tuple entry, a posterior_bias nudge).
    orig_time = time.time
    time.time = lambda: float(NOW)
    try:
        base = GaussianThompsonState()
        base.update_posterior(250, 12.5, 6.0, time_bucket="peak")
        base.update_posterior(400, 4.0, 3.0, time_bucket="low", congested=True)
        base.update_contextual("mid:peak:P", 250, 12.5, time_bucket="peak")
        base.record_posterior_nudge(275.0, 0.4)
        d0 = base.to_dict()
    finally:
        time.time = orig_time
    cases.append(_state_dict_case("current_format_roundtrip", d0))

    # 2: legacy blob — NO "weight_scheme" key at all (predates the marker),
    # so from_dict rescales BOTH a positive-rate window (a) and a
    # zero-rate window (b) to the exposure-only scheme. Weights are
    # repr-pinned separately below (test contract (b)).
    d1 = {
        "prior_mean_fee": 200,
        "prior_std_fee": 100,
        "observations": [
            # Small original weights so the rescale lands well below the
            # min(1.0, ...) clamp — a non-trivial repr-pinned value, not
            # just "both saturate to 1.0".
            [250, 12.5, 0.1, NOW - 3600, "normal"],
            [300, 0.0, 0.05, NOW - 7200, "low"],
        ],
    }
    legacy_case = _state_dict_case("legacy_weight_rescale_absent_marker", d1)
    legacy_state = GaussianThompsonState.from_dict(d1)
    legacy_case["rescaled_weights"] = [_r(o[2]) for o in legacy_state.observations]
    cases.append(legacy_case)

    # 2b: legacy blob with an explicit STALE weight_scheme value (not
    # merely absent) — same rescale path, different trigger.
    d1b = dict(d1)
    d1b["weight_scheme"] = "thompson_aimd_v1"
    cases.append(_state_dict_case("legacy_weight_rescale_stale_marker", d1b))

    # 3: 3-tuple contextual posteriors (legacy layout) alongside a current
    # 4-tuple one, converted on load.
    d2 = {
        "contextual_posteriors": {
            "low:normal:P": [250.0, 45.0, 12],
            "mid:peak:S": [310.0, 22.5, 8, NOW - 1000],
        },
    }
    ctx_case = _state_dict_case("legacy_ctx_3tuple_alongside_current_4tuple", d2)
    ctx_state = GaussianThompsonState.from_dict(d2)
    ctx_case["converted_ctx"] = {
        k: [_r(v[0]), _r(v[1]), v[2], v[3]] for k, v in ctx_state.contextual_posteriors.items()
    }
    cases.append(ctx_case)

    # 4: unknown injected top-level key on an otherwise-current-format
    # blob. Python's own from_dict/to_dict DROPS unknown keys; this port's
    # contract (see `_state_dict_case_with_unknown`) is to survive them,
    # re-emitted after all known keys in first-seen order.
    cases.append(_state_dict_case_with_unknown(
        "unknown_top_level_key_survives_roundtrip",
        d0, {"future_field": {"x": 1}},
    ))

    # 5: a 6-tuple zero_probe-flagged observation alongside a 6-tuple
    # congestion-flagged one, current weight_scheme, round-tripped.
    d4 = {
        "weight_scheme": "exposure_v2",
        "observations": [
            [250, 12.5, 0.8, NOW - 3600, "normal", "congestion"],
            [90, 0.0, 0.5, NOW - 1800, "low", "zero_probe"],
        ],
    }
    cases.append(_state_dict_case("six_tuple_congestion_and_zero_probe", d4))

    # 6: non-ASCII string in a preserved (unknown-key) leaf, to pin
    # dumps_python's ensure_ascii=True escaping end-to-end.
    cases.append(_state_dict_case_with_unknown(
        "non_ascii_unknown_value",
        d0, {"future_alias_note": "café ☃ router-étiquette"},
    ))

    # 7: a legacy float-typed observation fee (thompson_aimd_v1-era blobs
    # may carry `fee_ppm` as a JSON float) alongside a current int-typed fee
    # in the SAME observations list. `from_dict`/`to_dict` never cast
    # observation[0] (just like prior_mean_fee/prior_std_fee — see that
    # struct doc comment), so the JSON int/float typing of each observation
    # fee must survive the round trip independently, byte-identical to
    # Python's own `json.dumps`.
    d5 = {
        "weight_scheme": "exposure_v2",
        "observations": [
            [250.0, 12.5, 1.0, NOW - 3600, "peak"],
            [250, 12.5, 1.0, NOW - 3600, "peak"],
        ],
    }
    cases.append(_state_dict_case("float_and_int_observation_fee_roundtrip", d5))

    _write(outdir / "cases.json", {"now": NOW, "cases": cases})


# ---------------------------------------------------------------------------
# update / ceiling / sampling: Thompson dynamics + sampling paths (Phase 4
# Task 7) — update_posterior/update_contextual/record_posterior_nudge/
# apply_vegas_adjustment sequences with full state snapshots per step,
# supported_fee_ceiling + maybe_upward_probe_cap gate matrices, and seeded
# sample_fee/sample_fee_contextual draw scenarios. The REAL
# GaussianThompsonState methods are the oracle; time.time is monkeypatched
# per step and the global `random` module is seeded per scenario (the Rust
# port injects `now: i64` and `&mut PyRandom` instead).
# ---------------------------------------------------------------------------

H = 3600


def _dump_ctx_value(v):
    """[mean_repr, precision_repr, count, last_update] (current 4-tuple) or
    [mean_repr, std_repr, count] (legacy 3-tuple, preserved as stored)."""
    if len(v) == 3:
        return [_r(v[0]), _r(v[1]), v[2]]
    return [_r(v[0]), _r(v[1]), v[2], v[3]]


def _dump_bias(bias):
    return [[_r(t), _r(w), ts] for t, w, ts in bias]


def _full_snapshot(state):
    """Every GaussianThompsonState field Task 7's dynamics can touch."""
    return {
        "observations": [_dump_obs(o) for o in state.observations],
        "zero_revenue_streak": state.zero_revenue_streak,
        "zero_run_start_fee": _r(state.zero_run_start_fee),
        "zero_run_start_ts": state.zero_run_start_ts,
        "positive_rate_ref": _r(state.positive_rate_ref),
        "positive_rate_ref_ts": state.positive_rate_ref_ts,
        "meaningful_gap_ema_hours": _r(state.meaningful_gap_ema_hours),
        "last_meaningful_ts": state.last_meaningful_ts,
        "last_upward_probe_ts": state.last_upward_probe_ts,
        "posterior_mean": _r(state.posterior_mean),
        "posterior_std": _r(state.posterior_std),
        "posterior_coeffs": _rvec(state.posterior_coeffs),
        "posterior_precision": _rmat(state.posterior_precision),
        "noise_variance": _r(state.noise_variance),
        "charged_fee_mean": _r(state.charged_fee_mean),
        "last_fee_min": _r(state._last_fee_min),
        "last_fee_max": _r(state._last_fee_max),
        "posterior_bias": _dump_bias(state.posterior_bias),
        "contextual_posteriors": [
            [k, _dump_ctx_value(v)]
            for k, v in state.contextual_posteriors.items()
        ],
    }


_INITIAL_INT_FIELDS = (
    "zero_revenue_streak", "zero_run_start_ts", "positive_rate_ref_ts",
    "last_meaningful_ts", "last_upward_probe_ts",
)
_INITIAL_FLOAT_FIELDS = (
    "prior_mean_fee", "prior_std_fee", "posterior_mean", "posterior_std",
    "charged_fee_mean", "noise_variance", "zero_run_start_fee",
    "positive_rate_ref", "meaningful_gap_ema_hours",
)


def _apply_initial(state, initial):
    for key in _INITIAL_INT_FIELDS + _INITIAL_FLOAT_FIELDS:
        if key in initial:
            setattr(state, key, initial[key])
    if "observations" in initial:
        state.observations = [tuple(o) for o in initial["observations"]]
    if "posterior_bias" in initial:
        state.posterior_bias = [tuple(b) for b in initial["posterior_bias"]]
    if "contextual_posteriors" in initial:
        state.contextual_posteriors = {
            k: tuple(v) for k, v in initial["contextual_posteriors"].items()
        }
    if "posterior_coeffs" in initial:
        state.posterior_coeffs = list(initial["posterior_coeffs"])
    if "posterior_precision" in initial:
        state.posterior_precision = [
            row[:] for row in initial["posterior_precision"]
        ]
    if "last_fee_min" in initial:
        state._last_fee_min = initial["last_fee_min"]
    if "last_fee_max" in initial:
        state._last_fee_max = initial["last_fee_max"]


def _dump_initial(initial):
    out = {}
    for k, v in initial.items():
        if k == "observations":
            out[k] = [_dump_obs(o) for o in v]
        elif k == "posterior_bias":
            out[k] = _dump_bias(v)
        elif k == "contextual_posteriors":
            out[k] = [[key, _dump_ctx_value(val)] for key, val in v.items()]
        elif k == "posterior_coeffs":
            out[k] = _rvec(v)
        elif k == "posterior_precision":
            out[k] = _rmat(v)
        elif k in _INITIAL_INT_FIELDS:
            out[k] = v
        else:
            out[k] = _r(v)
    return out


def _run_update_sequence(name, steps, initial=None):
    state = GaussianThompsonState()
    if initial:
        _apply_initial(state, initial)
    out_steps = []
    for step in steps:
        now = step["now"]
        orig_time = time.time
        time.time = lambda t=now: float(t)
        try:
            op = step["op"]
            entry = {"op": op, "now": now}
            if op == "update_posterior":
                state.update_posterior(
                    step["fee"], step["revenue_rate"], step["hours"],
                    time_bucket=step.get("time_bucket", "normal"),
                    congested=step.get("congested", False),
                )
                entry["args"] = {
                    "fee": _r(step["fee"]),
                    "revenue_rate": _r(step["revenue_rate"]),
                    "hours": _r(step["hours"]),
                    "time_bucket": step.get("time_bucket", "normal"),
                    "congested": step.get("congested", False),
                }
            elif op == "nudge":
                state.record_posterior_nudge(step["target_fee"], step["weight"])
                entry["args"] = {
                    "target_fee": _r(step["target_fee"]),
                    "weight": _r(step["weight"]),
                }
            elif op == "vegas":
                state.apply_vegas_adjustment(
                    step["vegas_multiplier"], step["new_floor"])
                entry["args"] = {
                    "vegas_multiplier": _r(step["vegas_multiplier"]),
                    "new_floor": _r(step["new_floor"]),
                }
            elif op == "update_contextual":
                state.update_contextual(
                    step["context_key"], step["fee"], step["revenue_rate"],
                    time_bucket=step.get("time_bucket", "normal"),
                )
                entry["args"] = {
                    "context_key": step["context_key"],
                    "fee": _r(step["fee"]),
                    "revenue_rate": _r(step["revenue_rate"]),
                    "time_bucket": step.get("time_bucket", "normal"),
                }
            elif op == "consume_upward_probe":
                state.consume_upward_probe(now)
                entry["args"] = {}
            elif op == "recompute":
                state._recompute_posterior()
                entry["args"] = {}
            else:
                raise ValueError(op)

            floor_ppm = step.get("ceiling_floor_ppm")
            ceil_v = state.supported_fee_ceiling(now, floor_ppm)
            checks = {
                "real_observation_count": state.real_observation_count(),
                "ceiling_floor_ppm": (
                    _r(floor_ppm) if floor_ppm is not None else None),
                "supported_fee_ceiling": (
                    _r(ceil_v) if ceil_v is not None else None),
            }
            if "meaningful_probe_rates" in step:
                checks["is_meaningful_rate"] = [
                    [_r(r), state.is_meaningful_rate(r, now)]
                    for r in step["meaningful_probe_rates"]
                ]
            entry["expected"] = {
                "checks": checks,
                "state": _full_snapshot(state),
            }
        finally:
            time.time = orig_time
        out_steps.append(entry)
    return {
        "name": name,
        "initial": _dump_initial(initial) if initial else None,
        "steps": out_steps,
    }


def gen_update(outdir: Path) -> None:
    sequences = []

    def seq(name, steps, initial=None):
        sequences.append(_run_update_sequence(name, steps, initial=initial))

    # 1: positive_rate_ref seeding, then EMA-on-DECAYED-ref (py 776-779 blends
    # against the effective/decayed `ref`, NOT the stored positive_rate_ref)
    # and gap-EMA seeding + update.
    seq("meaningful_seeds_and_ema_on_decayed_ref", [
        {"op": "update_posterior", "now": NOW, "fee": 250,
         "revenue_rate": 100.0, "hours": 6.0},
        {"op": "update_posterior", "now": NOW + 48 * H, "fee": 250,
         "revenue_rate": 80.0, "hours": 6.0},
        {"op": "update_posterior", "now": NOW + 72 * H, "fee": 250,
         "revenue_rate": 120.0, "hours": 6.0},
    ])

    # 2: gap-EMA only updates when now > last_meaningful_ts (same-second
    # meaningful windows leave the gap EMA untouched).
    seq("gap_ema_requires_forward_time", [
        {"op": "update_posterior", "now": NOW, "fee": 300,
         "revenue_rate": 50.0, "hours": 3.0},
        {"op": "update_posterior", "now": NOW, "fee": 300,
         "revenue_rate": 60.0, "hours": 3.0},
        {"op": "update_posterior", "now": NOW + 6 * H, "fee": 300,
         "revenue_rate": 55.0, "hours": 6.0},
    ])

    # 3: trickle (rate below TRICKLE_RESET_FRAC of the decayed ref) EXTENDS
    # the streak and stamps zero_run_start; a later meaningful rate resets.
    seq("trickle_extends_streak", [
        {"op": "update_posterior", "now": NOW, "fee": 400,
         "revenue_rate": 500.0, "hours": 6.0,
         "meaningful_probe_rates": [0.0, 20.0, 50.0, 500.0]},
        {"op": "update_posterior", "now": NOW + H, "fee": 400,
         "revenue_rate": 20.0, "hours": 1.0,
         "meaningful_probe_rates": [20.0, 49.0, 51.0]},
        {"op": "update_posterior", "now": NOW + 2 * H, "fee": 400,
         "revenue_rate": 60.0, "hours": 1.0},
    ])

    # 4: the >= boundary — a rate exactly at TRICKLE_RESET_FRAC * ref (age 0,
    # undecayed) is meaningful; just below is a trickle. After step 1 the ref
    # is 100.0 (seeded, ts=now), so at the same now: 10.0 is meaningful
    # (resets), 9.99 would be a trickle (probed via is_meaningful_rate).
    seq("trickle_boundary_exact_fraction", [
        {"op": "update_posterior", "now": NOW, "fee": 200,
         "revenue_rate": 100.0, "hours": 6.0,
         "meaningful_probe_rates": [10.0, 9.99]},
        {"op": "update_posterior", "now": NOW, "fee": 200,
         "revenue_rate": 10.0, "hours": 6.0},
    ])

    # 5: sustained zero run -> probe injection at streak >= 4: probe fee
    # max(1, int(fee*0.9)) appended as ("zero_probe", rev 0.0) with the SAME
    # weight/ts/bucket; real_observation_count excludes probes throughout.
    seq("zero_run_probe_injection_starts", [
        {"op": "update_posterior", "now": NOW + i * H, "fee": 400,
         "revenue_rate": 0.0, "hours": 1.0}
        for i in range(4)
    ] + [
        {"op": "update_posterior", "now": NOW + 4 * H, "fee": 360,
         "revenue_rate": 0.0, "hours": 1.0},
    ])

    # 6: probe stops once posterior_mean falls below ZERO_PROBE_FLOOR_FRAC of
    # the zero-run start fee (descending charged fees drag the zero-regime
    # anchor mean down until the 0.3 * 1000 threshold blocks injection).
    seq("probe_stops_below_floor_frac", [
        {"op": "update_posterior", "now": NOW + i * H, "fee": fee,
         "revenue_rate": 0.0, "hours": 1.0}
        for i, fee in enumerate(
            [1000, 1000, 1000, 1000, 260, 250, 240, 230, 220, 210,
             150, 120, 100, 90, 80, 70])
    ])

    # 7a/7b: probe descent floor is relative to the EARNING anchor when one
    # exists, else the zero-run start fee. Identical stale posterior_mean
    # (250, set directly — the probe gate reads the PREVIOUS cycle's mean,
    # py 821) and zero run started at 900: with recent earning history at
    # ~800 the threshold is 0.3*≈800 ≈ 240 -> probe fires; without it the
    # threshold is 0.3*900 = 270 -> no probe.
    seq("probe_floor_ref_uses_earning_anchor", [
        {"op": "update_posterior", "now": NOW, "fee": 900,
         "revenue_rate": 0.0, "hours": 1.0},
    ], initial={
        "zero_revenue_streak": 3,
        "zero_run_start_fee": 900.0,
        "zero_run_start_ts": NOW - 3 * H,
        "posterior_mean": 250.0,
        "positive_rate_ref": 400.0,
        "positive_rate_ref_ts": NOW - 4 * H,
        "observations": [
            (800, 350.0, 1.0, NOW - 6 * H, "normal"),
            (900, 0.0, 1.0, NOW - 3 * H, "normal"),
            (900, 0.0, 1.0, NOW - 2 * H, "normal"),
            (900, 0.0, 1.0, NOW - H, "normal"),
        ],
    })
    seq("probe_floor_ref_falls_back_to_run_start", [
        {"op": "update_posterior", "now": NOW, "fee": 900,
         "revenue_rate": 0.0, "hours": 1.0},
    ], initial={
        "zero_revenue_streak": 3,
        "zero_run_start_fee": 900.0,
        "zero_run_start_ts": NOW - 3 * H,
        "posterior_mean": 250.0,
        "observations": [
            (900, 0.0, 1.0, NOW - 3 * H, "normal"),
            (900, 0.0, 1.0, NOW - 2 * H, "normal"),
            (900, 0.0, 1.0, NOW - H, "normal"),
        ],
    })

    # 8: probe requires probe_fee < fee — at fee 1, int(0.9) -> max(1, 0) = 1
    # is NOT < 1, so no probe despite an eligible streak.
    seq("probe_requires_strictly_lower_fee", [
        {"op": "update_posterior", "now": NOW, "fee": 1,
         "revenue_rate": 0.0, "hours": 1.0},
    ], initial={
        "zero_revenue_streak": 5,
        "zero_run_start_fee": 400.0,
        "zero_run_start_ts": NOW - 5 * H,
        "posterior_mean": 300.0,
    })

    # 9: congestion-flagged windows are REAL market windows: recorded as
    # 6-tuples, included in real_observation_count, but their revenue is
    # excluded from supported_fee_ceiling (slot protection, not a market
    # test) — the ceiling stays anchored on the un-flagged earning region.
    seq("congestion_counts_but_ceiling_excludes", [
        {"op": "update_posterior", "now": NOW, "fee": 300,
         "revenue_rate": 200.0, "hours": 6.0},
        {"op": "update_posterior", "now": NOW + H, "fee": 2000,
         "revenue_rate": 900.0, "hours": 6.0, "congested": True},
        {"op": "update_posterior", "now": NOW + 2 * H, "fee": 310,
         "revenue_rate": 180.0, "hours": 6.0},
    ])

    # 10: non-finite/non-positive hours default to 1.0 (weight = 1/6).
    seq("guard_bad_hours_defaults_to_one", [
        {"op": "update_posterior", "now": NOW, "fee": 250,
         "revenue_rate": 50.0, "hours": float("nan")},
        {"op": "update_posterior", "now": NOW + H, "fee": 250,
         "revenue_rate": 50.0, "hours": -3.0},
        {"op": "update_posterior", "now": NOW + 2 * H, "fee": 250,
         "revenue_rate": 50.0, "hours": 0.0},
    ])

    # 11: non-finite/negative rate is zeroed (starts a zero run).
    seq("guard_bad_rate_zeroed", [
        {"op": "update_posterior", "now": NOW, "fee": 250,
         "revenue_rate": float("nan"), "hours": 6.0},
        {"op": "update_posterior", "now": NOW + H, "fee": 250,
         "revenue_rate": -5.0, "hours": 6.0},
    ])

    # 12: non-finite or negative fee skips the observation ENTIRELY — no
    # streak/ref/observation change (steps 2-3 snapshots must equal step 1's).
    seq("guard_bad_fee_no_state_change", [
        {"op": "update_posterior", "now": NOW, "fee": 250,
         "revenue_rate": 50.0, "hours": 6.0},
        {"op": "update_posterior", "now": NOW + H, "fee": float("nan"),
         "revenue_rate": 70.0, "hours": 6.0},
        {"op": "update_posterior", "now": NOW + 2 * H, "fee": -1,
         "revenue_rate": 70.0, "hours": 6.0},
    ])

    # 13: exposure weight caps at 6h.
    seq("weight_caps_at_six_hours", [
        {"op": "update_posterior", "now": NOW, "fee": 250,
         "revenue_rate": 40.0, "hours": 12.0},
        {"op": "update_posterior", "now": NOW + H, "fee": 250,
         "revenue_rate": 40.0, "hours": 3.0},
        {"op": "update_posterior", "now": NOW + 2 * H, "fee": 250,
         "revenue_rate": 40.0, "hours": 6.0},
    ])

    # 14: prune to the LAST 200: 199 seeded observations + 1 real + 1 probe
    # appended in the same update -> 201 -> oldest dropped.
    rnd = random.Random(777)
    seeded_obs = [
        (int(rnd.uniform(100, 900)), max(0.0, rnd.gauss(80.0, 40.0)),
         rnd.uniform(0.3, 1.0), NOW - (199 - i) * H, "normal")
        for i in range(199)
    ]
    seq("prune_at_201_including_probe", [
        {"op": "update_posterior", "now": NOW, "fee": 500,
         "revenue_rate": 0.0, "hours": 6.0},
    ], initial={
        "zero_revenue_streak": 10,
        "zero_run_start_fee": 600.0,
        "zero_run_start_ts": NOW - 10 * H,
        "posterior_mean": 550.0,
        "observations": seeded_obs,
    })

    # 15: THE bias re-apply pin (Task 2 review obligation): a nudge blends
    # the mean immediately; the next update_posterior recomputes the
    # posterior from observations (erasing the in-place blend) and
    # _apply_posterior_bias re-applies the decayed nudge — the bias must
    # survive with its recorded (undecayed) weight in posterior_bias.
    seq("nudge_survives_recompute_via_bias_reapply", [
        {"op": "update_posterior", "now": NOW, "fee": 200,
         "revenue_rate": 30.0, "hours": 6.0},
        {"op": "nudge", "now": NOW, "target_fee": 300.0, "weight": 0.4},
        {"op": "update_posterior", "now": NOW + 2 * H, "fee": 210,
         "revenue_rate": 35.0, "hours": 6.0},
        {"op": "update_posterior", "now": NOW + 40 * H, "fee": 220,
         "revenue_rate": 32.0, "hours": 6.0},
    ])

    # 16: nudge dedup (M4): a target within 5% of an existing entry
    # REFRESHES it (max weight, new ts) with NO immediate re-blend; a
    # distinct target appends and blends.
    seq("nudge_dedup_refresh_no_reblend", [
        {"op": "nudge", "now": NOW, "target_fee": 300.0, "weight": 0.4},
        {"op": "nudge", "now": NOW + H, "target_fee": 310.0, "weight": 0.3},
        {"op": "nudge", "now": NOW + H, "target_fee": 310.0, "weight": 0.6},
        {"op": "nudge", "now": NOW + 2 * H, "target_fee": 400.0,
         "weight": 0.2},
    ])

    # 17: nudge memory cap: 50 pre-seeded distinct targets + 1 more evicts
    # the oldest (keep the LAST 50).
    seq("nudge_cap_evicts_oldest", [
        {"op": "nudge", "now": NOW, "target_fee": 5.0, "weight": 0.2},
    ], initial={
        "posterior_bias": [
            (10.0 * (1.1 ** k), 0.05, NOW - k) for k in range(50)
        ],
    })

    # 18: decayed-below-BIAS_MIN_WEIGHT nudges are pruned by the re-apply
    # pass that runs inside update_posterior's recompute.
    seq("nudge_decay_prunes_expired", [
        {"op": "update_posterior", "now": NOW, "fee": 250,
         "revenue_rate": 40.0, "hours": 6.0},
    ], initial={
        "posterior_bias": [
            (300.0, 0.5, NOW - 240 * H),   # 10 half-lives: 4.88e-4 < 1e-3
            (400.0, 0.5, NOW - 48 * H),    # 2 half-lives: 0.125, live
        ],
    })

    # 19: Vegas adjustment matrix: <=1.2 no-op; >1.2 boosts std (capped at
    # 2.0x) and routes the floor through the durable nudge channel only when
    # the floor exceeds the mean.
    seq("vegas_adjustment_matrix", [
        {"op": "vegas", "now": NOW, "vegas_multiplier": 1.2,
         "new_floor": 500.0},
        {"op": "vegas", "now": NOW, "vegas_multiplier": 1.19,
         "new_floor": 500.0},
        {"op": "vegas", "now": NOW, "vegas_multiplier": 1.5,
         "new_floor": 500.0},
        {"op": "vegas", "now": NOW + H, "vegas_multiplier": 3.0,
         "new_floor": 400.0},
        {"op": "vegas", "now": NOW + 2 * H, "vegas_multiplier": 1.5,
         "new_floor": 10.0},
    ])

    # 20: contextual init (hierarchical prior; role S widens by 1.25),
    # precision decay on re-update (7-day last_update decay + 0.98
    # per-update), malformed keys default to role P / time "normal".
    seq("contextual_init_roles_and_updates", [
        {"op": "update_contextual", "now": NOW, "context_key": "mid:peak:P",
         "fee": 300, "revenue_rate": 50.0, "time_bucket": "peak"},
        {"op": "update_contextual", "now": NOW, "context_key": "mid:peak:S",
         "fee": 320, "revenue_rate": 50.0, "time_bucket": "peak"},
        {"op": "update_contextual", "now": NOW + 6 * H,
         "context_key": "mid:peak:P", "fee": 340, "revenue_rate": 80.0,
         "time_bucket": "peak"},
        {"op": "update_contextual", "now": NOW + 6 * H,
         "context_key": "plain", "fee": 200, "revenue_rate": 10.0,
         "time_bucket": "peak"},
        {"op": "update_contextual", "now": NOW + 6 * H,
         "context_key": "low:peak", "fee": 210, "revenue_rate": 10.0,
         "time_bucket": "low"},
    ])

    # 21: cross-pollination: a full-weight (same-bucket) update also nudges
    # ADJACENT time contexts at 0.1x revenue-weight precision WITHOUT
    # incrementing their count; a reduced-weight update does not cross-poll.
    # The legacy 3-tuple context is converted in place when updated.
    seq("contextual_cross_pollination", [
        {"op": "update_contextual", "now": NOW,
         "context_key": "mid:normal:P", "fee": 300, "revenue_rate": 50.0,
         "time_bucket": "normal"},
        {"op": "update_contextual", "now": NOW + H,
         "context_key": "mid:peak:P", "fee": 500, "revenue_rate": 40.0,
         "time_bucket": "normal"},
        {"op": "update_contextual", "now": NOW + 2 * H,
         "context_key": "mid:low:S", "fee": 100, "revenue_rate": 30.0,
         "time_bucket": "low"},
    ], initial={
        "posterior_mean": 250.0,
        "posterior_std": 60.0,
        "contextual_posteriors": {
            "mid:low:P": (150.0, 0.002, 8, NOW - 24 * H),
            "mid:normal:P": (250.0, 0.001, 12, NOW - 12 * H),
            "mid:peak:P": (400.0, 0.0015, 6, NOW - 6 * H),
            "mid:normal:S": (260.0, 45.0, 7),   # legacy 3-tuple
        },
    })

    # 22: updating a stored legacy 3-tuple context converts it with the
    # SAME formula from_dict uses (precision = 1/max(std^2, MIN_STD^2),
    # last_update = 0 -> no age decay on first touch).
    seq("contextual_legacy_3tuple_update", [
        {"op": "update_contextual", "now": NOW, "context_key": "low:peak:P",
         "fee": 280, "revenue_rate": 25.0, "time_bucket": "peak"},
    ], initial={
        "contextual_posteriors": {
            "low:peak:P": (250.0, 45.0, 12),
        },
    })

    # 23: consume_upward_probe stamps the injected now.
    seq("consume_upward_probe_stamps_now", [
        {"op": "consume_upward_probe", "now": NOW + 5 * H},
    ])

    assert len(sequences) == 24, len(sequences)
    _write(outdir / "sequences.json", {"now": NOW, "sequences": sequences})

    # -----------------------------------------------------------------
    # contextual_prune.json: the OValue-order trap. All-count-1 overflow:
    # Python's stable sorted() keeps insertion order on ties, so the 131st
    # (just-inserted) key is itself pruned away; keys updated twice sort to
    # the front. dict(sorted_contexts[:104]) REORDERS the surviving map to
    # count-desc order — a BTreeMap (or unstable sort) changes which keys
    # survive AND their order.
    # -----------------------------------------------------------------
    state = GaussianThompsonState()
    prune_ops = []

    def ctx_op(key, fee, rate, bucket, now):
        prune_ops.append(
            {"context_key": key, "fee": fee, "revenue_rate": _r(rate),
             "time_bucket": bucket, "now": now})
        orig_time = time.time
        time.time = lambda t=now: float(t)
        try:
            state.update_contextual(key, fee, rate, time_bucket=bucket)
        finally:
            time.time = orig_time

    for i in range(130):
        ctx_op(f"b{i:03d}:normal:P", 200 + i, 20.0, "normal", NOW + i)
    # Second updates for three keys -> count 2 (they must survive the prune
    # and lead the reordered map).
    for i in (10, 20, 30):
        ctx_op(f"b{i:03d}:normal:P", 210 + i, 25.0, "normal", NOW + 200 + i)
    # 131st distinct key overflows: len 131 > 130 -> prune to 104 by count
    # desc (stable) -> the freshly inserted b130 (count 1, position 131) is
    # dropped immediately.
    ctx_op("b130:normal:P", 400, 30.0, "normal", NOW + 300)
    assert len(state.contextual_posteriors) == 104
    assert "b130:normal:P" not in state.contextual_posteriors
    keys = list(state.contextual_posteriors.keys())
    assert keys[:3] == ["b010:normal:P", "b020:normal:P", "b030:normal:P"]
    _write(outdir / "contextual_prune.json", {
        "now": NOW,
        "ops": prune_ops,
        "expected_keys_in_order": keys,
        "expected_contexts": [
            [k, _dump_ctx_value(v)]
            for k, v in state.contextual_posteriors.items()
        ],
    })

    # -----------------------------------------------------------------
    # failed_forward.json: the pure math of record_failed_forward
    # (fee_controller.py:8596-8605) + is_fee_relevant_failure (8504-8525).
    # The weight/implied-fee arithmetic is transcribed VERBATIM from the
    # method body (it is inline there, entangled with DB/threading state
    # that has no pure entry point) — source lines pinned in comments.
    # -----------------------------------------------------------------
    weight_cases = []
    for amount_msat in (0, -5, 1, 999, 123_456, 1_000_000, 100_000_000,
                        1_000_000_000, 5_000_000_000, 123_456_789):
        # py 8600-8605 verbatim:
        base_weight = 0.1
        amount_sats = amount_msat / 1000
        if amount_msat > 0:
            amount_boost = min(3.0, 1.0 + math.log10(max(1, amount_sats)) / 3.0)
            base_weight *= amount_boost
        weight_cases.append({
            "amount_msat": amount_msat,
            "amount_sats": _r(amount_sats),
            "expected_weight": _r(base_weight),
        })

    implied_cases = []
    for fee in (1, 2, 5, 100, 999, 1000, 2500, 4999):
        implied_cases.append({
            "current_fee_ppm": fee,
            "expected_implied_fee": int(fee * 0.8),  # py 8596 verbatim
        })

    relevance_cases = []
    for failcode, failreason in [
        (0x1000 | 12, None),
        (0x1000 | 12, "TEMPORARY_CHANNEL_FAILURE"),
        (0x4000 | 15, "WIRE_FEE_INSUFFICIENT"),  # failcode short-circuits
        (0, None),
        (None, "WIRE_FEE_INSUFFICIENT"),
        (None, "fee_insufficient (retry)"),
        (None, "TEMPORARY_CHANNEL_FAILURE"),
        (None, ""),
        (None, None),
    ]:
        relevance_cases.append({
            "failcode": failcode,
            "failreason": failreason,
            "expected": FeeController.is_fee_relevant_failure(
                failcode, failreason),
        })

    _write(outdir / "failed_forward.json", {
        "now": NOW,
        "weight_cases": weight_cases,
        "implied_fee_cases": implied_cases,
        "relevance_cases": relevance_cases,
    })


def gen_ceiling(outdir: Path) -> None:
    ceiling_cases = []

    def ceiling_case(name, observations, floor_ppm=None, now=NOW):
        state = GaussianThompsonState()
        state.observations = [tuple(o) for o in observations]
        v = state.supported_fee_ceiling(now, floor_ppm)
        ceiling_cases.append({
            "name": name,
            "now": now,
            "floor_ppm": _r(floor_ppm) if floor_ppm is not None else None,
            "observations": [_dump_obs(o) for o in observations],
            "expected": _r(v) if v is not None else None,
        })

    ceiling_case("no_earning_history_none", [
        (300, 0.0, 1.0, NOW - H, "normal"),
        (280, 0.0, 1.0, NOW - 2 * H, "normal"),
    ])
    ceiling_case("empty_observations_none", [])
    ceiling_case("basic_quantile_with_headroom", [
        (100, 50.0, 1.0, NOW - H, "normal"),
        (200, 80.0, 1.0, NOW - 2 * H, "normal"),
        (300, 20.0, 1.0, NOW - 3 * H, "normal"),
    ])
    ceiling_case("quantile_reaches_top_fee", [
        (100, 10.0, 1.0, NOW - H, "normal"),
        (500, 300.0, 1.0, NOW - 2 * H, "normal"),
    ])
    # Winsorization (>= 4 masses): the whale's mass is capped at 3x the
    # median, which moves the 0.90 quantile fee downward.
    ceiling_case("winsorized_whale_quantile", [
        (100, 50.0, 1.0, NOW - H, "normal"),
        (150, 60.0, 1.0, NOW - H, "normal"),
        (200, 55.0, 1.0, NOW - H, "normal"),
        (900, 5000.0, 1.0, NOW - H, "normal"),
    ])
    ceiling_case("congestion_revenue_excluded", [
        (100, 50.0, 1.0, NOW - H, "normal"),
        (2000, 900.0, 1.0, NOW - H, "normal", "congestion"),
        (150, 40.0, 1.0, NOW - 2 * H, "normal"),
    ])
    ceiling_case("zero_probe_carries_no_mass", [
        (100, 50.0, 1.0, NOW - H, "normal"),
        (90, 0.0, 1.0, NOW - H, "normal", "zero_probe"),
    ])
    # Floor escape: quantile at/below the floor widens to floor * 2.0.
    ceiling_case("floor_escape_at_floor", [
        (100, 50.0, 1.0, NOW - H, "normal"),
        (100, 45.0, 1.0, NOW - 2 * H, "normal"),
    ], floor_ppm=100.0)
    ceiling_case("floor_escape_only_binds_when_larger", [
        (100, 50.0, 1.0, NOW - H, "normal"),
    ], floor_ppm=50.0)
    ceiling_case("no_escape_quantile_above_floor", [
        (300, 50.0, 1.0, NOW - H, "normal"),
        (350, 45.0, 1.0, NOW - 2 * H, "normal"),
    ], floor_ppm=100.0)
    ceiling_case("floor_zero_never_escapes", [
        (100, 50.0, 1.0, NOW - H, "normal"),
    ], floor_ppm=0.0)
    ceiling_case("stale_mass_decays_out", [
        (100, 50.0, 1.0, NOW - 400 * 24 * H, "normal"),
    ])
    ceiling_case("same_fee_two_windows", [
        (250, 30.0, 1.0, NOW - H, "normal"),
        (250, 90.0, 0.5, NOW - 2 * H, "normal"),
        (400, 5.0, 1.0, NOW - 3 * H, "normal"),
    ])
    _write(outdir / "ceiling.json", {"now": NOW, "cases": ceiling_cases})

    # maybe_upward_probe_cap: guard order is cap-parse -> finite/positive ->
    # streak -> mean -> std -> interval (py 953-974).
    probe_cases = []

    def probe_case(name, *, streak=0, mean=400.0, std=80.0, last_ts=0,
                   cap=300.0, now=NOW):
        state = GaussianThompsonState()
        state.zero_revenue_streak = streak
        state.posterior_mean = mean
        state.posterior_std = std
        state.last_upward_probe_ts = last_ts
        v = state.maybe_upward_probe_cap(now, cap)
        probe_cases.append({
            "name": name,
            "now": now,
            "zero_revenue_streak": streak,
            "posterior_mean": _r(mean),
            "posterior_std": _r(std),
            "last_upward_probe_ts": last_ts,
            "supported_cap": _r(cap),
            "expected": _r(v) if v is not None else None,
        })

    probe_case("grants_stretch")
    probe_case("cap_nan_none", cap=float("nan"))
    probe_case("cap_inf_none", cap=float("inf"))
    probe_case("cap_zero_none", cap=0.0)
    probe_case("cap_negative_none", cap=-50.0)
    probe_case("streak_nonzero_none", streak=1)
    probe_case("streak_negative_none", streak=-1)
    probe_case("mean_at_cap_none", mean=300.0)
    probe_case("mean_below_cap_none", mean=299.0)
    probe_case("std_below_min_none", std=59.999)
    probe_case("std_exactly_min_grants", std=60.0)
    probe_case("within_interval_none", last_ts=NOW - 24 * H + 1)
    probe_case("interval_boundary_grants", last_ts=NOW - 24 * H)
    probe_case("zero_last_ts_skips_interval_gate", last_ts=0)
    _write(outdir / "probe_cap.json", {"now": NOW, "cases": probe_cases})


def _derive_seed(base_seed: int, component: str) -> int:
    from modules.cycle_context import CycleContext
    from modules.econ_types import UnixTime
    c = CycleContext(cycle_id="c", cycle_time=UnixTime(NOW), seed=base_seed,
                     snapshot_id="s")
    return c.derive_seed(component)


def _build_sampling_state(spec) -> GaussianThompsonState:
    state = GaussianThompsonState()
    _apply_initial(state, spec)
    return state


def gen_sampling(outdir: Path) -> None:
    cases = []

    def case(name, *, seed, spec, fn="sample_fee", floor=0, ceiling=5000,
             exploration_multiplier=None, pass_multiplier=True,
             context_key=None, expect_branch=None, expect_gauss=None,
             cholesky_expectation=None):
        state = _build_sampling_state(spec)

        if cholesky_expectation is not None:
            sigma = GaussianThompsonState._mat3_invert(
                state.posterior_precision)
            if cholesky_expectation == "fails":
                assert sigma is not None, name
                assert GaussianThompsonState._cholesky3(sigma) is None, name
            elif cholesky_expectation == "succeeds":
                assert sigma is not None, name
                assert GaussianThompsonState._cholesky3(sigma) is not None, name
            elif cholesky_expectation == "invert_fails":
                assert sigma is None, name

        random.seed(seed)
        counts = {"gauss": 0, "random": 0}
        orig_gauss, orig_random = random.gauss, random.random
        orig_time = time.time

        def counting_gauss(mu=0.0, sigma=1.0):
            counts["gauss"] += 1
            return orig_gauss(mu, sigma)

        def counting_random():
            counts["random"] += 1
            return orig_random()

        random.gauss = counting_gauss
        random.random = counting_random
        time.time = lambda: float(NOW)
        try:
            if fn == "sample_fee":
                if pass_multiplier:
                    fee = state.sample_fee(
                        floor, ceiling,
                        exploration_multiplier=exploration_multiplier)
                else:
                    fee = state.sample_fee(floor, ceiling)
            else:
                if pass_multiplier:
                    fee = state.sample_fee_contextual(
                        context_key, floor, ceiling,
                        exploration_multiplier=exploration_multiplier)
                else:
                    fee = state.sample_fee_contextual(context_key, floor,
                                                      ceiling)
        finally:
            random.gauss = orig_gauss
            random.random = orig_random
            time.time = orig_time

        if expect_gauss is not None:
            assert counts["gauss"] == expect_gauss, (name, counts)

        # Stream-position + gauss-cache pins: the NEXT gauss and random from
        # the SAME stream must match on the Rust side — this is the
        # draw-count parity contract (a missed or extra draw desyncs both).
        post_gauss = random.gauss(0.0, 1.0)
        post_random = random.random()

        cases.append({
            "name": name,
            "seed": seed,
            "now": NOW,
            "state": _dump_initial(spec),
            "call": {
                "fn": fn,
                "floor": floor,
                "ceiling": ceiling,
                "exploration_multiplier": (
                    _r(exploration_multiplier)
                    if exploration_multiplier is not None else None),
                "context_key": context_key,
            },
            "expected": {
                "fee": fee,
                "last_sampled_fee": state.last_sampled_fee,
                "last_sample_time": state.last_sample_time,
                "branch": expect_branch,
                "gauss_draws": counts["gauss"],
                "random_draws": counts["random"],
                "post_gauss": _r(post_gauss),
                "post_random": _r(post_random),
            },
        })

    seed_a = _derive_seed(0, "fee-sample")
    seed_b = _derive_seed(1, "fee-sample")
    seed_c = _derive_seed(2, "fee-sample")
    seed_d = _derive_seed(3, "fee-sample")

    # Real observations to pass the MIN_OBSERVATIONS sparse gate (their
    # values are irrelevant to sampling — only the count matters).
    five_real = [
        (200 + 10 * i, 40.0 + i, 1.0, NOW - (i + 1) * H, "normal")
        for i in range(5)
    ]
    # A PD precision matrix whose inverse is PD -> Cholesky succeeds.
    pd_precision = [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]
    # Invertible but INDEFINITE precision: Sigma = diag(1, -1, 1) -> the
    # Cholesky pivot goes negative -> diagonal-approximation fallback (the
    # draw count must stay 3).
    non_pd_precision = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    concave_coeffs = [-50.0, 20.0, 5.0]
    poly_spec = {
        "observations": five_real,
        "posterior_coeffs": concave_coeffs,
        "posterior_precision": pd_precision,
        "last_fee_min": 100.0,
        "last_fee_max": 400.0,
        "posterior_mean": 250.0,
        "posterior_std": 40.0,
        "charged_fee_mean": 220.0,
    }

    # --- sparse-prior path (ONE gauss; bias shift applies) ---
    case("sparse_fresh_prior", seed=seed_a, spec={},
         expect_branch="sparse_prior", expect_gauss=1)
    case("sparse_min_std_clamp", seed=seed_a,
         spec={"prior_mean_fee": 50.0, "prior_std_fee": 5.0},
         expect_branch="sparse_prior", expect_gauss=1)
    case("sparse_with_live_nudge", seed=seed_b,
         spec={"posterior_bias": [(300.0, 0.5, NOW - H)]},
         expect_branch="sparse_prior", expect_gauss=1)
    case("sparse_with_expired_nudge", seed=seed_b,
         spec={"posterior_bias": [(300.0, 0.5, NOW - 240 * H)]},
         expect_branch="sparse_prior", expect_gauss=1)
    case("sparse_multiplier_clamped_high", seed=seed_a, spec={},
         exploration_multiplier=3.5, expect_branch="sparse_prior",
         expect_gauss=1)
    case("sparse_multiplier_clamped_low", seed=seed_a, spec={},
         exploration_multiplier=0.3, expect_branch="sparse_prior",
         expect_gauss=1)
    case("sparse_multiplier_nan_fallback", seed=seed_a, spec={},
         exploration_multiplier=float("nan"), expect_branch="sparse_prior",
         expect_gauss=1)
    case("sparse_multiplier_inf_fallback", seed=seed_a, spec={},
         exploration_multiplier=float("inf"), expect_branch="sparse_prior",
         expect_gauss=1)
    case("sparse_multiplier_in_range", seed=seed_c, spec={},
         exploration_multiplier=1.3, expect_branch="sparse_prior",
         expect_gauss=1)
    # Probes don't count toward the sparse gate (4 real + 3 probes -> sparse);
    # congestion windows do (4 real + 1 congested -> polynomial path).
    case("sparse_gate_ignores_probes", seed=seed_c, spec={
        "observations": five_real[:4] + [
            (180 - i, 0.0, 1.0, NOW - i * H, "normal", "zero_probe")
            for i in range(3)
        ],
    }, expect_branch="sparse_prior", expect_gauss=1)
    case("sparse_gate_counts_congestion", seed=seed_c, spec={
        **poly_spec,
        "observations": five_real[:4] + [
            (400, 90.0, 1.0, NOW - 6 * H, "peak", "congestion")],
    }, expect_branch="poly_concave", expect_gauss=3,
        cholesky_expectation="succeeds")

    # --- polynomial path, Cholesky success (THREE gauss) ---
    case("poly_concave_cholesky_ok", seed=seed_a, spec=poly_spec,
         expect_branch="poly_concave", expect_gauss=3,
         cholesky_expectation="succeeds")
    case("poly_concave_with_live_nudge", seed=seed_b, spec={
        **poly_spec, "posterior_bias": [(600.0, 0.4, NOW - 2 * H)],
    }, expect_branch="poly_concave", expect_gauss=3,
        cholesky_expectation="succeeds")
    case("poly_concave_boost_max", seed=seed_c, spec=poly_spec,
         exploration_multiplier=2.0, expect_branch="poly_concave",
         expect_gauss=3, cholesky_expectation="succeeds")
    case("poly_concave_omitted_kwarg", seed=seed_a, spec=poly_spec,
         pass_multiplier=False, expect_branch="poly_concave",
         expect_gauss=3, cholesky_expectation="succeeds")

    # --- polynomial path, Cholesky FAILS -> diagonal fallback (still THREE
    # gauss draws before the concavity check) ---
    case("poly_cholesky_fallback_diag", seed=seed_a, spec={
        **poly_spec, "posterior_precision": non_pd_precision,
    }, expect_branch="poly_cholesky_fallback", expect_gauss=3,
        cholesky_expectation="fails")
    case("poly_cholesky_fallback_boosted", seed=seed_d, spec={
        **poly_spec, "posterior_precision": non_pd_precision,
    }, exploration_multiplier=1.7, expect_branch="poly_cholesky_fallback",
        expect_gauss=3, cholesky_expectation="fails")

    # --- polynomial returns None -> Gaussian posterior fallback ---
    # Non-concave sampled quadratic: 3 draws consumed, then ONE more gauss
    # (bias shift does NOT apply on this path).
    case("poly_non_concave_gauss_fallback", seed=seed_a, spec={
        **poly_spec,
        "posterior_coeffs": [0.5, 1.0, 2.0],
        "posterior_precision": [[10000.0, 0.0, 0.0], [0.0, 10000.0, 0.0],
                                [0.0, 0.0, 10000.0]],
        "posterior_bias": [(900.0, 0.9, NOW - H)],
    }, expect_branch="non_concave_gauss_fallback", expect_gauss=4,
        cholesky_expectation="succeeds")
    # Degenerate fee range: polynomial declines BEFORE any draw -> ONE gauss.
    case("poly_range_too_narrow_gauss", seed=seed_b, spec={
        **poly_spec, "last_fee_min": 200.0, "last_fee_max": 203.0,
    }, expect_branch="gauss_fallback_no_poly_draws", expect_gauss=1)
    # Singular precision: inversion fails before any draw -> ONE gauss.
    case("poly_invert_fails_gauss", seed=seed_c, spec={
        **poly_spec,
        "posterior_precision": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0]],
    }, expect_branch="gauss_fallback_no_poly_draws", expect_gauss=1,
        cholesky_expectation="invert_fails")
    case("gauss_fallback_min_std_boosted", seed=seed_d, spec={
        **poly_spec, "last_fee_min": 0.0, "last_fee_max": 0.0,
        "posterior_std": 4.0,
    }, exploration_multiplier=1.9,
        expect_branch="gauss_fallback_no_poly_draws", expect_gauss=1)

    # --- clamping (int truncation of max(floor, min(ceiling, sampled))) ---
    case("clamp_to_floor", seed=seed_a, spec=poly_spec, floor=495,
         ceiling=5000, expect_branch="poly_concave", expect_gauss=3)
    case("clamp_to_ceiling", seed=seed_a, spec=poly_spec, floor=0,
         ceiling=120, expect_branch="poly_concave", expect_gauss=3)

    # --- contextual offset path ---
    ctx4 = {"mid:peak:P": (400.0, 0.001, 20, NOW - H)}
    case("ctx_offset_capped", seed=seed_a, spec={
        **poly_spec, "contextual_posteriors": ctx4,
    }, fn="sample_fee_contextual", context_key="mid:peak:P",
        expect_branch="ctx_offset", expect_gauss=3)
    case("ctx_offset_within_cap", seed=seed_a, spec={
        **poly_spec,
        "contextual_posteriors": {"mid:peak:P": (230.0, 0.001, 20, NOW - H)},
    }, fn="sample_fee_contextual", context_key="mid:peak:P",
        expect_branch="ctx_offset", expect_gauss=3)
    case("ctx_legacy_3tuple_offset", seed=seed_b, spec={
        **poly_spec,
        "contextual_posteriors": {"low:normal:P": (350.0, 45.0, 12)},
    }, fn="sample_fee_contextual", context_key="low:normal:P",
        expect_branch="ctx_offset", expect_gauss=3)
    case("ctx_count_below_min_passthrough", seed=seed_a, spec={
        **poly_spec,
        "contextual_posteriors": {"mid:peak:P": (400.0, 0.001, 4, NOW - H)},
    }, fn="sample_fee_contextual", context_key="mid:peak:P",
        expect_branch="ctx_passthrough", expect_gauss=3)
    case("ctx_missing_key_passthrough", seed=seed_a, spec=poly_spec,
         fn="sample_fee_contextual", context_key="never:seen:P",
         expect_branch="ctx_passthrough", expect_gauss=3)
    case("ctx_nonfinite_mean_passthrough", seed=seed_a, spec={
        **poly_spec,
        "contextual_posteriors": {
            "mid:peak:P": (float("nan"), 0.001, 20, NOW - H)},
    }, fn="sample_fee_contextual", context_key="mid:peak:P",
        expect_branch="ctx_passthrough", expect_gauss=3)
    case("ctx_reference_falls_back_to_posterior_mean", seed=seed_b, spec={
        **poly_spec, "charged_fee_mean": 0.0,
        "contextual_posteriors": ctx4,
    }, fn="sample_fee_contextual", context_key="mid:peak:P",
        expect_branch="ctx_offset", expect_gauss=3)
    case("ctx_on_sparse_base", seed=seed_c, spec={
        "contextual_posteriors": ctx4, "charged_fee_mean": 150.0,
    }, fn="sample_fee_contextual", context_key="mid:peak:P",
        expect_branch="ctx_offset", expect_gauss=1)
    case("ctx_with_boost_forwarded", seed=seed_d, spec={
        **poly_spec, "contextual_posteriors": ctx4,
    }, fn="sample_fee_contextual", context_key="mid:peak:P",
        exploration_multiplier=1.5, expect_branch="ctx_offset",
        expect_gauss=3)

    assert len(cases) == 32, len(cases)
    _write(outdir / "draws.json", {
        "now": NOW,
        "seeds": {"seed_a": seed_a, "seed_b": seed_b, "seed_c": seed_c,
                  "seed_d": seed_d},
        "cases": cases,
    })

    # -----------------------------------------------------------------
    # shift.json: _posterior_bias_shift direct pins (sequential decayed
    # w/(1+w) blends; entries below BIAS_MIN_WEIGHT skipped, NOT pruned).
    # -----------------------------------------------------------------
    shift_cases = []

    def shift_case(name, bias, base, now=NOW):
        state = GaussianThompsonState()
        state.posterior_bias = [tuple(b) for b in bias]
        orig_time = time.time
        time.time = lambda t=now: float(t)
        try:
            v = state._posterior_bias_shift(base)
        finally:
            time.time = orig_time
        shift_cases.append({
            "name": name,
            "now": now,
            "posterior_bias": _dump_bias(bias),
            "base": _r(base),
            "expected": _r(v),
        })

    shift_case("empty_bias_zero", [], 150.0)
    shift_case("single_fresh_nudge", [(300.0, 0.5, NOW - H)], 150.0)
    shift_case("single_fresh_nudge_base_zero", [(300.0, 0.5, NOW - H)], 0.0)
    shift_case("negative_base", [(300.0, 0.5, NOW - H)], -50.0)
    shift_case("sequential_blends_order_matters",
               [(300.0, 0.5, NOW - H), (100.0, 0.8, NOW - 2 * H),
                (700.0, 0.2, NOW - 30 * H)], 200.0)
    shift_case("expired_entry_skipped",
               [(300.0, 0.5, NOW - 240 * H), (400.0, 0.5, NOW - 48 * H)],
               200.0)
    shift_case("future_ts_age_clamped_to_zero",
               [(300.0, 0.5, NOW + 10 * H)], 200.0)
    _write(outdir / "shift.json", {"now": NOW, "cases": shift_cases})


SUITES = {
    "pyrand": gen_pyrand,
    "mat3": gen_mat3,
    "rails": gen_rails,
    "posterior": gen_posterior,
    "discount": gen_discount,
    "pid": gen_pid,
    "state_dict": gen_state_dict,
    "update": gen_update,
    "ceiling": gen_ceiling,
    "sampling": gen_sampling,
}


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in SUITES:
        names = "|".join(SUITES)
        print(f"usage: {sys.argv[0]} {{{names}}} <outdir>", file=sys.stderr)
        sys.exit(1)
    SUITES[sys.argv[1]](Path(sys.argv[2]))


if __name__ == "__main__":
    main()
