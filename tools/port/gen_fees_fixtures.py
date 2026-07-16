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


SUITES = {
    "pyrand": gen_pyrand,
    "mat3": gen_mat3,
    "rails": gen_rails,
    "pid": gen_pid,
    "state_dict": gen_state_dict,
}


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in SUITES:
        names = "|".join(SUITES)
        print(f"usage: {sys.argv[0]} {{{names}}} <outdir>", file=sys.stderr)
        sys.exit(1)
    SUITES[sys.argv[1]](Path(sys.argv[2]))


if __name__ == "__main__":
    main()
