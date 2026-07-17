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
    ChannelCycleState,
    ChannelFeeState,
    FeeController,
    GaussianThompsonState,
    PIDState,
    VegasReflexState,
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
# floors: evidence-backed floors (Phase 4 Task 6) — _calculate_floor,
# _get_dynamic_chain_costs_live, _get_rebalance_cost_floor,
# _get_flow_adjusted_ceiling, _detect_congestion, _effective_min_fee_ppm.
# The real FeeController methods are the oracle; nothing here reimplements
# the algorithm. Chain-cost defaults come from ChainCostDefaults directly
# (imported, not hand-copied).
# ---------------------------------------------------------------------------

def gen_floors(outdir: Path) -> None:
    from modules.config import ChainCostDefaults

    # --- calculate_floor.json ---------------------------------------------
    floor_cases = []

    def _floor_case(name, *, capacity_sats, chain_costs, peer_latency, opener):
        fc = _controller()
        if peer_latency is not None:
            fc.database.get_peer_latency_stats.return_value = peer_latency
            peer_id = "03" + "ab" * 32
        else:
            peer_id = None
        expected = fc._calculate_floor(
            capacity_sats, chain_costs=chain_costs, peer_id=peer_id, opener=opener
        )
        floor_cases.append({
            "name": name,
            "capacity_sats": capacity_sats,
            "chain_costs": chain_costs,
            "peer_latency": peer_latency,
            "opener": opener,
            "expected_floor_ppm": expected,
        })

    _floor_case(
        "risk_premium_wins_no_stall",
        capacity_sats=2_000_000,
        chain_costs={"open_cost_sats": 2500, "close_cost_sats": 1500, "sat_per_vbyte": 15.0},
        peer_latency=None,
        opener="local",
    )
    _floor_case(
        # P8-002 regression pin: risk premium wins the max() AND the stall
        # markup must apply to the WINNING term (not the base floor that
        # lost the max()).
        "stall_multiplier_after_max_p8_002_regression",
        capacity_sats=2_000_000,
        chain_costs={"open_cost_sats": 2500, "close_cost_sats": 1500, "sat_per_vbyte": 15.0},
        peer_latency={"avg": 15.0, "std": 1.0},
        opener="local",
    )
    _floor_case(
        "stall_multiplier_only_no_dynamic_costs",
        capacity_sats=2_000_000,
        chain_costs=None,
        peer_latency={"avg": 0.0, "std": 6.0},
        opener="local",
    )
    _floor_case(
        "peer_latency_below_thresholds_no_stall",
        capacity_sats=2_000_000,
        chain_costs=None,
        peer_latency={"avg": 5.0, "std": 2.0},
        opener="local",
    )
    _floor_case(
        "risk_premium_zero_sat_per_vbyte_skipped",
        capacity_sats=2_000_000,
        chain_costs={"open_cost_sats": 2500, "close_cost_sats": 1500, "sat_per_vbyte": 0.0},
        peer_latency=None,
        opener="local",
    )
    _floor_case(
        "remote_opener_with_risk_premium",
        capacity_sats=500_000,
        chain_costs={"open_cost_sats": 2500, "close_cost_sats": 1500, "sat_per_vbyte": 20.0},
        peer_latency=None,
        opener="remote",
    )
    _floor_case(
        "avg_threshold_boundary_10_0_not_triggered",
        capacity_sats=2_000_000,
        chain_costs=None,
        peer_latency={"avg": 10.0, "std": 0.0},
        opener="local",
    )
    _floor_case(
        "avg_threshold_boundary_10_0001_triggered",
        capacity_sats=2_000_000,
        chain_costs=None,
        peer_latency={"avg": 10.0001, "std": 0.0},
        opener="local",
    )
    _write(outdir / "calculate_floor.json", {"now": NOW, "cases": floor_cases})

    # --- pinned_constants.json (ChainCostDefaults, transcribed) -----------
    _write(outdir / "pinned_constants.json", {
        "CHANNEL_OPEN_COST_SATS": ChainCostDefaults.CHANNEL_OPEN_COST_SATS,
        "CHANNEL_CLOSE_COST_SATS": ChainCostDefaults.CHANNEL_CLOSE_COST_SATS,
        "CHANNEL_LIFETIME_DAYS": ChainCostDefaults.CHANNEL_LIFETIME_DAYS,
        "DAILY_VOLUME_SATS": ChainCostDefaults.DAILY_VOLUME_SATS,
        "REBALANCE_FLOOR_WINDOW_DAYS": FeeController.REBALANCE_FLOOR_WINDOW_DAYS,
        "REBALANCE_FLOOR_MIN_SAMPLES": FeeController.REBALANCE_FLOOR_MIN_SAMPLES,
        "REBALANCE_FLOOR_MARGIN": _r(FeeController.REBALANCE_FLOOR_MARGIN),
        "SATURATED_OUTBOUND_RATIO": _r(FeeController.SATURATED_OUTBOUND_RATIO),
        "FLOW_BALANCED_WINDOW_SECONDS": FeeController.FLOW_BALANCED_WINDOW_SECONDS,
        "FLOW_BALANCED_MAX_NET_RATIO": _r(FeeController.FLOW_BALANCED_MAX_NET_RATIO),
        "FLOW_BALANCED_MIN_WEEKLY_TURNOVER": _r(FeeController.FLOW_BALANCED_MIN_WEEKLY_TURNOVER),
    })

    # --- dynamic_chain_costs.json ------------------------------------------
    dcc_cases = []

    def _dcc_case(name, perkb):
        fc = _controller()
        fc.data_service = MagicMock()
        fc.data_service.get_feerates.return_value = {"perkb": perkb}
        result = fc._get_dynamic_chain_costs_live()
        dcc_cases.append({
            "name": name,
            "perkb": perkb,
            "expected": None if result is None else {
                "open_cost_sats": result["open_cost_sats"],
                "close_cost_sats": result["close_cost_sats"],
                "sat_per_vbyte": _r(result["sat_per_vbyte"]),
            },
        })

    _dcc_case("opening_present", {"opening": 5000})
    _dcc_case("opening_zero_falls_to_mutual_close", {"opening": 0, "mutual_close": 3000})
    _dcc_case(
        "opening_null_mutual_close_zero_falls_to_unilateral",
        {"opening": None, "mutual_close": 0, "unilateral_close": 2000},
    )
    _dcc_case("all_missing_falls_to_floor_key", {"floor": 800})
    _dcc_case("all_missing_falls_to_1000_default", {})
    _dcc_case("high_fee_clamped_to_max", {"opening": 1_000_000})
    _dcc_case("low_fee_clamped_to_min", {"opening": 1})
    _write(outdir / "dynamic_chain_costs.json", {"now": NOW, "cases": dcc_cases})

    # --- rebalance_cost_floor.json ------------------------------------------
    rcf_cases = []

    def _rcf_case(name, *, flow_state, cost_history, peer_fallback):
        fc = _controller()
        orig_time = time.time
        time.time = lambda: float(NOW)
        try:
            fc.database.get_channel_cost_history.return_value = cost_history
            if peer_fallback is not None:
                fc.database.get_historical_inbound_fee_ppm.return_value = peer_fallback
            else:
                fc.database.get_historical_inbound_fee_ppm.return_value = None
            expected = fc._get_rebalance_cost_floor("chan1", "peer1", flow_state)
        finally:
            time.time = orig_time
        rcf_cases.append({
            "name": name,
            "flow_state": flow_state,
            "recent_costs": cost_history,
            "peer_fallback": peer_fallback,
            "now": NOW,
            "expected": expected,
        })

    _rcf_case("sink_returns_none", flow_state="sink",
              cost_history=[{"cost_sats": 100, "amount_sats": 10000, "timestamp": NOW - 1000}] * 5,
              peer_fallback={"confidence": "high", "avg_fee_ppm": 500})
    _rcf_case("dormant_returns_none", flow_state="dormant",
              cost_history=[{"cost_sats": 100, "amount_sats": 10000, "timestamp": NOW - 1000}] * 5,
              peer_fallback={"confidence": "high", "avg_fee_ppm": 500})
    _rcf_case(
        "four_samples_integer_division_pin",
        flow_state="source",
        cost_history=[
            {"cost_sats": 100, "amount_sats": 10000, "timestamp": NOW - 1000},
            {"cost_sats": 150, "amount_sats": 15000, "timestamp": NOW - 2000},
            {"cost_sats": 130, "amount_sats": 13000, "timestamp": NOW - 3000},
            {"cost_sats": 120, "amount_sats": 11000, "timestamp": NOW - 4000},
        ],
        peer_fallback=None,
    )
    _rcf_case(
        "insufficient_samples_falls_to_peer_fallback_medium",
        flow_state="router",
        cost_history=[
            {"cost_sats": 100, "amount_sats": 10000, "timestamp": NOW - 1000},
            {"cost_sats": 150, "amount_sats": 15000, "timestamp": NOW - 2000},
        ],
        peer_fallback={"confidence": "medium", "avg_fee_ppm": 77},
    )
    _rcf_case(
        "fallback_confidence_low_excluded_returns_none",
        flow_state="source",
        cost_history=[],
        peer_fallback={"confidence": "low", "avg_fee_ppm": 500},
    )
    _rcf_case(
        "fallback_avg_fee_ppm_zero_excluded_returns_none",
        flow_state="source",
        cost_history=[],
        peer_fallback={"confidence": "high", "avg_fee_ppm": 0},
    )
    _rcf_case(
        "samples_outside_window_excluded_falls_to_fallback",
        flow_state="source",
        cost_history=[
            # 2 stale (outside REBALANCE_FLOOR_WINDOW_DAYS=30d), 2 fresh
            # -> fresh count (2) < MIN_SAMPLES (4) -> falls through.
            {"cost_sats": 900, "amount_sats": 1000, "timestamp": NOW - 40 * 86400},
            {"cost_sats": 900, "amount_sats": 1000, "timestamp": NOW - 35 * 86400},
            {"cost_sats": 100, "amount_sats": 10000, "timestamp": NOW - 1000},
            {"cost_sats": 150, "amount_sats": 15000, "timestamp": NOW - 2000},
        ],
        peer_fallback={"confidence": "high", "avg_fee_ppm": 123},
    )
    _rcf_case(
        "volume_zero_falls_through_to_fallback",
        flow_state="source",
        cost_history=[
            {"cost_sats": 10, "amount_sats": 0, "timestamp": NOW - 1000},
            {"cost_sats": 10, "amount_sats": 0, "timestamp": NOW - 2000},
            {"cost_sats": 10, "amount_sats": 0, "timestamp": NOW - 3000},
            {"cost_sats": 10, "amount_sats": 0, "timestamp": NOW - 4000},
        ],
        peer_fallback={"confidence": "high", "avg_fee_ppm": 55},
    )
    _rcf_case(
        "no_data_at_all_returns_none",
        flow_state="source",
        cost_history=[],
        peer_fallback=None,
    )
    _write(outdir / "rebalance_cost_floor.json", {"now": NOW, "cases": rcf_cases})

    # --- flow_adjusted_ceiling.json -----------------------------------------
    fac_cases = []

    def _fac_case(name, *, current_fee, base_ceiling, last_forward_ts):
        fc = _controller()
        orig_time = time.time
        time.time = lambda: float(NOW)
        try:
            fc.database.get_last_forward_time.return_value = last_forward_ts
            expected = fc._get_flow_adjusted_ceiling("chan1", current_fee, base_ceiling)
        finally:
            time.time = orig_time
        fac_cases.append({
            "name": name,
            "current_fee": current_fee,
            "base_ceiling": base_ceiling,
            "last_forward_ts": last_forward_ts,
            "now": NOW,
            "expected": expected,
        })

    _fac_case("below_fee_threshold_returns_base_unchanged",
              current_fee=499, base_ceiling=1001, last_forward_ts=NOW - 100 * 86400)
    _fac_case("no_forwards_recorded_returns_base",
              current_fee=600, base_ceiling=1001, last_forward_ts=None)
    _fac_case("zero_timestamp_sentinel_returns_base",
              current_fee=600, base_ceiling=1001, last_forward_ts=0)
    _fac_case("just_under_moderate_2_99_days_no_reduction",
              current_fee=600, base_ceiling=1001, last_forward_ts=NOW - 258336)  # 2.99d
    _fac_case("exactly_3_days_moderate_reduction",
              current_fee=600, base_ceiling=1001, last_forward_ts=NOW - 3 * 86400)
    _fac_case("just_under_severe_6_99_days_moderate_reduction",
              current_fee=600, base_ceiling=1001, last_forward_ts=NOW - 604224)  # 6.99d
    _fac_case("exactly_7_days_severe_reduction",
              current_fee=600, base_ceiling=1001, last_forward_ts=NOW - 7 * 86400)
    _fac_case("beyond_severe_10_days",
              current_fee=600, base_ceiling=1001, last_forward_ts=NOW - 10 * 86400)
    _write(outdir / "flow_adjusted_ceiling.json", {"now": NOW, "cases": fac_cases})

    # --- congestion.json ------------------------------------------------------
    cong_cases = []

    def _cong_case(name, *, state, channel_info, htlc_congestion_threshold, flow_interval):
        fc = _controller()
        orig_time = time.time
        time.time = lambda: float(NOW)
        try:
            cfg = SimpleNamespace(
                htlc_congestion_threshold=htlc_congestion_threshold,
                flow_interval=flow_interval,
            )
            expected = fc._detect_congestion(state, channel_info, cfg)
        finally:
            time.time = orig_time
        cong_cases.append({
            "name": name,
            "state": state,
            "channel_info": channel_info,
            "htlc_congestion_threshold": _r(htlc_congestion_threshold),
            "flow_interval": flow_interval,
            "now": NOW,
            "expected": expected,
        })

    _cong_case("live_data_over_threshold_true",
               state=None,
               channel_info={"has_htlc_data": True, "max_accepted_htlcs": 10, "our_htlcs_in_flight": 9},
               htlc_congestion_threshold=0.8, flow_interval=1800)
    _cong_case("live_data_at_threshold_boundary_false",
               state=None,
               channel_info={"has_htlc_data": True, "max_accepted_htlcs": 10, "our_htlcs_in_flight": 8},
               htlc_congestion_threshold=0.8, flow_interval=1800)
    _cong_case("live_data_max_zero_falls_back_to_snapshot_none",
               state=None,
               channel_info={"has_htlc_data": True, "max_accepted_htlcs": 0, "our_htlcs_in_flight": 5},
               htlc_congestion_threshold=0.8, flow_interval=1800)
    _cong_case("snapshot_fresh_congested_true",
               state={"state": "congested", "updated_at": NOW - 100},
               channel_info=None,
               htlc_congestion_threshold=0.8, flow_interval=1800)
    _cong_case("snapshot_stale_beyond_2x_flow_interval_false",
               state={"state": "congested", "updated_at": NOW - 4000},
               channel_info=None,
               htlc_congestion_threshold=0.8, flow_interval=1800)
    _cong_case("snapshot_missing_updated_at_treated_fresh_true",
               state={"state": "congested"},
               channel_info=None,
               htlc_congestion_threshold=0.8, flow_interval=1800)
    _cong_case("snapshot_updated_at_zero_treated_fresh_true",
               state={"state": "congested", "updated_at": 0},
               channel_info=None,
               htlc_congestion_threshold=0.8, flow_interval=1800)
    _cong_case("snapshot_not_congested_false",
               state={"state": "idle", "updated_at": NOW},
               channel_info=None,
               htlc_congestion_threshold=0.8, flow_interval=1800)
    _cong_case("no_state_no_channel_info_false",
               state=None, channel_info=None,
               htlc_congestion_threshold=0.8, flow_interval=1800)
    _cong_case("live_htlc_data_false_falls_back_to_fresh_snapshot_true",
               state={"state": "congested", "updated_at": NOW - 10},
               channel_info={"has_htlc_data": False, "max_accepted_htlcs": 10, "our_htlcs_in_flight": 9},
               htlc_congestion_threshold=0.8, flow_interval=1800)
    _cong_case("stale_boundary_exactly_2x_flow_interval_still_fresh",
               state={"state": "congested", "updated_at": NOW - 3600},
               channel_info=None,
               htlc_congestion_threshold=0.8, flow_interval=1800)
    _write(outdir / "congestion.json", {"now": NOW, "cases": cong_cases})

    # --- effective_min_fee.json ------------------------------------------------
    emf_cases = []

    def _emf_case(name, *, min_fee_ppm, min_fee_ppm_saturated, flow_state, outbound_ratio,
                  capacity_sats, flow_window):
        fc = _controller()
        orig_time = time.time
        time.time = lambda: float(NOW)
        try:
            fc.database.get_all_channel_flow_windows.return_value = (
                {"chan1": flow_window} if flow_window is not None else {}
            )
            cfg = SimpleNamespace(min_fee_ppm=min_fee_ppm, min_fee_ppm_saturated=min_fee_ppm_saturated)
            expected = fc._effective_min_fee_ppm(
                cfg, flow_state=flow_state, outbound_ratio=outbound_ratio,
                channel_id="chan1", capacity_sats=capacity_sats,
            )
        finally:
            time.time = orig_time
        emf_cases.append({
            "name": name,
            "min_fee_ppm": min_fee_ppm,
            "min_fee_ppm_saturated": min_fee_ppm_saturated,
            "flow_state": flow_state,
            "outbound_ratio": None if outbound_ratio is None else _r(outbound_ratio),
            "capacity_sats": capacity_sats,
            "flow_window": None if flow_window is None else list(flow_window),
            "expected": expected,
        })

    _emf_case("below_saturated_and_not_source_returns_base",
              min_fee_ppm=100, min_fee_ppm_saturated=20, flow_state=None,
              outbound_ratio=0.5, capacity_sats=1_000_000, flow_window=None)
    _emf_case("sat_floor_negative_ignored_returns_base",
              min_fee_ppm=100, min_fee_ppm_saturated=-5, flow_state="source",
              outbound_ratio=0.9, capacity_sats=1_000_000, flow_window=None)
    _emf_case("sat_floor_equal_base_ignored_returns_base",
              min_fee_ppm=100, min_fee_ppm_saturated=100, flow_state="source",
              outbound_ratio=0.9, capacity_sats=1_000_000, flow_window=None)
    _emf_case("source_channel_no_flow_data_takes_sat_floor",
              min_fee_ppm=100, min_fee_ppm_saturated=20, flow_state="source",
              outbound_ratio=0.3, capacity_sats=1_000_000, flow_window=None)
    _emf_case("saturated_outbound_ratio_at_boundary_takes_sat_floor",
              min_fee_ppm=100, min_fee_ppm_saturated=20, flow_state=None,
              outbound_ratio=0.85, capacity_sats=1_000_000, flow_window=None)
    _emf_case("saturated_outbound_ratio_just_below_boundary_returns_base",
              min_fee_ppm=100, min_fee_ppm_saturated=20, flow_state=None,
              outbound_ratio=0.849999, capacity_sats=1_000_000, flow_window=None)
    _emf_case("flow_balanced_router_exemption_returns_base",
              min_fee_ppm=100, min_fee_ppm_saturated=20, flow_state="source",
              outbound_ratio=0.9, capacity_sats=1_000_000, flow_window=(200_000, 190_000, 42))
    _emf_case("flow_not_balanced_low_turnover_takes_sat_floor",
              min_fee_ppm=100, min_fee_ppm_saturated=20, flow_state="source",
              outbound_ratio=0.9, capacity_sats=1_000_000, flow_window=(10_000, 9_000, 3))
    _emf_case("flow_not_balanced_net_ratio_too_high_takes_sat_floor",
              min_fee_ppm=100, min_fee_ppm_saturated=20, flow_state="source",
              outbound_ratio=0.9, capacity_sats=1_000_000, flow_window=(300_000, 50_000, 12))
    _write(outdir / "effective_min_fee.json", {"now": NOW, "cases": emf_cases})


# ---------------------------------------------------------------------------
# vegas: VegasReflexState.update/get_floor_multiplier (Phase 4 Task 6).
# The real dataclass methods are the oracle. Pins decay-before-check
# ordering AND the short-circuited RNG-consumption trap: `random.random()`
# is called ONLY inside the 2x..4x branch when `consecutive_spikes < 2`.
# ---------------------------------------------------------------------------

def gen_vegas(outdir: Path) -> None:
    # 12-cycle ratio sequence hitting: decay-only, the 2x boundary exactly,
    # RNG-gated vs consecutive-confirmed spike triggers, the 4x boundary
    # exactly (immediate trigger, not the probabilistic branch), and a
    # decay-only cool-down before another immediate trigger.
    ratios = [1.0, 2.0, 2.5, 3.9, 4.5, 1.5, 2.1, 1.9, 2.9, 2.9, 0.5, 4.0]
    ma = 10.0
    seeds = [0, 1, 42, 12345]

    entries = []
    for seed in seeds:
        random.seed(seed)
        state = VegasReflexState()
        cycles = []
        orig_time = time.time
        try:
            for i, ratio in enumerate(ratios):
                cycle_now = NOW + i * 1800
                time.time = lambda t=cycle_now: float(t)
                current = ratio * ma
                state.update(current, ma)
                cycles.append({
                    "cycle": i,
                    "now": cycle_now,
                    "spike_ratio_input": _r(ratio),
                    "current_sat_vb": _r(current),
                    "ma_sat_vb": _r(ma),
                    "intensity": _r(state.intensity),
                    "consecutive_spikes": state.consecutive_spikes,
                    "floor_multiplier": _r(state.get_floor_multiplier()),
                })
        finally:
            time.time = orig_time
        entries.append({"seed": seed, "cycles": cycles})

    _write(outdir / "sequences.json", {"now": NOW, "entries": entries})

    # ma<=0 guard (py 2363-2364): ma_sat_vb clamped to 1.0 before the ratio
    # is computed.
    random.seed(0)
    guard_state = VegasReflexState()
    orig_time = time.time
    time.time = lambda: float(NOW)
    try:
        guard_state.update(2.5, 0.0)
        guard_intensity_after_1 = _r(guard_state.intensity)
        guard_consecutive_after_1 = guard_state.consecutive_spikes
        guard_state.update(2.5, -3.0)
        guard_intensity_after_2 = _r(guard_state.intensity)
        guard_consecutive_after_2 = guard_state.consecutive_spikes
    finally:
        time.time = orig_time
    _write(outdir / "ma_leq_zero_guard.json", {
        "now": NOW,
        "seed": 0,
        # ma<=0 -> treated as 1.0, so current_sat_vb=2.5 -> ratio=2.5 (2x-4x
        # branch) both cycles.
        "cycles": [
            {
                "current_sat_vb": _r(2.5), "ma_sat_vb_input": _r(0.0),
                "intensity": guard_intensity_after_1,
                "consecutive_spikes": guard_consecutive_after_1,
            },
            {
                "current_sat_vb": _r(2.5), "ma_sat_vb_input": _r(-3.0),
                "intensity": guard_intensity_after_2,
                "consecutive_spikes": guard_consecutive_after_2,
            },
        ],
    })


# ---------------------------------------------------------------------------
# v2_blob: fee_strategy_state.v2_state_json lossless round-trip (Phase 4
# Task 9). Exercises `_load_persisted_fee_strategy_row`,
# `_extract_fee_state_payload`, `_extract_cycle_state_payload`,
# `_build_merged_fee_strategy_row`, `ChannelFeeState.to_v2_dict`/
# `from_v2_dict` through the REAL `FeeController`/`ChannelFeeState`/
# `ChannelCycleState` classes. A scripted `_FakeDb` stands in for
# `self.database` (only `get_fee_strategy_state` is exercised on the
# cold-cache, no-cycle-batch path used here).
# ---------------------------------------------------------------------------

class _FakeDb:
    """Minimal `self.database` double: returns a fixed db_state row
    (single-channel generator, channel_id argument is ignored)."""

    def __init__(self, db_state):
        self._db_state = db_state

    def get_fee_strategy_state(self, channel_id):
        return dict(self._db_state)


_DEFAULT_ROW = {
    "last_revenue_rate": 12.5,
    "last_fee_ppm": 250,
    "trend_direction": 1,
    "step_ppm": 50,
    "consecutive_same_direction": 2,
    "last_update": NOW - 900,
    "last_broadcast_fee_ppm": 240,
    "is_sleeping": 0,
    "sleep_until": 0,
    "stable_cycles": 1,
    "forward_count_since_update": 6,
    "last_volume_sats": 15000,
    "last_state": "balanced",
}


def _row_for(channel_id: str, **overrides) -> dict:
    row = dict(_DEFAULT_ROW)
    row.update(overrides)
    row["channel_id"] = channel_id
    return row


def _merged_roundtrip(channel_id: str, row: dict, input_blob: str,
                       *, cycle_state=None, fee_state=None):
    """One `_build_merged_fee_strategy_row(channel_id, cycle_state=...,
    fee_state=...)` pass over a persisted `(row, input_blob)` pair through
    a FRESH `FeeController` (cold in-memory caches -> always re-loads from
    `_FakeDb`). With both callers `None` this is the exact no-op
    read-modify-write shape the live controller performs when a cycle
    touches neither cycle_state nor fee_state for a channel — the target
    Rust `build_merged_row(channel_id, None, None, persisted)` must
    reproduce it byte-identically."""
    db_row = dict(row)
    db_row["v2_state_json"] = input_blob
    fc = FeeController(MagicMock(), MagicMock(spec=Config), _FakeDb(db_row))
    row_fields, merged_v2 = fc._build_merged_fee_strategy_row(
        channel_id, cycle_state=cycle_state, fee_state=fee_state
    )
    python_roundtrip_blob = json.dumps(merged_v2)
    return row_fields, merged_v2, python_roundtrip_blob


def _pinned_fee_state(channel_id: str, row: dict, input_blob: str) -> dict:
    """`ChannelFeeState.from_v2_dict` truth for `input_blob`'s fee_state
    payload, floats repr-pinned — mirrors `_get_channel_fee_state_locked`'s
    load path (extract -> from_v2_dict -> unknown-version migration
    stamp)."""
    db_row = dict(row)
    db_row["v2_state_json"] = input_blob
    fc = FeeController(MagicMock(), MagicMock(spec=Config), _FakeDb(db_row))
    db_state, v2_data = fc._load_persisted_fee_strategy_row(channel_id)
    fee_v2_data = fc._extract_fee_state_payload(db_state, v2_data)
    known_versions = {"thompson_aimd_v1", "dts_pid_v1"}
    state = ChannelFeeState.from_v2_dict(fee_v2_data, db_state)
    if fee_v2_data.get("algorithm_version") not in known_versions:
        state.algorithm_version = "dts_pid_v1"
    return {
        "algorithm_version": state.algorithm_version,
        "last_revenue_rate": _r(state.last_revenue_rate),
        "last_fee_ppm": state.last_fee_ppm,
        "last_broadcast_fee_ppm": state.last_broadcast_fee_ppm,
        "last_update": state.last_update,
        "last_broadcast_at": state.last_broadcast_at,
        "last_state": state.last_state,
        "is_sleeping": state.is_sleeping,
        "sleep_until": state.sleep_until,
        "stable_cycles": state.stable_cycles,
        "forward_count_since_update": state.forward_count_since_update,
        "last_volume_sats": state.last_volume_sats,
        "last_gossip_refresh": state.last_gossip_refresh,
        "last_vegas_multiplier": _r(state.last_vegas_multiplier),
        "last_fee_profile": state.last_fee_profile,
        "last_context_key": state.last_context_key,
        "last_time_bucket": state.last_time_bucket,
        "last_corridor_role": state.last_corridor_role,
        "last_contextual_sample_used": state.last_contextual_sample_used,
        "dynamic_htlcmin_baseline_msat": state.dynamic_htlcmin_baseline_msat,
        "pid_kp": _r(state.pid.kp),
        "thompson_observations_count": len(state.thompson.observations),
    }


def _blob_case(name: str, channel_id: str, v2_data: dict, row_overrides=None) -> dict:
    row = _row_for(channel_id, **(row_overrides or {}))
    input_blob = json.dumps(v2_data)
    row_fields, merged_v2, python_roundtrip_blob = _merged_roundtrip(channel_id, row, input_blob)
    fee_state_pinned = _pinned_fee_state(channel_id, row, input_blob)
    return {
        "name": name,
        "channel_id": channel_id,
        "row": row,
        "input_blob": input_blob,
        "python_roundtrip_blob": python_roundtrip_blob,
        "fee_state_pinned": fee_state_pinned,
    }


def _current_fee_state_dict(**overrides) -> dict:
    d = {
        "algorithm_version": "dts_pid_v1",
        "thompson_state": GaussianThompsonState().to_dict(),
        "last_vegas_multiplier": 1.05,
        "last_gossip_refresh": NOW - 3600,
        "last_broadcast_at": NOW - 900,
        "pid_state": PIDState().to_dict(),
        "last_fee_profile": "active",
        "last_context_key": "mid:peak:P",
        "last_time_bucket": "peak",
        "last_corridor_role": "P",
        "last_contextual_sample_used": True,
        "dynamic_htlcmin_baseline_msat": 1000,
    }
    d.update(overrides)
    return d


def _current_cycle_state_dict(**overrides) -> dict:
    d = {
        "last_revenue_rate": 12.5,
        "last_fee_ppm": 250,
        "trend_direction": 1,
        "step_ppm": 50,
        "last_update": NOW - 900,
        "last_broadcast_at": NOW - 900,
        "consecutive_same_direction": 2,
        "is_sleeping": False,
        "sleep_until": 0,
        "stable_cycles": 1,
        "last_broadcast_fee_ppm": 240,
        "last_state": "balanced",
        "forward_count_since_update": 6,
        "last_volume_sats": 15000,
        "congestion_active": False,
        "congestion_quiet_cycles": 0,
        "congestion_entry_fee_ppm": 0,
        "pending_target_ppm": 0,
        "last_gossip_refresh": NOW - 3600,
        "dynamic_htlcmin_baseline_msat": 1000,
    }
    d.update(overrides)
    return d


def gen_v2_blob(outdir: Path) -> None:
    cases = []
    orig_time = time.time
    time.time = lambda: float(NOW)
    try:
        # 1: current nested-layout blob — both fee_state and cycle_state
        # nested, canonical shared scalars flat at top level (the shape
        # every live controller-written row has today).
        thompson1 = GaussianThompsonState()
        thompson1.update_posterior(250, 12.5, 6.0, time_bucket="peak")
        thompson1.update_contextual("mid:peak:P", 250, 12.5, time_bucket="peak")
        v2_1 = {
            "algorithm_version": "dts_pid_v1",
            "fee_state": _current_fee_state_dict(thompson_state=thompson1.to_dict()),
            "cycle_state": _current_cycle_state_dict(),
            "last_gossip_refresh": NOW - 3600,
            "last_broadcast_at": NOW - 900,
            "dynamic_htlcmin_baseline_msat": 1000,
        }
        cases.append(_blob_case("current_nested_layout", "chan_current_nested", v2_1))

        # 2: pre-nesting flat blob — thompson_state/pid_state/shared
        # scalars at the v2_data TOP level, no "fee_state"/"cycle_state"
        # keys at all (predates the nesting migration).
        thompson2 = GaussianThompsonState()
        thompson2.update_posterior(180, 3.0, 2.0, time_bucket="low")
        v2_2 = {
            "algorithm_version": "dts_pid_v1",
            "thompson_state": thompson2.to_dict(),
            "last_vegas_multiplier": 1.0,
            "last_gossip_refresh": NOW - 7200,
            "last_broadcast_at": NOW - 1800,
            "pid_state": PIDState().to_dict(),
            "dynamic_htlcmin_baseline_msat": None,
        }
        cases.append(_blob_case(
            "pre_nesting_flat_layout", "chan_flat_legacy", v2_2,
            row_overrides={"last_update": NOW - 1800},
        ))

        # 3: thompson_aimd_v1 legacy blob — flat layout, the pre-rename
        # algorithm_version marker, and a legacy thompson dict (no
        # weight_scheme key -> weight rescale on load).
        v2_3 = {
            "algorithm_version": "thompson_aimd_v1",
            "thompson_state": {
                "prior_mean_fee": 200,
                "prior_std_fee": 100,
                "observations": [
                    [220, 8.0, 0.1, NOW - 5400, "normal"],
                    [260, 0.0, 0.05, NOW - 9000, "low"],
                ],
            },
            "last_gossip_refresh": 0,
            "last_broadcast_at": NOW - 3600,
            "pid_state": PIDState().to_dict(),
            "dynamic_htlcmin_baseline_msat": None,
        }
        cases.append(_blob_case(
            "thompson_aimd_v1_legacy", "chan_aimd_legacy", v2_3,
            row_overrides={"last_update": NOW - 3600},
        ))

        # 4: 5-tuple-only observations (current time_bucket layout, no 6th
        # congestion/zero_probe flag anywhere).
        thompson4 = GaussianThompsonState()
        thompson4.update_posterior(300, 15.0, 4.0, time_bucket="peak")
        thompson4.update_posterior(310, 14.0, 4.0, time_bucket="normal")
        v2_4 = {
            "algorithm_version": "dts_pid_v1",
            "fee_state": _current_fee_state_dict(thompson_state=thompson4.to_dict()),
            "cycle_state": _current_cycle_state_dict(),
            "last_gossip_refresh": NOW - 3600,
            "last_broadcast_at": NOW - 900,
            "dynamic_htlcmin_baseline_msat": 1000,
        }
        cases.append(_blob_case("five_tuple_observations_only", "chan_five_tuple", v2_4))

        # 5: 6-tuple zero_probe/congestion-flagged observations alongside
        # plain 5-tuples.
        thompson5 = GaussianThompsonState()
        thompson5.update_posterior(250, 12.5, 0.8, time_bucket="normal", congested=True)
        thompson5.update_posterior(90, 0.0, 0.5, time_bucket="low")
        v2_5 = {
            "algorithm_version": "dts_pid_v1",
            "fee_state": _current_fee_state_dict(thompson_state=thompson5.to_dict()),
            "cycle_state": _current_cycle_state_dict(),
            "last_gossip_refresh": NOW - 3600,
            "last_broadcast_at": NOW - 900,
            "dynamic_htlcmin_baseline_msat": 1000,
        }
        cases.append(_blob_case("six_tuple_congestion_and_zero_probe", "chan_six_tuple", v2_5))

        # 6: 3-tuple (legacy) contextual posteriors alongside the current
        # 4-tuple layout.
        thompson6 = GaussianThompsonState()
        thompson6.contextual_posteriors["low:normal:P"] = (250.0, 45.0, 12)
        thompson6.contextual_posteriors["mid:peak:S"] = (310.0, 22.5, 8, NOW - 1000)
        v2_6 = {
            "algorithm_version": "dts_pid_v1",
            "fee_state": _current_fee_state_dict(thompson_state=thompson6.to_dict()),
            "cycle_state": _current_cycle_state_dict(),
            "last_gossip_refresh": NOW - 3600,
            "last_broadcast_at": NOW - 900,
            "dynamic_htlcmin_baseline_msat": 1000,
        }
        cases.append(_blob_case("three_tuple_contextual_posteriors", "chan_ctx_legacy", v2_6))

        # 7: missing pid_state AND missing algorithm_version inside
        # fee_state — from_v2_dict falls back to PIDState() and
        # `"migrated"`, which the controller then stamps `"dts_pid_v1"`
        # (unknown-version migration path).
        fee_state_7 = _current_fee_state_dict()
        del fee_state_7["pid_state"]
        del fee_state_7["algorithm_version"]
        v2_7 = {
            "algorithm_version": "dts_pid_v1",
            "fee_state": fee_state_7,
            "cycle_state": _current_cycle_state_dict(),
            "last_gossip_refresh": NOW - 3600,
            "last_broadcast_at": NOW - 900,
            "dynamic_htlcmin_baseline_msat": 1000,
        }
        cases.append(_blob_case("missing_pid_state", "chan_missing_pid", v2_7))

        # 8: missing cycle_state entirely — cycle payload built purely
        # from the flat db row columns + top-level shared scalars.
        v2_8 = {
            "algorithm_version": "dts_pid_v1",
            "fee_state": _current_fee_state_dict(),
            "last_gossip_refresh": NOW - 3600,
            "last_broadcast_at": NOW - 900,
            "dynamic_htlcmin_baseline_msat": 1000,
        }
        cases.append(_blob_case("missing_cycle_state", "chan_missing_cycle", v2_8))

        # 9: unknown keys injected at THREE levels (top of v2_data, inside
        # fee_state, inside thompson_state). The MERGED write shape drops
        # unrelated top-level junk by design (only algorithm_version/
        # fee_state/cycle_state/3 shared scalars are re-emitted) but
        # preserves fee_state's own unknown key (fee_payload = dict(nested))
        # and thompson_state's own unknown key (GaussianThompsonState.extra)
        # — the RAW parse->re-emit test (separate from this merge test)
        # preserves all three.
        thompson9 = GaussianThompsonState()
        thompson9_dict = thompson9.to_dict()
        thompson9_dict["future_thompson_field"] = {"nested": True}
        fee_state_9 = _current_fee_state_dict(thompson_state=thompson9_dict)
        fee_state_9["future_fee_field"] = [1, 2, 3]
        v2_9 = {
            "algorithm_version": "dts_pid_v1",
            "fee_state": fee_state_9,
            "cycle_state": _current_cycle_state_dict(),
            "last_gossip_refresh": NOW - 3600,
            "last_broadcast_at": NOW - 900,
            "dynamic_htlcmin_baseline_msat": 1000,
            "future_top_level_field": "dropped-by-merge",
        }
        cases.append(_blob_case("unknown_keys_three_levels", "chan_unknown_keys", v2_9))

        # 10: non-ASCII in last_context_key.
        v2_10 = {
            "algorithm_version": "dts_pid_v1",
            "fee_state": _current_fee_state_dict(last_context_key="café:☃:P"),
            "cycle_state": _current_cycle_state_dict(),
            "last_gossip_refresh": NOW - 3600,
            "last_broadcast_at": NOW - 900,
            "dynamic_htlcmin_baseline_msat": 1000,
        }
        cases.append(_blob_case("non_ascii_last_context_key", "chan_non_ascii", v2_10))
    finally:
        time.time = orig_time

    _write(outdir / "blobs.json", {"now": NOW, "cases": cases})

    # ------------------------------------------------------------------
    # merge_matrix: the explicit-shared-field resolution algorithm
    # (`_resolve_shared_field` + the fee/cycle caller-preference rule,
    # py 3799-3826). 9 cases = 3 caller combos (cycle-only, fee-only,
    # both) x 3 shared-field-provenance states on the PRIMARY caller
    # (explicit-and-different-from-persisted, explicit-but-EQUAL-to-the-
    # default value 0 [proves tracking is a real "was it assigned" flag,
    # not a truthiness/!= 0 check], untouched-at-construction-default).
    # When both callers are given, `cycle_state` is always primary (the
    # real function's rule) — the "both" cases additionally give
    # `fee_state` an EXPLICIT, DIFFERENT value (777) to prove it is
    # correctly ignored.
    persisted_value = 999
    persisted_v2 = {
        "algorithm_version": "dts_pid_v1",
        "fee_state": _current_fee_state_dict(last_gossip_refresh=persisted_value),
        "cycle_state": _current_cycle_state_dict(last_gossip_refresh=persisted_value),
        "last_gossip_refresh": persisted_value,
        "last_broadcast_at": NOW - 900,
        "dynamic_htlcmin_baseline_msat": 1000,
    }

    def _cycle_with(value, *, explicit):
        cs = ChannelCycleState()
        if explicit:
            cs.last_gossip_refresh = value  # tracked via __setattr__
        else:
            object.__setattr__(cs, "last_gossip_refresh", value)  # untracked
        return cs

    def _fee_with(value, *, explicit):
        fs = ChannelFeeState()
        if explicit:
            fs.last_gossip_refresh = value
        else:
            object.__setattr__(fs, "last_gossip_refresh", value)
        return fs

    matrix_cases = []
    field_states = [
        ("explicit_diff", 555, True),
        ("explicit_zero", 0, True),
        ("untouched_default", 0, False),
    ]
    for combo in ("cycle_only", "fee_only", "both"):
        for state_name, value, explicit in field_states:
            name = f"{combo}__{state_name}"
            cycle_state = None
            fee_state = None
            if combo in ("cycle_only", "both"):
                cycle_state = _cycle_with(value, explicit=explicit)
            if combo in ("fee_only", "both"):
                fee_state = _fee_with(777 if combo == "both" else value, explicit=True if combo == "both" else explicit)
            channel_id = f"merge_{name}"
            row = _row_for(channel_id)
            input_blob = json.dumps(persisted_v2)
            _, merged_v2, _ = _merged_roundtrip(
                channel_id, row, input_blob, cycle_state=cycle_state, fee_state=fee_state
            )
            matrix_cases.append({
                "name": name,
                "channel_id": channel_id,
                "row": row,
                "input_blob": input_blob,
                "cycle_state_value": (None if cycle_state is None else cycle_state.last_gossip_refresh),
                "cycle_state_explicit": (
                    None if cycle_state is None
                    else "last_gossip_refresh" in cycle_state.explicit_shared_fields()
                ),
                "fee_state_value": (None if fee_state is None else fee_state.last_gossip_refresh),
                "fee_state_explicit": (
                    None if fee_state is None
                    else "last_gossip_refresh" in fee_state.explicit_shared_fields()
                ),
                "expected_last_gossip_refresh": merged_v2["last_gossip_refresh"],
            })

    _write(outdir / "merge_matrix.json", {
        "now": NOW,
        "persisted_last_gossip_refresh": persisted_value,
        "cases": matrix_cases,
    })


SUITES = {
    "pyrand": gen_pyrand,
    "mat3": gen_mat3,
    "rails": gen_rails,
    "posterior": gen_posterior,
    "discount": gen_discount,
    "pid": gen_pid,
    "state_dict": gen_state_dict,
    "floors": gen_floors,
    "vegas": gen_vegas,
    "v2_blob": gen_v2_blob,
}


def gen_v2_prod_check(dump_path: Path, outdir: Path) -> None:
    """Task 9 Step 2 (production blobs — controller-run only): read the
    operator's `sqlite3 -readonly ... -json` export of `fee_strategy_state`
    (a JSON array of row dicts with exactly the SELECT's column names — see
    `p4-task-9-brief.md` / the Phase 4 plan's Task 9 section for the exact
    command) and, for EVERY row, compute the Python truth for the
    `_extract_fee_state_payload -> ChannelFeeState.from_v2_dict ->
    (unknown-version migration stamp) -> to_v2_dict -> json.dumps` path.
    Writes `<outdir>/expected_to_v2_dict.json`: `{channel_id: expected_json_string}`.
    `_extract_fee_state_payload` is a `@staticmethod` — no FeeController
    instance (and therefore no scripted `self.database`/`self.plugin`
    double) is needed for this path at all."""
    rows = json.loads(dump_path.read_text())
    if isinstance(rows, dict):
        rows = [rows]

    known_versions = {"thompson_aimd_v1", "dts_pid_v1"}
    expected = {}
    warnings = []
    for row in rows:
        channel_id = row.get("channel_id", "<unknown>")
        db_state = {k: v for k, v in row.items() if k != "v2_state_json"}
        raw = row.get("v2_state_json") or "{}"
        try:
            v2_data = json.loads(raw) if raw else {}
        except json.JSONDecodeError as e:
            warnings.append(f"{channel_id}: v2_state_json JSON parse failed: {e}")
            continue

        fee_v2_data = FeeController._extract_fee_state_payload(db_state, v2_data)
        if fee_v2_data.get("algorithm_version") not in known_versions:
            warnings.append(
                f"{channel_id}: unrecognized algorithm_version "
                f"{fee_v2_data.get('algorithm_version')!r} (migration path)"
            )
        state = ChannelFeeState.from_v2_dict(fee_v2_data, db_state)
        if fee_v2_data.get("algorithm_version") not in known_versions:
            state.algorithm_version = "dts_pid_v1"
        expected[channel_id] = json.dumps(state.to_v2_dict())

    outdir.mkdir(parents=True, exist_ok=True)
    _write(outdir / "expected_to_v2_dict.json", {"expected": expected, "warnings": warnings})
    if warnings:
        print(f"gen_v2_prod_check: {len(warnings)} warning(s):", file=sys.stderr)
        for w in warnings:
            print(f"  {w}", file=sys.stderr)


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "v2_prod_check":
        gen_v2_prod_check(Path(sys.argv[2]), Path(sys.argv[3]))
        return
    if len(sys.argv) != 3 or sys.argv[1] not in SUITES:
        names = "|".join(SUITES)
        print(f"usage: {sys.argv[0]} {{{names}}} <outdir>", file=sys.stderr)
        print(f"       {sys.argv[0]} v2_prod_check <fee_strategy_state.json> <outdir>", file=sys.stderr)
        sys.exit(1)
    SUITES[sys.argv[1]](Path(sys.argv[2]))


if __name__ == "__main__":
    main()
