#!/usr/bin/env python3
"""Generate bit-parity Kalman flow-filter fixtures for the Rust port.

Run from the repo root (~/bin/cl_revenue_ops-port):

    python3 tools/port/gen_kalman_fixtures.py <output.json>

The output feeds `crates/revops-analytics/src/kalman.rs` /
`crates/revops-analytics/tests/kalman.rs` in the Rust port
(cl-revenue-ops-r), committed there as `fixtures/kalman.json` (Phase 3 Task
3). Every value in this fixture comes from calling the REAL
`modules/flow_analysis.py` code (`KalmanFlowState`, `KalmanFlowFilter`,
`_calculate_kalman_volatility`, `_compute_raw_kalman_observation`,
`_calculate_confidence`, `estimate_depletion_hours`) -- nothing here is
hand-computed. Floats are pinned as `struct.pack('<d', v).hex()` bit
patterns so the Rust replay test can assert `f64::to_bits` equality with no
epsilon.

Ground-truth notes (confirmed by running the actual code, not by reading
the brief in isolation):

- `KalmanFlowState._safe()` (I-7 fix) only substitutes a field's default
  when the key is MISSING (`d.get(key) is None`); a key that is PRESENT
  but non-finite (NaN/Inf) passes straight through unchanged. A
  NaN-poisoned-but-present dict does NOT get defaulted by `from_dict`
  alone -- the NaN only gets scrubbed later, if/when the filter's own
  `_has_nan()` guard fires inside `predict()`/`update()`. This fixture
  therefore encodes "missing vs present-and-nonfinite" as two distinct
  cases and lets the real `from_dict()` output be the arbiter, rather than
  asserting the (incorrect) "non-finite always defaults" shorthand.
- `KalmanFlowFilter.update()` has NO upfront `_has_nan()` guard (unlike
  `predict()`): a NaN already present in the incoming state (e.g. a
  poisoned `covariance`) propagates through the whole update arithmetic and
  is only caught by the NaN check at the very end, which resets the state
  to fresh defaults and returns 0.0 -- captured here as
  `update_nan_state_reset`.
- `_compute_raw_kalman_observation` and `_calculate_confidence` call
  `time.time()` internally for their "now" reference; this generator
  monkeypatches `flow_analysis.time.time` to a fixed value per case so the
  Rust port's injected `now` parameter reproduces the exact same number.
"""
import json
import math
import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules import flow_analysis as fa  # noqa: E402


def bits(v: float) -> str:
    return struct.pack("<d", float(v)).hex()


def state_dict(state: "fa.KalmanFlowState") -> dict:
    d = state.to_dict()
    return {
        "flow_ratio": bits(d["flow_ratio"]),
        "flow_velocity": bits(d["flow_velocity"]),
        "variance_ratio": bits(d["variance_ratio"]),
        "variance_velocity": bits(d["variance_velocity"]),
        "covariance": bits(d["covariance"]),
        "last_update": int(d["last_update"]),
        "innovation_variance": bits(d["innovation_variance"]),
        "last_innovation": bits(d["last_innovation"]),
        "observation_count": int(d["observation_count"]),
    }


def take_nan_recovery(kf: "fa.KalmanFlowFilter") -> int:
    """Mirror the Rust `take_nan_recovery_count`: read then reset to 0."""
    n = kf._nan_recovery_count
    kf._nan_recovery_count = 0
    return n


def run_predict(kf, dt_hours, volatility):
    kf.predict(dt_hours, volatility)
    return {
        "kind": "predict",
        "dt_hours": bits(dt_hours),
        "volatility": bits(volatility),
        "state_after": state_dict(kf.state),
        "nan_recovery_delta": take_nan_recovery(kf),
    }


def run_update(kf, observed_ratio, confidence, now):
    fa.time.time = lambda: float(now)
    innovation = kf.update(observed_ratio, confidence)
    return {
        "kind": "update",
        "observed_ratio": bits(observed_ratio),
        "confidence": bits(confidence),
        "now": int(now),
        "innovation": bits(innovation),
        "state_after": state_dict(kf.state),
        "nan_recovery_delta": take_nan_recovery(kf),
    }


CONSTS = (
    "KALMAN_MAX_VELOCITY", "KALMAN_MIN_VELOCITY",
    "KALMAN_BASE_PROCESS_NOISE", "KALMAN_VELOCITY_PROCESS_NOISE",
    "KALMAN_MIN_PROCESS_NOISE", "KALMAN_MAX_PROCESS_NOISE",
    "KALMAN_BASE_MEASUREMENT_NOISE", "KALMAN_MIN_MEASUREMENT_NOISE",
    "KALMAN_MAX_MEASUREMENT_NOISE", "KALMAN_INITIAL_VARIANCE",
    "KALMAN_VOLATILITY_SCALING", "KALMAN_CONFIDENCE_SCALING",
    "KALMAN_CONVERGENCE_UNCERTAINTY",
    "MIN_CONFIDENCE", "MAX_CONFIDENCE",
    "CONFIDENCE_RECENCY_HALFLIFE_DAYS",
    "DEPLETION_MIN_DRAIN_SATS_PER_DAY",
)
INT_CONSTS = ("KALMAN_MIN_OBSERVATIONS", "KALMAN_MIN_UPDATE_INTERVAL_SECS",
              "MIN_FORWARDS_FOR_HIGH_CONFIDENCE")


def gen_constants():
    out = {name: bits(getattr(fa, name)) for name in CONSTS}
    out.update({name: int(getattr(fa, name)) for name in INT_CONSTS})
    return out


def gen_sequences(rng: random.Random, n_sequences: int) -> list:
    sequences = []
    fake_now = 1_800_000_000
    for seq_idx in range(n_sequences):
        fresh = seq_idx % 3 != 0  # 2/3 fresh, 1/3 persisted-state
        if fresh:
            kf = fa.KalmanFlowFilter()
            initial_state = None
        else:
            st = fa.KalmanFlowState(
                flow_ratio=rng.uniform(-0.9, 0.9),
                flow_velocity=rng.uniform(fa.KALMAN_MIN_VELOCITY, fa.KALMAN_MAX_VELOCITY),
                variance_ratio=rng.uniform(1e-3, 0.3),
                variance_velocity=rng.uniform(1e-3, 0.3),
                covariance=rng.uniform(-0.05, 0.05),
                last_update=fake_now,
                innovation_variance=rng.uniform(0.001, 0.5),
                last_innovation=rng.uniform(-0.5, 0.5),
                observation_count=rng.randint(0, 50),
            )
            kf = fa.KalmanFlowFilter(st)
            initial_state = state_dict(kf.state)

        steps = []
        n_steps = rng.randint(8, 14)
        for _ in range(n_steps):
            fake_now += rng.randint(1, 100_000)
            if rng.random() < 0.45:
                dt_hours = rng.uniform(0.01, 200.0)
                volatility = rng.uniform(0.4, 2.2)
                steps.append(run_predict(kf, dt_hours, volatility))
            else:
                roll = rng.random()
                if roll < 0.08:
                    observed_ratio = float("nan")
                elif roll < 0.14:
                    observed_ratio = float("inf") if rng.random() < 0.5 else float("-inf")
                else:
                    observed_ratio = rng.uniform(-1.5, 1.5)
                confidence = rng.uniform(0.0, 1.2)
                steps.append(run_update(kf, observed_ratio, confidence, fake_now))

        sequences.append({
            "id": f"seq{seq_idx:03d}",
            "fresh": fresh,
            "initial_state": initial_state,
            "steps": steps,
        })
    return sequences


def gen_hand_cases():
    cases = {}

    # dt<=0 no-op (both zero and negative dt): state untouched.
    for label, dt in (("dt_zero_noop", 0.0), ("dt_negative_noop", -5.0)):
        st = fa.KalmanFlowState(flow_ratio=0.37, flow_velocity=0.004,
                                 variance_ratio=0.08, variance_velocity=0.02,
                                 covariance=0.01, last_update=1_800_000_000,
                                 innovation_variance=0.05, last_innovation=0.02,
                                 observation_count=9)
        kf = fa.KalmanFlowFilter(st)
        before = state_dict(kf.state)
        step = run_predict(kf, dt, 1.3)
        cases[label] = {"initial_state": before, "step": step}

    # NaN already present in state -> predict() resets before computing.
    st = fa.KalmanFlowState(flow_ratio=0.1, flow_velocity=0.0,
                             variance_ratio=0.05, variance_velocity=0.05,
                             covariance=float("nan"), last_update=1_800_000_000,
                             innovation_variance=0.02, last_innovation=0.0,
                             observation_count=4)
    kf = fa.KalmanFlowFilter(st)
    before = state_dict_allow_nan(kf.state)
    step = run_predict(kf, 2.0, 1.0)
    cases["predict_nan_state_reset"] = {"initial_state": before, "step": step}

    # NaN already present in state -> update() has no upfront guard, NaN
    # propagates through the whole computation and is only caught by the
    # final _has_nan() check, which resets to defaults and returns 0.0.
    st = fa.KalmanFlowState(flow_ratio=0.2, flow_velocity=0.01,
                             variance_ratio=0.05, variance_velocity=0.05,
                             covariance=float("nan"), last_update=1_800_001_000,
                             innovation_variance=0.02, last_innovation=0.1,
                             observation_count=3)
    kf = fa.KalmanFlowFilter(st)
    before = state_dict_allow_nan(kf.state)
    step = run_update(kf, 0.3, 0.7, 1_800_005_000)
    cases["update_nan_state_reset"] = {"initial_state": before, "step": step}

    # det<=0 covariance clamp: a stale, disproportionately large covariance
    # relative to tiny variances propagates through the covariance-predict
    # equations to a state with non-positive determinant, forcing
    # _ensure_positive_definite's clamp branch (verified empirically: the
    # resulting covariance sits at the +/-0.9*sqrt(p00*p11) bound, not the
    # raw propagated value).
    st = fa.KalmanFlowState(flow_ratio=0.1, flow_velocity=0.0,
                             variance_ratio=1e-4, variance_velocity=1e-4,
                             covariance=0.5, last_update=1_800_000_000,
                             innovation_variance=0.02, last_innovation=0.0,
                             observation_count=1)
    kf = fa.KalmanFlowFilter(st)
    before = state_dict(kf.state)
    step = run_predict(kf, 0.01, 0.4)
    p00 = struct.unpack("<d", bytes.fromhex(step["state_after"]["variance_ratio"]))[0]
    p01 = struct.unpack("<d", bytes.fromhex(step["state_after"]["covariance"]))[0]
    p11 = struct.unpack("<d", bytes.fromhex(step["state_after"]["variance_velocity"]))[0]
    assert p00 * p11 - p01 * p01 > 0, "expected clamp to restore a positive determinant"
    assert abs(abs(p01) - 0.9 * math.sqrt(p00 * p11)) < 1e-9, "expected covariance pinned to the clamp bound"
    cases["det_nonpositive_clamp"] = {"initial_state": before, "step": step}

    # Observation NaN/Inf: state must be completely untouched, return 0.0.
    for label, obs in (("observation_nan_no_touch", float("nan")),
                       ("observation_pos_inf_no_touch", float("inf")),
                       ("observation_neg_inf_no_touch", float("-inf"))):
        st = fa.KalmanFlowState(flow_ratio=0.42, flow_velocity=-0.003,
                                 variance_ratio=0.06, variance_velocity=0.04,
                                 covariance=-0.01, last_update=1_800_000_000,
                                 innovation_variance=0.03, last_innovation=0.15,
                                 observation_count=11)
        kf = fa.KalmanFlowFilter(st)
        before = state_dict(kf.state)
        step = run_update(kf, obs, 0.9, 1_800_010_000)
        assert step["state_after"] == before, "observation reject must not touch state"
        assert step["innovation"] == bits(0.0)
        cases[label] = {"initial_state": before, "step": step}

    return cases


def state_dict_allow_nan(state) -> dict:
    """Like state_dict but for states we deliberately poisoned with NaN --
    used only as the *initial_state* snapshot for hand cases (never as an
    output truth value, since NaN payload bits are canonical-but-arbitrary;
    Rust reconstructs the same input state from these bits directly)."""
    d = state.to_dict()
    return {k: (bits(v) if isinstance(v, float) else int(v)) for k, v in d.items()}


def gen_state_dict_roundtrip_cases():
    """from_dict()/to_dict() roundtrip cases exercising the real `_safe`
    semantics: missing key -> that key's default; present-but-non-finite ->
    passes through unchanged (ground truth, see module docstring)."""
    cases = []

    # Fully populated dict: everything should pass straight through.
    full = {
        "flow_ratio": 0.33, "flow_velocity": 0.002, "variance_ratio": 0.07,
        "variance_velocity": 0.03, "covariance": -0.004, "last_update": 1_800_020_000,
        "innovation_variance": 0.04, "last_innovation": -0.11, "observation_count": 6,
    }
    cases.append(_roundtrip_case("fully_populated", full))

    # Missing keys entirely -> defaults for each.
    missing_all = {}
    cases.append(_roundtrip_case("all_missing_defaults", missing_all))

    # Some keys missing, others present (incl. an explicit 0.0, which the
    # I-7 fix must NOT treat as absent).
    partial = {"flow_ratio": 0.0, "covariance": 0.0, "observation_count": 0}
    cases.append(_roundtrip_case("partial_with_explicit_zero", partial))

    # Present-but-non-finite values pass through unchanged (NOT defaulted --
    # this is the ground-truth-corrected case; see module docstring).
    poisoned = {
        "flow_ratio": float("nan"), "variance_ratio": float("inf"),
        "covariance": float("-inf"), "last_update": 1_800_030_000,
        "innovation_variance": float("nan"),
    }
    cases.append(_roundtrip_case("present_nonfinite_passes_through", poisoned,
                                  allow_nan_in_expected=True))

    # last_update / observation_count explicit None value in the dict (as
    # opposed to the key being absent) -- `d.get(key) or 0` treats an
    # explicit None the same as absent (falsy), landing on the default 0.
    explicit_none_ints = {"last_update": None, "observation_count": None,
                           "flow_ratio": 0.5}
    cases.append(_roundtrip_case("explicit_none_ints_default_to_zero", explicit_none_ints))

    return cases


def _roundtrip_case(label, input_dict, allow_nan_in_expected=False):
    state = fa.KalmanFlowState.from_dict(input_dict)
    encoded_input = {}
    for k, v in input_dict.items():
        if v is None:
            encoded_input[k] = None
        elif isinstance(v, float):
            encoded_input[k] = {"bits": bits(v)}
        else:
            encoded_input[k] = v
    expected = state_dict_allow_nan(state) if allow_nan_in_expected else state_dict(state)
    return {"label": label, "input": encoded_input, "expected_state": expected}


def gen_volatility_cases():
    cases = []
    bucket_sets = [
        [],
        [{"out": 10, "in": 5}],
        [{"out": 10, "in": 5}, {"out": 12, "in": 4}],  # still < 3 buckets
        [{"out": 100, "in": 90}, {"out": 110, "in": 85}, {"out": 90, "in": 95}],
        [{"out": 500, "in": 480}, {"out": 520, "in": 470}, {"out": 300, "in": 600},
         {"out": 700, "in": 200}],
        [{"out": 5, "in": 3}, {"out": 4, "in": 6}, {"out": 3, "in": 2}],  # mean_flow < 1000
        [{"out": 10_000, "in": 9_000}, {"out": 10_500, "in": 8_800},
         {"out": 9_500, "in": 9_200}, {"out": 15_000, "in": 3_000},
         {"out": 2_000, "in": 18_000}],
        [{"out": 0, "in": 0}, {"out": 0, "in": 0}, {"out": 0, "in": 0}],
        [{"in": 7}, {"out": 3}, {}],  # missing keys default via `.get(x, 0) or 0`
    ]
    for i, buckets in enumerate(bucket_sets):
        vol = fa.FlowAnalyzer._calculate_kalman_volatility(None, buckets)
        cases.append({
            "id": f"vol{i:02d}",
            "daily_buckets": [{"out": b.get("out", 0), "in": b.get("in", 0)} for b in buckets],
            "volatility": bits(vol),
        })
    return cases


def gen_raw_observation_cases(rng: random.Random):
    cases = []
    now = 1_900_000_000.0

    def mk(label, capacity, entries, now_val):
        fa.time.time = lambda: now_val
        ratio, count = fa.FlowAnalyzer._compute_raw_kalman_observation(None, "chan", capacity, entries)
        return {
            "id": label,
            "capacity": capacity,
            "entries": [{"timestamp": bits(e["timestamp"]), "net_msat": int(e["net_msat"])} for e in entries],
            "now": bits(now_val),
            "ratio": bits(ratio),
            "count": count,
        }

    cases.append(mk("empty_entries", 1_000_000, [], now))
    cases.append(mk("zero_capacity", 0, [{"timestamp": now - 100, "net_msat": 5000}], now))
    cases.append(mk("negative_capacity", -10, [{"timestamp": now - 100, "net_msat": 5000}], now))
    cases.append(mk("all_outside_window", 1_000_000,
                    [{"timestamp": now - 90_000, "net_msat": 500_000},
                     {"timestamp": now - 200_000, "net_msat": -300_000}], now))
    cases.append(mk("mixed_window", 5_000_000,
                    [{"timestamp": now - 1000, "net_msat": 300_000},
                     {"timestamp": now - 50_000, "net_msat": -1_200_000},  # outside 24h
                     {"timestamp": now - 3600, "net_msat": -150_000},
                     {"timestamp": now - 86_399, "net_msat": 900_000}], now))
    cases.append(mk("saturating_positive", 100_000,
                    [{"timestamp": now - 10, "net_msat": v} for v in (5_000_000, 6_000_000, 7_000_000)], now))
    cases.append(mk("saturating_negative", 100_000,
                    [{"timestamp": now - 10, "net_msat": -v} for v in (5_000_000, 6_000_000, 7_000_000)], now))
    cases.append(mk("boundary_at_86400", 2_000_000,
                    [{"timestamp": now - 86_400, "net_msat": 10_000},
                     {"timestamp": now - 86_401, "net_msat": 10_000}], now))

    # A handful of randomized cases for broader arithmetic coverage.
    for i in range(20):
        capacity = rng.randint(100_000, 50_000_000)
        n_entries = rng.randint(0, 8)
        entries = []
        for _ in range(n_entries):
            age = rng.uniform(0, 172_800)
            entries.append({"timestamp": now - age, "net_msat": rng.randint(-5_000_000, 5_000_000)})
        cases.append(mk(f"rand{i:02d}", capacity, entries, now))

    return cases


def gen_confidence_cases(rng: random.Random):
    cases = []
    now = 1_950_000_000

    def mk(label, forward_count, last_forward_ts, now_val):
        fa.time.time = lambda: float(now_val)
        conf = fa.FlowAnalyzer._calculate_confidence(None, forward_count, last_forward_ts)
        return {
            "id": label,
            "forward_count": forward_count,
            "last_forward_ts": last_forward_ts,
            "now": now_val,
            "confidence": bits(conf),
        }

    cases.append(mk("zero_forwards_never_routed", 0, 0, now))
    cases.append(mk("below_high_confidence_threshold", 5, now - 3600, now))
    cases.append(mk("exactly_high_confidence_threshold", 20, now - 3600, now))
    cases.append(mk("above_high_confidence_threshold", 45, now, now))
    cases.append(mk("stale_recency_one_halflife", 20, now - int(3.0 * 86400), now))
    cases.append(mk("stale_recency_many_halflives", 20, now - int(30.0 * 86400), now))
    cases.append(mk("last_forward_in_future_negative_days", 10, now + 3600, now))
    for i in range(15):
        forward_count = rng.randint(0, 60)
        days_ago = rng.uniform(0, 60)
        last_ts = now - int(days_ago * 86400)
        cases.append(mk(f"rand{i:02d}", forward_count, last_ts, now))
    return cases


def gen_depletion_cases(rng: random.Random):
    cases = []

    def mk(label, local_sats, capacity_sats, kalman_ratio, kalman_velocity):
        hours = fa.estimate_depletion_hours(local_sats, capacity_sats, kalman_ratio, kalman_velocity)
        return {
            "id": label,
            "local_sats": bits(local_sats) if isinstance(local_sats, float) else local_sats,
            "capacity_sats": bits(capacity_sats) if isinstance(capacity_sats, float) else capacity_sats,
            "kalman_ratio": bits(kalman_ratio) if isinstance(kalman_ratio, float) else kalman_ratio,
            "kalman_velocity": bits(kalman_velocity) if isinstance(kalman_velocity, float) else kalman_velocity,
            "hours": (bits(hours) if hours is not None else None),
        }

    cases.append(mk("audit_table_example", 5_000_000.0, 10_000_000.0, 0.12, 0.0))
    cases.append(mk("zero_capacity_none", 1000.0, 0.0, 0.5, 0.0))
    cases.append(mk("negative_capacity_none", 1000.0, -5.0, 0.5, 0.0))
    cases.append(mk("negative_local_none", -1.0, 10_000.0, 0.5, 0.0))
    cases.append(mk("nan_ratio_none", 1000.0, 10_000.0, float("nan"), 0.0))
    cases.append(mk("inf_velocity_none", 1000.0, 10_000.0, 0.1, float("inf")))
    cases.append(mk("below_min_drain_none", 10.0, 10_000.0, 0.00001, 0.0))
    cases.append(mk("negative_ratio_only_velocity_drains", 1000.0, 10_000.0, -0.5, 0.05))
    cases.append(mk("velocity_makes_net_negative_clamped_to_zero_none", 100.0, 10_000.0, 0.0, -1.0))
    cases.append(mk("zero_local_zero_hours", 0.0, 10_000.0, 0.2, 0.0))
    cases.append(mk("large_positive_velocity_amplifies_drain", 100_000.0, 10_000_000.0, 0.05, 0.02))
    for i in range(15):
        local = rng.uniform(0, 20_000_000)
        capacity = rng.uniform(100_000, 20_000_000)
        ratio = rng.uniform(-1.0, 1.0)
        velocity = rng.uniform(fa.KALMAN_MIN_VELOCITY, fa.KALMAN_MAX_VELOCITY)
        cases.append(mk(f"rand{i:02d}", local, capacity, ratio, velocity))
    return cases


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <output.json>", file=sys.stderr)
        sys.exit(1)

    real_time = fa.time.time
    try:
        rng = random.Random(20260717)
        out = {
            "meta": {"seed": 20260717, "source": "modules/flow_analysis.py"},
            "constants": gen_constants(),
            "sequences": gen_sequences(rng, n_sequences=26),
            "hand_cases": gen_hand_cases(),
            "state_dict_roundtrip": gen_state_dict_roundtrip_cases(),
            "volatility_cases": gen_volatility_cases(),
            "raw_observation_cases": gen_raw_observation_cases(rng),
            "confidence_cases": gen_confidence_cases(rng),
            "depletion_cases": gen_depletion_cases(rng),
        }
    finally:
        fa.time.time = real_time

    total_steps = sum(len(s["steps"]) for s in out["sequences"])
    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {sys.argv[1]} ({len(out['sequences'])} sequences, "
          f"{total_steps} steps, {len(out['hand_cases'])} hand cases, "
          f"{len(out['state_dict_roundtrip'])} roundtrip cases, "
          f"{len(out['volatility_cases'])} volatility cases, "
          f"{len(out['raw_observation_cases'])} raw-observation cases, "
          f"{len(out['confidence_cases'])} confidence cases, "
          f"{len(out['depletion_cases'])} depletion cases)")


if __name__ == "__main__":
    main()
