#!/usr/bin/env python3
"""Generate bit-parity flow-pipeline + demand-flow fixtures for the Rust port.

Run from the repo root (~/bin/cl_revenue_ops-port):

    python3 tools/port/gen_flow_fixtures.py <output.json>

Feeds `crates/revops-analytics/src/flow.rs` / `demand_flow.rs` and their
`tests/flow.rs` / `tests/demand_flow.rs` in the Rust port
(cl-revenue-ops-r), committed there as `fixtures/flow.json` (Phase 3 Task
7). Every value comes from calling the REAL `modules/flow_analysis.py` /
`modules/demand_flow.py` code -- nothing here is hand-computed. Floats are
pinned as `struct.pack('<d', v).hex()` bit patterns (Rust asserts
`f64::to_bits` equality), EXCEPT the numpy-derived `TemporalProfile`
fields (`burstiness`, `diurnal_strength`), which are stored as plain JSON
numbers for an epsilon comparison -- see the "Numpy-derived TemporalProfile
fields" note in `flow.rs`'s module doc comment for why those two are not
claimed to be bit-reproducible by a from-scratch Rust reduction.

Ground-truth notes (confirmed by running the actual code):

- `FlowAnalyzer._apply_kalman_filter` / `_apply_kalman_reclassification`
  are instance methods, but neither touches `self.database` on the path
  exercised here: passing a list for `pending_kalman_saves` makes the
  per-channel DB write branch dead code (state is queued into the list
  instead), and `_compute_raw_kalman_observation` doesn't read `self` at
  all. A minimal duck-typed `FakeAnalyzer` (a `_kalman_lock`, a
  `_kalman_filters` dict this script pre-seeds directly instead of going
  through `_get_kalman_filter`'s DB-load path, a no-op `plugin.log`, and a
  `config`/`database.get_fee_strategy_state` stub for the DTS-widening
  cases) is enough to call the real bound methods and get real Python
  arithmetic back.
- The "step" (Untouched/PredictOnly/Updated) isn't a separate return value
  in Python -- it's reconstructed here exactly like the Rust port does:
  `state_snapshot is None` (nothing appended to `pending_kalman_saves`) is
  Untouched; otherwise Updated iff `has_observation`, else PredictOnly.
- `DemandFlowClassifier`'s three methods are fully stateless (no `self.*`
  reads at all) -- called directly, no stubbing needed.
"""
import json
import math
import os
import struct
import sys
import threading
import time as time_module
import types
from random import Random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules import flow_analysis as fa  # noqa: E402
from modules import demand_flow as df  # noqa: E402


def bits(v: float) -> str:
    return struct.pack("<d", float(v)).hex()


# =============================================================================
# Constants
# =============================================================================

F64_CONSTS = ("MAX_VELOCITY", "MIN_VELOCITY", "VELOCITY_OUTLIER_THRESHOLD",
              "BASE_EMA_DECAY", "DECAY_RANGE", "TEMPORAL_EMA_ALPHA")
INT_CONSTS = ("TEMPORAL_GRADUATION_DAYS", "TEMPORAL_MIN_DAILY_FORWARDS")


def gen_constants():
    out = {name: bits(getattr(fa, name)) for name in F64_CONSTS}
    out.update({name: int(getattr(fa, name)) for name in INT_CONSTS})
    return out


# =============================================================================
# _calculate_velocity
# =============================================================================

def gen_velocity_cases(rng: Random):
    cases = []
    now = 1_900_000_000

    def mk(label, flow_ratio, previous_ratio, previous_timestamp, now_val):
        fa.time.time = lambda: float(now_val)
        v = fa.FlowAnalyzer._calculate_velocity(None, flow_ratio, previous_ratio, previous_timestamp)
        return {
            "id": label,
            "flow_ratio": bits(flow_ratio),
            "previous_ratio": bits(previous_ratio),
            "previous_timestamp": previous_timestamp,
            "now": now_val,
            "velocity": bits(v),
        }

    cases.append(mk("no_previous_timestamp", 0.3, 0.1, 0, now))
    cases.append(mk("negative_previous_timestamp", 0.3, 0.1, -5, now))
    cases.append(mk("too_recent_below_30min", 0.3, 0.1, now - 1000, now))
    cases.append(mk("exactly_30min_boundary", 0.3, 0.1, now - 1800, now))
    cases.append(mk("normal_positive_velocity", 0.4, 0.1, now - 3600, now))
    cases.append(mk("normal_negative_velocity", 0.1, 0.4, now - 3600, now))
    cases.append(mk("outlier_clamped_positive", 0.9, -0.9, now - 3600, now))
    cases.append(mk("outlier_clamped_negative", -0.9, 0.9, now - 3600, now))
    cases.append(mk("zero_flow_ratio_floor_applies", 0.02, 0.0, now - 3600, now))
    cases.append(mk("bound_hits_max_velocity", 1.0, -1.0, now - 3600, now))
    cases.append(mk("bound_hits_min_velocity", -1.0, 1.0, now - 3600, now))
    for i in range(20):
        flow_ratio = rng.uniform(-1.0, 1.0)
        previous_ratio = rng.uniform(-1.0, 1.0)
        previous_timestamp = now - rng.randint(1800, 500_000)
        cases.append(mk(f"rand{i:02d}", flow_ratio, previous_ratio, previous_timestamp, now))
    return cases


# =============================================================================
# _calculate_adaptive_decay
# =============================================================================

def gen_decay_cases(rng: Random):
    cases = []

    def mk(label, buckets):
        decay = fa.FlowAnalyzer._calculate_adaptive_decay(None, buckets)
        return {
            "id": label,
            "daily_buckets": [{"out": b.get("out", 0), "in": b.get("in", 0)} for b in buckets],
            "decay": bits(decay),
        }

    cases.append(mk("empty", []))
    cases.append(mk("one_bucket", [{"out": 100, "in": 50}]))
    cases.append(mk("two_buckets", [{"out": 100, "in": 50}, {"out": 90, "in": 60}]))
    cases.append(mk("three_buckets_low_activity_below_1000", [
        {"out": 5, "in": 3}, {"out": 4, "in": 6}, {"out": 3, "in": 2}]))
    cases.append(mk("stable_low_volatility", [
        {"out": 10_000, "in": 9_900}, {"out": 10_050, "in": 9_950}, {"out": 9_980, "in": 9_890},
        {"out": 10_010, "in": 9_920}, {"out": 9_995, "in": 9_905}]))
    cases.append(mk("volatile_high_volatility", [
        {"out": 50_000, "in": 1_000}, {"out": 1_000, "in": 60_000}, {"out": 40_000, "in": 2_000},
        {"out": 2_000, "in": 45_000}]))
    cases.append(mk("moderate_interpolated", [
        {"out": 20_000, "in": 15_000}, {"out": 18_000, "in": 17_000}, {"out": 22_000, "in": 14_000},
        {"out": 19_500, "in": 16_200}]))
    cases.append(mk("missing_keys_default_zero", [{"in": 5000}, {"out": 3000}, {}]))
    cases.append(mk("all_zero_buckets", [{"out": 0, "in": 0}] * 5))
    for i in range(20):
        n = rng.randint(3, 12)
        buckets = [{"out": rng.randint(0, 200_000), "in": rng.randint(0, 200_000)} for _ in range(n)]
        cases.append(mk(f"rand{i:02d}", buckets))
    return cases


# =============================================================================
# _calculate_ema_flow
# =============================================================================

def gen_ema_cases(rng: Random):
    cases = []

    def mk(label, buckets, decay_factor):
        ema_in, ema_out, total_in, total_out, forward_count, last_forward_ts = \
            fa.FlowAnalyzer._calculate_ema_flow(None, buckets, decay_factor)
        return {
            "id": label,
            "daily_buckets": [
                {"in": b.get("in", 0), "out": b.get("out", 0), "count": b.get("count", 0),
                 "last_ts": b.get("last_ts", 0)}
                for b in buckets
            ],
            "decay_factor": bits(decay_factor),
            "ema_in": bits(ema_in),
            "ema_out": bits(ema_out),
            "total_in": total_in,
            "total_out": total_out,
            "forward_count": forward_count,
            "last_forward_ts": last_forward_ts,
        }

    cases.append(mk("empty", [], 0.8))
    cases.append(mk("single_bucket", [{"in": 100, "out": 200, "count": 3, "last_ts": 1_900_000_000}], 0.8))
    cases.append(mk("missing_keys_default_zero", [{}, {"in": 50}, {"out": 20, "count": 2}], 0.8))
    cases.append(mk("decay_065_fast", [
        {"in": 1000, "out": 2000, "count": 5, "last_ts": 1_900_090_000},
        {"in": 900, "out": 1800, "count": 4, "last_ts": 1_900_000_000},
        {"in": 800, "out": 1600, "count": 3, "last_ts": 1_899_900_000},
    ], 0.65))
    cases.append(mk("decay_095_slow", [
        {"in": 1000, "out": 2000, "count": 5, "last_ts": 1_900_090_000},
        {"in": 900, "out": 1800, "count": 4, "last_ts": 1_900_000_000},
        {"in": 800, "out": 1600, "count": 3, "last_ts": 1_899_900_000},
    ], 0.95))
    cases.append(mk("last_ts_not_monotonic_takes_max_across_all", [
        {"in": 100, "out": 50, "count": 1, "last_ts": 1_000},
        {"in": 100, "out": 50, "count": 1, "last_ts": 5_000},
        {"in": 100, "out": 50, "count": 1, "last_ts": 2_000},
    ], 0.8))
    cases.append(mk("seven_day_window", [
        {"in": rng.randint(0, 500_000), "out": rng.randint(0, 500_000), "count": rng.randint(0, 40),
         "last_ts": 1_900_000_000 - age * 80_000}
        for age in range(7)
    ], 0.8))
    for i in range(20):
        n = rng.randint(1, 20)
        decay_factor = rng.uniform(0.65, 0.95)
        buckets = [
            {"in": rng.randint(0, 1_000_000), "out": rng.randint(0, 1_000_000),
             "count": rng.randint(0, 100), "last_ts": 1_900_000_000 - age * rng.randint(1000, 90_000)}
            for age in range(n)
        ]
        cases.append(mk(f"rand{i:02d}", buckets, decay_factor))
    return cases


# =============================================================================
# apply_kalman_filter (the 300s no-touch gate + first-run dt / cap-168h)
# =============================================================================

class _FakeDatabase:
    def __init__(self, fee_state=None):
        self._fee_state = fee_state

    def get_kalman_state(self, channel_id):
        return None  # _kalman_filters is pre-seeded directly; never consulted.

    def get_fee_strategy_state(self, channel_id):
        return self._fee_state

    def save_kalman_state(self, channel_id, state):
        raise AssertionError("save_kalman_state must not be reached when pending_kalman_saves is a list")


class _FakePlugin:
    def log(self, *_args, **_kwargs):
        pass


class _FakeConfig:
    source_threshold = 0.05
    sink_threshold = -0.05


def make_fake_analyzer(channel_id, initial_state=None, fee_state=None):
    fake = types.SimpleNamespace()
    fake._kalman_lock = threading.Lock()
    kf = fa.KalmanFlowFilter(initial_state) if initial_state is not None else fa.KalmanFlowFilter()
    fake._kalman_filters = {channel_id: kf}
    fake.plugin = _FakePlugin()
    fake.database = _FakeDatabase(fee_state=fee_state)
    fake.config = _FakeConfig()
    # `_apply_kalman_filter` calls `self._get_kalman_filter(channel_id)` --
    # the channel is always pre-seeded above, so this just mirrors
    # `_get_kalman_filter`'s fast path (`if channel_id in
    # self._kalman_filters: return it`) without needing a real DB.
    fake._get_kalman_filter = lambda cid: fake._kalman_filters[cid]
    # The remaining `self.*` calls inside `_apply_kalman_filter` /
    # `_apply_kalman_reclassification` are all methods that never read
    # `self` internally (verified by inspection) -- delegate to the real
    # unbound implementations with a dummy `self=None` rather than
    # reimplementing them.
    fake._calculate_kalman_volatility = (
        lambda daily_buckets: fa.FlowAnalyzer._calculate_kalman_volatility(None, daily_buckets)
    )
    fake._calculate_confidence = (
        lambda forward_count, last_forward_ts: fa.FlowAnalyzer._calculate_confidence(
            None, forward_count, last_forward_ts)
    )
    fake._compute_raw_kalman_observation = (
        lambda cid, capacity, net_flow_entries: fa.FlowAnalyzer._compute_raw_kalman_observation(
            None, cid, capacity, net_flow_entries)
    )
    # `_apply_kalman_reclassification` calls `self._apply_kalman_filter(...)`
    # -- route back through the same fake so it reuses the stubs above.
    fake._apply_kalman_filter = (
        lambda **kwargs: fa.FlowAnalyzer._apply_kalman_filter(fake, **kwargs)
    )
    return fake, kf


def state_dict(state) -> dict:
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


def step_label(pending, has_observation):
    if not pending:
        return "untouched"
    return "updated" if has_observation else "predict_only"


def gen_kalman_filter_cases(rng: Random):
    cases = []

    def mk(label, *, initial_state, observed_ratio, confidence, daily_buckets,
           has_observation, now_val):
        channel_id = f"chan_{label}"
        fake, kf = make_fake_analyzer(channel_id, initial_state=initial_state)
        before = state_dict(kf.state) if initial_state is not None else None
        fa.time.time = lambda: float(now_val)
        pending = []
        result = fa.FlowAnalyzer._apply_kalman_filter(
            fake, channel_id, observed_ratio, confidence, daily_buckets,
            has_observation=has_observation, pending_kalman_saves=pending,
        )
        kalman_ratio, kalman_velocity, uncertainty, regime_change, obs_count = result
        return {
            "id": label,
            "initial_state": before,
            "observed_ratio": bits(observed_ratio),
            "confidence": bits(confidence),
            "daily_buckets": [{"out": b.get("out", 0), "in": b.get("in", 0)} for b in daily_buckets],
            "has_observation": has_observation,
            "now": now_val,
            "flow_ratio": bits(kalman_ratio),
            "flow_velocity": bits(kalman_velocity),
            "uncertainty": bits(uncertainty),
            "regime_change": regime_change,
            "observation_count": obs_count,
            "step": step_label(pending, has_observation),
            "final_state": state_dict(fake._kalman_filters[channel_id].state),
        }

    base_buckets = [{"out": 10_000, "in": 9_000}, {"out": 10_500, "in": 8_800}, {"out": 9_500, "in": 9_200}]

    # First run (last_update==0) -> dt_hours assumed 24.0.
    cases.append(mk("first_run_dt_24h", initial_state=None, observed_ratio=0.15, confidence=0.8,
                     daily_buckets=base_buckets, has_observation=True, now_val=2_000_000_000))

    # Long gap capped at 168h (7 days).
    st_stale = fa.KalmanFlowState(flow_ratio=0.05, flow_velocity=0.0, variance_ratio=0.05,
                                   variance_velocity=0.05, covariance=0.0,
                                   last_update=2_000_000_000 - 400 * 3600,
                                   innovation_variance=0.02, last_innovation=0.0, observation_count=8)
    cases.append(mk("long_gap_capped_at_168h", initial_state=st_stale, observed_ratio=0.2, confidence=0.7,
                     daily_buckets=base_buckets, has_observation=True, now_val=2_000_000_000))

    # Recently updated (< 300s) -> Untouched: state must not move.
    st_recent = fa.KalmanFlowState(flow_ratio=0.22, flow_velocity=0.001, variance_ratio=0.04,
                                    variance_velocity=0.03, covariance=0.002,
                                    last_update=2_000_000_000 - 100,
                                    innovation_variance=0.015, last_innovation=0.05, observation_count=6)
    cases.append(mk("untouched_within_300s", initial_state=st_recent, observed_ratio=0.5, confidence=0.9,
                     daily_buckets=base_buckets, has_observation=True, now_val=2_000_000_000))

    # Exactly at the 300s boundary -> NOT recently-updated (Python: `< 300`,
    # so `now - last_update == 300` fails the gate and DOES advance).
    st_boundary = fa.KalmanFlowState(flow_ratio=0.1, flow_velocity=0.0, variance_ratio=0.05,
                                      variance_velocity=0.05, covariance=0.0,
                                      last_update=2_000_000_000 - 300,
                                      innovation_variance=0.02, last_innovation=0.0, observation_count=6)
    cases.append(mk("exactly_300s_boundary_advances", initial_state=st_boundary, observed_ratio=0.1,
                     confidence=0.8, daily_buckets=base_buckets, has_observation=True,
                     now_val=2_000_000_000))

    # Predict-only: no raw observation this cycle, but last_update still bumps.
    st_predict_only = fa.KalmanFlowState(flow_ratio=0.08, flow_velocity=0.0002, variance_ratio=0.06,
                                          variance_velocity=0.04, covariance=0.001,
                                          last_update=2_000_000_000 - 3600,
                                          innovation_variance=0.02, last_innovation=0.01,
                                          observation_count=7)
    cases.append(mk("predict_only_no_observation", initial_state=st_predict_only, observed_ratio=0.0,
                     confidence=0.5, daily_buckets=base_buckets, has_observation=False,
                     now_val=2_000_000_000))

    # Fresh filter, predict-only, first run.
    cases.append(mk("fresh_predict_only_first_run", initial_state=None, observed_ratio=0.0,
                     confidence=0.5, daily_buckets=base_buckets, has_observation=False,
                     now_val=2_000_000_000))

    # observation_count climbs toward convergence gate across repeated
    # updates spaced >300s apart.
    channel_id = "chan_convergence_walk"
    fake, kf = make_fake_analyzer(channel_id, initial_state=None)
    now_val = 2_000_000_000
    walk_steps = []
    for i in range(8):
        now_val += 3600
        fa.time.time = lambda nv=now_val: float(nv)
        pending = []
        result = fa.FlowAnalyzer._apply_kalman_filter(
            fake, channel_id, 0.3, 0.85, base_buckets, has_observation=True, pending_kalman_saves=pending,
        )
        kalman_ratio, kalman_velocity, uncertainty, regime_change, obs_count = result
        walk_steps.append({
            "now": now_val,
            "flow_ratio": bits(kalman_ratio),
            "flow_velocity": bits(kalman_velocity),
            "uncertainty": bits(uncertainty),
            "regime_change": regime_change,
            "observation_count": obs_count,
            "step": step_label(pending, True),
        })
    cases.append({"id": "convergence_walk_sequence", "steps": walk_steps})

    return cases


# =============================================================================
# apply_kalman_reclassification (convergence gate, DTS widening, congestion
# skip, hysteresis via previous_state)
# =============================================================================

def fee_state_with_variance(variance, nested=True):
    if nested:
        payload = {"fee_state": {"thompson_state": {"posterior_variance": variance}}}
    else:
        payload = {"thompson_state": {"posterior_variance": variance}}
    return {"v2_state_json": json.dumps(payload)}


def gen_reclassification_cases(rng: Random):
    cases = []

    def mk(label, *, capacity, our_balance, daily_volume, is_congested,
           daily_buckets, raw_entries, last_forward_ts, previous_state,
           fee_state, now_val, initial_kalman_state=None, fallback_confidence=1.0):
        channel_id = f"chan_{label}"
        fake, kf = make_fake_analyzer(channel_id, initial_state=initial_kalman_state, fee_state=fee_state)
        initial_state_out = state_dict(kf.state)
        fa.time.time = lambda: float(now_val)

        metrics = fa.FlowMetrics(
            channel_id=channel_id, peer_id="02" + "ab" * 32, sats_in=0, sats_out=0,
            capacity=capacity, flow_ratio=0.0, state=fa.ChannelState.UNKNOWN,
            daily_volume=daily_volume, is_congested=is_congested, confidence=fallback_confidence,
        )
        pending = []
        fa.FlowAnalyzer._apply_kalman_reclassification(
            fake, metrics, channel_id, capacity, our_balance, daily_buckets, raw_entries,
            last_forward_ts, pending_kalman_saves=pending, previous_state=previous_state,
        )

        # Recompute has_observation independently (same 24h-window rule) so
        # the fixture records the step without depending on internals.
        raw_ratio, raw_count = fa.FlowAnalyzer._compute_raw_kalman_observation(
            fake, channel_id, capacity, raw_entries,
        )
        has_observation = raw_count > 0

        overridden = metrics.state != fa.ChannelState.UNKNOWN
        return {
            "id": label,
            "initial_state": initial_state_out,
            "capacity": capacity,
            "our_balance": our_balance,
            "daily_volume": daily_volume,
            "is_congested": is_congested,
            "daily_buckets": [{"out": b.get("out", 0), "in": b.get("in", 0)} for b in daily_buckets],
            "raw_entries": [{"timestamp": bits(e["timestamp"]), "net_msat": int(e["net_msat"])}
                            for e in raw_entries],
            "last_forward_ts": last_forward_ts,
            "previous_state": previous_state,
            "posterior_variance": fee_state_variance_of(fee_state),
            "source_threshold": bits(fake.config.source_threshold),
            "sink_threshold": bits(fake.config.sink_threshold),
            "now": now_val,
            "fallback_confidence": bits(fallback_confidence),
            "kalman_flow_ratio": bits(metrics.kalman_flow_ratio),
            "kalman_velocity": bits(metrics.kalman_velocity),
            "kalman_uncertainty": bits(metrics.kalman_uncertainty),
            "kalman_regime_change": metrics.kalman_regime_change,
            "step": step_label(pending, has_observation),
            "state_overridden": overridden,
            "resulting_state": metrics.state.name if overridden else None,
        }

    def fee_state_variance_of(fee_state):
        if not fee_state:
            return None
        payload = json.loads(fee_state["v2_state_json"])
        ts = (payload.get("fee_state") or {}).get("thompson_state") or payload.get("thompson_state", {})
        return ts.get("posterior_variance")

    now_val = 2_100_000_000

    # Factories, NOT shared instances: `KalmanFlowFilter.__init__` does
    # `self.state = state or KalmanFlowState()` -- it stores the object BY
    # REFERENCE, so `_apply_kalman_filter` mutates it in place (bumping
    # `last_update` to `now_val`, among other fields). A single shared
    # `KalmanFlowState` instance reused across multiple `mk()` calls would
    # have its `last_update` silently advanced by the FIRST case that
    # touches it, corrupting the "recently updated" gate for every
    # subsequent case that thinks it's starting from the original values.
    def converged_state():
        return fa.KalmanFlowState(
            flow_ratio=0.2, flow_velocity=0.0, variance_ratio=0.01, variance_velocity=0.01,
            covariance=0.0, last_update=now_val - 3600, innovation_variance=0.01, last_innovation=0.0,
            observation_count=10,
        )

    def custom_state(flow_ratio):
        """A converged filter at an arbitrary `flow_ratio`, otherwise
        identical to `converged_state()` -- for cases that need the
        Kalman-threshold branch to fall through to
        `classification::classify_balance_position` instead of deciding at
        the outer source/sink-threshold check."""
        return fa.KalmanFlowState(
            flow_ratio=flow_ratio, flow_velocity=0.0, variance_ratio=0.01, variance_velocity=0.01,
            covariance=0.0, last_update=now_val - 3600, innovation_variance=0.01, last_innovation=0.0,
            observation_count=10,
        )

    def unconverged_state():
        return fa.KalmanFlowState(
            flow_ratio=0.2, flow_velocity=0.0, variance_ratio=0.2, variance_velocity=0.2,
            covariance=0.0, last_update=now_val - 3600, innovation_variance=0.05, last_innovation=0.0,
            observation_count=2,
        )

    # Straddles the widened-vs-unwidened threshold boundary: the resulting
    # kalman_flow_ratio (~0.0525, verified empirically) sits ABOVE the bare
    # source_threshold (0.05) but BELOW the DTS-widened one (0.05 * 1.5 =
    # 0.075) -- the one initial state where widening actually flips the
    # classification outcome (SOURCE -> BALANCED_ACTIVE), unlike
    # `converged_state()` whose flow_ratio (0.2) stays SOURCE either way.
    def straddling_state():
        return fa.KalmanFlowState(
            flow_ratio=0.055, flow_velocity=0.0, variance_ratio=0.01, variance_velocity=0.01,
            covariance=0.0, last_update=now_val - 3600, innovation_variance=0.01, last_innovation=0.0,
            observation_count=10,
        )

    entries_strong_source = [{"timestamp": now_val - 100, "net_msat": 900_000}]
    entries_strong_sink = [{"timestamp": now_val - 100, "net_msat": -900_000}]
    entries_none = []

    cases.append(mk(
        "converged_source_override", capacity=1_000_000, our_balance=800_000, daily_volume=50_000,
        is_congested=False, daily_buckets=[{"out": 10_000, "in": 9_000}] * 3,
        raw_entries=entries_strong_source, last_forward_ts=now_val - 100, previous_state=None,
        fee_state=None, now_val=now_val, initial_kalman_state=converged_state(),
    ))
    cases.append(mk(
        "unconverged_low_obs_count_no_override", capacity=1_000_000, our_balance=800_000,
        daily_volume=50_000, is_congested=False, daily_buckets=[{"out": 10_000, "in": 9_000}] * 3,
        raw_entries=entries_strong_source, last_forward_ts=now_val - 100, previous_state=None,
        fee_state=None, now_val=now_val, initial_kalman_state=unconverged_state(),
    ))
    cases.append(mk(
        "congested_skips_override_even_when_converged", capacity=1_000_000, our_balance=800_000,
        daily_volume=50_000, is_congested=True, daily_buckets=[{"out": 10_000, "in": 9_000}] * 3,
        raw_entries=entries_strong_source, last_forward_ts=now_val - 100, previous_state=None,
        fee_state=None, now_val=now_val, initial_kalman_state=converged_state(),
    ))
    cases.append(mk(
        "dts_widening_variance_above_10000_biases_balanced", capacity=1_000_000, our_balance=500_000,
        daily_volume=50_000, is_congested=False, daily_buckets=[{"out": 10_000, "in": 9_000}] * 3,
        raw_entries=entries_strong_source, last_forward_ts=now_val - 100, previous_state=None,
        fee_state=fee_state_with_variance(15_000, nested=True), now_val=now_val,
        initial_kalman_state=straddling_state(),
    ))
    cases.append(mk(
        "dts_variance_at_threshold_no_widening", capacity=1_000_000, our_balance=500_000,
        daily_volume=50_000, is_congested=False, daily_buckets=[{"out": 10_000, "in": 9_000}] * 3,
        raw_entries=entries_strong_source, last_forward_ts=now_val - 100, previous_state=None,
        fee_state=fee_state_with_variance(10_000, nested=True), now_val=now_val,
        initial_kalman_state=straddling_state(),
    ))
    cases.append(mk(
        "dts_widening_flat_fallback_shape", capacity=1_000_000, our_balance=500_000,
        daily_volume=50_000, is_congested=False, daily_buckets=[{"out": 10_000, "in": 9_000}] * 3,
        raw_entries=entries_strong_source, last_forward_ts=now_val - 100, previous_state=None,
        fee_state=fee_state_with_variance(20_000, nested=False), now_val=now_val,
        initial_kalman_state=straddling_state(),
    ))
    cases.append(mk(
        "no_capacity_outbound_ratio_defaults_half", capacity=0, our_balance=0,
        daily_volume=0, is_congested=False, daily_buckets=[], raw_entries=entries_none,
        last_forward_ts=0, previous_state=None, fee_state=None, now_val=now_val,
        initial_kalman_state=custom_state(0.0),
    ))
    # Hysteresis: kalman_ratio=0.0 stays inside the default +/-0.05
    # threshold band -> falls through to `classify_balance_position`.
    # outbound_ratio=0.75 sits BETWEEN the SINK exit band (0.72) and enter
    # band (0.78): with `previous_state="sink"` this keeps SINK (exit band
    # applies); the same inputs with `previous_state=None` would NOT
    # qualify (0.75 < 0.78 enter band) -- verified empirically.
    cases.append(mk(
        "balance_fallback_sink_via_hysteresis_previous_sink", capacity=1_000_000, our_balance=750_000,
        daily_volume=1_000, is_congested=False, daily_buckets=[{"out": 1_000, "in": 900}] * 3,
        raw_entries=entries_none, last_forward_ts=0, previous_state="sink", fee_state=None,
        now_val=now_val, initial_kalman_state=custom_state(0.0), fallback_confidence=0.3,
    ))
    # F1c veto: kalman_ratio=0.06 is ABOVE KALMAN_BALANCE_VETO_RATIO (0.05)
    # -- outbound_ratio=0.9 alone would qualify as SINK (> 0.78 enter band),
    # but the veto ("draining but currently full is not a SINK") forces it
    # away from SINK. DTS widening (posterior_variance=15000 -> thresholds
    # *1.5 = 0.075) keeps 0.06 inside the balance-position branch instead of
    # tripping the outer SOURCE check first; verified empirically the
    # result lands on BALANCED (not SINK, not DORMANT: turnover=0.001 <
    # 1%/day, and |0.06| >= the 0.01 dormant threshold).
    cases.append(mk(
        "balance_fallback_source_kalman_veto", capacity=1_000_000, our_balance=900_000,
        daily_volume=1_000, is_congested=False, daily_buckets=[{"out": 1_000, "in": 900}] * 3,
        raw_entries=entries_none, last_forward_ts=0, previous_state=None,
        fee_state=fee_state_with_variance(15_000, nested=True),
        now_val=now_val, initial_kalman_state=custom_state(0.06), fallback_confidence=0.3,
    ))
    cases.append(mk(
        "sink_direction_strong_negative_flow", capacity=1_000_000, our_balance=500_000,
        daily_volume=50_000, is_congested=False, daily_buckets=[{"out": 9_000, "in": 10_000}] * 3,
        raw_entries=entries_strong_sink, last_forward_ts=now_val - 100, previous_state=None,
        fee_state=None, now_val=now_val, initial_kalman_state=converged_state(),
    ))
    cases.append(mk(
        "no_raw_observation_falls_back_to_metrics_confidence", capacity=1_000_000, our_balance=500_000,
        daily_volume=50_000, is_congested=False, daily_buckets=[{"out": 10_000, "in": 9_000}] * 3,
        raw_entries=entries_none, last_forward_ts=0, previous_state=None, fee_state=None,
        now_val=now_val, initial_kalman_state=converged_state(), fallback_confidence=0.42,
    ))

    for i in range(10):
        capacity = rng.randint(200_000, 20_000_000)
        our_balance = rng.randint(0, capacity)
        daily_volume = rng.randint(0, capacity // 2 if capacity > 1 else 1)
        entries = [{"timestamp": now_val - rng.randint(0, 80_000),
                    "net_msat": rng.randint(-2_000_000, 2_000_000)} for _ in range(rng.randint(0, 4))]
        buckets = [{"out": rng.randint(0, 50_000), "in": rng.randint(0, 50_000)} for _ in range(3)]
        prev = rng.choice([None, "source", "sink", "balanced", "balanced_active", "dormant"])
        variance = rng.choice([None, 5_000, 12_000, 25_000])
        fee_state = fee_state_with_variance(variance, nested=rng.random() < 0.5) if variance is not None else None
        state = rng.choice([converged_state(), unconverged_state()])
        cases.append(mk(
            f"rand{i:02d}", capacity=capacity, our_balance=our_balance, daily_volume=daily_volume,
            is_congested=rng.random() < 0.1, daily_buckets=buckets, raw_entries=entries,
            last_forward_ts=now_val - rng.randint(0, 200_000), previous_state=prev,
            fee_state=fee_state, now_val=now_val, initial_kalman_state=state,
        ))

    return cases


# =============================================================================
# TemporalProfile / update_temporal_profile
# =============================================================================

def histogram_bucket(out_sats=0.0, in_sats=0.0, count=0.0):
    return {"out_sats": out_sats, "in_sats": in_sats, "count": count}


def temporal_profile_dict(tp: "fa.TemporalProfile") -> dict:
    d = tp.to_dict()
    return {
        "hourly_out": [bits(v) for v in d["hourly_out"]],
        "hourly_in": [bits(v) for v in d["hourly_in"]],
        "hourly_count": [bits(v) for v in d["hourly_count"]],
        "peak_hours": list(d["peak_hours"]),
        "quiet_hours": list(d["quiet_hours"]),
        # Numpy-derived: plain floats (epsilon compare in Rust), see module
        # docstring.
        "burstiness": d["burstiness"],
        "diurnal_strength": d["diurnal_strength"],
        "dominant_bucket": d["dominant_bucket"],
        "observation_days": int(d["observation_days"]),
        "last_observation_day": int(d["last_observation_day"]),
        "last_updated": int(d["last_updated"]),
    }


def gen_temporal_profile_cases(rng: Random):
    cases = []
    now0 = 2_200_000_000

    def run_sequence(label, updates):
        """updates: list of (histogram_24, daily_forwards, now_val)."""
        existing = fa.TemporalProfile()
        steps = []
        for histogram, daily_forwards, now_val in updates:
            fa.time.time = lambda nv=now_val: float(nv)
            updated = fa.update_temporal_profile(existing, histogram, daily_forwards)
            steps.append({
                # Bit-pattern hex, NOT plain JSON numbers: serde_json's
                # default (non-"arbitrary_precision") float parser is not
                # always correctly-rounded to the nearest f64 (observed
                # empirically -- a 1-ULP miss on a value like
                # 49.734184829473676 -- whereas Rust's own
                # `str::parse::<f64>()` IS correctly-rounded and matches
                # Python bit-for-bit). Every float in this fixture format
                # goes through `bits()` for exactly this reason; these
                # three were the one place that had drifted from that
                # convention.
                "histogram": [{"out_sats": bits(h["out_sats"]), "in_sats": bits(h["in_sats"]),
                               "count": bits(h["count"])}
                              for h in histogram],
                "daily_forwards": daily_forwards,
                "now": now_val,
                "profile_after": temporal_profile_dict(updated),
            })
            existing = updated
        return {"id": label, "steps": steps}

    # First update (all-zero existing) copies raw values; graduates only
    # once daily_forwards crosses the threshold AND the epoch day changes.
    flat_hist = [histogram_bucket(out_sats=100.0 + h, in_sats=50.0 + h, count=5.0) for h in range(24)]
    diurnal_hist = [
        histogram_bucket(out_sats=(1000.0 if 8 <= h < 20 else 10.0), in_sats=(800.0 if 8 <= h < 20 else 5.0),
                         count=(20.0 if 8 <= h < 20 else 1.0))
        for h in range(24)
    ]
    zero_hist = [histogram_bucket() for _ in range(24)]

    cases.append(run_sequence("first_update_copies_raw", [
        (flat_hist, 5, now0),
    ]))

    cases.append(run_sequence("ema_blend_second_update_same_day", [
        (flat_hist, 15, now0),
        ([histogram_bucket(out_sats=200.0 + h, in_sats=100.0 + h, count=8.0) for h in range(24)],
         15, now0 + 3600),
    ]))

    cases.append(run_sequence("graduation_advances_once_per_day", [
        (flat_hist, 15, now0),
        (flat_hist, 15, now0 + 3600),  # same epoch day -> no extra graduation
        (flat_hist, 15, now0 + 90_000),  # next day -> observation_days += 1
    ]))

    cases.append(run_sequence("below_min_daily_forwards_never_graduates", [
        (flat_hist, 3, now0),
        (flat_hist, 3, now0 + 90_000),
    ]))

    cases.append(run_sequence("all_zero_histogram_derived_fields_empty", [
        (zero_hist, 0, now0),
    ]))

    cases.append(run_sequence("diurnal_pattern_strong_signal", [
        (diurnal_hist, 20, now0),
    ]))

    # Seven-day graduation walk (one update per day, crossing into
    # `graduated` at day 7).
    week_updates = []
    for day in range(9):
        week_updates.append((flat_hist, 15, now0 + day * 90_000))
    cases.append(run_sequence("seven_day_graduation_walk", week_updates))

    for i in range(6):
        n_updates = rng.randint(1, 5)
        updates = []
        t = now0 + rng.randint(0, 5) * 90_000
        for _ in range(n_updates):
            hist = [histogram_bucket(
                out_sats=rng.uniform(0, 5000), in_sats=rng.uniform(0, 5000), count=rng.uniform(0, 50),
            ) for _ in range(24)]
            daily_forwards = rng.randint(0, 30)
            updates.append((hist, daily_forwards, t))
            t += rng.randint(1, 5) * 3600
        cases.append(run_sequence(f"rand{i:02d}", updates))

    return cases


# =============================================================================
# demand_flow: classify_peers / classify_candidate / find_sink_adjacent_candidates
# =============================================================================

def gen_classify_peers_cases(rng: Random):
    classifier = df.DemandFlowClassifier()
    cases = []

    def mk(label, flows):
        ns_flows = [types.SimpleNamespace(peer_id=f["peer_id"], sats_in=f["sats_in"], sats_out=f["sats_out"])
                    for f in flows]
        profiles = classifier.classify_peers({f"ch{i}": v for i, v in enumerate(ns_flows)})
        return {
            "id": label,
            "flows": flows,
            "profiles": {
                pid: {
                    "role": p.role,
                    "confidence": bits(p.confidence),
                    "net_flow_ratio": (bits(p.net_flow_ratio) if p.net_flow_ratio is not None else None),
                }
                for pid, p in profiles.items()
            },
        }

    cases.append(mk("empty", []))
    cases.append(mk("single_peer_zero_flow", [{"peer_id": "peerA", "sats_in": 0, "sats_out": 0}]))
    cases.append(mk("single_peer_source_leaning", [{"peer_id": "peerB", "sats_in": 900_000, "sats_out": 100_000}]))
    cases.append(mk("single_peer_sink_leaning", [{"peer_id": "peerC", "sats_in": 100_000, "sats_out": 900_000}]))
    cases.append(mk("single_peer_router_balanced", [{"peer_id": "peerD", "sats_in": 500_000, "sats_out": 520_000}]))
    cases.append(mk("aggregates_across_multiple_channels_same_peer", [
        {"peer_id": "peerE", "sats_in": 100_000, "sats_out": 50_000},
        {"peer_id": "peerE", "sats_in": 200_000, "sats_out": 20_000},
    ]))
    cases.append(mk("empty_peer_id_skipped", [{"peer_id": "", "sats_in": 100, "sats_out": 100}]))
    cases.append(mk("ratio_exactly_at_0_3_boundary_is_router", [
        {"peer_id": "peerF", "sats_in": 650_000, "sats_out": 350_000},
    ]))
    cases.append(mk("high_volume_confidence_capped_at_0_9", [
        {"peer_id": "peerG", "sats_in": 5_000_000_000, "sats_out": 100_000},
    ]))
    cases.append(mk("low_volume_confidence_floors_at_0_1", [
        {"peer_id": "peerH", "sats_in": 2, "sats_out": 1},
    ]))
    for i in range(10):
        n = rng.randint(1, 5)
        flows = [{"peer_id": f"rpeer{i}_{j}", "sats_in": rng.randint(0, 2_000_000),
                  "sats_out": rng.randint(0, 2_000_000)} for j in range(n)]
        cases.append(mk(f"rand{i:02d}", flows))
    return cases


def gen_classify_candidate_cases(rng: Random):
    classifier = df.DemandFlowClassifier()
    cases = []

    def mk(label, node_id, node_info, channels):
        profile = classifier.classify_candidate(node_id, node_info, channels)
        return {
            "id": label,
            "node_id": node_id,
            "node_info": node_info,
            "channels": channels,
            "role": profile.role,
            "confidence": bits(profile.confidence),
            "gossip_signals": {k: bits(v) for k, v in profile.gossip_signals.items()},
            "has_liquidity_ads": profile.has_liquidity_ads,
        }

    cases.append(mk("no_signals_unknown", "node1", {"alias": "randomnode"}, []))
    cases.append(mk("exchange_alias_source", "node2", {"alias": "Kraken Hub 3"}, []))
    cases.append(mk("sink_alias_wallet", "node3", {"alias": "MyWallet Pay"}, []))
    cases.append(mk("lsp_alias_router", "node4", {"alias": "Voltage LSP node"}, []))
    cases.append(mk("no_node_info_defaults_empty_alias", "node5", None, []))
    cases.append(mk("structure_hub_router", "node6", {"alias": "bignode"}, [
        {"active": True, "amount_msat": 60_000_000_000} for _ in range(101)
    ]))
    cases.append(mk("structure_sink_many_small_channels", "node7", {"alias": "smallnode"}, [
        {"active": True, "amount_msat": 100_000_000} for _ in range(31)
    ]))
    cases.append(mk("structure_source_few_large_channels", "node8", {"alias": "bignode2"}, [
        {"active": True, "amount_msat": 6_000_000_000} for _ in range(5)
    ]))
    cases.append(mk("fee_sink_low_fee_majority", "node9", {"alias": "feenode"}, [
        {"active": True, "base_fee_millisatoshi": 0, "fee_per_millionth": 10} for _ in range(6)
    ] + [{"active": True, "base_fee_millisatoshi": 1000, "fee_per_millionth": 500}]))
    cases.append(mk("fee_extractive_high_fee_majority", "node10", {"alias": "greedynode"}, [
        {"active": True, "fee_per_millionth": 800} for _ in range(6)
    ] + [{"active": True, "fee_per_millionth": 10}]))
    cases.append(mk("liquidity_ads_true", "node11", {"alias": "adnode", "option_will_fund": {"lease_fee_base_msat": 0}}, []))
    cases.append(mk("liquidity_ads_false_value", "node12", {"alias": "adnode2", "option_will_fund": False}, []))
    cases.append(mk("inactive_channels_ignored_for_structure", "node13", {"alias": "inactivenode"}, [
        {"active": False, "amount_msat": 60_000_000_000} for _ in range(101)
    ]))
    cases.append(mk("tie_source_sink_prefers_source", "node14", {"alias": "krakenwallet"}, []))
    cases.append(mk("tie_sink_router_prefers_sink", "node15", {"alias": "walletvoltage"}, []))
    cases.append(mk("missing_alias_key", "node16", {}, []))
    cases.append(mk("default_fee_values_used_when_missing", "node17", {"alias": "defaultfeenode"}, [
        {"active": True} for _ in range(4)
    ]))
    for i in range(10):
        alias = rng.choice(["kraken1", "mywallet", "phoenixnode", "randomalias", "btcpay-store"])
        n = rng.randint(0, 15)
        channels = [
            {
                "active": rng.random() < 0.8,
                "amount_msat": rng.randint(1_000_000, 8_000_000_000),
                "base_fee_millisatoshi": rng.choice([0, 1000]),
                "fee_per_millionth": rng.randint(0, 900),
            }
            for _ in range(n)
        ]
        cases.append(mk(f"rand{i:02d}", f"randnode{i}", {"alias": alias}, channels))
    return cases


def gen_sink_adjacent_cases(rng: Random):
    classifier = df.DemandFlowClassifier()
    cases = []

    def mk(label, sink_profiles_ordered, sink_channels, existing_peers):
        # Preserve insertion order via a plain dict (Python 3.7+ dict
        # literal / comprehension order == insertion order).
        sink_profiles = {p.node_id: p for p in sink_profiles_ordered}
        result = classifier.find_sink_adjacent_candidates(sink_profiles, sink_channels, existing_peers)
        return {
            "id": label,
            "sink_profiles_order": [p.node_id for p in sink_profiles_ordered],
            "sink_profiles": {
                p.node_id: {"confidence": bits(p.confidence),
                            "net_flow_ratio": (bits(p.net_flow_ratio) if p.net_flow_ratio is not None else None)}
                for p in sink_profiles_ordered
            },
            "sink_channels": sink_channels,
            "existing_peers": sorted(existing_peers),
            "candidates": [
                {
                    "peer_id": c["peer_id"],
                    "source": c["source"],
                    "score": bits(c["score"]),
                    "reason": c["reason"],
                    "sink_peer_id": c["sink_peer_id"],
                    "is_sink_adjacent": c["is_sink_adjacent"],
                }
                for c in result
            ],
        }

    def prof(node_id, confidence, net_flow_ratio):
        return df.NodeFlowProfile(node_id=node_id, role="sink", confidence=confidence,
                                   net_flow_ratio=net_flow_ratio)

    cases.append(mk("empty_sink_profiles", [], {}, set()))

    sinkA = prof("sinkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", 0.8, -0.5)
    cases.append(mk("no_channels_for_sink", [sinkA], {}, set()))

    sinkB = prof("sinkBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB", 0.6, -0.4)
    chans_b = [
        {"destination": "destX", "active": True},
        {"destination": "destY", "active": False},  # inactive -> skipped
        {"destination": "", "active": True},          # empty destination -> skipped
    ]
    cases.append(mk("basic_scoring_and_inactive_skip", [sinkB], {sinkB.node_id: chans_b}, set()))

    sinkC = prof("sinkCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC", 0.7, -0.6)
    chans_c = [{"destination": "destExisting", "active": True}]
    cases.append(mk("existing_peer_excluded", [sinkC], {sinkC.node_id: chans_c}, {"destExisting"}))

    sinkD = prof("sinkDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD", 0.5, 0.55)
    chans_d = [
        {"destination": "destDup", "active": True},
        {"destination": "destDup", "active": True},  # duplicate -> deduped, first wins
        {"destination": "destUnique", "active": True},
    ]
    cases.append(mk("dedup_keeps_first_occurrence", [sinkD], {sinkD.node_id: chans_d}, set()))

    # More than 5 sink profiles -> only top 5 by |net_flow_ratio| considered.
    many_sinks = [prof(f"sink{n:02d}" + "F" * 60, 0.5, (-0.9 + n * 0.05)) for n in range(8)]
    many_channels = {s.node_id: [{"destination": f"dest_{s.node_id}", "active": True}] for s in many_sinks}
    cases.append(mk("more_than_five_sinks_truncated", many_sinks, many_channels, set()))

    # Tie in |net_flow_ratio| -> stable ordering keeps original insertion
    # order for the tied pair (NOT reversed): sinkTieA before sinkTieB.
    sink_tie_a = prof("sinkTieAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", 0.9, 0.4)
    sink_tie_b = prof("sinkTieBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB", 0.9, -0.4)
    tie_channels = {
        sink_tie_a.node_id: [{"destination": "destTieA", "active": True}],
        sink_tie_b.node_id: [{"destination": "destTieB", "active": True}],
    }
    cases.append(mk("tie_preserves_original_relative_order", [sink_tie_a, sink_tie_b], tie_channels, set()))
    # Same tie pair, opposite insertion order -> output order must flip too
    # (proving it's insertion-order-driven, not id-driven).
    cases.append(mk("tie_reversed_insertion_order", [sink_tie_b, sink_tie_a], tie_channels, set()))

    # More than 10 resulting candidates -> truncated to 10, sorted by score desc.
    big_sink = prof("sinkBIGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG", 0.9, -0.8)
    big_channels = [{"destination": f"dest{n:02d}", "active": True} for n in range(15)]
    cases.append(mk("more_than_ten_candidates_truncated", [big_sink], {big_sink.node_id: big_channels}, set()))

    for i in range(6):
        n_sinks = rng.randint(1, 6)
        sinks = [
            prof(f"randsink{i}_{j}" + "A" * 40, rng.uniform(0.1, 0.9), rng.uniform(-0.9, 0.9))
            for j in range(n_sinks)
        ]
        channels_map = {}
        for s in sinks:
            n_ch = rng.randint(0, 4)
            channels_map[s.node_id] = [
                {"destination": f"randdest{i}_{s.node_id}_{k}", "active": rng.random() < 0.8}
                for k in range(n_ch)
            ]
        existing = {f"randdest{i}_{sinks[0].node_id}_0"} if sinks and rng.random() < 0.3 else set()
        cases.append(mk(f"rand{i:02d}", sinks, channels_map, existing))

    return cases


# =============================================================================
# main
# =============================================================================

def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <output.json>", file=sys.stderr)
        sys.exit(1)

    real_time = fa.time.time
    try:
        rng = Random(20260717)
        out = {
            "meta": {"seed": 20260717, "source": "modules/flow_analysis.py + modules/demand_flow.py"},
            "constants": gen_constants(),
            "velocity_cases": gen_velocity_cases(rng),
            "decay_cases": gen_decay_cases(rng),
            "ema_cases": gen_ema_cases(rng),
            "kalman_filter_cases": gen_kalman_filter_cases(rng),
            "reclassification_cases": gen_reclassification_cases(rng),
            "temporal_profile_cases": gen_temporal_profile_cases(rng),
            "classify_peers_cases": gen_classify_peers_cases(rng),
            "classify_candidate_cases": gen_classify_candidate_cases(rng),
            "sink_adjacent_cases": gen_sink_adjacent_cases(rng),
        }
    finally:
        fa.time.time = real_time

    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")

    print(
        f"wrote {sys.argv[1]}: "
        f"{len(out['velocity_cases'])} velocity, "
        f"{len(out['decay_cases'])} decay, "
        f"{len(out['ema_cases'])} ema, "
        f"{len(out['kalman_filter_cases'])} kalman_filter, "
        f"{len(out['reclassification_cases'])} reclassification, "
        f"{len(out['temporal_profile_cases'])} temporal_profile, "
        f"{len(out['classify_peers_cases'])} classify_peers, "
        f"{len(out['classify_candidate_cases'])} classify_candidate, "
        f"{len(out['sink_adjacent_cases'])} sink_adjacent cases"
    )


if __name__ == "__main__":
    main()
