"""Structural loop-out: config, scarcity factor, credit, and envelope gate."""

from unittest.mock import MagicMock

import pytest

from tests.plugin_test_utils import load_plugin_module


def test_config_defaults():
    from modules.config import Config
    cfg = Config()
    assert cfg.receivable_ratio_target == 0.30
    assert cfg.receivable_ratio_floor == 0.20
    assert cfg.boltz_structural_budget_sats_per_day == 0  # off by default
    assert cfg.drain_fee_discount_max == 0.0               # off by default


def test_config_validation_ranges():
    from modules.config import Config
    with pytest.raises(ValueError):
        Config(receivable_ratio_target=1.5)
    with pytest.raises(ValueError):
        Config(drain_fee_discount_max=0.9)


def test_config_floor_must_not_exceed_target():
    from modules.config import Config
    with pytest.raises(ValueError):
        Config(receivable_ratio_floor=0.4, receivable_ratio_target=0.3)


def test_plugin_options_registered():
    mod = load_plugin_module()
    for name in (
        "revenue-ops-receivable-ratio-target",
        "revenue-ops-receivable-ratio-floor",
        "revenue-ops-boltz-structural-budget-sats",
        "revenue-ops-drain-fee-discount-max",
    ):
        assert name in mod.plugin.options, name


def _channels_info(entries):
    """entries: list of (spendable_sats, receivable_sats)."""
    return {
        f"{i}x1x0": {
            "peer_id": "02" + ("%02d" % i) * 33,
            "spendable_msat": s * 1000,
            "receivable_msat": r * 1000,
            "capacity": s + r,
        }
        for i, (s, r) in enumerate(entries, start=100)
    }


def _scarcity_module(entries):
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    mod.fee_controller = MagicMock()
    mod.fee_controller._get_channels_info.return_value = _channels_info(entries)
    from modules.config import Config
    mod.config = Config()
    return mod


def test_receivable_status_starved_node():
    # 97% local everywhere => receivable_ratio 0.03, fully starved
    mod = _scarcity_module([(970_000, 30_000)] * 4)
    status = mod._node_receivable_status()
    assert status["receivable_ratio"] == pytest.approx(0.03, abs=0.001)
    assert status["scarcity"] == pytest.approx(1.0)  # clamped at floor


def test_receivable_status_healthy_node():
    mod = _scarcity_module([(500_000, 500_000)] * 4)
    status = mod._node_receivable_status()
    assert status["receivable_ratio"] == pytest.approx(0.5, abs=0.001)
    assert status["scarcity"] == 0.0


def test_receivable_status_scales_between_floor_and_target():
    # ratio 0.25 sits halfway between floor 0.20 and target 0.30
    mod = _scarcity_module([(750_000, 250_000)] * 4)
    status = mod._node_receivable_status()
    assert status["scarcity"] == pytest.approx(0.5, abs=0.01)


def test_receivable_status_safe_on_error():
    mod = _scarcity_module([])
    mod.fee_controller._get_channels_info.side_effect = RuntimeError("boom")
    status = mod._node_receivable_status()
    assert status == {"receivable_ratio": None, "scarcity": 0.0,
                      "total_capacity_sats": 0, "total_receivable_sats": 0}


# ---------------------------------------------------------------------------
# Fix 1: ConfigSnapshot carries the four new drain-demand fields
# ---------------------------------------------------------------------------

def test_snapshot_carries_new_drain_fields():
    """ConfigSnapshot.from_config (via Config.snapshot) must include all four
    structural loop-out fields with the correct values."""
    from modules.config import Config, ConfigSnapshot

    cfg = Config(boltz_structural_budget_sats_per_day=500)
    snap = cfg.snapshot()

    assert snap.boltz_structural_budget_sats_per_day == 500
    assert snap.receivable_ratio_target == pytest.approx(0.30)
    assert snap.receivable_ratio_floor == pytest.approx(0.20)
    assert snap.drain_fee_discount_max == pytest.approx(0.0)


def test_snapshot_drain_fields_propagate_non_defaults():
    """Non-default values for all four fields survive the snapshot round-trip."""
    from modules.config import Config

    cfg = Config(
        receivable_ratio_target=0.40,
        receivable_ratio_floor=0.25,
        boltz_structural_budget_sats_per_day=1_000,
        drain_fee_discount_max=0.05,
    )
    snap = cfg.snapshot()

    assert snap.receivable_ratio_target == pytest.approx(0.40)
    assert snap.receivable_ratio_floor == pytest.approx(0.25)
    assert snap.boltz_structural_budget_sats_per_day == 1_000
    assert snap.drain_fee_discount_max == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Fix 2: Negative msat values are clamped to zero — scarcity stays fail-open
# ---------------------------------------------------------------------------

def _channels_info_raw(entries):
    """Like _channels_info but entries are (spendable_msat, receivable_msat) raw msat."""
    return {
        f"{i}x1x0": {
            "peer_id": "02" + ("%02d" % i) * 33,
            "spendable_msat": s,
            "receivable_msat": r,
            "capacity": (max(0, s) + max(0, r)) // 1000,
        }
        for i, (s, r) in enumerate(entries, start=200)
    }


def test_negative_receivable_msat_clamped_to_zero():
    """A single channel with receivable_msat=-900_000_000 mixed with healthy
    channels must not drive scarcity to 1.0 — the bad value is treated as 0."""
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    mod.fee_controller = MagicMock()
    # Three healthy channels: 500k/500k sats each → ratio 0.50 normally
    # Plus one poisoned channel: spendable=500k sats, receivable=-900k sats (msat)
    healthy = [(500_000 * 1000, 500_000 * 1000)] * 3
    poisoned = [(500_000 * 1000, -900_000_000)]
    mod.fee_controller._get_channels_info.return_value = _channels_info_raw(
        healthy + poisoned
    )
    from modules.config import Config
    mod.config = Config()

    status = mod._node_receivable_status()
    # Negative receivable is zeroed; total_cap = (500k+500k)*3 + (500k+0)*1 = 3_500_000 sats
    # total_recv = 500k*3 + 0 = 1_500_000 sats; ratio ~ 0.4286 → above target → scarcity 0.0
    assert status["scarcity"] != 1.0, "negative msat must not force scarcity=1.0"
    assert status["scarcity"] == pytest.approx(0.0)
    assert status["receivable_ratio"] is not None
    assert status["receivable_ratio"] > 0.0


# ---------------------------------------------------------------------------
# Fix 3a: Zero-capacity channels dict → safe dict returned
# ---------------------------------------------------------------------------

def test_zero_capacity_returns_safe_dict():
    """All channels with spendable=0 and receivable=0 yields total_cap=0,
    triggering the safe-dict early return."""
    mod = _scarcity_module([(0, 0)] * 4)
    status = mod._node_receivable_status()
    assert status == {"receivable_ratio": None, "scarcity": 0.0,
                      "total_capacity_sats": 0, "total_receivable_sats": 0}


# ---------------------------------------------------------------------------
# Fix 3b: Exact boundary tests — ratio==target → scarcity 0.0, ratio==floor → scarcity 1.0
# ---------------------------------------------------------------------------

def test_scarcity_zero_at_exact_target():
    """When receivable_ratio == receivable_ratio_target (0.30), scarcity must be 0.0."""
    # target=0.30: recv / (spend+recv) = 0.30 → recv=3, spend=7 (sats ratio 3:7)
    # Use 300k receivable + 700k spendable per channel.
    mod = _scarcity_module([(700_000, 300_000)] * 4)
    status = mod._node_receivable_status()
    assert status["receivable_ratio"] == pytest.approx(0.30, abs=1e-6)
    assert status["scarcity"] == pytest.approx(0.0, abs=1e-6)


def test_scarcity_one_at_exact_floor():
    """When receivable_ratio == receivable_ratio_floor (0.20), scarcity must be 1.0."""
    # floor=0.20: recv / (spend+recv) = 0.20 → recv=2, spend=8 (ratio 2:8)
    # Use 200k receivable + 800k spendable per channel.
    mod = _scarcity_module([(800_000, 200_000)] * 4)
    status = mod._node_receivable_status()
    assert status["receivable_ratio"] == pytest.approx(0.20, abs=1e-6)
    assert status["scarcity"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Task 8: Boltz loop-outs earn a scarcity-scaled structural inbound credit
# ---------------------------------------------------------------------------

from tests.test_boltz_balance_plan_bias import _make_planner_module


def _make_prof_mock():
    """Minimal profitability object that survives require_profitable=True
    filters but contributes zero historical revenue (daily_contrib == 0),
    so only the structural credit can move the profit guard."""
    prof = MagicMock()
    prof.marginal_roi = 1.0
    prof.marginal_roi_percent = 1.0
    prof.net_profit_sats = 0
    prof.roi_percent = 0.0
    prof.is_operationally_profitable = False
    prof.days_open = 30
    prof.revenue.total_contribution_sats = 0
    return prof


def _structural_module(scarcity, realized_ppm=500, structural_budget=1000):
    mod = _make_planner_module()
    from modules.config import Config
    mod.config = Config(boltz_structural_budget_sats_per_day=structural_budget)
    mod._node_receivable_status = lambda: {
        "receivable_ratio": 0.05, "scarcity": scarcity,
        "total_capacity_sats": 10_000_000, "total_receivable_sats": 500_000,
    }
    mod.database.get_node_realized_fee_ppm.return_value = realized_ppm
    # Fallback cascade stages default to "no data" so each test controls
    # exactly which source prices the credit.
    mod.database.get_node_realized_fee_ppm_30d.return_value = 0
    # Profitability data so require_profitable=True does not skip the channel.
    pa = MagicMock()
    pa.analyze_all_channels.return_value = None
    pa.get_profitability.return_value = _make_prof_mock()
    mod.profitability_analyzer = pa
    # 97% local => loop_out direction (high trigger default 80%)
    mod.fee_controller._get_channels_info.return_value = {
        "100x1x0": {
            "peer_id": "02" + "b" * 64,
            "capacity": 10_000_000,
            "spendable_msat": 9_700_000_000,
            "receivable_msat": 300_000_000,
        }
    }
    return mod


def test_starved_node_loopout_earns_structural_credit():
    """Fully starved node: the structural credit alone (2500 sats vs 120 sats
    threshold) carries the loop-out past the profit guard.

    Arithmetic (marginal, amortized credit): amount 1M sats x 0.5 turnover
    per horizon x scarcity 1.0 x (realized 500ppm - channel posterior 0ppm)
    / 1e6 x STRUCTURAL_AMORTIZATION_HORIZONS (10) = 2500. The freed inbound
    keeps working beyond one ~3-day horizon; pricing only one horizon made
    the guard unflippable against real reverse-swap costs (~1000-2500 ppm).
    Spend stays bounded by the daily envelope, not by this estimate."""
    mod = _structural_module(scarcity=1.0)

    plan = mod._build_boltz_balance_plan(require_profitable=True)

    assert "error" not in plan
    recs = plan["recommendations"]
    assert len(recs) == 1
    econ = recs[0]["economics"]
    assert econ["structural_uplift_sats"] == 2500
    assert econ["passes_profit_guard"] is True
    assert econ["structural"] is True


def test_worked_example_starved_node_clears_realistic_swap_fee():
    """Calibration worked example: 1M amount, scarcity 1.0, realized 500 ppm,
    no posterior, swap fee 200 sats. Credit = 1M x 0.5 x 1.0 x 500/1e6 x 10
    = 2500 sats vs threshold 200 x 1.2 = 240 — passes comfortably. Before
    amortization the credit was 250, leaving essentially no room against
    real reverse-swap costs."""
    mod = _structural_module(scarcity=1.0, realized_ppm=500)
    bm = mod._require_boltz_manager()
    bm.quote.return_value = {"estimated_total_fee_sats": 200}

    plan = mod._build_boltz_balance_plan(
        require_profitable=True, profit_margin_factor=1.2
    )

    assert "error" not in plan
    econ = plan["recommendations"][0]["economics"]
    assert econ["estimated_swap_fee_sats"] == 200
    assert econ["structural_uplift_sats"] == 2500
    assert econ["passes_profit_guard"] is True
    assert econ["structural"] is True


def test_worked_example_healthy_node_still_exactly_zero():
    """Healthy node (scarcity 0) under the same realistic 200-sat fee: the
    amortization multiplier must not leak any credit — exactly 0."""
    mod = _structural_module(scarcity=0.0, realized_ppm=500)
    bm = mod._require_boltz_manager()
    bm.quote.return_value = {"estimated_total_fee_sats": 200}

    plan = mod._build_boltz_balance_plan(
        require_profitable=True, profit_margin_factor=1.2
    )

    assert "error" not in plan
    for rec in plan["recommendations"]:
        econ = rec["economics"]
        assert econ["structural_uplift_sats"] == 0
        assert econ["structural"] is False
        assert econ["passes_profit_guard"] is False


def test_structural_credit_is_marginal_over_channel_posterior():
    """Audit double-count case: DTS posterior 800ppm (tight) + realized
    800ppm + scarcity 1.0 must yield ~0 structural credit (was 600): when
    DTS already prices the channel's own volume, structural only credits the
    node-level premium over it — here the premium is zero."""
    mod = _structural_module(scarcity=1.0, realized_ppm=800)
    mod.fee_controller.get_dts_summary.return_value = {
        "broadcast_fee_ppm": 0,
        "posterior_mean": 800,
        "posterior_std": 50,  # tight: < 100
    }

    plan = mod._build_boltz_balance_plan(require_profitable=True)

    assert "error" not in plan
    econ = plan["recommendations"][0]["economics"]
    assert econ["structural_uplift_sats"] == 0


def test_structural_credit_pays_only_node_premium_over_posterior():
    """Realized 800ppm, tight channel posterior 300ppm: the credit prices
    only the 500ppm premium, amortized:
    1M x 0.5 x 1.0 x (800-300)/1e6 x 10 = 2500."""
    mod = _structural_module(scarcity=1.0, realized_ppm=800)
    mod.fee_controller.get_dts_summary.return_value = {
        "broadcast_fee_ppm": 0,
        "posterior_mean": 300,
        "posterior_std": 50,
    }

    plan = mod._build_boltz_balance_plan(require_profitable=True)

    assert "error" not in plan
    econ = plan["recommendations"][0]["economics"]
    assert econ["structural_uplift_sats"] == 2500


def test_loose_posterior_credits_full_realized_rate():
    """posterior_std >= 100 means DTS does not price this channel's volume:
    the full realized rate is credited (channel_posterior_ppm treated as 0)."""
    mod = _structural_module(scarcity=1.0, realized_ppm=500)
    mod.fee_controller.get_dts_summary.return_value = {
        "broadcast_fee_ppm": 0,
        "posterior_mean": 800,
        "posterior_std": 200,  # loose
    }

    plan = mod._build_boltz_balance_plan(require_profitable=True)

    assert "error" not in plan
    econ = plan["recommendations"][0]["economics"]
    # 1M x 0.5 x 1.0 x 500/1e6 x 10 horizons
    assert econ["structural_uplift_sats"] == 2500


def test_uplifts_independent_of_expected_horizon_days():
    """Neither uplift may scale with the caller's expected_horizon_days: the
    DTS uplift prices one horizon's turnover; the structural credit amortizes
    over the FIXED STRUCTURAL_AMORTIZATION_HORIZONS constant (not a caller
    knob), so changing expected_horizon_days must move neither."""
    mod3 = _structural_module(scarcity=1.0)
    mod3.fee_controller.get_dts_summary.return_value = {
        "broadcast_fee_ppm": 0, "posterior_mean": 400, "posterior_std": 50,
    }
    mod6 = _structural_module(scarcity=1.0)
    mod6.fee_controller.get_dts_summary.return_value = {
        "broadcast_fee_ppm": 0, "posterior_mean": 400, "posterior_std": 50,
    }

    plan3 = mod3._build_boltz_balance_plan(require_profitable=True, expected_horizon_days=3.0)
    plan6 = mod6._build_boltz_balance_plan(require_profitable=True, expected_horizon_days=6.0)

    econ3 = plan3["recommendations"][0]["economics"]
    econ6 = plan6["recommendations"][0]["economics"]
    # DTS uplift: 1M x 0.5 x 400/1e6 = 200, horizon-independent.
    assert econ3["dts_uplift_sats"] == 200
    assert econ6["dts_uplift_sats"] == econ3["dts_uplift_sats"]
    # Structural: 1M x 0.5 x 1.0 x (500-400)/1e6 x 10 = 500, independent of
    # expected_horizon_days (the x10 is the fixed amortization constant).
    assert econ3["structural_uplift_sats"] == 500
    assert econ6["structural_uplift_sats"] == econ3["structural_uplift_sats"]


def test_healthy_node_gets_zero_credit_and_unchanged_guard():
    """Healthy node (scarcity 0): credit must be exactly zero and the guard
    must behave exactly as before — the dead channel still fails it."""
    mod = _structural_module(scarcity=0.0)

    plan = mod._build_boltz_balance_plan(require_profitable=True)

    assert "error" not in plan
    for rec in plan["recommendations"]:
        econ = rec["economics"]
        assert econ["structural_uplift_sats"] == 0
        assert econ["structural"] is False
        assert econ["passes_profit_guard"] is False


def test_structural_budget_zero_disables_credit_entirely():
    """Envelope disabled (budget 0): even a fully starved node earns no credit."""
    mod = _structural_module(scarcity=1.0, structural_budget=0)

    plan = mod._build_boltz_balance_plan(require_profitable=True)

    assert "error" not in plan
    for rec in plan["recommendations"]:
        econ = rec["economics"]
        assert econ["structural_uplift_sats"] == 0
        assert econ["structural"] is False


def test_credit_scales_with_scarcity():
    full = _structural_module(scarcity=1.0)
    half = _structural_module(scarcity=0.5)

    full_plan = full._build_boltz_balance_plan(require_profitable=True)
    half_plan = half._build_boltz_balance_plan(require_profitable=True)

    full_credit = full_plan["recommendations"][0]["economics"]["structural_uplift_sats"]
    half_credit = half_plan["recommendations"][0]["economics"]["structural_uplift_sats"]
    assert full_credit > 0 and half_credit > 0
    assert full_credit == pytest.approx(2 * half_credit, rel=0.05)


def test_realized_ppm_zero_disables_credit():
    """No realized earn rate (new node, no forwards): credit must be zero."""
    mod = _structural_module(scarcity=1.0, realized_ppm=0)

    plan = mod._build_boltz_balance_plan(require_profitable=True)

    assert "error" not in plan
    for rec in plan["recommendations"]:
        assert rec["economics"]["structural_uplift_sats"] == 0


def test_db_and_status_called_once_not_per_candidate():
    """Receivable status and realized fee rate are hoisted out of the channel
    loop: one evaluation per plan build, regardless of candidate count."""
    mod = _structural_module(scarcity=1.0)
    status_calls = {"n": 0}
    base_status = mod._node_receivable_status

    def counting_status():
        status_calls["n"] += 1
        return base_status()

    mod._node_receivable_status = counting_status
    # TWO loop-out candidates, both at 97% local.
    mod.fee_controller._get_channels_info.return_value = {
        "100x1x0": {
            "peer_id": "02" + "b" * 64,
            "capacity": 10_000_000,
            "spendable_msat": 9_700_000_000,
            "receivable_msat": 300_000_000,
        },
        "101x1x0": {
            "peer_id": "02" + "c" * 64,
            "capacity": 10_000_000,
            "spendable_msat": 9_700_000_000,
            "receivable_msat": 300_000_000,
        },
    }

    plan = mod._build_boltz_balance_plan(require_profitable=True)

    assert "error" not in plan
    assert len(plan["recommendations"]) == 2
    assert mod.database.get_node_realized_fee_ppm.call_count == 1
    assert status_calls["n"] == 1


# ---------------------------------------------------------------------------
# Task 9 part C: volume floor, require_profitable gating, Task 8 gap tests
# ---------------------------------------------------------------------------

def test_realized_ppm_query_uses_volume_floor():
    """The hoist must request the realized ppm with the 1M-sat volume floor so
    a handful of lucky high-fee forwards cannot price the structural credit."""
    mod = _structural_module(scarcity=1.0)

    plan = mod._build_boltz_balance_plan(require_profitable=True)

    assert "error" not in plan
    mod.database.get_node_realized_fee_ppm.assert_called_once_with(
        window_days=7, min_volume_sats=1_000_000
    )


def _set_channel_policy(mod, fee_ppm, capacity=10_000_000):
    """Give every channel in the module's channels dict an own out-fee policy."""
    for ch in mod.fee_controller._get_channels_info.return_value.values():
        ch["fee_proportional_millionths"] = fee_ppm
        ch["capacity"] = capacity


def test_cascade_prefers_realized_7d():
    """7d window with volume clears the floor: it prices the credit and the
    later fallback stages are never consulted."""
    mod = _structural_module(scarcity=1.0, realized_ppm=500)

    plan = mod._build_boltz_balance_plan(require_profitable=True)

    assert "error" not in plan
    assert plan["structural_credit"]["source"] == "realized_7d"
    assert plan["structural_credit"]["realized_ppm"] == 500
    mod.database.get_node_realized_fee_ppm_30d.assert_not_called()
    assert plan["recommendations"][0]["economics"]["structural_uplift_sats"] > 0


def test_cascade_falls_back_to_realized_30d():
    """Inbound-starved chicken-and-egg: the 7d window returns 0 (below the
    volume floor), so the 30d rollup-backed rate prices the credit."""
    mod = _structural_module(scarcity=1.0, realized_ppm=0)
    mod.database.get_node_realized_fee_ppm_30d.return_value = 400

    plan = mod._build_boltz_balance_plan(require_profitable=True)

    assert "error" not in plan
    assert plan["structural_credit"]["source"] == "realized_30d"
    assert plan["structural_credit"]["realized_ppm"] == 400
    mod.database.get_node_realized_fee_ppm_30d.assert_called_once_with(
        min_volume_sats=1_000_000
    )
    assert plan["recommendations"][0]["economics"]["structural_uplift_sats"] > 0


def test_cascade_falls_back_to_policy_derived():
    """Both realized windows empty (the node cannot route -- exactly why it
    needs the drain): price the credit from our OWN out-fee policies,
    capacity-weighted and discounted by OWN_POLICY_HAIRCUT (asks that never
    cleared at volume are optimistic)."""
    mod = _structural_module(scarcity=1.0, realized_ppm=0)
    _set_channel_policy(mod, fee_ppm=1000)

    plan = mod._build_boltz_balance_plan(require_profitable=True)

    assert "error" not in plan
    assert plan["structural_credit"]["source"] == "policy_derived"
    # capacity-weighted mean 1000 ppm x 0.5 haircut = 500 ppm
    assert plan["structural_credit"]["realized_ppm"] == 500
    assert plan["recommendations"][0]["economics"]["structural_uplift_sats"] > 0


def test_policy_derived_is_capacity_weighted():
    """Two channels, 10M @ 1200 ppm and 5M @ 300 ppm: weighted mean is
    (10*1200 + 5*300)/15 = 900 ppm, haircut to 450."""
    mod = _structural_module(scarcity=1.0, realized_ppm=0)
    mod.fee_controller._get_channels_info.return_value = {
        "100x1x0": {
            "peer_id": "02" + "b" * 64,
            "capacity": 10_000_000,
            "spendable_msat": 9_700_000_000,
            "receivable_msat": 300_000_000,
            "fee_proportional_millionths": 1200,
        },
        "101x1x0": {
            "peer_id": "02" + "c" * 64,
            "capacity": 5_000_000,
            "spendable_msat": 4_850_000_000,
            "receivable_msat": 150_000_000,
            "fee_proportional_millionths": 300,
        },
    }

    plan = mod._build_boltz_balance_plan(require_profitable=True)

    assert "error" not in plan
    assert plan["structural_credit"]["source"] == "policy_derived"
    assert plan["structural_credit"]["realized_ppm"] == 450


def test_cascade_no_source_disables_credit():
    """No realized data anywhere and zero own policies: no source can price
    the credit, so it stays disabled (source None, uplift 0)."""
    mod = _structural_module(scarcity=1.0, realized_ppm=0)
    _set_channel_policy(mod, fee_ppm=0)

    plan = mod._build_boltz_balance_plan(require_profitable=True)

    assert "error" not in plan
    assert plan["structural_credit"]["source"] is None
    assert plan["structural_credit"]["realized_ppm"] == 0
    for rec in plan["recommendations"]:
        assert rec["economics"]["structural_uplift_sats"] == 0


def test_cascade_30d_error_falls_through_to_policy():
    """A failing 30d query must not kill the plan build -- the cascade
    degrades to the policy-derived stage."""
    mod = _structural_module(scarcity=1.0, realized_ppm=0)
    mod.database.get_node_realized_fee_ppm_30d.side_effect = RuntimeError("db down")
    _set_channel_policy(mod, fee_ppm=800)

    plan = mod._build_boltz_balance_plan(require_profitable=True)

    assert "error" not in plan
    assert plan["structural_credit"]["source"] == "policy_derived"
    assert plan["structural_credit"]["realized_ppm"] == 400


def test_cascade_not_consulted_when_node_healthy():
    """scarcity 0: no pricing source is consulted at all."""
    mod = _structural_module(scarcity=0.0)

    plan = mod._build_boltz_balance_plan(require_profitable=True)

    assert "error" not in plan
    mod.database.get_node_realized_fee_ppm.assert_not_called()
    mod.database.get_node_realized_fee_ppm_30d.assert_not_called()
    assert plan["structural_credit"]["source"] is None


def test_structural_flag_false_when_require_profitable_disabled():
    """require_profitable=False: every candidate passes the guard regardless of
    the credit, so the pass never DEPENDED on it — structural must be False."""
    mod = _structural_module(scarcity=1.0)

    plan = mod._build_boltz_balance_plan(require_profitable=False)

    assert "error" not in plan
    recs = plan["recommendations"]
    assert len(recs) == 1
    econ = recs[0]["economics"]
    assert econ["structural_uplift_sats"] > 0  # credit is still reported
    assert econ["structural"] is False


def test_loop_in_candidate_gets_zero_structural_credit():
    """Starved node, loop-in direction: the credit is loop-out only (loop-in
    consumes receivable instead of creating it)."""
    mod = _structural_module(scarcity=1.0)
    # 10% local => loop_in direction (low trigger default 40%)
    mod.fee_controller._get_channels_info.return_value = {
        "100x1x0": {
            "peer_id": "02" + "b" * 64,
            "capacity": 10_000_000,
            "spendable_msat": 1_000_000_000,
            "receivable_msat": 9_000_000_000,
        }
    }

    plan = mod._build_boltz_balance_plan(require_profitable=True)

    assert "error" not in plan
    recs = plan["recommendations"]
    assert len(recs) == 1
    assert recs[0]["direction"] == "loop_in"
    econ = recs[0]["economics"]
    assert econ["structural_uplift_sats"] == 0
    assert econ["structural"] is False


def test_candidate_passing_without_credit_not_structural():
    """A loop-out whose historical contribution alone clears the guard is NOT
    structural even when the starved node grants a positive credit."""
    mod = _structural_module(scarcity=1.0)
    prof = _make_prof_mock()
    # 30k sats over 30 days => 1000 sats/day; with horizon 3d and severity
    # ~0.85 the historical uplift (~2550) clears the 120-sat threshold alone.
    prof.revenue.total_contribution_sats = 30_000
    mod.profitability_analyzer.get_profitability.return_value = prof

    plan = mod._build_boltz_balance_plan(require_profitable=True)

    assert "error" not in plan
    econ = plan["recommendations"][0]["economics"]
    assert econ["passes_profit_guard"] is True
    assert econ["structural_uplift_sats"] > 0
    assert econ["structural"] is False


# ---------------------------------------------------------------------------
# Task 9 parts A+B: execution-time envelope gate + structural spend recording
# ---------------------------------------------------------------------------

CYCLE_PEER = "02" + "b" * 64


def _cycle_plan(structural=True, est_fee=100):
    """Minimal precomputed balance plan with one loop-out recommendation."""
    return {
        "generated_at": 1,
        "pending_swap_count": 0,
        "budget": {"remaining_24h_sats_estimate": 100_000},
        "recommendations": [
            {
                "channel_id": "100x1x0",
                "peer_id": CYCLE_PEER,
                "direction": "loop_out",
                "amount_sats": 200_000,
                "economics": {
                    "passes_profit_guard": True,
                    "estimated_swap_fee_sats": est_fee,
                    "structural_uplift_sats": 500 if structural else 0,
                    "structural": structural,
                },
            }
        ],
        "total_candidates": 1,
        "skipped_count": 0,
        "skipped_examples": [],
    }


def _cycle_module(envelope=1000, spent=0):
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    from modules.config import Config
    mod.config = Config(boltz_structural_budget_sats_per_day=envelope)
    mod.database = MagicMock()
    mod.database.get_category_spend_sats.return_value = spent
    bm = MagicMock()
    bm.loop_out.return_value = {"status": "accepted"}
    mod._require_boltz_manager = MagicMock(return_value=bm)
    return mod, bm


def test_structural_candidate_blocked_when_envelope_spent():
    """100 already spent + 100 estimated > 150 envelope: skip, never swap."""
    mod, bm = _cycle_module(envelope=150, spent=100)

    result = mod._execute_boltz_balance_cycle(
        dry_run=False, precomputed_plan=_cycle_plan(est_fee=100)
    )

    assert result["executed"] == []
    skip = result["skipped"][0]
    assert skip["reason"] == "structural_envelope_exhausted"
    assert skip["envelope_sats"] == 150
    assert skip["spent_24h_sats"] == 100
    assert skip["estimated_fee_sats"] == 100
    bm.loop_out.assert_not_called()


def test_structural_candidate_executes_within_envelope():
    mod, bm = _cycle_module(envelope=1000, spent=0)

    result = mod._execute_boltz_balance_cycle(
        dry_run=False, precomputed_plan=_cycle_plan(est_fee=100)
    )

    assert len(result["executed"]) == 1
    assert result["executed"][0]["status"] == "accepted"
    # The gate consulted the 24h structural spend exactly once.
    mod.database.get_category_spend_sats.assert_called_once()
    kwargs = mod.database.get_category_spend_sats.call_args.kwargs
    assert kwargs["subcategory"] == "structural"


def test_structural_spend_recorded_after_execution():
    """Structural execution must flow structural=True into boltz_manager so
    record_boltz_spend writes the fee under subcategory='structural' (the
    same boltz:{swap_id} event the tactical budget already counts once)."""
    mod, bm = _cycle_module(envelope=1000, spent=0)

    result = mod._execute_boltz_balance_cycle(
        dry_run=False, precomputed_plan=_cycle_plan(est_fee=100)
    )

    assert len(result["executed"]) == 1
    assert bm.loop_out.call_args.kwargs["structural"] is True


def test_non_structural_candidate_ignores_envelope():
    """A candidate that passes without the credit executes even when the
    structural envelope is exhausted (or disabled) — and never queries it."""
    mod, bm = _cycle_module(envelope=0, spent=10_000)

    result = mod._execute_boltz_balance_cycle(
        dry_run=False, precomputed_plan=_cycle_plan(structural=False)
    )

    assert len(result["executed"]) == 1
    mod.database.get_category_spend_sats.assert_not_called()
    assert bm.loop_out.call_args.kwargs["structural"] is False


def test_structural_envelope_provider_reads_live_config():
    """The provider wired into BoltzCliManager must read the LIVE config at
    call time (the envelope is runtime-mutable via revenue-config)."""
    mod = load_plugin_module()
    from modules.config import Config
    mod.config = Config(boltz_structural_budget_sats_per_day=750)
    assert mod._structural_envelope_sats_provider() == 750
    mod.config = Config()  # envelope off by default
    assert mod._structural_envelope_sats_provider() == 0


def _treasury_cycle_module(structural=True):
    mod, bm = _cycle_module(envelope=1000, spent=0)
    plan = _cycle_plan(structural=structural)
    plan["status"] = "ok"
    plan["treasury"] = {"deficit_sats": 900_000, "preferred_currency": "BTC"}
    mod._build_boltz_expansion_treasury_plan = MagicMock(return_value=plan)
    return mod, bm


def test_treasury_cycle_skips_structural_candidates():
    """The expansion-treasury executor has no structural-envelope accounting,
    so a candidate that only passes the profit guard via the scarcity credit
    must not fund the treasury — it is skipped (pre-credit behavior)."""
    mod, bm = _treasury_cycle_module(structural=True)

    result = mod.revenue_boltz_expansion_treasury_cycle(mod.plugin, dry_run=False)

    assert result["executed"] == []
    assert result["skipped"][0]["reason"] == "structural_credit_requires_balance_cycle"
    bm.loop_out.assert_not_called()


def test_treasury_cycle_executes_non_structural_candidates():
    mod, bm = _treasury_cycle_module(structural=False)

    result = mod.revenue_boltz_expansion_treasury_cycle(mod.plugin, dry_run=False)

    assert len(result["executed"]) == 1
    bm.loop_out.assert_called_once()


def _two_candidate_plan(est_fee=100):
    """Precomputed plan with TWO structural loop-out candidates on distinct
    channels/peers (distinct so the per-channel cooldown pre-claim cannot
    interfere with the second candidate)."""
    import copy

    plan = _cycle_plan(est_fee=est_fee)
    rec2 = copy.deepcopy(plan["recommendations"][0])
    rec2["channel_id"] = "101x1x0"
    rec2["peer_id"] = "02" + "c" * 64
    plan["recommendations"].append(rec2)
    plan["total_candidates"] = 2
    return plan


def test_envelope_accumulates_across_candidates_in_one_cycle():
    """Two structural candidates, envelope 150, each fee 100, max_actions=2:
    the gate must re-query the recorded 24h structural spend PER CANDIDATE,
    so the first execution (which records 100) exhausts the envelope for the
    second (100 + 100 > 150). Exactly one executes; the second is skipped.

    This pins the in-cycle race protection: if someone hoists the spend
    query out of the candidate loop, both candidates see spent=0 and both
    execute — and this test must fail."""
    mod, bm = _cycle_module(envelope=150, spent=0)

    # Stateful fakes: get_category_spend_sats returns the running sum of the
    # fees recorded so far; each accepted loop_out records a 100-sat fee.
    recorded_fees = []
    mod.database.get_category_spend_sats.side_effect = (
        lambda *a, **kw: sum(recorded_fees)
    )

    def _loop_out_records_spend(**kwargs):
        recorded_fees.append(100)
        return {"status": "accepted"}

    bm.loop_out.side_effect = _loop_out_records_spend

    result = mod._execute_boltz_balance_cycle(
        dry_run=False,
        max_actions=2,
        precomputed_plan=_two_candidate_plan(est_fee=100),
    )

    assert len(result["executed"]) == 1
    assert result["executed"][0]["status"] == "accepted"
    assert bm.loop_out.call_count == 1
    skips = [s for s in result["skipped"]
             if s["reason"] == "structural_envelope_exhausted"]
    assert len(skips) == 1
    assert skips[0]["spent_24h_sats"] == 100
    assert skips[0]["estimated_fee_sats"] == 100
    assert skips[0]["envelope_sats"] == 150
    # The gate consulted the spend once per structural candidate.
    assert mod.database.get_category_spend_sats.call_count == 2


def test_gate_fails_closed_on_spend_query_error():
    """Unknown 24h spend => assume the envelope is exhausted."""
    mod, bm = _cycle_module(envelope=1000)
    mod.database.get_category_spend_sats.side_effect = RuntimeError("db down")

    result = mod._execute_boltz_balance_cycle(
        dry_run=False, precomputed_plan=_cycle_plan(est_fee=100)
    )

    assert result["executed"] == []
    assert result["skipped"][0]["reason"] == "structural_envelope_exhausted"
    bm.loop_out.assert_not_called()


# ---------------------------------------------------------------------------
# Task 10: drain-demand residual from rebalancer prioritises channels first
# ---------------------------------------------------------------------------

def _drain_module():
    """Two 97%-local channels; neither has structural data (budget=0 keeps it
    neutral so structural credit does not interfere with ordering)."""
    mod = _make_planner_module()
    from modules.config import Config
    mod.config = Config(boltz_structural_budget_sats_per_day=0)
    mod._node_receivable_status = lambda: {
        "receivable_ratio": 0.03, "scarcity": 0.0,
        "total_capacity_sats": 20_000_000, "total_receivable_sats": 600_000,
    }
    pa = MagicMock()
    pa.analyze_all_channels.return_value = None
    pa.get_profitability.return_value = _make_prof_mock()
    mod.profitability_analyzer = pa
    # Two identical 97%-local channels; different peer ids so they are independent.
    mod.fee_controller._get_channels_info.return_value = {
        "100x1x0": {
            "peer_id": "02" + "a" * 64,
            "capacity": 10_000_000,
            "spendable_msat": 9_700_000_000,
            "receivable_msat": 300_000_000,
        },
        "200x2x0": {
            "peer_id": "02" + "c" * 64,
            "capacity": 10_000_000,
            "spendable_msat": 9_700_000_000,
            "receivable_msat": 300_000_000,
        },
    }
    return mod


def test_drain_demand_channels_rank_first():
    """Channel in drain-demand must sort to position 0 (drain_bonus=1.5 lifts
    its multi_goal_value above the identical non-demand channel)."""
    mod = _drain_module()
    mod.rebalancer = MagicMock()
    demand = MagicMock()
    entry = MagicMock()
    entry.channel_id = "200x2x0"
    demand.entries = [entry]
    mod.rebalancer.rebalance_engine_v2.get_drain_demand.return_value = demand

    plan = mod._build_boltz_balance_plan(require_profitable=False)

    assert "error" not in plan
    recs = plan["recommendations"]
    assert len(recs) == 2
    assert recs[0]["channel_id"] == "200x2x0", (
        f"Expected demand channel first, got {recs[0]['channel_id']}"
    )
    assert recs[0]["score"]["in_drain_demand"] is True
    assert recs[1]["score"]["in_drain_demand"] is False


def test_drain_demand_colon_scids_normalized():
    """entry.channel_id in colon form (200:2:0) must match the x-form scid_display."""
    mod = _drain_module()
    mod.rebalancer = MagicMock()
    demand = MagicMock()
    entry = MagicMock()
    entry.channel_id = "200:2:0"   # colon form
    demand.entries = [entry]
    mod.rebalancer.rebalance_engine_v2.get_drain_demand.return_value = demand

    plan = mod._build_boltz_balance_plan(require_profitable=False)

    assert "error" not in plan
    recs = plan["recommendations"]
    assert recs[0]["channel_id"] == "200x2x0"
    assert recs[0]["score"]["in_drain_demand"] is True


def test_missing_engine_neutralizes():
    """mod.rebalancer = None must not crash the plan; all in_drain_demand False."""
    mod = _drain_module()
    mod.rebalancer = None

    plan = mod._build_boltz_balance_plan(require_profitable=False)

    assert "error" not in plan
    for rec in plan["recommendations"]:
        assert rec["score"]["in_drain_demand"] is False


def test_drain_demand_fetch_error_neutralizes():
    """get_drain_demand raising must not crash the plan build; every
    candidate simply scores with in_drain_demand False."""
    mod = _drain_module()
    mod.rebalancer = MagicMock()
    mod.rebalancer.rebalance_engine_v2.get_drain_demand.side_effect = (
        RuntimeError("engine exploded")
    )

    plan = mod._build_boltz_balance_plan(require_profitable=False)

    assert "error" not in plan
    recs = plan["recommendations"]
    assert len(recs) == 2
    for rec in recs:
        assert rec["score"]["in_drain_demand"] is False


def test_drain_demand_fetched_once():
    """get_drain_demand must be called exactly once per plan build, even when
    there are multiple loop-out candidates."""
    mod = _drain_module()
    mod.rebalancer = MagicMock()
    demand = MagicMock()
    demand.entries = []
    mod.rebalancer.rebalance_engine_v2.get_drain_demand.return_value = demand

    plan = mod._build_boltz_balance_plan(require_profitable=False)

    assert "error" not in plan
    assert len(plan["recommendations"]) == 2
    mod.rebalancer.rebalance_engine_v2.get_drain_demand.assert_called_once()


# ---------------------------------------------------------------------------
# Runtime mutability: drain knobs and the Boltz auto-cycle toggle must be
# settable via revenue-config without a lightningd restart (config-file
# edits for plugin options only load at daemon startup).
# ---------------------------------------------------------------------------


def _runtime_db():
    db = MagicMock()
    store = {}
    db.set_config_override.side_effect = lambda k, v: store.update({k: v}) or 1
    db.get_config_override.side_effect = lambda k: store.get(k)
    return db


def test_drain_knobs_are_public_runtime_keys():
    from modules.config import PUBLIC_RUNTIME_KEYS
    for key in (
        "boltz_auto_cycle_enabled",
        "boltz_structural_budget_sats_per_day",
        "receivable_ratio_target",
        "receivable_ratio_floor",
        "drain_fee_discount_max",
    ):
        assert key in PUBLIC_RUNTIME_KEYS, key


def test_update_runtime_enables_auto_cycle():
    from modules.config import Config
    cfg = Config()
    result = cfg.update_runtime(_runtime_db(), "boltz_auto_cycle_enabled", "true")
    assert result.get("status") == "success"
    assert cfg.boltz_auto_cycle_enabled is True


def test_update_runtime_sets_structural_budget():
    from modules.config import Config
    cfg = Config()
    result = cfg.update_runtime(
        _runtime_db(), "boltz_structural_budget_sats_per_day", "200")
    assert result.get("status") == "success"
    assert cfg.boltz_structural_budget_sats_per_day == 200


def test_update_runtime_rejects_floor_above_target():
    from modules.config import Config
    cfg = Config()  # target 0.30
    result = cfg.update_runtime(_runtime_db(), "receivable_ratio_floor", "0.4")
    assert "error" in result
    assert cfg.receivable_ratio_floor == 0.20


def test_update_runtime_rejects_target_below_floor():
    from modules.config import Config
    cfg = Config()  # floor 0.20
    result = cfg.update_runtime(_runtime_db(), "receivable_ratio_target", "0.1")
    assert "error" in result
    assert cfg.receivable_ratio_target == 0.30


def test_update_runtime_rejects_out_of_range_budget():
    from modules.config import Config
    cfg = Config()
    result = cfg.update_runtime(
        _runtime_db(), "boltz_structural_budget_sats_per_day", "2000000")
    assert "error" in result
