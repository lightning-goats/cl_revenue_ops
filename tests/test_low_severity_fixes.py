"""Low-severity audit fixes: deprecated alias gating, option defaults,
policy validation, router correctness, spend-reserve serialization, and
rounding consistency."""

import threading
import time
from unittest.mock import MagicMock

import pytest

from tests.plugin_test_utils import load_plugin_module


PEER = "02" + "a" * 64
OTHER = "03" + "b" * 64


# ---------------------------------------------------------------------------
# Deprecated revenue-ignore / revenue-unignore must respect the policy gate
# ---------------------------------------------------------------------------


def _gated_module():
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    mod.policy_manager = MagicMock()
    return mod


def test_revenue_ignore_blocked_without_internal_override():
    mod = _gated_module()

    result = mod.revenue_ignore(mod.plugin, PEER)

    assert "error" in result
    mod.policy_manager.set_policy.assert_not_called()


def test_revenue_ignore_allowed_with_internal_override():
    mod = _gated_module()

    result = mod.revenue_ignore(mod.plugin, PEER, internal=True)

    assert result.get("status") == "success"
    mod.policy_manager.set_policy.assert_called_once()


def test_revenue_unignore_blocked_without_internal_override():
    mod = _gated_module()

    result = mod.revenue_unignore(mod.plugin, PEER)

    assert "error" in result
    mod.policy_manager.delete_policy.assert_not_called()


def test_revenue_unignore_rejects_malformed_peer_id():
    mod = _gated_module()

    result = mod.revenue_unignore(mod.plugin, "not-a-pubkey", internal=True)

    assert "error" in result
    mod.policy_manager.delete_policy.assert_not_called()



# ---------------------------------------------------------------------------
# Static policy without a fee target must be rejected at write time
# ---------------------------------------------------------------------------


def test_set_policy_rejects_static_without_fee_target(mock_database, mock_plugin):
    from modules.policy_manager import PolicyManager

    pm = PolicyManager(mock_database, mock_plugin)

    with pytest.raises(ValueError, match="fee_ppm"):
        pm.set_policy(PEER, strategy="static")


def test_set_policy_accepts_static_with_zero_target(mock_database, mock_plugin):
    from modules.policy_manager import PolicyManager, FeeStrategy

    pm = PolicyManager(mock_database, mock_plugin)

    policy = pm.set_policy(PEER, strategy="static", fee_ppm_target=0)

    assert policy.strategy == FeeStrategy.STATIC
    assert policy.fee_ppm_target == 0


# ---------------------------------------------------------------------------
# Router v2: final-hop policy must come from the requested dest channel
# ---------------------------------------------------------------------------


def _parallel_channels_rpc():
    rpc = MagicMock()
    rpc.listpeerchannels.return_value = {
        "channels": [
            {
                "peer_id": PEER,
                "short_channel_id": "100x1x0",
                "updates": {
                    "remote": {
                        "fee_proportional_millionths": 1000,
                        "fee_base_msat": 1000,
                        "cltv_expiry_delta": 80,
                    }
                },
            },
            {
                "peer_id": PEER,
                "short_channel_id": "200x1x0",
                "updates": {
                    "remote": {
                        "fee_proportional_millionths": 50,
                        "fee_base_msat": 0,
                        "cltv_expiry_delta": 34,
                    }
                },
            },
        ]
    }
    return rpc


def _router_v2():
    from modules.rebalance_router_v2 import RebalanceRouter

    plugin = MagicMock()
    plugin.rpc = _parallel_channels_rpc()
    router = RebalanceRouter.__new__(RebalanceRouter)
    router.plugin = plugin
    router.data_service = None
    router.our_node_id = "03" + "f" * 64
    return router


def test_final_hop_policy_matches_requested_channel():
    router = _router_v2()

    policy = router._get_final_hop_policy(PEER, dest_channel_id="200x1x0")

    assert policy["fee_ppm"] == 50
    assert policy["fee_base_msat"] == 0


def test_final_hop_cltv_matches_requested_channel():
    router = _router_v2()

    cltv = router._get_dest_channel_cltv(PEER, dest_channel_id="200x1x0")

    assert cltv == 34


def test_final_hop_policy_falls_back_without_channel_filter():
    router = _router_v2()

    policy = router._get_final_hop_policy(PEER)

    assert policy["fee_ppm"] == 1000  # first channel, legacy behavior


# ---------------------------------------------------------------------------
# Router v3: exclude layer names must be unique under concurrency
# ---------------------------------------------------------------------------


def test_exclude_layer_names_unique_across_threads():
    from modules.rebalance_router_v3 import RebalanceRouterV3

    names = []
    names_lock = threading.Lock()

    def make_router():
        router = RebalanceRouterV3.__new__(RebalanceRouterV3)
        router.plugin = MagicMock()
        router.data_service = None
        # __init__ is bypassed; _exclude_layer consults the cycle state to
        # decide between cached and per-call layer lifetimes.
        router._cycle_state = threading.local()
        return router

    def worker():
        router = make_router()
        for _ in range(50):
            with router._exclude_layer(["100x1x0/1"]) as layer_name:
                with names_lock:
                    names.append(layer_name)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(names) == 200
    assert len(set(names)) == 200


# ---------------------------------------------------------------------------
# revenue-spend-reserve: budget check + reserve must be serialized
# ---------------------------------------------------------------------------


def test_spend_reserve_serializes_check_and_reserve():
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    mod.database = MagicMock()
    mod.database.reserve_spend.return_value = True

    in_check = threading.Event()
    release = threading.Event()
    call_times = []

    def budget_status(window_hours=None, force_fresh=False):
        # DD1/P2-011: the gating path now calls with force_fresh=True (live).
        call_times.append(time.monotonic())
        if len(call_times) == 1:
            in_check.set()
            release.wait(timeout=5)
        return {"remaining_sats": 1000, "effective_budget_sats": 1000, "window_hours": 24}

    mod._total_cost_budget_status = budget_status

    results = {}

    def first():
        results["a"] = mod.revenue_spend_reserve(
            mod.plugin, reservation_id="r-a", amount_sats=100, category="test"
        )

    def second():
        results["b"] = mod.revenue_spend_reserve(
            mod.plugin, reservation_id="r-b", amount_sats=100, category="test"
        )

    t1 = threading.Thread(target=first)
    t1.start()
    assert in_check.wait(timeout=5)
    t2 = threading.Thread(target=second)
    t2.start()
    time.sleep(0.3)
    # While the first caller is inside the check+reserve critical section,
    # the second caller must not have entered the budget check.
    budget_calls_during_hold = len(call_times)
    release.set()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert budget_calls_during_hold == 1
    assert results["a"].get("status") == "success"
    assert results["b"].get("status") == "success"


# ---------------------------------------------------------------------------
# Lifetime revenue rounding must match the other revenue reports (ceil)
# ---------------------------------------------------------------------------


def test_lifetime_report_revenue_uses_ceiling(mock_plugin):
    from modules.profitability_analyzer import ChannelProfitabilityAnalyzer

    analyzer = ChannelProfitabilityAnalyzer.__new__(ChannelProfitabilityAnalyzer)
    analyzer.database = MagicMock()
    analyzer.database.get_lifetime_stats.return_value = {
        "total_revenue_msat": 500,  # sub-sat revenue must stay visible
        "total_opening_cost_sats": 0,
        "total_closure_cost_sats": 0,
        "total_rebalance_cost_sats": 0,
        "total_forwards": 1,
    }
    analyzer.database.get_closed_channels_summary.return_value = {}
    analyzer.plugin = mock_plugin

    report = analyzer.get_lifetime_report()

    assert report["lifetime_revenue_sats"] == 1
