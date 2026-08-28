import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = ROOT / "tools" / "polar_clboss_runner.py"
    spec = importlib.util.spec_from_file_location("polar_clboss_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_identity_assignment_crosses_between_replicas():
    runner = load_runner()

    assert runner.assignment_for(1) == {
        "revenue_ops": "identity-a",
        "clboss": "identity-b",
    }
    assert runner.assignment_for(2) == {
        "revenue_ops": "identity-b",
        "clboss": "identity-a",
    }
    assert runner.assignment_for(3) == runner.assignment_for(1)


def test_container_name_is_narrowly_scoped():
    runner = load_runner()

    assert runner.container_name(4, 2, "identity-b") == "polar-n4-clboss-r2-identity-b"
    with pytest.raises(runner.RunnerError, match="unknown contender"):
        runner.container_name(4, 2, "../backend1")


def test_docker_running_fails_closed_when_daemon_is_unavailable(monkeypatch):
    runner = load_runner()
    result = type(
        "Result",
        (),
        {"returncode": 1, "stdout": "", "stderr": "permission denied connecting to Docker"},
    )()
    monkeypatch.setattr(runner, "_run", lambda *_args, **_kwargs: result)

    with pytest.raises(runner.RunnerError, match="cannot inspect Docker"):
        runner.docker_running("polar-n4-clboss-r1-identity-a")


def test_docker_running_reports_only_a_known_missing_container_as_absent(monkeypatch):
    runner = load_runner()
    result = type(
        "Result", (), {"returncode": 1, "stdout": "", "stderr": "No such object: missing"}
    )()
    monkeypatch.setattr(runner, "_run", lambda *_args, **_kwargs: result)

    assert runner.docker_running("missing") is False
    assert runner.docker_exists("missing") is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [(123, 123), ("123msat", 123), ({"msat": 123}, 123), ({"msat": "123msat"}, 123)],
)
def test_msat_value_accepts_all_cln_encodings(value, expected):
    runner = load_runner()

    assert runner.msat_value(value) == expected


@pytest.mark.parametrize("value", [None, True, "bogus", "-1msat"])
def test_msat_value_rejects_malformed_or_negative_values(value):
    runner = load_runner()

    with pytest.raises(runner.RunnerError, match="msat value"):
        runner.msat_value(value)


def test_onchain_address_prefers_v26_p2tr_and_accepts_legacy_bech32():
    runner = load_runner()

    assert runner.onchain_address({"p2tr": "bcrt1ptest", "bech32": "bcrt1qtest"}) == "bcrt1ptest"
    assert runner.onchain_address({"bech32": "bcrt1qtest"}) == "bcrt1qtest"
    with pytest.raises(runner.RunnerError, match="regtest segwit"):
        runner.onchain_address({"p2tr": "bc1pmainnet"})


def test_wait_wallet_funds_handles_v26_amount_encoding(monkeypatch):
    runner = load_runner()
    calls = iter(
        [
            {"outputs": []},
            {"outputs": [{"status": "confirmed", "amount_msat": "2200000000msat"}]},
        ]
    )
    monkeypatch.setattr(runner, "cln_rpc", lambda *_args: next(calls))
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)

    runner.wait_wallet_funds("polar-n4-clboss-r1-identity-a", timeout_seconds=1)


def test_wallet_funding_leaves_emergency_reserve_and_fee_headroom(monkeypatch):
    runner = load_runner()
    addresses = iter(({"p2tr": "bcrt1pa"}, {"p2tr": "bcrt1pb"}))
    commands = []
    monkeypatch.setattr(runner, "cln_rpc", lambda *_args: next(addresses))
    monkeypatch.setattr(
        runner,
        "_run",
        lambda command, **_kwargs: commands.append(command)
        or type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    )
    monkeypatch.setattr(runner, "_mine", lambda *_args: None)

    runner._fund_wallet("polar-n4-clboss-r1-identity-a", object(), 4)

    funded = [command[-1] for command in commands if "sendtoaddress" in command]
    assert funded == ["0.01100000", "0.01100000"]
    assert runner.FUNDING_UTXO_SATS - runner.CHANNEL_CAPACITY_SATS >= 100_000


def test_totals_delta_counts_policy_change_without_inventing_rebalance_cost():
    runner = load_runner()
    before = runner.Totals(
        2, 20_000, 20, 500_000, (("1x1x1", 1, 10),),
        min_local_balance_ppm=400_000,
        max_local_balance_ppm=600_000,
        worst_channel_imbalance_ppm=200_000,
    )
    after = runner.Totals(
        5, 65_000, 65, 490_000, (("1x1x1", 1, 15),),
        min_local_balance_ppm=70_000,
        max_local_balance_ppm=910_000,
        worst_channel_imbalance_ppm=860_000,
    )

    assert runner.totals_delta(before, after) == {
        "forward_count": 3,
        "volume_msat": 45_000,
        "routing_fee_msat": 45,
        "mean_local_liquidity_sats": 495_000,
        "ending_min_local_balance_ppm": 70_000,
        "ending_max_local_balance_ppm": 910_000,
        "ending_worst_channel_imbalance_ppm": 860_000,
        "channels": {},
        "policy_changes": 1,
        "rebalance_cost_msat": 0,
    }


def test_totals_delta_accounts_for_completed_circular_payment_cost():
    runner = load_runner()
    before = runner.Totals(0, 0, 0, 500_000, (), rebalance_cost_msat=100)
    after = runner.Totals(0, 0, 0, 500_000, (), rebalance_cost_msat=375)

    assert runner.totals_delta(before, after)["rebalance_cost_msat"] == 275


def test_contender_totals_normalizes_channel_depletion_by_capacity(monkeypatch):
    runner = load_runner()
    channels = [
        {
            "short_channel_id": "1x1x0",
            "to_us_msat": 100_000,
            "total_msat": 1_000_000,
            "updates": {"local": {"fee_base_msat": 1, "fee_proportional_millionths": 5}},
        },
        {
            "short_channel_id": "1x1x1",
            "to_us_msat": 1_800_000,
            "total_msat": 2_000_000,
            "updates": {"local": {"fee_base_msat": 1, "fee_proportional_millionths": 10}},
        },
    ]

    def rpc(_container, method, *_args):
        if method == "listforwards":
            return {"forwards": []}
        if method == "getinfo":
            return {"id": "node-id"}
        if method == "listsendpays":
            return {"payments": []}
        raise AssertionError(method)

    monkeypatch.setattr(runner, "cln_rpc", rpc)
    monkeypatch.setattr(runner, "active_channels", lambda _container: channels)

    totals = runner.contender_totals("polar-n4-clboss-r1-identity-a")

    assert totals.mean_local_liquidity_sats == 950
    assert totals.min_local_balance_ppm == 100_000
    assert totals.max_local_balance_ppm == 900_000
    assert totals.worst_channel_imbalance_ppm == 800_000


def test_totals_delta_reports_treatment_lane_economics():
    runner = load_runner()
    before = runner.Totals(
        1, 5_000, 5, 500_000, (),
        channel_metrics=(("1x1x0", 1, 5_000, 5),),
    )
    after = runner.Totals(
        4, 50_000, 50, 500_000, (),
        channel_metrics=(
            ("1x1x0", 3, 35_000, 35),
            ("1x1x1", 1, 15_000, 15),
        ),
    )

    assert runner.totals_delta(before, after)["channels"] == {
        "1x1x0": {"forward_count": 2, "volume_msat": 30_000, "routing_fee_msat": 30},
        "1x1x1": {"forward_count": 1, "volume_msat": 15_000, "routing_fee_msat": 15},
    }


def test_rebalance_cost_counts_only_completed_self_payments(monkeypatch):
    runner = load_runner()

    def rpc(_container, method, *_args):
        if method == "getinfo":
            return {"id": "our-node"}
        assert method == "listsendpays"
        return {
            "payments": [
                {
                    "status": "complete", "destination": "our-node",
                    "amount_msat": "100000msat", "amount_sent_msat": "100250msat",
                },
                {
                    "status": "pending", "destination": "our-node",
                    "amount_msat": 100000, "amount_sent_msat": 100999,
                },
                {
                    "status": "complete", "destination": "someone-else",
                    "amount_msat": 100000, "amount_sent_msat": 100999,
                },
            ]
        }

    monkeypatch.setattr(runner, "cln_rpc", rpc)

    assert runner.rebalance_cost_msat("polar-n4-clboss-r1-identity-a") == 250


def test_clboss_spend_cap_disables_rebalancing_at_limit(monkeypatch):
    runner = load_runner()
    state = {
        "assignment": {"clboss": "identity-a"},
        "contenders": {
            "identity-a": {"container": "polar-n4-clboss-r1-identity-a"},
        },
        "full_stack": {
            "spend_cap_sats_per_controller": 1_000,
            "clboss_rebalance_cost_baseline_msat": 250,
        },
    }
    calls = []
    monkeypatch.setattr(runner, "rebalance_cost_msat", lambda _container: 1_000_250)
    monkeypatch.setattr(
        runner, "cln_rpc", lambda container, method, *args: calls.append((container, method, args)) or {}
    )

    monitor = runner.enforce_clboss_spend_cap(state)

    assert monitor == {
        "cap_msat": 1_000_000,
        "spent_msat": 1_000_000,
        "cap_enforced": True,
        "disabled_now": True,
    }
    assert calls == [
        (
            "polar-n4-clboss-r1-identity-a",
            "setconfig",
            ("clboss-rebalance-mode", "off"),
        )
    ]
    assert state["full_stack"]["clboss_spend_cap_enforced_at_msat"] == 1_000_000


def test_clboss_spend_cap_does_not_disable_below_limit(monkeypatch):
    runner = load_runner()
    state = {
        "assignment": {"clboss": "identity-b"},
        "contenders": {
            "identity-b": {"container": "polar-n4-clboss-r2-identity-b"},
        },
        "full_stack": {
            "spend_cap_sats_per_controller": 1_000,
            "clboss_rebalance_cost_baseline_msat": 500,
        },
    }
    monkeypatch.setattr(runner, "rebalance_cost_msat", lambda _container: 999_500)
    monkeypatch.setattr(
        runner,
        "cln_rpc",
        lambda *_args: pytest.fail("CLBOSS must stay enabled below the cap"),
    )

    monitor = runner.enforce_clboss_spend_cap(state)

    assert monitor["spent_msat"] == 999_000
    assert monitor["cap_enforced"] is False
    assert monitor["disabled_now"] is False


def test_setup_spend_baseline_waits_for_late_accounting(monkeypatch):
    runner = load_runner()
    samples = iter((0, 0, 310))
    sleeps = []

    def rpc(_container, _method, *_args):
        return {
            "capital_controls": {
                "total_liquidity_breakdown": {
                    "actual_spent_by_category": {
                        "rebalance": 0,
                        "open": next(samples),
                        "close": 0,
                    }
                }
            }
        }

    monkeypatch.setattr(runner, "cln_rpc", rpc)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)

    baseline = runner.wait_for_setup_spend_baseline(
        "polar-n4-clboss-r1-identity-a", attempts=3, poll_seconds=0.25
    )

    assert baseline == 310
    assert sleeps == [0.25, 0.25]


def test_start_revenue_uses_accelerated_tournament_cadences(monkeypatch):
    runner = load_runner()
    calls = []
    cadence_keys = (
        "revenue-ops-flow-interval",
        "revenue-ops-fee-interval",
        "revenue-ops-rebalance-interval",
    )

    def rpc(container, method, *args):
        calls.append((container, method, args))
        if method == "listconfigs":
            return {
                "configs": {
                    key: {"value_str": str(runner.TOURNAMENT_CYCLE_SECONDS)}
                    for key in cadence_keys
                }
            }
        if method == "revenue-status":
            return {
                "operator_controls": {
                    "values": {"paused": True, "daily_budget_sats": 0}
                }
            }
        return {}

    monkeypatch.setattr(runner, "cln_rpc", rpc)

    runner.start_revenue("polar-n4-clboss-r1-identity-a")

    start_args = next(args for _container, method, args in calls if method == "-k")
    for key in cadence_keys:
        assert f"{key}={runner.TOURNAMENT_CYCLE_SECONDS}" in start_args


def test_acquisition_treatment_pins_one_lane_and_restores_controls(monkeypatch, tmp_path):
    runner = load_runner()
    state_file = runner.state_path(tmp_path, 1)
    cln_channel_id = "a" * 64
    lnd_channel_id = "b" * 64
    cln_peer = "02" + "c" * 64
    lnd_peer = "03" + "d" * 64
    runner.write_json_atomic(
        state_file,
        {
            "schema": runner.SCHEMA,
            "status": "isolated_full_stack_ready",
            "events": [],
            "assignment": {"revenue_ops": "identity-a", "clboss": "identity-b"},
            "contenders": {
                "identity-a": {"container": "polar-n4-clboss-r1-identity-a"},
                "identity-b": {"container": "polar-n4-clboss-r1-identity-b"},
            },
            "channels": [
                {
                    "funder": "identity-a", "sink": "cln-sink",
                    "result": {"channel_id": cln_channel_id},
                },
                {
                    "funder": "identity-a", "sink": "lnd-sink",
                    "result": {"channel_id": lnd_channel_id},
                },
            ],
        },
    )
    rows = [
        {
            "channel_id": cln_channel_id, "short_channel_id": "1x1x0",
            "peer_id": cln_peer,
            "updates": {"local": {"fee_base_msat": 0, "fee_proportional_millionths": 18}},
        },
        {
            "channel_id": lnd_channel_id, "short_channel_id": "1x1x1",
            "peer_id": lnd_peer,
            "updates": {"local": {"fee_base_msat": 0, "fee_proportional_millionths": 15}},
        },
    ]
    runtime = {"min_fee_ppm_saturated": 0, "policies": {}}

    def rpc(_container, method, *args):
        if method == "revenue-policy" and args == ("list",):
            return {"policies": []}
        if method == "revenue-status":
            return {"operator_controls": {"values": {
                "min_fee_ppm_saturated": runtime["min_fee_ppm_saturated"]
            }}}
        if method == "revenue-config":
            runtime["min_fee_ppm_saturated"] = int(args[-1])
            return {"status": "success"}
        if method == "revenue-fee-cycle":
            return {"status": "success"}
        if method == "-k" and args[0] == "revenue-policy":
            params = dict(arg.split("=", 1) for arg in args[1:] if "=" in arg)
            peer = params["peer_id"]
            if params["action"] == "set":
                fee = int(params["fee_ppm"])
                runtime["policies"][peer] = fee
                next(row for row in rows if row["peer_id"] == peer)["updates"]["local"][
                    "fee_proportional_millionths"
                ] = fee
                return {"status": "success"}
            runtime["policies"].pop(peer, None)
            return {"status": "success"}
        raise AssertionError((method, args))

    monkeypatch.setattr(runner, "cln_rpc", rpc)
    monkeypatch.setattr(runner, "active_channels", lambda _container: rows)

    active = runner.apply_acquisition_treatment(
        replica=1, results_dir=tmp_path, family="cln", fee_ppm=0
    )

    assert active["status"] == "acquisition_ready"
    assert active["acquisition_treatment"]["treatment_lane"]["fee_ppm"] == 18
    assert runtime == {
        "min_fee_ppm_saturated": 0,
        "policies": {cln_peer: 0, lnd_peer: 15},
    }

    restored = runner.restore_acquisition(replica=1, results_dir=tmp_path)

    assert restored["status"] == "isolated_full_stack_ready"
    assert restored["acquisition_treatment"]["status"] == "restored"
    assert runtime == {"min_fee_ppm_saturated": 0, "policies": {}}


def test_acquisition_treatment_rejects_negative_fee(tmp_path):
    runner = load_runner()

    with pytest.raises(runner.RunnerError, match="nonnegative integer"):
        runner.apply_acquisition_treatment(
            replica=1, results_dir=tmp_path, family="cln", fee_ppm=-1
        )


def test_totals_delta_fails_closed_on_counter_regression():
    runner = load_runner()
    before = runner.Totals(2, 20_000, 20, 500_000, ())
    after = runner.Totals(1, 20_000, 20, 500_000, ())

    with pytest.raises(runner.RunnerError, match="regressed"):
        runner.totals_delta(before, after)


def test_family_metrics_reconcile_by_scored_lane():
    runner = load_runner()
    delta = {
        "forward_count": 3,
        "volume_msat": 30_000,
        "routing_fee_msat": 30,
        "channels": {
            "1x1x0": {"forward_count": 2, "volume_msat": 20_000, "routing_fee_msat": 20},
            "1x1x1": {"forward_count": 1, "volume_msat": 10_000, "routing_fee_msat": 10},
        },
    }
    lanes = {
        "1x1x0": {"family": "cln", "side": "sink"},
        "1x1x1": {"family": "lnd", "side": "payer"},
    }

    assert runner.family_metrics(delta, lanes) == {
        "cln": {"forward_count": 2, "volume_msat": 20_000, "routing_fee_msat": 20},
        "lnd": {"forward_count": 1, "volume_msat": 10_000, "routing_fee_msat": 10},
    }


def test_family_metrics_fail_closed_on_unmapped_nonzero_channel():
    runner = load_runner()
    delta = {
        "forward_count": 1,
        "volume_msat": 10_000,
        "routing_fee_msat": 10,
        "channels": {
            "unknown": {"forward_count": 1, "volume_msat": 10_000, "routing_fee_msat": 10},
        },
    }

    with pytest.raises(runner.RunnerError, match="cannot attribute"):
        runner.family_metrics(delta, {})


def test_resolve_lane_map_requires_each_family_and_side(monkeypatch):
    runner = load_runner()
    peers = {
        "cln-payer-id": {"family": "cln", "side": "payer"},
        "cln-sink-id": {"family": "cln", "side": "sink"},
        "lnd-payer-id": {"family": "lnd", "side": "payer"},
        "lnd-sink-id": {"family": "lnd", "side": "sink"},
    }
    rows = [
        {"short_channel_id": f"1x1x{i}", "peer_id": peer_id}
        for i, peer_id in enumerate(peers)
    ]
    state = {"contenders": {"identity-a": {"container": "contender"}}}
    monkeypatch.setattr(runner, "peer_families", lambda _network_id: peers)
    monkeypatch.setattr(runner, "active_channels", lambda _container: rows)

    mapped = runner.resolve_lane_map(state)

    assert {tuple((row["family"], row["side"])) for row in mapped["identity-a"].values()} == {
        ("cln", "payer"), ("cln", "sink"), ("lnd", "payer"), ("lnd", "sink"),
    }


def test_automatic_acquisition_waits_for_native_episode_without_forced_cycle(
    monkeypatch, tmp_path
):
    runner = load_runner()
    path = runner.state_path(tmp_path, 1)
    runner.write_json_atomic(path, {
        "schema": runner.SCHEMA,
        "status": "isolated_full_stack_ready",
        "events": [],
        "assignment": {"revenue_ops": "identity-a", "clboss": "identity-b"},
        "contenders": {
            "identity-a": {"container": "revenue"},
            "identity-b": {"container": "clboss"},
        },
    })
    lane_before = {
        "cln": {"family": "cln", "channel_id": "a", "peer_id": "pa", "fee_ppm": 15},
        "lnd": {"family": "lnd", "channel_id": "b", "peer_id": "pb", "fee_ppm": 15},
    }
    lane_active = {**lane_before, "cln": {**lane_before["cln"], "fee_ppm": 0}}
    lane_calls = iter((lane_before, lane_active))
    rpc_calls = []

    def rpc(_container, method, *args):
        rpc_calls.append((method, args))
        if method == "revenue-policy":
            return {"policies": []}
        if method == "revenue-status":
            return {"operator_controls": {"values": {
                "acquisition_experiment_enabled": False,
                "min_fee_ppm_saturated": 5,
            }}}
        if method == "revenue-config":
            return {"status": "success"}
        raise AssertionError((method, args))

    monkeypatch.setattr(runner, "cln_rpc", rpc)
    monkeypatch.setattr(runner, "acquisition_lanes", lambda _state: next(lane_calls))
    monkeypatch.setattr(runner, "_acquisition_rows", lambda _container: [{
        "id": 9, "state": "active", "channel_id": "a",
        "baseline_fee_ppm": 15, "target_fee_ppm": 0,
    }])

    state = runner.start_automatic_acquisition(
        replica=1, results_dir=tmp_path, attempts=1, poll_seconds=0,
    )

    assert state["status"] == "automatic_acquisition_ready"
    assert state["automatic_acquisition"]["lane"]["family"] == "cln"
    assert all(method != "revenue-fee-cycle" for method, _args in rpc_calls)


def test_retention_waits_for_restore_then_prices_lane_without_forced_cycle(
    monkeypatch, tmp_path
):
    runner = load_runner()
    path = runner.state_path(tmp_path, 2)
    episode = {
        "id": 11, "state": "active", "channel_id": "b",
        "baseline_fee_ppm": 15, "target_fee_ppm": 0,
    }
    lane = {"family": "lnd", "channel_id": "b", "peer_id": "pb", "fee_ppm": 0}
    runner.write_json_atomic(path, {
        "schema": runner.SCHEMA,
        "status": "smoke_complete",
        "events": [],
        "assignment": {"revenue_ops": "identity-b", "clboss": "identity-a"},
        "contenders": {
            "identity-a": {"container": "clboss"},
            "identity-b": {"container": "revenue"},
        },
        "automatic_acquisition": {"status": "active", "episode": episode, "lane": lane},
    })
    runtime = {"fee": 15}
    rpc_calls = []
    monkeypatch.setattr(runner, "_revenue_config_set", lambda *_args: {"status": "success"})
    monkeypatch.setattr(runner, "_acquisition_rows", lambda _container: [{
        **episode, "state": "completed", "restored_fee_ppm": 15,
    }])
    monkeypatch.setattr(runner, "acquisition_lanes", lambda _state: {
        "cln": {"family": "cln", "channel_id": "a", "peer_id": "pa", "fee_ppm": 15},
        "lnd": {**lane, "fee_ppm": runtime["fee"]},
    })

    def policy(_container, action, peer_id, *, fee_ppm=None):
        assert (action, peer_id) == ("set", "pb")
        runtime["fee"] = fee_ppm
        return {"status": "success"}

    monkeypatch.setattr(runner, "_policy_write", policy)
    monkeypatch.setattr(
        runner, "cln_rpc",
        lambda _container, method, *_args: rpc_calls.append(method) or {},
    )

    state = runner.start_retention_treatment(
        replica=2, results_dir=tmp_path, fee_ppm=1, attempts=1, poll_seconds=0,
    )

    assert state["status"] == "retention_ready"
    assert state["retention_treatment"]["fee_ppm"] == 1
    assert "revenue-fee-cycle" not in rpc_calls


def test_reconciled_traffic_never_retries_ambiguous_payment(monkeypatch):
    runner = load_runner()

    class Bridge:
        def __init__(self):
            self.calls = []

        def call(self, tool, arguments):
            self.calls.append((tool, arguments))
            if tool == "create_invoice":
                return {"invoice": "bolt11-redacted"}
            raise runner.PolarMcpError("post-dispatch UI failure")

    bridge = Bridge()
    monkeypatch.setattr(runner, "_invoice_hash", lambda _invoice: "payment-hash")
    monkeypatch.setattr(runner, "invoice_settled", lambda _sink, _hash: True)

    records = runner.run_reconciled_traffic(
        bridge,
        network_id=4,
        rounds=1,
        amount_sats=5_000,
        pause_seconds=0,
        lanes=(("cln-payer", "cln-sink"),),
    )

    assert [tool for tool, _arguments in bridge.calls] == ["create_invoice", "pay_invoice"]
    assert records[0]["payment"]["success"] is True
    assert records[0]["payment"]["reconciled_after_mcp_error"] is True


def test_reconciled_traffic_preserves_unknown_operation(monkeypatch):
    runner = load_runner()

    class Bridge:
        def call(self, tool, _arguments):
            if tool == "create_invoice":
                return {"invoice": "bolt11-redacted"}
            raise runner.PolarMcpError("post-dispatch UI failure")

    monkeypatch.setattr(runner, "_invoice_hash", lambda _invoice: "payment-hash")
    monkeypatch.setattr(runner, "invoice_settled", lambda _sink, _hash: False)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)

    with pytest.raises(runner.ReconciliationError) as caught:
        runner.run_reconciled_traffic(
            Bridge(),
            network_id=4,
            rounds=1,
            amount_sats=5_000,
            pause_seconds=0,
            lanes=(("lnd-payer", "lnd-sink"),),
        )
    assert caught.value.records == []
    assert caught.value.operation["outcome"] == "unknown_do_not_retry"
    assert caught.value.operation["payment_hash"] == "payment-hash"


def test_smoke_checkpoints_prior_schedule_progress_on_unknown_payment(monkeypatch, tmp_path):
    runner = load_runner()
    state_file = runner.state_path(tmp_path, 1)
    runner.write_json_atomic(
        state_file,
        {
            "schema": runner.SCHEMA,
            "status": "smoke_complete",
            "events": [],
            "assignment": {"revenue_ops": "identity-a", "clboss": "identity-b"},
            "contenders": {
                "identity-a": {"container": "polar-n4-clboss-r1-identity-a"},
                "identity-b": {"container": "polar-n4-clboss-r1-identity-b"},
            },
        },
    )
    totals = runner.Totals(0, 0, 0, 500_000, ())
    monkeypatch.setattr(runner, "contender_totals", lambda _container: totals)
    monkeypatch.setattr(
        runner,
        "traffic_schedule",
        lambda _rounds, _amount, _pattern: (
            ("cln", "forward", 5_000), ("cln", "reverse", 5_000)
        ),
    )
    monkeypatch.setattr(runner, "select_traffic_lanes", lambda direction, _family: ((direction, "sink"),))
    completed = {
        "round": 0,
        "payer": "forward",
        "sink": "sink",
        "amount_sats": 5_000,
        "payment": {"success": True},
    }
    calls = iter(
        (
            [completed],
            runner.ReconciliationError(
                "unknown", [], {"payment_hash": "hash", "outcome": "unknown_do_not_retry"}
            ),
        )
    )

    def traffic(*_args, **_kwargs):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(runner, "run_reconciled_traffic", traffic)

    with pytest.raises(runner.ReconciliationError):
        runner.run_smoke(
            object(), replica=1, results_dir=tmp_path, rounds=1, amount_sats=5_000,
            pause_seconds=0,
        )

    state = runner.read_state(state_file)
    result = state["events"][-1]["result"]
    assert state["status"] == "traffic_outcome_unknown"
    assert result["completed_count"] == 1
    assert result["partial_contenders"] == {
        "revenue_ops": {
            "forward_count": 0,
            "volume_msat": 0,
            "routing_fee_msat": 0,
            "mean_local_liquidity_sats": 500_000,
            "ending_min_local_balance_ppm": 0,
            "ending_max_local_balance_ppm": 0,
            "ending_worst_channel_imbalance_ppm": 0,
            "channels": {},
            "policy_changes": 0,
            "rebalance_cost_msat": 0,
        },
        "clboss": {
            "forward_count": 0,
            "volume_msat": 0,
            "routing_fee_msat": 0,
            "mean_local_liquidity_sats": 500_000,
            "ending_min_local_balance_ppm": 0,
            "ending_max_local_balance_ppm": 0,
            "ending_worst_channel_imbalance_ppm": 0,
            "channels": {},
            "policy_changes": 0,
            "rebalance_cost_msat": 0,
        },
    }
    progress = json.loads(Path(result["progress_file"]).read_text(encoding="utf-8"))
    assert progress["records"] == [completed]
    assert progress["records"][0] | {"payment": {}} == {
        "round": 0,
        "family": "cln",
        "direction": "forward",
        "payer": "forward",
        "sink": "sink",
        "amount_sats": 5_000,
        "payment": {},
    }
    assert progress["uncertain_operation"]["payment_hash"] == "hash"
    assert progress["uncertain_operation"]["family"] == "cln"
    assert progress["uncertain_operation"]["direction"] == "reverse"
    assert progress["uncertain_operation"]["round"] == 0
    assert progress["partial_contenders"] == result["partial_contenders"]


def test_completed_smoke_emits_strict_family_economics(monkeypatch, tmp_path):
    runner = load_runner()
    import polar_clboss_competition as competition

    state_file = runner.state_path(tmp_path, 1)
    revenue_scids = ("1x1x0", "1x1x1", "1x1x2", "1x1x3")
    clboss_scids = ("2x1x0", "2x1x1", "2x1x2", "2x1x3")
    roles = (
        ("cln", "payer"), ("cln", "sink"),
        ("lnd", "payer"), ("lnd", "sink"),
    )
    runner.write_json_atomic(state_file, {
        "schema": runner.SCHEMA,
        "status": "isolated_full_stack_ready",
        "league": "full_stack",
        "events": [],
        "assignment": {"revenue_ops": "identity-a", "clboss": "identity-b"},
        "contenders": {
            "identity-a": {"container": "revenue"},
            "identity-b": {"container": "clboss"},
        },
        "lane_map": {
            "identity-a": {
                scid: {"family": family, "side": side}
                for scid, (family, side) in zip(revenue_scids, roles)
            },
            "identity-b": {
                scid: {"family": family, "side": side}
                for scid, (family, side) in zip(clboss_scids, roles)
            },
        },
    })
    revenue_policy = tuple((scid, 0, 1) for scid in revenue_scids)
    clboss_policy = tuple((scid, 0, 1) for scid in clboss_scids)
    samples = {
        "revenue": iter((
            runner.Totals(0, 0, 0, 500_000, revenue_policy),
            runner.Totals(
            1, 10_000_000, 10, 500_000, revenue_policy,
            channel_metrics=(("1x1x0", 1, 10_000_000, 10),),
            ),
        )),
        "clboss": iter((
            runner.Totals(0, 0, 0, 500_000, clboss_policy),
            runner.Totals(
            1, 10_000_000, 10, 500_000, clboss_policy,
            channel_metrics=(("2x1x2", 1, 10_000_000, 10),),
            ),
        )),
    }
    monkeypatch.setattr(
        runner, "contender_totals", lambda container: next(samples[container])
    )
    monkeypatch.setattr(runner, "enforce_clboss_spend_cap", lambda _state: None)
    monkeypatch.setattr(
        runner, "run_reconciled_traffic",
        lambda *_args, **_kwargs: [{
            "round": 0, "payer": "payer", "sink": "sink",
            "amount_sats": 5_000, "payment_hash": "hash",
            "payment": {"success": True},
        }],
    )

    block = runner.run_smoke(
        object(), replica=1, results_dir=tmp_path, rounds=1,
        amount_sats=5_000, pause_seconds=0, traffic_pattern="forward-pressure",
    )

    assert block["cache_mode"] == "cold"
    assert block["traffic"]["fallback_settled"] == 0
    assert block["families"]["cln"]["contenders"]["revenue_ops"]["forward_count"] == 1
    assert block["families"]["lnd"]["contenders"]["clboss"]["forward_count"] == 1
    competition.validate_evidence({
        "schema": competition.SCHEMA_EVIDENCE,
        "assignments": [{
            "replica": "replica-1",
            "controllers": {"revenue_ops": "identity-a", "clboss": "identity-b"},
        }],
        "blocks": [block],
    })


def test_traffic_schedule_interleaves_directions_and_seeds_reserve_once():
    runner = load_runner()

    assert runner.traffic_schedule(2, 5_000) == (
        ("cln", "forward", 5_000 + runner.REVERSE_FEE_BUFFER_SATS),
        ("cln", "reverse", 5_000),
        ("lnd", "forward", 5_000 + runner.REVERSE_FEE_BUFFER_SATS),
        ("lnd", "reverse", 5_000),
        ("cln", "forward", 5_000),
        ("cln", "reverse", 5_000),
        ("lnd", "forward", 5_000),
        ("lnd", "reverse", 5_000),
    )


def test_forward_pressure_schedule_is_one_way_without_reserve_seed():
    runner = load_runner()

    assert runner.traffic_schedule(2, 10_000, "forward-pressure") == (
        ("cln", "forward", 10_000),
        ("lnd", "forward", 10_000),
        ("cln", "forward", 10_000),
        ("lnd", "forward", 10_000),
    )


def test_full_stack_offsets_revenue_setup_spend_and_pins_clboss_controls(
    monkeypatch, tmp_path
):
    runner = load_runner()
    path = runner.state_path(tmp_path, 2)
    runner.write_json_atomic(
        path,
        {
            "schema": runner.SCHEMA,
            "status": "isolated_fee_only_ready",
            "events": [],
            "assignment": {"revenue_ops": "identity-b", "clboss": "identity-a"},
            "contenders": {
                "identity-a": {"container": "polar-n4-clboss-r2-identity-a"},
                "identity-b": {"container": "polar-n4-clboss-r2-identity-b"},
            },
        },
    )
    calls = []
    expected = {
        "clboss-rebalance-mode": "xrebalance",
        "clboss-xrebalance-gain": "1",
        "clboss-xrebalance-grant": "0",
        "clboss-xrebalance-route-cost-floor": "auto",
    }

    def rpc(container, method, *args):
        calls.append((container, method, args))
        if method == "revenue-rebalance-debug":
            return {
                "capital_controls": {
                    "total_liquidity_breakdown": {
                        "actual_spent_by_category": {
                            "rebalance": 0, "open": 369, "close": 0, "boltz": 0,
                        }
                    }
                }
            }
        if method == "revenue-status":
            return {
                "operator_controls": {
                    "values": {"paused": False, "daily_budget_sats": 1_369}
                }
            }
        if method == "listconfigs":
            return {"configs": {key: {"value_str": value} for key, value in expected.items()}}
        return {}

    monkeypatch.setattr(runner, "cln_rpc", rpc)
    monkeypatch.setattr(runner, "rebalance_cost_msat", lambda _container: 0)

    state = runner.enable_full_stack(replica=2, results_dir=tmp_path, spend_cap_sats=1_000)

    assert state["status"] == "isolated_full_stack_ready"
    assert state["full_stack"]["revenue_runtime_budget_sats"] == 1_369
    assert any(
        method == "revenue-config" and args[-1] == "1369"
        for _container, method, args in calls
    )
    assert any(
        method == "setconfig" and args == ("clboss-rebalance-mode", "xrebalance")
        for _container, method, args in calls
    )


def test_state_round_trip_is_schema_checked(tmp_path):
    runner = load_runner()
    path = tmp_path / "state.json"
    payload = {"schema": runner.SCHEMA, "events": []}

    runner.write_json_atomic(path, payload)

    assert runner.read_state(path) == payload
    runner.write_json_atomic(path, {"schema": "unexpected"})
    with pytest.raises(runner.RunnerError, match="schema"):
        runner.read_state(path)


def test_launch_uses_official_entrypoint_contract(monkeypatch, tmp_path):
    runner = load_runner()
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return type(
            "Result", (), {"returncode": 1, "stdout": "", "stderr": "No such object: test"}
        )()

    monkeypatch.setattr(runner, "_run", fake_run)
    data_dir = tmp_path / "lightning"

    runner.launch_contender(
        name="polar-n4-clboss-r1-identity-a",
        identity="identity-a",
        data_dir=data_dir,
        image="test-image",
        network_id=4,
    )

    launch = commands[-1]
    assert "LIGHTNINGD_NETWORK=regtest" in launch
    assert "lightningd" not in launch
    assert "--network=regtest" not in launch
    assert (data_dir / "regtest").is_dir()


def test_cleanup_refuses_container_removal_while_channels_remain(monkeypatch, tmp_path):
    runner = load_runner()
    path = runner.state_path(tmp_path, 1)
    runner.write_json_atomic(
        path,
        {
            "schema": runner.SCHEMA,
            "network_id": 4,
            "assignment": {},
            "events": [],
            "contenders": {
                "identity-a": {"container": "polar-n4-clboss-r1-identity-a"},
            },
        },
    )
    removed = []
    monkeypatch.setattr(runner, "_stop_plugins", lambda _state: None)
    monkeypatch.setattr(runner, "docker_running", lambda _name: True)
    monkeypatch.setattr(runner, "_mine", lambda *_args: None)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    ticks = iter((0, 0, 61))
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        runner,
        "channel_rows",
        lambda _container: [{"short_channel_id": "1x1x1"}],
    )

    def fake_run(command, **_kwargs):
        if command[:3] == ["docker", "rm", "--force"]:
            removed.append(command[-1])
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(runner, "_run", fake_run)

    with pytest.raises(runner.RunnerError, match="active channels"):
        runner.cleanup(object(), replica=1, results_dir=tmp_path)
    assert removed == []


def test_competition_image_pins_all_source_revisions():
    dockerfile = (ROOT / "tools" / "polar-clboss" / "Dockerfile").read_text(encoding="utf-8")

    assert "elementsproject/lightningd:v26.06.6" in dockerfile
    assert "CLBOSS_COMMIT=8cb4e9215eba58b049375f234f5f073d0c7fc622" in dockerfile
    assert "XREBALANCE_COMMIT=fb70bf13cd9f3f79b14100bfdb8f2966884a4142" in dockerfile
    assert 'test "$(git -C /src/clboss rev-parse HEAD)" = "${CLBOSS_COMMIT}"' in dockerfile
    assert 'test "$(git -C /src/xrebalance rev-parse HEAD)" = "${XREBALANCE_COMMIT}"' in dockerfile
    assert "cargo build --release --locked" in dockerfile
