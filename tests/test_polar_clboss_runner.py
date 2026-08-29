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
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
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


def test_market_profiles_seed_explicit_realistic_and_acquisition_fees(monkeypatch):
    runner = load_runner()
    calls = []

    def rpc(container, method, *args):
        calls.append((container, method, args))
        return {"channels": [{}, {}, {}, {}]}

    monkeypatch.setattr(runner, "cln_rpc", rpc)
    contenders = {
        "identity-a": {"container": "a"},
        "identity-b": {"container": "b"},
    }
    runner.set_initial_fees(contenders, **runner.MARKET_PROFILES["realistic"])

    assert calls == [
        ("a", "setchannel", ("all", 500, 150)),
        ("b", "setchannel", ("all", 500, 150)),
    ]
    assert runner.MARKET_PROFILES["acquisition"] == {
        "fee_base_msat": 1,
        "fee_ppm": 10,
    }


def test_wait_gossip_policies_requires_exact_directional_crossed_readback(
    monkeypatch,
):
    runner = load_runner()
    calls = []
    rounds = {"a": 0, "b": 0}
    contenders = {
        "identity-a": {"container": "a"},
        "identity-b": {"container": "b"},
    }

    def rpc(container, method, *args):
        assert method == "listchannels"
        assert not args
        calls.append(container)
        rounds[container] += 1
        ppm = 10_000 if rounds[container] > 1 else 10
        return {"channels": [{
            "short_channel_id": "1x1x0",
            "source": "background-id",
            "fee_per_millionth": ppm,
        }]}

    monkeypatch.setattr(runner, "cln_rpc", rpc)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)

    result = runner.wait_gossip_policies(
        contenders,
        [{
            "short_channel_id": "1x1x0",
            "source": "background-id",
            "fee_ppm": 10_000,
        }],
        attempts=2,
        poll_seconds=0,
    )

    assert calls == ["a", "b", "a", "b"]
    assert all(
        row["1x1x0:background-id"] == 10_000 for row in result.values()
    )


def test_apply_background_reconnects_before_policy_and_verifies_gossip(monkeypatch):
    runner = load_runner()
    calls = []
    verified = []
    state = {
        "network_id": 4,
        "contenders": {
            "identity-a": {"container": "a"},
            "identity-b": {"container": "b"},
        },
        "background_policies": {
            "cln": {"revenue-node": [{
                "short_channel_id": "1x1x0",
                "source": "background-id",
                "peer_id": "sink-id",
                "peer_host": "lnd-sink",
                "fee_base_msat": 1,
                "fee_ppm": 10,
            }]},
            "lnd": [{
                "channel_point": "funding:0",
                "fee_base_msat": 1,
                "fee_ppm": 10,
                "time_lock_delta": 40,
                "min_htlc_msat": 1,
                "max_htlc_msat": 1_000_000,
            }],
        },
    }
    monkeypatch.setattr(
        runner, "_connect_cln",
        lambda *args: calls.append(("connect", *args)),
    )
    monkeypatch.setattr(
        runner, "cln_rpc",
        lambda *args: calls.append(("rpc", *args)) or {"channels": [{}]},
    )
    monkeypatch.setattr(
        runner, "_set_lnd_policy",
        lambda *args, **kwargs: calls.append(("lnd", *args, kwargs)),
    )
    monkeypatch.setattr(
        runner, "wait_gossip_policies",
        lambda contenders, expected: verified.append((contenders, expected)) or {
            "identity-a": {}, "identity-b": {}
        },
    )

    result = runner.apply_background_ppm(state, 10_000)

    assert calls[0] == (
        "connect", "polar-n4-revenue-node", "sink-id", "lnd-sink"
    )
    assert calls[1] == (
        "rpc", "polar-n4-revenue-node", "setchannel", "1x1x0", 1, 10_000
    )
    assert verified[0][1] == [{
        "short_channel_id": "1x1x0",
        "source": "background-id",
        "fee_ppm": 10_000,
    }]
    assert set(result) == {"identity-a", "identity-b"}


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
            "identity-a": {"container": "revenue", "node_id": "revenue-id"},
            "identity-b": {"container": "clboss", "node_id": "clboss-id"},
        },
        "automatic_acquisition": {
            "status": "restored",
            "episode": {"id": 8},
        },
    })
    lane_before = {
        "cln": {
            "family": "cln", "channel_id": "a", "short_channel_id": "1x1x0",
            "peer_id": "pa", "fee_ppm": 15,
        },
        "lnd": {
            "family": "lnd", "channel_id": "b", "short_channel_id": "2x1x0",
            "peer_id": "pb", "fee_ppm": 15,
        },
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
        "id": 9, "state": "active", "channel_id": "1x1x0",
        "baseline_fee_ppm": 15, "target_fee_ppm": 0,
    }])

    state = runner.start_automatic_acquisition(
        replica=1, results_dir=tmp_path, attempts=1, poll_seconds=0,
    )

    assert state["status"] == "automatic_acquisition_ready"
    assert state["automatic_acquisition"]["lane"]["family"] == "cln"
    assert state["automatic_acquisition_history"] == [{
        "status": "restored", "episode": {"id": 8},
    }]
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


def test_refresh_automatic_phase_recognizes_native_paid_retention(monkeypatch):
    runner = load_runner()
    state = {
        "assignment": {"revenue_ops": "identity-a", "clboss": "identity-b"},
        "contenders": {
            "identity-a": {"container": "revenue"},
            "identity-b": {"container": "clboss"},
        },
        "automatic_acquisition": {
            "status": "active",
            "episode": {"id": 7, "state": "active", "channel_id": "1x1x0"},
            "lane": {"channel_id": "funding-a", "short_channel_id": "1x1x0"},
        },
    }
    live_episode = {
        "id": 7,
        "state": "active",
        "channel_id": "1x1x0",
        "phase": "retention",
        "target_fee_ppm": 0,
        "target_base_fee_msat": 0,
        "retention_fee_ppm": 0,
        "retention_base_fee_msat": 4,
    }
    live_lane = {
        "family": "cln",
        "channel_id": "funding-a",
        "short_channel_id": "1x1x0",
        "peer_id": "peer-a",
        "fee_base_msat": 4,
        "fee_ppm": 0,
    }
    monkeypatch.setattr(runner, "_acquisition_rows", lambda _container: [live_episode])
    monkeypatch.setattr(
        runner,
        "acquisition_lanes",
        lambda _state: {
            "cln": live_lane,
            "lnd": {
                "family": "lnd",
                "channel_id": "funding-b",
                "short_channel_id": "2x1x0",
                "peer_id": "peer-b",
                "fee_base_msat": 0,
                "fee_ppm": 15,
            },
        },
    )

    assert runner.refresh_automatic_acquisition_phase(state) == "retention"
    assert state["automatic_acquisition"]["phase"] == "retention"
    assert state["automatic_acquisition"]["episode"] == live_episode
    assert state["automatic_acquisition"]["lane"] == live_lane


def test_refresh_automatic_phase_fails_closed_after_native_episode_exit(monkeypatch):
    runner = load_runner()
    state = {
        "assignment": {"revenue_ops": "identity-a", "clboss": "identity-b"},
        "contenders": {
            "identity-a": {"container": "revenue"},
            "identity-b": {"container": "clboss"},
        },
        "automatic_acquisition": {
            "status": "active",
            "episode": {"id": 7, "state": "active", "channel_id": "1x1x0"},
        },
    }
    monkeypatch.setattr(
        runner,
        "_acquisition_rows",
        lambda _container: [{"id": 7, "state": "completed"}],
    )

    with pytest.raises(runner.RunnerError, match="no longer exactly one active"):
        runner.refresh_automatic_acquisition_phase(state)


def test_refresh_automatic_phase_reconciles_native_channel_rollover(monkeypatch):
    runner = load_runner()
    state = {
        "assignment": {"revenue_ops": "identity-a", "clboss": "identity-b"},
        "contenders": {
            "identity-a": {"container": "revenue"},
            "identity-b": {"container": "clboss"},
        },
        "automatic_acquisition": {
            "status": "active",
            "episode": {"id": 7, "state": "active", "channel_id": "1x1x0"},
            "lane": {"channel_id": "funding-a", "short_channel_id": "1x1x0"},
        },
    }
    completed = {
        "id": 7,
        "state": "completed",
        "channel_id": "1x1x0",
        "exit_reason": "retention_volume_cap",
        "baseline_fee_ppm": 15,
        "baseline_base_fee_msat": 0,
        "restored_fee_ppm": 15,
        "restored_base_fee_msat": 0,
    }
    active = {
        "id": 8,
        "state": "active",
        "channel_id": "2x1x0",
        "phase": "acquisition",
        "target_fee_ppm": 0,
        "target_base_fee_msat": 0,
    }
    restored_lane = {
        "family": "cln", "channel_id": "funding-a", "short_channel_id": "1x1x0",
        "peer_id": "peer-a", "fee_base_msat": 0, "fee_ppm": 15,
    }
    next_lane = {
        "family": "lnd", "channel_id": "funding-b", "short_channel_id": "2x1x0",
        "peer_id": "peer-b", "fee_base_msat": 0, "fee_ppm": 0,
    }
    monkeypatch.setattr(
        runner, "_acquisition_rows", lambda _container: [active, completed]
    )
    monkeypatch.setattr(
        runner, "acquisition_lanes",
        lambda _state: {"cln": restored_lane, "lnd": next_lane},
    )

    assert runner.refresh_automatic_acquisition_phase(state) == "acquisition"
    automatic = state["automatic_acquisition"]
    assert automatic["episode"] == active
    assert automatic["lane"] == next_lane
    assert automatic["rollovers"] == [{
        "completed_episode": completed,
        "restored_lane": restored_lane,
        "restoration_evidence": {
            "baseline_fee_ppm": 15,
            "baseline_base_fee_msat": 0,
            "restored_fee_ppm": 15,
            "restored_base_fee_msat": 0,
            "current_fee_ppm": 15,
            "current_base_fee_msat": 0,
        },
        "next_episode_id": 8,
    }]


def test_refresh_automatic_rollover_allows_later_ordinary_repricing(monkeypatch):
    runner = load_runner()
    state = {
        "assignment": {"revenue_ops": "identity-a", "clboss": "identity-b"},
        "contenders": {
            "identity-a": {"container": "revenue"},
            "identity-b": {"container": "clboss"},
        },
        "automatic_acquisition": {
            "status": "active",
            "episode": {"id": 7, "state": "active", "channel_id": "1x1x0"},
            "lane": {"channel_id": "funding-a", "short_channel_id": "1x1x0"},
        },
    }
    completed = {
        "id": 7, "state": "completed", "channel_id": "1x1x0",
        "baseline_fee_ppm": 150, "baseline_base_fee_msat": 500,
        "restored_fee_ppm": 150, "restored_base_fee_msat": 500,
    }
    active = {
        "id": 8, "state": "active", "channel_id": "2x1x0",
        "phase": "acquisition", "target_fee_ppm": 0,
        "target_base_fee_msat": 0,
    }
    monkeypatch.setattr(
        runner, "_acquisition_rows", lambda _container: [active, completed]
    )
    monkeypatch.setattr(runner, "acquisition_lanes", lambda _state: {
        "cln": {
            "family": "cln", "channel_id": "funding-a", "short_channel_id": "1x1x0",
            "peer_id": "peer-a", "fee_base_msat": 400, "fee_ppm": 125,
        },
        "lnd": {
            "family": "lnd", "channel_id": "funding-b", "short_channel_id": "2x1x0",
            "peer_id": "peer-b", "fee_base_msat": 0, "fee_ppm": 0,
        },
    })

    assert runner.refresh_automatic_acquisition_phase(state) == "acquisition"
    evidence = state["automatic_acquisition"]["rollovers"][0][
        "restoration_evidence"
    ]
    assert evidence == {
        "baseline_fee_ppm": 150,
        "baseline_base_fee_msat": 500,
        "restored_fee_ppm": 150,
        "restored_base_fee_msat": 500,
        "current_fee_ppm": 125,
        "current_base_fee_msat": 400,
    }


def test_refresh_automatic_rollover_fails_closed_on_restore_mismatch(monkeypatch):
    runner = load_runner()
    state = {
        "assignment": {"revenue_ops": "identity-a", "clboss": "identity-b"},
        "contenders": {
            "identity-a": {"container": "revenue"},
            "identity-b": {"container": "clboss"},
        },
        "automatic_acquisition": {
            "status": "active",
            "episode": {"id": 7, "state": "active", "channel_id": "1x1x0"},
            "lane": {"channel_id": "funding-a", "short_channel_id": "1x1x0"},
        },
    }
    monkeypatch.setattr(runner, "_acquisition_rows", lambda _container: [
        {
            "id": 7, "state": "completed", "channel_id": "1x1x0",
            "baseline_fee_ppm": 15, "baseline_base_fee_msat": 0,
            "restored_fee_ppm": 14, "restored_base_fee_msat": 0,
        },
        {
            "id": 8, "state": "active", "channel_id": "2x1x0",
            "phase": "acquisition", "target_fee_ppm": 0,
            "target_base_fee_msat": 0,
        },
    ])
    monkeypatch.setattr(runner, "acquisition_lanes", lambda _state: {
        "cln": {
            "family": "cln", "channel_id": "funding-a", "short_channel_id": "1x1x0",
            "peer_id": "peer-a", "fee_base_msat": 0, "fee_ppm": 0,
        },
        "lnd": {
            "family": "lnd", "channel_id": "funding-b", "short_channel_id": "2x1x0",
            "peer_id": "peer-b", "fee_base_msat": 0, "fee_ppm": 0,
        },
    })

    with pytest.raises(runner.RunnerError, match="did not restore"):
        runner.refresh_automatic_acquisition_phase(state)


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


def test_reconciled_traffic_records_authoritative_terminal_failure_without_retry(
    monkeypatch,
):
    runner = load_runner()

    class Bridge:
        def __init__(self):
            self.calls = []

        def call(self, tool, arguments):
            self.calls.append((tool, arguments))
            if tool == "create_invoice":
                return {"invoice": "bolt11-redacted"}
            raise runner.PolarMcpError("post-dispatch timeout")

    bridge = Bridge()
    monkeypatch.setattr(runner, "_invoice_hash", lambda _invoice: "payment-hash")
    monkeypatch.setattr(
        runner, "payment_outcome", lambda _payer, _sink, _hash: "failed"
    )

    records = runner.run_reconciled_traffic(
        bridge,
        network_id=4,
        rounds=1,
        amount_sats=5_000,
        pause_seconds=0,
        lanes=(("cln-payer", "cln-sink"),),
        reconciliation_attempts=1,
        reconciliation_poll_seconds=0,
    )

    assert [tool for tool, _arguments in bridge.calls] == [
        "create_invoice", "pay_invoice"
    ]
    assert records[0]["payment"] == {
        "success": False,
        "reconciled_terminal_failure": True,
        "bridge_error": "post-dispatch timeout",
    }


@pytest.mark.parametrize(
    ("payer", "payload", "expected"),
    [
        ("cln-payer", {"pays": [{"payment_hash": "hash", "status": "complete"}]}, "settled"),
        ("cln-payer", {"pays": [{"payment_hash": "hash", "status": "pending"}]}, "pending"),
        ("cln-payer", {"pays": [{"payment_hash": "hash", "status": "failed"}]}, "failed"),
        ("cln-payer", {"pays": "malformed"}, "unknown"),
        ("lnd-payer", {"payments": [{"payment_hash": "HASH", "status": "SUCCEEDED"}]}, "settled"),
        ("lnd-payer", {"payments": [{"payment_hash": "hash", "status": "IN_FLIGHT"}]}, "pending"),
        ("lnd-payer", {"payments": [{"payment_hash": "hash", "status": "FAILED"}]}, "failed"),
        ("lnd-payer", {"payments": None}, "unknown"),
    ],
)
def test_payer_payment_outcome_is_aggregate_and_malformed_safe(
    monkeypatch, payer, payload, expected,
):
    runner = load_runner()
    monkeypatch.setattr(runner, "cln_rpc", lambda *_args: payload)
    monkeypatch.setattr(runner, "lnd_rpc", lambda *_args: payload)

    assert runner.payer_payment_outcome(payer, "hash") == expected


def test_payment_outcome_treats_rpc_errors_as_unknown(monkeypatch):
    runner = load_runner()

    def broken(*_args):
        raise runner.RunnerError("read failed")

    monkeypatch.setattr(runner, "invoice_settled", broken)
    monkeypatch.setattr(runner, "cln_rpc", broken)

    assert runner.payment_outcome("cln-payer", "cln-sink", "hash") == "unknown"


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        ([{"payment": {"success": True}}], False),
        ([{"payment": {"success": False}}], True),
        ([{"payment": "malformed"}], True),
        (["malformed"], True),
    ],
)
def test_terminal_payment_failure_stops_depleted_block(records, expected):
    runner = load_runner()

    assert runner.has_terminal_payment_failure(records) is expected


def test_reconciled_traffic_preserves_unknown_operation(monkeypatch):
    runner = load_runner()

    class Bridge:
        def call(self, tool, _arguments):
            if tool == "create_invoice":
                return {"invoice": "bolt11-redacted"}
            raise runner.PolarMcpError("post-dispatch UI failure")

    monkeypatch.setattr(runner, "_invoice_hash", lambda _invoice: "payment-hash")
    monkeypatch.setattr(
        runner, "payment_outcome", lambda _payer, _sink, _hash: "pending"
    )
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)

    with pytest.raises(runner.ReconciliationError) as caught:
        runner.run_reconciled_traffic(
            Bridge(),
            network_id=4,
            rounds=1,
            amount_sats=5_000,
            pause_seconds=0,
            lanes=(("lnd-payer", "lnd-sink"),),
            reconciliation_attempts=2,
            reconciliation_poll_seconds=0,
        )
    assert caught.value.records == []
    assert caught.value.operation["outcome"] == "unknown_do_not_retry"
    assert caught.value.operation["last_observed_outcome"] == "pending"
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
        lambda _rounds, _amount, _pattern, _amount_profile, _traffic_family: (
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


def test_completed_smoke_accepts_multipart_htlcs_at_exact_traffic_volume(
    monkeypatch, tmp_path
):
    runner = load_runner()
    path = runner.state_path(tmp_path, 93)
    roles = (("cln", "payer"), ("cln", "sink"), ("lnd", "payer"), ("lnd", "sink"))
    runner.write_json_atomic(path, {
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
            identity: {
                f"{prefix}x1x{index}": {"family": family, "side": side}
                for index, (family, side) in enumerate(roles)
            }
            for identity, prefix in (("identity-a", 1), ("identity-b", 2))
        },
    })
    revenue_policy = tuple((f"1x1x{i}", 0, 150) for i in range(4))
    clboss_policy = tuple((f"2x1x{i}", 0, 1) for i in range(4))
    samples = {
        "revenue": iter((
            runner.Totals(0, 0, 0, 500_000, revenue_policy),
            runner.Totals(
                2, 50_000_000, 7_500, 500_000, revenue_policy,
                channel_metrics=(
                    ("1x1x3", 2, 50_000_000, 7_500),
                ),
            ),
        )),
        "clboss": iter((
            runner.Totals(0, 0, 0, 500_000, clboss_policy),
            runner.Totals(0, 0, 0, 500_000, clboss_policy),
        )),
    }
    monkeypatch.setattr(
        runner, "contender_totals", lambda container: next(samples[container])
    )
    monkeypatch.setattr(
        runner, "run_reconciled_traffic", lambda *_args, **_kwargs: [{
            "payer": "payer", "sink": "sink", "amount_sats": 50_000,
            "payment_hash": "hash", "payment": {"success": True},
        }]
    )

    block = runner.run_smoke(
        object(), replica=93, results_dir=tmp_path, rounds=1,
        amount_sats=50_000, pause_seconds=0,
        traffic_pattern="forward-pressure", traffic_family="lnd",
    )

    assert block["safety_violations"] == []
    assert block["traffic"] == {
        **block["traffic"],
        "fallback_settled": 0,
        "settled_volume_msat": 50_000_000,
        "contender_volume_msat": 50_000_000,
        "fallback_volume_msat": 0,
        "extra_volume_msat": 0,
        "contender_forward_count": 2,
        "multipart_forward_splits": 1,
        "attribution_method": "exact_family_volume",
    }


@pytest.mark.parametrize(
    ("contender_volume_msat", "traffic_field", "traffic_value", "violation"),
    (
        (
            60_000_000, "extra_volume_msat", 10_000_000,
            "unattributed_extra_contender_volume",
        ),
        (40_000_000, "fallback_volume_msat", 10_000_000, "fallback_settled"),
    ),
)
def test_completed_smoke_rejects_mismatched_contender_volume(
    monkeypatch, tmp_path, contender_volume_msat, traffic_field, traffic_value,
    violation,
):
    runner = load_runner()
    path = runner.state_path(tmp_path, 94)
    roles = (("cln", "payer"), ("cln", "sink"), ("lnd", "payer"), ("lnd", "sink"))
    runner.write_json_atomic(path, {
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
            identity: {
                f"{prefix}x1x{index}": {"family": family, "side": side}
                for index, (family, side) in enumerate(roles)
            }
            for identity, prefix in (("identity-a", 1), ("identity-b", 2))
        },
    })
    policies = {
        "revenue": tuple((f"1x1x{i}", 0, 150) for i in range(4)),
        "clboss": tuple((f"2x1x{i}", 0, 1) for i in range(4)),
    }
    samples = {
        "revenue": iter((
            runner.Totals(0, 0, 0, 500_000, policies["revenue"]),
            runner.Totals(
                2, contender_volume_msat, 8_000, 500_000, policies["revenue"],
                channel_metrics=(("1x1x3", 2, contender_volume_msat, 8_000),),
            ),
        )),
        "clboss": iter((
            runner.Totals(0, 0, 0, 500_000, policies["clboss"]),
            runner.Totals(0, 0, 0, 500_000, policies["clboss"]),
        )),
    }
    monkeypatch.setattr(
        runner, "contender_totals", lambda container: next(samples[container])
    )
    monkeypatch.setattr(
        runner, "run_reconciled_traffic", lambda *_args, **_kwargs: [{
            "payer": "payer", "sink": "sink", "amount_sats": 50_000,
            "payment_hash": "hash", "payment": {"success": True},
        }]
    )

    block = runner.run_smoke(
        object(), replica=94, results_dir=tmp_path, rounds=1,
        amount_sats=50_000, pause_seconds=0,
        traffic_pattern="forward-pressure", traffic_family="lnd",
    )

    assert block["traffic"][traffic_field] == traffic_value
    assert block["safety_violations"] == [violation]


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


def test_realistic_amount_profile_cycles_market_sized_payments():
    runner = load_runner()

    schedule = runner.traffic_schedule(
        4, 999, "forward-pressure", amount_profile="realistic"
    )

    assert schedule == (
        ("cln", "forward", 5_000),
        ("lnd", "forward", 5_000),
        ("cln", "forward", 15_000),
        ("lnd", "forward", 15_000),
        ("cln", "forward", 35_000),
        ("lnd", "forward", 35_000),
        ("cln", "forward", 100_000),
        ("lnd", "forward", 100_000),
    )


def test_family_scoped_schedule_excludes_unrelated_client_family():
    runner = load_runner()

    assert runner.traffic_schedule(
        2, 5_000, traffic_family="cln"
    ) == (
        ("cln", "forward", 5_000 + runner.REVERSE_FEE_BUFFER_SATS),
        ("cln", "reverse", 5_000),
        ("cln", "forward", 5_000),
        ("cln", "reverse", 5_000),
    )


def test_automatic_treatment_restore_is_idempotent():
    runner = load_runner()
    state = {
        "automatic_acquisition": {"status": "restored"},
        "retention_treatment": {"status": "restored"},
    }

    assert runner.restore_automatic_treatments(state) is False


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
        "clboss-xrebalance-per-hour": str(runner.CLBOSS_REBALANCES_PER_HOUR),
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
    assert state["full_stack"]["revenue_rebalance_allowance_sats"] == 1_000
    assert state["full_stack"]["clboss_spend_policy"] == "native_unbounded"
    assert any(
        method == "revenue-config" and args[-1] == "1369"
        for _container, method, args in calls
    )
    assert any(
        method == "setconfig" and args == ("clboss-rebalance-mode", "xrebalance")
        for _container, method, args in calls
    )


def test_accelerate_restarts_revenue_and_leaves_clboss_spending_unbounded(
    monkeypatch, tmp_path
):
    runner = load_runner()
    path = runner.state_path(tmp_path, 7)
    runner.write_json_atomic(path, {
        "schema": runner.SCHEMA,
        "network_id": 4,
        "status": "return_paths_ready",
        "events": [],
        "assignment": {"revenue_ops": "identity-a", "clboss": "identity-b"},
        "contenders": {
            "identity-a": {"container": "revenue"},
            "identity-b": {"container": "clboss"},
        },
        "full_stack": {
            "revenue_runtime_budget_sats": 1_310,
            "spend_cap_sats_per_controller": 1_000,
        },
    })
    calls = []

    def rpc(container, method, *args):
        calls.append((container, method, args))
        if method == "listconfigs":
            if container == "revenue":
                return {"configs": {
                    key: {"value_str": str(runner.TOURNAMENT_CYCLE_SECONDS)}
                    for key in (
                        "revenue-ops-flow-interval", "revenue-ops-fee-interval",
                        "revenue-ops-rebalance-interval",
                    )
                }}
            return {"configs": {
                "clboss-xrebalance-per-hour": {
                    "value_str": str(runner.CLBOSS_REBALANCES_PER_HOUR)
                }
            }}
        if method == "revenue-status":
            return {"operator_controls": {"values": {
                "paused": False, "daily_budget_sats": 1_310,
            }}}
        return {}

    monkeypatch.setattr(runner, "cln_rpc", rpc)
    monkeypatch.setattr(
        runner, "_run",
        lambda *_args, **_kwargs: type(
            "Result", (), {"returncode": 0, "stdout": "", "stderr": ""}
        )(),
    )

    state = runner.accelerate_controllers(replica=7, results_dir=tmp_path)

    assert state["full_stack"]["clboss_spend_policy"] == "native_unbounded"
    assert state["full_stack"]["revenue_rebalance_allowance_sats"] == 1_000
    assert "spend_cap_sats_per_controller" not in state["full_stack"]
    assert state["full_stack"]["cadence"] == {
        "revenue_seconds": runner.TOURNAMENT_CYCLE_SECONDS,
        "clboss_rebalances_per_hour": runner.CLBOSS_REBALANCES_PER_HOUR,
    }
    start = next(args for container, method, args in calls
                 if container == "revenue" and method == "-k")
    assert "revenue-ops-daily-budget-sats=1310" in start
    assert all(
        method != "revenue-rebalance-cycle" for _container, method, _args in calls
    )
    assert any(
        method == "revenue-config" and args == ("set", "paused", "false")
        for _container, method, args in calls
    )
    assert not any(
        method == "setconfig" and args == ("clboss-rebalance-mode", "off")
        for _container, method, args in calls
    )


def test_controlled_depletion_holds_both_controllers_and_routes_equal_pressure(
    monkeypatch, tmp_path
):
    runner = load_runner()
    path = runner.state_path(tmp_path, 9)
    runner.write_json_atomic(path, {
        "schema": runner.SCHEMA,
        "network_id": 4,
        "status": "isolated_full_stack_ready",
        "league": "full_stack",
        "events": [],
        "assignment": {"revenue_ops": "identity-a", "clboss": "identity-b"},
        "contenders": {
            "identity-a": {"container": "revenue", "node_id": "revenue-id"},
            "identity-b": {"container": "clboss", "node_id": "clboss-id"},
        },
        "lane_map": {},
    })
    snapshots = iter((
        {
            "revenue_ops": {
                "source": {"short_channel_id": "1x1x0", "local_balance_ppm": 500_000},
                "depleted": {"short_channel_id": "2x1x0", "local_balance_ppm": 500_000},
            },
            "clboss": {
                "source": {"short_channel_id": "3x1x0", "local_balance_ppm": 500_000},
                "depleted": {"short_channel_id": "4x1x0", "local_balance_ppm": 500_000},
            },
        },
        {
            "revenue_ops": {
                "source": {"short_channel_id": "1x1x0", "local_balance_ppm": 750_000},
                "depleted": {"short_channel_id": "2x1x0", "local_balance_ppm": 250_000},
            },
            "clboss": {
                "source": {"short_channel_id": "3x1x0", "local_balance_ppm": 750_000},
                "depleted": {"short_channel_id": "4x1x0", "local_balance_ppm": 250_000},
            },
        },
    ))
    routed = []
    calls = []
    gossip = []

    def rpc(container, method, *args):
        calls.append((container, method, args))
        if method == "revenue-status":
            return {"operator_controls": {"values": {"paused": True}}}
        if method == "clboss-status":
            return {"rebalance_mode": {"mode": "off"}}
        if method == "setchannel":
            return {"channels": [{}]}
        return {}

    monkeypatch.setattr(runner, "cln_rpc", rpc)
    monkeypatch.setattr(
        runner, "wait_gossip_policies",
        lambda contenders, expected: gossip.append((contenders, expected)) or {},
    )
    monkeypatch.setattr(
        runner, "controlled_depletion_snapshot",
        lambda _state, *, family: next(snapshots)
    )
    monkeypatch.setattr(
        runner,
        "_directed_cln_fixture_payment",
        lambda _bridge, **kwargs: routed.append(kwargs) or {
            "controller": kwargs["label"], "amount_sats": kwargs["amount_sats"]
        },
    )

    state = runner.prepare_controlled_depletion(
        object(), replica=9, results_dir=tmp_path, amount_sats=750_000,
        fixture_fee_ppm=120,
    )

    assert state["status"] == "controlled_depletion_ready"
    assert state["controlled_depletion"]["controllers_held"] is True
    assert state["controlled_depletion"]["family"] == "cln"
    assert [row["label"] for row in routed] == ["revenue_ops", "clboss"]
    assert [row["target_scid"] for row in routed] == ["1x1x0", "3x1x0"]
    assert all(row["amount_sats"] == 750_000 for row in routed)
    assert state["controlled_depletion"]["fixture_fee_ppm"] == 120
    assert gossip[0][1] == [
        {"short_channel_id": "2x1x0", "source": "revenue-id", "fee_ppm": 120},
        {"short_channel_id": "4x1x0", "source": "clboss-id", "fee_ppm": 120},
    ]
    assert {
        (container, args)
        for container, method, args in calls
        if method == "setchannel"
    } == {
        ("revenue", ("2x1x0", 0, 120)),
        ("clboss", ("4x1x0", 0, 120)),
    }
    assert any(
        method == "setconfig" and args == ("clboss-rebalance-mode", "off")
        for _container, method, args in calls
    )
    assert all(method != "revenue-rebalance-cycle" for _container, method, _args in calls)


def test_controlled_lnd_depletion_uses_exact_lnd_first_hops(monkeypatch, tmp_path):
    runner = load_runner()
    path = runner.state_path(tmp_path, 10)
    runner.write_json_atomic(path, {
        "schema": runner.SCHEMA,
        "network_id": 4,
        "status": "isolated_full_stack_ready",
        "league": "full_stack",
        "events": [],
        "assignment": {"revenue_ops": "identity-a", "clboss": "identity-b"},
        "contenders": {
            "identity-a": {"container": "revenue"},
            "identity-b": {"container": "clboss"},
        },
        "lane_map": {},
    })
    snapshots = iter((
        {
            "revenue_ops": {
                "source": {"short_channel_id": "11x1x0", "local_balance_ppm": 500_000},
                "depleted": {"short_channel_id": "12x1x0", "local_balance_ppm": 500_000},
            },
            "clboss": {
                "source": {"short_channel_id": "13x1x0", "local_balance_ppm": 500_000},
                "depleted": {"short_channel_id": "14x1x0", "local_balance_ppm": 500_000},
            },
        },
        {
            "revenue_ops": {
                "source": {"short_channel_id": "11x1x0", "local_balance_ppm": 750_000},
                "depleted": {"short_channel_id": "12x1x0", "local_balance_ppm": 250_000},
            },
            "clboss": {
                "source": {"short_channel_id": "13x1x0", "local_balance_ppm": 750_000},
                "depleted": {"short_channel_id": "14x1x0", "local_balance_ppm": 250_000},
            },
        },
    ))
    snapshot_families = []
    routed = []

    def rpc(_container, method, *_args):
        if method == "revenue-status":
            return {"operator_controls": {"values": {"paused": True}}}
        if method == "clboss-status":
            return {"rebalance_mode": {"mode": "off"}}
        return {}

    monkeypatch.setattr(runner, "cln_rpc", rpc)
    monkeypatch.setattr(
        runner,
        "controlled_depletion_snapshot",
        lambda _state, *, family: (
            snapshot_families.append(family) or next(snapshots)
        ),
    )
    monkeypatch.setattr(
        runner,
        "_directed_lnd_fixture_payment",
        lambda _bridge, **kwargs: routed.append(kwargs) or {
            "controller": kwargs["label"], "amount_sats": kwargs["amount_sats"]
        },
    )

    state = runner.prepare_controlled_depletion(
        object(), replica=10, results_dir=tmp_path, amount_sats=750_000,
        family="lnd",
    )

    assert snapshot_families == ["lnd", "lnd"]
    assert state["controlled_depletion"]["family"] == "lnd"
    assert [row["target_scid"] for row in routed] == ["11x1x0", "13x1x0"]
    assert all(row["amount_sats"] == 750_000 for row in routed)


def test_directed_lnd_fixture_payment_pins_first_and_last_hop(monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner, "_live_lnd_channels", lambda _container: [{
        "active": True,
        "scid_str": "21x1x0",
        "chan_id": "funding-hash",
        "scid": "2308974418337792",
        "remote_pubkey": "02contender",
    }])
    calls = []

    def rpc(container, *args):
        calls.append((container, args))
        return {"status": "SUCCEEDED", "payment_hash": "abcd"}

    class Bridge:
        def call(self, method, params):
            assert method == "create_invoice"
            assert params["nodeName"] == "lnd-sink"
            assert params["amount"] == 750_000
            return {"invoice": "lnbcrt-invoice"}

    monkeypatch.setattr(runner, "lnd_rpc", rpc)

    result = runner._directed_lnd_fixture_payment(
        Bridge(), network_id=4, target_scid="21x1x0",
        amount_sats=750_000, label="revenue_ops",
    )

    assert result["payment_hash"] == "abcd"
    assert calls == [("polar-n4-lnd-payer", (
        "payinvoice", "--force", "--json", "--fee_limit=1000",
        "--timeout=30s",
        "--outgoing_chan_id=2308974418337792",
        "--last_hop=02contender", "lnbcrt-invoice",
    ))]


def test_directed_lnd_fixture_payment_rejects_malformed_channel(monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner, "_live_lnd_channels", lambda _container: [{
        "active": True,
        "scid_str": "21x1x0",
        "chan_id": "funding-hash",
        "scid": None,
        "remote_pubkey": "02contender",
    }])

    with pytest.raises(runner.RunnerError, match="numeric routing id"):
        runner._directed_lnd_fixture_payment(
            object(), network_id=4, target_scid="21x1x0",
            amount_sats=750_000, label="revenue_ops",
        )


def test_controlled_depletion_resume_uses_native_cycles_only(monkeypatch, tmp_path):
    runner = load_runner()
    path = tmp_path / "state.json"
    state = {
        "events": [],
        "assignment": {"revenue_ops": "identity-a", "clboss": "identity-b"},
        "contenders": {
            "identity-a": {"container": "revenue"},
            "identity-b": {"container": "clboss"},
        },
        "controlled_depletion": {"controllers_held": True},
    }
    calls = []

    def rpc(container, method, *args):
        calls.append((container, method, args))
        if method == "revenue-status":
            return {"operator_controls": {"values": {"paused": False}}}
        if method == "clboss-status":
            return {"rebalance_mode": {"mode": "xrebalance"}}
        return {}

    monkeypatch.setattr(runner, "cln_rpc", rpc)
    activation = runner.resume_controlled_depletion_controllers(path, state)

    assert activation["forced_cycles"] is False
    assert state["controlled_depletion"]["controllers_held"] is False
    assert all(method != "revenue-rebalance-cycle" for _container, method, _args in calls)
    assert any(
        method == "setconfig" and args == ("clboss-rebalance-mode", "xrebalance")
        for _container, method, args in calls
    )


def test_retired_return_paths_archive_for_one_warm_epoch(monkeypatch, tmp_path):
    runner = load_runner()
    path = tmp_path / "state.json"
    old_paths = [
        {"family": "cln", "short_channel_id": "1x1x0"},
        {"family": "lnd", "channel_point": "funding:1"},
    ]
    state = {
        "status": "smoke_complete",
        "events": [],
        "return_paths": old_paths,
        "return_path_closes_dispatched": True,
        "return_paths_retired": {
            "confirmed_absent": True,
            "observation_block": "rebalance-1",
        },
        "return_path_fixture": {"fee_ppm": 1},
    }
    runner.write_json_atomic(path, state)
    monkeypatch.setattr(runner, "_live_return_paths", lambda _state: {})

    assert runner.prepare_return_path_renewal(path, state) is True

    assert state["return_paths"] == []
    assert state["return_path_closes_dispatched"] is False
    assert "return_paths_retired" not in state
    assert "return_path_fixture" not in state
    assert state["return_path_history"] == [{
        "paths": old_paths,
        "retired": {
            "confirmed_absent": True,
            "observation_block": "rebalance-1",
        },
        "fixture": {"fee_ppm": 1},
    }]
    assert state["events"][-1]["event"] == "return_paths_archived_for_warm_epoch"


def test_return_path_renewal_fails_closed_if_retired_path_is_live(
    monkeypatch, tmp_path
):
    runner = load_runner()
    path = tmp_path / "state.json"
    state = {
        "status": "smoke_complete",
        "events": [],
        "return_paths": [{"family": "lnd", "channel_point": "funding:1"}],
        "return_path_closes_dispatched": True,
        "return_paths_retired": {"confirmed_absent": True},
    }
    monkeypatch.setattr(
        runner, "_live_return_paths", lambda _state: {"lnd": ["funding:1"]}
    )

    with pytest.raises(runner.RunnerError, match="still live"):
        runner.prepare_return_path_renewal(path, state)

    assert state["return_path_closes_dispatched"] is True
    assert state["return_paths"] == [
        {"family": "lnd", "channel_point": "funding:1"}
    ]


def test_post_demand_hold_resumes_once_without_forced_cycles(monkeypatch, tmp_path):
    runner = load_runner()
    path = tmp_path / "state.json"
    state = {
        "events": [{"at": 20, "event": "smoke_complete"}],
        "assignment": {"revenue_ops": "identity-a", "clboss": "identity-b"},
        "contenders": {
            "identity-a": {"container": "revenue"},
            "identity-b": {"container": "clboss"},
        },
        "post_rebalance_controllers_held": {
            "revenue_ops": True,
            "clboss": True,
            "forced_cycles": False,
            "at": 10,
        },
    }
    calls = []

    def rpc(container, method, *args):
        calls.append((container, method, args))
        if method == "revenue-status":
            return {"operator_controls": {"values": {"paused": False}}}
        if method == "clboss-status":
            return {"rebalance_mode": {"mode": "xrebalance"}}
        return {}

    monkeypatch.setattr(runner, "cln_rpc", rpc)

    activation = runner.resume_post_demand_controllers(path, state)

    assert activation["source"] == "post_rebalance_demand"
    assert activation["forced_cycles"] is False
    assert state["post_rebalance_controllers_held"]["revenue_ops"] is False
    assert state["post_rebalance_controllers_held"]["clboss"] is False
    assert runner.resume_post_demand_controllers(path, state) is None
    assert all(method != "revenue-rebalance-cycle" for _container, method, _args in calls)


def test_post_demand_hold_requires_newer_completed_demand_block(tmp_path):
    runner = load_runner()
    state = {
        "events": [{"at": 9, "event": "smoke_complete"}],
        "post_rebalance_controllers_held": {
            "revenue_ops": True,
            "clboss": True,
            "forced_cycles": False,
            "at": 10,
        },
    }

    with pytest.raises(runner.RunnerError, match="completed demand block"):
        runner.resume_post_demand_controllers(tmp_path / "state.json", state)


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


def test_return_paths_require_completed_full_stack_block(tmp_path):
    runner = load_runner()
    path = runner.state_path(tmp_path, 3)
    runner.write_json_atomic(path, {
        "schema": runner.SCHEMA,
        "network_id": 4,
        "status": "isolated_full_stack_ready",
        "league": "full_stack",
        "events": [],
    })

    with pytest.raises(runner.RunnerError, match="completed full-stack"):
        runner.provision_return_paths(
            object(), replica=3, results_dir=tmp_path, capacity_sats=2_000_000
        )


def test_payer_wallet_topup_is_bounded_to_confirmed_shortfall(monkeypatch):
    runner = load_runner()
    funded = []

    def cln(_container, method, *_args):
        if method == "listfunds":
            return {"outputs": [
                {"status": "confirmed", "amount_msat": 2_500_000_000},
                {"status": "unconfirmed", "amount_msat": 9_000_000_000},
            ]}
        raise AssertionError(method)

    def lnd(_container, method, *_args):
        if method == "walletbalance":
            return {"confirmed_balance": "300000"}
        if method == "newaddress":
            return {"address": "bcrt1plnd"}
        raise AssertionError(method)

    monkeypatch.setattr(runner, "cln_rpc", cln)
    monkeypatch.setattr(runner, "lnd_rpc", lnd)
    monkeypatch.setattr(
        runner, "_send_fake_onchain_funds",
        lambda _bridge, _network, address, amount: funded.append((address, amount)),
    )

    result = runner.ensure_payer_wallet_funds(object(), 4)

    assert result == {
        "cln": {"before_sats": 2_500_000, "topup_sats": 0},
        "lnd": {"before_sats": 300_000, "topup_sats": 1_800_000},
    }
    assert funded == [("bcrt1plnd", 1_800_000)]


def test_payer_wallet_topup_fails_closed_on_malformed_lnd_balance(monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner, "cln_rpc", lambda *_args: {"outputs": []})
    monkeypatch.setattr(
        runner, "lnd_rpc", lambda *_args: {"confirmed_balance": "unknown"}
    )

    with pytest.raises(runner.RunnerError, match="nonnegative integer"):
        runner.ensure_payer_wallet_funds(object(), 4)


def test_return_paths_open_only_after_score_and_checkpoint_exact_channels(
    monkeypatch, tmp_path
):
    runner = load_runner()
    path = runner.state_path(tmp_path, 4)
    runner.write_json_atomic(path, {
        "schema": runner.SCHEMA,
        "network_id": 4,
        "status": "smoke_complete",
        "league": "full_stack",
        "events": [],
        "contenders": {
            "identity-a": {"container": "a"},
            "identity-b": {"container": "b"},
        },
    })
    calls = []
    funded = []
    gossip = []
    lnd_rows = iter(([], [{
        "active": True,
        "remote_pubkey": "lnd-sink-id",
        "channel_point": "funding:1",
        "scid": "123",
        "scid_str": "1x1x1",
    }]))

    def cln(container, method, *args):
        calls.append((container, method, args))
        if method == "getinfo":
            return {
                "id": "cln-payer-id" if container.endswith("cln-payer")
                else "cln-sink-id"
            }
        if method == "newaddr":
            return {"p2tr": "bcrt1ptest"}
        if method == "fundchannel":
            return {"channel_id": "cln-channel-id"}
        if method == "setchannel":
            return {"channels": [{}]}
        return {}

    def lnd(container, method, *args):
        calls.append((container, method, args))
        if method == "getinfo":
            return {
                "identity_pubkey": (
                    "lnd-sink-id" if container.endswith("lnd-sink") else "lnd-payer-id"
                )
            }
        if method == "newaddress":
            return {"address": "bcrt1plnd"}
        if method == "openchannel":
            return {"funding_txid": "funding"}
        if method == "getchaninfo":
            return {
                "node1_pub": "lnd-payer-id",
                "node1_policy": {
                    "time_lock_delta": 40,
                    "min_htlc": "1",
                    "max_htlc_msat": "1980000000",
                },
            }
        return {}

    monkeypatch.setattr(runner, "cln_rpc", cln)
    monkeypatch.setattr(runner, "lnd_rpc", lnd)
    monkeypatch.setattr(runner, "channel_rows", lambda _container: [])
    monkeypatch.setattr(runner, "_live_lnd_channels", lambda _container: next(lnd_rows))
    monkeypatch.setattr(
        runner, "_send_fake_onchain_funds",
        lambda _bridge, _network, address, amount: funded.append((address, amount)),
    )
    monkeypatch.setattr(runner, "_mine", lambda *_args: None)
    monkeypatch.setattr(
        runner, "_wait_return_paths", lambda paths: [
            {**paths[0], "short_channel_id": "1x1x0"},
            {**paths[1], "short_channel_id": "1x1x1", "channel_point": "funding:1"},
        ],
    )
    monkeypatch.setattr(
        runner, "wait_gossip_policies",
        lambda contenders, expected: gossip.append((contenders, expected)) or {},
    )

    state = runner.provision_return_paths(
        object(), replica=4, results_dir=tmp_path, capacity_sats=2_000_000
    )

    assert state["status"] == "return_paths_ready"
    assert state["return_path_fixture"]["present_during_scored_traffic"] is False
    assert state["return_path_fixture"] == {
        "present_during_scored_traffic": False,
        "fee_base_msat": runner.RETURN_PATH_FEE_BASE_MSAT,
        "fee_ppm": runner.RETURN_PATH_FEE_PPM,
        "purpose": "isolate post-pressure circular-route availability",
    }
    assert gossip[0][1] == [
        {
            "short_channel_id": "1x1x0",
            "source": "cln-payer-id",
            "fee_ppm": runner.RETURN_PATH_FEE_PPM,
        },
        {
            "short_channel_id": "1x1x1",
            "source": "lnd-payer-id",
            "fee_ppm": runner.RETURN_PATH_FEE_PPM,
        },
    ]
    assert {row["family"] for row in state["return_paths"]} == {"cln", "lnd"}
    assert funded == [
        ("bcrt1ptest", 2_100_000),
        ("bcrt1plnd", 2_100_000),
    ]
    assert all(method not in {"revenue-rebalance-cycle", "revenue-rebalance"}
               for _container, method, _args in calls)


def test_return_path_resume_reconciles_partial_open_without_refunding(
    monkeypatch, tmp_path
):
    runner = load_runner()
    path = runner.state_path(tmp_path, 5)
    partial = [
        {
            "family": "cln", "source_container": "cln-payer",
            "sink_container": "cln-sink", "peer_id": "cln-sink-id",
            "channel_id": "cln-channel-id", "capacity_sats": 2_000_000,
        },
        {
            "family": "lnd", "source_container": "lnd-payer",
            "sink_container": "lnd-sink", "peer_id": "lnd-sink-id",
            "funding_txid": "funding", "capacity_sats": 2_000_000,
        },
    ]
    runner.write_json_atomic(path, {
        "schema": runner.SCHEMA,
        "network_id": 4,
        "status": "controlled_depletion_ready",
        "league": "full_stack",
        "events": [
            {"event": "return_path_open_dispatched", "family": "cln"},
            {"event": "return_path_open_dispatched", "family": "lnd"},
        ],
        "contenders": {
            "identity-a": {"container": "a"},
            "identity-b": {"container": "b"},
        },
        "return_paths": partial,
    })
    calls = []

    def cln(container, method, *args):
        calls.append((container, method, args))
        if method == "getinfo":
            return {"id": "cln-payer-id" if container.endswith("payer") else "cln-sink-id"}
        if method == "setchannel":
            return {"channels": [{}]}
        raise AssertionError((container, method, args))

    def lnd(container, method, *args):
        calls.append((container, method, args))
        if method == "getinfo":
            return {"identity_pubkey": "lnd-payer-id" if container.endswith("payer") else "lnd-sink-id"}
        raise AssertionError((container, method, args))

    monkeypatch.setattr(runner, "cln_rpc", cln)
    monkeypatch.setattr(runner, "lnd_rpc", lnd)
    monkeypatch.setattr(
        runner, "_wait_return_paths", lambda paths: [
            {**paths[0], "short_channel_id": "1x1x0"},
            {**paths[1], "short_channel_id": "1x1x1", "channel_point": "funding:1"},
        ],
    )
    monkeypatch.setattr(runner, "_live_lnd_channels", lambda _container: [{
        "channel_point": "funding:1", "scid": "123",
    }])
    monkeypatch.setattr(
        runner, "_wait_lnd_local_policy", lambda *_args: {
            "time_lock_delta": 40, "min_htlc": "1", "max_htlc_msat": "1980000000",
        },
    )
    monkeypatch.setattr(runner, "_set_lnd_policy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "wait_gossip_policies", lambda *_args, **_kwargs: {})

    state = runner.provision_return_paths(
        object(), replica=5, results_dir=tmp_path, fee_ppm=10
    )

    assert state["status"] == "return_paths_ready"
    assert state["return_path_fixture"]["fee_ppm"] == 10
    assert any(
        event["event"] == "return_paths_reconciled_after_partial_open"
        for event in state["events"]
    )
    assert all(method not in {"newaddr", "fundchannel", "openchannel"}
               for _container, method, _args in calls)


def test_lnd_policy_wait_fails_closed_on_malformed_or_absent_graph(monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner, "lnd_rpc", lambda *_args: {"node1_pub": "other"})
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)

    with pytest.raises(runner.RunnerError, match="policy readback failed"):
        runner._wait_lnd_local_policy(
            "polar-n4-lnd-payer", "123", "payer", attempts=2, poll_seconds=0
        )


def test_return_path_snapshot_fails_closed_on_malformed_lnd_balance(
    monkeypatch
):
    runner = load_runner()
    state = {"return_paths": [
        {
            "family": "cln", "source_container": "cln-payer",
            "channel_id": "a",
        },
        {
            "family": "lnd", "source_container": "lnd-payer",
            "channel_point": "b:1",
        },
    ]}
    monkeypatch.setattr(runner, "channel_rows", lambda _container: [{
        "channel_id": "a", "state": "CHANNELD_NORMAL", "short_channel_id": "1x1x0",
        "to_us_msat": 1_000, "total_msat": 2_000,
    }])
    monkeypatch.setattr(runner, "_live_lnd_channels", lambda _container: [{
        "channel_point": "b:1", "active": True, "scid_str": "1x1x1",
        "local_balance": "malformed", "capacity": "2000000",
    }])

    with pytest.raises(runner.RunnerError, match="nonnegative integer"):
        runner.return_path_snapshot(state)


def test_observe_rebalances_uses_native_cycles_and_records_balance_improvement(
    monkeypatch, tmp_path
):
    runner = load_runner()
    path = runner.state_path(tmp_path, 5)
    runner.write_json_atomic(path, {
        "schema": runner.SCHEMA,
        "network_id": 4,
        "status": "return_paths_ready",
        "events": [],
        "assignment": {"revenue_ops": "identity-a", "clboss": "identity-b"},
        "contenders": {
            "identity-a": {"container": "revenue"},
            "identity-b": {"container": "clboss"},
        },
        "full_stack": {
            "revenue_rebalance_allowance_sats": 1_000,
            "clboss_rebalance_cost_baseline_msat": 0,
        },
        "return_paths": [{"family": "cln"}, {"family": "lnd"}],
    })
    totals = {
        "revenue": iter((
            runner.Totals(0, 0, 0, 500_000, (), worst_channel_imbalance_ppm=900_000),
            runner.Totals(0, 0, 0, 500_000, (), worst_channel_imbalance_ppm=500_000),
        )),
        "clboss": iter((
            runner.Totals(0, 0, 0, 500_000, (), worst_channel_imbalance_ppm=900_000),
            runner.Totals(0, 0, 0, 500_000, (), worst_channel_imbalance_ppm=900_000),
        )),
    }
    circular = {
        "revenue": iter((
            {"completed_count": 0, "delivered_msat": 0, "cost_msat": 0},
            {"completed_count": 1, "delivered_msat": 200_000_000, "cost_msat": 2_000},
        )),
        "clboss": iter((
            {"completed_count": 0, "delivered_msat": 0, "cost_msat": 0},
            {"completed_count": 0, "delivered_msat": 0, "cost_msat": 0},
        )),
    }
    rpc_calls = []
    monkeypatch.setattr(runner, "contender_totals", lambda container: next(totals[container]))
    monkeypatch.setattr(runner, "rebalance_totals", lambda container: next(circular[container]))
    monkeypatch.setattr(runner, "return_path_snapshot", lambda _state: {
        "cln": {"active": True}, "lnd": {"active": True},
    })
    monkeypatch.setattr(
        runner, "cln_rpc",
        lambda container, method, *args: rpc_calls.append((container, method, args)) or {},
    )
    wall_clock = iter((1_000.0, 5_000.0, 5_000.0))
    monkeypatch.setattr(runner.time, "time", lambda: next(wall_clock))
    result = runner.observe_rebalances(
        replica=5, results_dir=tmp_path, observe_seconds=0
    )

    assert result["controllers"]["revenue_ops"]["circular_payments"] == {
        "completed_count": 1, "delivered_msat": 200_000_000, "cost_msat": 2_000,
    }
    assert result["controllers"]["revenue_ops"][
        "worst_imbalance_improvement_ppm"
    ] == 400_000
    assert result["clboss_spend_policy"] == "native_unbounded"
    assert result["duration_seconds"] < 1.0
    saved = runner.read_state(path)
    assert saved["status"] == "rebalance_observed"
    assert saved["last_rebalance_observation"].startswith("rebalance-")
    assert all(method != "revenue-rebalance-cycle" for _container, method, _args in rpc_calls)
    assert all(
        not (method == "setconfig" and args == ("clboss-rebalance-mode", "off"))
        for _container, method, args in rpc_calls
    )


def test_retire_return_paths_confirms_direct_bypass_absent(monkeypatch, tmp_path):
    runner = load_runner()
    path = runner.state_path(tmp_path, 6)
    runner.write_json_atomic(path, {
        "schema": runner.SCHEMA,
        "network_id": 4,
        "status": "rebalance_observed",
        "events": [],
        "last_rebalance_observation": "rebalance-123",
        "assignment": {"revenue_ops": "identity-a", "clboss": "identity-b"},
        "contenders": {
            "identity-a": {"container": "revenue"},
            "identity-b": {"container": "clboss"},
        },
        "return_paths": [{"family": "cln"}, {"family": "lnd"}],
    })
    mines = []
    live = iter(({"cln": ["1x1x0"]}, {}))

    def close(state):
        state["return_path_closes_dispatched"] = True
        return True

    monkeypatch.setattr(runner, "_close_return_paths", close)
    monkeypatch.setattr(
        runner,
        "cln_rpc",
        lambda container, method, *args: (
            {"operator_controls": {"values": {"paused": True}}}
            if method == "revenue-status"
            else {"rebalance_mode": {"mode": "off"}}
            if method == "clboss-status"
            else {}
        ),
    )
    monkeypatch.setattr(runner, "_mine", lambda *args: mines.append(args))
    monkeypatch.setattr(runner, "_live_return_paths", lambda _state: next(live))
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)

    state = runner.retire_return_paths(
        object(), replica=6, results_dir=tmp_path, timeout_seconds=1
    )

    assert len(mines) == 1
    assert mines[0][1:] == (4, 6)
    assert state["status"] == "post_rebalance_ready"
    assert state["post_rebalance_controllers_held"]["forced_cycles"] is False
    assert state["return_paths_retired"] == {
        "confirmed_absent": True,
        "at": state["return_paths_retired"]["at"],
        "observation_block": "rebalance-123",
        "purpose": "prevent direct fixture bypass during post-rebalance demand",
    }
    assert state["events"][-1]["event"] == "return_paths_retired_for_scoring"


def test_post_rebalance_smoke_requires_retired_paths_and_records_lineage(
    monkeypatch, tmp_path
):
    runner = load_runner()
    path = runner.state_path(tmp_path, 8)
    roles = (
        ("cln", "payer"), ("cln", "sink"),
        ("lnd", "payer"), ("lnd", "sink"),
    )
    state = {
        "schema": runner.SCHEMA,
        "network_id": 4,
        "status": "post_rebalance_ready",
        "league": "full_stack",
        "market_profile": "realistic",
        "events": [],
        "assignment": {"revenue_ops": "identity-a", "clboss": "identity-b"},
        "contenders": {
            "identity-a": {"container": "revenue"},
            "identity-b": {"container": "clboss"},
        },
        "lane_map": {
            identity: {
                f"{prefix}x1x{index}": {"family": family, "side": side}
                for index, (family, side) in enumerate(roles)
            }
            for identity, prefix in (("identity-a", 1), ("identity-b", 2))
        },
        "last_rebalance_observation": "rebalance-456",
        "return_paths_retired": {"confirmed_absent": True, "at": 456},
        "post_rebalance_controllers_held": {
            "revenue_ops": True,
            "clboss": True,
            "forced_cycles": False,
            "at": 456,
        },
        "controlled_depletion": {"after": {"revenue_ops": {}, "clboss": {}}},
    }
    runner.write_json_atomic(path, state)
    policies = {
        "revenue": tuple((f"1x1x{i}", 0, 150) for i in range(4)),
        "clboss": tuple((f"2x1x{i}", 0, 1) for i in range(4)),
    }
    samples = {
        "revenue": iter((
            runner.Totals(0, 0, 0, 500_000, policies["revenue"]),
            runner.Totals(
                1, 5_000_000, 750, 500_000, policies["revenue"],
                channel_metrics=(("1x1x1", 1, 5_000_000, 750),),
            ),
        )),
        "clboss": iter((
            runner.Totals(0, 0, 0, 500_000, policies["clboss"]),
            runner.Totals(0, 0, 0, 500_000, policies["clboss"]),
        )),
    }
    monkeypatch.setattr(
        runner, "contender_totals", lambda container: next(samples[container])
    )
    monkeypatch.setattr(
        runner,
        "run_reconciled_traffic",
        lambda *_args, **_kwargs: [{
            "round": 0, "payer": "payer", "sink": "sink",
            "amount_sats": 5_000, "payment_hash": "hash",
            "payment": {"success": True},
        }],
    )

    block = runner.run_smoke(
        object(), replica=8, results_dir=tmp_path, rounds=1,
        amount_sats=5_000, pause_seconds=0,
        traffic_pattern="forward-pressure", traffic_family="cln",
    )

    assert block["phase"] == "post_rebalance_demand"
    assert block["post_rebalance"]["observation_block"] == "rebalance-456"
    assert block["post_rebalance"]["return_paths_retired"]["confirmed_absent"] is True
    assert block["traffic"]["fallback_settled"] == 0


def test_post_rebalance_smoke_rejects_malformed_hold_before_traffic(
    monkeypatch, tmp_path
):
    runner = load_runner()
    runner.write_json_atomic(runner.state_path(tmp_path, 9), {
        "schema": runner.SCHEMA,
        "status": "post_rebalance_ready",
        "events": [],
        "last_rebalance_observation": "rebalance-789",
        "return_paths_retired": {"confirmed_absent": True, "at": 789},
        "post_rebalance_controllers_held": {
            "revenue_ops": True,
            "clboss": False,
            "forced_cycles": False,
        },
    })
    dispatched = []
    monkeypatch.setattr(
        runner,
        "run_reconciled_traffic",
        lambda *_args, **_kwargs: dispatched.append(True),
    )

    with pytest.raises(runner.RunnerError, match="both controllers held"):
        runner.run_smoke(
            object(), replica=9, results_dir=tmp_path, rounds=1,
            amount_sats=5_000, pause_seconds=0,
        )

    assert dispatched == []


def test_rebalance_observation_is_not_directly_traffic_ready(tmp_path):
    runner = load_runner()
    runner.write_json_atomic(runner.state_path(tmp_path, 10), {
        "schema": runner.SCHEMA,
        "status": "rebalance_observed",
        "events": [],
    })

    with pytest.raises(runner.RunnerError, match="not traffic-ready"):
        runner.run_smoke(
            object(), replica=10, results_dir=tmp_path, rounds=1,
            amount_sats=5_000, pause_seconds=0,
        )


def test_return_path_cleanup_dispatches_exact_cln_and_lnd_closes(monkeypatch):
    runner = load_runner()
    state = {"return_paths": [
        {
            "family": "cln", "source_container": "polar-n4-cln-payer",
            "short_channel_id": "1x1x0", "channel_id": "cln-funding-id",
        },
        {
            "family": "lnd", "source_container": "polar-n4-lnd-payer",
            "channel_point": "funding:1",
        },
    ]}
    shell_calls = []
    monkeypatch.setattr(runner, "channel_rows", lambda _container: [{
        "channel_id": "cln-funding-id", "state": "CHANNELD_NORMAL",
    }])
    monkeypatch.setattr(
        runner, "_run",
        lambda command, **_kwargs: shell_calls.append(command) or type(
            "Result", (), {"returncode": 0, "stdout": "", "stderr": ""}
        )(),
    )

    assert runner._close_return_paths(state) is True
    assert shell_calls[0][-2:] == ["close", "1x1x0"]
    assert shell_calls[1][-3:] == ["closechannel", "--chan_point", "funding:1"]
    assert state["return_path_closes_dispatched"] is True


def test_competition_image_pins_all_source_revisions():
    runner = load_runner()
    dockerfile = (ROOT / "tools" / "polar-clboss" / "Dockerfile").read_text(encoding="utf-8")

    assert runner.IMAGE == "cl-revenue-ops-polar-clboss:565cd0f"
    assert runner.EXPECTED_REVENUE_REVISION.startswith("565cd0f")
    assert "elementsproject/lightningd:v26.06.6" in dockerfile
    assert "clightning-v26.06.7-Ubuntu-22.04-amd64.tar.xz" in dockerfile
    assert runner.EXPECTED_CLN_ARTIFACT_DIGEST in dockerfile
    assert 'test "$(lightningd --version)" = "v26.06.7"' in dockerfile
    assert "CLBOSS_COMMIT=8cb4e9215eba58b049375f234f5f073d0c7fc622" in dockerfile
    assert "XREBALANCE_COMMIT=fb70bf13cd9f3f79b14100bfdb8f2966884a4142" in dockerfile
    assert 'test "$(git -C /src/clboss rev-parse HEAD)" = "${CLBOSS_COMMIT}"' in dockerfile
    assert 'test "$(git -C /src/xrebalance rev-parse HEAD)" = "${XREBALANCE_COMMIT}"' in dockerfile
    assert "cargo build --release --locked" in dockerfile


def test_equal_targeted_pressure_is_crossed_checkpointed_and_unscored(
    monkeypatch, tmp_path
):
    runner = load_runner()
    path = tmp_path / "replica-67" / "state.json"
    state = {
        "status": "post_rebalance_ready",
        "network_id": 4,
        "post_rebalance_controllers_held": {
            "revenue_ops": True,
            "clboss": True,
            "forced_cycles": False,
        },
    }
    snapshots = iter([
        {
            "revenue_ops": {
                "source": {"short_channel_id": "1x1x0"},
                "depleted": {"local_balance_ppm": 300_000},
            },
            "clboss": {
                "source": {"short_channel_id": "2x1x0"},
                "depleted": {"local_balance_ppm": 250_000},
            },
        },
        {
            "revenue_ops": {
                "source": {"short_channel_id": "1x1x0"},
                "depleted": {"local_balance_ppm": 145_000},
            },
            "clboss": {
                "source": {"short_channel_id": "2x1x0"},
                "depleted": {"local_balance_ppm": 95_000},
            },
        },
    ])
    calls = []
    events = []
    monkeypatch.setattr(runner, "state_path", lambda *_args: path)
    monkeypatch.setattr(runner, "read_state", lambda _path: state)
    monkeypatch.setattr(
        runner, "controlled_depletion_snapshot", lambda *_args, **_kwargs: next(snapshots)
    )
    monkeypatch.setattr(
        runner,
        "_directed_lnd_fixture_payment",
        lambda _bridge, **kwargs: calls.append(kwargs) or {"success": True, **kwargs},
    )
    monkeypatch.setattr(
        runner,
        "_checkpoint",
        lambda _path, _state, event, **_kwargs: events.append(event),
    )

    saved = runner.apply_equal_targeted_pressure(
        object(), replica=67, results_dir=tmp_path
    )

    assert [call["label"] for call in calls] == [
        "targeted-revenue_ops", "targeted-clboss",
        "targeted-revenue_ops", "targeted-clboss",
        "targeted-revenue_ops", "targeted-clboss",
        "targeted-revenue_ops", "targeted-clboss",
    ]
    assert saved["targeted_pressure"]["competitive_scoring"] is False
    assert saved["targeted_pressure"]["total_sats_per_controller"] == 155_000
    assert saved["status"] == "post_rebalance_ready"
    assert events == [
        "equal_targeted_pressure_started", "equal_targeted_pressure_complete"
    ]


def test_equal_targeted_pressure_rejects_malformed_hold(monkeypatch, tmp_path):
    runner = load_runner()
    monkeypatch.setattr(
        runner,
        "read_state",
        lambda _path: {
            "status": "post_rebalance_ready",
            "post_rebalance_controllers_held": {"revenue_ops": True},
        },
    )

    with pytest.raises(runner.RunnerError, match="both controllers held"):
        runner.apply_equal_targeted_pressure(
            object(), replica=67, results_dir=tmp_path
        )


def test_preflight_rejects_wrong_revision_behind_default_image(monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner, "network_record", lambda *_args: {
        "id": 4,
        "name": "lab",
        "status": "Started",
        "nodes": {"lightning": []},
    })
    monkeypatch.setattr(runner, "docker_running", lambda _name: True)
    monkeypatch.setattr(
        runner,
        "_run",
        lambda _command: type(
            "Result",
            (),
            {
                "stdout": json.dumps([{
                    "Id": "sha256:stale",
                    "Config": {"Labels": {
                        "org.opencontainers.image.revision.revenue_ops": "stale",
                    }},
                }]),
            },
        )(),
    )

    class Bridge:
        def health(self):
            return {"status": "ok"}

        def call(self, method, _params):
            assert method == "list_networks"
            return {"networks": [{"id": 4, "status": "Started"}]}

    with pytest.raises(runner.RunnerError, match="unexpected revenue_ops revision"):
        runner.preflight(Bridge(), 4, runner.IMAGE)


def test_preflight_rejects_unverified_cln_artifact_behind_default_image(monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner, "network_record", lambda *_args: {
        "id": 4,
        "name": "lab",
        "status": "Started",
        "nodes": {"lightning": []},
    })
    monkeypatch.setattr(runner, "docker_running", lambda _name: True)
    monkeypatch.setattr(
        runner,
        "_run",
        lambda _command: type(
            "Result",
            (),
            {
                "stdout": json.dumps([{
                    "Id": "sha256:wrong-cln",
                    "Config": {"Labels": {
                        "org.opencontainers.image.revision.revenue_ops": (
                            runner.EXPECTED_REVENUE_REVISION
                        ),
                        "org.opencontainers.image.version.cln": "v26.06.7",
                        "org.opencontainers.image.digest.cln": "sha256:wrong",
                    }},
                }]),
            },
        )(),
    )

    class Bridge:
        def health(self):
            return {"status": "ok"}

        def call(self, method, _params):
            assert method == "list_networks"
            return {"networks": [{"id": 4, "status": "Started"}]}

    with pytest.raises(runner.RunnerError, match="unexpected CLN artifact digest"):
        runner.preflight(Bridge(), 4, runner.IMAGE)
