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
    before = runner.Totals(2, 20_000, 20, 500_000, (("1x1x1", 1, 10),))
    after = runner.Totals(5, 65_000, 65, 490_000, (("1x1x1", 1, 15),))

    assert runner.totals_delta(before, after) == {
        "forward_count": 3,
        "volume_msat": 45_000,
        "routing_fee_msat": 45,
        "mean_local_liquidity_sats": 495_000,
        "policy_changes": 1,
        "rebalance_cost_msat": 0,
    }


def test_totals_delta_accounts_for_completed_circular_payment_cost():
    runner = load_runner()
    before = runner.Totals(0, 0, 0, 500_000, (), rebalance_cost_msat=100)
    after = runner.Totals(0, 0, 0, 500_000, (), rebalance_cost_msat=375)

    assert runner.totals_delta(before, after)["rebalance_cost_msat"] == 275


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


def test_totals_delta_fails_closed_on_counter_regression():
    runner = load_runner()
    before = runner.Totals(2, 20_000, 20, 500_000, ())
    after = runner.Totals(1, 20_000, 20, 500_000, ())

    with pytest.raises(runner.RunnerError, match="regressed"):
        runner.totals_delta(before, after)


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
        lambda _rounds, _amount: (("cln", "forward", 5_000), ("cln", "reverse", 5_000)),
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
            "policy_changes": 0,
            "rebalance_cost_msat": 0,
        },
        "clboss": {
            "forward_count": 0,
            "volume_msat": 0,
            "routing_fee_msat": 0,
            "mean_local_liquidity_sats": 500_000,
            "policy_changes": 0,
            "rebalance_cost_msat": 0,
        },
    }
    progress = json.loads(Path(result["progress_file"]).read_text(encoding="utf-8"))
    assert progress["records"] == [completed]
    assert progress["uncertain_operation"]["payment_hash"] == "hash"
    assert progress["partial_contenders"] == result["partial_contenders"]


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
