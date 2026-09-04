"""The Grand Prix base runner is matched, resumable, and mutation-gated."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "grand_prix_runner.py"
CALIBRATION = ROOT / "tests" / "fixtures" / "competitive_improvement" / "calibration.v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("grand_prix_runner", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _topology():
    manifest_path = ROOT / "tools" / "grand_prix_manifest.py"
    spec = importlib.util.spec_from_file_location("grand_prix_manifest_for_runner", manifest_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    return module.build_topology(calibration, public_seed=20260901)


class FakeBridge:
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    def call(self, tool, arguments):
        self.calls.append((tool, arguments))
        response = self.responses.get(tool, {})
        return response(arguments) if callable(response) else response


def test_plan_partitions_22_docker_nodes_and_two_contenders():
    runner = _module()
    result = runner.runtime_plan(_topology())
    assert result["docker_nodes"] == 22
    assert result["external_contenders"] == 2
    assert result["implementation_counts"] == {"c-lightning": 15, "LND": 7}
    assert result["background_channels"] == 25
    assert result["contender_channels_deferred"] == 32
    assert result["mutations_required"] is False


def test_assignment_and_container_names_cross_safely():
    runner = _module()
    assert runner.assignment_for(1) == {"revenue_ops": "identity-a", "clboss": "identity-b"}
    assert runner.assignment_for(2) == {"revenue_ops": "identity-b", "clboss": "identity-a"}
    assert runner.contender_container(5, 1, "identity-a") == (
        "revenue-gp-n5-grand-prix-r1-identity-a"
    )
    with pytest.raises(runner.RunnerError, match="unknown contender"):
        runner.contender_container(5, 1, "../backend1")


def test_docker_backend_uses_run_scoped_names():
    runner = _module()
    assert runner._base_container_name(5, "cln-payer") == "revenue-gp-n5-cln-payer"
    assert runner._docker_network_name(5) == "revenue-gp-n5"
    assert runner.contender_container(5, 2, "identity-b") == (
        "revenue-gp-n5-grand-prix-r2-identity-b"
    )


def test_revenue_wrapper_prefers_generic_and_falls_back_for_frozen_image(monkeypatch):
    runner = _module()
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return type("Result", (), {"returncode": 0 if command[-1].endswith("polar-wrapper") else 1})()

    monkeypatch.setattr(runner, "_run", fake_run)
    container = "revenue-gp-n5-grand-prix-r2-identity-b"
    assert runner._revenue_plugin_path(container) == runner.LEGACY_REVENUE_PLUGIN
    assert [call[-1] for call in calls] == [
        runner.REVENUE_PLUGIN,
        runner.LEGACY_REVENUE_PLUGIN,
    ]


def test_revenue_wrapper_resolution_fails_closed(monkeypatch):
    runner = _module()
    monkeypatch.setattr(
        runner,
        "_run",
        lambda *_args, **_kwargs: type("Result", (), {"returncode": 1})(),
    )
    with pytest.raises(runner.RunnerError, match="neither the current nor legacy"):
        runner._revenue_plugin_path("revenue-gp-n5-grand-prix-r2-identity-b")
    with pytest.raises(runner.RunnerError, match="unsafe contender"):
        runner._revenue_plugin_path("../unsafe")


def test_msat_normalizes_supported_cln_encodings():
    runner = _module()
    assert runner._msat(12) == 12
    assert runner._msat("13msat") == 13
    assert runner._msat({"msat": 14}) == 14
    with pytest.raises(runner.RunnerError, match="invalid msat"):
        runner._msat("malformed")


def test_lnd_readiness_probe_retries_short_timeouts(monkeypatch):
    runner = _module()
    calls = []

    def fake_rpc(container, command, *, timeout):
        calls.append((container, command, timeout))
        if len(calls) == 1:
            raise runner.RunnerError("command timed out")
        return {"confirmed_balance": "1"}

    monkeypatch.setattr(runner, "_lnd_rpc", fake_rpc)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    assert runner._wait_lnd_read_rpc(
        "revenue-gp-n5-lnd-payer", "walletbalance"
    ) == {"confirmed_balance": "1"}
    assert calls == [
        ("revenue-gp-n5-lnd-payer", "walletbalance", 10),
        ("revenue-gp-n5-lnd-payer", "walletbalance", 10),
    ]


def test_lnd_readiness_probe_rejects_action_rpc():
    runner = _module()
    with pytest.raises(runner.RunnerError, match="non-read-only"):
        runner._wait_lnd_read_rpc(
            "revenue-gp-n5-lnd-payer", "openchannel"
        )


def test_lnd_base_node_readiness_uses_topology_validated_wrapper(monkeypatch):
    runner = _module()
    topology = _topology()
    calls = []

    def fake_rpc(network_id, node_name, received_topology, command, *, timeout):
        calls.append(
            (network_id, node_name, received_topology, command, timeout)
        )
        return {"identity_pubkey": "02sink"}

    monkeypatch.setattr(runner, "_lnd_node_rpc", fake_rpc)
    assert runner._wait_lnd_node_read_rpc(
        1_788_000_001, "lnd-sink", topology, "getinfo"
    ) == {"identity_pubkey": "02sink"}
    assert calls == [
        (1_788_000_001, "lnd-sink", topology, "getinfo", 10)
    ]
    with pytest.raises(runner.RunnerError, match="non-read-only"):
        runner._wait_lnd_node_read_rpc(
            1_788_000_001, "lnd-sink", topology, "openchannel"
        )


def test_cln_wallet_output_totals_exclude_reserved_and_channels(monkeypatch):
    runner = _module()
    monkeypatch.setattr(runner, "_cln_rpc", lambda *_args: {
        "outputs": [
            {"amount_msat": 3_000, "status": "confirmed", "reserved": False},
            {"amount_msat": "4000msat", "status": "unconfirmed"},
            {"amount_msat": 5_000, "status": "confirmed", "reserved": True},
        ],
        "channels": [{"our_amount_msat": 9_000}],
    })
    assert runner._cln_wallet_output_totals("safe-container") == (3, 4)


def test_contender_channel_portfolios_remain_exactly_matched():
    runner = _module()
    rows = runner.contender_channels(_topology())
    vectors = {}
    for identity in ("identity-a", "identity-b"):
        vectors[identity] = sorted(
            (row["capacity_sats"], row["fee_ppm"], row["initial_source_ratio"])
            for row in rows if row["contender_lane"] == identity
        )
    assert len(vectors["identity-a"]) == 16
    assert vectors["identity-a"] == vectors["identity-b"]


def test_contender_opening_uses_manifest_ratio_push_shaping():
    source = TOOL.read_text(encoding="utf-8")
    assert 'f"push_msat={push_sats * 1000}msat"' in source
    assert '"--push_amt", push_sats' in source


def test_mutating_commands_require_apply(tmp_path):
    runner = _module()
    topology_path = tmp_path / "topology.json"
    topology_path.write_text(json.dumps(_topology()), encoding="utf-8")
    with pytest.raises(runner.RunnerError, match="pass --apply"):
        runner.main(["create-base", "--topology", str(topology_path)])


def test_create_base_uses_manifest_counts_and_deterministic_renames(tmp_path):
    runner = _module()
    topology = _topology()
    generated = []
    for implementation, count in (("c-lightning", 15), ("LND", 7)):
        generated.extend(
            {"name": f"generated-{implementation}-{index}", "implementation": implementation}
            for index in range(count)
        )
    network = {"id": 9, "name": runner.DEFAULT_NAME, "nodes": {"lightning": generated}}
    bridge = FakeBridge({
        "list_networks": {"networks": []},
        "create_network": {"network": network},
        "rename_node": {},
    })
    state = runner.create_base(
        bridge, topology, name=runner.DEFAULT_NAME, state_path=tmp_path / "state.json"
    )
    create = next(arguments for tool, arguments in bridge.calls if tool == "create_network")
    assert create["nodes"] == [
        {"implementation": "bitcoind", "count": 1},
        {"implementation": "c-lightning", "count": 15},
        {"implementation": "LND", "count": 7},
    ]
    assert len([call for call in bridge.calls if call[0] == "rename_node"]) == 22
    assert state["network_id"] == 9
    assert state["status"] == "base_created"


def test_background_wiring_never_touches_contenders_and_checkpoints(tmp_path):
    runner = _module()
    topology = _topology()
    state_path = tmp_path / "state.json"
    state = {
        "schema": runner.SCHEMA,
        "network_id": 9,
        "network_name": runner.DEFAULT_NAME,
        "topology_digest": runner._digest(topology),
        "status": "base_started",
        "background_channels": [],
        "events": [],
    }
    runner._write_json_atomic(state_path, state)

    def node_info(arguments):
        return {"info": {"pubkey": f"pk-{arguments['nodeName']}"}}

    bridge = FakeBridge({
        "get_node_info": node_info,
        "list_channels": {"channels": []},
        "open_channel": {},
        "mine_blocks": {},
    })
    result = runner.wire_background(bridge, topology, state_path=state_path)
    opens = [arguments for tool, arguments in bridge.calls if tool == "open_channel"]
    assert len(opens) == 25
    assert all("identity-" not in row["fromNode"] for row in opens)
    assert all("identity-" not in row["toNode"] for row in opens)
    assert result["status"] == "background_ready"
    assert len(result["background_channels"]) == 25
    assert len([row for row in result["events"] if row["event"] == "background_channel_opened"]) == 25


def test_background_wiring_continues_after_uncertain_mining_timeout(tmp_path):
    runner = _module()
    topology = _topology()
    state_path = tmp_path / "state.json"
    state = {
        "schema": runner.SCHEMA,
        "network_id": 9,
        "network_name": runner.DEFAULT_NAME,
        "topology_digest": runner._digest(topology),
        "status": "base_started",
        "background_channels": [],
        "events": [],
    }
    runner._write_json_atomic(state_path, state)

    def node_info(arguments):
        return {"info": {"pubkey": f"pk-{arguments['nodeName']}"}}

    def mining_timeout(_arguments):
        raise runner.DockerLabError("Tool execution timed out")

    bridge = FakeBridge({
        "get_node_info": node_info,
        "list_channels": {"channels": []},
        "open_channel": {},
        "mine_blocks": mining_timeout,
    })
    result = runner.wire_background(bridge, topology, state_path=state_path)
    assert result["status"] == "background_ready"
    assert len(result["background_channels"]) == 25
    assert len([
        row for row in result["events"]
        if row["event"] == "background_mine_timeout_uncertain"
    ]) == 25


def test_background_wiring_still_fails_on_non_timeout_mining_error(tmp_path):
    runner = _module()
    topology = _topology()
    state_path = tmp_path / "state.json"
    state = {
        "schema": runner.SCHEMA,
        "network_id": 9,
        "network_name": runner.DEFAULT_NAME,
        "topology_digest": runner._digest(topology),
        "status": "base_started",
        "background_channels": [],
        "events": [],
    }
    runner._write_json_atomic(state_path, state)

    def node_info(arguments):
        return {"info": {"pubkey": f"pk-{arguments['nodeName']}"}}

    def mining_failure(_arguments):
        raise runner.DockerLabError("backend rejected mutation")

    bridge = FakeBridge({
        "get_node_info": node_info,
        "list_channels": {"channels": []},
        "open_channel": {},
        "mine_blocks": mining_failure,
    })
    with pytest.raises(runner.DockerLabError, match="backend rejected"):
        runner.wire_background(bridge, topology, state_path=state_path)


def test_read_only_channel_reconciliation_retries_timeout(monkeypatch):
    runner = _module()
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    attempts = 0

    def flaky_list(_arguments):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise runner.DockerLabError("Tool execution timed out")
        return {"channels": [{"status": "Open"}]}

    bridge = FakeBridge({"list_channels": flaky_list})
    rows = runner._list_channels_with_timeout_retries(
        bridge,
        network_id=9,
        node_name="hub-1",
    )
    assert rows == [{"status": "Open"}]
    assert attempts == 2


def test_read_only_channel_reconciliation_survives_extended_bridge_stall(monkeypatch):
    runner = _module()
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    attempts = 0

    def flaky_list(_arguments):
        nonlocal attempts
        attempts += 1
        if attempts < 8:
            raise runner.DockerLabError("Tool execution timed out")
        return {"channels": [{"status": "Open"}]}

    rows = runner._list_channels_with_timeout_retries(
        FakeBridge({"list_channels": flaky_list}),
        network_id=9,
        node_name="hub-1",
    )
    assert rows == [{"status": "Open"}]
    assert attempts == 8


def test_native_base_channel_normalization(monkeypatch):
    runner = _module()
    topology = _topology()
    monkeypatch.setattr(runner, "_cln_rpc", lambda *_args, **_kwargs: {
        "channels": [{
            "peer_id": "cln-peer",
            "total_msat": 2_000_000_000,
            "state": "CHANNELD_NORMAL",
        }]
    })
    monkeypatch.setattr(runner, "_lnd_node_rpc", lambda *_args, **_kwargs: {
        "channels": [{
            "remote_pubkey": "lnd-peer",
            "capacity": "3000000",
            "active": True,
        }]
    })
    assert runner._native_base_channels(9, "cln-payer", topology) == [{
        "pubkey": "cln-peer", "capacity": "2000000", "status": "Open",
    }]
    assert runner._native_base_channels(9, "lnd-payer", topology) == [{
        "pubkey": "lnd-peer", "capacity": "3000000", "status": "Open",
    }]


def test_native_background_reconciliation_polls_exact_channel(tmp_path, monkeypatch):
    runner = _module()
    topology = _topology()
    state_path = tmp_path / "state.json"
    runner._write_json_atomic(state_path, {
        "schema": runner.SCHEMA,
        "network_id": 9,
        "network_name": runner.DEFAULT_NAME,
        "topology_digest": runner._digest(topology),
        "status": "base_started",
        "background_channels": [],
        "events": [],
    })
    first = runner.background_channels(topology)[0]
    capacity = int(first["capacity_sats"])
    peer_id = "native-peer"
    probes = 0

    def native_channels(*_args):
        nonlocal probes
        probes += 1
        if probes < 3:
            return []
        return [{"pubkey": peer_id, "capacity": str(capacity), "status": "Open"}]

    def stop_after_first(*_args, **_kwargs):
        if probes >= 3:
            raise runner.RunnerError("stop after first checkpoint")
        return peer_id

    bridge = FakeBridge({
        "open_channel": lambda _arguments: (_ for _ in ()).throw(
            runner.DockerLabError("Tool execution timed out")
        ),
    })
    monkeypatch.setattr(runner, "_native_base_pubkey", stop_after_first)
    monkeypatch.setattr(runner, "_native_base_channels", native_channels)
    monkeypatch.setattr(runner, "_native_mine_blocks", lambda *_args: None)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    with pytest.raises(runner.RunnerError, match="stop after first checkpoint"):
        runner.wire_background(
            bridge, topology, state_path=state_path, native_io=True
        )
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert len(saved["background_channels"]) == 1
    assert probes == 3


def test_background_channel_reconciliation_fails_closed_on_duplicate(tmp_path):
    runner = _module()
    topology = _topology()
    state_path = tmp_path / "state.json"
    runner._write_json_atomic(state_path, {
        "schema": runner.SCHEMA,
        "network_id": 9,
        "network_name": runner.DEFAULT_NAME,
        "topology_digest": runner._digest(topology),
        "status": "base_started",
        "background_channels": [],
        "events": [],
    })
    first = runner.background_channels(topology)[0]
    peer = (
        first["destination"]
        if float(first["initial_source_ratio"]) >= 0.5
        else first["source"]
    )
    capacity = int(first["capacity_sats"])
    duplicate = {
        "pubkey": f"pk-{peer}",
        "capacity": str(capacity),
        "status": "Open",
    }
    bridge = FakeBridge({
        "get_node_info": {"info": {"pubkey": f"pk-{peer}"}},
        "list_channels": {"channels": [duplicate, dict(duplicate)]},
    })
    with pytest.raises(runner.RunnerError, match="topology mismatch"):
        runner.wire_background(bridge, topology, state_path=state_path)
    assert not [call for call in bridge.calls if call[0] == "open_channel"]


def test_mutation_lock_rejects_overlapping_processes(tmp_path):
    runner = _module()
    state_path = tmp_path / "state.json"
    first = runner._acquire_mutation_lock(state_path)
    try:
        with pytest.raises(runner.RunnerError, match="do not overlap retries"):
            runner._acquire_mutation_lock(state_path)
    finally:
        runner.fcntl.flock(first.fileno(), runner.fcntl.LOCK_UN)
        first.close()


def test_invalid_or_malformed_state_fails_closed(tmp_path):
    runner = _module()
    path = tmp_path / "state.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(runner.RunnerError, match="schema"):
        runner._read_state(path, _topology())


def test_runner_source_has_no_production_or_plugin_action_rpc():
    source = TOOL.read_text(encoding="utf-8")
    assert "ssh" not in source
    assert "revenue-rebalance" not in source
    assert "revenue-fee-cycle" not in source
    assert "start_revenue" not in source


def test_payment_results_are_redacted_to_accounting_fields():
    runner = _module()
    result = runner._compact_payment_result({
        "status": "complete", "amount_msat": 10, "amount_sent_msat": 11,
        "payment_preimage": "do-not-persist",
    })
    assert result == {"status": "complete", "amount_msat": 10, "amount_sent_msat": 11}


def test_fee_plan_covers_both_directions_and_keeps_contenders_matched():
    runner = _module()
    rows = runner.fee_policy_plan(_topology())
    assert len(rows) == 114
    assert len({(row["node"], row["peer"]) for row in rows}) == 114
    contender_vectors = {
        identity: sorted(row["fee_ppm"] for row in rows if row["node"] == identity)
        for identity in ("identity-a", "identity-b")
    }
    assert len(contender_vectors["identity-a"]) == 16
    assert contender_vectors["identity-a"] == contender_vectors["identity-b"]


def test_controller_mutation_is_explicitly_apply_gated(tmp_path):
    runner = _module()
    topology_path = tmp_path / "topology.json"
    topology_path.write_text(json.dumps(_topology()), encoding="utf-8")
    with pytest.raises(runner.RunnerError, match="pass --apply"):
        runner.main(["start-controllers", "--topology", str(topology_path)])


def test_payer_top_up_is_explicitly_apply_gated(tmp_path):
    runner = _module()
    topology_path = tmp_path / "topology.json"
    topology_path.write_text(json.dumps(_topology()), encoding="utf-8")
    with pytest.raises(runner.RunnerError, match="pass --apply"):
        runner.main(["top-up-payers", "--topology", str(topology_path)])


@pytest.mark.parametrize(
    ("mode", "ceiling", "dynamic_htlcmax", "message"),
    [
        ("unknown", 2000, True, "market mode"),
        ("premium", 0, True, "max fee"),
        ("premium", 100001, True, "max fee"),
        ("premium", True, True, "max fee"),
        ("premium", 2000, "off", "dynamic htlcmax"),
    ],
)
def test_controller_arm_rejects_malformed_tuning_before_mutation(
    tmp_path, mode, ceiling, dynamic_htlcmax, message
):
    runner = _module()
    with pytest.raises(runner.RunnerError, match=message):
        runner.start_controllers(
            _topology(),
            state_path=tmp_path / "missing-state.json",
            revenue_market_mode=mode,
            revenue_max_fee_ppm=ceiling,
            revenue_dynamic_htlcmax=dynamic_htlcmax,
        )


def test_equivalent_controller_catalog_is_frozen_and_loadable():
    runner = _module()
    model, validation = runner._equivalent_model_context(
        "ln_operator", runner.EQUIVALENT_CONTROLLER_CONFIG
    )
    assert model["comparison_class"] == "algorithm_equivalent"
    assert model["rebalance_mode"] == "off_for_fee_only_comparison"
    assert validation["catalog_digest"].startswith("sha256:")


def test_equivalent_policy_application_uses_only_returned_intents(monkeypatch):
    runner = _module()
    model, _validation = runner._equivalent_model_context(
        "ln_operator", runner.EQUIVALENT_CONTROLLER_CONFIG
    )
    calls = []

    def fake_rpc(_container, *arguments):
        calls.append(arguments)
        if arguments == ("listpeerchannels",):
            return {"channels": [{
                "peer_connected": True, "peer_id": "fake-peer",
                "fee_proportional_millionths": 100,
                "total_msat": 1_000_000, "to_us_msat": 100_000,
            }]}
        return {}

    monkeypatch.setattr(runner, "_cln_rpc", fake_rpc)
    result = runner._apply_equivalent_competitor_policy("safe-container", model)
    assert result == {
        "eligible_channels": 1,
        "changed_channels": 1,
        "target_fee_ppm": {"min": 722, "max": 722},
    }
    assert calls == [
        ("listpeerchannels",),
        ("setchannel", "fake-peer", 0, 722),
    ]


def test_equivalent_refresh_is_noop_for_clboss_and_pinned_for_torq(monkeypatch):
    runner = _module()
    assert runner._refresh_equivalent_competitor({
        "controller_readback": {"competitor": {"id": "clboss"}},
        "contenders": {}, "assignment": {},
    }) is None
    model, _validation = runner._equivalent_model_context(
        "torq", runner.EQUIVALENT_CONTROLLER_CONFIG
    )
    state = {
        "controller_readback": {"competitor": {
            "id": "torq", "model": model, "refresh_count": 1,
        }},
        "contenders": {"identity-b": {"container": "safe-container"}},
        "assignment": {"clboss": "identity-b"},
    }
    monkeypatch.setattr(
        runner, "_apply_equivalent_competitor_policy",
        lambda container, frozen: {
            "eligible_channels": 16, "changed_channels": 2,
            "target_fee_ppm": {"min": 50, "max": 500},
        },
    )
    response = runner._refresh_equivalent_competitor(state)
    assert response["changed_channels"] == 2
    assert state["controller_readback"]["competitor"]["refresh_count"] == 2


def test_torq_refresh_requires_managed_channel_balance_change():
    runner = _module()
    state = {
        "controller_readback": {"competitor": {
            "id": "torq",
            "trigger": {"minimum_balance_change_sats": 50_000},
        }},
        "assignment": {"clboss": "identity-a"},
    }
    record = {
        "outcome": "settled",
        "contender_delta": {
            "identity-a": {"volume_msat": 0},
            "identity-b": {"volume_msat": 750_000_000},
        },
    }
    assert runner._equivalent_refresh_due(state, record) is False
    record["contender_delta"]["identity-a"]["volume_msat"] = 49_999_000
    assert runner._equivalent_refresh_due(state, record) is False
    record["contender_delta"]["identity-a"]["volume_msat"] = 50_000_000
    assert runner._equivalent_refresh_due(state, record) is True
    assert runner._equivalent_refresh_due(state, {"malformed": True}) is False


def test_unknown_competitor_is_rejected_before_state_or_rpc(tmp_path):
    runner = _module()
    with pytest.raises(runner.RunnerError, match="unsupported competitor"):
        runner.start_controllers(
            _topology(), state_path=tmp_path / "missing.json",
            competitor_controller="not-a-product",
        )


def test_controller_arm_is_pinned_in_plugin_start_and_readback_source():
    source = TOOL.read_text(encoding="utf-8")
    assert 'f"revenue-ops-market-fee-mode={revenue_market_mode}"' in source
    assert 'f"revenue-ops-max-fee-ppm={revenue_max_fee_ppm}"' in source
    assert '"revenue-config", "get", "market_fee_mode"' in source
    assert '"revenue-config", "get", "max_fee_ppm"' in source
    assert '"revenue-config", "get", "enable_dynamic_htlcmax"' in source
    assert "time.sleep(CONTROLLER_WARMUP_SECONDS)" in source
    assert '"warmup_seconds": CONTROLLER_WARMUP_SECONDS' in source
    assert 'state["competitor_controller"] = competitor_controller' in source
    assert '"equivalent_competitor_refreshed"' in source


def test_custom_image_requires_experiment_patch_attestation_source():
    source = TOOL.read_text(encoding="utf-8")
    assert "org.opencontainers.image.experiment.patch_digest" in source
    dockerfile = ROOT / "tools" / "grand-prix" / "Dockerfile"
    body = dockerfile.read_text(encoding="utf-8")
    assert "ARG EXPERIMENT_PATCH_DIGEST" in body
    assert "COPY modules/fee_controller.py" in body
    assert "COPY modules/admission_policy.py" in body


def test_metric_delta_is_capital_normalized_and_handles_zero_volume():
    runner = _module()
    before = {identity: {"settled_count": 1, "volume_msat": 10, "fee_msat": 2}
              for identity in ("identity-a", "identity-b")}
    after = {
        "identity-a": {"settled_count": 3, "volume_msat": 110, "fee_msat": 12},
        "identity-b": {"settled_count": 1, "volume_msat": 10, "fee_msat": 2},
    }
    result = runner._metric_delta(before, after)
    assert result["identity-a"]["settled_count"] == 2
    assert result["identity-a"]["fee_ppm_on_forwarded_volume"] == 100_000
    assert result["identity-b"]["fee_ppm_on_forwarded_volume"] == 0


def test_channel_policy_snapshot_is_anonymous_aggregate(monkeypatch):
    runner = _module()
    monkeypatch.setattr(runner, "_cln_rpc", lambda *_args, **_kwargs: {
        "channels": [
            {"peer_id": "secret-a", "fee_proportional_millionths": 100,
             "total_msat": "1000msat", "to_us_msat": "250msat", "peer_connected": True},
            {"peer_id": "secret-b", "fee_proportional_millionths": 300,
             "total_msat": {"msat": 3000}, "to_us_msat": 2250, "peer_connected": False},
            {"malformed": True},
        ]
    })

    result = runner._channel_policy_snapshot("safe-container")

    assert result == {
        "channels": 2,
        "active_channels": 1,
        "fee_ppm": {"min": 100, "median": 200.0, "mean": 200.0, "max": 300},
        "local_balance_ratio": 0.625,
    }
    assert "peer_id" not in str(result)


def test_channel_policy_snapshot_rejects_malformed_input(monkeypatch):
    runner = _module()
    monkeypatch.setattr(runner, "_cln_rpc", lambda *_args, **_kwargs: {"channels": [{}]})
    with pytest.raises(runner.RunnerError, match="no usable"):
        runner._channel_policy_snapshot("safe-container")


def test_native_payment_attempts_have_bounded_retry_source():
    source = TOOL.read_text(encoding="utf-8")
    assert '"xpay"' in source
    assert '"retry_for=5"' in source
    assert 'f"maxfee={max_fee_msat}msat"' in source
    assert '"--timeout", "5s"' in source
    assert '"--fee_limit_percent", "10"' in source


def test_public_traffic_persists_anonymous_per_payment_contender_deltas():
    source = TOOL.read_text(encoding="utf-8")
    assert 'record["contender_delta"] = _metric_delta(metric_cursor, contender_after)' in source
    assert 'record["contender_after"] = contender_after' in source
    assert 'run["per_payment_attribution_complete"] = True' in source
    assert 'run["post_traffic_unattributed_delta"]' in source


def test_contender_launch_has_no_implicit_image_default():
    runner = _module()
    args = runner.build_parser().parse_args([
        "launch-contenders", "--topology", str(CALIBRATION)
    ])
    assert args.image is None


@pytest.mark.parametrize(
    ("payer", "sink", "family"),
    [
        ("cln-sink", "cln-payer", "cln"),
        ("lnd-sink", "lnd-payer", "lnd"),
    ],
)
def test_reverse_unknown_payment_reconciliation_is_read_only(
    tmp_path, monkeypatch, payer, sink, family
):
    runner = _module()
    topology = _topology()
    item = topology["traffic"][0]
    item.update({"direction": "reverse", "payer": payer, "sink": sink})
    state_path = tmp_path / "state.json"
    state = {
        "schema": runner.SCHEMA,
        "network_id": 9,
        "network_name": runner.DEFAULT_NAME,
        "topology_digest": runner._digest(topology),
        "replica": 1,
        "status": "public_traffic_unknown",
        "public_traffic": {"records": []},
        "events": [
            {"event": "public_payment_unknown", "sequence": 0},
            {"event": "public_timeout_blocks_mined", "blocks": 144},
        ],
    }
    runner._write_json_atomic(state_path, state)
    calls = []

    def fake_cln(container, *arguments, **kwargs):
        calls.append(("cln", container, arguments))
        if arguments[0] == "listinvoices":
            return {"invoices": [{"status": "unpaid"}]}
        return {"payments": []}

    def fake_lnd(network_id, node, topology_value, *arguments):
        calls.append(("lnd", node, arguments))
        if arguments[0] == "listinvoices":
            label = "grand-prix-public-r1-0"
            return {"invoices": [{"memo": label, "state": "open"}]}
        return {"payments": []}

    monkeypatch.setattr(runner, "_cln_rpc", fake_cln)
    monkeypatch.setattr(runner, "_lnd_node_rpc", fake_lnd)
    result = runner.reconcile_public_unknown(topology, state_path=state_path)

    assert result["public_traffic"]["records"][0]["outcome"] == "failed"
    assert [row[0] for row in calls] == [family, family]
    assert all(call[2][0] in {"listinvoices", "listsendpays", "listpayments"} for call in calls)


def test_reconciliation_invalidates_replica_after_payer_channel_goes_onchain(
    tmp_path, monkeypatch
):
    runner = _module()
    topology = _topology()
    topology["traffic"][0].update({
        "payer": "cln-payer",
        "sink": "cln-sink",
    })
    state_path = tmp_path / "state.json"
    state = {
        "schema": runner.SCHEMA,
        "network_id": 9,
        "network_name": runner.DEFAULT_NAME,
        "topology_digest": runner._digest(topology),
        "replica": 1,
        "status": "public_traffic_unknown",
        "public_traffic": {"records": []},
        "events": [
            {"event": "public_payment_unknown", "sequence": 0},
            {"event": "public_timeout_blocks_mined", "blocks": 144},
        ],
    }
    runner._write_json_atomic(state_path, state)

    def fake_cln(_container, *arguments, **_kwargs):
        if arguments[0] == "listinvoices":
            return {"invoices": [{"status": "open"}]}
        if arguments[0] == "listsendpays":
            return {"payments": [{"status": "pending"}]}
        if arguments[0] == "listpeerchannels":
            return {"channels": [{"state": "ONCHAIN"}]}
        raise AssertionError(arguments)

    monkeypatch.setattr(runner, "_cln_rpc", fake_cln)
    result = runner.reconcile_public_unknown(topology, state_path=state_path)

    assert result["status"] == "public_traffic_invalid"
    assert result["public_traffic"]["records"] == []
    event = result["events"][-1]
    assert event["event"] == "public_reconciliation_invalidated"
    assert event["sequence"] == 0
    assert event["reason"] == "payer_channel_onchain"
    assert isinstance(event["at"], int)


def test_timeout_advance_is_unknown_state_only_bounded_and_checkpointed(tmp_path):
    runner = _module()
    topology = _topology()
    state_path = tmp_path / "state.json"
    state = {
        "schema": runner.SCHEMA,
        "network_id": 9,
        "network_name": runner.DEFAULT_NAME,
        "topology_digest": runner._digest(topology),
        "status": "public_traffic_unknown",
        "events": [],
    }
    runner._write_json_atomic(state_path, state)
    bridge = FakeBridge({"mine_blocks": {}})

    result = runner.advance_public_timeout(
        bridge, topology, state_path=state_path, blocks=144
    )

    assert bridge.calls == [("mine_blocks", {"networkId": 9, "blocks": 144})]
    assert result["status"] == "public_traffic_unknown"
    assert result["events"][-1]["event"] == "public_timeout_blocks_mined"
    with pytest.raises(runner.RunnerError, match="2016"):
        runner.advance_public_timeout(
            bridge, topology, state_path=state_path, blocks=2017
        )


def test_timeout_advance_rejects_non_unknown_state_without_mining(tmp_path):
    runner = _module()
    topology = _topology()
    state_path = tmp_path / "state.json"
    runner._write_json_atomic(state_path, {
        "schema": runner.SCHEMA,
        "network_id": 9,
        "topology_digest": runner._digest(topology),
        "status": "public_traffic_partial",
    })
    bridge = FakeBridge()

    with pytest.raises(runner.RunnerError, match="only for an unknown"):
        runner.advance_public_timeout(
            bridge, topology, state_path=state_path, blocks=144
        )
    assert bridge.calls == []
