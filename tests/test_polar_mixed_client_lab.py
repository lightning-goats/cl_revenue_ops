import importlib.util
import sys
from pathlib import Path

import pytest


def load_lab():
    path = Path(__file__).resolve().parents[1] / "tools" / "polar_mixed_client_lab.py"
    spec = importlib.util.spec_from_file_location("polar_mixed_client_lab", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_role_renames_assigns_stable_mixed_client_roles():
    lab = load_lab()
    network = {
        "nodes": {
            "lightning": [
                {"name": "alice", "implementation": "LND"},
                {"name": "bob", "implementation": "LND"},
                {"name": "carol", "implementation": "c-lightning"},
                {"name": "dave", "implementation": "c-lightning"},
                {"name": "erin", "implementation": "c-lightning"},
                {"name": "frank", "implementation": "c-lightning"},
                {"name": "grace", "implementation": "LND"},
            ]
        }
    }

    assert lab.role_renames(network) == [
        ("carol", "revenue-node"),
        ("dave", "cln-competitor"),
        ("erin", "cln-payer"),
        ("frank", "cln-sink"),
        ("alice", "lnd-competitor"),
        ("bob", "lnd-payer"),
        ("grace", "lnd-sink"),
    ]


def test_role_renames_fails_closed_on_incomplete_topology():
    lab = load_lab()

    with pytest.raises(lab.PolarMcpError, match="expected 4 c-lightning nodes"):
        lab.role_renames({"nodes": {"lightning": []}})


def test_traffic_rejects_nonpositive_inputs_without_mcp_calls():
    lab = load_lab()

    with pytest.raises(ValueError, match="must be positive"):
        lab.run_deterministic_traffic(object(), 1, 0, 10, 0)


def test_payment_bridge_error_is_unknown_and_never_retried():
    lab = load_lab()

    class Bridge:
        def __init__(self):
            self.calls = []

        def call(self, tool, arguments):
            self.calls.append((tool, arguments))
            if tool == "create_invoice":
                return {"invoice": "bolt11-redacted"}
            raise lab.PolarMcpError("HTTP 500 after dispatch")

    bridge = Bridge()
    with pytest.raises(lab.PolarTrafficError) as caught:
        lab.run_deterministic_traffic(
            bridge, 4, 10, 50_000, 0, (("lnd-payer", "lnd-sink"),)
        )

    assert [tool for tool, _args in bridge.calls] == [
        "create_invoice",
        "pay_invoice",
    ]
    assert caught.value.completed_records == []
    assert caught.value.uncertain_operation == {
        "round": 0,
        "payer": "lnd-payer",
        "sink": "lnd-sink",
        "amount_sats": 50_000,
        "payment_outcome": "unknown_do_not_retry",
        "invoice_created": True,
        "error": "HTTP 500 after dispatch",
    }
    assert "bolt11" not in str(caught.value.uncertain_operation)


def test_traffic_lane_selector_supports_reverse_single_client_lane():
    lab = load_lab()

    assert lab.select_traffic_lanes("reverse", "lnd") == (
        ("lnd-sink", "lnd-payer"),
    )


def test_traffic_lane_selector_supports_both_directions():
    lab = load_lab()

    assert lab.select_traffic_lanes("both", "cln") == (
        ("cln-payer", "cln-sink"),
        ("cln-sink", "cln-payer"),
    )


def test_both_direction_traffic_seeds_explicit_return_fee_buffer():
    lab = load_lab()

    assert lab.traffic_batches("both", "lnd", 20_000, 100) == (
        ((("lnd-payer", "lnd-sink"),), 20_100),
        ((("lnd-sink", "lnd-payer"),), 20_000),
    )


def test_reverse_fee_buffer_must_be_nonnegative():
    lab = load_lab()

    with pytest.raises(ValueError, match="must be nonnegative"):
        lab.traffic_batches("both", "all", 20_000, -1)


def test_default_reverse_buffer_covers_two_million_sat_channel_reserve():
    lab = load_lab()

    assert lab.DEFAULT_REVERSE_FEE_BUFFER_SATS > 20_000


def test_required_channel_check_requires_matching_peer_capacity_and_state():
    lab = load_lab()

    assert lab.has_required_channel(
        [{"pubkey": "destination", "capacity": "2000000", "status": "Open"}],
        "destination",
        2_000_000,
    )
    assert not lab.has_required_channel(
        [{"pubkey": "destination", "capacity": "2000000", "status": "Closed"}],
        "destination",
        2_000_000,
    )


def test_required_channel_check_accepts_pending_channel_for_retry_safety():
    lab = load_lab()

    assert lab.has_required_channel(
        [{"pubkey": "destination", "capacity": "2000000", "status": "Opening"}],
        "destination",
        2_000_000,
    )


def test_create_lab_refuses_running_network_before_creating_an_orphan():
    lab = load_lab()

    class Bridge:
        def __init__(self):
            self.calls = []

        def call(self, tool, arguments):
            self.calls.append((tool, arguments))
            if tool == "list_networks":
                return {
                    "networks": [
                        {"id": 1, "name": "existing-lab", "status": "Started"},
                    ]
                }
            raise AssertionError(f"unexpected mutation: {tool}")

    bridge = Bridge()
    with pytest.raises(lab.PolarMcpError, match=r"existing-lab \(id=1\)"):
        lab.create_lab(bridge, "new-lab", "test")

    assert bridge.calls == [("list_networks", {})]
