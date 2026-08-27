import importlib.util
import sys
from pathlib import Path

import pytest


def load_campaign():
    tools = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools))
    path = tools / "polar_endurance_campaign.py"
    spec = importlib.util.spec_from_file_location("polar_endurance_campaign", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def snapshot(revenue=0, cln=0, lnd=0):
    return {
        "routers": {
            "revenue-node": {"settled_count": revenue, "fee_msat": revenue * 2, "volume_msat": revenue * 10},
            "cln-competitor": {"settled_count": cln, "fee_msat": cln * 2, "volume_msat": cln * 10},
            "lnd-competitor": {"settled_count": lnd, "fee_msat": lnd * 2, "volume_msat": lnd * 10},
        }
    }


def test_network_id_and_csv_values_fail_closed():
    campaign = load_campaign()
    assert campaign.positive_network_id("4") == 4
    assert campaign.csv_positive_ints("5,15,35") == (5, 15, 35)
    with pytest.raises(Exception, match="positive integer"):
        campaign.positive_network_id("4;touch /tmp/bad")
    with pytest.raises(Exception, match="positive"):
        campaign.csv_positive_ints("5,0")


def test_phase_summary_is_client_specific_and_attributes_all_routes():
    campaign = load_campaign()
    records = [
        {"payment": {"success": True}},
        {"payment": {"success": True}},
        {"payment": {"success": True}},
    ]
    result = campaign.phase_summary(
        "cln", 25, records, snapshot(), snapshot(revenue=1, cln=2)
    )
    assert result["passed"] is True
    assert result["revenue_route_share"] == pytest.approx(1 / 3, abs=1e-6)
    assert result["revenue_fee_msat"] == 2


def test_router_counter_regression_is_rejected():
    campaign = load_campaign()
    with pytest.raises(campaign.CampaignError, match="regressed"):
        campaign.router_delta(snapshot(revenue=2), snapshot(revenue=1))


def test_safe_final_state_requires_every_cleanup_rail():
    campaign = load_campaign()
    healthy = {
        "module_health": {
            "config": {"config": {"paused": True, "daily_budget_sats": 0}},
            "budget": {"reserved_24h_sats": 0},
            "econ_reconcile": {"divergences": []},
        }
    }
    assert campaign.safe_final_state(healthy, dry_run=True) is True
    assert campaign.safe_final_state(healthy, dry_run=False) is False
    healthy["module_health"]["config"]["config"]["paused"] = False
    assert campaign.safe_final_state(healthy, dry_run=True) is False


def test_live_controller_restart_precedes_budget_and_unpause():
    campaign = load_campaign()

    class Node:
        def __init__(self):
            self.calls = []

        def set_dry_run(self, enabled):
            self.calls.append(("dry_run", enabled))

        def set_config(self, key, value):
            self.calls.append((key, value))

    node = Node()
    campaign.enable_live_controller(node, 1_000)
    assert node.calls == [
        ("dry_run", False),
        ("daily_budget_sats", 1_000),
        ("paused", False),
    ]


def test_completed_resume_does_not_enter_live_mode():
    campaign = load_campaign()

    class Node:
        def set_dry_run(self, _enabled):
            raise AssertionError("completed campaign must not restart live")

        def set_config(self, _key, _value):
            raise AssertionError("completed campaign must not mutate live controls")

    assert campaign.enable_live_controller_if_needed(Node(), 1_000, 60, 60) is False


@pytest.mark.parametrize(
    ("completed", "target", "max_new", "expected"),
    [
        (0, 60, 0, 60),
        (0, 60, 10, 10),
        (10, 60, 10, 20),
        (55, 60, 10, 60),
        (60, 60, 10, 60),
    ],
)
def test_endurance_epoch_limit_is_resumable(completed, target, max_new, expected):
    campaign = load_campaign()
    assert campaign.endurance_epoch_limit(completed, target, max_new) == expected


def test_revenue_policy_capture_rejects_malformed_rows(monkeypatch):
    campaign = load_campaign()
    node = campaign.RevenueNode(4)
    monkeypatch.setattr(node, "rpc", lambda *_args: {"channels": [{}]})
    with pytest.raises(campaign.CampaignError, match="active local policy"):
        node.policies()


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"configs": {"revenue-ops-dry-run": {"value_str": "true"}}}, True),
        ({"configs": {"revenue-ops-dry-run": {"value_bool": False}}}, False),
    ],
)
def test_dry_run_readback_supports_cln_shapes(monkeypatch, payload, expected):
    campaign = load_campaign()
    node = campaign.RevenueNode(4)
    monkeypatch.setattr(node, "rpc", lambda *_args: payload)
    assert node.dry_run() is expected


def test_existing_output_requires_explicit_resume(tmp_path, monkeypatch):
    campaign = load_campaign()
    output = tmp_path / "campaign.json"
    output.write_text("{}", encoding="utf-8")
    args = type("Args", (), {
        "bridge_url": "http://polar.invalid",
        "network_id": 4,
        "output": output,
        "resume": False,
    })()
    monkeypatch.setattr(campaign.PolarMcp, "health", lambda _self: {})
    monkeypatch.setattr(campaign.RevenueNode, "policies", lambda _self: (
        campaign.ChannelPolicy("1x1x1", 1, 10),
    ))
    with pytest.raises(campaign.CampaignError, match="already exists"):
        campaign.run_campaign(args)
