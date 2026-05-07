from pathlib import Path

from pyln.testing.fixtures import *  # noqa: F401,F403


ROOT = Path(__file__).resolve().parents[1]


def test_cl_revenue_ops_starts_and_serves_status(node_factory, tmp_path):
    plugin_path = ROOT / "cl-revenue-ops.py"

    l1 = node_factory.get_node(
        options={
            "plugin": str(plugin_path),
            "revenue-ops-db-path": str(tmp_path / "revenue_ops.db"),
            "revenue-ops-dry-run": "true",
            "revenue-ops-flow-interval": "3600",
            "revenue-ops-fee-interval": "3600",
            "revenue-ops-rebalance-interval": "3600",
            "revenue-ops-boltz-auto-cycle-enabled": "false",
            "revenue-ops-hive-hints-enabled": "false",
            "revenue-ops-planner-enabled": "false",
        }
    )

    status = l1.rpc.call("revenue-status")
    assert status["status"] == "running"
    assert status["version"]

    plugin_list = l1.rpc.plugin("list")
    assert any("cl-revenue-ops.py" in plugin["name"] for plugin in plugin_list["plugins"])
