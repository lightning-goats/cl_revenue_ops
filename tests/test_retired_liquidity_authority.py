"""Characterize the executor roots approved for retirement."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "cl-revenue-ops.py"
RETIRED_MODULES = {
    "lnplus": ROOT / "modules/lnplus_swaps.py",
    "boltz": ROOT / "modules/boltz_manager.py",
    "planner": ROOT / "modules/capacity_planner.py",
    "demand_flow": ROOT / "modules/demand_flow.py",
    "protection": ROOT / "modules/protection_service.py",
}


def _registered_rpcs(prefix: str) -> set[str]:
    source = PLUGIN.read_text(encoding="utf-8")
    return {
        name
        for name in re.findall(r'@plugin\.method\(\s*"([a-z-]+)"', source)
        if name.startswith(prefix)
    }


def _option_literals(prefix: str) -> set[str]:
    sources = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in ("cl-revenue-ops.py", "modules/config.py")
    )
    return set(re.findall(rf'["\']({re.escape(prefix)}[a-z0-9-]+)["\']', sources))


def test_remaining_retiring_module_roots_are_present_at_characterization_checkpoint():
    expected = {"boltz", "planner", "demand_flow", "protection"}
    actual = {
        name for name, path in RETIRED_MODULES.items()
        if name != "lnplus" and path.is_file()
    }
    assert actual == expected


def test_lnplus_module_and_import_are_absent():
    assert not RETIRED_MODULES["lnplus"].exists()
    assert "modules.lnplus_swaps" not in PLUGIN.read_text(encoding="utf-8")


def test_remaining_retired_rpc_and_option_families_are_nonempty():
    assert _registered_rpcs("revenue-boltz-")
    assert _registered_rpcs("revenue-planner-")
    assert _option_literals("revenue-ops-boltz-")
    assert _option_literals("revenue-ops-planner-")


def test_lnplus_rpc_and_option_families_are_absent():
    assert not _registered_rpcs("revenue-lnplus-")
    assert not _option_literals("revenue-ops-lnplus-")


def test_direct_mutation_verbs_have_the_expected_owner():
    expected = {
        "boltz": ("subprocess.run", "boltzcli"),
        "planner": ("fund_channel", "close_channel", "diagnostic_rebalance"),
    }
    for owner, symbols in expected.items():
        source = RETIRED_MODULES[owner].read_text(encoding="utf-8")
        for symbol in symbols:
            assert symbol in source, f"{owner} lost audited mutation symbol {symbol}"


def test_lnplus_network_signing_and_state_writers_are_absent():
    sources = {
        "plugin": PLUGIN.read_text(encoding="utf-8"),
        "database": (ROOT / "modules/database.py").read_text(encoding="utf-8"),
        "data_service": (ROOT / "modules/data_service.py").read_text(encoding="utf-8"),
    }
    for symbol in (
        "LNPlusClient", "SwapEvaluator", "SwapLifecycle",
        "revenue-lnplus-", "lnplus_record_swap", "lnplus_update_swap",
        "lnplus_bump_peer", "lnplus_prune_terminal",
    ):
        assert all(symbol not in source for source in sources.values())
    assert "def connect_peer(" not in sources["data_service"]
    assert "def sign_message(" not in sources["data_service"]


def test_historical_schema_and_generic_ledger_policy_helpers_are_present():
    source = (ROOT / "modules/database.py").read_text(encoding="utf-8")
    for ddl in (
        "CREATE TABLE IF NOT EXISTS lnplus_swaps",
        "CREATE TABLE IF NOT EXISTS lnplus_peers",
        "CREATE TABLE IF NOT EXISTS planner_actions",
        "CREATE TABLE IF NOT EXISTS spend_reservations",
        "CREATE TABLE IF NOT EXISTS spend_events",
        "CREATE TABLE IF NOT EXISTS peer_policies",
    ):
        assert ddl in source
    for helper in (
        "def reserve_spend(", "def release_spend_reservation(",
        "def get_spend_ledger_summary(", "def get_policy(", "def upsert_policy(",
    ):
        assert helper in source
