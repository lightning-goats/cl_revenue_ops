"""
Regression guard: ensure removed architectures do not quietly return.

These tests verify that hive integration, legacy standalone fee algorithms
(AIMD, Hill Climbing, discrete Thompson), and removed feature flags remain
absent from the codebase.  The current architecture is DTS+PID only.
"""
import ast
import inspect
import os
import re
from pathlib import Path
import subprocess


def _source_files():
    """Yield all .py source files under modules/ and the main plugin."""
    root = os.path.dirname(os.path.dirname(__file__))
    for name in os.listdir(os.path.join(root, "modules")):
        if name.endswith(".py"):
            yield os.path.join(root, "modules", name)
    yield os.path.join(root, "cl-revenue-ops.py")


ROOT = Path(__file__).resolve().parents[1]
FINAL_RUNTIME_FILES = [
    path for path in sorted((ROOT / "modules").glob("*.py"))
    if path.name not in {
        "lnplus_swaps.py", "boltz_manager.py", "capacity_planner.py",
        "demand_flow.py", "protection_service.py",
    }
] + [ROOT / "cl-revenue-ops.py"]
REBALANCE_BOUNDARY_FILES = [
    ROOT / "modules" / "rebalance_engine_v2.py",
    ROOT / "modules" / "rebalance_router_v2.py",
    ROOT / "modules" / "rebalance_router_v3.py",
]


def _is_forbidden_coordinator_import(module_name):
    """Match retired coordinator names as import-path components."""
    components = re.split(r"[._-]+", str(module_name or "").lower())
    return bool({"hive", "mycelium", "sling"}.intersection(components))


class TestNoHiveReintroduction:
    """Hive integration must not return."""

    def test_no_tracked_hive_bridge_module(self):
        result = subprocess.run(
            ["git", "ls-files", "modules/hive_bridge.py"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.stdout.strip() == ""

    def test_no_hive_bridge_import_in_source(self):
        for path in _source_files():
            source = open(path).read()
            assert "from .hive_bridge import" not in source, f"hive_bridge import found in {path}"
            assert "from modules.hive_bridge" not in source, f"hive_bridge import found in {path}"

    def test_no_fee_strategy_hive(self):
        from modules.policy_manager import FeeStrategy
        assert not hasattr(FeeStrategy, "HIVE")

    def test_no_is_hive_peer_method(self):
        from modules.policy_manager import PolicyManager
        assert not hasattr(PolicyManager, "is_hive_peer")

    def test_no_tracked_hive_modules(self):
        """Standalone Phases 0-5 (2026-07-09): the dedicated hive modules
        must stay deleted."""
        removed = [
            "modules/hive_hints.py",
            "modules/hive_router.py",
            "modules/hive_runtime.py",
            "modules/rebalance_hive_router.py",
            "modules/rebalance_coordination_overlay.py",
        ]
        result = subprocess.run(
            ["git", "ls-files", *removed],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.stdout.strip() == ""

    def test_no_hive_references_in_runtime_source(self):
        """The Phase 5 grep gate, pinned: modules/ and cl-revenue-ops.py
        carry no hive/mycelium code. Historical changelog comments in the
        plugin header and explanatory retirement comments are exempt."""
        import re

        pattern = re.compile(r"hive|mycelium", re.IGNORECASE)
        for path in _source_files():
            for lineno, line in enumerate(open(path).readlines(), 1):
                if not pattern.search(line):
                    continue
                stripped = line.strip()
                # 'archived'/'archive' contains 'hive' — incidental.
                if pattern.search(re.sub(r"archiv\w*", "", stripped, flags=re.I)) is None:
                    continue
                assert stripped.startswith("#") or stripped.startswith("'"), (
                    f"non-comment hive/mycelium reference in {path}:{lineno}: "
                    f"{stripped!r}"
                )

    def test_no_hive_symbols_on_core_classes(self):
        from modules.fee_controller import FeeController
        from modules.rebalance_engine_v2 import RebalanceEngine
        from modules.rebalance_types_v2 import PairCandidate
        from modules.config import Config

        for cls, attrs in [
            (FeeController, ["_get_hive_fee_bias", "_get_temporal_fee_adjustment",
                             "_is_fleet_sibling", "_maybe_reseed_skewed_prior",
                             "_check_hive_member_fee"]),
            (RebalanceEngine, ["_apply_metabolic_rebalance_bias",
                               "_apply_immune_rebalance_bias",
                               "_get_hive_rebalance_bias"]),
        ]:
            for attr in attrs:
                assert not hasattr(cls, attr), f"{cls.__name__}.{attr} returned"

        for field_name in ("hive_source_rebalance_bias", "metabolic_rebalance_bias",
                           "immune_rebalance_bias"):
            assert field_name not in PairCandidate.__dataclass_fields__

        cfg_fields = Config.__dataclass_fields__
        for key in ("hive_hints_enabled", "hive_zero_fee_stale_grace_seconds",
                    "hive_equalization_enabled", "hive_push_enabled",
                    "hive_rebalance_bootstrap_budget_sats",
                    "base_fee_msat_intra_fleet", "base_fee_msat_non_hive",
                    "fee_ppm_intra_fleet", "lnplus_fleet_pubkeys",
                    "rebalance_coordination_reserved_slots"):
            assert key not in cfg_fields, f"Config.{key} returned"


class TestNoLegacyFeeAlgorithms:
    """Legacy standalone fee algorithms (AIMD, Hill Climbing, discrete Thompson) must not return."""

    def test_no_aimd_defense_state(self):
        from modules import fee_controller
        assert not hasattr(fee_controller, "AIMDDefenseState")

    def test_no_thompson_sampling_state(self):
        from modules import fee_controller
        assert not hasattr(fee_controller, "ThompsonSamplingState")

    def test_no_historical_response_curve(self):
        from modules import fee_controller
        assert not hasattr(fee_controller, "HistoricalResponseCurve")

    def test_no_scarcity_multiplier(self):
        from modules import fee_controller
        assert not hasattr(fee_controller, "calculate_scarcity_multiplier")

    def test_no_legacy_reason_codes(self):
        from modules.fee_controller import FeeReasonCode
        removed = ["THOMPSON_COLD_START", "THOMPSON_AIMD_DEFENSE", "SCARCITY",
                    "ANCHOR_BLEND", "YOUNG_CHANNEL_CAP", "HIGH_VOLATILITY_REDUCE",
                    "HIGH_FAILURE_CONSERVATIVE", "POLICY_HIVE"]
        for code in removed:
            assert not hasattr(FeeReasonCode, code), f"Removed reason code {code} found"

    def test_no_legacy_feature_flags(self):
        from modules.fee_controller import FeeController
        removed_flags = ["ENABLE_THOMPSON_AIMD", "ENABLE_SIMPLIFIED_FEE_PATH",
                         "ENABLE_DTS_PID", "ENABLE_BALANCE_FLOOR",
                         "ENABLE_SATURATION_FLOOR", "ENABLE_SATURATION_DRAIN",
                         "ENABLE_COLD_START", "ENABLE_HIVE_INTELLIGENCE",
                         "ENABLE_HIVE_COORDINATION", "ENABLE_PHEROMONE_BIAS",
                         "ENABLE_COMPETITION_AVOIDANCE"]
        for flag in removed_flags:
            assert not hasattr(FeeController, flag), f"Removed flag {flag} found"

    def test_dts_pid_sample_is_default_reason(self):
        from modules.fee_controller import FeeReasonCode
        assert hasattr(FeeReasonCode, "DTS_PID_SAMPLE")


class TestRebalanceDataServiceBoundary:
    """The active rebalance path must stay behind DataService."""

    def test_hot_path_constructors_accept_data_service(self):
        from modules.rebalance_engine_v2 import RebalanceEngine
        from modules.rebalance_router_v2 import RebalanceRouter
        from modules.rebalance_router_v3 import RebalanceRouterV3

        assert "data_service" in inspect.signature(RebalanceEngine.__init__).parameters
        assert "data_service" in inspect.signature(RebalanceRouter.__init__).parameters
        assert "data_service" in inspect.signature(RebalanceRouterV3.__init__).parameters

    def test_hot_path_modules_reference_data_service_wrappers(self):
        expectations = {
            "modules/rebalance_engine_v2.py": [
                "self._data_service.get_askrene_layers",
                "self._data_service.get_node_id",
                "self._data_service.get_peer_channels",
            ],
            "modules/rebalance_router_v2.py": [
                "self.data_service.get_peer_channels",
                "self.data_service.get_channels",
                "self.data_service.get_configs",
                "self.data_service.get_route",
            ],
            "modules/rebalance_router_v3.py": [
                "self.data_service.get_askrene_layers",
                "self.data_service.get_routes",
                "self.data_service.askrene_create_layer",
                "self.data_service.askrene_update_channel",
                "self.data_service.askrene_remove_layer",
            ],
        }

        for rel, required_snippets in expectations.items():
            source = (ROOT / rel).read_text()
            for snippet in required_snippets:
                assert snippet in source, f"{snippet} missing from {rel}"

    def test_main_wires_data_service_into_rebalance_engine(self):
        source = (ROOT / "cl-revenue-ops.py").read_text()
        tree = ast.parse(source, filename="cl-revenue-ops.py")

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "RebalanceEngine":
                continue
            for keyword in node.keywords:
                if keyword.arg == "data_service":
                    assert isinstance(keyword.value, ast.Name)
                    assert keyword.value.id == "data_service"
                    return
            raise AssertionError("RebalanceEngine call missing data_service=...")

        raise AssertionError("RebalanceEngine call not found in cl-revenue-ops.py")


class TestForwardArchiveReadOnlyBoundary:
    def test_forward_history_rpc_only_calls_archive_history(self):
        source = (ROOT / "cl-revenue-ops.py").read_text()
        tree = ast.parse(source, filename="cl-revenue-ops.py")
        function = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "revenue_forward_history"
            ),
            None,
        )
        assert function is not None
        assert function.args.kwarg is None

        def dotted_name(node):
            parts = []
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                parts.append(node.id)
            return ".".join(reversed(parts))

        database_calls = sorted(
            dotted_name(node.func)
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and dotted_name(node.func).startswith("database.")
        )
        assert database_calls == ["database.forward_archive.history"]


class TestNoSlingDependency:
    """Sling must not be an active runtime dependency."""

    def test_runtime_source_has_no_sling_references(self):
        for path in _source_files():
            source = Path(path).read_text(encoding="utf-8").lower()
            assert "sling" not in source, f"runtime Sling reference found in {path}"

    def test_dependency_files_have_no_sling_references(self):
        for path in (ROOT / "requirements.txt", ROOT / "pyproject.toml"):
            source = path.read_text(encoding="utf-8").lower()
            assert "sling" not in source, f"dependency Sling reference found in {path}"


class TestFinalRuntimeAuthorityBoundary:
    def test_forbidden_import_match_uses_module_components_not_word_substrings(self):
        assert _is_forbidden_coordinator_import("modules.hive_router") is True
        assert _is_forbidden_coordinator_import("modules.mycelium_runtime") is True
        assert _is_forbidden_coordinator_import("modules.sling") is True
        assert _is_forbidden_coordinator_import("modules.forward_archive") is False

    def test_no_sling_or_coordinator_imports_in_post_removal_source_set(self):
        for path in FINAL_RUNTIME_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name.lower() for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.append(str(node.module or "").lower())
            for module_name in imported:
                assert not _is_forbidden_coordinator_import(module_name), (
                    f"forbidden runtime import {module_name} returned in {path}"
                )
