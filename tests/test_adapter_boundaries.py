"""Phase 3E: adapter-isolation boundary guard (Workstream G).

The adapter set is now EXPLICIT:
- CLN adapter: modules/data_service.py (typed wrappers, cache coherence)
  plus its execution arm modules/rebalance_native_executor_v2.py (the
  timeout-managed sendpay pipeline — part of the adapter boundary, not
  a bypass).
- Boltz adapter: modules/boltz_manager.py (boltzcli subprocess).

Policy/decision modules must stay pure of the execution surface. The
mutating-verb inventory itself is pinned by
tests/test_mutation_path_inventory.py; this guard pins the LAYERING.
"""
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent

# Pure decision modules: no RPC handles, no subprocess, no HTTP.
PURE_POLICY_MODULES = [
    "modules/classification.py",
    "modules/admission_policy.py",
    "modules/protection_service.py",
    "modules/rebalance_modes.py",
    "modules/rebalance_planner_v2.py",
    "modules/rebalance_state_v2.py",
    "modules/econ_types.py",
    "modules/econ_intents.py",
    "modules/econ_arbiter.py",
    "modules/econ_reconcile.py",
    "modules/reason_codes.py",
    "modules/cycle_context.py",
]

FORBIDDEN_IN_POLICY = (
    ".rpc.", "subprocess", "urllib", "requests.", "boltzcli",
)


def test_policy_modules_never_touch_execution_surfaces():
    for module in PURE_POLICY_MODULES:
        source = (REPO / module).read_text()
        for forbidden in FORBIDDEN_IN_POLICY:
            assert forbidden not in source, f"{module}: {forbidden}"


def test_cln_adapter_does_not_retain_lnplus_only_surfaces():
    source = (REPO / "modules" / "data_service.py").read_text()
    assert "def connect_peer(" not in source
    assert "def sign_message(" not in source
    assert not (REPO / "modules" / "lnplus_swaps.py").exists()


def test_external_adapters_do_not_leak_wire_formats():
    """The Boltz manager may parse boltzcli output; nothing outside that
    adapter may invoke boltzcli or the retired LN+ HTTP base URL."""
    adapter_files = {"boltz_manager.py"}
    for path in sorted((REPO / "modules").glob("*.py")):
        if path.name in adapter_files:
            continue
        source = path.read_text()
        # Scope to actual wire usage, not prose mentions in comments.
        assert '"boltzcli' not in source and "'boltzcli" not in source, \
            path.name
        assert "https://lightningnetwork.plus" not in source, path.name
