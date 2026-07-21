"""Phase 0 pin: the registered RPC surface of cl-revenue-ops.

Adding/removing/renaming an RPC method fails this test until
docs/refactor/phase0/compatibility-catalog.md is updated in the same
commit. Refactor invariant 2: existing primary RPCs remain compatible
unless a separately approved migration changes them.
"""
import pathlib
import re

PLUGIN_PY = pathlib.Path(__file__).resolve().parent.parent / "cl-revenue-ops.py"

EXPECTED_RPC_METHODS = frozenset({
    "revenue-rebalance-cycle", "revenue-status", "revenue-rebalance-debug",
    "revenue-fee-debug", "revenue-fee-cycle", "revenue-fee-authority-status",
    "revenue-analyze",
    "revenue-wake-all", "revenue-capacity-report", "revenue-planner-status",
    "revenue-lnplus-status", "revenue-lnplus-breaker-clear",
    "revenue-lnplus-abandon", "revenue-lnplus-backfill",
    "revenue-planner-candidate-sources", "revenue-planner-candidates",
    "revenue-planner-execute", "revenue-planner-history", "revenue-set-fee",
    "revenue-rebalance", "revenue-profitability", "revenue-history",
    "revenue-ignore", "revenue-unignore", "revenue-list-ignored",
    "revenue-ban", "revenue-unban", "revenue-list-banned", "revenue-policy",
    "revenue-report", "revenue-hot-channel-protection-peers",
    "revenue-config", "revenue-dashboard", "revenue-health",
    "revenue-econ-snapshot", "revenue-econ-reconcile",
    "revenue-econ-cycle", "revenue-profile-preview",
    "revenue-cleanup-closed", "revenue-clear-reservations",
    "revenue-total-cost-budget", "revenue-capex-status",
    "revenue-spend-ledger", "revenue-spend-reserve", "revenue-spend-release",
    "revenue-spend-release-stale", "revenue-spend-settle",
    "revenue-boltz-quote", "revenue-boltz-loop-out", "revenue-boltz-loop-in",
    "revenue-boltz-status", "revenue-boltz-history",
    "revenue-boltz-external-pay-ignores", "revenue-boltz-budget",
    "revenue-boltz-wallet", "revenue-boltz-refund", "revenue-boltz-claim",
    "revenue-boltz-chainswap", "revenue-boltz-withdraw",
    "revenue-boltz-deposit", "revenue-boltz-backup",
    "revenue-boltz-backup-verify", "revenue-boltz-balance-recommendations",
    "revenue-boltz-auto-cycle-status", "revenue-boltz-auto-cycle-run-now",
    "revenue-boltz-balance-cycle", "revenue-boltz-expansion-treasury-status",
    "revenue-boltz-expansion-treasury-recommendations",
    "revenue-boltz-expansion-treasury-cycle",
})


def _registered_methods():
    text = PLUGIN_PY.read_text()
    return frozenset(re.findall(r'@plugin\.method\(\s*"([a-z-]+)"', text))


def test_rpc_surface_matches():
    actual = _registered_methods()
    assert actual == EXPECTED_RPC_METHODS, (
        "Registered RPC surface changed — update this pin AND "
        "docs/refactor/phase0/compatibility-catalog.md together.\n"
        f"added={sorted(actual - EXPECTED_RPC_METHODS)} "
        f"removed={sorted(EXPECTED_RPC_METHODS - actual)}"
    )


def test_expected_count():
    # 64 at baseline 5e8f747; + econ-shadow diagnostics (no compat
    # promise yet): revenue-econ-snapshot (Phase 1), revenue-econ-
    # reconcile (Phase 2B), revenue-econ-cycle (Workstream H shadow),
    # revenue-profile-preview (PR 8, read-only risk-profile diff), and
    # revenue-fee-authority-status (Python fee-authority handoff status).
    assert len(EXPECTED_RPC_METHODS) == 69
