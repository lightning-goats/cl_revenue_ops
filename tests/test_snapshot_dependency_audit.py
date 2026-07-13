"""PR 2 (gap-closure Phase B): snapshot-dependency enforcement pin.

Mirrors the scan in docs/refactor/phase0/snapshot-dependency-audit.md.
Every mutable-source read in a policy module is counted per category;
the counts are pinned. A new read anywhere trips this test and must be
classified in the audit doc before the pin is updated. As PRs 3a-3e
migrate improper reads onto the canonical snapshot, these counts go
DOWN — update the pin with each migration, never upward without an
audit-doc entry.
"""
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

MODULES = {
    "fee": "modules/fee_controller.py",
    "rebalance": "modules/rebalance_engine_v2.py",
    "planner": "modules/capacity_planner.py",
    "boltz": "modules/boltz_manager.py",
    "lnplus": "modules/lnplus_swaps.py",
    "admission": "modules/admission_policy.py",
    "protection": "modules/protection_service.py",
    "profitability": "modules/profitability_analyzer.py",
    "treasury": "modules/capex_budget.py",
}

PATTERNS = {
    "analyzer_cache": (
        r"self\.(profitability|flow_analyzer|flow)\.\w+\("
        r"|(?<!self\.)\bprofitability\.\w+\("
        r"|\bflow_analyzer\.\w+\("
    ),
    "live_rpc": (
        r"data_service\.(get|list)\w*\("
        r"|plugin\.rpc\.\w+\("
        r"|self\.rpc\.\w+\("
    ),
    "database": r"database\.\w*get\w*\(|self\.database\.\w+\(",
    "wall_clock": r"time\.time\(\)",
}

# Pinned 2026-07-13 (audit baseline). Keys: (module, category) -> count.
PINNED_COUNTS = {
    ("fee", "analyzer_cache"): 1,
    ("fee", "live_rpc"): 9,
    ("fee", "database"): 27,
    ("fee", "wall_clock"): 38,
    ("rebalance", "analyzer_cache"): 0,
    ("rebalance", "live_rpc"): 14,
    ("rebalance", "database"): 18,
    ("rebalance", "wall_clock"): 10,
    ("planner", "analyzer_cache"): 6,
    # 3b: 24 -> 20 (dead _has_direct_peer_channel/_is_peer_connected
    # removed; 4 live-RPC sites gone)
    ("planner", "live_rpc"): 20,
    ("planner", "database"): 11,
    ("planner", "wall_clock"): 11,
    ("boltz", "analyzer_cache"): 0,
    ("boltz", "live_rpc"): 7,
    ("boltz", "database"): 0,
    ("boltz", "wall_clock"): 5,
    ("lnplus", "analyzer_cache"): 0,
    ("lnplus", "live_rpc"): 12,
    ("lnplus", "database"): 0,
    ("lnplus", "wall_clock"): 12,
    ("admission", "analyzer_cache"): 0,
    ("admission", "live_rpc"): 0,
    ("admission", "database"): 0,
    ("admission", "wall_clock"): 0,
    ("protection", "analyzer_cache"): 0,
    ("protection", "live_rpc"): 0,
    ("protection", "database"): 0,
    ("protection", "wall_clock"): 0,
    ("profitability", "analyzer_cache"): 0,
    ("profitability", "live_rpc"): 5,
    ("profitability", "database"): 28,
    ("profitability", "wall_clock"): 24,
    ("treasury", "analyzer_cache"): 0,
    ("treasury", "live_rpc"): 0,
    ("treasury", "database"): 4,
    ("treasury", "wall_clock"): 1,
}


def _count(path: pathlib.Path, pattern: str) -> int:
    rx = re.compile(pattern)
    return sum(
        1
        for line in path.read_text().split("\n")
        if rx.search(line) and not line.strip().startswith("#")
    )


@pytest.mark.parametrize("module,category",
                         sorted(PINNED_COUNTS),
                         ids=lambda v: str(v))
def test_mutable_read_counts_pinned(module, category):
    path = REPO / MODULES[module]
    actual = _count(path, PATTERNS[category])
    pinned = PINNED_COUNTS[(module, category)]
    assert actual == pinned, (
        f"{module} ({MODULES[module]}) has {actual} '{category}' read "
        f"sites, pinned at {pinned}. If you ADDED a mutable-source read "
        f"to a policy module, classify it in docs/refactor/phase0/"
        f"snapshot-dependency-audit.md first. If a snapshot-migration PR "
        f"REMOVED reads, lower the pin."
    )


def test_admission_and_protection_stay_pure():
    """The two already-pure policy modules must never regress."""
    for module in ("admission", "protection"):
        for category in PATTERNS:
            assert PINNED_COUNTS[(module, category)] == 0
            assert _count(REPO / MODULES[module], PATTERNS[category]) == 0


def test_synthetic_snapshot_ids_still_present_until_migrated():
    """Documents the CURRENT (improper) snapshot_id labels the migration
    replaces. When a PR threads real snapshot ids, remove its entry here
    and check the box in the audit doc's migration work list."""
    expected = {
        # 3a DONE: rebalance threads real snapshot refs; the synthetic
        # label survives only as the documented fail-open FALLBACK.
        "modules/rebalance_engine_v2.py": 'or f"rebalance-cycle-',
        # 3b DONE: planner threads real snapshot refs; synthetic label
        # survives only as the fail-open fallback.
        "modules/capacity_planner.py": 'or f"planner-cycle-',
        # 3c DONE: Boltz threads real snapshot refs; synthetic label
        # survives only as the fail-open fallback.
        "modules/boltz_manager.py": 'or f"boltz-swap-',
        "modules/lnplus_swaps.py": 'snapshot_id=f"lnplus-swap-',
    }
    for rel, marker in expected.items():
        assert marker in (REPO / rel).read_text(), (
            f"{rel}: synthetic snapshot_id marker gone — if this was the "
            f"snapshot migration, update this pin and the audit doc")
