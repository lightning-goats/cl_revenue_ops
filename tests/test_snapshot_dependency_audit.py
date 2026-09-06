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
    "admission": "modules/admission_policy.py",
    "profitability": "modules/profitability_analyzer.py",
    "treasury": "modules/capex_budget.py",
}

PATTERNS = {
    "analyzer_cache": (
        r"self\.(profitability|flow_analyzer|flow)\.\w+\("
        r"|(?<!self\.)\bprofitability\.\w+\("
        r"|\bflow_analyzer\.\w+\("
        # fee_controller's guarded canonical refresh first binds
        # self.profitability to a local variable; keep that mutable-source
        # read visible to the dependency pin instead of silently lowering it.
        r"|\banalyzer\.get_profitability\("
    ),
    "live_rpc": (
        r"data_service\.(get|list)\w*\("
        r"|plugin\.rpc\.\w+\("
        r"|self\.rpc\.\w+\("
    ),
    "database": r"database\.\w*get\w*\(|self\.database\.\w+\(",
    "wall_clock": r"time\.time\(\)|\bdecision_now\(",
}

# Pinned 2026-07-13 (audit baseline). Keys: (module, category) -> count.
PINNED_COUNTS = {
    ("fee", "analyzer_cache"): 1,
    ("fee", "live_rpc"): 9,
    # +3 bounded-acquisition reads: cold-lane probe/idle qualification and
    # one atomic episode volume/count/minimum aggregate. All are
    # replay-captured evidence.
    # +1 coherent earned-fee/label-shortfall read (captured historical window).
    ("fee", "database"): 31,
    # Fee replay clock seam: 30 effective reads now use decision_now(),
    # including one notification-safe yield-inventory wake cooldown; 10
    # cache-TTL reads remain raw and one dead contextual read was removed.
    ("fee", "wall_clock"): 39,
    ("rebalance", "analyzer_cache"): 0,
    ("rebalance", "live_rpc"): 14,
    # 18 -> 20 (audit 2026-08-01 wave2): _recover_missing_pending_row
    # retries the 'pending' history insert (record_rebalance +
    # update_rebalance_result) when the original insert failed and the
    # payment ended payment_pending — execution/reconciliation writes, not
    # decision reads; classified in the audit doc.
    ("rebalance", "database"): 20,
    # 10 -> 11 (2026-08-01, task 26/78): reconcile_pending_settlements
    # reads time.time() once per sweep to compute row AGE for the stale-hold
    # escalation. Measurement of elapsed time, not a policy input; classified
    # in docs/refactor/phase0/snapshot-dependency-audit.md.
    # 11 -> 13 (audit 2026-08-01 wave2 FIX 1): cost-row timestamps in the
    # atomic success settlement (_settle_rebalance_success and the
    # reconcile atomic branch) — execution timestamping, same class as the
    # legacy sites they mirror; classified in the audit doc.
    ("rebalance", "wall_clock"): 13,
    ("admission", "analyzer_cache"): 0,
    ("admission", "live_rpc"): 0,
    ("admission", "database"): 0,
    ("admission", "wall_clock"): 0,
    ("profitability", "analyzer_cache"): 0,
    ("profitability", "live_rpc"): 5,
    ("profitability", "database"): 27,
    ("profitability", "wall_clock"): 23,
    ("treasury", "analyzer_cache"): 0,
    ("treasury", "live_rpc"): 0,
    ("treasury", "database"): 3,
    ("treasury", "wall_clock"): 0,
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


def test_admission_stays_pure():
    """The retained pure admission policy module must never regress."""
    for module in ("admission",):
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
    }
    for rel, marker in expected.items():
        assert marker in (REPO / rel).read_text(), (
            f"{rel}: synthetic snapshot_id marker gone — if this was the "
            f"snapshot migration, update this pin and the audit doc")
