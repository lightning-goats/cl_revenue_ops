"""Thin pytest wrapper around tools/audit/stress_concurrency.py.

Runs a short (CI-length) concurrency soak and asserts the structural safety
invariants that MUST hold (integrity, no-deadlock, no-thread-exception). The
budget-TOCTOU invariant (P1-003) is expected to fail against the current tree,
so it is captured as an xfail sentinel: if it ever starts holding (someone
closes P1-003), the xpass is visible without breaking CI.

For the full 30-minute soak, run the tool directly:

    python3 tools/audit/stress_concurrency.py --soak 1800 --json soak.json
"""

import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.audit import stress_concurrency as sc  # noqa: E402


def _run(soak=6, threads=4, seed=20260701, budget=10000):
    args = argparse.Namespace(
        soak=soak,
        threads=threads,
        budget=budget,
        seed=seed,
        watchdog=20.0,
        json=None,
        allow_known_budget_toctou=True,
    )
    harness = sc.StressHarness(args)
    return harness.run()


@pytest.fixture(scope="module")
def report():
    return _run()


def test_substrate_matches_production(report):
    """Temp DB must be WAL + autocommit, exactly as production configures it."""
    assert str(report["substrate"]["journal_mode"]).lower() == "wal"
    assert report["substrate"]["isolation_level"] is None


def test_harness_made_progress(report):
    """Sanity: the firehose actually exercised the money path."""
    assert report["progress_ops"] > 1000
    # At least some reservations landed (otherwise the budget test is vacuous).
    assert report["max_reserved_observed_sats"] > 0


def test_integrity_check_ok(report):
    assert report["results"]["integrity"], report["integrity_check"]


def test_no_deadlock(report):
    assert report["results"]["no_deadlock"], "watchdog detected a progress stall (deadlock)"


def test_no_unhandled_thread_exception(report):
    assert report["results"]["no_thread_exception"], report["thread_exceptions"][:3]


@pytest.mark.xfail(
    reason="P1-003: cross-category budget reserve is a TOCTOU (open finding); "
    "the stress harness reproduces overspend. xpass here means it was closed.",
    strict=False,
)
def test_budget_never_exceeded_P1_003(report):
    assert report["results"]["budget_never_exceeded"], (
        f"budget exceeded: {report['budget_violations'][:1]}"
    )
