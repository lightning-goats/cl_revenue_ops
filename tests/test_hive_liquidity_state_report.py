"""Producer side of the fleet liquidity pipeline (audit item 1a).

The ["revenue", "liquidity-state"] datastore payload was previously written
ONLY on early-suppression paths (no-slots / capital-controls) with
depleted/saturated lists HARDCODED empty, so cl-hive's
_member_liquidity_state stayed empty in production forever.

These tests pin the honest producer contract:
- the NORMAL v2-engine path reports real liquidity state each cycle,
  derived from the engine's state snapshot (depleted = below the planner
  band-low, source = above band-high) and the selected pair candidates;
- channel entries carry capacity in BOTH keys: capacity_msat (primary,
  what cl-hive's liquidity_coordinator reads) and capacity_sats (compat).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.config import Config
from modules.rebalancer import EVRebalancer
from modules.rebalance_engine_v2 import CycleResult
from modules.rebalance_types_v2 import PairCandidate

SRC_PEER = "03" + "a" * 64
DST_PEER = "03" + "b" * 64
MID_PEER = "03" + "c" * 64


def _channel(channel_id, peer_id, capacity_sats, local_ratio):
    return SimpleNamespace(
        channel_id=channel_id,
        peer_id=peer_id,
        capacity_sats=capacity_sats,
        local_ratio=local_ratio,
    )


def _snapshot(channels):
    return SimpleNamespace(channels=tuple(channels))


def _pair(amount_sats=250_000):
    return PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=amount_sats,
        pair_budget_sats=500,
    )


def _make_rebalancer(mock_plugin, mock_database, cycle_result):
    cfg = Config(
        dry_run=False,
        low_liquidity_threshold=0.35,
        high_liquidity_threshold=0.65,
    )
    mock_database.cleanup_stale_reservations.return_value = 0

    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r._check_capital_controls = MagicMock(return_value=True)

    engine = MagicMock()
    engine.run_cycle.return_value = cycle_result
    engine.get_last_cycle_debug.return_value = {}
    r.rebalance_engine_v2 = engine

    data_service = MagicMock()
    data_service.datastore_push.return_value = True
    r.data_service = data_service
    return r, data_service


def _pushed_payload(data_service):
    calls = [
        c for c in data_service.datastore_push.call_args_list
        if c.args[0] == ["revenue", "liquidity-state"]
    ]
    assert calls, "no liquidity-state datastore push happened"
    return calls[-1].args[1]


def test_normal_cycle_reports_real_depleted_and_saturated_channels(
    mock_plugin, mock_database
):
    cycle_result = CycleResult(
        candidates=[_pair()],
        snapshot=_snapshot([
            _channel("100x1x0", SRC_PEER, 2_000_000, 0.90),   # above band-high
            _channel("200x1x0", DST_PEER, 1_000_000, 0.10),   # below band-low
            _channel("300x1x0", MID_PEER, 5_000_000, 0.50),   # in band
        ]),
    )
    r, data_service = _make_rebalancer(mock_plugin, mock_database, cycle_result)

    r.find_rebalance_candidates()

    payload = _pushed_payload(data_service)
    assert payload["depleted_channels"] == [{
        "peer_id": DST_PEER,
        "local_pct": 0.10,
        "capacity_msat": 1_000_000_000,
        "capacity_sats": 1_000_000,
    }]
    assert payload["saturated_channels"] == [{
        "peer_id": SRC_PEER,
        "local_pct": 0.90,
        "capacity_msat": 2_000_000_000,
        "capacity_sats": 2_000_000,
    }]
    # In-band channel reported on neither side.
    flat = payload["depleted_channels"] + payload["saturated_channels"]
    assert all(entry["peer_id"] != MID_PEER for entry in flat)


def test_normal_cycle_reports_selected_pairs_as_liquidity_needs(
    mock_plugin, mock_database
):
    cycle_result = CycleResult(
        candidates=[_pair(amount_sats=250_000)],
        snapshot=_snapshot([
            _channel("100x1x0", SRC_PEER, 2_000_000, 0.90),
            _channel("200x1x0", DST_PEER, 1_000_000, 0.10),
        ]),
    )
    r, data_service = _make_rebalancer(mock_plugin, mock_database, cycle_result)

    r.find_rebalance_candidates()

    needs = _pushed_payload(data_service)["liquidity_needs"]
    assert len(needs) == 1
    need = needs[0]
    assert need["source_peer_id"] == SRC_PEER
    assert need["destination_peer_id"] == DST_PEER
    assert need["capacity_sats"] == 250_000
    assert need["capacity_msat"] == 250_000_000


def test_normal_cycle_reports_even_with_no_candidates(mock_plugin, mock_database):
    # A hold cycle must still report channel state — that is exactly when
    # the fleet needs to know about depletion the local engine cannot fix.
    cycle_result = CycleResult(
        candidates=[],
        snapshot=_snapshot([
            _channel("200x1x0", DST_PEER, 1_000_000, 0.05),
        ]),
    )
    r, data_service = _make_rebalancer(mock_plugin, mock_database, cycle_result)

    r.find_rebalance_candidates()

    payload = _pushed_payload(data_service)
    assert len(payload["depleted_channels"]) == 1
    assert payload["liquidity_needs"] == []


def test_report_failure_does_not_break_cycle(mock_plugin, mock_database):
    cycle_result = CycleResult(
        candidates=[],
        snapshot=_snapshot([_channel("200x1x0", DST_PEER, 1_000_000, 0.05)]),
    )
    r, data_service = _make_rebalancer(mock_plugin, mock_database, cycle_result)
    data_service.datastore_push.side_effect = RuntimeError("datastore down")

    # Must not raise.
    result = r.find_rebalance_candidates()
    assert result == []


def test_payload_builder_emits_capacity_in_both_keys(mock_plugin, mock_database):
    cfg = Config(dry_run=True)
    r = EVRebalancer(mock_plugin, cfg, mock_database)

    payload = r._build_hive_liquidity_state_payload(
        depleted_channels=[("200x1x0", {"peer_id": DST_PEER, "capacity": 7_000},
                            0.12)],
        source_channels=[("100x1x0", {"peer_id": SRC_PEER, "capacity": 9_000},
                          0.88)],
        candidates=[_pair(amount_sats=1_000)],
    )

    dep = payload["depleted_channels"][0]
    sat = payload["saturated_channels"][0]
    assert (dep["capacity_msat"], dep["capacity_sats"]) == (7_000_000, 7_000)
    assert (sat["capacity_msat"], sat["capacity_sats"]) == (9_000_000, 9_000)
    need = payload["liquidity_needs"][0]
    assert (need["capacity_msat"], need["capacity_sats"]) == (1_000_000, 1_000)
