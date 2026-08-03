"""Formal cross-plugin contract tests for cl-mycelium / cl_revenue_ops IPC."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from modules.capex_budget import ChannelCapexBudget
from modules.profitability_analyzer import (
    ChannelCosts,
    ChannelProfitability,
    ChannelProfitabilityAnalyzer,
    ChannelRevenue,
    ProfitabilityClass,
)
from modules.segment_observations import SegmentObservationStore
from tests.plugin_test_utils import load_plugin_module


CONTRACT_DOCS = {
    ("revenue", "profitability-summary"): "docs/contracts/REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md",
    ("revenue", "capex-summary"): "docs/contracts/REVENUE_CAPEX_SUMMARY_CONTRACT.md",
    ("revenue", "segment-observations"): "docs/contracts/REVENUE_SEGMENT_OBSERVATIONS_CONTRACT.md",
}

REQUIRED_CONTRACT_SECTIONS = (
    "## Producer",
    "## Consumer",
    "## Generated At",
    "## TTL And Freshness",
    "## Units",
    "## Required Fields",
    "## Optional Fields",
    "## Stale Behavior",
    "## Malformed Behavior",
    "## Neutral Fallback Behavior",
    "## Versioning",
    "## Example Payload",
)


def _jsonable(payload: dict) -> None:
    json.dumps(payload, sort_keys=True)


def _profitability_analyzer() -> ChannelProfitabilityAnalyzer:
    plugin = MagicMock()
    config = MagicMock()
    config.estimated_open_cost_sats = 0
    database = MagicMock()
    analyzer = ChannelProfitabilityAnalyzer(plugin, config, database)
    analyzer.data_service = MagicMock()
    analyzer.data_service.datastore_push.return_value = True
    analyzer._cache_timestamp = int(time.time())
    return analyzer


def _profitability(channel_id: str = "100x1x0") -> ChannelProfitability:
    return ChannelProfitability(
        channel_id=channel_id,
        peer_id="02" + "c" * 64,
        capacity_sats=2_000_000,
        costs=ChannelCosts(
            channel_id=channel_id,
            peer_id="02" + "c" * 64,
            open_cost_sats=123,
            rebalance_cost_sats=2,
        ),
        revenue=ChannelRevenue(
            channel_id=channel_id,
            fees_earned_msat=1,
            volume_routed_msat=1_234_567,
            forward_count=3,
            sourced_volume_msat=9_876_543,
            sourced_fee_contribution_msat=1_001,
            sourced_forward_count=2,
        ),
        net_profit_sats=7,
        roi_percent=12.345,
        classification=ProfitabilityClass.PROFITABLE,
        cost_per_sat_routed=0.01,
        fee_per_sat_routed=0.02,
        days_open=30,
        last_routed=int(time.time()) - 60,
    )


def test_contract_documents_exist_and_define_required_sections():
    for key, path_text in CONTRACT_DOCS.items():
        path = Path(path_text)
        assert path.exists(), f"missing contract doc for {key}: {path_text}"
        text = path.read_text()
        assert f"`{list(key)}`" in text or f'["{key[0]}", "{key[1]}"]' in text
        for section in REQUIRED_CONTRACT_SECTIONS:
            assert section in text, f"{path_text} missing {section}"


def test_profitability_summary_producer_payload_matches_contract_and_rounding():
    analyzer = _profitability_analyzer()
    analyzer._push_profitability_summary({"100x1x0": _profitability()})

    key, payload = analyzer.data_service.datastore_push.call_args.args
    _jsonable(payload)
    assert key == ["revenue", "profitability-summary"]
    assert isinstance(payload["timestamp"], int)
    assert "100x1x0" in payload["channels"]

    channel = payload["channels"]["100x1x0"]
    required = {
        "channel_id",
        "peer_id",
        "class",
        "net_profit_sats",
        "roi_pct",
        "days_open",
        "role",
        "fee_multiplier",
        "forward_count",
        "sourced_forward_count",
        "total_forward_count",
        "fees_earned_msat",
        "fees_earned_sats",
        "volume_routed_msat",
        "sourced_volume_msat",
        "sourced_fee_contribution_msat",
        "sourced_fee_contribution_sats",
        "total_contribution_msat",
        "total_contribution_sats",
        "open_cost_msat",
        "rebalance_cost_msat",
        "net_pnl_msat",
    }
    assert required.issubset(channel)
    assert channel["fees_earned_msat"] == 1
    assert channel["fees_earned_sats"] == 1
    assert channel["sourced_fee_contribution_msat"] == 1_001
    assert channel["sourced_fee_contribution_sats"] == 2
    assert channel["total_contribution_msat"] == 1_001
    assert channel["total_contribution_sats"] == 2
    assert channel["open_cost_msat"] == 123_000
    assert channel["rebalance_cost_msat"] == 2_000
    assert channel["roi_pct"] == 12.35


def test_capex_summary_producer_payload_matches_contract():
    mod = load_plugin_module()
    mod.data_service = MagicMock()
    mod.capex_engine = MagicMock()
    mod.capex_engine.compute_allocations.return_value = SimpleNamespace(
        priority_class="growth",
        global_envelope_sats=10_000,
        total_fleet_contribution_sats=20_000,
        allocated_by_priority_sats={"growth": 1_500},
        channel_budgets={
            "100x1x0": ChannelCapexBudget(
                channel_id="100x1x0",
                budget_msat=700_000,
                tier="proven",
                tier_ppm=750,
                priority_class="growth",
            )
        },
    )

    result = mod.revenue_capex_status(mod.plugin)

    key, payload = mod.data_service.datastore_push.call_args.args
    _jsonable(payload)
    _jsonable(result)
    assert key == ["revenue", "capex-summary"]
    assert result["status"] == "ok"
    assert isinstance(result["timestamp"], int)
    assert result["generated_at"] == result["timestamp"]
    assert result["ttl_seconds"] == 1800
    assert result["channel_count"] == 1
    assert result["channels"]["100x1x0"]["budget_sats"] == 700
    assert isinstance(payload["timestamp"], int)
    assert payload["timestamp"] == result["timestamp"]
    assert payload["generated_at"] == result["timestamp"]
    assert payload["ttl_seconds"] == 1800
    assert payload["status"] == "ok"
    assert payload == {
        "timestamp": payload["timestamp"],
        "generated_at": payload["generated_at"],
        "ttl_seconds": 1800,
        "status": "ok",
        "priority_class": "growth",
        "global_envelope_sats": 10_000,
        "total_fleet_contribution_sats": 20_000,
        "allocated_by_priority_sats": {"growth": 1_500},
        "channel_count": 1,
    }


def test_segment_observations_producer_payload_matches_contract_and_stale_behavior():
    store = SegmentObservationStore(ttl_seconds=900)
    observer = "02" + "d" * 64
    store.record_failure(
        short_channel_id="123x1x0",
        direction=1,
        amount_sats=420_000,
        failure_class="liquidity",
        confidence=1.5,
        observed_at=1_700_000_000,
        source_channel_id="100x1x0",
        dest_channel_id="123x1x0",
        route_policy="network",
        router_kind="v2",
        correlation_id="corr-1",
    )
    store.record_failure(
        short_channel_id="999x1x0",
        direction=0,
        amount_sats=10_000,
        failure_class="fee",
        confidence=0.2,
        observed_at=1_699_000_000,
    )

    snapshot = store.export_snapshot(observer_member_id=observer, now=1_700_000_100)

    _jsonable(snapshot)
    assert snapshot["generated_at"] == 1_700_000_100
    assert snapshot["ttl_seconds"] == 900
    assert snapshot["schema_version"] == 1
    assert snapshot["observer_member_id"] == observer
    assert len(snapshot["segment_observations"]) == 1
    observation = snapshot["segment_observations"][0]
    assert observation["amount_bucket_sats"] == 250_000
    assert observation["outcome"] == "failure"
    assert observation["failure_class"] == "liquidity"
    assert observation["confidence"] == 1.0


