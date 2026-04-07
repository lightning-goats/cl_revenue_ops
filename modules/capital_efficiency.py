"""Capital-efficiency analysis for channel budgeting and planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Dict


@dataclass
class ChannelEfficiency:
    """Per-channel capital-efficiency metrics."""

    channel_id: str
    rpsd: float
    efficiency_rank: float
    forward_velocity: float
    is_dead_capital: bool
    dead_capital_stage: str


@dataclass
class FleetEfficiency:
    """Fleet-wide capital-efficiency snapshot."""

    channel_efficiencies: Dict[str, ChannelEfficiency] = field(default_factory=dict)
    median_rpsd: float = 0.0
    dead_capital_count: int = 0
    dead_capital_sats: int = 0
    total_deployed_sats: int = 0


class CapitalEfficiencyAnalyzer:
    """Compute efficiency rankings from profitability and flow snapshots."""

    def __init__(
        self,
        profitability_analyzer,
        flow_analyzer,
        database,
        hive_hints=None,
        config=None,
    ):
        self._profitability = profitability_analyzer
        self._flow = flow_analyzer
        self._database = database
        self._hive_hints = hive_hints
        self._config = config

    def analyze(self) -> FleetEfficiency:
        """Build a fleet snapshot from current profitability and flow data."""
        profitability = self._profitability.analyze_all_channels() or {}
        flow = self._flow.analyze_all_channels() or {}
        stages = self._database.get_dead_capital_stages() if self._database else {}

        channel_ids = list(profitability.keys())
        rpsd_by_channel = {
            channel_id: self._calculate_rpsd(profitability[channel_id])
            for channel_id in channel_ids
        }
        ranks = self._calculate_percentile_ranks(rpsd_by_channel)

        grace_days = int(getattr(self._config, "capex_grace_days", 14) or 14)
        flow_window_days = int(
            getattr(self._config, "flow_window_days", 0)
            or getattr(getattr(self._flow, "config", None), "flow_window_days", 0)
            or 7
        )

        snapshot = FleetEfficiency(
            median_rpsd=float(median(rpsd_by_channel.values())) if rpsd_by_channel else 0.0,
            total_deployed_sats=sum(
                max(0, int(getattr(profitability[channel_id], "capacity_sats", 0) or 0))
                for channel_id in channel_ids
            ),
        )

        for channel_id in channel_ids:
            prof = profitability[channel_id]
            flow_metrics = flow.get(channel_id)
            is_dead_capital = self._is_dead_capital(
                prof=prof,
                flow_metrics=flow_metrics,
                grace_days=grace_days,
            )

            if is_dead_capital:
                snapshot.dead_capital_count += 1
                snapshot.dead_capital_sats += max(0, int(getattr(prof, "capacity_sats", 0) or 0))

            forward_count = int(getattr(flow_metrics, "forward_count", 0) or 0) if flow_metrics else 0
            snapshot.channel_efficiencies[channel_id] = ChannelEfficiency(
                channel_id=channel_id,
                rpsd=rpsd_by_channel[channel_id],
                efficiency_rank=ranks[channel_id],
                forward_velocity=forward_count / max(flow_window_days, 1),
                is_dead_capital=is_dead_capital,
                dead_capital_stage=str(stages.get(channel_id, {}).get("stage", "none")),
            )

        return snapshot

    def _calculate_rpsd(self, profitability) -> float:
        """Revenue per sat deployed, scaled to ppm."""
        capacity_sats = max(0, int(getattr(profitability, "capacity_sats", 0) or 0))
        if capacity_sats <= 0:
            return 0.0

        revenue = getattr(profitability, "revenue", None)
        fees_earned_sats = getattr(revenue, "fees_earned_sats", None)
        if fees_earned_sats is None:
            fees_earned_msat = int(getattr(revenue, "fees_earned_msat", 0) or 0)
            fees_earned_sats = fees_earned_msat / 1000
        return float(fees_earned_sats) / capacity_sats * 1_000_000

    def _calculate_percentile_ranks(self, rpsd_by_channel: Dict[str, float]) -> Dict[str, float]:
        """Map each channel to a 0..1 percentile rank by RPSD."""
        if not rpsd_by_channel:
            return {}

        sorted_pairs = sorted(rpsd_by_channel.items(), key=lambda item: (item[1], item[0]))
        if len(sorted_pairs) == 1:
            channel_id, _ = sorted_pairs[0]
            return {channel_id: 1.0}

        denominator = len(sorted_pairs) - 1
        return {
            channel_id: index / denominator
            for index, (channel_id, _value) in enumerate(sorted_pairs)
        }

    def _is_dead_capital(self, prof, flow_metrics, grace_days: int) -> bool:
        """Classify dead capital conservatively from flow and age."""
        if flow_metrics is None:
            return False

        if int(getattr(flow_metrics, "forward_count", 0) or 0) != 0:
            return False

        if int(getattr(prof, "days_open", 0) or 0) <= grace_days:
            return False

        if self._hive_hints is not None:
            try:
                if self._hive_hints.is_hive_member(getattr(prof, "peer_id", None)):
                    return False
            except Exception:
                pass

        return True
