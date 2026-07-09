"""Capital-efficiency analysis for channel budgeting and planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Dict

from .utils import normalize_scid


@dataclass
class ChannelEfficiency:
    """Per-channel capital-efficiency metrics.

    rpsd is lifetime GROSS revenue per sat deployed (ppm).
    windowed_net_rpsd is the 30-day NET (marginal_profit_30d_sats) per sat
    deployed (ppm) — F7 (audit): lifetime-gross alone let long-dead former
    earners keep outranking currently productive channels.
    efficiency_rank blends both percentile ranks 50/50 when the
    profitability snapshot exposes 30d marginal profit.
    """

    channel_id: str
    rpsd: float
    efficiency_rank: float
    forward_velocity: float
    is_dead_capital: bool
    dead_capital_stage: str
    windowed_net_rpsd: float = 0.0


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
        config=None,
    ):
        self._profitability = profitability_analyzer
        self._flow = flow_analyzer
        self._database = database
        self._config = config

    def analyze(self) -> FleetEfficiency:
        """Build a fleet snapshot from current profitability and flow data."""
        profitability = self._profitability.analyze_all_channels() or {}
        flow = self._flow.analyze_all_channels() or {}
        try:
            raw_stages = self._database.get_dead_capital_stages() if self._database else {}
        except Exception:
            raw_stages = {}
        flow_by_normalized_channel = {
            normalize_scid(channel_id): metrics
            for channel_id, metrics in (flow or {}).items()
            if normalize_scid(channel_id)
        }
        stages = {
            normalize_scid(channel_id): value
            for channel_id, value in (raw_stages or {}).items()
            if normalize_scid(channel_id)
        }

        channel_ids = list(profitability.keys())
        rpsd_by_channel = {
            channel_id: self._calculate_rpsd(profitability[channel_id])
            for channel_id in channel_ids
        }
        ranks = self._calculate_percentile_ranks(rpsd_by_channel)

        # F7 (audit): blend in a windowed-NET rank when (and only when) the
        # profitability objects expose 30d marginal profit. Lifetime gross
        # alone is sticky: a channel that earned well a year ago and has
        # bled since keeps a top rank forever.
        windowed_rpsd_by_channel: Dict[str, float] = {}
        for channel_id in channel_ids:
            value = self._calculate_windowed_net_rpsd(profitability[channel_id])
            if value is None:
                windowed_rpsd_by_channel = {}
                break
            windowed_rpsd_by_channel[channel_id] = value
        if windowed_rpsd_by_channel:
            windowed_ranks = self._calculate_percentile_ranks(windowed_rpsd_by_channel)
            ranks = {
                channel_id: 0.5 * ranks[channel_id] + 0.5 * windowed_ranks[channel_id]
                for channel_id in channel_ids
            }

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
            normalized_channel_id = normalize_scid(channel_id)
            flow_metrics = (
                flow.get(channel_id)
                or flow.get(normalized_channel_id)
                or flow_by_normalized_channel.get(normalized_channel_id)
            )
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
                dead_capital_stage=str(stages.get(normalized_channel_id, {}).get("stage", "none")),
                windowed_net_rpsd=windowed_rpsd_by_channel.get(channel_id, 0.0),
            )

        return snapshot

    def _calculate_rpsd(self, profitability) -> float:
        """Revenue per sat deployed, scaled to ppm."""
        capacity_sats = max(0, int(getattr(profitability, "capacity_sats", 0) or 0))
        if capacity_sats <= 0:
            return 0.0

        revenue = getattr(profitability, "revenue", None)
        fees_earned_msat = getattr(revenue, "fees_earned_msat", None)
        if fees_earned_msat is None:
            fees_earned_sats = float(getattr(revenue, "fees_earned_sats", 0) or 0)
            fees_earned_msat = int(fees_earned_sats * 1000)
        else:
            fees_earned_msat = int(fees_earned_msat or 0)
        return float(fees_earned_msat) * 1000.0 / capacity_sats

    def _calculate_windowed_net_rpsd(self, profitability) -> float | None:
        """30-day NET profit per sat deployed, scaled to ppm.

        Returns None when the profitability object does not expose a numeric
        marginal_profit_30d_sats — callers must then fall back to pure
        lifetime ranking (F7: the blend only activates when the windowed
        signal genuinely exists; it is never synthesized).
        """
        if not hasattr(profitability, "marginal_profit_30d_sats"):
            return None
        raw = getattr(profitability, "marginal_profit_30d_sats")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        capacity_sats = max(0, int(getattr(profitability, "capacity_sats", 0) or 0))
        if capacity_sats <= 0:
            return 0.0
        return float(raw) * 1_000_000.0 / capacity_sats

    def _calculate_percentile_ranks(self, rpsd_by_channel: Dict[str, float]) -> Dict[str, float]:
        """Map each channel to a 0..1 percentile rank by RPSD."""
        if not rpsd_by_channel:
            return {}

        sorted_pairs = sorted(rpsd_by_channel.items(), key=lambda item: (item[1], item[0]))
        if len(sorted_pairs) == 1:
            channel_id, _ = sorted_pairs[0]
            return {channel_id: 1.0}

        denominator = len(sorted_pairs) - 1
        ranks: Dict[str, float] = {}
        index = 0
        while index < len(sorted_pairs):
            end = index
            value = sorted_pairs[index][1]
            while end + 1 < len(sorted_pairs) and sorted_pairs[end + 1][1] == value:
                end += 1
            avg_rank = ((index + end) / 2) / denominator
            for tied_index in range(index, end + 1):
                channel_id, _ = sorted_pairs[tied_index]
                ranks[channel_id] = avg_rank
            index = end + 1
        return ranks

    def _is_dead_capital(self, prof, flow_metrics, grace_days: int) -> bool:
        """Classify dead capital conservatively from flow and age."""
        if flow_metrics is None:
            return False

        if int(getattr(flow_metrics, "forward_count", 0) or 0) != 0:
            return False

        if int(getattr(prof, "days_open", 0) or 0) <= grace_days:
            return False

        return True
