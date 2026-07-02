"""Per-channel windowed forwarding facts for the rebalancer.

Pure adapter over Database.get_channel_flow_window. Turns
(channel_id, capacity, now, config) into a ChannelFlowFacts bundle feeding
realized-utilization EV (#2) and the live-activity penalty (#1).

Fail-open: on any DB error or zero capacity, returns neutral facts
(utilization = the 0.5 prior, zero net-flow) so the rebalancer degrades to
current behavior and never crashes.
"""
from dataclasses import dataclass

UTILIZATION_PRIOR = 0.5


@dataclass(frozen=True)
class ChannelFlowFacts:
    channel_id: str
    out_sats_window: int
    in_sats_window: int
    forward_count_window: int
    realized_utilization: float
    utilization_is_realized: bool


def _neutral(channel_id: str) -> ChannelFlowFacts:
    return ChannelFlowFacts(
        channel_id=channel_id, out_sats_window=0, in_sats_window=0,
        forward_count_window=0, realized_utilization=UTILIZATION_PRIOR,
        utilization_is_realized=False,
    )


def _cfg_get(cfg, name, default):
    """Read a config value, falling back to default only when absent or None.

    Uses an ``is None`` check rather than truthiness so a legitimate falsy
    setting (e.g. ``rebalance_utilization_min_forwards = 0`` to disable the
    thin-history gate, or ``rebalance_utilization_floor = 0.0``) is respected
    instead of being silently replaced by the default.
    """
    value = getattr(cfg, name, default)
    return default if value is None else value


def compute_channel_flow_facts(db, channel_id: str, capacity_sats: int, now: int, cfg) -> ChannelFlowFacts:
    if capacity_sats <= 0:
        return _neutral(channel_id)
    try:
        activity_window = int(_cfg_get(cfg, "rebalance_activity_window_seconds", 3600))
        util_days = int(_cfg_get(cfg, "rebalance_utilization_window_days", 7))
        floor = float(_cfg_get(cfg, "rebalance_utilization_floor", 0.05))
        ceiling = float(_cfg_get(cfg, "rebalance_utilization_ceiling", 1.0))
        min_forwards = int(_cfg_get(cfg, "rebalance_utilization_min_forwards", 5))

        short_since = now - activity_window
        long_since = now - util_days * 86_400

        short_out, short_in, _ = db.get_channel_flow_window(channel_id, short_since)
        long_out, _long_in, long_count = db.get_channel_flow_window(channel_id, long_since)

        if long_count >= min_forwards:
            raw = long_out / float(capacity_sats)
            realized = max(floor, min(ceiling, raw))
            is_realized = True
        else:
            realized = UTILIZATION_PRIOR
            is_realized = False

        return ChannelFlowFacts(
            channel_id=channel_id,
            out_sats_window=max(0, int(short_out)),
            in_sats_window=max(0, int(short_in)),
            forward_count_window=int(long_count),
            realized_utilization=realized,
            utilization_is_realized=is_realized,
        )
    except Exception:
        return _neutral(channel_id)
