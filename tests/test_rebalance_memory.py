"""Tests for rebalance routing memory."""

from modules.rebalance_memory import RebalanceRoutingMemory


def test_channel_and_node_bans_expire():
    memory = RebalanceRoutingMemory()

    memory.ban_channel("100x1x0/1", ttl_seconds=60, now=1000.0)
    memory.ban_node("node_abc", ttl_seconds=60, now=1000.0)

    assert sorted(memory.current_excludes(now=1020.0)) == ["100x1x0/1", "node_abc"]
    assert memory.current_excludes(now=1061.0) == []


def test_constraints_expire_and_return_max_amount():
    memory = RebalanceRoutingMemory()

    memory.constrain_channel("100x1x0/1", 123_456, ttl_seconds=60, now=1000.0)

    assert memory.max_amount_for("100x1x0/1", now=1001.0) == 123_456
    assert memory.max_amount_for("100x1x0/1", now=1061.0) is None
