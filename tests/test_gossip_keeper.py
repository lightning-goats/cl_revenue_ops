"""Tests for gossip keepalive target discovery."""

from types import SimpleNamespace
from unittest.mock import MagicMock


def _make_manager(hive_bridge=None):
    from modules.gossip_keeper import GossipKeepaliveManager

    plugin = MagicMock()
    plugin.log = MagicMock()
    plugin.rpc = MagicMock()
    plugin.rpc.getinfo.return_value = {"id": "02" + "f" * 64}

    cfg = SimpleNamespace(enable_gossip_keepalives=True, target_gossip_peers=3)
    return GossipKeepaliveManager(plugin=plugin, config=cfg, hive_bridge=hive_bridge)


def test_connected_peer_count_uses_all_connected_peers():
    manager = _make_manager()

    count = manager.count_connected_peers(
        {
            "peers": [
                {"id": "02" + "a" * 64, "connected": True},
                {"id": "02" + "b" * 64, "connected": False},
                {"id": "02" + "c" * 64, "connected": True},
            ]
        }
    )

    assert count == 2


def test_extract_channel_peer_ids_returns_peers_with_channels():
    manager = _make_manager()

    peer_ids = manager.extract_channel_peer_ids(
        {
            "channels": [
                {"peer_id": "02" + "a" * 64},
                {"peer_id": "03" + "b" * 64},
                {"peer_id": ""},
            ]
        }
    )

    assert peer_ids == {"02" + "a" * 64, "03" + "b" * 64}


def test_filter_candidates_removes_self_connected_and_channel_peers():
    manager = _make_manager()

    filtered = manager.filter_candidates(
        [
            "02" + "f" * 64,  # self
            "02" + "a" * 64,  # connected
            "03" + "b" * 64,  # channel
            "02" + "c" * 64,  # allowed
            "02" + "c" * 64,  # duplicate
        ],
        connected_peer_ids={"02" + "a" * 64},
        channel_peer_ids={"03" + "b" * 64},
    )

    assert filtered == ["02" + "c" * 64]


def test_get_ranked_targets_prefers_hive_candidates_before_public_candidates():
    hive_bridge = MagicMock()
    hive_bridge.get_priority_gossip_targets.return_value = [
        "02" + "a" * 64,
        "03" + "b" * 64,
    ]
    manager = _make_manager(hive_bridge=hive_bridge)

    ranked = manager.get_ranked_targets(
        connected_peer_ids=set(),
        channel_peer_ids=set(),
        public_candidates=[
            "02" + "c" * 64,
            "03" + "d" * 64,
        ],
    )

    assert ranked == [
        "02" + "a" * 64,
        "03" + "b" * 64,
        "02" + "c" * 64,
        "03" + "d" * 64,
    ]
