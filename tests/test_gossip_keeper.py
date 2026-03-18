"""Tests for gossip keepalive target discovery."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock


def _make_manager(target_gossip_peers=3):
    from modules.gossip_keeper import GossipKeepaliveManager

    plugin = MagicMock()
    plugin.log = MagicMock()
    plugin.rpc = MagicMock()
    plugin.rpc.getinfo.return_value = {"id": "02" + "f" * 64}

    cfg = SimpleNamespace(
        enable_gossip_keepalives=True,
        target_gossip_peers=target_gossip_peers,
    )
    manager = GossipKeepaliveManager(plugin=plugin, config=cfg)
    return manager


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


def test_get_ranked_targets_returns_filtered_public_candidates():
    manager = _make_manager()

    ranked = manager.get_ranked_targets(
        connected_peer_ids=set(),
        channel_peer_ids=set(),
        public_candidates=[
            "02" + "c" * 64,
            "03" + "d" * 64,
        ],
    )

    assert ranked == [
        "02" + "c" * 64,
        "03" + "d" * 64,
    ]


def test_public_targets_rank_by_active_edges_then_capacity():
    manager = _make_manager()

    ranked = manager.rank_public_graph_targets(
        {
            "nodes": [
                {"nodeid": "02" + "a" * 64},
                {"nodeid": "02" + "b" * 64},
                {"nodeid": "02" + "c" * 64},
            ]
        },
        {
            "channels": [
                {
                    "short_channel_id": "1x1x0",
                    "source": "02" + "a" * 64,
                    "destination": "03" + "1" * 64,
                    "satoshis": 5_000_000,
                    "active": True,
                },
                {
                    "short_channel_id": "2x1x0",
                    "source": "02" + "a" * 64,
                    "destination": "03" + "2" * 64,
                    "satoshis": 2_000_000,
                    "active": True,
                },
                {
                    "short_channel_id": "3x1x0",
                    "source": "02" + "b" * 64,
                    "destination": "03" + "3" * 64,
                    "satoshis": 9_000_000,
                    "active": True,
                },
                {
                    "short_channel_id": "4x1x0",
                    "source": "02" + "c" * 64,
                    "destination": "03" + "4" * 64,
                    "satoshis": 20_000_000,
                    "active": False,
                },
            ]
        },
        excluded_peer_ids=set(),
    )

    assert ranked == ["02" + "a" * 64, "02" + "b" * 64]


def test_public_target_ranking_deduplicates_directional_channel_rows():
    manager = _make_manager()
    peer_a = "02" + "a" * 64
    peer_b = "02" + "b" * 64
    remote_x = "03" + "1" * 64
    remote_y = "03" + "2" * 64

    ranked = manager.rank_public_graph_targets(
        {
            "nodes": [
                {"nodeid": peer_a},
                {"nodeid": peer_b},
            ]
        },
        {
            "channels": [
                {
                    "short_channel_id": "1x1x0",
                    "source": peer_a,
                    "destination": remote_x,
                    "satoshis": 5_000_000,
                    "active": True,
                },
                {
                    "short_channel_id": "1x1x0",
                    "source": remote_x,
                    "destination": peer_a,
                    "satoshis": 5_000_000,
                    "active": True,
                },
                {
                    "short_channel_id": "2x1x0",
                    "source": peer_b,
                    "destination": remote_y,
                    "satoshis": 6_000_000,
                    "active": True,
                },
            ]
        },
        excluded_peer_ids=set(),
    )

    assert ranked == [peer_b, peer_a]


def test_maintain_connections_connects_only_deficit():
    manager = _make_manager(target_gossip_peers=3)
    manager.plugin.rpc.listpeers.return_value = {
        "peers": [
            {"id": "02" + "1" * 64, "connected": True},
        ]
    }
    manager.plugin.rpc.listpeerchannels.return_value = {"channels": []}
    manager.plugin.rpc.listnodes.return_value = {
        "nodes": [
            {"nodeid": "02" + "a" * 64},
            {"nodeid": "02" + "b" * 64},
            {"nodeid": "02" + "c" * 64},
        ]
    }
    manager.plugin.rpc.listchannels.return_value = {
        "channels": [
            {
                "short_channel_id": "1x1x0",
                "source": "02" + "a" * 64,
                "destination": "03" + "1" * 64,
                "satoshis": 5_000_000,
                "active": True,
            },
            {
                "short_channel_id": "2x1x0",
                "source": "02" + "b" * 64,
                "destination": "03" + "2" * 64,
                "satoshis": 4_000_000,
                "active": True,
            },
            {
                "short_channel_id": "3x1x0",
                "source": "02" + "c" * 64,
                "destination": "03" + "3" * 64,
                "satoshis": 3_000_000,
                "active": True,
            },
        ]
    }

    manager.maintain_connections()

    assert manager.plugin.rpc.connect.call_count == 2
    connect_ids = [call.args[0] for call in manager.plugin.rpc.connect.call_args_list]
    assert connect_ids == ["02" + "a" * 64, "02" + "b" * 64]


def test_maintain_connections_skips_peers_under_backoff():
    manager = _make_manager(target_gossip_peers=1)
    blocked_peer_id = "02" + "a" * 64
    allowed_peer_id = "02" + "b" * 64
    manager._connect_backoff_until[blocked_peer_id] = time.time() + 60
    manager.plugin.rpc.listpeers.return_value = {"peers": []}
    manager.plugin.rpc.listpeerchannels.return_value = {"channels": []}
    manager.plugin.rpc.listnodes.return_value = {
        "nodes": [
            {"nodeid": blocked_peer_id},
            {"nodeid": allowed_peer_id},
        ]
    }
    manager.plugin.rpc.listchannels.return_value = {
        "channels": [
            {
                "short_channel_id": "1x1x0",
                "source": blocked_peer_id,
                "destination": "03" + "1" * 64,
                "satoshis": 5_000_000,
                "active": True,
            },
            {
                "short_channel_id": "2x1x0",
                "source": allowed_peer_id,
                "destination": "03" + "2" * 64,
                "satoshis": 4_000_000,
                "active": True,
            },
        ]
    }

    manager.maintain_connections()

    manager.plugin.rpc.connect.assert_called_once_with(allowed_peer_id)


def test_failed_connect_adds_backoff():
    manager = _make_manager(target_gossip_peers=1)
    target_peer_id = "02" + "a" * 64
    manager.plugin.rpc.listpeers.return_value = {"peers": []}
    manager.plugin.rpc.listpeerchannels.return_value = {"channels": []}
    manager.plugin.rpc.listnodes.return_value = {"nodes": [{"nodeid": target_peer_id}]}
    manager.plugin.rpc.listchannels.return_value = {
        "channels": [
            {
                "short_channel_id": "1x1x0",
                "source": target_peer_id,
                "destination": "03" + "1" * 64,
                "satoshis": 5_000_000,
                "active": True,
            },
        ]
    }
    manager.plugin.rpc.connect.side_effect = RuntimeError("connection refused")

    manager.maintain_connections()

    assert manager._connect_failures[target_peer_id] == 1
    assert manager._connect_backoff_until[target_peer_id] > time.time()


def test_maintain_connections_uses_explicit_node_address_when_available():
    manager = _make_manager(target_gossip_peers=1)
    target_peer_id = "02" + "a" * 64
    manager.plugin.rpc.listpeers.return_value = {"peers": []}
    manager.plugin.rpc.listpeerchannels.return_value = {"channels": []}
    manager.plugin.rpc.listnodes.return_value = {
        "nodes": [
            {
                "nodeid": target_peer_id,
                "addresses": [
                    {"type": "ipv4", "address": "1.2.3.4", "port": 9735},
                ],
            }
        ]
    }
    manager.plugin.rpc.listchannels.return_value = {
        "channels": [
            {
                "short_channel_id": "1x1x0",
                "source": target_peer_id,
                "destination": "03" + "1" * 64,
                "satoshis": 5_000_000,
                "active": True,
            },
        ]
    }

    manager.maintain_connections()

    manager.plugin.rpc.connect.assert_called_once_with(target_peer_id, "1.2.3.4", 9735)
