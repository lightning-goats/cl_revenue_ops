"""
Pytest fixtures for cl-revenue-ops tests.

Provides mock database, plugin, and RPC fixtures.
"""

import pytest
import sqlite3
import tempfile
import os
import sys
from unittest.mock import MagicMock, patch

# Mock pyln.client before importing modules that depend on it
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

# Add modules to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def temp_db_path():
    """Create a temporary database file path."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def mock_plugin():
    """Create a mock plugin with basic functionality."""
    plugin = MagicMock()
    plugin.log = MagicMock()
    plugin.rpc = MagicMock()
    return plugin


@pytest.fixture
def mock_rpc():
    """Create a mock RPC interface."""
    rpc = MagicMock()

    # Default return values
    rpc.getinfo.return_value = {
        "id": "02" + "a" * 64,
        "alias": "test-node",
        "network": "regtest"
    }

    rpc.listchannels.return_value = {"channels": []}
    rpc.listfunds.return_value = {"channels": [], "outputs": []}
    rpc.listpeers.return_value = {"peers": []}

    rpc.plugin.return_value = {
        "plugins": [
            {"name": "cl-hive", "active": True},
            {"name": "cl-revenue-ops", "active": True}
        ]
    }

    return rpc


@pytest.fixture
def mock_database():
    """Create a mock database with common methods."""
    db = MagicMock()

    # Policy methods
    db.get_policy.return_value = None
    db.set_policy.return_value = True
    db.get_all_policies.return_value = []
    db.get_policies_changed_since.return_value = []
    db.delete_policy.return_value = True

    # Channel/peer methods
    db.get_channel_stats.return_value = {}
    db.get_peer_stats.return_value = {}
    db.get_channel_profitability.return_value = None

    return db


@pytest.fixture
def sample_peer_ids():
    """Sample peer IDs for testing."""
    return [
        "02" + "a" * 64,
        "02" + "b" * 64,
        "02" + "c" * 64,
        "03" + "d" * 64,
        "03" + "e" * 64,
    ]


@pytest.fixture
def sample_channel_id():
    """Sample channel ID for testing."""
    return "123x456x0"


@pytest.fixture
def sample_policy_updates(sample_peer_ids):
    """Sample policy batch updates."""
    return [
        {
            "peer_id": sample_peer_ids[0],
            "strategy": "hive",
            "rebalance_mode": "enabled"
        },
        {
            "peer_id": sample_peer_ids[1],
            "strategy": "hive",
            "rebalance_mode": "sink_only"
        },
        {
            "peer_id": sample_peer_ids[2],
            "strategy": "dynamic",
            "rebalance_mode": "enabled"
        }
    ]


@pytest.fixture
def sample_channel_closed_payload(sample_peer_ids, sample_channel_id):
    """Sample channel closed notification payload."""
    return {
        "peer_id": sample_peer_ids[0],
        "channel_id": sample_channel_id,
        "closer": "local",
        "close_type": "mutual",
        "capacity_sats": 1_000_000,
        "duration_days": 30,
        "total_revenue_sats": 5000,
        "total_rebalance_cost_sats": 500,
        "net_pnl_sats": 4500,
        "forward_count": 150,
        "forward_volume_sats": 50_000_000,
        "our_fee_ppm": 100,
        "their_fee_ppm": 200,
        "routing_score": 0.75,
        "profitability_score": 0.85
    }


@pytest.fixture
def sample_channel_opened_payload(sample_peer_ids, sample_channel_id):
    """Sample channel opened notification payload."""
    return {
        "peer_id": sample_peer_ids[0],
        "channel_id": sample_channel_id,
        "opener": "local",
        "capacity_sats": 2_000_000,
        "our_funding_sats": 2_000_000,
        "their_funding_sats": 0
    }


@pytest.fixture
def mock_hive_bridge(mock_plugin, mock_rpc):
    """
    Create a HiveFeeIntelligenceBridge with hive availability pre-set.

    This properly sets both _hive_available and _availability_check_time
    so that is_available() returns the cached value without doing a fresh check.
    """
    import time
    from modules.hive_bridge import HiveFeeIntelligenceBridge

    mock_plugin.rpc = mock_rpc
    bridge = HiveFeeIntelligenceBridge(mock_plugin, None)
    bridge._hive_available = True
    bridge._availability_check_time = time.time()  # Set fresh timestamp
    return bridge
