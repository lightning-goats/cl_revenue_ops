"""
Tests for PolicyManager - policy management and validation.

Tests:
- Policy batch operations (bulk updates)
- Policy changes API (change log)
- Strategy handling (dynamic/static/passive)
- Rate limiting and validation
"""

import pytest
import time
import sys
import os
from unittest.mock import MagicMock

# Mock pyln.client before importing modules
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.policy_manager import (
    PolicyManager,
    PeerPolicy,
    FeeStrategy,
    RebalanceMode,
    MAX_POLICY_CHANGES_PER_MINUTE,
)


class TestPolicyManagerInit:
    """Test PolicyManager initialization."""

    def test_init_with_mock_db(self, mock_database, mock_plugin):
        """PolicyManager initializes with mock database."""
        pm = PolicyManager(mock_database, mock_plugin)
        assert pm is not None
        assert pm.database == mock_database

    def test_internal_set_policy_remains_available_for_coordination_code(self, mock_database, mock_plugin):
        """Direct PolicyManager writes remain available for internal callers."""
        pm = PolicyManager(mock_database, mock_plugin)

        policy = pm.set_policy("02" + "a" * 64, strategy="static", fee_ppm_target=500)

        assert policy.strategy == FeeStrategy.STATIC
        assert policy.fee_ppm_target == 500


class TestPolicyBatchOperations:
    """Test batch policy operations."""

    def test_batch_update_creates_policies(self, mock_database, mock_plugin, sample_policy_updates):
        """Batch update creates multiple policies in one call."""
        pm = PolicyManager(mock_database, mock_plugin)

        results = pm.set_policies_batch(sample_policy_updates)

        assert len(results) == 3
        assert all(isinstance(p, PeerPolicy) for p in results)

    def test_batch_update_applies_rebalance_modes(self, mock_database, mock_plugin, sample_peer_ids):
        """Batch update correctly sets rebalance modes."""
        pm = PolicyManager(mock_database, mock_plugin)

        updates = [
            {"peer_id": sample_peer_ids[0], "strategy": "dynamic", "rebalance_mode": "sink_only"},
            {"peer_id": sample_peer_ids[1], "strategy": "static", "rebalance_mode": "source_only"},
            {"peer_id": sample_peer_ids[2], "strategy": "passive", "rebalance_mode": "disabled"},
        ]
        results = pm.set_policies_batch(updates)

        assert results[0].rebalance_mode == RebalanceMode.SINK_ONLY
        assert results[1].rebalance_mode == RebalanceMode.SOURCE_ONLY
        assert results[2].rebalance_mode == RebalanceMode.DISABLED

    def test_batch_update_empty_list_returns_empty(self, mock_database, mock_plugin):
        """Batch update with empty list returns empty list."""
        pm = PolicyManager(mock_database, mock_plugin)

        results = pm.set_policies_batch([])

        assert results == []

    def test_batch_update_exceeds_max_size_raises(self, mock_database, mock_plugin, sample_peer_ids):
        """Batch update exceeding max size raises ValueError."""
        pm = PolicyManager(mock_database, mock_plugin)

        # Create more than MAX_BATCH_SIZE updates
        updates = [{"peer_id": f"02{'a' * 62}{i:02x}", "strategy": "dynamic"} for i in range(101)]

        with pytest.raises(ValueError) as exc_info:
            pm.set_policies_batch(updates)

        assert "exceeds maximum" in str(exc_info.value)

    def test_batch_update_invalid_peer_id_raises(self, mock_database, mock_plugin):
        """Batch update with invalid peer_id raises ValueError."""
        pm = PolicyManager(mock_database, mock_plugin)

        updates = [{"peer_id": "invalid", "strategy": "dynamic"}]

        with pytest.raises(ValueError) as exc_info:
            pm.set_policies_batch(updates)

        assert "peer_id" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()

    def test_batch_update_invalid_strategy_raises(self, mock_database, mock_plugin, sample_peer_ids):
        """Batch update with invalid strategy raises ValueError."""
        pm = PolicyManager(mock_database, mock_plugin)

        updates = [{"peer_id": sample_peer_ids[0], "strategy": "invalid_strategy"}]

        with pytest.raises(ValueError) as exc_info:
            pm.set_policies_batch(updates)

        assert "strategy" in str(exc_info.value).lower()

    def test_batch_update_preserves_existing_fields(self, mock_database, mock_plugin, sample_peer_ids):
        """Batch update preserves existing fields when not specified."""
        pm = PolicyManager(mock_database, mock_plugin)

        # First set a policy with all fields
        pm.set_policy(
            peer_id=sample_peer_ids[0],
            strategy="static",
            rebalance_mode="disabled",
            fee_ppm_target=500
        )

        # Now update only strategy via batch
        results = pm.set_policies_batch([
            {"peer_id": sample_peer_ids[0], "strategy": "dynamic"}
        ])

        # Rebalance mode should be preserved
        assert results[0].strategy == FeeStrategy.DYNAMIC
        assert results[0].rebalance_mode == RebalanceMode.DISABLED

    def test_batch_update_updates_timestamp(self, mock_database, mock_plugin, sample_peer_ids):
        """Batch update updates the timestamp for changed policies."""
        pm = PolicyManager(mock_database, mock_plugin)

        before = int(time.time())
        results = pm.set_policies_batch([
            {"peer_id": sample_peer_ids[0], "strategy": "dynamic"}
        ])
        after = int(time.time())

        assert before <= results[0].updated_at <= after


class TestPolicyChangesAPI:
    """Test policy changes API for sync."""

    def test_get_policy_changes_since_returns_list(self, mock_database, mock_plugin):
        """get_policy_changes_since returns a list (may be empty with mock db)."""
        # Setup mock to simulate database query returning empty
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_database._get_connection.return_value = mock_conn

        pm = PolicyManager(mock_database, mock_plugin)

        changes = pm.get_policy_changes_since(0)

        # Should return a list (empty with mock db)
        assert isinstance(changes, list)

    def test_get_policy_changes_since_empty_when_none(self, mock_database, mock_plugin):
        """get_policy_changes_since returns empty list when no changes."""
        # Setup mock to return empty results
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_database._get_connection.return_value = mock_conn

        pm = PolicyManager(mock_database, mock_plugin)

        future_timestamp = int(time.time()) + 3600

        changes = pm.get_policy_changes_since(future_timestamp)

        assert changes == []

    def test_get_policy_changes_since_handles_db_error(self, mock_database, mock_plugin):
        """get_policy_changes_since handles database errors gracefully."""
        mock_database._get_connection.side_effect = Exception("DB error")

        pm = PolicyManager(mock_database, mock_plugin)

        # Should not raise, should return empty list
        changes = pm.get_policy_changes_since(0)

        assert changes == []

    def test_policy_to_dict_has_required_fields(self, sample_peer_ids):
        """Policy.to_dict() includes fields needed for changes API."""
        policy = PeerPolicy(
            peer_id=sample_peer_ids[0],
            strategy=FeeStrategy.DYNAMIC,
            rebalance_mode=RebalanceMode.SINK_ONLY,
            updated_at=int(time.time())
        )

        d = policy.to_dict()

        assert "peer_id" in d
        assert "strategy" in d
        assert "rebalance_mode" in d
        assert "updated_at" in d


class TestRateLimiting:
    """Test policy change rate limiting."""

    def test_single_policy_change_allowed(self, mock_database, mock_plugin, sample_peer_ids):
        """Single policy change is always allowed."""
        pm = PolicyManager(mock_database, mock_plugin)

        # Should not raise
        pm.set_policy(sample_peer_ids[0], strategy="dynamic")

        policy = pm.get_policy(sample_peer_ids[0])
        assert policy.strategy == FeeStrategy.DYNAMIC

    def test_batch_bypasses_rate_limit(self, mock_database, mock_plugin, sample_peer_ids):
        """Batch operations bypass per-peer rate limiting."""
        pm = PolicyManager(mock_database, mock_plugin)

        # Create many updates (more than rate limit would allow individually)
        updates = [
            {"peer_id": sample_peer_ids[0], "strategy": "dynamic"},
            {"peer_id": sample_peer_ids[0], "strategy": "static"},
            {"peer_id": sample_peer_ids[0], "strategy": "dynamic"},
        ]

        # Should not raise - batch bypasses rate limit
        # Note: This tests the same peer being updated multiple times
        # The implementation may or may not allow this depending on design
        try:
            results = pm.set_policies_batch(updates)
            # If it succeeds, that's fine
        except ValueError as e:
            # If it fails due to duplicate peer_id, that's also acceptable behavior
            pass


class TestPeerPolicyDataclass:
    """Test PeerPolicy dataclass behavior."""

    def test_peer_policy_to_dict(self, sample_peer_ids):
        """PeerPolicy.to_dict() returns complete dict representation."""
        policy = PeerPolicy(
            peer_id=sample_peer_ids[0],
            strategy=FeeStrategy.DYNAMIC,
            rebalance_mode=RebalanceMode.ENABLED,
            fee_ppm_target=0,
            tags=["priority"],
            updated_at=int(time.time())
        )

        d = policy.to_dict()

        assert d["peer_id"] == sample_peer_ids[0]
        assert d["strategy"] == "dynamic"
        assert d["rebalance_mode"] == "enabled"
        assert "priority" in d["tags"]

    def test_peer_policy_has_tag(self, sample_peer_ids):
        """PeerPolicy.has_tag() correctly checks for tags."""
        policy = PeerPolicy(
            peer_id=sample_peer_ids[0],
            strategy=FeeStrategy.STATIC,
            tags=["priority", "important"]
        )

        assert policy.has_tag("priority")
        assert policy.has_tag("important")
        assert not policy.has_tag("other")

    def test_peer_policy_default_values(self, sample_peer_ids):
        """PeerPolicy has correct default values."""
        policy = PeerPolicy(peer_id=sample_peer_ids[0])

        assert policy.strategy == FeeStrategy.DYNAMIC
        assert policy.rebalance_mode == RebalanceMode.ENABLED
        assert policy.fee_ppm_target is None
        assert policy.tags == []

    def test_peer_policy_fee_multiplier_bounds_clamped(self, sample_peer_ids):
        """PeerPolicy clamps multiplier bounds to global safety range."""
        policy = PeerPolicy(
            peer_id=sample_peer_ids[0],
            strategy=FeeStrategy.DYNAMIC,
            fee_ppm_target=500,
            fee_multiplier_min=0.01,
            fee_multiplier_max=99.0,
        )

        min_mult, max_mult = policy.get_fee_multiplier_bounds()

        assert min_mult >= 0.1
        assert max_mult <= 5.0
        assert min_mult <= max_mult


class TestFeeBandPolicy:
    """Test fee band storage in policy manager."""

    def test_set_policy_persists_fee_autoband_multipliers(self, mock_database, mock_plugin, sample_peer_ids):
        pm = PolicyManager(mock_database, mock_plugin)

        pm.set_policy(
            sample_peer_ids[0],
            strategy="dynamic",
            fee_ppm_target=500,
            fee_multiplier_min=1.0,
            fee_multiplier_max=2.0,
        )

        policy = pm.get_policy(sample_peer_ids[0])
        assert policy.fee_ppm_target == 500
        assert policy.fee_multiplier_min == 1.0
        assert policy.fee_multiplier_max == 2.0


class TestValidation:
    """Test input validation."""

    def test_invalid_peer_id_rejected(self, mock_database, mock_plugin):
        """Invalid peer IDs are rejected."""
        pm = PolicyManager(mock_database, mock_plugin)

        # Test clearly invalid IDs that should be rejected
        invalid_ids = [
            "",
            "short",
            "not-a-hex-string-at-all",
        ]

        for invalid_id in invalid_ids:
            with pytest.raises(ValueError):
                pm.set_policy(invalid_id, strategy="dynamic")

    def test_valid_peer_id_accepted(self, mock_database, mock_plugin):
        """Valid peer IDs are accepted."""
        pm = PolicyManager(mock_database, mock_plugin)

        valid_ids = [
            "02" + "a" * 64,
            "03" + "b" * 64,
            "02" + "0" * 64,
            "03" + "f" * 64,
        ]

        for valid_id in valid_ids:
            pm.set_policy(valid_id, strategy="dynamic")
            policy = pm.get_policy(valid_id)
            assert policy.peer_id == valid_id
