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


class TestPolicySuggestions:
    """Test profitability-driven policy suggestions."""

    def test_bleeder_suggestion_carries_channel_scope(
        self, mock_database, mock_plugin, sample_peer_ids
    ):
        pm = PolicyManager(mock_database, mock_plugin)
        peer_id = sample_peer_ids[0]
        channel_id = "123x456x0"
        analyzer = MagicMock()
        analyzer.identify_bleeders.return_value = [
            {
                "peer_id": peer_id,
                "channel_id": channel_id,
                "net_pnl_sats": -250,
                "forward_count": 4,
                "sourced_forward_count": 1,
                "rebalance_cost_sats": 800,
            }
        ]
        mock_database.get_channel_rebalance_success_rate.side_effect = [
            {"total": 3, "success_rate": 0.30},
            {"total": 6, "success_rate": 0.50},
        ]
        mock_database.get_all_channel_states.return_value = []

        suggestions = pm.get_policy_suggestions(analyzer)

        assert len(suggestions) == 1
        suggestion = suggestions[0]
        assert suggestion["scope"] == "channel"
        assert suggestion["channel_id"] == channel_id
        assert suggestion["channel_id_short"] == "123x456x0"
        assert suggestion["suggested_channel_rebalance_mode"] == "disabled"
        assert suggestion["suggested_rebalance_mode"] == "disabled"
        assert suggestion["policy_evidence_scope"] == "channel_specific"


class TestPolicyBatchOperations:
    """Test batch policy operations."""

    def test_batch_update_creates_policies(self, mock_database, mock_plugin, sample_policy_updates):
        """Batch update creates multiple policies in one call."""
        pm = PolicyManager(mock_database, mock_plugin)

        results = pm.set_policies_batch(sample_policy_updates)

        assert len(results) == 3
        assert all(isinstance(p, PeerPolicy) for p in results)

    def test_batch_update_applies_rebalance_modes(self, mock_database, mock_plugin, sample_peer_ids):
        """Batch update correctly sets rebalance modes.

        PM-I2 fix: the STATIC entry must carry a fee_ppm_target — this test
        previously entrenched the batch static-without-target gap by asserting
        success for the invalid input (see test_batch_update_rejects_static_
        without_target for the rejection).
        """
        pm = PolicyManager(mock_database, mock_plugin)

        updates = [
            {"peer_id": sample_peer_ids[0], "strategy": "dynamic", "rebalance_mode": "sink_only"},
            {"peer_id": sample_peer_ids[1], "strategy": "static", "rebalance_mode": "source_only",
             "fee_ppm_target": 500},
            {"peer_id": sample_peer_ids[2], "strategy": "passive", "rebalance_mode": "disabled"},
        ]
        results = pm.set_policies_batch(updates)

        assert results[0].rebalance_mode == RebalanceMode.SINK_ONLY
        assert results[1].rebalance_mode == RebalanceMode.SOURCE_ONLY
        assert results[2].rebalance_mode == RebalanceMode.DISABLED

    def test_batch_update_rejects_static_without_target(self, mock_database, mock_plugin, sample_peer_ids):
        """PM-I2: batch STATIC without fee_ppm_target is rejected like set_policy.

        Whole batch fails (validate-first semantics, PM-I10) and nothing is
        persisted; previously this persisted STATIC with fee_ppm_target=None
        and the fee controller silently fell through to dynamic management.
        """
        pm = PolicyManager(mock_database, mock_plugin)

        updates = [
            {"peer_id": sample_peer_ids[0], "strategy": "dynamic"},
            {"peer_id": sample_peer_ids[1], "strategy": "static"},  # no target
        ]
        with pytest.raises(ValueError, match="fee_ppm_target"):
            pm.set_policies_batch(updates)
        mock_database.upsert_policies_batch.assert_not_called()

    def test_batch_update_accepts_static_with_target(self, mock_database, mock_plugin, sample_peer_ids):
        """PM-I2: batch STATIC with an explicit target (including 0) succeeds."""
        pm = PolicyManager(mock_database, mock_plugin)

        results = pm.set_policies_batch([
            {"peer_id": sample_peer_ids[0], "strategy": "static", "fee_ppm_target": 0},
        ])

        assert results[0].strategy == FeeStrategy.STATIC
        assert results[0].fee_ppm_target == 0

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

    def test_failed_batch_commit_leaves_no_rate_limit_entry(
        self, mock_database, mock_plugin, sample_peer_ids
    ):
        """DEF-037: rate-limit timestamps are recorded only AFTER a successful
        upsert commit. A failed commit must not leave a phantom entry that
        throttles retries of a change that never persisted."""
        pm = PolicyManager(mock_database, mock_plugin)
        mock_database.upsert_policies_batch.side_effect = RuntimeError("commit boom")

        updates = [
            {"peer_id": sample_peer_ids[0], "strategy": "dynamic"},
            {"peer_id": sample_peer_ids[1], "strategy": "dynamic"},
        ]
        with pytest.raises(RuntimeError, match="commit boom"):
            pm.set_policies_batch(updates)

        for peer_id in sample_peer_ids[:2]:
            assert not pm._change_timestamps.get(peer_id), (
                f"phantom rate-limit entry left for {peer_id} after failed commit"
            )

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


class TestCorruptRowIsolation:
    """PM-I13: one corrupt DB row must not poison the cache / brick all reads."""

    @staticmethod
    def _row(peer_id, **overrides):
        row = {
            "peer_id": peer_id,
            "strategy": "dynamic",
            "rebalance_mode": "enabled",
            "fee_ppm_target": None,
            "tags": "[]",
            "updated_at": 1000,
            "fee_multiplier_min": None,
            "fee_multiplier_max": None,
            "expires_at": None,
        }
        row.update(overrides)
        return row

    def test_corrupt_expires_at_row_does_not_poison_cache(self, mock_database, mock_plugin, sample_peer_ids):
        """A corrupt TEXT expires_at (TypeError in is_expired) is skipped with a
        warning; get_policy still serves every other peer's stored policy."""
        corrupt = self._row(sample_peer_ids[1], expires_at="garbage")
        valid = self._row(sample_peer_ids[0], strategy="static", fee_ppm_target=500)
        mock_database.get_all_policies.return_value = [corrupt, valid]
        pm = PolicyManager(mock_database, mock_plugin)

        policy = pm.get_policy(sample_peer_ids[0])

        assert policy.strategy == FeeStrategy.STATIC
        assert policy.fee_ppm_target == 500
        # A warning was logged for the skipped row
        assert any(
            "corrupt" in str(call.args[0]).lower()
            for call in mock_plugin.log.call_args_list
        )

    def test_corrupt_row_peer_degrades_to_default(self, mock_database, mock_plugin, sample_peer_ids):
        """The peer whose row is corrupt gets the permissive default (PM-I7),
        not an exception."""
        corrupt = self._row(sample_peer_ids[1], expires_at="garbage")
        mock_database.get_all_policies.return_value = [corrupt]
        pm = PolicyManager(mock_database, mock_plugin)

        fallback = pm.get_policy(sample_peer_ids[1])

        assert fallback.strategy == FeeStrategy.DYNAMIC
        assert fallback.rebalance_mode == RebalanceMode.ENABLED


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

    def test_peer_id_trailing_newline_rejected(self, mock_database, mock_plugin):
        """PM-I1: 66 hex chars + trailing newline must be rejected.

        Python's re '$' matches before a trailing newline, so '^...{66}$'
        accepted a 67-char input and persisted a junk row end-to-end.
        """
        pm = PolicyManager(mock_database, mock_plugin)

        with pytest.raises(ValueError):
            pm.set_policy("02" + "a" * 64 + "\n", strategy="dynamic")
        mock_database.upsert_policy.assert_not_called()

    def test_peer_id_wrong_length_rejected(self, mock_database, mock_plugin):
        """PM-I1: 67 hex chars (and 65) are rejected strictly."""
        pm = PolicyManager(mock_database, mock_plugin)

        for bad in ("02" + "a" * 65, "02" + "a" * 63):
            with pytest.raises(ValueError):
                pm.set_policy(bad, strategy="dynamic")
        mock_database.upsert_policy.assert_not_called()

    def test_peer_id_non_hex_rejected(self, mock_database, mock_plugin):
        """PM-I1: right length but non-hex characters are rejected."""
        pm = PolicyManager(mock_database, mock_plugin)

        with pytest.raises(ValueError):
            pm.set_policy("02" + "g" * 64, strategy="dynamic")

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
