"""Tests for hive hint bias integration in fee controller."""

import time
import pytest
from unittest.mock import MagicMock

from modules.fee_controller import FeeController
from modules.hive_hints import HiveHintAdapter


@pytest.fixture
def mock_plugin():
    p = MagicMock()
    p.rpc = MagicMock()
    p.log = MagicMock()
    return p


@pytest.fixture
def mock_config():
    c = MagicMock()
    c.min_fee_ppm = 10
    c.max_fee_ppm = 5000
    c.vegas_decay_rate = 0.95
    # Path B (2026-04-22): intra-fleet ppm is now configurable. Tests that
    # exercise _check_hive_member_fee explicitly set this; default to 0 so
    # pre-existing assertions that expect 0 still pass.
    c.fee_ppm_intra_fleet = 0
    c.neighbor_median_min_competitors = 2
    c.snapshot = MagicMock(return_value=c)
    return c


@pytest.fixture
def mock_database():
    return MagicMock()


class TestFeeHiveBias:
    def test_get_hive_fee_bias_with_adapter(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02aabb": {
                    "corridor_role": "owner",
                    "competition_bias": 1,
                    "traffic_confidence": 1.0,
                },
            },
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter.poll()
        fc.hive_hints = adapter
        bias = fc._get_hive_fee_bias("02aabb")
        assert bias > 1.0
        assert bias <= 1.1

    def test_get_hive_fee_bias_no_adapter(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        fc.hive_hints = None
        assert fc._get_hive_fee_bias("02aabb") == 1.0

    def test_get_hive_fee_bias_exception_returns_neutral(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.get_fee_bias.side_effect = Exception("boom")
        fc.hive_hints = adapter
        assert fc._get_hive_fee_bias("02aabb") == 1.0

    def test_bias_within_hard_cap(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.get_fee_bias.return_value = 1.5
        fc.hive_hints = adapter
        bias = fc._get_hive_fee_bias("02aabb")
        assert 0.9 <= bias <= 1.1

    def test_metabolic_fee_bias_is_bounded_scoring_input(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.get_fee_bias.return_value = 1.0
        adapter.get_metabolic_fee_bias.return_value = 1.05
        fc.hive_hints = adapter

        assert fc._get_hive_fee_bias("02aabb") == pytest.approx(1.05)

    def test_metabolic_fee_bias_cannot_escape_fee_hard_cap(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.get_fee_bias.return_value = 1.1
        adapter.get_metabolic_fee_bias.return_value = 1.05
        fc.hive_hints = adapter

        assert fc._get_hive_fee_bias("02aabb") == pytest.approx(1.1)

    def test_immune_fee_bias_is_bounded_scoring_input(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.get_fee_bias.return_value = 1.0
        adapter.get_metabolic_fee_bias.return_value = 1.0
        adapter.get_immune_fee_bias.return_value = 0.95
        fc.hive_hints = adapter

        assert fc._get_hive_fee_bias("02aabb") == pytest.approx(0.95)

    def test_immune_fee_bias_cannot_escape_fee_hard_cap(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.get_fee_bias.return_value = 0.9
        adapter.get_metabolic_fee_bias.return_value = 1.0
        adapter.get_immune_fee_bias.return_value = 0.95
        fc.hive_hints = adapter

        assert fc._get_hive_fee_bias("02aabb") == pytest.approx(0.9)

    def test_hive_exploration_multiplier_uses_fee_elasticity(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.get_centrality.return_value = 0.05
        adapter.get_corridor_role.return_value = "owner"
        adapter.get_fee_elasticity.return_value = 0.25
        fc.hive_hints = adapter

        multiplier = fc._get_hive_exploration_multiplier("02aabb")

        assert multiplier > 1.5

    def test_hive_fee_debug_exposes_bounded_hint_attribution(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.get_status.return_value = {
            "snapshot_fresh": True,
            "snapshot_usable": True,
            "snapshot_source": "datastore",
            "snapshot_age_seconds": 12,
            "effective_ttl_seconds": 900,
        }
        adapter.get_membership_status.return_value = {
            "known": True,
            "member": True,
            "fresh": True,
            "usable": True,
            "source": "datastore",
        }
        adapter.get_fee_bias.return_value = 1.5
        adapter.get_traffic_confidence.return_value = 0.9
        adapter.get_peak_hours.return_value = []
        adapter.get_centrality.return_value = 0.05
        adapter.get_corridor_role.return_value = "owner"
        adapter.get_fee_elasticity.return_value = 0.25
        adapter.get_peer_quality_score.return_value = 0.8
        adapter.get_fleet_fee_prior.return_value = 120
        adapter.get_optimal_fee_estimate.return_value = 150
        fc.hive_hints = adapter

        debug = fc.get_hive_fee_hint_debug("02aabb")

        assert debug["snapshot_fresh"] is True
        assert debug["snapshot_source"] == "datastore"
        assert debug["membership"]["member"] is True
        assert debug["fee_bias"] == 1.1
        assert debug["exploration_multiplier"] > 1.0
        assert debug["peer_quality_score"] == 0.8


class TestMemberFeePolicy:
    def test_hive_member_uses_zero_fee_policy(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.is_hive_member.return_value = True
        adapter.is_fresh.return_value = True
        adapter.is_usable.return_value = True
        adapter._effective_ttl.return_value = 900
        fc.hive_hints = adapter
        result = fc._check_hive_member_fee("02member")
        assert result == 0
        assert fc._consume_hive_member_advisory("02member") is True
        assert fc._consume_hive_member_advisory("02member") is False

    def test_configured_intra_fleet_ppm_does_not_override_zero_member_fee(self, mock_plugin, mock_config, mock_database):
        """Hive membership forces zero fees even when older intra-fleet ppm config is nonzero."""
        mock_config.fee_ppm_intra_fleet = 1
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.is_hive_member.return_value = True
        adapter.is_fresh.return_value = True
        adapter.is_usable.return_value = True
        adapter._effective_ttl.return_value = 900
        fc.hive_hints = adapter
        result = fc._check_hive_member_fee("02member")
        assert result == 0
        assert fc._consume_hive_member_advisory("02member") is True

    def test_non_member_returns_none(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.is_hive_member.return_value = False
        fc.hive_hints = adapter
        result = fc._check_hive_member_fee("02nonmember")
        assert result is None

    def test_no_adapter_returns_none(self, mock_plugin, mock_config, mock_database):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        fc.hive_hints = None
        result = fc._check_hive_member_fee("02peer")
        assert result is None

    def test_grace_period_keeps_membership_without_rearming_advisory_after_stale(self, mock_plugin, mock_config, mock_database):
        """Stale-but-usable hints keep membership sticky without forcing repeated reprices."""
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.is_hive_member.return_value = True
        adapter.is_fresh.return_value = True
        adapter.is_usable.return_value = True
        adapter._effective_ttl.return_value = 900
        fc.hive_hints = adapter
        assert fc._check_hive_member_fee("02peer") == 0
        assert fc._consume_hive_member_advisory("02peer") is True

        adapter.is_hive_member.return_value = False
        adapter.is_fresh.return_value = False
        adapter.is_usable.return_value = True
        assert fc._check_hive_member_fee("02peer") == 0
        assert fc._consume_hive_member_advisory("02peer") is False

    def test_explicit_hive_unavailable_clears_sticky_membership(self, mock_plugin, mock_config, mock_database):
        """When cl-hive is disabled, dynamic fee control should reprice immediately."""
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.is_hive_member.return_value = True
        adapter.is_usable.return_value = True
        adapter._effective_ttl.return_value = 900
        fc.hive_hints = adapter
        assert fc._check_hive_member_fee("02peer") == 0
        assert fc._consume_hive_member_advisory("02peer") is True

        adapter.is_hive_member.return_value = False
        adapter.is_usable.return_value = False

        assert fc._check_hive_member_fee("02peer") is None
        assert "02peer" not in fc._hive_member_set_at
        assert fc._consume_hive_member_advisory("02peer") is False
        assert fc._consume_hive_member_release("02peer") is True
        assert fc._consume_hive_member_release("02peer") is False

    def test_grace_period_expires(self, mock_plugin, mock_config, mock_database):
        """After grace period expires, zero-fee policy deactivates."""
        import time as _time
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adapter = MagicMock()
        adapter.is_hive_member.return_value = True
        adapter.is_fresh.return_value = True
        adapter.is_usable.return_value = True
        adapter._effective_ttl.return_value = 900
        fc.hive_hints = adapter
        fc._check_hive_member_fee("02peer")
        fc._hive_member_set_at["02peer"] = int(_time.time()) - 1801
        adapter.is_hive_member.return_value = False
        adapter.is_fresh.return_value = False
        adapter.is_usable.return_value = True
        assert fc._check_hive_member_fee("02peer") is None
        assert fc._consume_hive_member_advisory("02peer") is False


class TestHiveInfluenceBoundsPinning:
    """FC-I6 pinning tests (contract amended 2026-07-01).

    The hive influences the fee decision through THREE declared channels:
    (1) the fee-bias x temporal multiplier clamped [0.9, 1.1] (covered by
    TestFeeHiveBias above), (2) the exploration multiplier clamped
    [0.75, 2.0] that scales DTS draw noise, and (3) the fleet fee prior
    accepted only in [1, 10000] ppm (else None) that seeds the Thompson
    prior mean. These tests lock the bounds of (2) and (3) so any future
    widening of hive authority breaks the suite instead of shipping
    silently.
    """

    # ---- Channel 2: exploration multiplier [0.75, 2.0] ----

    def test_exploration_bound_constants_pinned(self):
        from modules.fee_controller import GaussianThompsonState
        assert GaussianThompsonState.EXPLORATION_BOOST_MIN == 0.75
        assert GaussianThompsonState.EXPLORATION_BOOST_MAX == 2.0

    def test_scale_variance_clamps_at_upper_boundary(self):
        from modules.fee_controller import GaussianThompsonState
        st = GaussianThompsonState()
        st.scale_variance(1_000_000.0)
        assert st.exploration_boost == 2.0

    def test_scale_variance_clamps_at_lower_boundary(self):
        from modules.fee_controller import GaussianThompsonState
        st = GaussianThompsonState()
        st.scale_variance(1e-9)
        assert st.exploration_boost == 0.75

    def test_resolve_exploration_boost_clamps_explicit_argument(self):
        from modules.fee_controller import GaussianThompsonState
        st = GaussianThompsonState()
        assert st._resolve_exploration_boost(50.0) == 2.0
        assert st._resolve_exploration_boost(0.0001) == 0.75
        # In-range values pass through unclamped.
        assert st._resolve_exploration_boost(1.3) == pytest.approx(1.3)

    def test_hive_exploration_multiplier_bounded_for_adversarial_hints(
        self, mock_plugin, mock_config, mock_database
    ):
        """Whatever the hint adapter returns (poisoned centrality/elasticity
        included), the multiplier handed to scale_variance stays in
        [0.75, 2.0]."""
        fc = FeeController(mock_plugin, mock_config, mock_database)
        adversarial_cases = [
            (1e9, "owner", 1e-9),        # max upward composite
            (1e9, "owner", float("nan")),
            (0.05, "owner", 0.25),        # suite's known >1.5 case
            (0.0, "none", 1e9),           # max downward composite
            (float("inf"), "owner", 0.5),
        ]
        for centrality, role, elasticity in adversarial_cases:
            adapter = MagicMock()
            adapter.get_centrality.return_value = centrality
            adapter.get_corridor_role.return_value = role
            adapter.get_fee_elasticity.return_value = elasticity
            fc.hive_hints = adapter
            m = fc._get_hive_exploration_multiplier("02aabb")
            assert 0.75 <= m <= 2.0, (
                f"exploration multiplier escaped [0.75, 2.0] for "
                f"centrality={centrality}, role={role}, "
                f"elasticity={elasticity}: {m}"
            )

    # ---- Channel 3: fleet fee prior [1, 10000]-or-None ----

    def _adapter_with_fleet_prior(self, mock_plugin, fleet_fee_median):
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02aabb": {"fleet_fee_median": fleet_fee_median},
            },
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter.poll()
        return adapter

    def test_fleet_prior_bound_constant_pinned(self):
        assert HiveHintAdapter.MAX_FLEET_FEE_PRIOR_PPM == 10_000

    @pytest.mark.parametrize(
        ("fleet_fee_median", "expected"),
        [
            (1, 1),            # lower boundary accepted
            (10_000, 10_000),  # upper boundary accepted
            (2_500, 2_500),
            (0, None),         # below lower boundary rejected
            (10_001, None),    # above upper boundary rejected
            (-5, None),
            (float("nan"), None),
            (float("inf"), None),
            (True, None),      # bools are not fees
            (None, None),
        ],
    )
    def test_fleet_fee_prior_accepts_only_1_to_10000_else_none(
        self, mock_plugin, fleet_fee_median, expected
    ):
        adapter = self._adapter_with_fleet_prior(mock_plugin, fleet_fee_median)
        assert adapter.get_fleet_fee_prior("02aabb") == expected

    def test_fee_controller_prior_selection_honors_none(
        self, mock_plugin, mock_config, mock_database
    ):
        """When the fleet prior neutralizes to None, _select_best_fee_prior
        must not seed from hive data (falls back to network/None)."""
        fc = FeeController(mock_plugin, mock_config, mock_database)
        fc.hive_hints = self._adapter_with_fleet_prior(mock_plugin, 10_001)
        assert fc._select_best_fee_prior("02aabb", "100x1x0", allow_rpc=False) is None

    def test_fee_controller_prior_selection_uses_bounded_fleet_value(
        self, mock_plugin, mock_config, mock_database
    ):
        fc = FeeController(mock_plugin, mock_config, mock_database)
        fc.hive_hints = self._adapter_with_fleet_prior(mock_plugin, 10_000)
        prior = fc._select_best_fee_prior("02aabb", "100x1x0", allow_rpc=False)
        assert prior is not None
        assert prior["source"] == "fleet"
        assert 1 <= prior["mean"] <= 10_000
