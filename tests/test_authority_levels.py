"""Phase 4 (Workstream I): global authority ladder enforced at the
governor. observe < fees < liquidity < capital; default 'capital'
preserves current behavior; unknown values fail closed to observe."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.econ_intents import Explanation, make_intent
from modules.econ_types import Micro, Msat, SignedMsat, UnixTime
from modules.governor_facade import (
    AUTHORITY_LEVELS,
    GovernorFacade,
    authority_allows,
)

NOW = 1_752_400_000


class TestLadder:
    def test_ladder_ordering(self):
        assert AUTHORITY_LEVELS == {"observe": 0, "fees": 1,
                                    "liquidity": 2, "capital": 3}

    def test_capital_allows_everything(self):
        for required in ("observe", "fees", "liquidity", "capital"):
            assert authority_allows("capital", required)

    def test_observe_allows_nothing_actionable(self):
        for required in ("fees", "liquidity", "capital"):
            assert not authority_allows("observe", required)

    def test_fees_level(self):
        assert authority_allows("fees", "fees")
        assert not authority_allows("fees", "liquidity")
        assert not authority_allows("fees", "capital")

    def test_unknown_configured_fails_closed(self):
        assert not authority_allows("captial", "fees")  # typo
        assert not authority_allows(None, "fees")
        assert not authority_allows("", "fees")

    def test_unknown_required_fails_closed(self):
        # An unknown requirement demands the TOP level.
        assert not authority_allows("liquidity", "wormhole")
        assert authority_allows("capital", "wormhole")


class TestFacadeGate:
    def _env(self):
        return make_intent(
            intent_type="REBALANCE", snapshot_id="s1",
            created_at=UnixTime(NOW), expires_at=UnixTime(NOW + 600),
            target="111x222x0", amount_msat=None,
            expected_benefit_msat=SignedMsat(0), max_cost_msat=Msat(3000),
            capital_committed_msat=Msat(0), confidence_micro=Micro(0),
            reason_codes=(), explanation=Explanation("t", (("x", 1),)),
            preconditions=(), priority=50, budget_bucket="rebalance",
            origin_policy="test", reversible=False)

    def _facade(self, authority_check):
        return GovernorFacade(
            reserve_spend=MagicMock(return_value=True),
            release_spend=MagicMock(return_value=True),
            is_paused=lambda: False,
            authority_check=authority_check)

    def test_blocked_below_level_without_reserving(self):
        reserve = MagicMock(return_value=True)
        facade = GovernorFacade(
            reserve_spend=reserve, release_spend=MagicMock(),
            is_paused=lambda: False,
            authority_check=lambda: authority_allows("fees", "liquidity"))
        decision = facade.authorize(self._env(), NOW)
        assert decision.authorized is False
        assert decision.reason_code == "AUTHORITY_LEVEL_BLOCKED"
        reserve.assert_not_called()

    def test_allowed_at_level(self):
        facade = self._facade(
            lambda: authority_allows("liquidity", "liquidity"))
        assert facade.authorize(self._env(), NOW).authorized is True

    def test_raising_check_fails_closed(self):
        facade = self._facade(MagicMock(side_effect=RuntimeError("boom")))
        decision = facade.authorize(self._env(), NOW)
        assert decision.reason_code == "AUTHORITY_LEVEL_BLOCKED"

    def test_no_check_means_ungated(self):
        facade = self._facade(None)
        assert facade.authorize(self._env(), NOW).authorized is True


def test_path_level_assignments():
    """Structural pin: each retained governed path declares its authority level."""
    import pathlib
    repo = pathlib.Path(__file__).resolve().parent.parent
    assert '"liquidity"),' in (repo / "modules" /
                               "rebalance_engine_v2.py").read_text()
    assert '"fees"),' in (repo / "modules" /
                          "fee_controller.py").read_text()
    assert not (repo / "modules" / "lnplus_swaps.py").exists()
