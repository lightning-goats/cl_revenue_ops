"""
Tests for _handle_policy_change: policy set/delete must wake the affected
peer's sleeping channels so the next fee cycle applies the new policy.
"""

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

from modules.fee_controller import (
    FeeController,
    ChannelFeeState,
    ChannelCycleState,
)
from modules.config import Config
from modules.policy_manager import PeerPolicy, FeeStrategy, RebalanceMode

PEER_A = "02" + "a" * 64
PEER_B = "03" + "b" * 64
CHAN_A1 = "111x1x0"
CHAN_A2 = "111x2x0"
CHAN_B1 = "222x1x0"


from modules.fee_authority import FeeAuthorityGate

def _make_fc(database=None):
    plugin = MagicMock()
    config = MagicMock(spec=Config)
    db = database or MagicMock()
    return FeeController(plugin, config, db, fee_authority_gate=FeeAuthorityGate())


def _sleeping_cycle():
    cycle = ChannelCycleState()
    cycle.is_sleeping = True
    cycle.sleep_until = 2_000_000_000
    cycle.stable_cycles = 5
    return cycle


def _policy(peer_id):
    return PeerPolicy(
        peer_id=peer_id,
        strategy=FeeStrategy.STATIC,
        rebalance_mode=RebalanceMode.ENABLED,
    )


def _db_with_channels(mapping):
    db = MagicMock()
    db.get_all_channel_states.return_value = [
        {"channel_id": cid, "peer_id": pid} for cid, pid in mapping.items()
    ]
    # _save_cycle_state merges with the persisted fee_strategy_state row
    db.get_fee_strategy_state.side_effect = lambda cid: {
        "channel_id": cid, "v2_state_json": "{}",
    }
    return db


class TestHandlePolicyChange:
    def test_wakes_sleeping_cycle_state_for_peer(self):
        db = _db_with_channels({CHAN_A1: PEER_A, CHAN_A2: PEER_A, CHAN_B1: PEER_B})
        fc = _make_fc(db)
        fc._cycle_states[CHAN_A1] = _sleeping_cycle()
        fc._cycle_states[CHAN_A2] = _sleeping_cycle()
        fc._cycle_states[CHAN_B1] = _sleeping_cycle()

        fc._handle_policy_change(PEER_A, _policy(PEER_A))

        for cid in (CHAN_A1, CHAN_A2):
            assert fc._cycle_states[cid].is_sleeping is False
            assert fc._cycle_states[cid].sleep_until == 0
            assert fc._cycle_states[cid].stable_cycles == 0
        # Other peer's channel untouched
        assert fc._cycle_states[CHAN_B1].is_sleeping is True

    def test_wakes_sleeping_channel_fee_state_for_peer(self):
        db = _db_with_channels({CHAN_A1: PEER_A})
        fc = _make_fc(db)
        state = ChannelFeeState()
        state.is_sleeping = True
        state.sleep_until = 2_000_000_000
        state.stable_cycles = 3
        fc._channel_fee_states[CHAN_A1] = state

        fc._handle_policy_change(PEER_A, _policy(PEER_A))

        assert state.is_sleeping is False
        assert state.sleep_until == 0
        assert state.stable_cycles == 0

    def test_awake_channels_are_left_alone(self):
        db = _db_with_channels({CHAN_A1: PEER_A})
        fc = _make_fc(db)
        cycle = ChannelCycleState()
        cycle.is_sleeping = False
        cycle.stable_cycles = 2
        fc._cycle_states[CHAN_A1] = cycle

        fc._handle_policy_change(PEER_A, _policy(PEER_A))

        # No wake needed; stable_cycles untouched for awake channels
        assert cycle.stable_cycles == 2

    def test_no_channels_for_peer_is_noop(self):
        db = _db_with_channels({CHAN_B1: PEER_B})
        fc = _make_fc(db)
        fc._handle_policy_change(PEER_A, _policy(PEER_A))

    def test_database_error_does_not_raise(self):
        db = MagicMock()
        db.get_all_channel_states.side_effect = RuntimeError("db down")
        fc = _make_fc(db)
        fc._handle_policy_change(PEER_A, _policy(PEER_A))
