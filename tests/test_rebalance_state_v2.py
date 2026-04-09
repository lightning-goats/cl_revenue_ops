from unittest.mock import MagicMock

def test_build_state_snapshot_derives_value_classes_and_budget():
    from modules.capex_budget import CapexAllocations, ChannelCapexBudget
    from modules.rebalance_state_v2 import NormalizedV2ChannelInput, build_state_snapshot

    allocations = CapexAllocations(
        channel_budgets={
            "111x1x0": ChannelCapexBudget(
                channel_id="111x1x0",
                budget_msat=250_000,
            ),
            "222x2x0": ChannelCapexBudget(
                channel_id="222x2x0",
                budget_msat=0,
            ),
        }
    )

    state = build_state_snapshot(
        [
            NormalizedV2ChannelInput(
                channel_id="111x1x0",
                peer_id="02" + "a" * 64,
                capacity_sats=1_000_000,
                local_sats=650_000,
                actual_inbound_fee_ppm=120,
                is_hive_member=True,
                is_profitable=True,
                is_active=True,
                cooldown_active=False,
            ),
            NormalizedV2ChannelInput(
                channel_id="222x2x0",
                peer_id="02" + "b" * 64,
                capacity_sats=500_000,
                local_sats=100_000,
                actual_inbound_fee_ppm=80,
                is_hive_member=False,
                is_profitable=False,
                is_active=True,
                cooldown_active=True,
            ),
        ],
        allocations,
    )

    assert [channel.channel_id for channel in state.channels] == ["111x1x0", "222x2x0"]
    assert state.channels[0].local_ratio == 0.65
    assert state.channels[0].actual_inbound_fee_ppm == 120
    assert state.channels[0].value_class == "hive"
    assert state.channels[0].is_valuable is True
    assert state.channels[0].remaining_budget_sats == 250
    assert state.channels[0].cooldown_active is False

    assert state.channels[1].local_ratio == 0.2
    assert state.channels[1].value_class == "active"
    assert state.channels[1].is_valuable is True
    assert state.channels[1].remaining_budget_sats == 0
    assert state.channels[1].cooldown_active is True


def test_build_state_snapshot_normalizes_mapping_booleans_and_budget_defaults():
    from modules.rebalance_state_v2 import build_state_snapshot

    state = build_state_snapshot(
        [
            {
                "channel_id": "333x3x0",
                "peer_id": "02" + "c" * 64,
                "capacity_sats": 400_000,
                "local_sats": 40_000,
                "peer_inbound_fee_ppm": 95,
                "is_hive_member": "false",
                "is_profitable": "0",
                "is_active": "yes",
                "cooldown_active": "off",
            }
        ],
        {},
    )

    assert len(state.channels) == 1
    assert state.channels[0].actual_inbound_fee_ppm == 95
    assert state.channels[0].value_class == "active"
    assert state.channels[0].is_valuable is True
    assert state.channels[0].remaining_budget_sats == 0
    assert state.channels[0].cooldown_active is False


def test_rebalancer_build_rebalance_state_v2_delegates_to_builder(mock_plugin, mock_database):
    from modules.config import Config
    import modules.rebalancer as rebalancer_module
    from modules.rebalancer import EVRebalancer

    cfg = Config(dry_run=True)
    r = EVRebalancer(mock_plugin, cfg, mock_database)
    builder = MagicMock(return_value="snapshot")
    rebalancer_module.build_rebalance_state_v2_snapshot = builder

    result = r.build_rebalance_state_v2(["channels"], "allocations")

    assert result == "snapshot"
    builder.assert_called_once_with(["channels"], "allocations")
