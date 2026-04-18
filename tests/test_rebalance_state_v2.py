from unittest.mock import MagicMock

def test_build_state_snapshot_derives_value_classes_and_budget():
    from modules.capex_budget import CapexAllocations, ChannelCapexBudget
    from modules.rebalance_state_v2 import ChannelInput, build_state_snapshot

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
            ChannelInput(
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
            ChannelInput(
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


def test_build_state_snapshot_emits_role_eligibility_metadata():
    """Phase 1.1: per-channel state must expose source vs destination eligibility
    so the planner and operator surface can separate "would never drain from this"
    from "would never refill into this"."""
    from modules.capex_budget import CapexAllocations, ChannelCapexBudget
    from modules.rebalance_state_v2 import ChannelInput, build_state_snapshot

    allocations = CapexAllocations(
        channel_budgets={
            "111x1x0": ChannelCapexBudget(channel_id="111x1x0", budget_msat=1_000_000),
            "333x3x0": ChannelCapexBudget(channel_id="333x3x0", budget_msat=1_000_000),
        }
    )

    state = build_state_snapshot(
        [
            ChannelInput(  # over-local, profitable, not in cooldown -> drainable + refillable
                channel_id="111x1x0",
                peer_id="02" + "a" * 64,
                capacity_sats=1_000_000,
                local_sats=900_000,
                is_profitable=True,
                is_active=True,
            ),
            ChannelInput(  # neutral over-remote, no budget -> not refill-eligible
                channel_id="222x2x0",
                peer_id="02" + "b" * 64,
                capacity_sats=1_000_000,
                local_sats=100_000,
            ),
            ChannelInput(  # depleted profitable but cooldown active -> source not eligible
                channel_id="333x3x0",
                peer_id="02" + "c" * 64,
                capacity_sats=1_000_000,
                local_sats=66_000,
                is_profitable=True,
                cooldown_active=True,
            ),
        ],
        allocations,
    )

    drainable, neutral, depleted = state.channels

    assert drainable.source_eligible is True
    assert drainable.dest_eligible is True
    assert drainable.source_reason == ""
    assert drainable.dest_reason == ""
    assert drainable.source_drain_score > 0.0  # well above the high band
    assert drainable.dest_urgency == 0.0       # not depleted

    # Neutral over-remote: dest still rejected (no value class), but source is
    # eligible because Phase 2 source gate only checks cooldown protection.
    assert neutral.source_eligible is True
    assert neutral.dest_eligible is False
    assert neutral.source_reason == ""
    assert neutral.dest_reason == "not_valuable"
    assert neutral.dest_urgency > 0.0          # well below the low band
    assert neutral.source_drain_score == 0.0   # not over-local

    # Depleted profitable in cooldown: cooldown blocks both roles.
    assert depleted.source_eligible is False
    assert depleted.dest_eligible is False
    assert depleted.source_reason == "cooldown"
    assert depleted.dest_reason == "cooldown"
    assert depleted.dest_urgency > 0.5         # severely depleted


def test_source_eligibility_allows_neutral_channels_with_no_budget():
    """Phase 2.1+2.2: a neutral over-local channel with zero budget is still a
    valid drain source -- it does not need to already be a value channel and
    does not consume capex budget. Destinations remain conservative."""
    from modules.rebalance_state_v2 import ChannelInput, build_state_snapshot

    state = build_state_snapshot(
        [
            ChannelInput(  # neutral, no budget, healthy -> drainable as source
                channel_id="200x2x0",
                peer_id="02" + "a" * 64,
                capacity_sats=1_000_000,
                local_sats=950_000,
                is_profitable=False,
                is_active=False,
            ),
        ],
        {},
    )

    channel = state.channels[0]
    assert channel.value_class == "neutral"
    # Source: ELIGIBLE (Phase 2 relaxation)
    assert channel.source_eligible is True
    assert channel.source_reason == ""
    # Destination: still NOT eligible (conservative)
    assert channel.dest_eligible is False
    assert channel.dest_reason in {"not_valuable", "no_budget"}


def test_destination_eligibility_keeps_conservative_value_gate():
    """Phase 2.1: destinations still require value class + budget + no cooldown."""
    from modules.capex_budget import CapexAllocations, ChannelCapexBudget
    from modules.rebalance_state_v2 import ChannelInput, build_state_snapshot

    allocations = CapexAllocations(
        channel_budgets={
            "100x1x0": ChannelCapexBudget(channel_id="100x1x0", budget_msat=1_000_000),
        }
    )
    state = build_state_snapshot(
        [
            ChannelInput(
                channel_id="100x1x0",
                peer_id="02" + "a" * 64,
                capacity_sats=1_000_000,
                local_sats=200_000,
                is_profitable=True,
            ),
        ],
        allocations,
    )

    channel = state.channels[0]
    assert channel.dest_eligible is True
    assert channel.dest_reason == ""
    assert channel.source_eligible is True  # also drainable, but not over-local


def test_source_eligibility_blocked_only_by_cooldown():
    """Phase 2.2: source-side gate only blocks on cooldown protection. The
    cooldown flag is the source-protection mechanism for now; budget and
    value gates do not apply to sources."""
    from modules.rebalance_state_v2 import ChannelInput, build_state_snapshot

    state = build_state_snapshot(
        [
            ChannelInput(
                channel_id="201x2x0",
                peer_id="02" + "b" * 64,
                capacity_sats=1_000_000,
                local_sats=950_000,
                cooldown_active=True,
            ),
        ],
        {},
    )

    channel = state.channels[0]
    assert channel.source_eligible is False
    assert channel.source_reason == "cooldown"


def test_rebalancer_build_state_v2_delegates_to_builder(mock_plugin, mock_database):
    from modules.config import Config
    import modules.rebalancer as rebalancer_module
    from modules.rebalancer import EVRebalancer

    cfg = Config(dry_run=True)
    r = EVRebalancer(mock_plugin, cfg, mock_database)
    builder = MagicMock(return_value="snapshot")
    rebalancer_module.build_state_snapshot_v2 = builder

    result = r.build_state_v2(["channels"], "allocations")

    assert result == "snapshot"
    builder.assert_called_once_with(["channels"], "allocations")
