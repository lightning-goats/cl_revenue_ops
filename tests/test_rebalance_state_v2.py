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
    assert state.channels[0].value_class == "profitable"
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
            ChannelInput(  # moderately depleted profitable in cooldown -> both roles blocked
                channel_id="333x3x0",
                peer_id="02" + "c" * 64,
                capacity_sats=1_000_000,
                local_sats=200_000,
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

    # Depleted profitable in cooldown (local_ratio above emergency floor):
    # cooldown still blocks both roles -- Phase 3 drift override only fires
    # for emergency-low local ratios.
    assert depleted.source_eligible is False
    assert depleted.dest_eligible is False
    assert depleted.source_reason == "cooldown"
    assert depleted.dest_reason == "cooldown"
    assert depleted.dest_urgency > 0.0         # depleted but not at emergency


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


def test_emergency_local_ratio_overrides_destination_cooldown():
    """Phase 3.1+3.2: a destination that has drifted below the emergency
    local ratio is refill-eligible even when channel-level cooldown is
    active. The Polar S9 case (depleted profitable channel held in
    cooldown after drifting back to 6.6% local) becomes refillable."""
    from modules.capex_budget import CapexAllocations, ChannelCapexBudget
    from modules.rebalance_state_v2 import ChannelInput, build_state_snapshot

    allocations = CapexAllocations(
        channel_budgets={
            "100x1x0": ChannelCapexBudget(channel_id="100x1x0", budget_msat=1_000_000),
            "200x2x0": ChannelCapexBudget(channel_id="200x2x0", budget_msat=1_000_000),
        }
    )
    state = build_state_snapshot(
        [
            # severely depleted profitable in cooldown -> override fires
            ChannelInput(
                channel_id="100x1x0",
                peer_id="02" + "a" * 64,
                capacity_sats=1_000_000,
                local_sats=66_000,
                is_profitable=True,
                cooldown_active=True,
            ),
            # moderately depleted profitable in cooldown -> still blocked
            ChannelInput(
                channel_id="200x2x0",
                peer_id="02" + "b" * 64,
                capacity_sats=1_000_000,
                local_sats=250_000,
                is_profitable=True,
                cooldown_active=True,
            ),
        ],
        allocations,
        target_emergency_low=0.10,
    )

    severe, moderate = state.channels
    # Severe drift defeats the cooldown gate for the destination role.
    assert severe.dest_eligible is True
    assert severe.dest_reason == ""
    assert severe.cooldown_active is True
    # Moderate drift does not defeat cooldown -- still blocked.
    assert moderate.dest_eligible is False
    assert moderate.dest_reason == "cooldown"


def test_emergency_override_does_not_relax_source_cooldown():
    """Phase 3: drift override is for refill destinations only. Source-side
    cooldown protection (recently drained from this channel) stays in force
    so we don't immediately re-drain a recovering channel."""
    from modules.rebalance_state_v2 import ChannelInput, build_state_snapshot

    state = build_state_snapshot(
        [
            ChannelInput(
                channel_id="300x3x0",
                peer_id="02" + "c" * 64,
                capacity_sats=1_000_000,
                local_sats=950_000,
                cooldown_active=True,
            ),
        ],
        {},
        target_emergency_low=0.10,
    )

    channel = state.channels[0]
    # Source still cooldown-protected even though channel is drainable.
    assert channel.source_eligible is False
    assert channel.source_reason == "cooldown"


def test_explicit_cooldown_override_input_overrides_dest_cooldown():
    """Phase 3.3: when the engine has computed a drift override from anchor
    state, it can pass cooldown_override=True on the channel input. The
    state builder honors that even when the channel is above the emergency
    threshold."""
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
                local_sats=200_000,         # 20% local -- above emergency floor
                is_profitable=True,
                cooldown_active=True,
                cooldown_override=True,     # engine computed drift override
            ),
        ],
        allocations,
        target_emergency_low=0.10,
    )

    channel = state.channels[0]
    assert channel.dest_eligible is True
    assert channel.dest_reason == ""


def test_polar_s7_tiny_oscillation_does_not_trigger_drift_override():
    """Phase 3.4 anti-churn: a channel that wandered slightly below its
    post-rebalance anchor should still respect the cooldown. The override is
    for severe drift, not noise."""
    from modules.capex_budget import CapexAllocations, ChannelCapexBudget
    from modules.rebalance_state_v2 import ChannelInput, build_state_snapshot

    allocations = CapexAllocations(
        channel_budgets={
            "777x7x0": ChannelCapexBudget(channel_id="777x7x0", budget_msat=1_000_000),
        }
    )
    state = build_state_snapshot(
        [
            # Healthy mid-band channel in cooldown after a recent rebalance.
            # Override should NOT fire -- emergency_low is far away (0.45 vs 0.10),
            # and cooldown_override flag is False (no anchor-based drift either).
            ChannelInput(
                channel_id="777x7x0",
                peer_id="02" + "7" * 64,
                capacity_sats=1_000_000,
                local_sats=450_000,
                is_profitable=True,
                cooldown_active=True,
                cooldown_override=False,
            ),
        ],
        allocations,
        target_emergency_low=0.10,
    )

    channel = state.channels[0]
    assert channel.dest_eligible is False
    assert channel.dest_reason == "cooldown"


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


def test_channel_state_has_flow_fact_fields_with_neutral_defaults():
    from modules.rebalance_state_v2 import ChannelState

    ch = ChannelState(
        channel_id="A", peer_id="B", capacity_sats=1_000_000, local_ratio=0.5,
        actual_inbound_fee_ppm=0, value_class="neutral", is_valuable=False,
        remaining_budget_sats=0, cooldown_active=False,
    )
    assert ch.realized_utilization == 0.5
    assert ch.utilization_is_realized is False
    assert ch.activity_out_sats == 0
    assert ch.activity_in_sats == 0
    assert ch.target_band_low == 0.35
    assert ch.target_band_high == 0.65


def test_build_state_snapshot_injects_flow_facts():
    from modules.capex_budget import CapexAllocations, ChannelCapexBudget
    from modules.rebalance_flow_facts import ChannelFlowFacts
    from modules.rebalance_state_v2 import ChannelInput, build_state_snapshot

    allocations = CapexAllocations(
        channel_budgets={
            "A": ChannelCapexBudget(channel_id="A", budget_msat=250_000),
        }
    )
    facts = {
        "A": ChannelFlowFacts(
            channel_id="A", out_sats_window=1000, in_sats_window=0,
            forward_count_window=9, realized_utilization=0.7,
            utilization_is_realized=True,
        )
    }

    state = build_state_snapshot(
        [
            ChannelInput(
                channel_id="A",
                peer_id="02" + "a" * 64,
                capacity_sats=1_000_000,
                local_sats=650_000,
                actual_inbound_fee_ppm=120,
                is_profitable=True,
                is_active=True,
                cooldown_active=False,
            ),
        ],
        allocations,
        flow_facts=facts,
    )

    channel = state.channels[0]
    assert channel.realized_utilization == 0.7
    assert channel.utilization_is_realized is True
    assert channel.activity_out_sats == 1000


def test_build_state_snapshot_injects_target_band():
    from modules.capex_budget import CapexAllocations, ChannelCapexBudget
    from modules.rebalance_state_v2 import ChannelInput, build_state_snapshot

    allocations = CapexAllocations(
        channel_budgets={
            "A": ChannelCapexBudget(channel_id="A", budget_msat=250_000),
        }
    )
    bands = {"A": (0.20, 0.80)}

    state = build_state_snapshot(
        [
            ChannelInput(
                channel_id="A",
                peer_id="02" + "a" * 64,
                capacity_sats=1_000_000,
                local_sats=650_000,
                actual_inbound_fee_ppm=120,
                is_profitable=True,
                is_active=True,
                cooldown_active=False,
            ),
        ],
        allocations,
        target_bands=bands,
    )

    channel = state.channels[0]
    assert channel.target_band_low == 0.20
    assert channel.target_band_high == 0.80


def test_build_state_snapshot_band_falls_back_to_cfg_thresholds():
    """When no target_bands map is given but a cfg is, the per-channel band
    falls back to cfg.low_liquidity_threshold / high_liquidity_threshold."""
    from types import SimpleNamespace
    from modules.capex_budget import CapexAllocations, ChannelCapexBudget
    from modules.rebalance_state_v2 import ChannelInput, build_state_snapshot

    allocations = CapexAllocations(
        channel_budgets={"A": ChannelCapexBudget(channel_id="A", budget_msat=250_000)}
    )
    cfg = SimpleNamespace(low_liquidity_threshold=0.25, high_liquidity_threshold=0.75)

    state = build_state_snapshot(
        [
            ChannelInput(
                channel_id="A", peer_id="02" + "a" * 64, capacity_sats=1_000_000,
                local_sats=650_000, actual_inbound_fee_ppm=120,
                is_profitable=True, is_active=True, cooldown_active=False,
            ),
        ],
        allocations,
        cfg=cfg,
    )
    channel = state.channels[0]
    assert channel.target_band_low == 0.25
    assert channel.target_band_high == 0.75


def test_build_state_snapshot_scores_use_per_channel_band():
    """FIX 2(b) coherence: dest_urgency/source_drain_score must be computed
    against the channel's OWN per-channel band (from target_bands), not the
    flat function-scope target_band_low/high kwargs -- otherwise a channel
    classified "inside its own wider band" would still carry a nonzero
    source_drain_score/dest_urgency computed against the narrower flat band,
    which is incoherent with classification (Task 7)."""
    from modules.capex_budget import CapexAllocations, ChannelCapexBudget
    from modules.rebalance_state_v2 import ChannelInput, build_state_snapshot

    allocations = CapexAllocations(
        channel_budgets={"A": ChannelCapexBudget(channel_id="A", budget_msat=250_000)}
    )
    bands = {"A": (0.20, 0.80)}

    state = build_state_snapshot(
        [
            ChannelInput(
                channel_id="A",
                peer_id="02" + "a" * 64,
                capacity_sats=1_000_000,
                # ratio 0.75: above the flat high (0.65) but comfortably
                # inside this channel's own wider band (0.20, 0.80).
                local_sats=750_000,
                is_profitable=True,
                is_active=True,
            ),
        ],
        allocations,
        target_band_low=0.35,
        target_band_high=0.65,
        target_bands=bands,
    )

    channel = state.channels[0]
    assert channel.local_ratio == 0.75
    # Against the OWN band (0.20, 0.80) the channel is inside band: zero
    # drain score / urgency, NOT the flat-band answer (source_drain_score
    # would be > 0 against the narrower flat 0.65 high bound).
    assert channel.source_drain_score == 0.0
    assert channel.dest_urgency == 0.0


def test_build_state_snapshot_scores_fall_back_to_flat_band_when_no_per_channel_band():
    """Invariant: when a channel has no per-channel band entry, dest_urgency/
    source_drain_score must fall back to the flat function-scope
    target_band_low/high kwargs -- identical to pre-feature behavior."""
    from modules.capex_budget import CapexAllocations, ChannelCapexBudget
    from modules.rebalance_state_v2 import ChannelInput, build_state_snapshot

    allocations = CapexAllocations(
        channel_budgets={"A": ChannelCapexBudget(channel_id="A", budget_msat=250_000)}
    )

    state = build_state_snapshot(
        [
            ChannelInput(
                channel_id="A",
                peer_id="02" + "a" * 64,
                capacity_sats=1_000_000,
                local_sats=750_000,  # ratio 0.75, above flat high 0.65
                is_profitable=True,
                is_active=True,
            ),
        ],
        allocations,
        target_band_low=0.35,
        target_band_high=0.65,
        # no target_bands map -- must use the flat kwargs.
    )

    channel = state.channels[0]
    assert channel.local_ratio == 0.75
    assert channel.source_drain_score > 0.0
    assert channel.dest_urgency == 0.0
