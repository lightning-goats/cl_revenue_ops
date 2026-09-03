"""Acceptance suite for the post-remediation rebalancer.

Phase 6 Steps 1, 2, 4 of docs/superpowers/plans/2026-04-18-rebalancer-post-polar-remediation.md.

These tests encode the lab scenarios that drove the remediation and pin
the acceptance criteria in the plan against the unit-testable surface.

Step 3 (live lab reruns) is operator-only and not covered here.
"""

from modules.capex_budget import CapexAllocations, ChannelCapexBudget
from modules.rebalance_planner_v2 import RebalancePlanner
from modules.rebalance_state_v2 import ChannelInput, build_state_snapshot


def _allocations(*channel_ids):
    return CapexAllocations(
        channel_budgets={
            cid: ChannelCapexBudget(channel_id=cid, budget_msat=1_000_000)
            for cid in channel_ids
        }
    )


# ---------------------------------------------------------------------------
# S2: sink-pressure -- depleted profitable + over-local neutrals.
# Pre-fix: planner discarded all neutral over-local channels as not_valuable
# and the depleted profitable destination ended on no_partner.
# Post-fix (Phase 2): a neutral over-local channel is a valid drain source
# and a pair forms.
# ---------------------------------------------------------------------------

def test_s2_sink_pressure_forms_pair():
    snapshot = build_state_snapshot(
        [
            ChannelInput(  # depleted profitable -- needs refill
                channel_id="159x1x0",
                peer_id="02" + "1" * 64,
                capacity_sats=1_000_000,
                local_sats=100_000,
                is_profitable=True,
            ),
            ChannelInput(  # extreme over-local neutral -- now a valid source
                channel_id="243x1x0",
                peer_id="02" + "2" * 64,
                capacity_sats=1_000_000,
                local_sats=950_000,
            ),
            ChannelInput(
                channel_id="255x1x0",
                peer_id="02" + "3" * 64,
                capacity_sats=1_000_000,
                local_sats=950_000,
            ),
        ],
        _allocations("159x1x0"),
    )

    plan = RebalancePlanner().plan(snapshot)

    assert len(plan.selected) == 1
    pair = plan.selected[0]
    assert pair.dest_channel_id == "159x1x0"
    assert pair.source_channel_id in {"243x1x0", "255x1x0"}


# ---------------------------------------------------------------------------
# S9: rebalance provocation -- 6.6% local profitable in cooldown + 100%-local
# neutrals.
# Pre-fix: depleted destination blocked by blanket cooldown, neutrals
# rejected as not_valuable. considered_pairs=0.
# Post-fix (Phase 2 + Phase 3): emergency-local override unblocks the
# destination and a pair forms with one of the neutral sources.
# ---------------------------------------------------------------------------

def test_s9_emergency_drift_unblocks_cooldown():
    snapshot = build_state_snapshot(
        [
            ChannelInput(  # depleted profitable in cooldown (drift-overridden)
                channel_id="123x1x0",
                peer_id="02" + "9" * 64,
                capacity_sats=1_000_000,
                local_sats=66_000,        # 6.6% local
                is_profitable=True,
                cooldown_active=True,
            ),
            ChannelInput(  # 100% local neutral source
                channel_id="200x2x0",
                peer_id="02" + "8" * 64,
                capacity_sats=1_000_000,
                local_sats=1_000_000,
            ),
            ChannelInput(
                channel_id="201x2x0",
                peer_id="02" + "7" * 64,
                capacity_sats=1_000_000,
                local_sats=1_000_000,
            ),
        ],
        _allocations("123x1x0"),
        target_emergency_low=0.10,
    )

    plan = RebalancePlanner().plan(snapshot)

    assert len(plan.selected) == 1
    pair = plan.selected[0]
    assert pair.dest_channel_id == "123x1x0"


# ---------------------------------------------------------------------------
# S7: capital-burn trap -- tiny oscillation around mid-band.
# Acceptance: no low-value churn. The pair either is not formed (no over-band
# drift) or scores below the hold margin in the engine.
# ---------------------------------------------------------------------------

def test_s7_oscillation_does_not_form_pair():
    """Mid-band drift between 0.45 and 0.55 should not produce a candidate
    -- both channels stay inside the band."""
    snapshot = build_state_snapshot(
        [
            ChannelInput(
                channel_id="aaa1x1x0",
                peer_id="02" + "a" * 64,
                capacity_sats=1_000_000,
                local_sats=550_000,
                is_profitable=True,
            ),
            ChannelInput(
                channel_id="bbb1x1x0",
                peer_id="02" + "b" * 64,
                capacity_sats=1_000_000,
                local_sats=450_000,
                is_profitable=True,
            ),
        ],
        _allocations("aaa1x1x0", "bbb1x1x0"),
    )

    plan = RebalancePlanner().plan(snapshot)

    assert plan.selected == []
    assert all(s.reason == "inside_band" for s in plan.skipped)


# ---------------------------------------------------------------------------
# S0: idle / quiet conditions -- balanced channels, nothing to do.
# Acceptance: no pointless autonomous rebalances.
# ---------------------------------------------------------------------------

def test_s0_idle_state_yields_no_pairs():
    snapshot = build_state_snapshot(
        [
            ChannelInput(
                channel_id="quiet1",
                peer_id="02" + "q" * 64,
                capacity_sats=1_000_000,
                local_sats=500_000,
                is_profitable=True,
            ),
        ],
        _allocations("quiet1"),
    )

    plan = RebalancePlanner().plan(snapshot)

    assert plan.selected == []
