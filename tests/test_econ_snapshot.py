"""Phase 1: canonical EconomicSnapshot types, builder, canonical JSON."""
import dataclasses
import json
import pathlib
from types import SimpleNamespace

import pytest

from modules.econ_snapshot import (
    LIFECYCLES,
    ROLES,
    BudgetState,
    ChannelSnapshot,
    EconomicSnapshot,
    NodeState,
    Protection,
    build_channel_snapshot,
    canonical_json,
    to_wire,
)
from modules.econ_types import (
    ChannelId,
    EconArithmeticError,
    Micro,
    Msat,
    PeerId,
    SignedMsat,
    UnixTime,
)

SCHEMA_PATH = (pathlib.Path(__file__).resolve().parent.parent
               / "schemas" / "economic_snapshot.v0.schema.json")

PEER = "02" + "a" * 64


def _chan(cid="123x456x0", **over):
    base = dict(
        channel_id=ChannelId(cid),
        peer_id=PeerId(PEER),
        capacity_msat=Msat(2_000_000_000_000),
        local_msat=Msat(1_200_000_000_000),
        remote_msat=Msat(800_000_000_000),
        spendable_msat=Msat(1_180_000_000_000),
        receivable_msat=Msat(780_000_000_000),
        exit_revenue_msat=Msat(2_000_000),
        sourced_value_msat=Msat(1_500_000),
        rebalance_cost_msat=Msat(800_000),
        capital_cost_msat=Msat(400_000),
        net_value_msat=SignedMsat(2_300_000),
        exit_volume_msat=Msat(900_000_000_000),
        sourced_volume_msat=Msat(700_000_000_000),
        forward_count=142,
        sourced_forward_count=96,
        role="ROUTER",
        lifecycle="PRODUCTIVE",
        protections=(Protection("lnplus_contract", "lnplus",
                                UnixTime(1_755_000_000)),),
        confidence_micro=Micro(850_000),
    )
    base.update(over)
    return ChannelSnapshot(**base)


def _node():
    return NodeState(
        total_local_msat=Msat(500_000_000_000),
        total_remote_msat=Msat(300_000_000_000),
        receivable_objective_msat=Msat(400_000_000_000),
        onchain_confirmed_msat=Msat(100_000_000_000),
        reserved_msat=Msat(5_000_000_000),
        daily_budget=BudgetState(cap_msat=Msat(10_000_000_000),
                                 reserved_msat=Msat(2_000_000_000),
                                 spent_msat=Msat(1_000_000_000)),
        pending_operations=(),
        external_obligations=(),
    )


def _snap(channels):
    return EconomicSnapshot(
        snapshot_id="cycle-000001",
        observed_at=UnixTime(1_752_300_000),
        evidence_window_seconds=2_592_000,
        node=_node(),
        channels=tuple(channels),
    )


def test_frozen_and_role_validation():
    snap = _snap([_chan()])
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.snapshot_id = "x"
    with pytest.raises(EconArithmeticError):
        _chan(role="NOT_A_ROLE")
    with pytest.raises(EconArithmeticError):
        _chan(lifecycle="NOT_A_STATE")
    assert "INBOUND_GATEWAY" in ROLES
    assert "RECYCLING" in LIFECYCLES


def test_channels_sorted_by_channel_id():
    snap = _snap([_chan("900x1x0"), _chan("100x1x0"), _chan("500x1x0")])
    ids = [c.channel_id.value for c in snap.channels]
    assert ids == sorted(ids)


def test_to_wire_validates_against_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())
    wire = to_wire(_snap([_chan()]))
    jsonschema.validate(wire, schema)
    assert wire["schema_name"] == "economic_snapshot"
    assert wire["schema_version"] == 0
    assert wire["channels"][0]["role"] == "ROUTER"
    assert wire["channels"][0]["capacity_msat"] == 2_000_000_000_000


def test_canonical_json_order_insensitive():
    a = {"b": 2, "a": {"y": 1, "x": [1, 2]}}
    b = {}
    b["a"] = {}
    b["a"]["x"] = [1, 2]
    b["a"]["y"] = 1
    b["b"] = 2
    assert canonical_json(a) == canonical_json(b)
    assert canonical_json(a) == '{"a":{"x":[1,2],"y":1},"b":2}'


def _prof_like(fees_earned_sats=2000, sourced_sats=500,
               rebalance_cost_sats=1000, open_cost_sats=500,
               net_profit_sats=1500):
    return SimpleNamespace(
        revenue=SimpleNamespace(
            fees_earned_msat=fees_earned_sats * 1000,
            sourced_fee_contribution_msat=sourced_sats * 1000,
            volume_routed_msat=1_000_000_000,
            forward_count=100,
        ),
        costs=SimpleNamespace(
            rebalance_cost_sats=rebalance_cost_sats,
            open_cost_sats=open_cost_sats,
        ),
        net_profit_sats=net_profit_sats,
        sourced_forward_count_30d=40,
    )


def _lpc_channel():
    """listpeerchannels-shaped normalized dict (ints are msat)."""
    return {
        "short_channel_id": "123x456x0",
        "peer_id": PEER,
        "total_msat": 2_000_000_000,
        "to_us_msat": 1_200_000_000,
        "spendable_msat": 1_180_000_000,
        "receivable_msat": 780_000_000,
    }


def test_builder_maps_profitability():
    cs = build_channel_snapshot(channel=_lpc_channel(), prof=_prof_like(),
                                flow_confidence=0.85, role="ROUTER")
    assert cs.exit_revenue_msat == Msat(2_000_000)
    assert cs.sourced_value_msat == Msat(500_000)
    assert cs.rebalance_cost_msat == Msat(1_000_000)
    assert cs.capital_cost_msat == Msat(500_000)
    assert cs.net_value_msat == SignedMsat(1_500_000)
    assert cs.remote_msat == Msat(800_000_000)  # capacity - local
    assert cs.forward_count == 100
    assert cs.sourced_forward_count == 40
    assert cs.confidence_micro == Micro(850_000)


def test_builder_without_prof_yields_zero_economics():
    cs = build_channel_snapshot(channel=_lpc_channel())
    assert cs.exit_revenue_msat == Msat(0)
    assert cs.net_value_msat == SignedMsat(0)
    assert cs.role == "UNKNOWN"
    assert cs.confidence_micro == Micro(0)
