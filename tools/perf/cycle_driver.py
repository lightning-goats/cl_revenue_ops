"""
Synthetic driver for profiling the fee / profitability / rebalance cycles.

Builds the real module stack (Config, Database, PolicyManager,
ChannelProfitabilityAnalyzer, FeeController, EVRebalancer + RebalanceEngine)
over a synthetic DataService whose underlying RPC returns a fixed 36-channel
topology. This lets cProfile see the real per-channel CPU work of each cycle
at production DB scale, without a live node.

The cycles are RPC-coupled; any path the synthetic RPC cannot satisfy raises
and is reported by the caller as an error row (the DB-read baseline remains
authoritative).
"""

import time
from typing import Any, Dict, List
from unittest.mock import MagicMock

N_CHANNELS = 36
DAY = 86_400


from modules.fee_authority import FeeAuthorityGate

def _scids(n: int) -> List[str]:
    return [f"{800000 + i}x{i}x0" for i in range(n)]


def _peers(n: int) -> List[str]:
    return ["02" + f"{i:064x}"[:64] for i in range(n)]


def _peer_channels_payload() -> Dict[str, Any]:
    scids = _scids(N_CHANNELS)
    peers = _peers(N_CHANNELS)
    channels = []
    for i, (scid, peer) in enumerate(zip(scids, peers)):
        cap = 5_000_000_000  # msat
        spend = int(cap * (0.2 + 0.6 * ((i % 5) / 5.0)))
        channels.append({
            "short_channel_id": scid,
            "channel_id": f"{i:064x}",
            "state": "CHANNELD_NORMAL",
            "peer_id": peer,
            "total_msat": cap,
            "spendable_msat": spend,
            "receivable_msat": cap - spend,
            "to_us_msat": spend,
            "funding_txid": f"{i:064x}",
            "opener": "local" if i % 2 == 0 else "remote",
            "fee_base_msat": 1000,
            "fee_proportional_millionths": 100 + i * 5,
            "htlc_minimum_msat": 1000,
            "htlc_maximum_msat": cap,
            "max_accepted_htlcs": 483,
            "htlcs": [],
            "updates": {"local": {
                "fee_base_msat": 1000,
                "fee_proportional_millionths": 100 + i * 5,
                "htlc_minimum_msat": 1000,
                "htlc_maximum_msat": cap,
            }},
            "peer_connected": True,
        })
    return {"channels": channels}


def _gossip_channels_payload(peers: List[str]) -> Dict[str, Any]:
    scids = _scids(N_CHANNELS)
    out = []
    for i, (scid, peer) in enumerate(zip(scids, peers)):
        out.append({
            "source": peer,
            "destination": peers[(i + 1) % len(peers)],
            "short_channel_id": scid,
            "base_fee_millisatoshi": 1000,
            "fee_per_millionth": 200 + i,
            "active": True,
            "amount_msat": 5_000_000_000,
            "htlc_minimum_msat": 1000,
            "htlc_maximum_msat": 4_900_000_000,
        })
    return {"channels": out}


def _make_rpc():
    scids = _scids(N_CHANNELS)
    peers = _peers(N_CHANNELS)
    node_id = peers[0]
    rpc = MagicMock()
    rpc.getinfo.return_value = {
        "id": node_id, "alias": "perf-node", "network": "bitcoin",
        "blockheight": 860000, "version": "v26.06.1",
    }
    rpc.listconfigs.return_value = {"configs": {}}
    rpc.listpeerchannels.return_value = _peer_channels_payload()
    rpc.listpeers.return_value = {"peers": [{"id": p, "connected": True} for p in peers]}
    rpc.listfunds.return_value = {
        "channels": [], "outputs": [{"amount_msat": 10_000_000, "status": "confirmed"}],
    }
    rpc.listforwards.return_value = {"forwards": []}
    rpc.listnodes.return_value = {"nodes": []}
    rpc.feerates.return_value = {"perkb": {"opening": 10000}, "onchain_fee_estimates": {}}
    rpc.setchannel.return_value = {"channels": []}

    def _listchannels(**kwargs):
        return _gossip_channels_payload(peers)
    rpc.listchannels.side_effect = lambda *a, **k: _gossip_channels_payload(peers)

    def _call(method, payload=None, *a, **k):
        if method == "listclosedchannels":
            return {"closedchannels": []}
        if method == "askrene-listlayers":
            return {"layers": []}
        if method in ("bkpr-listaccountevents", "bkpr-listbalances", "bkpr-inspect"):
            return {"events": [], "accounts": []}
        if method == "getroutes":
            return {"routes": []}
        return {}
    rpc.call.side_effect = _call
    return rpc


def build_stack(database) -> Dict[str, Any]:
    """Construct the real cycle objects over a synthetic RPC. Returns a dict
    with keys profitability / fee_controller / rebalance_engine (values may be
    None if construction fails)."""
    from modules.config import Config
    from modules.data_service import DataService
    from modules.policy_manager import PolicyManager
    from modules.profitability_analyzer import ChannelProfitabilityAnalyzer
    from modules.fee_controller import FeeController

    plugin = MagicMock()
    plugin.log = MagicMock()
    plugin.rpc = _make_rpc()

    config = Config()
    data_service = DataService(plugin)

    stack: Dict[str, Any] = {"profitability": None, "fee_controller": None,
                             "rebalance_engine": None}

    policy_manager = PolicyManager(database, plugin)

    profitability = ChannelProfitabilityAnalyzer(plugin, config, database)
    profitability.data_service = data_service
    stack["profitability"] = profitability

    try:
        fc = FeeController(plugin, config, database, policy_manager, profitability, fee_authority_gate=FeeAuthorityGate())
        fc.data_service = data_service
        stack["fee_controller"] = fc
    except Exception:
        stack["fee_controller"] = None

    try:
        from modules.rebalancer import EVRebalancer
        from modules.rebalance_engine_v2 import RebalanceEngine
        rb = EVRebalancer(plugin, config, database, policy_manager)
        rb.set_profitability_analyzer(profitability)
        rb.data_service = data_service
        engine = RebalanceEngine(
            plugin=plugin, config=config, database=database,
            capex_engine=None, profitability=profitability,
            hive_hints=None, data_service=data_service, hive_router=None,
            segment_observation_store=None,
        )
        stack["rebalance_engine"] = engine
    except Exception:
        stack["rebalance_engine"] = None

    return stack
