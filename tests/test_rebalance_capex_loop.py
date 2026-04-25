import importlib.util
import sys
from pathlib import Path


def load_loop():
    repo = Path(__file__).resolve().parents[1]
    tools_dir = repo / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    path = tools_dir / "rebalance_capex_loop.py"
    spec = importlib.util.spec_from_file_location("rebalance_capex_loop", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prepare_pure_hive_corridor_returns_structured_perturbation(monkeypatch):
    loop = load_loop()
    node_ids = {
        "revenue-node": "rev-id",
        "source": "source-id",
        "relay": "relay-id",
    }
    channels = {
        ("revenue-node", "source-id"): {
            "short_channel_id": "100x1x0",
            "to_us_msat": 700_000_000,
            "total_msat": 1_000_000_000,
        },
        ("revenue-node", "relay-id"): {
            "short_channel_id": "200x1x0",
            "to_us_msat": 300_000_000,
            "total_msat": 1_000_000_000,
        },
        ("source", "relay-id"): {
            "short_channel_id": "300x1x0",
            "to_us_msat": 270_000_000,
            "total_msat": 1_000_000_000,
        },
    }

    monkeypatch.setattr(loop, "_node_id", lambda node: node_ids[node])
    monkeypatch.setattr(
        loop,
        "_peer_channel",
        lambda node, peer_id: channels.get((node, peer_id), {}),
    )
    monkeypatch.setattr(
        loop,
        "_set_channel_policy",
        lambda **kwargs: {"ok": True, **kwargs},
    )
    monkeypatch.setattr(
        loop,
        "_pay_between",
        lambda **kwargs: {"ok": True, **kwargs},
    )
    monkeypatch.setattr(loop.tournament, "rpc_result_ok", lambda result: result.get("ok", False))

    result = loop.prepare_pure_hive_corridor(
        path_nodes=["source", "relay"],
        source_target_ratio=0.98,
        dest_target_ratio=0.02,
        corridor_target_ratio=0.55,
        corridor_base_msat=1,
        corridor_fee_ppm=1,
    )

    assert result["ok"] is True
    assert result["nodes"]["path_nodes"] == ["source", "relay"]
    assert result["ids"]["source"] == "source-id"
    assert result["targets"]["corridor_target_ratio"] == 0.55
    assert result["before"]["revenue_source"]["short_channel_id"] == "100x1x0"
    assert result["payments"]["source_to_revenue_source_balance"]["amount_sats"] == 280_000
    assert result["payments"]["revenue_to_dest_dest_balance"]["amount_sats"] == 280_000
    assert result["payments"]["corridor_1_source_to_relay_liquidity"]["amount_sats"] == 280_000


def test_revenue_config_overrides_are_transient(monkeypatch):
    loop = load_loop()
    calls = []

    def fake_cln(*args):
        calls.append(args)
        return {"ok": True}

    monkeypatch.setattr(loop.tournament, "cln", fake_cln)

    result = loop.set_revenue_config_overrides(
        {
            "revenue-ops-pair-fee-cap-ppm": 5000,
            "revenue-ops-rebalance-hold-margin": 0.2,
        }
    )

    assert result["ok"] is True
    assert result["requires_restart"] is False
    assert calls == [
        (
            "revenue-node",
            "setconfig",
            "config=revenue-ops-pair-fee-cap-ppm",
            "val=5000",
            "transient=true",
        ),
        (
            "revenue-node",
            "setconfig",
            "config=revenue-ops-rebalance-hold-margin",
            "val=0.2",
            "transient=true",
        ),
    ]


def test_hive_hints_disabled_uses_transient_config(monkeypatch):
    loop = load_loop()
    calls = []

    def fake_cln(*args):
        calls.append(args)
        return {"ok": True}

    monkeypatch.setattr(loop.tournament, "cln", fake_cln)

    result = loop.set_hive_hints_disabled()

    assert result["ok"] is True
    assert calls == [
        (
            "revenue-node",
            "setconfig",
            "revenue-ops-hive-hints-enabled",
            "false",
            "true",
        )
    ]


def test_pay_between_uses_single_hop_direct_sendpay(monkeypatch):
    loop = load_loop()
    calls = []
    route = [{"id": "payee-id", "channel": "1x1x0", "amount_msat": 5000000, "delay": 9}]

    def fake_cln(node, *args, **kwargs):
        calls.append((node, args, kwargs))
        method = args[0]
        if method == "invoice":
            return {
                "ok": True,
                "bolt11": "lnbcrt...",
                "payment_hash": "hash",
                "payment_secret": "secret",
            }
        if method == "connect":
            return {"ok": True}
        if method == "getroute":
            return {"ok": True, "route": route}
        if method == "sendpay":
            return {"ok": True, "status": "pending"}
        if method == "waitsendpay":
            return {"ok": True, "status": "complete"}
        raise AssertionError(method)

    monkeypatch.setattr(loop, "_node_id", lambda node: f"{node}-id")
    monkeypatch.setattr(loop.tournament, "cln", fake_cln)
    monkeypatch.setattr(loop.tournament, "rpc_result_ok", lambda result: result.get("ok", False))

    result = loop._pay_between(
        payer="payer",
        payee="payee",
        amount_sats=5000,
        label_prefix="corridor-test",
    )

    assert result["ok"] is True
    assert calls[1][1] == ("connect", "payee-id@payee:9735")
    assert calls[2][1] == (
        "getroute",
        "id=payee-id",
        "amount_msat=5000000",
        "riskfactor=1",
        "maxhops=1",
    )
    assert calls[3][1][0] == "sendpay"
    assert 'route=[{"id":"payee-id","channel":"1x1x0","amount_msat":5000000,"delay":9}]' in calls[3][1]
    assert calls[4][1] == ("waitsendpay", "hash", "45")


def test_pay_between_falls_back_to_manual_peer_channel(monkeypatch):
    loop = load_loop()
    calls = []

    def fake_cln(node, *args, **kwargs):
        calls.append((node, args, kwargs))
        method = args[0]
        if method == "invoice":
            return {
                "ok": True,
                "bolt11": "lnbcrt...",
                "payment_hash": "hash",
                "payment_secret": "secret",
            }
        if method == "connect":
            return {"ok": True}
        if method == "getroute":
            return {"ok": False, "message": "Shortest route was 2"}
        if method == "sendpay":
            return {"ok": True, "status": "pending"}
        if method == "waitsendpay":
            return {"ok": True, "status": "complete"}
        raise AssertionError(method)

    monkeypatch.setattr(loop, "_node_id", lambda node: f"{node}-id")
    monkeypatch.setattr(
        loop,
        "_peer_channel",
        lambda node, peer_id: {"short_channel_id": "9x1x0"},
    )
    monkeypatch.setattr(loop.tournament, "cln", fake_cln)
    monkeypatch.setattr(loop.tournament, "rpc_result_ok", lambda result: result.get("ok", False))

    result = loop._pay_between(
        payer="payer",
        payee="payee",
        amount_sats=7000,
        label_prefix="corridor-test",
    )

    assert result["ok"] is True
    assert result["route"]["fallback"] == "manual_single_hop_peer_channel"
    assert 'route=[{"id":"payee-id","channel":"9x1x0","amount_msat":7000000,"delay":9,"style":"tlv"}]' in calls[3][1]
