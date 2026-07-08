import os
import sys
import tempfile
import time
from unittest.mock import MagicMock

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import Database
from modules.config import Config


def _make_db():
    path = os.path.join(tempfile.mkdtemp(prefix="lnplus_test_"), "test.db")
    db = Database(path, MagicMock())
    db.initialize()
    return db


class TestLnplusSwapTables:
    def test_record_and_get_swap(self):
        db = _make_db()
        db.lnplus_record_swap("s123", "applied", 5_000_000, 3,
                              outbound_peer="02" + "ab" * 32,
                              our_identifier="C", planner_action_id=7)
        row = db.lnplus_get_swap("s123")
        assert row["status"] == "applied"
        assert row["capacity_sats"] == 5_000_000
        assert row["duration_months"] == 3
        assert row["our_identifier"] == "C"
        assert row["planner_action_id"] == 7
        assert row["applied_at"] > 0

    def test_update_swap_status_and_fields(self):
        db = _make_db()
        db.lnplus_record_swap("s1", "applied", 2_000_000, 3)
        db.lnplus_update_swap("s1", status="opening",
                              deadline_at=int(time.time()) + 172800,
                              incoming_peer="03" + "cd" * 32)
        row = db.lnplus_get_swap("s1")
        assert row["status"] == "opening"
        assert row["deadline_at"] > time.time()
        assert row["incoming_peer"].startswith("03")

    def test_update_rejects_unknown_field(self):
        db = _make_db()
        db.lnplus_record_swap("s1", "applied", 2_000_000, 3)
        try:
            db.lnplus_update_swap("s1", bogus_column="x")
            assert False, "should have raised"
        except ValueError:
            pass

    def test_inflight_and_reservation(self):
        db = _make_db()
        db.lnplus_record_swap("s1", "applied", 2_000_000, 3)
        db.lnplus_record_swap("s2", "opening", 3_000_000, 3)
        db.lnplus_record_swap("s3", "active", 9_000_000, 3)
        db.lnplus_record_swap("s4", "ended", 1_000_000, 3)
        inflight = db.lnplus_inflight_swaps()
        assert {r["swap_id"] for r in inflight} == {"s1", "s2"}
        assert db.lnplus_reserved_sats() == 5_000_000

    def test_get_swaps_by_status(self):
        db = _make_db()
        db.lnplus_record_swap("s1", "active", 2_000_000, 3)
        rows = db.lnplus_get_swaps_by_status(["active"])
        assert len(rows) == 1 and rows[0]["swap_id"] == "s1"

    def test_peer_bump_and_get(self):
        db = _make_db()
        pk = "02" + "ee" * 32
        assert db.lnplus_get_peer(pk) is None
        db.lnplus_bump_peer(pk)
        db.lnplus_bump_peer(pk, defection=True, rating="negative")
        peer = db.lnplus_get_peer(pk)
        assert peer["swaps_count"] == 2
        assert peer["defections"] == 1
        assert peer["ratings_given_negative"] == 1
        assert peer["ratings_given_positive"] == 0


class TestLnplusConfig:
    def test_defaults(self):
        cfg = Config()
        assert cfg.lnplus_swaps_enabled is True
        assert cfg.lnplus_execute_applications is True
        assert cfg.lnplus_swap_preference_margin == 0.2
        assert cfg.lnplus_max_duration_months == 3
        assert cfg.lnplus_min_peer_positive_ratings == 5
        assert cfg.lnplus_max_participants == 4
        assert cfg.lnplus_apply_feerate_ceiling == 5000
        assert cfg.lnplus_pending_timeout_days == 7
        assert cfg.lnplus_inbound_credit_factor == 0.5
        assert cfg.lnplus_fleet_pubkeys == ''
        assert cfg.lnplus_watcher_interval == 3600

    def test_public_runtime_keys(self):
        cfg = Config()
        for key in ("lnplus_swaps_enabled", "lnplus_execute_applications",
                    "lnplus_swap_preference_margin", "lnplus_inbound_credit_factor",
                    "lnplus_apply_feerate_ceiling", "lnplus_max_duration_months",
                    "lnplus_min_peer_positive_ratings"):
            assert Config.is_public_runtime_key(key), key

    def test_runtime_update_roundtrip(self):
        db = _make_db()
        cfg = Config()
        result = cfg.update_runtime(db, "lnplus_execute_applications", "false")
        assert result.get("status") == "success"
        assert cfg.lnplus_execute_applications is False

    def test_margin_range_rejected(self):
        db = _make_db()
        cfg = Config()
        result = cfg.update_runtime(db, "lnplus_swap_preference_margin", "-5")
        assert "error" in result or result.get("status") != "success"


import json as _json
from unittest.mock import patch

from modules.lnplus_swaps import (
    LNPlusClient, LNPlusError, _valid_pubkey, _parse_ts,
)


class _FakeHTTPResponse:
    def __init__(self, payload):
        self._raw = _json.dumps(payload).encode()
    def read(self, n=-1):
        out, self._raw = self._raw[:n if n and n > 0 else len(self._raw)], b""
        return out
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def _make_client(urlopen_payloads):
    """urlopen_payloads: list of payloads returned per successive HTTP call."""
    plugin = MagicMock()
    rpc = MagicMock()
    rpc.signmessage.return_value = {"zbase": "d75qtmgijm79rpooshmgzjwji9gj7dsdat8remuskyjp9oq0ygdd"}
    client = LNPlusClient(plugin, rpc)
    responses = [_FakeHTTPResponse(p) for p in urlopen_payloads]
    patcher = patch("modules.lnplus_swaps.urllib.request.urlopen",
                    side_effect=responses)
    return client, rpc, patcher


class TestLnplusClient:
    def test_helpers(self):
        assert _valid_pubkey("02" + "ab" * 32)
        assert not _valid_pubkey("xx")
        assert not _valid_pubkey(None)
        assert _parse_ts("2026-08-01T00:00:00Z") == 1785542400
        assert _parse_ts(1785542400) == 1785542400
        assert _parse_ts("garbage") is None

    def test_auth_flow_signs_challenge(self):
        challenge = {"message": "lnplus-auth-abc123", "expires_at": "2099-01-01T00:00:00Z"}
        client, rpc, patcher = _make_client([challenge, {"pending": [], "opening": [], "completed": []}])
        with patcher:
            result = client.get_my_swaps()
        rpc.signmessage.assert_called_once_with("lnplus-auth-abc123")
        assert result["pending"] == []

    def test_auth_rejects_suspicious_challenge(self):
        # Never sign something that looks like an invoice.
        challenge = {"message": "lnbc1500n1p...", "expires_at": "2099-01-01T00:00:00Z"}
        client, rpc, patcher = _make_client([challenge])
        with patcher:
            try:
                client.get_my_swaps()
                assert False, "should refuse to sign invoice-like challenge"
            except LNPlusError:
                pass
        rpc.signmessage.assert_not_called()

    def test_http_error_wrapped(self):
        client = LNPlusClient(MagicMock(), MagicMock())
        import urllib.error
        err = urllib.error.HTTPError("u", 422, "Unprocessable", {}, None)
        err.read = lambda n=-1: b'{"errors": {"id": ["already applied"]}}'
        with patch("modules.lnplus_swaps.urllib.request.urlopen", side_effect=err):
            try:
                client.get_swap("s1")
                assert False
            except LNPlusError as e:
                assert e.http_status == 422

    def test_get_swap_id_cannot_escape_path(self):
        client = LNPlusClient(MagicMock(), MagicMock())
        captured = {}
        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            return _FakeHTTPResponse({"id": "x"})
        with patch("modules.lnplus_swaps.urllib.request.urlopen", side_effect=fake_urlopen):
            client.get_swap("../../evil")
        assert "/api/2/get_swap/id=..%2F..%2Fevil" in captured["url"]


from modules.lnplus_swaps import SwapEvaluator, NEG_RATIO_MAX, TOR_RELIABILITY

PK_A = "02" + "aa" * 32
PK_B = "03" + "bb" * 32
FLEET_PK = "02" + "ff" * 32


def _swap_fixture(**overrides):
    swap = {
        "id": "sw1", "status": "pending",
        "capacity_sats": 5_000_000, "duration_months": 3,
        "participant_max_count": 3,
        "participant_applied_count": 2,
        "participant_waiting_for_count": 1,
        "clearnet_connection_allowed": True,
        "tor_connection_allowed": True,
        "platform": "any",
        "participants": [
            {"participant_identifier": "A", "pubkey": PK_A,
             "positive_ratings_count": 20, "negative_ratings_count": 0,
             "address_1": "1.2.3.4:9735", "capacity_sats": 100_000_000,
             "channels_count": 40},
            {"participant_identifier": "B", "pubkey": PK_B,
             "positive_ratings_count": 12, "negative_ratings_count": 1,
             "address_1": "5.6.7.8:9735", "capacity_sats": 80_000_000,
             "channels_count": 25},
        ],
    }
    swap.update(overrides)
    return swap


def _make_evaluator(cfg_overrides=None, swaps=None, inflight=False, breaker=None):
    cfg = Config()
    for k, v in (cfg_overrides or {}).items():
        setattr(cfg, k, v)
    plugin, rpc = MagicMock(), MagicMock()
    rpc.feerates.return_value = {"perkw": {"opening": 2500}}
    rpc.listfunds.return_value = {"outputs": [
        {"amount_msat": 100_000_000_000, "status": "confirmed", "reserved": False}]}
    db = _make_db()
    client = MagicMock()
    client.get_applicable_swaps.return_value = swaps if swaps is not None else [_swap_fixture()]
    planner = MagicMock()
    planner._calculate_open_ev.return_value = 1000.0
    planner._estimate_open_cost.return_value = 2000
    planner._capex_engine.get_fleet_exploration_budget.return_value = 1_000_000_000
    lifecycle = MagicMock()
    lifecycle.breaker_tripped.return_value = breaker
    lifecycle.has_inflight.return_value = inflight
    lifecycle.reconcile_ok.return_value = True
    ev = SwapEvaluator(plugin, rpc, db, cfg, client, planner, lifecycle)
    return ev, cfg, client, db


class TestSwapEvaluatorGates:
    def test_disabled_short_circuits(self):
        ev, cfg, client, _ = _make_evaluator({"lnplus_swaps_enabled": False})
        result = ev.run_cycle(cfg, 500.0)
        assert result["applied"] is False and result["recommended"] is False
        client.get_applicable_swaps.assert_not_called()

    def test_breaker_blocks(self):
        ev, cfg, client, _ = _make_evaluator(breaker="missed deadline sw9")
        result = ev.run_cycle(cfg, 500.0)
        assert result["applied"] is False
        client.get_applicable_swaps.assert_not_called()

    def test_serialization_blocks(self):
        ev, cfg, client, _ = _make_evaluator(inflight=True)
        result = ev.run_cycle(cfg, 500.0)
        assert result["applied"] is False
        client.get_applicable_swaps.assert_not_called()

    def test_feerate_ceiling_blocks(self):
        ev, cfg, client, _ = _make_evaluator()
        ev.rpc.feerates.return_value = {"perkw": {"opening": 99999}}
        result = ev.run_cycle(cfg, 500.0)
        assert result["applied"] is False
        client.get_applicable_swaps.assert_not_called()

    def test_rejects_not_last_slot(self):
        swap = _swap_fixture(participant_waiting_for_count=2)
        ev, cfg, _, _ = _make_evaluator(swaps=[swap])
        result = ev.run_cycle(cfg, 0.0)
        assert result["applied"] is False
        assert any(r["gate"] == "fill_state" for r in result["rejections"])

    def test_rejects_long_duration(self):
        swap = _swap_fixture(duration_months=6)
        ev, cfg, _, _ = _make_evaluator(swaps=[swap])
        result = ev.run_cycle(cfg, 0.0)
        assert any(r["gate"] == "terms" for r in result["rejections"])

    def test_rejects_lnd_platform(self):
        swap = _swap_fixture(platform="lnd")
        ev, cfg, _, _ = _make_evaluator(swaps=[swap])
        result = ev.run_cycle(cfg, 0.0)
        assert any(r["gate"] == "terms" for r in result["rejections"])

    def test_rejects_low_rated_participant(self):
        swap = _swap_fixture()
        swap["participants"][1]["positive_ratings_count"] = 1
        ev, cfg, _, _ = _make_evaluator(swaps=[swap])
        result = ev.run_cycle(cfg, 0.0)
        assert any(r["gate"] == "peer_quality" for r in result["rejections"])

    def test_rejects_fleet_member_participant(self):
        swap = _swap_fixture()
        swap["participants"][1]["pubkey"] = FLEET_PK
        ev, cfg, _, _ = _make_evaluator(
            {"lnplus_fleet_pubkeys": FLEET_PK}, swaps=[swap])
        result = ev.run_cycle(cfg, 0.0)
        assert any(r["gate"] == "fleet_dedup" for r in result["rejections"])

    def test_rejects_bad_negative_ratio(self):
        swap = _swap_fixture()
        swap["participants"][0].update(positive_ratings_count=10,
                                       negative_ratings_count=5)
        ev, cfg, _, _ = _make_evaluator(swaps=[swap])
        result = ev.run_cycle(cfg, 0.0)
        assert any(r["gate"] == "peer_quality" for r in result["rejections"])

    def test_rejects_peer_vetoed_by_planner_scoring(self):
        ev, cfg, _, _ = _make_evaluator()
        ev._planner._score_candidate.return_value = 0.2   # below SCORE_FLOOR
        result = ev.run_cycle(cfg, 0.0)
        assert any(r["gate"] == "peer_quality" and "score" in r["reason"]
                   for r in result["rejections"])

    def test_infer_assignment_triangle(self):
        ev, cfg, _, _ = _make_evaluator()
        a = ev._infer_assignment(_swap_fixture())
        assert a["our_identifier"] == "C"
        assert a["outbound_peer"] == PK_A     # C opens to A (wraps)
        assert a["incoming_peer"] == PK_B     # B opens to C


class TestSwapEvAndApply:
    def test_swap_ev_computation(self):
        ev, cfg, _, _ = _make_evaluator()
        # outbound_ev = inbound_corridor = 1000.0 (mocked), open cost 2000
        # replacement = 5M * 0.005 = 25000 -> inbound_credit = 1000
        # min pos ratings = 12 -> reliability = 0.6 + 0.4*(12/50) = 0.696 (clearnet)
        # haircut = 0.3 * 800 * (3/12) = 60 (best_regular_ev=800)
        value, assignment = ev._swap_ev(_swap_fixture(), cfg, best_regular_ev=800.0)
        expected = 1000.0 + 1000.0 * 0.696 * 0.5 - 60.0 - 2000
        assert abs(value - expected) < 1.0
        assert assignment["outbound_peer"] == PK_A

    def test_tor_only_counterparty_discounted(self):
        swap = _swap_fixture()
        for p in swap["participants"]:
            p["address_1"] = "abcdef.onion:9735"
            p["address_2"] = None
        ev, cfg, _, _ = _make_evaluator(swaps=[swap])
        tor_value, _ = ev._swap_ev(swap, cfg, best_regular_ev=0.0)
        clear_value, _ = ev._swap_ev(_swap_fixture(), cfg, best_regular_ev=0.0)
        assert tor_value < clear_value

    def test_swap_wins_within_margin_and_applies(self):
        ev, cfg, client, db = _make_evaluator()
        ev._planner._calculate_open_ev.return_value = 5000.0
        client.create_application.return_value = {"id": "sw1", "status": "pending"}
        db.record_planner_action = MagicMock(return_value=42)
        db.update_planner_action = MagicMock()
        # swap_ev clearly positive; best_regular_ev inside margin
        result = ev.run_cycle(cfg, best_regular_ev=3000.0)
        assert result["applied"] is True
        assert result["swap_id"] == "sw1"
        client.create_application.assert_called_once_with("sw1")
        # intent-first: DB row exists with status applied
        row = ev._db.lnplus_get_swap("sw1")
        assert row["status"] == "applied"
        db.record_planner_action.assert_called_once()
        assert db.record_planner_action.call_args.kwargs["action_type"] == "swap_apply"

    def test_regular_open_beats_margin(self):
        ev, cfg, client, _ = _make_evaluator()
        # swap EV positive but small: 2500 + 2500*0.696*0.5 - 0.3*10000*(3/12) - 2000 = 620
        # margin: 10000 > 620 * 1.2 -> regular open wins
        ev._planner._calculate_open_ev.return_value = 2500.0
        result = ev.run_cycle(cfg, best_regular_ev=10_000.0)
        assert result["applied"] is False
        client.create_application.assert_not_called()
        assert any(r["gate"] == "preference_margin" for r in result["rejections"])

    def test_insufficient_onchain_funds_blocks(self):
        ev, cfg, client, _ = _make_evaluator()
        ev._planner._calculate_open_ev.return_value = 5000.0
        # confirmed on-chain funds below capacity + open cost
        ev.rpc.listfunds.return_value = {"outputs": [
            {"amount_msat": 1_000_000_000, "status": "confirmed", "reserved": False}]}
        result = ev.run_cycle(cfg, best_regular_ev=0.0)
        assert result["applied"] is False
        assert any(r["gate"] == "economics" and "funds" in r["reason"]
                   for r in result["rejections"])

    def test_negative_ev_rejected(self):
        ev, cfg, client, _ = _make_evaluator()
        ev._planner._calculate_open_ev.return_value = 0.0   # ev = -open_cost < 0
        result = ev.run_cycle(cfg, best_regular_ev=0.0)
        assert result["applied"] is False
        assert any(r["gate"] == "economics" for r in result["rejections"])

    def test_recommend_only_mode(self):
        ev, cfg, client, db = _make_evaluator({"lnplus_execute_applications": False})
        ev._planner._calculate_open_ev.return_value = 5000.0
        db.record_planner_action = MagicMock(return_value=43)
        db.update_planner_action = MagicMock()
        result = ev.run_cycle(cfg, best_regular_ev=0.0)
        assert result["recommended"] is True and result["applied"] is False
        client.create_application.assert_not_called()
        db.update_planner_action.assert_called_with(43, status="recommended")

    def test_capex_budget_blocks(self):
        ev, cfg, client, _ = _make_evaluator()
        ev._planner._calculate_open_ev.return_value = 5000.0
        ev._planner._capex_engine.get_fleet_exploration_budget.return_value = 0
        result = ev.run_cycle(cfg, best_regular_ev=0.0)
        assert result["applied"] is False
        assert any(r["gate"] == "economics" for r in result["rejections"])

    def test_application_failure_marks_failed(self):
        ev, cfg, client, db = _make_evaluator()
        ev._planner._calculate_open_ev.return_value = 5000.0
        from modules.lnplus_swaps import LNPlusError
        client.create_application.side_effect = LNPlusError("slot taken", http_status=422)
        db.record_planner_action = MagicMock(return_value=44)
        db.update_planner_action = MagicMock()
        result = ev.run_cycle(cfg, best_regular_ev=0.0)
        assert result["applied"] is False
        row = ev._db.lnplus_get_swap("sw1")
        assert row["status"] == "failed"
        db.update_planner_action.assert_called_with(44, status="failed")

    def test_capex_engine_error_fails_closed(self):
        ev, cfg, client, _ = _make_evaluator()
        ev._planner._calculate_open_ev.return_value = 5000.0
        ev._planner._capex_engine.get_fleet_exploration_budget.side_effect = RuntimeError("boom")
        result = ev.run_cycle(cfg, best_regular_ev=0.0)
        assert result["applied"] is False
        client.create_application.assert_not_called()
        assert any(r["gate"] == "economics" and "capex" in r["reason"]
                   for r in result["rejections"])


from modules.lnplus_swaps import SwapLifecycle


def _make_lifecycle(my_swaps=None, local_rows=None):
    plugin, rpc = MagicMock(), MagicMock()
    db = _make_db()
    client = MagicMock()
    client.get_my_swaps.return_value = my_swaps or {"pending": [], "opening": [], "completed": []}
    policy = MagicMock()
    ignore_fn = MagicMock()
    from modules.config import Config
    cfg = Config()
    lc = SwapLifecycle(plugin, rpc, db, cfg, client, policy, ignore_peer_fn=ignore_fn)
    for row in (local_rows or []):
        db.lnplus_record_swap(**row)
    return lc, db, rpc, client, policy, ignore_fn


class TestSwapLifecycle:
    def test_breaker_roundtrip(self):
        lc, db, *_ = _make_lifecycle()
        assert lc.breaker_tripped() is None
        lc.trip_breaker("missed deadline sw1")
        assert "sw1" in lc.breaker_tripped()
        lc.clear_breaker()
        assert lc.breaker_tripped() is None

    def test_has_inflight(self):
        lc, db, *_ = _make_lifecycle(local_rows=[
            dict(swap_id="s1", status="applied", capacity_sats=1_000_000, duration_months=3)])
        assert lc.has_inflight() is True

    def test_reconcile_divergence_trips_breaker(self):
        # Local row 'applied' but LN+ has nothing -> divergence
        lc, *_ = _make_lifecycle(local_rows=[
            dict(swap_id="s1", status="applied", capacity_sats=1_000_000, duration_months=3)])
        assert lc.reconcile_ok() is False
        assert lc.breaker_tripped() is not None

    def test_opening_executes_fundchannel_and_completes(self):
        peer = PK_A
        deadline = int(time.time()) + 40 * 3600
        my = {"pending": [], "completed": [], "opening": [
            {"id": "s1", "outgoing_peer_pubkey": peer,
             "outgoing_peer_clearnet_address": "1.2.3.4:9735",
             "deadline": deadline, "capacity_sats": 2_000_000}]}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my, local_rows=[
            dict(swap_id="s1", status="applied", capacity_sats=2_000_000,
                 duration_months=3, outbound_peer=peer)])
        rpc.listpeerchannels.return_value = {"channels": []}
        rpc.fundchannel.return_value = {"txid": "ff" * 32}
        lc.run_watcher_once()
        rpc.fundchannel.assert_called_once()
        assert rpc.fundchannel.call_args.kwargs.get("feerate") == "slow"  # >24h left
        client.complete_application.assert_called_once_with("s1")
        row = db.lnplus_get_swap("s1")
        assert row["status"] == "opened"
        assert row["channel_funding_txid"] == "ff" * 32

    def test_open_is_idempotent_when_channel_exists(self):
        peer = PK_A
        my = {"pending": [], "completed": [], "opening": [
            {"id": "s1", "outgoing_peer_pubkey": peer,
             "deadline": int(time.time()) + 10 * 3600, "capacity_sats": 2_000_000}]}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my, local_rows=[
            dict(swap_id="s1", status="applied", capacity_sats=2_000_000,
                 duration_months=3, outbound_peer=peer)])
        rpc.listpeerchannels.return_value = {"channels": [
            {"peer_id": peer, "state": "CHANNELD_AWAITING_LOCKIN",
             "funding_txid": "aa" * 32}]}
        lc.run_watcher_once()
        rpc.fundchannel.assert_not_called()
        client.complete_application.assert_called_once()

    def test_feerate_escalates_near_deadline(self):
        peer = PK_A
        my = {"pending": [], "completed": [], "opening": [
            {"id": "s1", "outgoing_peer_pubkey": peer,
             "outgoing_peer_clearnet_address": "1.2.3.4:9735",
             "deadline": int(time.time()) + 2 * 3600, "capacity_sats": 2_000_000}]}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my, local_rows=[
            dict(swap_id="s1", status="applied", capacity_sats=2_000_000,
                 duration_months=3, outbound_peer=peer)])
        rpc.listpeerchannels.return_value = {"channels": []}
        rpc.fundchannel.return_value = {"txid": "ff" * 32}
        lc.run_watcher_once()
        assert rpc.fundchannel.call_args.kwargs.get("feerate") == "urgent"

    def test_missed_deadline_trips_breaker(self):
        peer = PK_A
        my = {"pending": [], "completed": [], "opening": [
            {"id": "s1", "outgoing_peer_pubkey": peer,
             "deadline": int(time.time()) - 3600, "capacity_sats": 2_000_000}]}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my, local_rows=[
            dict(swap_id="s1", status="applied", capacity_sats=2_000_000,
                 duration_months=3, outbound_peer=peer)])
        rpc.listpeerchannels.return_value = {"channels": []}
        rpc.connect.side_effect = Exception("unreachable")
        lc.run_watcher_once()
        assert lc.breaker_tripped() is not None

    def test_completion_activates_protection(self):
        peer = PK_A
        ends = "2026-10-05T00:00:00Z"
        my = {"pending": [], "opening": [], "completed": [
            {"id": "s1", "incoming_peer_pubkey": PK_B, "ends": ends}]}
        lc, db, rpc, client, policy, _ = _make_lifecycle(my_swaps=my, local_rows=[
            dict(swap_id="s1", status="applied", capacity_sats=2_000_000,
                 duration_months=3, outbound_peer=peer)])
        db.lnplus_update_swap("s1", status="opened")
        lc.run_watcher_once()
        row = db.lnplus_get_swap("s1")
        assert row["status"] == "active"
        assert row["ends_at"] is not None
        policy.add_tag.assert_called_once_with(peer, "no_close")

    def test_contract_end_rates_and_releases(self):
        peer, incoming = PK_A, PK_B
        lc, db, rpc, client, policy, ignore_fn = _make_lifecycle()
        db.lnplus_record_swap("s1", "active", 2_000_000, 3,
                              outbound_peer=peer, incoming_peer=incoming)
        db.lnplus_update_swap("s1", ends_at=int(time.time()) - 60)
        rpc.listpeerchannels.return_value = {"channels": [
            {"peer_id": incoming, "state": "CHANNELD_NORMAL"}]}
        lc.run_watcher_once()
        row = db.lnplus_get_swap("s1")
        assert row["status"] == "ended"
        policy.remove_tag.assert_called_once_with(peer, "no_close")
        client.create_rating.assert_called_once_with("s1", "positive")
        ignore_fn.assert_not_called()

    def test_defection_rated_negative_and_ignored(self):
        peer, incoming = PK_A, PK_B
        lc, db, rpc, client, policy, ignore_fn = _make_lifecycle()
        db.lnplus_record_swap("s1", "active", 2_000_000, 3,
                              outbound_peer=peer, incoming_peer=incoming)
        db.lnplus_update_swap("s1", ends_at=int(time.time()) - 60)
        rpc.listpeerchannels.return_value = {"channels": []}   # incoming never opened / closed early
        lc.run_watcher_once()
        client.create_rating.assert_called_once_with("s1", "negative")
        ignore_fn.assert_called_once()
        assert db.lnplus_get_peer(incoming)["defections"] == 1

    def test_pending_timeout_withdraws(self):
        lc, db, rpc, client, *_ = _make_lifecycle(
            my_swaps={"pending": [{"id": "s1"}], "opening": [], "completed": []})
        db.lnplus_record_swap("s1", "applied", 2_000_000, 3)
        # age the row past the timeout
        conn = db._get_connection()
        conn.execute("UPDATE lnplus_swaps SET applied_at = ? WHERE swap_id = 's1'",
                     (int(time.time()) - 9 * 86400,))
        lc.run_watcher_once()
        client.delete_application.assert_called_once_with("s1")
        assert db.lnplus_get_swap("s1")["status"] == "withdrawn"

    def test_lnplus_outage_does_not_trip_breaker(self):
        from modules.lnplus_swaps import LNPlusError
        lc, db, rpc, client, *_ = _make_lifecycle()
        client.get_my_swaps.side_effect = LNPlusError("down")
        result = lc.run_watcher_once()
        assert lc.breaker_tripped() is None
        assert result.get("skipped") == "lnplus unreachable"

    # -- atomic spend reservation around swap-open (fix round 1) -------------
    def test_open_reserves_before_fundchannel(self):
        peer = PK_A
        deadline = int(time.time()) + 40 * 3600
        my = {"pending": [], "completed": [], "opening": [
            {"id": "s1", "outgoing_peer_pubkey": peer,
             "outgoing_peer_clearnet_address": "1.2.3.4:9735",
             "deadline": deadline, "capacity_sats": 2_000_000}]}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my, local_rows=[
            dict(swap_id="s1", status="applied", capacity_sats=2_000_000,
                 duration_months=3, outbound_peer=peer)])
        rpc.listpeerchannels.return_value = {"channels": []}
        rpc.fundchannel.return_value = {"txid": "ff" * 32}

        call_order = []
        real_reserve_spend = db.reserve_spend

        def _spy_reserve_spend(*args, **kwargs):
            call_order.append("reserve_spend")
            return real_reserve_spend(*args, **kwargs)

        def _spy_fundchannel(*args, **kwargs):
            call_order.append("fundchannel")
            return {"txid": "ff" * 32}

        rpc.fundchannel.side_effect = _spy_fundchannel
        with patch.object(db, "reserve_spend", side_effect=_spy_reserve_spend) as spy:
            lc.run_watcher_once()
            spy.assert_called_once()
            kwargs = spy.call_args.kwargs
            assert kwargs["category"] == "channel_open"
            assert kwargs["subcategory"] == "lnplus_swap"
            assert kwargs["metadata"]["swap_id"] == "s1"

        assert call_order == ["reserve_spend", "fundchannel"]

        # The reservation should have been settled ('spent'), not left active.
        conn = db._get_connection()
        rows = conn.execute(
            "SELECT status, category, reserved_sats FROM spend_reservations "
            "WHERE reservation_id LIKE 'lnplus-open-s1-%'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["status"] == "spent"
        assert rows[0]["category"] == "channel_open"

    def test_reservation_failure_blocks_fundchannel(self):
        peer = PK_A
        deadline = int(time.time()) + 40 * 3600
        my = {"pending": [], "completed": [], "opening": [
            {"id": "s1", "outgoing_peer_pubkey": peer,
             "outgoing_peer_clearnet_address": "1.2.3.4:9735",
             "deadline": deadline, "capacity_sats": 2_000_000}]}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my, local_rows=[
            dict(swap_id="s1", status="applied", capacity_sats=2_000_000,
                 duration_months=3, outbound_peer=peer)])
        rpc.listpeerchannels.return_value = {"channels": []}
        with patch.object(db, "reserve_spend", return_value=False):
            lc.run_watcher_once()
        rpc.fundchannel.assert_not_called()
        row = db.lnplus_get_swap("s1")
        assert row["status"] == "opening"
        assert lc.breaker_tripped() is None

    def test_fundchannel_failure_releases_reservation(self):
        peer = PK_A
        deadline = int(time.time()) + 40 * 3600
        my = {"pending": [], "completed": [], "opening": [
            {"id": "s1", "outgoing_peer_pubkey": peer,
             "outgoing_peer_clearnet_address": "1.2.3.4:9735",
             "deadline": deadline, "capacity_sats": 2_000_000}]}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my, local_rows=[
            dict(swap_id="s1", status="applied", capacity_sats=2_000_000,
                 duration_months=3, outbound_peer=peer)])
        rpc.listpeerchannels.return_value = {"channels": []}
        rpc.fundchannel.side_effect = Exception("fundchannel boom")
        with patch.object(db, "release_spend_reservation",
                           wraps=db.release_spend_reservation) as spy:
            result = lc.run_watcher_once()
        spy.assert_called_once()
        assert "s1" not in result.get("errors", [])  # handled internally, no crash
        conn = db._get_connection()
        rows = conn.execute(
            "SELECT status FROM spend_reservations WHERE reservation_id LIKE 'lnplus-open-s1-%'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["status"] == "released"

    def test_deadline_null_gets_local_fallback(self):
        peer = PK_A
        my = {"pending": [], "completed": [], "opening": [
            {"id": "s1", "outgoing_peer_pubkey": peer,
             "outgoing_peer_clearnet_address": "1.2.3.4:9735",
             "capacity_sats": 2_000_000}]}  # no 'deadline' key at all
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my, local_rows=[
            dict(swap_id="s1", status="applied", capacity_sats=2_000_000,
                 duration_months=3, outbound_peer=peer)])
        rpc.listpeerchannels.return_value = {"channels": []}
        rpc.fundchannel.return_value = {"txid": "ff" * 32}
        before = int(time.time())
        lc.run_watcher_once()
        after = int(time.time())
        row = db.lnplus_get_swap("s1")
        assert row["deadline_at"] is not None
        expected_low = before + 48 * 3600
        expected_high = after + 48 * 3600
        assert expected_low <= row["deadline_at"] <= expected_high

    def test_opened_rows_skipped_in_opening_phase(self):
        peer = PK_A
        my = {"pending": [], "completed": [], "opening": [
            {"id": "s1", "outgoing_peer_pubkey": peer,
             "deadline": int(time.time()) + 40 * 3600, "capacity_sats": 2_000_000}]}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my, local_rows=[
            dict(swap_id="s1", status="opened", capacity_sats=2_000_000,
                 duration_months=3, outbound_peer=peer)])
        db.lnplus_update_swap("s1", channel_funding_txid="ff" * 32, opened_at=int(time.time()))
        lc.run_watcher_once()
        rpc.fundchannel.assert_not_called()
        client.complete_application.assert_not_called()
        row = db.lnplus_get_swap("s1")
        assert row["status"] == "opened"


class TestLnplusIntegration:
    def test_full_swap_lifecycle(self):
        """apply (evaluator) -> fill -> open (watcher) -> activate -> end -> rate."""
        from modules.config import Config
        cfg = Config()
        plugin, rpc = MagicMock(), MagicMock()
        rpc.feerates.return_value = {"perkw": {"opening": 2000}}
        rpc.getinfo.return_value = {"id": FLEET_PK}
        # Gate 7 (confirmed on-chain funds) is exercised elsewhere by
        # TestSwapEvAndApply; here we only need it to not block the apply,
        # so give the mock rpc a generous confirmed balance (mirrors the
        # _make_evaluator fixture's listfunds shape).
        rpc.listfunds.return_value = {"outputs": [
            {"amount_msat": 100_000_000_000, "status": "confirmed", "reserved": False}]}
        db = _make_db()
        client = MagicMock()
        planner = MagicMock()
        planner._calculate_open_ev.return_value = 8000.0
        planner._estimate_open_cost.return_value = 2000
        planner._capex_engine.get_fleet_exploration_budget.return_value = 50_000
        policy = MagicMock()
        ignore_fn = MagicMock()

        from modules.lnplus_swaps import SwapEvaluator, SwapLifecycle
        lifecycle = SwapLifecycle(plugin, rpc, db, cfg, client, policy,
                                  ignore_peer_fn=ignore_fn,
                                  estimate_open_cost_fn=lambda: 2000)
        evaluator = SwapEvaluator(plugin, rpc, db, cfg, client, planner, lifecycle)

        # Phase 1: apply
        client.get_my_swaps.return_value = {"pending": [], "opening": [], "completed": []}
        client.get_applicable_swaps.return_value = [_swap_fixture()]
        client.create_application.return_value = {"id": "sw1"}
        summary = evaluator.run_cycle(cfg, best_regular_ev=1000.0)
        assert summary["applied"] is True
        assert lifecycle.has_inflight() is True
        # Serialization: a second cycle must not even fetch swaps
        client.get_applicable_swaps.reset_mock()
        summary2 = evaluator.run_cycle(cfg, best_regular_ev=1000.0)
        assert summary2["applied"] is False
        client.get_applicable_swaps.assert_not_called()

        # Phase 2: swap fills -> watcher opens our channel
        client.get_my_swaps.return_value = {"pending": [], "completed": [], "opening": [
            {"id": "sw1", "outgoing_peer_pubkey": PK_A,
             "outgoing_peer_clearnet_address": "1.2.3.4:9735",
             "deadline": int(time.time()) + 40 * 3600, "capacity_sats": 5_000_000}]}
        rpc.listpeerchannels.return_value = {"channels": []}
        rpc.fundchannel.return_value = {"txid": "ab" * 32}
        lifecycle.run_watcher_once()
        assert db.lnplus_get_swap("sw1")["status"] == "opened"

        # Phase 3: all sides open -> active + protected
        client.get_my_swaps.return_value = {"pending": [], "opening": [], "completed": [
            {"id": "sw1", "incoming_peer_pubkey": PK_B,
             "ends": "2026-10-05T00:00:00Z"}]}
        lifecycle.run_watcher_once()
        assert db.lnplus_get_swap("sw1")["status"] == "active"
        policy.add_tag.assert_called_with(PK_A, "no_close")

        # Phase 4: contract ends -> release + positive rating
        db.lnplus_update_swap("sw1", ends_at=int(time.time()) - 1)
        rpc.listpeerchannels.return_value = {"channels": [
            {"peer_id": PK_B, "state": "CHANNELD_NORMAL"}]}
        lifecycle.run_watcher_once()
        assert db.lnplus_get_swap("sw1")["status"] == "ended"
        policy.remove_tag.assert_called_with(PK_A, "no_close")
        client.create_rating.assert_called_with("sw1", "positive")
        ignore_fn.assert_not_called()
        assert lifecycle.has_inflight() is False
