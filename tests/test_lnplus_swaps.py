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

    def test_reserved_sats_excludes_already_funded_rows(self):
        """B8: once channel_funding_txid is set the capacity already left
        listfunds — it must not still be subtracted from available_sats."""
        db = _make_db()
        db.lnplus_record_swap("s1", "opening", 2_000_000, 3)
        db.lnplus_update_swap("s1", channel_funding_txid="aa" * 32)
        db.lnplus_record_swap("s2", "opening", 3_000_000, 3)
        assert db.lnplus_reserved_sats() == 3_000_000

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

    def test_parse_ts_naive_string_treated_as_utc(self):
        """B3: a TZ-less ISO timestamp must not be interpreted in the node's
        local timezone by .timestamp() — LN+ deadlines/ends are UTC. Assert
        equality with the explicit-Z form rather than depending on the
        test runner's local TZ (monkeypatching TZ is unreliable)."""
        assert _parse_ts("2026-08-01T00:00:00") == _parse_ts("2026-08-01T00:00:00Z")

    def test_auth_flow_signs_challenge(self):
        # LN+ docs show get_my_swaps wrapping its response in a one-element
        # array — this is the documented reality, so the primary happy-path
        # test now serves the ARRAY-wrapped payload.
        challenge = {"message": "lnplus-auth-abc123", "expires_at": "2099-01-01T00:00:00Z"}
        client, rpc, patcher = _make_client(
            [challenge, [{"pending": [], "opening": [], "completed": []}]])
        with patcher:
            result = client.get_my_swaps()
        rpc.signmessage.assert_called_once_with("lnplus-auth-abc123")
        assert result["pending"] == []

    def test_get_my_swaps_bare_dict_still_accepted(self):
        """Docs are ambiguous about whether the envelope is always present
        — a bare dict (no array wrapper) must still be accepted."""
        challenge = {"message": "lnplus-auth-abc123", "expires_at": "2099-01-01T00:00:00Z"}
        client, rpc, patcher = _make_client(
            [challenge, {"pending": [{"id": "p1"}], "opening": [], "completed": []}])
        with patcher:
            result = client.get_my_swaps()
        assert result["pending"] == [{"id": "p1"}]

    def test_get_my_swaps_empty_list_returns_empty_buckets(self):
        challenge = {"message": "lnplus-auth-abc123", "expires_at": "2099-01-01T00:00:00Z"}
        client, rpc, patcher = _make_client([challenge, []])
        with patcher:
            result = client.get_my_swaps()
        assert result == {"pending": [], "opening": [], "completed": []}

    def test_get_my_swaps_garbage_payload_raises(self):
        challenge = {"message": "lnplus-auth-abc123", "expires_at": "2099-01-01T00:00:00Z"}
        client, rpc, patcher = _make_client([challenge, "not a payload"])
        with patcher:
            try:
                client.get_my_swaps()
                assert False, "should have raised"
            except LNPlusError:
                pass

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

    def test_get_my_swaps_normalizes_integer_ids_to_strings(self):
        """ID normalization at client boundary: LN+ may return integer ids,
        but local DB stores them as strings. Normalize on the way in."""
        challenge = {"message": "lnplus-auth-abc123", "expires_at": "2099-01-01T00:00:00Z"}
        client, rpc, patcher = _make_client(
            [challenge, [{"pending": [{"id": 123}], "opening": [{"id": 456}], "completed": []}]])
        with patcher:
            result = client.get_my_swaps()
        assert result["pending"][0]["id"] == "123"
        assert result["opening"][0]["id"] == "456"
        assert isinstance(result["pending"][0]["id"], str)
        assert isinstance(result["opening"][0]["id"], str)


from modules.lnplus_swaps import SwapEvaluator, NEG_RATIO_MAX, TOR_RELIABILITY, _IDENTIFIERS

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
    # I5(a): default to no pre-existing channel to the inferred outbound
    # peer, so the new existing-channel gate does not spuriously reject
    # every other evaluator test (a bare MagicMock's .get("channels", [])
    # is itself a truthy MagicMock, not an empty list).
    rpc.listpeerchannels.return_value = {"channels": []}
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

    def test_rejects_oversized_capacity(self):
        """I2(a): a swap above planner_max_channel_sats must be rejected at
        the terms gate, not slip through to the funds/EV gates."""
        cfg_tmp = Config()
        swap = _swap_fixture(capacity_sats=cfg_tmp.planner_max_channel_sats + 1)
        ev, cfg, _, _ = _make_evaluator(swaps=[swap])
        result = ev.run_cycle(cfg, 0.0)
        assert result["applied"] is False
        assert any(r["gate"] == "terms" and "above planner max" in r["reason"]
                   for r in result["rejections"])

    def test_rejects_existing_channel_to_assigned_outbound_peer(self):
        """I5(a): a swap must be rejected if we already have a channel to
        the inferred outbound peer (PK_A, per test_infer_assignment_triangle)
        — joining would waste the serialization slot on a swap that cannot
        add new capacity, and could later be misread as our swap channel."""
        ev, cfg, _, _ = _make_evaluator()
        ev.rpc.listpeerchannels.return_value = {"channels": [
            {"peer_id": PK_A, "state": "CHANNELD_NORMAL"}]}
        result = ev.run_cycle(cfg, 0.0)
        assert result["applied"] is False
        assert any(r["gate"] == "terms" and "existing channel" in r["reason"]
                   for r in result["rejections"])

    def test_rejects_full_identifier_set_no_free_slot(self):
        """B11: a full identifier set (participant count == cfg's own
        max_participants ceiling, so every _IDENTIFIERS slot is occupied)
        must be rejected cleanly via a gate ('terms:no free participant
        slot'), not raise a bare StopIteration through the gate chain."""
        max_participants = 4
        letters = _IDENTIFIERS[:max_participants]
        participants = [
            {"participant_identifier": letter,
             "pubkey": "02" + f"{i:02x}" * 32,
             "positive_ratings_count": 20, "negative_ratings_count": 0,
             "address_1": "1.2.3.4:9735"}
            for i, letter in enumerate(letters, start=1)
        ]
        swap = _swap_fixture(participant_max_count=max_participants,
                             participant_waiting_for_count=1,
                             participants=participants)
        ev, cfg, _, _ = _make_evaluator(swaps=[swap])
        result = ev.run_cycle(cfg, 0.0)
        assert result["applied"] is False
        assert any(r["gate"] == "terms" and "no free participant slot" in r["reason"]
                   for r in result["rejections"])

    def test_cancelled_participant_does_not_veto_peer_gate(self):
        """B12: a cancelled participant's stats are stale and their slot is
        effectively free — they must not veto gate 5 even with a
        disqualifying rating."""
        swap = _swap_fixture()
        swap["participants"][1]["positive_ratings_count"] = 1
        swap["participants"][1]["cancelled"] = True
        ev, cfg, _, _ = _make_evaluator(swaps=[swap])
        result = ev.run_cycle(cfg, 0.0)
        assert not any(r["gate"] == "peer_quality" for r in result["rejections"])

    def test_infer_assignment_treats_cancelled_identifier_as_free(self):
        """B12: _infer_assignment must exclude a cancelled participant from
        the identifier map — their letter is free for us to take rather
        than treated as occupied."""
        swap = _swap_fixture()
        swap["participants"][1]["cancelled"] = True   # B is cancelled
        ev, cfg, _, _ = _make_evaluator()
        a = ev._infer_assignment(swap)
        assert a["our_identifier"] == "B"   # B's slot reads as free
        assert a["incoming_peer"] == PK_A   # A -> B (us)

    def test_infer_assignment_triangle(self):
        ev, cfg, _, _ = _make_evaluator()
        a = ev._infer_assignment(_swap_fixture())
        assert a["our_identifier"] == "C"
        assert a["outbound_peer"] == PK_A     # C opens to A (wraps)
        assert a["incoming_peer"] == PK_B     # B opens to C


class TestSwapEvAndApply:
    def test_swap_ev_computation(self):
        ev, cfg, _, _ = _make_evaluator()
        # outbound_ev = inbound_corridor = 1000.0 (mocked)
        # I4: no separate "- open_cost" term here — _calculate_open_ev (the
        # mocked outbound_ev above) already nets open+close on-chain cost
        # internally, same as a regular candidate's _planned_ev. open_cost
        # is only re-used by the capex/funds gates in _select_and_apply.
        # replacement = 5M * 0.005 = 25000 -> inbound_credit = 1000
        # min pos ratings = 12 -> reliability = 0.6 + 0.4*(12/50) = 0.696 (clearnet)
        # haircut = 0.3 * 800 * (3/12) = 60 (best_regular_ev=800)
        value, assignment = ev._swap_ev(_swap_fixture(), cfg, best_regular_ev=800.0)
        expected = 1000.0 + 1000.0 * 0.696 * 0.5 - 60.0
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
        # swap EV positive but small: 2500 + 2500*0.696*0.5 - 0.3*10000*(3/12) = 2620
        # (I4: no "- open_cost" term — see test_swap_ev_computation)
        # margin: 10000 > 2620 * 1.2 (=3144) -> regular open wins
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

    def test_funds_cover_capacity_but_not_reserve_blocks(self):
        """I2(b): confirmed funds cover capacity+open_cost exactly but leave
        nothing for the wallet reserve floor (min_wallet_reserve) — must be
        rejected, mirroring the planner's own reserve-respecting sizer."""
        ev, cfg, client, _ = _make_evaluator()
        ev._planner._calculate_open_ev.return_value = 5000.0
        # swap capacity 5_000_000 (fixture default) + open_cost 2000 covered
        # exactly, but min_wallet_reserve (default 1_000_000) is not.
        ev.rpc.listfunds.return_value = {"outputs": [
            {"amount_msat": 5_002_000_000, "status": "confirmed", "reserved": False}]}
        result = ev.run_cycle(cfg, best_regular_ev=0.0)
        assert result["applied"] is False
        assert any(r["gate"] == "economics" and "reserve" in r["reason"]
                   for r in result["rejections"])
        client.create_application.assert_not_called()

    def test_negative_ev_rejected(self):
        ev, cfg, client, _ = _make_evaluator()
        # I4: with open_cost no longer subtracted here, an all-zero EV input
        # computes to exactly 0 rather than a negative number — still
        # non-positive, so _select_and_apply's `value <= 0` check rejects it.
        ev._planner._calculate_open_ev.return_value = 0.0   # ev = 0 <= 0
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

    def test_trip_breaker_preserves_first_cause(self):
        """B10: a second trip must not overwrite the stored reason — the
        first cause is the diagnostic signal the operator needs."""
        lc, db, *_ = _make_lifecycle()
        lc.trip_breaker("first cause: missed deadline sw1")
        lc.trip_breaker("second cause: unrelated noise sw2")
        reason = lc.breaker_tripped()
        assert "first cause" in reason
        assert "second cause" not in reason

    def test_has_inflight(self):
        lc, db, *_ = _make_lifecycle(local_rows=[
            dict(swap_id="s1", status="applied", capacity_sats=1_000_000, duration_months=3)])
        assert lc.has_inflight() is True

    def test_reconcile_divergence_trips_breaker(self):
        # Local row 'opening' (funds may already be committed on-chain) but
        # LN+ has nothing -> divergence. B4 carves 'applied' rows out into a
        # cancelled_remote transition instead (see
        # test_reconcile_applied_row_vanished_becomes_cancelled_remote) —
        # 'opening'/'opened' rows still trip the breaker exactly as before,
        # since funds may already be committed and this needs an operator.
        lc, *_ = _make_lifecycle(local_rows=[
            dict(swap_id="s1", status="opening", capacity_sats=1_000_000, duration_months=3)])
        assert lc.reconcile_ok() is False
        assert lc.breaker_tripped() is not None

    def test_reconcile_applied_row_vanished_becomes_cancelled_remote(self):
        """B4: an 'applied' row absent from pending/opening/completed on a
        successful fetch is a REMOTE cancellation, not our defection — it
        must transition to cancelled_remote (terminal, frees the
        serialization slot + reservation) rather than trip the breaker
        every pass forever."""
        lc, db, *_ = _make_lifecycle(local_rows=[
            dict(swap_id="s1", status="applied", capacity_sats=1_000_000, duration_months=3)])
        conn = db._get_connection()
        conn.execute("UPDATE lnplus_swaps SET applied_at = ? WHERE swap_id = 's1'",
                     (int(time.time()) - 3600,))
        assert lc.reconcile_ok() is True
        assert lc.breaker_tripped() is None
        row = db.lnplus_get_swap("s1")
        assert row["status"] == "cancelled_remote"
        assert lc.has_inflight() is False

    def test_reconcile_fresh_application_grace_window(self):
        """B9: a local 'applied' row with applied_at=now must not be
        evaluated for divergence at all — the evaluator may have applied
        milliseconds after the watcher's own get_my_swaps fetch ran."""
        lc, db, *_ = _make_lifecycle(local_rows=[
            dict(swap_id="s1", status="applied", capacity_sats=1_000_000, duration_months=3)])
        assert lc.reconcile_ok() is True
        assert lc.breaker_tripped() is None
        row = db.lnplus_get_swap("s1")
        assert row["status"] == "applied"   # untouched — not cancelled_remote either

    def test_reconcile_pending_ghost_with_terminal_local_row_deletes_not_trips(self):
        """B5(b): a pending entry LN+ still lists that matches a local row
        we've already knowingly walked away from (failed) is cleanup, not a
        genuine ghost — delete_application attempted, breaker NOT tripped."""
        lc, db, rpc, client, *_ = _make_lifecycle(
            my_swaps={"pending": [{"id": "s1"}], "opening": [], "completed": []})
        db.set_config_override(SwapLifecycle._BACKFILL_FLAG, str(int(time.time())))
        db.lnplus_record_swap("s1", "failed", 1_000_000, 3)
        assert lc.reconcile_ok() is True
        client.delete_application.assert_called_once_with("s1")
        assert lc.breaker_tripped() is None

    def test_reconcile_pending_ghost_with_no_local_row_still_trips(self):
        """B5(b) counterpart: a true untracked ghost (no local row at all)
        must still trip the breaker."""
        lc, db, rpc, client, *_ = _make_lifecycle(
            my_swaps={"pending": [{"id": "ghost1"}], "opening": [], "completed": []})
        db.set_config_override(SwapLifecycle._BACKFILL_FLAG, str(int(time.time())))
        assert lc.reconcile_ok() is False
        assert "ghost1" in lc.breaker_tripped()
        client.delete_application.assert_not_called()

    def test_reconcile_trips_breaker_on_pending_ghost(self):
        """I1 regression: gate 0 only checked the 'opening' list for
        LN+-side entries with no local record. A pending application LN+
        knows about, with no local row at all, is an untracked live
        commitment ('ghost') and must trip the breaker too.

        Audit wave A / Part 2: an unknown pending entry on the very FIRST
        pass is now adopted by backfill_from_lnplus rather than treated as
        a ghost (see TestLnplusBackfill for that case) — the backfill flag
        is primed here to exercise genuine post-backfill ghost detection,
        which must still trip the breaker."""
        lc, db, rpc, client, *_ = _make_lifecycle(
            my_swaps={"pending": [{"id": "ghost1"}], "opening": [], "completed": []})
        db.set_config_override(SwapLifecycle._BACKFILL_FLAG, str(int(time.time())))
        assert lc.reconcile_ok() is False
        assert "ghost1" in lc.breaker_tripped()

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

    def test_invalid_outgoing_peer_pubkey_skips_row(self):
        """I3: an invalid outgoing_peer_pubkey from LN+ must never be written
        into the row or flow into connect/fundchannel — skip the row this
        pass entirely."""
        my = {"pending": [], "completed": [], "opening": [
            {"id": "s1", "outgoing_peer_pubkey": "not-a-real-pubkey",
             "deadline": int(time.time()) + 40 * 3600, "capacity_sats": 2_000_000}]}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my, local_rows=[
            dict(swap_id="s1", status="applied", capacity_sats=2_000_000,
                 duration_months=3)])
        rpc.listpeerchannels.return_value = {"channels": []}
        lc.run_watcher_once()
        rpc.fundchannel.assert_not_called()
        rpc.connect.assert_not_called()
        row = db.lnplus_get_swap("s1")
        assert row["status"] == "applied"
        assert row["outbound_peer"] is None

    def test_malformed_connect_address_falls_back_to_bare_pubkey(self):
        """I3: a malformed connect address from LN+ must not be interpolated
        into the connect target — fall back to a bare-pubkey connect."""
        peer = PK_A
        my = {"pending": [], "completed": [], "opening": [
            {"id": "s1", "outgoing_peer_pubkey": peer,
             "outgoing_peer_clearnet_address": "1.2.3.4:9735; rm -rf /",
             "deadline": int(time.time()) + 40 * 3600, "capacity_sats": 2_000_000}]}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my, local_rows=[
            dict(swap_id="s1", status="applied", capacity_sats=2_000_000,
                 duration_months=3, outbound_peer=peer)])
        rpc.listpeerchannels.return_value = {"channels": []}
        rpc.fundchannel.return_value = {"txid": "ff" * 32}
        lc.run_watcher_once()
        rpc.connect.assert_called_once_with(peer)

    def test_open_is_idempotent_when_channel_exists(self):
        peer = PK_A
        my = {"pending": [], "completed": [], "opening": [
            {"id": "s1", "outgoing_peer_pubkey": peer,
             "deadline": int(time.time()) + 10 * 3600, "capacity_sats": 2_000_000}]}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my, local_rows=[
            dict(swap_id="s1", status="applied", capacity_sats=2_000_000,
                 duration_months=3, outbound_peer=peer)])
        # I5(b): must match the row's committed capacity (2_000_000 sats) to
        # be recognized as OUR swap channel, not just any open channel to peer.
        rpc.listpeerchannels.return_value = {"channels": [
            {"peer_id": peer, "state": "CHANNELD_AWAITING_LOCKIN",
             "total_msat": 2_000_000_000, "funding_txid": "aa" * 32}]}
        lc.run_watcher_once()
        rpc.fundchannel.assert_not_called()
        client.complete_application.assert_called_once()

    def test_dual_fund_inflated_total_msat_matched_via_to_us_msat(self):
        """B7: under experimental-dual-fund a peer contribution inflates
        total_msat past our committed capacity — a crash-retry must still
        recognize the channel as ours via to_us_msat (our own promised
        contribution) rather than funding a SECOND channel."""
        peer = PK_A
        my = {"pending": [], "completed": [], "opening": [
            {"id": "s1", "outgoing_peer_pubkey": peer,
             "deadline": int(time.time()) + 40 * 3600, "capacity_sats": 5_000_000}]}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my, local_rows=[
            dict(swap_id="s1", status="applied", capacity_sats=5_000_000,
                 duration_months=3, outbound_peer=peer)])
        rpc.listpeerchannels.return_value = {"channels": [
            {"peer_id": peer, "state": "CHANNELD_AWAITING_LOCKIN",
             "total_msat": 7_000_000_000, "to_us_msat": 5_000_000_000,
             "funding_txid": "aa" * 32}]}
        lc.run_watcher_once()
        rpc.fundchannel.assert_not_called()
        client.complete_application.assert_called_once()

    def test_opens_new_channel_when_only_wrong_capacity_channel_exists(self):
        """I5(b): a pre-existing channel to the assigned peer that does NOT
        match the swap's committed capacity must not be claimed as our swap
        channel — the watcher must fund a NEW channel of the correct
        capacity rather than silently skip the open."""
        peer = PK_A
        deadline = int(time.time()) + 40 * 3600
        my = {"pending": [], "completed": [], "opening": [
            {"id": "s1", "outgoing_peer_pubkey": peer,
             "outgoing_peer_clearnet_address": "1.2.3.4:9735",
             "deadline": deadline, "capacity_sats": 2_000_000}]}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my, local_rows=[
            dict(swap_id="s1", status="applied", capacity_sats=2_000_000,
                 duration_months=3, outbound_peer=peer)])
        # A pre-existing channel to `peer` exists, but at a DIFFERENT
        # capacity than the swap terms (e.g. a regular planner open from
        # before the swap) — must not be mistaken for our swap channel.
        rpc.listpeerchannels.return_value = {"channels": [
            {"peer_id": peer, "state": "CHANNELD_NORMAL",
             "total_msat": 9_000_000_000}]}
        rpc.fundchannel.return_value = {"txid": "ff" * 32}
        lc.run_watcher_once()
        rpc.fundchannel.assert_called_once()
        assert rpc.fundchannel.call_args.args[1] == 2_000_000
        row = db.lnplus_get_swap("s1")
        assert row["status"] == "opened"
        assert row["channel_funding_txid"] == "ff" * 32

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

    def test_finalize_rpc_failure_defers_no_negative_rating(self):
        """B2: listpeerchannels raising must not be read as "no channel" ->
        permanent negative rating for an innocent peer. Row stays active,
        no rating/ignore/bump — retried next hourly pass."""
        peer, incoming = PK_A, PK_B
        lc, db, rpc, client, policy, ignore_fn = _make_lifecycle()
        db.lnplus_record_swap("s1", "active", 2_000_000, 3,
                              outbound_peer=peer, incoming_peer=incoming)
        db.lnplus_update_swap("s1", ends_at=int(time.time()) - 60)
        rpc.listpeerchannels.side_effect = Exception("rpc timeout")
        lc.run_watcher_once()
        row = db.lnplus_get_swap("s1")
        assert row["status"] == "active"
        client.create_rating.assert_not_called()
        ignore_fn.assert_not_called()
        assert db.lnplus_get_peer(incoming) is None

    def test_finalize_rpc_empty_channels_still_negative(self):
        """B2 counterpart: an RPC that ANSWERS with no matching channel is
        still a genuine defection — unchanged from prior behavior."""
        peer, incoming = PK_A, PK_B
        lc, db, rpc, client, policy, ignore_fn = _make_lifecycle()
        db.lnplus_record_swap("s1", "active", 2_000_000, 3,
                              outbound_peer=peer, incoming_peer=incoming)
        db.lnplus_update_swap("s1", ends_at=int(time.time()) - 60)
        rpc.listpeerchannels.return_value = {"channels": []}
        lc.run_watcher_once()
        row = db.lnplus_get_swap("s1")
        assert row["status"] == "ended"
        client.create_rating.assert_called_once_with("s1", "negative")
        ignore_fn.assert_called_once()

    def test_get_status_after_watcher_pass_carries_last_watcher_pass(self):
        """B13: get_status must surface recent_failed, backfill_done, and
        last_watcher_pass (in-memory, recorded at the end of every pass)."""
        lc, db, rpc, client, *_ = _make_lifecycle()
        db.lnplus_record_swap("s1", "failed", 1_000_000, 3)
        db.lnplus_update_swap("s1", outcome="abandoned by operator")
        before = int(time.time())
        lc.run_watcher_once()
        status = lc.get_status()
        assert status["last_watcher_pass"] is not None
        assert status["last_watcher_pass"]["ts"] >= before
        assert "summary" in status["last_watcher_pass"]
        assert status["backfill_done"] is True
        assert any(r["swap_id"] == "s1" for r in status["recent_failed"])

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

    def test_abandoned_row_not_resurrected_by_stale_lnplus_opening(self):
        """C2 regression: a row abandoned locally (status 'failed', e.g. via
        revenue-lnplus-abandon) must stay terminal even if LN+ still lists
        the swap under 'opening' for a stale cycle or two. The watcher used
        to flip status back to 'opening' unconditionally and fund it."""
        peer = PK_A
        my = {"pending": [], "completed": [], "opening": [
            {"id": "s1", "outgoing_peer_pubkey": peer,
             "deadline": int(time.time()) + 40 * 3600, "capacity_sats": 2_000_000}]}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my, local_rows=[
            dict(swap_id="s1", status="failed", capacity_sats=2_000_000,
                 duration_months=3, outbound_peer=peer)])
        rpc.listpeerchannels.return_value = {"channels": []}
        rpc.fundchannel.return_value = {"txid": "ff" * 32}
        lc.run_watcher_once()
        rpc.fundchannel.assert_not_called()
        client.complete_application.assert_not_called()
        row = db.lnplus_get_swap("s1")
        assert row["status"] == "failed"

    def test_phase_3b_skips_dead_swap_on_successful_fetch(self):
        """B1: a local 'opening' row LN+ no longer recognizes (creator
        cancelled / we were banned / swap deleted) must NOT be funded when
        get_my_swaps succeeds and returns it in neither opening nor
        completed."""
        peer = PK_A
        lc, db, rpc, client, *_ = _make_lifecycle(
            my_swaps={"pending": [], "opening": [], "completed": []},
            local_rows=[dict(swap_id="s1", status="opening", capacity_sats=2_000_000,
                             duration_months=3, outbound_peer=peer)])
        db.lnplus_update_swap("s1", deadline_at=int(time.time()) + 40 * 3600)
        rpc.listpeerchannels.return_value = {"channels": []}
        lc.run_watcher_once()
        rpc.fundchannel.assert_not_called()
        row = db.lnplus_get_swap("s1")
        assert row["status"] == "opening"   # untouched, not funded

    def test_phase_3b_still_drives_from_ledger_on_outage(self):
        """B1 counterpart: when get_my_swaps itself fails (outage), phase 3b
        must still drive off the local ledger unconditionally — a funded
        deadline cannot wait on LN+ being reachable."""
        from modules.lnplus_swaps import LNPlusError
        peer = PK_A
        lc, db, rpc, client, *_ = _make_lifecycle(
            local_rows=[dict(swap_id="s1", status="opening", capacity_sats=2_000_000,
                             duration_months=3, outbound_peer=peer)])
        db.lnplus_update_swap("s1", deadline_at=int(time.time()) + 40 * 3600)
        client.get_my_swaps.side_effect = LNPlusError("down")
        rpc.listpeerchannels.return_value = {"channels": []}
        rpc.fundchannel.return_value = {"txid": "ff" * 32}
        lc.run_watcher_once()
        rpc.fundchannel.assert_called_once()

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
        # I5(a): no pre-existing channel to the inferred outbound peer at
        # apply time (overridden in phase 2 below for the watcher's own
        # idempotency check).
        rpc.listpeerchannels.return_value = {"channels": []}
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


class TestLnplusBackfill:
    """Audit wave A / Part 2: adopt pre-existing (manual) LN+ swaps into
    the local ledger. See docs/plans/2026-07-05-lnplus-swap-automation-design.md
    and .superpowers/sdd/audit-wave-a-brief.md for the mechanism."""

    def test_pending_entry_imports_applied_row(self):
        my = {"pending": [{"id": "p1", "capacity_sats": 3_000_000, "duration_months": 3}],
              "opening": [], "completed": []}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my)
        result = lc.backfill_from_lnplus()
        row = db.lnplus_get_swap("p1")
        assert row["status"] == "applied"
        assert row["capacity_sats"] == 3_000_000
        assert row["duration_months"] == 3
        assert row["outbound_peer"] is None
        assert row["incoming_peer"] is None
        assert row["our_identifier"] is None
        assert row["applied_at"] > 0
        assert result["imported"]["pending"] == 1

    def test_opening_entry_imports_opening_row(self):
        deadline = int(time.time()) + 40 * 3600
        my = {"pending": [], "opening": [
            {"id": "o1", "outgoing_peer_pubkey": PK_A, "deadline": deadline,
             "capacity_sats": 2_000_000, "duration_months": 3}], "completed": []}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my)
        result = lc.backfill_from_lnplus()
        row = db.lnplus_get_swap("o1")
        assert row["status"] == "opening"
        assert row["outbound_peer"] == PK_A
        assert row["deadline_at"] == deadline
        assert row["capacity_sats"] == 2_000_000
        assert result["imported"]["opening"] == 1

    def test_running_contract_imports_opened_row_with_derived_outbound(self):
        ends = int(time.time()) + 30 * 86400
        detail = {"participants": [
            {"participant_identifier": "A", "pubkey": FLEET_PK},
            {"participant_identifier": "B", "pubkey": PK_A},
            {"participant_identifier": "C", "pubkey": PK_B}]}
        my = {"pending": [], "opening": [], "completed": [
            {"id": "c1", "incoming_peer_pubkey": PK_B, "ends": ends,
             "capacity_sats": 4_000_000, "duration_months": 3}]}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my)
        rpc.getinfo.return_value = {"id": FLEET_PK}
        client.get_swap.return_value = detail
        result = lc.backfill_from_lnplus()
        row = db.lnplus_get_swap("c1")
        assert row["status"] == "opened"
        assert row["outbound_peer"] == PK_A   # A (us) -> next letter B -> PK_A
        assert row["incoming_peer"] == PK_B
        assert row["ends_at"] == ends
        assert row["opened_at"] is None
        assert result["imported"]["active"] == 1

    def test_ended_contract_imports_ended_row(self):
        ends = int(time.time()) - 86400
        my = {"pending": [], "opening": [], "completed": [
            {"id": "e1", "incoming_peer_pubkey": PK_B, "ends": ends,
             "capacity_sats": 1_500_000, "duration_months": 3}]}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my)
        result = lc.backfill_from_lnplus()
        row = db.lnplus_get_swap("e1")
        assert row["status"] == "ended"
        assert row["outcome"] == "imported_pre_automation"
        assert row["ends_at"] == ends
        assert row["incoming_peer"] == PK_B
        client.create_rating.assert_not_called()
        assert result["imported"]["ended"] == 1

    def test_rerun_is_noop(self):
        ends_running = int(time.time()) + 30 * 86400
        ends_ended = int(time.time()) - 86400
        my = {"pending": [{"id": "p1", "capacity_sats": 1_000_000, "duration_months": 3}],
              "opening": [{"id": "o1", "outgoing_peer_pubkey": PK_A,
                          "deadline": int(time.time()) + 40 * 3600,
                          "capacity_sats": 2_000_000, "duration_months": 3}],
              "completed": [
                  {"id": "c1", "incoming_peer_pubkey": PK_B, "ends": ends_running,
                   "capacity_sats": 3_000_000, "duration_months": 3},
                  {"id": "e1", "incoming_peer_pubkey": PK_B, "ends": ends_ended,
                   "capacity_sats": 1_500_000, "duration_months": 3}]}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my)
        rpc.getinfo.return_value = {"id": FLEET_PK}
        client.get_swap.return_value = {"participants": [
            {"participant_identifier": "A", "pubkey": FLEET_PK},
            {"participant_identifier": "B", "pubkey": PK_A}]}
        lc.backfill_from_lnplus()
        before = {sid: db.lnplus_get_swap(sid) for sid in ("p1", "o1", "c1", "e1")}
        result2 = lc.backfill_from_lnplus()
        after = {sid: db.lnplus_get_swap(sid) for sid in ("p1", "o1", "c1", "e1")}
        assert before == after
        assert result2["imported"] == {"pending": 0, "opening": 0, "active": 0, "ended": 0}
        assert set(result2["skipped"]) == {"p1", "o1", "c1", "e1"}

    def test_automation_owned_row_untouched(self):
        my = {"pending": [], "opening": [
            {"id": "s1", "outgoing_peer_pubkey": PK_A,
             "deadline": int(time.time()) + 40 * 3600,
             "capacity_sats": 2_000_000, "duration_months": 3}], "completed": []}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my, local_rows=[
            dict(swap_id="s1", status="opening", capacity_sats=9_999_999,
                 duration_months=6, outbound_peer=PK_B)])
        db.lnplus_update_swap("s1", channel_funding_txid="aa" * 32)
        result = lc.backfill_from_lnplus()
        row = db.lnplus_get_swap("s1")
        assert row["capacity_sats"] == 9_999_999
        assert row["outbound_peer"] == PK_B
        assert row["channel_funding_txid"] == "aa" * 32
        assert "s1" in result["skipped"]
        assert result["imported"] == {"pending": 0, "opening": 0, "active": 0, "ended": 0}

    def test_first_watcher_pass_imports_and_sets_flag_no_breaker(self):
        my = {"pending": [{"id": "p1", "capacity_sats": 1_000_000, "duration_months": 3}],
              "opening": [{"id": "o1", "outgoing_peer_pubkey": PK_A,
                          "deadline": int(time.time()) + 40 * 3600,
                          "capacity_sats": 2_000_000, "duration_months": 3}],
              "completed": []}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my)
        rpc.listpeerchannels.return_value = {"channels": []}
        rpc.fundchannel.return_value = {"txid": "ff" * 32}
        lc.run_watcher_once()
        assert lc.breaker_tripped() is None
        assert db.get_config_override(SwapLifecycle._BACKFILL_FLAG) is not None

    def test_new_ghost_after_flag_set_trips_breaker(self):
        lc, db, rpc, client, *_ = _make_lifecycle(
            my_swaps={"pending": [], "opening": [], "completed": []})
        lc.run_watcher_once()  # sets flag, nothing to import
        assert db.get_config_override(SwapLifecycle._BACKFILL_FLAG) is not None
        client.get_my_swaps.return_value = {
            "pending": [{"id": "ghost1"}], "opening": [], "completed": []}
        lc.run_watcher_once()
        assert lc.breaker_tripped() is not None
        assert "ghost1" in lc.breaker_tripped()

    def test_imported_opening_entry_opens_same_pass(self):
        deadline = int(time.time()) + 40 * 3600
        my = {"pending": [], "opening": [
            {"id": "o1", "outgoing_peer_pubkey": PK_A,
             "outgoing_peer_clearnet_address": "1.2.3.4:9735",
             "deadline": deadline, "capacity_sats": 2_000_000,
             "duration_months": 3}], "completed": []}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my)
        rpc.listpeerchannels.return_value = {"channels": []}
        rpc.fundchannel.return_value = {"txid": "ff" * 32}
        lc.run_watcher_once()
        rpc.fundchannel.assert_called_once()
        assert rpc.fundchannel.call_args.args[1] == 2_000_000
        row = db.lnplus_get_swap("o1")
        assert row["status"] == "opened"

    def test_imported_running_contract_activates_same_pass(self):
        ends = int(time.time()) + 30 * 86400
        my = {"pending": [], "opening": [], "completed": [
            {"id": "c1", "incoming_peer_pubkey": PK_B, "ends": ends,
             "capacity_sats": 3_000_000, "duration_months": 3}]}
        lc, db, rpc, client, policy, _ = _make_lifecycle(my_swaps=my)
        rpc.getinfo.return_value = {"id": FLEET_PK}
        client.get_swap.return_value = {"participants": [
            {"participant_identifier": "A", "pubkey": FLEET_PK},
            {"participant_identifier": "B", "pubkey": PK_A}]}
        lc.run_watcher_once()
        row = db.lnplus_get_swap("c1")
        assert row["status"] == "active"
        assert row["ends_at"] == ends
        policy.add_tag.assert_called_once_with(PK_A, "no_close")

    def test_imported_running_contract_get_swap_failure_leaves_outbound_null(self):
        ends = int(time.time()) + 30 * 86400
        my = {"pending": [], "opening": [], "completed": [
            {"id": "c1", "incoming_peer_pubkey": PK_B, "ends": ends,
             "capacity_sats": 3_000_000, "duration_months": 3}]}
        lc, db, rpc, client, policy, _ = _make_lifecycle(my_swaps=my)
        client.get_swap.side_effect = Exception("lnplus down")
        result = lc.run_watcher_once()   # must not raise
        row = db.lnplus_get_swap("c1")
        assert row is not None
        assert row["outbound_peer"] is None
        assert row["status"] == "active"   # phase 4 still activates this pass
        policy.add_tag.assert_not_called()  # nothing to tag with a NULL peer
        error_logged = any(
            call.kwargs.get("level") == "error" and "c1" in call.args[0]
            for call in lc._plugin.log.call_args_list)
        assert error_logged
        assert "c1" not in result.get("errors", [])   # handled, no crash

    def test_imported_ended_contract_no_rating(self):
        ends = int(time.time()) - 86400
        my = {"pending": [], "opening": [], "completed": [
            {"id": "e1", "incoming_peer_pubkey": PK_B, "ends": ends,
             "capacity_sats": 1_000_000, "duration_months": 3}]}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my)
        lc.run_watcher_once()
        row = db.lnplus_get_swap("e1")
        assert row["status"] == "ended"
        client.create_rating.assert_not_called()

    def test_backfill_failure_blocks_flag_and_divergence_check(self):
        my = {"pending": [
            {"id": "p1", "capacity_sats": 1_000_000, "duration_months": 3},
            {"id": "p2", "capacity_sats": 1_000_000, "duration_months": 3}],
            "opening": [], "completed": []}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my)
        real_record = db.lnplus_record_swap
        call_count = {"n": 0}

        def _flaky_record(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("db boom")
            return real_record(*args, **kwargs)

        with patch.object(db, "lnplus_record_swap", side_effect=_flaky_record):
            ok = lc._reconcile(my)
        assert ok is False
        assert db.get_config_override(SwapLifecycle._BACKFILL_FLAG) is None
        assert lc.breaker_tripped() is None

    def test_invalid_pubkeys_result_in_null_fields(self):
        ends = int(time.time()) - 86400
        my = {"pending": [],
              "opening": [{"id": "o1", "outgoing_peer_pubkey": "not-a-pubkey",
                          "deadline": int(time.time()) + 40 * 3600,
                          "capacity_sats": 2_000_000, "duration_months": 3}],
              "completed": [{"id": "e1", "incoming_peer_pubkey": "also-bad",
                            "ends": ends, "capacity_sats": 1_000_000,
                            "duration_months": 3}]}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my)
        result = lc.backfill_from_lnplus()
        row_o = db.lnplus_get_swap("o1")
        row_e = db.lnplus_get_swap("e1")
        assert row_o["outbound_peer"] is None
        assert row_e["incoming_peer"] is None
        assert any("o1" in w for w in result["warnings"])
        assert any("e1" in w for w in result["warnings"])

    def test_backfill_returns_per_category_counts(self):
        my = {"pending": [{"id": "p1", "capacity_sats": 1_000_000, "duration_months": 3}],
              "opening": [{"id": "o1", "outgoing_peer_pubkey": PK_A,
                          "deadline": int(time.time()) + 40 * 3600,
                          "capacity_sats": 2_000_000, "duration_months": 3}],
              "completed": [
                  {"id": "c1", "incoming_peer_pubkey": PK_B,
                   "ends": int(time.time()) + 30 * 86400,
                   "capacity_sats": 3_000_000, "duration_months": 3},
                  {"id": "e1", "incoming_peer_pubkey": PK_B,
                   "ends": int(time.time()) - 86400,
                   "capacity_sats": 1_500_000, "duration_months": 3}]}
        lc, db, rpc, client, *_ = _make_lifecycle(my_swaps=my)
        rpc.getinfo.return_value = {"id": FLEET_PK}
        client.get_swap.return_value = {"participants": [
            {"participant_identifier": "A", "pubkey": FLEET_PK},
            {"participant_identifier": "B", "pubkey": PK_A}]}
        result = lc.backfill_from_lnplus()
        assert result["imported"] == {"pending": 1, "opening": 1, "active": 1, "ended": 1}
        assert result["skipped"] == []


from types import SimpleNamespace
from tests.plugin_test_utils import load_plugin_module


class TestLnplusPluginWiring:
    """Audit wave B: fixes that live in cl-revenue-ops.py rather than
    modules/lnplus_swaps.py (B5(a) abandon RPC, B6 watcher interval clamp +
    dynamic config refresh)."""

    def test_abandon_applied_row_deletes_live_application(self):
        """B5(a): abandoning an 'applied' row leaves a LIVE application on
        LN+'s side unless we also ask LN+ to delete it (best-effort — the
        reconcile pending-ghost path in B5(b) covers a delete failure)."""
        mod = load_plugin_module()
        mod.database = MagicMock()
        mod.database.lnplus_get_swap.return_value = {"status": "applied"}
        mod.lnplus_lifecycle = MagicMock()
        mod.lnplus_client = MagicMock()
        result = mod.revenue_lnplus_abandon(mod.plugin, swap_id="s1")
        assert result["status"] == "abandoned"
        mod.lnplus_client.delete_application.assert_called_once_with("s1")
        mod.lnplus_lifecycle.trip_breaker.assert_called_once()

    def test_abandon_opening_row_does_not_call_delete_application(self):
        """B5(a): delete_application is only valid while a swap is pending
        on LN+'s side — an 'opening'/'opened' row must not attempt it."""
        mod = load_plugin_module()
        mod.database = MagicMock()
        mod.database.lnplus_get_swap.return_value = {"status": "opening"}
        mod.lnplus_lifecycle = MagicMock()
        mod.lnplus_client = MagicMock()
        result = mod.revenue_lnplus_abandon(mod.plugin, swap_id="s1")
        assert result["status"] == "abandoned"
        mod.lnplus_client.delete_application.assert_not_called()

    def test_abandon_delete_application_failure_does_not_raise(self):
        """B5(a): delete_application is best-effort — a failure must not
        crash the abandon RPC."""
        mod = load_plugin_module()
        mod.database = MagicMock()
        mod.database.lnplus_get_swap.return_value = {"status": "applied"}
        mod.lnplus_lifecycle = MagicMock()
        mod.lnplus_client = MagicMock()
        mod.lnplus_client.delete_application.side_effect = Exception("lnplus down")
        result = mod.revenue_lnplus_abandon(mod.plugin, swap_id="s1")
        assert result["status"] == "abandoned"

    def test_watcher_interval_clamp_helper(self):
        """B6(a): one pass per >=4h still gives >=12 retries inside a 48h
        deadline — clamp to [300s, 14400s]."""
        mod = load_plugin_module()
        assert mod._clamp_lnplus_watcher_interval(100) == 300
        assert mod._clamp_lnplus_watcher_interval(999_999) == 14400
        assert mod._clamp_lnplus_watcher_interval(1800) == 1800

    def test_fleet_pubkeys_and_watcher_interval_options_are_dynamic(self):
        """B6(b): fleet growth (new pubkeys) and interval tuning must not
        require a plugin restart."""
        mod = load_plugin_module()
        for option in ("revenue-ops-lnplus-fleet-pubkeys",
                       "revenue-ops-lnplus-watcher-interval"):
            assert mod.plugin.options[option]["dynamic"] is True

    def test_dynamic_refresh_updates_fleet_pubkeys_and_watcher_interval(self):
        """B6(b): _refresh_dynamic_config's cast-tuple loop must cover both
        new options (extending the loop to support a "str" cast for
        fleet_pubkeys)."""
        mod = load_plugin_module()
        mod.config = Config()
        mod.boltz_manager = None
        fake_rpc = MagicMock()
        fake_rpc.listconfigs.return_value = {"configs": {
            "revenue-ops-lnplus-fleet-pubkeys": {"value_str": PK_A + "," + PK_B},
            "revenue-ops-lnplus-watcher-interval": {"value_str": "900"},
        }}
        mod.safe_plugin = SimpleNamespace(rpc=fake_rpc)
        mod._refresh_dynamic_config()
        assert mod.config.lnplus_fleet_pubkeys == PK_A + "," + PK_B
        assert mod.config.lnplus_watcher_interval == 900
