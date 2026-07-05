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
