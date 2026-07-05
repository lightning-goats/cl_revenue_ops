"""LN+ (lightningnetwork.plus) liquidity swap automation.

Three collaborators, wired in cl-revenue-ops.py init:
  LNPlusClient   — stdlib-urllib HTTPS client + signmessage auth
  SwapEvaluator  — pre-application gate chain (spec gates 0-9)
  SwapLifecycle  — obligations watcher / state machine (spec gates 10-14)

Design spec: docs/plans/2026-07-05-lnplus-swap-automation-design.md
Application to a swap is an IRREVERSIBLE COMMITMENT (48h open deadline once
filled). Every gate lives before create_application; everything after only
executes obligations safely.
"""

import json
import re
import time
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

BASE_URL = "https://lightningnetwork.plus/api/2"
_MAX_RESPONSE_BYTES = 1_000_000
_CHALLENGE_MAX_LEN = 500
_CHALLENGE_FORBIDDEN_PREFIXES = ("lnbc", "lntb", "lnbcrt", "cbor", "psbt")
_PUBKEY_RE = re.compile(r"^0[23][0-9a-fA-F]{64}$")


def _valid_pubkey(value) -> bool:
    return isinstance(value, str) and bool(_PUBKEY_RE.match(value))


def _parse_ts(value) -> Optional[int]:
    """ISO-8601 string or epoch number -> epoch seconds, else None."""
    if isinstance(value, (int, float)) and value > 0:
        return int(value)
    if isinstance(value, str):
        try:
            return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
        except ValueError:
            return None
    return None


class LNPlusError(Exception):
    def __init__(self, message: str, http_status: Optional[int] = None):
        super().__init__(message)
        self.http_status = http_status


class LNPlusClient:
    """Thin HTTPS client for the LN+ v2 API. No business logic."""

    def __init__(self, plugin, rpc, base_url: str = BASE_URL, timeout_seconds: int = 30):
        self._plugin = plugin
        self._rpc = rpc
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    # -- transport -------------------------------------------------------
    def _request(self, path: str, params: Optional[Dict] = None, method: str = "GET") -> Any:
        url = f"{self._base_url}/{path}"
        data = None
        headers = {"Accept": "application/json", "User-Agent": "cl-revenue-ops"}
        if method == "POST":
            data = urllib.parse.urlencode(params or {}).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        elif params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read(_MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as e:
            body = b""
            try:
                body = e.read(_MAX_RESPONSE_BYTES)
            except Exception:
                pass
            raise LNPlusError(f"LN+ HTTP {e.code} on {path}: {body[:500]!r}",
                              http_status=e.code)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            raise LNPlusError(f"LN+ unreachable on {path}: {e}")
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise LNPlusError(f"LN+ response too large on {path}")
        try:
            return json.loads(raw)
        except (ValueError, UnicodeDecodeError) as e:
            raise LNPlusError(f"LN+ invalid JSON on {path}: {e}")

    # -- auth --------------------------------------------------------------
    def _auth_params(self) -> Dict[str, str]:
        challenge = self._request("get_message")
        message = challenge.get("message") if isinstance(challenge, dict) else None
        self._validate_challenge(message)
        signed = self._rpc.signmessage(message)
        signature = signed.get("zbase") if isinstance(signed, dict) else None
        if not signature:
            raise LNPlusError("signmessage returned no zbase signature")
        return {"message": message, "signature": signature}

    @staticmethod
    def _validate_challenge(message) -> None:
        """Gate 15: only sign strings that look like LN+ auth challenges."""
        if not isinstance(message, str) or not message:
            raise LNPlusError("LN+ challenge missing")
        if len(message) > _CHALLENGE_MAX_LEN:
            raise LNPlusError("LN+ challenge suspiciously long — refusing to sign")
        if not message.isprintable():
            raise LNPlusError("LN+ challenge not printable — refusing to sign")
        lowered = message.lower()
        for prefix in _CHALLENGE_FORBIDDEN_PREFIXES:
            if lowered.startswith(prefix):
                raise LNPlusError("LN+ challenge looks like an invoice/PSBT — refusing to sign")

    # -- endpoints ---------------------------------------------------------
    def get_applicable_swaps(self) -> List[Dict]:
        result = self._request("get_applicable_swaps", self._auth_params(), method="POST")
        swaps = result.get("swaps", result) if isinstance(result, dict) else result
        return swaps if isinstance(swaps, list) else []

    def get_swap(self, swap_id) -> Dict:
        return self._request(f"get_swap/id={urllib.parse.quote(str(swap_id))}")

    def get_my_swaps(self) -> Dict:
        result = self._request("get_my_swaps", self._auth_params(), method="POST")
        if not isinstance(result, dict):
            raise LNPlusError("get_my_swaps: unexpected payload")
        return {k: result.get(k) or [] for k in ("pending", "opening", "completed")}

    def create_application(self, swap_id) -> Dict:
        params = self._auth_params()
        params["id"] = str(swap_id)
        return self._request("create_application", params, method="POST")

    def delete_application(self, swap_id) -> Dict:
        params = self._auth_params()
        params["id"] = str(swap_id)
        return self._request("delete_application", params, method="POST")

    def complete_application(self, swap_id) -> Dict:
        params = self._auth_params()
        params["id"] = str(swap_id)
        return self._request("complete_application", params, method="POST")

    def create_rating(self, swap_id, rating: str) -> Dict:
        if rating not in ("positive", "negative"):
            raise LNPlusError(f"invalid rating {rating!r}")
        params = self._auth_params()
        params["id"] = str(swap_id)
        params["rating"] = rating
        return self._request("create_rating", params, method="POST")
