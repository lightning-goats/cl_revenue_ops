"""
Boltz CLI integration for cl-revenue-ops.

Provides a synchronous manager used by plugin RPC methods to call `boltzcli`
against a local `boltzd` instance. Designed as a best-effort integration that
works even when boltzd autoswap is unstable; revenue ops can still perform
manual quotes/swaps and enforce a local daily fee budget for swap fees.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .utils import parse_msat


class BoltzCliError(RuntimeError):
    """Raised when boltzcli execution fails."""


# ---------------------------------------------------------------------------
# On-chain address validation (P1-006, address-validation part)
#
# Self-contained (no third-party dependency) structural validation of a
# withdrawal destination. Its purpose is defensive: reject an obviously
# malformed / wrong-network destination before it is handed to boltzcli.
# It is intentionally conservative and does NOT attempt to enumerate every
# exotic address encoding.
#
#   * bech32 / bech32m (BIP173/BIP350): full checksum + witness-program
#     validation. Covers BTC (bc/tb/bcrt) and Liquid unconfidential segwit
#     (ex/tex/ert).
#   * base58check: full double-SHA256 checksum + version-byte gating. Covers
#     BTC legacy/P2SH (mainnet + test/regtest).
#   * Liquid confidential segwit (blech32: lq/tlq/el) uses a distinct 12-symbol
#     checksum we do not fully verify; we accept it on a strict prefix +
#     charset + length structural check (documented limitation). Combined with
#     the `--` argv terminator (P1-015) this still blocks flag-injection and
#     junk input.
#
# "network" here means the asset network (Bitcoin vs Liquid), which is what
# the wallet currency selects; mainnet/testnet/regtest are all accepted for a
# given currency (the configured boltzd instance is bound to one chain).
# ---------------------------------------------------------------------------

_BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

_BTC_BECH32_HRPS = {"bc", "tb", "bcrt"}
_LBTC_BECH32_HRPS = {"ex", "tex", "ert"}          # unconfidential Liquid segwit
_LBTC_BLECH32_HRPS = {"lq", "tlq", "el"}          # confidential Liquid segwit
# BTC base58 version bytes: P2PKH/P2SH for mainnet and test/regtest.
_BTC_BASE58_VERSIONS = {0x00, 0x05, 0x6f, 0xc4}


def _bech32_polymod(values) -> int:
    generator = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
    chk = 1
    for v in values:
        top = chk >> 25
        chk = ((chk & 0x1ffffff) << 5) ^ v
        for i in range(5):
            chk ^= generator[i] if ((top >> i) & 1) else 0
    return chk


def _bech32_hrp_expand(hrp: str):
    return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]


def _bech32_decode(addr: str):
    """Return (hrp, data_without_checksum, spec) or (None, None, None).

    spec is 'bech32' or 'bech32m'. Mixed-case input is rejected per BIP173.
    """
    if any(ord(c) < 33 or ord(c) > 126 for c in addr):
        return None, None, None
    if addr.lower() != addr and addr.upper() != addr:
        return None, None, None
    addr = addr.lower()
    if len(addr) > 90:
        return None, None, None
    pos = addr.rfind("1")
    if pos < 1 or pos + 7 > len(addr):
        return None, None, None
    hrp = addr[:pos]
    try:
        data = [_BECH32_CHARSET.index(c) for c in addr[pos + 1:]]
    except ValueError:
        return None, None, None
    const = _bech32_polymod(_bech32_hrp_expand(hrp) + data)
    if const == 1:
        spec = "bech32"
    elif const == 0x2bc830a3:
        spec = "bech32m"
    else:
        return None, None, None
    return hrp, data[:-6], spec


def _convertbits(data, frombits, tobits, pad=True):
    acc = 0
    bits = 0
    ret = []
    maxv = (1 << tobits) - 1
    max_acc = (1 << (frombits + tobits - 1)) - 1
    for value in data:
        if value < 0 or (value >> frombits):
            return None
        acc = ((acc << frombits) | value) & max_acc
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append((acc >> bits) & maxv)
    if pad:
        if bits:
            ret.append((acc << (tobits - bits)) & maxv)
    elif bits >= frombits or ((acc << (tobits - bits)) & maxv):
        return None
    return ret


def _valid_segwit_program(data) -> bool:
    """data is the 5-bit payload (witver + program), checksum already stripped."""
    if not data:
        return False
    witver = data[0]
    if witver < 0 or witver > 16:
        return False
    prog = _convertbits(data[1:], 5, 8, False)
    if prog is None:
        return False
    if len(prog) < 2 or len(prog) > 40:
        return False
    if witver == 0 and len(prog) not in (20, 32):
        return False
    return True


def _base58check_decode(s: str):
    """Return the decoded payload (version byte + data, checksum stripped) or None."""
    if not s:
        return None
    num = 0
    for ch in s:
        idx = _B58_ALPHABET.find(ch)
        if idx == -1:
            return None
        num = num * 58 + idx
    combined = num.to_bytes((num.bit_length() + 7) // 8, "big") if num else b""
    n_pad = len(s) - len(s.lstrip("1"))
    raw = (b"\x00" * n_pad) + combined
    if len(raw) < 5:
        return None
    payload, checksum = raw[:-4], raw[-4:]
    if hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4] != checksum:
        return None
    return payload


def _validate_onchain_address(address: str, currency: str) -> bool:
    """Structurally validate an on-chain withdrawal destination for currency.

    Returns True only for a well-formed address of the correct asset network.
    """
    if not isinstance(address, str):
        return False
    addr = address.strip()
    if not addr or any(c.isspace() for c in addr):
        return False

    cur = (currency or "").strip().upper()
    if cur in ("L-BTC", "LBTC"):
        cur = "LBTC"

    lower = addr.lower()

    # Confidential Liquid segwit (blech32): strict structural check only.
    if cur == "LBTC":
        for hrp in _LBTC_BLECH32_HRPS:
            if lower.startswith(hrp + "1"):
                body = lower[len(hrp) + 1:]
                if 6 <= len(body) <= 200 and all(c in _BECH32_CHARSET for c in body):
                    return True
                return False

    # bech32 / bech32m segwit.
    hrp, data, spec = _bech32_decode(addr)
    if hrp is not None:
        if cur == "BTC" and hrp not in _BTC_BECH32_HRPS:
            return False
        if cur == "LBTC" and hrp not in _LBTC_BECH32_HRPS:
            return False
        if cur not in ("BTC", "LBTC"):
            return False
        return _valid_segwit_program(data)

    # base58check legacy / P2SH.
    payload = _base58check_decode(addr)
    if payload is not None and len(payload) >= 1:
        version = payload[0]
        if cur == "BTC":
            return version in _BTC_BASE58_VERSIONS
        if cur == "LBTC":
            # Liquid uses distinct version bytes; a BTC-network version byte
            # here would be a wrong-network address.
            return version not in _BTC_BASE58_VERSIONS
        return False

    return False


def _validate_swap_destination(destination: str) -> bool:
    """Validate a swap destination address whose asset network is not known
    from the call signature (refund/claim). Accept a structurally valid BTC or
    Liquid address; reject garbage / flag-like values. The literal "wallet"
    keyword (send to the internal wallet) is handled by the caller.
    """
    return (
        _validate_onchain_address(destination, "BTC")
        or _validate_onchain_address(destination, "LBTC")
    )


@dataclass
class BoltzCliConfig:
    enabled: bool = False
    cli_path: str = "/usr/local/bin/boltzcli"
    datadir: str = "/var/lib/boltz"
    use_sudo: bool = False
    sudo_user: str = "boltz"
    timeout_seconds: int = 60
    daily_budget_sats: int = 3000
    enforce_budget: bool = True
    btc_wallet: str = "CLN"
    lbtc_wallet: str = "LOOP-LBTC"
    routing_fee_limit_ppm: int = 0  # 0 = no limit (boltzcli default)
    # DD3 / P1-006: hard cap on a single on-chain withdraw so a typo or
    # automation bug cannot sweep the wallet in one call. 0 disables the cap.
    max_withdraw_sats: int = 10_000_000


class BoltzCliManager:
    def __init__(self, plugin, rpc, config: BoltzCliConfig):
        self.plugin = plugin
        self.rpc = rpc
        self.cfg = config
        self._swap_journal_file = os.path.join(self.cfg.datadir, "cl_revenue_ops_swap_journal.json")
        self._ignored_external_swaps_file = os.path.join(self.cfg.datadir, "cl_revenue_ops_ignored_external_swaps.json")
        # B1/B3 FIX: Serialize file-based load-modify-save to prevent lost updates
        self._journal_lock = threading.Lock()
        self._ignored_swaps_lock = threading.Lock()
        # P0-1 FIX: Serialize budget-check + swap-create to prevent TOCTOU race
        self._swap_creation_lock = threading.Lock()
        self.data_service = None
        # Optional callback set by cl-revenue-ops plugin to provide non-Boltz liquidity costs
        # (e.g. market rebalance spend/reservations) for unified budget accounting.
        self.external_liquidity_cost_provider = None
        # Optional callback returning unified budget limit info for all liquidity costs.
        self.global_budget_limit_provider = None
        # Optional callback returning the configured daily structural envelope
        # (boltz_structural_budget_sats_per_day) in sats. Read at call time so
        # runtime config changes apply. When > 0, structural loop-outs skip
        # the per-channel capex gate (the envelope + unified budget gate them
        # instead); when absent/0/unreadable, no bypass (fail closed).
        self.structural_envelope_provider = None
        # Capability cache: some CLN+boltzd combinations reject reverse-swap chanIds.
        self._reverse_chanids_supported: Optional[bool] = None
        self._capex_engine = None
        # P4-014: context for the in-flight swap's unified-budget reservation,
        # stashed by _loop_out_locked (under _swap_creation_lock) so the loop_out
        # wrapper can settle-or-release it on every exit including exceptions.
        self._pending_swap_budget = None

    def set_capex_engine(self, engine):
        """Inject the unified capex budget engine."""
        self._capex_engine = engine

    def check_tactical_budget(self, estimated_fee_sats: int, channel_id=None) -> dict:
        """Check if the capex tactical budget allows a swap.

        Pure treasury swaps (channel_id=None) are gated by tactical budget.
        Channel-targeted swaps are not gated by the tactical budget; they are
        gated by the channel's own capex budget via check_channel_capex_budget
        (swap paths must call both).
        Without an engine, the gate is not applied (backward compat).

        Returns:
            {"allowed": bool, "reason": str or None}
        """
        if not self._capex_engine:
            return {"allowed": True, "reason": None}

        # Channel-targeted swaps use the per-channel capex budget gate instead
        # (see check_channel_capex_budget, applied by loop_in/loop_out).
        if channel_id is not None:
            return {"allowed": True, "reason": None}

        # Pure treasury: check tactical budget
        tactical = self._capex_engine.get_tactical_budget()
        if estimated_fee_sats > tactical:
            return {
                "allowed": False,
                "reason": (
                    f"Tactical budget: estimated fee {estimated_fee_sats} sats "
                    f"exceeds tactical budget {tactical} sats"
                ),
            }
        return {"allowed": True, "reason": None}

    def _structural_envelope_sats(self) -> int:
        """Configured daily structural envelope in sats (0 = disabled).

        Fail closed: no provider, a falsy value, or a provider error all
        report 0 — structural swaps then stay behind the conservative
        per-channel capex gate.
        """
        provider = self.structural_envelope_provider
        if provider is None:
            return 0
        try:
            return max(0, int(provider() or 0))
        except Exception:
            return 0

    def check_channel_capex_budget(self, estimated_fee_sats: int, channel_id=None) -> dict:
        """Gate a channel-targeted swap on the channel's remaining capex budget.

        Conservative by design: if the capex engine has no allocation for the
        channel (unknown channel, or allocations never computed), the remaining
        budget is 0 and the swap is rejected. Without an engine, or for pure
        treasury swaps (channel_id=None), the gate is not applied.

        Returns:
            {"allowed": bool, "reason": str or None, ...}
        """
        if not self._capex_engine or channel_id is None:
            return {"allowed": True, "reason": None}

        scid = str(channel_id).replace(':', 'x')
        try:
            budget = self._capex_engine.get_channel_budget(scid)
            remaining = max(0, int(getattr(budget, "budget_sats", 0) or 0))
            tier = str(getattr(budget, "tier", "unknown"))
        except Exception as e:
            # Fail closed: a swap spends real fees; do not spend against an
            # unreadable budget.
            try:
                self.plugin.log(
                    f"BOLTZ: channel capex budget lookup failed for {scid}: {e}; rejecting swap",
                    level="warn",
                )
            except Exception:
                pass
            return {
                "allowed": False,
                "reason": f"Channel capex budget lookup failed for {scid}: {e}",
                "channel_id": scid,
            }

        fee = max(0, int(estimated_fee_sats or 0))
        if fee > remaining:
            reason = (
                f"Channel capex budget: estimated fee {fee} sats exceeds remaining "
                f"channel budget {remaining} sats for {scid} (tier={tier})"
            )
            try:
                self.plugin.log(f"BOLTZ: {reason}", level="info")
            except Exception:
                pass
            return {
                "allowed": False,
                "reason": reason,
                "channel_id": scid,
                "remaining_budget_sats": remaining,
                "tier": tier,
            }
        try:
            self.plugin.log(
                f"BOLTZ: channel capex gate passed for {scid}: fee {fee} sats <= "
                f"remaining budget {remaining} sats (tier={tier})",
                level="debug",
            )
        except Exception:
            pass
        return {
            "allowed": True,
            "reason": None,
            "channel_id": scid,
            "remaining_budget_sats": remaining,
            "tier": tier,
        }

    def compute_cost_attribution(self, cost_sats: int, channel_id=None) -> dict:
        """Compute cost attribution for a Boltz swap.

        Uses the capex engine's attribute_boltz_cost() for the split.
        Without an engine, all cost is attributed to tactical (safe default).

        NOTE (2026-06 audit): the 50/50 channel/tactical split returned by
        attribute_boltz_cost() is INFORMATIONAL ONLY — it is dead-lettered.
        record_boltz_spend() writes the FULL fee once into the unified
        "boltz" spend category; the split is only persisted as journal
        metadata (capex_attribution) and never depletes either budget
        partially. Do not treat the split as budget accounting.

        Returns:
            {"channel": amount_sats, "tactical": amount_sats}
        """
        if self._capex_engine:
            return self._capex_engine.attribute_boltz_cost(cost_sats, channel_id=channel_id)
        # No engine: attribute everything to tactical
        return {"channel": 0, "tactical": cost_sats}

    @property
    def enabled(self) -> bool:
        """Compatibility property for callers that check boltz_manager.enabled."""
        return bool(getattr(self.cfg, "enabled", False))

    # ---------------------------------------------------------------------
    # Core command execution helpers
    # ---------------------------------------------------------------------
    def _ensure_enabled(self) -> None:
        if not self.cfg.enabled:
            raise BoltzCliError("Boltz CLI integration disabled (set revenue-ops-boltz-enabled=true)")

    def _base_cmd(self) -> List[str]:
        cmd: List[str] = []
        if self.cfg.use_sudo:
            cmd.extend(["sudo", "-n", "-u", self.cfg.sudo_user])
        cmd.extend([self.cfg.cli_path, "--datadir", self.cfg.datadir])
        return cmd

    def _run(self, args: List[str], timeout: Optional[int] = None) -> str:
        self._ensure_enabled()
        cmd = self._base_cmd() + args
        timeout = timeout or self.cfg.timeout_seconds
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as e:
            raise BoltzCliError(f"boltzcli executable not found: {e}")
        except subprocess.TimeoutExpired:
            raise BoltzCliError(f"boltzcli timed out after {timeout}s: {' '.join(shlex.quote(x) for x in cmd)}")

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        if proc.returncode != 0:
            msg = stderr or stdout or f"boltzcli exited with code {proc.returncode}"
            raise BoltzCliError(msg)
        return stdout

    def _run_json(self, args: List[str], timeout: Optional[int] = None) -> Any:
        out = self._run(args, timeout=timeout)
        try:
            return json.loads(out)
        except json.JSONDecodeError as e:
            raise BoltzCliError(f"Invalid JSON from boltzcli: {e}: {out[:300]}")

    def _detect_reverse_chanids_support(self) -> Optional[bool]:
        """Best-effort capability probe for reverse-swap --chan-id support.

        CLN backends on some boltzd versions reject chanIds for reverse swaps.
        Cache the result so we avoid one failed attempt on every loop-out.
        """
        if self._reverse_chanids_supported is not None:
            return self._reverse_chanids_supported
        try:
            info = self._run_json(["getinfo"], timeout=max(self.cfg.timeout_seconds, 30))
        except Exception:
            return self._reverse_chanids_supported
        node_hint = ""
        if isinstance(info, dict):
            for k in ("node", "Node", "lightningNode"):
                v = info.get(k)
                if isinstance(v, str) and v.strip():
                    node_hint = v.strip()
                    break
            if not node_hint:
                node_hint = json.dumps(info, sort_keys=True)
        else:
            node_hint = str(info)
        if "cln" in node_hint.lower():
            self._reverse_chanids_supported = False
            try:
                self.plugin.log("BOLTZ: reverse-swap chanIds disabled for CLN backend", level="info")
            except Exception:
                pass
        return self._reverse_chanids_supported

    # ---------------------------------------------------------------------
    # Parsing helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _parse_int(v: Any, default: int = 0) -> int:
        if v is None:
            return default
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, float)):
            try:
                return int(v)
            except Exception:
                return default
        s = str(v).strip()
        if not s:
            return default
        try:
            return int(s)
        except Exception:
            try:
                return int(float(s))
            except Exception:
                return default

    @classmethod
    def _parse_timestamp(cls, v: Any) -> Optional[int]:
        # boltz sometimes emits weird zero/default timestamps (e.g. -62135596800)
        ts = cls._parse_int(v, default=0)
        if ts <= 0:
            return None
        # heuristic sanity guard: 2001-01-01 onwards
        if ts < 978307200:
            return None
        return ts

    @staticmethod
    def _norm_currency(currency: Optional[str], default: str) -> str:
        c = (currency or default).strip().upper()
        if c in ("L-BTC", "LBTC"):
            return "LBTC"
        if c in ("BTC",):
            return "BTC"
        return c

    @staticmethod
    def _swap_cli_currency(currency: Optional[str], default: str) -> str:
        return BoltzCliManager._norm_currency(currency, default).lower()

    def _wallet_list(self) -> Dict[str, Any]:
        return self._run_json(["wallet", "list", "--json"])

    def _resolve_wallet(self, currency: str, explicit_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        currency = self._norm_currency(currency, currency)
        wallets_obj = self._wallet_list()
        wallets = wallets_obj.get("wallets", []) if isinstance(wallets_obj, dict) else []

        if explicit_name:
            for w in wallets:
                if str(w.get("name")) == explicit_name:
                    return w
            return None

        preferred_name = self.cfg.btc_wallet if currency == "BTC" else self.cfg.lbtc_wallet if currency == "LBTC" else None
        if preferred_name:
            for w in wallets:
                if str(w.get("name")) == preferred_name and str(w.get("currency", "")).upper() == currency:
                    return w

        for w in wallets:
            if str(w.get("currency", "")).upper() == currency and not bool(w.get("readonly", False)):
                return w
        return None

    def _resolve_wallet_name(self, currency: str, explicit_name: Optional[str] = None) -> str:
        w = self._resolve_wallet(currency, explicit_name)
        if not w:
            raise BoltzCliError(f"No writable {currency} wallet found in boltzd")
        return str(w.get("name"))

    def _resolve_peer_channel_ids(self, peer_id: str) -> List[str]:
        try:
            result = self.data_service.get_peer_channels(peer_id=peer_id) if self.data_service else self.rpc.listpeerchannels(peer_id)
        except Exception as e:
            self.plugin.log(f"BOLTZ: listpeerchannels failed for peer {peer_id[:12]}...: {e}", level='warn')
            return []

        chs = result.get('channels', []) if isinstance(result, dict) else []
        scids: List[str] = []
        for ch in chs:
            if str(ch.get('state', '')) != 'CHANNELD_NORMAL':
                continue
            scid = (ch.get('short_channel_id') or '').replace(':', 'x')
            if scid:
                scids.append(scid)
        return scids

    def _resolve_first_hop_target(self, *, channel_id: Optional[str] = None, peer_id: Optional[str] = None) -> Tuple[Optional[str], Optional[str], List[str]]:
        """Resolve a preferred first-hop peer/channel from local channels.

        Returns (peer_id, channel_id, warnings). The channel is best-effort exact only if we have a
        unique channel to the preferred peer; CLN `pay` excludes are used to force the first hop by peer.
        """
        target_scid = (str(channel_id).replace(':', 'x') if channel_id else None)
        warnings: List[str] = []
        try:
            result = self.data_service.get_peer_channels() if self.data_service else self.rpc.call("listpeerchannels")
        except Exception as e:
            raise BoltzCliError(f"listpeerchannels failed while resolving first-hop target: {e}")
        channels = result.get("channels", []) if isinstance(result, dict) else []

        resolved_peer = str(peer_id).strip() if peer_id else None
        resolved_scid = target_scid
        if target_scid:
            match = None
            for ch in channels:
                scid = str(ch.get("short_channel_id") or "").replace(':', 'x')
                if scid != target_scid:
                    continue
                if str(ch.get("state") or "") != "CHANNELD_NORMAL":
                    continue
                match = ch
                break
            if not match:
                raise BoltzCliError(f"Preferred first-hop channel not found/normal: {target_scid}")
            ch_peer = str(match.get("peer_id") or "").strip()
            if resolved_peer and ch_peer and ch_peer != resolved_peer:
                raise BoltzCliError(f"channel_id {target_scid} belongs to {ch_peer[:12]}..., not requested peer {resolved_peer[:12]}...")
            resolved_peer = ch_peer or resolved_peer
            resolved_scid = target_scid

        if not resolved_peer:
            raise BoltzCliError("external routed reverse swap requires channel_id or peer_id")

        same_peer_normals = []
        for ch in channels:
            if str(ch.get("state") or "") != "CHANNELD_NORMAL":
                continue
            if str(ch.get("peer_id") or "") != resolved_peer:
                continue
            scid = str(ch.get("short_channel_id") or "").replace(':', 'x')
            if scid:
                same_peer_normals.append(scid)
        if resolved_scid and len(same_peer_normals) > 1:
            warnings.append("multiple normal channels to preferred peer; first hop pinned to peer, exact channel best-effort")
        elif not resolved_scid and len(same_peer_normals) == 1:
            resolved_scid = same_peer_normals[0]
        return resolved_peer, resolved_scid, warnings

    def _build_first_hop_excludes(self, preferred_peer_id: str, preferred_channel_id: Optional[str] = None) -> Tuple[List[str], List[str]]:
        """Build CLN `pay` exclude list to force the first hop via preferred peer.

        Excludes direct channels (SCID/direction) to all non-preferred peers.
        Using channel-level excludes (not node-level) so these peers can still
        appear as intermediate hops on multi-hop routes to the destination.
        """
        warnings: List[str] = []
        try:
            result = self.data_service.get_peer_channels() if self.data_service else self.rpc.call("listpeerchannels")
        except Exception as e:
            raise BoltzCliError(f"listpeerchannels failed while building excludes: {e}")
        channels = result.get("channels", []) if isinstance(result, dict) else []

        exclude: List[str] = []
        preferred_scid = (str(preferred_channel_id).replace(':', 'x') if preferred_channel_id else None)
        alt_same_peer: List[str] = []

        for ch in channels:
            if str(ch.get("state") or "") != "CHANNELD_NORMAL":
                continue
            peer = str(ch.get("peer_id") or "").strip()
            if not peer:
                continue
            scid = str(ch.get("short_channel_id") or "").replace(':', 'x')
            if not scid:
                continue
            if peer == preferred_peer_id:
                if preferred_scid and scid != preferred_scid:
                    alt_same_peer.append(scid)
                continue
            # Exclude the direct channel, not the node ��� the node can still
            # appear as an intermediate hop on routes to the destination.
            exclude.append(f"{scid}/0")
            exclude.append(f"{scid}/1")

        if alt_same_peer:
            for scid in alt_same_peer:
                exclude.append(f"{scid}/0")
                exclude.append(f"{scid}/1")
            warnings.append(f"excluded {len(alt_same_peer)} alternate channel(s) to preferred peer")

        return exclude, warnings

    def _extract_reverse_swap_invoice(self, payload: Any) -> Optional[str]:
        """Best-effort extract the LN invoice that must be paid for an external reverse swap."""
        def _scan(obj: Any, depth: int = 0) -> Optional[str]:
            if depth > 4:
                return None
            if isinstance(obj, dict):
                for key in ("invoice", "swapInvoice", "payInvoice", "invoiceToPay"):
                    v = obj.get(key)
                    if isinstance(v, str) and v.strip().lower().startswith("ln"):
                        return v.strip()
                for v in obj.values():
                    found = _scan(v, depth + 1)
                    if found:
                        return found
            elif isinstance(obj, list):
                for v in obj:
                    found = _scan(v, depth + 1)
                    if found:
                        return found
            return None
        return _scan(payload)

    def _decodepay_payee_pubkey(self, decode: Any) -> Optional[str]:
        """Extract payee pubkey from CLN decodepay output (best effort)."""
        def _scan(obj: Any, depth: int = 0) -> Optional[str]:
            if depth > 4:
                return None
            if isinstance(obj, dict):
                for key in ("payee", "payee_id", "destination", "nodeid", "payeeNodeKey"):
                    v = obj.get(key)
                    if isinstance(v, str):
                        s = v.strip().lower()
                        if len(s) == 66 and all(c in '0123456789abcdef' for c in s):
                            return s
                for v in obj.values():
                    found = _scan(v, depth + 1)
                    if found:
                        return found
            elif isinstance(obj, list):
                for v in obj:
                    found = _scan(v, depth + 1)
                    if found:
                        return found
            return None
        return _scan(decode)

    def _lookup_pays_for_invoice(self, invoice: str) -> Dict[str, Any]:
        """Best-effort CLN pay lookup for a bolt11 invoice after timeouts/errors."""
        out: Dict[str, Any] = {"available": False, "matches": []}
        try:
            res = self.data_service.list_pays(bolt11=invoice) if self.data_service else self.rpc.call("listpays", {"bolt11": invoice})
            pays = res.get("pays", []) if isinstance(res, dict) else []
            out = {"available": True, "source": "listpays", "matches": pays}
            if pays:
                return out
        except Exception as e:
            out = {"available": False, "error": str(e), "source": "listpays"}
        try:
            res2 = self.data_service.list_pays() if self.data_service else self.rpc.call("listpays")
            pays2 = res2.get("pays", []) if isinstance(res2, dict) else []
            matches = [p for p in pays2 if isinstance(p, dict) and str(p.get("bolt11") or "") == str(invoice)]
            return {"available": True, "source": "listpays_scan", "matches": matches}
        except Exception as e2:
            out["scan_error"] = str(e2)
            return out

    @staticmethod
    def _decodepay_amount_msat(decode: Any) -> Optional[int]:
        """Extract the invoice amount in msat from CLN decode output."""
        if not isinstance(decode, dict):
            return None
        for key in ("amount_msat", "msatoshi", "amount"):
            value = decode.get(key)
            if value is None:
                continue
            try:
                amount = parse_msat(value)
            except Exception:
                continue
            if amount > 0:
                return amount
        return None

    def _pay_invoice_via_first_hop(self, invoice: str, *, preferred_peer_id: str, preferred_channel_id: Optional[str] = None,
                                   retry_for: int = 120,
                                   expected_amount_sats: Optional[int] = None) -> Dict[str, Any]:
        if not invoice or not str(invoice).lower().startswith("ln"):
            raise BoltzCliError("Invalid bolt11 invoice for external reverse swap payment")
        try:
            decode = self.data_service.decode(invoice) if self.data_service else self.rpc.call("decode", {"string": invoice})
        except Exception as e:
            raise BoltzCliError(f"decode failed for external reverse swap invoice: {e}")

        # The invoice is produced by boltzd / the Boltz API — an external
        # service. Never pay a principal we did not ask for.
        if expected_amount_sats is not None:
            invoice_msat = self._decodepay_amount_msat(decode)
            expected_msat = int(expected_amount_sats) * 1000
            if invoice_msat is None:
                raise BoltzCliError(
                    "external reverse swap invoice has no decodable amount; "
                    f"refusing to pay (expected {expected_amount_sats} sats)"
                )
            if invoice_msat > expected_msat:
                raise BoltzCliError(
                    "external reverse swap invoice amount exceeds the requested "
                    f"swap amount: invoice {invoice_msat} msat > expected {expected_msat} msat"
                )

        exclude, warnings = self._build_first_hop_excludes(preferred_peer_id, preferred_channel_id)
        payee_pubkey = self._decodepay_payee_pubkey(decode)
        if payee_pubkey:
            # Never node-exclude the destination: the exclude list must only
            # contain channel-scoped "scid/dir" entries, but guard against a
            # payee pubkey slipping in so the route can still terminate there.
            if payee_pubkey in exclude:
                exclude = [e for e in exclude if e != payee_pubkey]
                warnings.append("removed payee pubkey from exclude set to permit route termination")
            if payee_pubkey != preferred_peer_id:
                # If the payee is a direct peer, exclude only our direct
                # edge(s) to it (channel-level), so the payment is forced
                # through the preferred first hop but can still terminate at
                # the payee via a multi-hop route.
                try:
                    direct_scids = self._resolve_peer_channel_ids(payee_pubkey)
                except Exception:
                    direct_scids = []
                if direct_scids:
                    for scid in direct_scids:
                        for direction in ("0", "1"):
                            entry = f"{scid}/{direction}"
                            if entry not in exclude:
                                exclude.append(entry)
                    warnings.append(f"payee is a direct peer; excluded {len(direct_scids)} direct payee channel(s) instead of payee node")
        pay_params: Dict[str, Any] = {"bolt11": invoice}
        if exclude:
            pay_params["exclude"] = exclude
        if retry_for is not None:
            try:
                pay_params["retry_for"] = max(1, int(retry_for))
            except Exception:
                pass

        try:
            pay_result = self.data_service.pay(**pay_params) if self.data_service else self.rpc.call("pay", pay_params)
            status = "submitted"
            pay_error = None
            pay_lookup = None
        except Exception as e:
            pay_result = None
            pay_error = str(e)
            pay_lookup = self._lookup_pays_for_invoice(invoice)
            # CLN pay can outlive the plugin RPC timeout; preserve context instead of failing hard.
            if "timeout" in pay_error.lower():
                status = "timeout"
                warnings.append("CLN pay RPC timed out; payment may still be in progress or completed")
            else:
                status = "error"
        return {
            "status": status,
            "preferred_peer_id": preferred_peer_id,
            "preferred_channel_id": preferred_channel_id,
            "exclude_count": len(exclude),
            "exclude": exclude,
            "warnings": warnings,
            "decodepay": decode,
            "pay_params": pay_params,
            "pay_result": pay_result,
            "pay_error": pay_error,
            "pay_lookup": pay_lookup,
        }

    def _normalize_swap_entry(self, entry: Dict[str, Any], *, wrapper_key: Optional[str] = None) -> Dict[str, Any]:
        """Flatten nested swap entries from boltzcli listswaps/swapinfo outputs."""
        if not isinstance(entry, dict):
            return {}
        out = dict(entry)
        if wrapper_key and wrapper_key not in out:
            out["_swap_wrapper"] = wrapper_key
        # Normalize common nested shapes emitted by boltzcli (swap/reverseSwap/chainSwap/channelCreation)
        for k in ("swap", "reverseSwap", "chainSwap", "channelCreation"):
            nested = out.get(k)
            if isinstance(nested, dict) and nested.get("id"):
                merged = dict(out)
                merged.pop(k, None)
                # Preserve wrapper metadata while letting nested fields take precedence.
                merged.update(nested)
                merged["_swap_wrapper"] = k
                return merged
        return out

    def _extract_swap_list(self, swaps_json: Any) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if isinstance(swaps_json, dict):
            # Common flat list keys.
            for key in ('swaps', 'list', 'allSwaps'):
                val = swaps_json.get(key)
                if isinstance(val, list):
                    items.extend(self._normalize_swap_entry(s) for s in val if isinstance(s, dict))
            # Some boltzcli versions split by type.
            for key in ('reverseSwaps', 'submarineSwaps', 'chainSwaps', 'channelCreations'):
                val = swaps_json.get(key)
                if isinstance(val, list):
                    items.extend(self._normalize_swap_entry(s, wrapper_key=key) for s in val if isinstance(s, dict))
            # Some outputs contain one nested object (e.g. swapinfo-like payload).
            for key in ('swap', 'reverseSwap', 'chainSwap', 'channelCreation'):
                val = swaps_json.get(key)
                if isinstance(val, dict) and val.get('id'):
                    items.append(self._normalize_swap_entry({key: val}, wrapper_key=key))
        elif isinstance(swaps_json, list):
            items.extend(self._normalize_swap_entry(s) for s in swaps_json if isinstance(s, dict))

        # Deduplicate by id while preserving first-seen ordering.
        dedup: List[Dict[str, Any]] = []
        seen = set()
        for it in items:
            if not isinstance(it, dict) or not it:
                continue
            sid = str(it.get('id') or '')
            key = sid or json.dumps(it, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(it)
        return dedup

    def _listswaps_json(self, *, manual_only: bool = False, pending_only: bool = False) -> Any:
        """Best-effort listswaps wrapper with flag compatibility fallbacks.

        Some boltzcli versions do not support filters like --manual / --pending.
        In that case we fall back to unfiltered listswaps and filter client-side.
        """
        args = ["listswaps", "--json"]
        if manual_only:
            args.append("--manual")
        if pending_only:
            args.append("--pending")
        try:
            return self._run_json(args)
        except BoltzCliError:
            if not (manual_only or pending_only):
                raise

        data = self._listswaps_json()
        swaps = self._extract_swap_list(data)
        if manual_only:
            swaps = [s for s in swaps if not bool(s.get("isAuto", False))]
        if pending_only:
            swaps = [s for s in swaps if not self._is_completed_swap(s)]
        return {"swaps": swaps, "_compat_fallback": True}

    def _estimate_swap_fee_sats(self, swap: Dict[str, Any]) -> int:
        # Prefer explicit top-level fields where available.
        total = 0
        seen_named = False
        for key in ("boltzFee", "networkFee", "serviceFee", "routingFee", "onchainFee"):
            if key in swap:
                total += max(0, self._parse_int(swap.get(key), 0))
                seen_named = True
        if seen_named:
            return total

        # Fallback: recursively sum fee-like integer fields (excluding
        # percentages/rates). E-4.4 (2026-07 econ audit): guarded against
        # double-counting — when a dict level carries NUMERIC fee-like
        # fields, that level is authoritative and its subtree is NOT also
        # descended (payloads with a parent total plus a nested per-component
        # breakdown used to sum BOTH, inflating spend accounting and
        # over-throttling the budget gate).
        def rec(obj: Any) -> int:
            if isinstance(obj, dict):
                level_total = 0
                for k, v in obj.items():
                    lk = str(k).lower()
                    if (
                        "fee" in lk
                        and not any(x in lk for x in ("percent", "percentage", "ppm", "rate"))
                        and isinstance(v, (int, float, str))
                    ):
                        level_total += max(0, self._parse_int(v, 0))
                if level_total > 0:
                    return level_total
                return sum(
                    rec(v) for v in obj.values() if isinstance(v, (dict, list))
                )
            if isinstance(obj, list):
                return sum(rec(v) for v in obj)
            return 0

        return rec(swap)

    def _estimate_reverse_routing_fee_sats(
        self, amount_sats: int, routing_fee_limit_ppm: Optional[int] = None
    ) -> int:
        """Plan-time LN routing-fee estimate for a reverse swap (E-4.4).

        A reverse swap PAYS a lightning invoice to Boltz; the quote covers
        Boltz service + on-chain fees but NOT our routing cost, which is
        bounded by routing_fee_limit_ppm. Omitting it understated every
        loop-out's planned cost. 0/unset limit -> 0 (boltzcli then imposes
        no explicit routing cap; there is no honest bound to plan with).
        """
        limit_ppm = routing_fee_limit_ppm
        if limit_ppm is None:
            limit_ppm = getattr(self.cfg, "routing_fee_limit_ppm", 0)
        limit_ppm = max(0, self._parse_int(limit_ppm, 0))
        if limit_ppm <= 0 or amount_sats <= 0:
            return 0
        return (int(amount_sats) * limit_ppm + 999_999) // 1_000_000

    def _swap_created_ts(self, swap: Dict[str, Any]) -> Optional[int]:
        for key in ("createdAt", "updatedAt", "created_at", "updated_at"):
            ts = self._parse_timestamp(swap.get(key))
            if ts:
                return ts
        return None

    def _swap_completed_ts(self, swap: Dict[str, Any]) -> Optional[int]:
        """Best-effort completion timestamp for budget windowing.

        Prefer update timestamps for completed swaps, then fall back to creation time.
        """
        for key in ("updatedAt", "updated_at", "createdAt", "created_at"):
            ts = self._parse_timestamp(swap.get(key))
            if ts:
                return ts
        return None

    def _swap_status_text(self, swap: Dict[str, Any]) -> str:
        return str(swap.get('state') or swap.get('status') or '').lower()

    def _is_completed_swap(self, swap: Dict[str, Any]) -> bool:
        st = self._swap_status_text(swap)
        if "abandoned" in st:
            # "abandoned" contains "done" — an abandoned swap is terminal
            # but NOT completed (its estimated fee must not count as spend).
            return False
        return any(token in st for token in ("success", "completed", "claimed", "done"))

    def _get_external_liquidity_costs(self) -> Dict[str, Any]:
        provider = getattr(self, "external_liquidity_cost_provider", None)
        if not callable(provider):
            return {
                "source": "none",
                "spent_24h_sats": 0,
                "reserved_24h_sats": 0,
            }
        try:
            data = provider()
        except Exception as e:
            self.plugin.log(f"BOLTZ: external liquidity cost provider failed: {e}", level="warn")
            return {
                "source": "provider_error",
                "error": str(e),
                "spent_24h_sats": 0,
                "reserved_24h_sats": 0,
            }
        if not isinstance(data, dict):
            return {
                "source": "provider_invalid",
                "spent_24h_sats": 0,
                "reserved_24h_sats": 0,
            }
        return {
            "source": str(data.get("source") or "external"),
            "spent_24h_sats": max(0, self._parse_int(data.get("spent_24h_sats"), 0)),
            "reserved_24h_sats": max(0, self._parse_int(data.get("reserved_24h_sats"), 0)),
            **{k: v for k, v in data.items() if k not in ("source", "spent_24h_sats", "reserved_24h_sats")},
        }

    def _get_global_budget_limit(self) -> Dict[str, Any]:
        """Return the unified liquidity budget.

        When the global provider is set (normal operation), Boltz draws from
        the same universal budget as the rebalancer and capacity planner.
        The boltz_daily_budget_sats config is only used as a last-resort
        fallback if the unified provider is not registered.
        """
        provider = getattr(self, "global_budget_limit_provider", None)
        if not callable(provider):
            return {"budget_sats": max(0, int(self.cfg.daily_budget_sats)), "source": "boltz_cfg"}
        try:
            data = provider()
            if isinstance(data, dict):
                if "effective_budget_sats" in data:
                    return {
                        "budget_sats": max(0, self._parse_int(data.get("effective_budget_sats"), 0)),
                        "source": str(data.get("source") or "unified"),
                        **{k: v for k, v in data.items() if k != "effective_budget_sats"},
                    }
                if "budget_sats" in data:
                    return {
                        "budget_sats": max(0, self._parse_int(data.get("budget_sats"), 0)),
                        "source": str(data.get("source") or "unified"),
                        **{k: v for k, v in data.items() if k != "budget_sats"},
                    }
            if isinstance(data, (int, float, str)):
                return {"budget_sats": max(0, self._parse_int(data, 0)), "source": "unified_scalar"}
        except Exception as e:
            self.plugin.log(f"BOLTZ: unified budget provider failed: {e}", level="warn")
        return {"budget_sats": max(0, int(self.cfg.daily_budget_sats)), "source": "boltz_cfg_fallback"}

    def get_boltz_cost_components(
        self,
        window_hours: int = 24,
        global_budget_cap_sats: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Boltz-only spend component summary (no external costs, no unified budget math).

        IMPORTANT: this method must never call self.global_budget_limit_provider.
        The unified budget status itself aggregates Boltz costs through this
        method, so calling the provider here mutually recurses (each level
        spawning boltzcli subprocesses) until RecursionError. Callers that
        want the reserved estimate capped at the unified budget pass the cap
        explicitly via global_budget_cap_sats instead.
        """
        swaps_json = self._listswaps_json(manual_only=True)
        swaps = self._extract_swap_list(swaps_json)
        swaps = self._augment_with_swap_journal(swaps, limit_hint=50)
        now = int(time.time())
        window_hours = max(1, min(168, int(window_hours or 24)))
        cutoff = now - (window_hours * 3600)

        boltz_spent = 0
        counted: List[Dict[str, Any]] = []
        unknown_ts = 0
        for s in swaps:
            ts = self._swap_completed_ts(s)
            if ts is None:
                unknown_ts += 1
                continue
            if ts < cutoff:
                continue
            if not self._is_completed_swap(s):
                continue
            fee_sats = self._estimate_swap_fee_sats(s)
            boltz_spent += max(0, fee_sats)
            counted.append({
                "id": s.get("id"),
                "created_at": self._swap_created_ts(s),
                "counted_timestamp": ts,
                "fee_sats_estimate": fee_sats,
                "state": s.get("state"),
                "status": s.get("status"),
            })
        # C2 FIX: Count pending (non-terminal) swaps as reserved budget
        reserved = 0
        reserved_count = 0
        for s in swaps:
            if self._is_terminal_swap(s):
                continue
            ts = self._swap_created_ts(s)
            if ts is None or ts < cutoff:
                continue
            fee_est = self._estimate_swap_fee_sats(s)
            if fee_est > 0:
                reserved += fee_est
                reserved_count += 1

        # Cap reserved at remaining budget to prevent over-estimation
        # from blocking the unified capital control (fee estimate fallback can overcount).
        # Use the tighter of Boltz-specific and global (unified) budget.
        boltz_budget = max(0, int(getattr(self.cfg, "daily_budget_sats", 0) or 0))
        cap_budget = boltz_budget
        if global_budget_cap_sats is not None:
            global_budget = max(0, self._parse_int(global_budget_cap_sats, 0))
            if global_budget > 0:
                cap_budget = min(cap_budget, global_budget) if cap_budget > 0 else global_budget
        if cap_budget > 0:
            max_reservable = max(0, cap_budget - boltz_spent)
            reserved = min(reserved, max_reservable)

        return {
            "spent_24h_sats": boltz_spent,
            "reserved_24h_sats": reserved,
            "reserved_swaps": reserved_count,
            "counted_swaps": len(counted),
            "skipped_without_timestamp": unknown_ts,
            "counted_details": counted[:20],
            "window_seconds": window_hours * 3600,
            "source": "boltz",
        }

    def _swap_entry_error_text(self, swap: Optional[Dict[str, Any]]) -> str:
        if not isinstance(swap, dict):
            return ""
        return str(swap.get("error") or "").strip()

    def _is_error_swap(self, swap: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(swap, dict):
            return False
        if self._swap_entry_error_text(swap):
            return True
        st = self._swap_status_text(swap)
        return any(token in st for token in ("error", "failed"))

    def _is_terminal_swap(self, swap: Optional[Dict[str, Any]]) -> bool:
        """Check if swap is in any terminal state (completed, error, refunded, abandoned)."""
        if not isinstance(swap, dict):
            return False
        if self._is_completed_swap(swap) or self._is_error_swap(swap):
            return True
        st = self._swap_status_text(swap)
        return any(token in st for token in ("refund", "abandon"))

    def _contains_chanids_cln_error(self, payload: Any) -> bool:
        needle = "chanids are not supported for cln"
        if isinstance(payload, dict):
            if needle in str(payload.get("error") or "").lower():
                return True
            for s in self._extract_swap_list(payload):
                if needle in self._swap_entry_error_text(s).lower():
                    return True
        return False

    def _primary_swap_entry(self, payload: Any) -> Optional[Dict[str, Any]]:
        if isinstance(payload, dict):
            extracted = self._extract_swap_list(payload)
            if extracted:
                return extracted[0]
            if payload.get("id"):
                return self._normalize_swap_entry(payload)
        return None

    def _load_swap_journal(self) -> List[Dict[str, Any]]:
        try:
            with open(self._swap_journal_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
        except Exception:
            return []
        return []

    # DD8/RES-3: keep-last-180-days retention window plus a generous keep-last-N
    # count cap. Completed-swap records are historical only, so pruning entries
    # older than the window on the next write bounds on-disk growth without any
    # behavioral effect.
    _SWAP_JOURNAL_MAX_AGE_SECONDS = 180 * 24 * 60 * 60
    _SWAP_JOURNAL_MAX_ENTRIES = 200

    def _prune_swap_journal_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop entries older than the retention window; cap the count.

        Age is measured by the newest available timestamp on the record
        (recorded_at or created_at). Entries with no usable timestamp are kept
        (age is unknown) but are still subject to the keep-last-N count cap.
        """
        cutoff = int(time.time()) - self._SWAP_JOURNAL_MAX_AGE_SECONDS
        kept: List[Dict[str, Any]] = []
        for rec in entries:
            if not isinstance(rec, dict):
                continue
            newest = max(
                self._parse_int(rec.get("recorded_at"), 0),
                self._parse_int(rec.get("created_at"), 0),
            )
            if newest and newest < cutoff:
                continue  # older than the retention window
            kept.append(rec)
        return kept[-self._SWAP_JOURNAL_MAX_ENTRIES:]

    def _save_swap_journal(self, entries: List[Dict[str, Any]]) -> None:
        try:
            os.makedirs(os.path.dirname(self._swap_journal_file), exist_ok=True)
            tmp = self._swap_journal_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._prune_swap_journal_entries(entries), f, sort_keys=True)
            os.replace(tmp, self._swap_journal_file)
        except Exception as e:
            self.plugin.log(f"BOLTZ: failed to write swap journal: {e}", level="warn")

    def _load_ignored_external_swaps(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self._ignored_external_swaps_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                out = {}
                for k, v in data.items():
                    if isinstance(v, dict):
                        out[str(k)] = dict(v)
                return out
            if isinstance(data, list):
                out = {}
                for rec in data:
                    if not isinstance(rec, dict):
                        continue
                    sid = str(rec.get("id") or "").strip()
                    if sid:
                        out[sid] = dict(rec)
                return out
        except Exception:
            return {}
        return {}

    def _save_ignored_external_swaps(self, entries: Dict[str, Dict[str, Any]]) -> None:
        try:
            os.makedirs(os.path.dirname(self._ignored_external_swaps_file), exist_ok=True)
            tmp = self._ignored_external_swaps_file + ".tmp"
            serial = dict(sorted(((str(k), v) for k, v in entries.items() if isinstance(v, dict)), key=lambda kv: str(kv[0])))
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(serial, f, sort_keys=True)
            os.replace(tmp, self._ignored_external_swaps_file)
        except Exception as e:
            self.plugin.log(f"BOLTZ: failed to write ignored external swaps file: {e}", level="warn")

    @staticmethod
    def _annotate_ignored_swap(entry: Optional[Dict[str, Any]], ignored: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(entry, dict):
            return entry
        sid = str(entry.get("id") or "").strip()
        if not sid:
            return entry
        rec = ignored.get(sid)
        if not rec:
            return entry
        out = dict(entry)
        out["ignored_external_swap"] = True
        out["ignored_external_swap_meta"] = dict(rec)
        return out

    def _load_swap_journal_index(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for rec in self._load_swap_journal():
            if not isinstance(rec, dict):
                continue
            sid = str(rec.get("id") or "").strip()
            if not sid:
                continue
            out[sid] = dict(rec)
        return out

    @staticmethod
    def _annotate_journal_swap(entry: Optional[Dict[str, Any]], journal_index: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(entry, dict):
            return entry
        sid = str(entry.get("id") or "").strip()
        if not sid:
            return entry
        rec = journal_index.get(sid)
        if not rec:
            return entry
        out = dict(entry)
        # Expose a small set of common fields directly and preserve full journal metadata nested.
        for k in (
            "source",
            "peer_id",
            "requested_channel_ids",
            "requested_first_hop_peer_id",
            "requested_first_hop_channel_id",
            "routing_mode",
            "cln_payment_status",
            "cln_payment_error",
            "cln_pay_lookup",
            "cln_pay_summary",
            "recorded_at",
        ):
            if rec.get(k) is not None and out.get(k) is None:
                out[k] = rec.get(k)
        out["journal_meta"] = dict(rec)
        return out

    def manage_external_pay_ignores(self, action: str = "list", swap_id: Optional[str] = None, note: Optional[str] = None) -> Dict[str, Any]:
        # B3 FIX: Serialize load-modify-save to prevent lost updates
        with self._ignored_swaps_lock:
            return self._manage_external_pay_ignores_locked(action, swap_id, note)

    def _manage_external_pay_ignores_locked(self, action: str = "list", swap_id: Optional[str] = None, note: Optional[str] = None) -> Dict[str, Any]:
        act = str(action or "list").strip().lower()
        ignores = self._load_ignored_external_swaps()
        now = int(time.time())
        if act == "list":
            items = []
            for sid, rec in ignores.items():
                item = dict(rec)
                item.setdefault("id", sid)
                try:
                    st = self.swap_status(sid)
                    item["swap_status"] = st.get("swapinfo_entry") or st.get("listswaps_entry")
                except Exception as e:
                    item["status_error"] = str(e)
                items.append(item)
            items.sort(key=lambda x: int(x.get("added_at") or 0), reverse=True)
            return {"status": "success", "action": "list", "ignores": items, "count": len(items)}
        if act == "clear":
            cleared = len(ignores)
            self._save_ignored_external_swaps({})
            return {"status": "success", "action": "clear", "cleared": cleared}
        sid = str(swap_id or "").strip()
        if not sid:
            raise BoltzCliError("swap_id is required for action add/remove")
        if act == "remove":
            existed = sid in ignores
            if existed:
                ignores.pop(sid, None)
                self._save_ignored_external_swaps(ignores)
            return {"status": "success", "action": "remove", "swap_id": sid, "removed": bool(existed)}
        if act != "add":
            raise BoltzCliError("action must be list, add, remove, or clear")
        st = self.swap_status(sid)
        sw = st.get("swapinfo_entry") or st.get("listswaps_entry")
        if not isinstance(sw, dict):
            raise BoltzCliError(f"swap {sid} not found")
        is_external = bool(sw.get("externalPay"))
        wrapper = str(sw.get("_swap_wrapper") or "")
        state = str(sw.get("state") or "")
        status_txt = str(sw.get("status") or "")
        if not is_external:
            raise BoltzCliError("Only external-pay swaps can be ignored")
        if wrapper not in ("reverseSwap", "") and str(sw.get("type") or "").upper() != "REVERSE":
            raise BoltzCliError("Only reverse swaps can be ignored")
        rec = dict(ignores.get(sid) or {})
        rec.update({
            "id": sid,
            "added_at": int(rec.get("added_at") or now),
            "updated_at": now,
            "note": (str(note).strip() if note is not None else rec.get("note")),
            "state_at_add": state,
            "status_at_add": status_txt,
            "externalPay": True,
        })
        created = self._parse_timestamp(sw.get("createdAt"))
        if created:
            rec["created_at"] = created
        ignores[sid] = rec
        self._save_ignored_external_swaps(ignores)
        return {"status": "success", "action": "add", "swap_id": sid, "ignore": rec, "swap": sw}

    # Sources that correspond to actually executing a swap (creation paths).
    # Only these deplete the capex "boltz" spend category; status probes and
    # journal lookups must not double-spend the budget.
    _SPEND_RECORD_SOURCES = frozenset({"loop_in", "loop_out", "loop_out_external_create", "chainswap"})

    def _record_swap_result(self, payload: Any, *, source: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        # B1 FIX: Serialize load-modify-save to prevent lost updates from concurrent threads
        with self._journal_lock:
            self._record_swap_result_locked(payload, source=source, metadata=metadata)

    def _record_swap_result_locked(self, payload: Any, *, source: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        now = int(time.time())
        entries = self._load_swap_journal()
        by_id = {str(e.get("id")): e for e in entries if e.get("id")}
        for s in self._extract_swap_list(payload):
            sid = str(s.get("id") or "").strip()
            if not sid:
                continue
            rec = by_id.get(sid, {"id": sid})
            rec.update({
                "id": sid,
                "recorded_at": now,
                "source": source,
            })
            st = self._swap_created_ts(s)
            if st:
                rec["created_at"] = st
            if metadata:
                rec.update({k: v for k, v in metadata.items() if v is not None})
            # Compute capex cost attribution
            fee = self._estimate_swap_fee_sats(s)
            trigger_ch = (metadata or {}).get("trigger_channel_id")
            if not trigger_ch:
                # loop_out paths report the target channel via requested_channel_ids
                req_chs = (metadata or {}).get("requested_channel_ids")
                if isinstance(req_chs, list) and len(req_chs) == 1 and req_chs[0]:
                    trigger_ch = str(req_chs[0])
            attribution = self.compute_cost_attribution(fee, channel_id=trigger_ch)
            rec["capex_attribution"] = attribution
            by_id[sid] = rec
            # Deplete the capex "boltz" spend category for executed swaps so
            # the tactical budget actually decreases as swap fees are spent.
            if (
                self._capex_engine is not None
                and source in self._SPEND_RECORD_SOURCES
                and fee > 0
                and not self._is_error_swap(s)
            ):
                try:
                    self._capex_engine.record_boltz_spend(
                        swap_id=sid,
                        fee_sats=fee,
                        channel_id=trigger_ch,
                        source=f"boltz_manager:{source}",
                        metadata={"attribution": attribution},
                        # Structural swaps (executed on the inbound-scarcity
                        # credit) are partitioned so the daily structural
                        # envelope can sum them; same boltz:{id} event, so
                        # the unified boltz category never double-counts.
                        subcategory=(
                            "structural"
                            if (metadata or {}).get("structural")
                            else "swap_fee"
                        ),
                    )
                except Exception as e:
                    try:
                        self.plugin.log(
                            f"BOLTZ: failed to record capex spend event for swap {sid}: {e}",
                            level="warn",
                        )
                    except Exception:
                        pass
        if by_id:
            merged = list(by_id.values())
            merged.sort(key=lambda x: int(x.get("recorded_at") or 0))
            self._save_swap_journal(merged)

    def _augment_with_swap_journal(self, swaps: List[Dict[str, Any]], *, limit_hint: Optional[int] = None) -> List[Dict[str, Any]]:
        existing = {str(s.get("id") or "") for s in swaps if isinstance(s, dict)}
        journal = self._load_swap_journal()
        if not journal:
            return swaps
        journal.sort(key=lambda x: int(x.get("recorded_at") or 0), reverse=True)
        max_fetch = max(10, (int(limit_hint) * 3) if limit_hint else 40)
        added = 0
        for rec in journal:
            if added >= max_fetch:
                break
            sid = str(rec.get("id") or "").strip()
            if not sid or sid in existing:
                continue
            try:
                info = self._run_json(["swapinfo", sid], timeout=max(self.cfg.timeout_seconds, 120))
            except Exception:
                continue
            extracted = self._extract_swap_list(info)
            if not extracted:
                continue
            s = extracted[0]
            if not self._swap_created_ts(s):
                ts = self._parse_timestamp(rec.get("created_at") or rec.get("recorded_at"))
                if ts:
                    s["createdAt"] = ts
            swaps.append(s)
            existing.add(sid)
            added += 1
        return swaps

    # ---------------------------------------------------------------------
    # Budget helpers
    # ---------------------------------------------------------------------
    def get_budget_status(self) -> Dict[str, Any]:
        budget_info = self._get_global_budget_limit()
        budget = max(0, self._parse_int(budget_info.get("budget_sats"), self.cfg.daily_budget_sats))
        try:
            local = self.get_boltz_cost_components(window_hours=24, global_budget_cap_sats=budget)
        except TypeError:
            # Test doubles may stub get_boltz_cost_components with the old signature.
            local = self.get_boltz_cost_components(window_hours=24)
        boltz_spent = max(0, self._parse_int(local.get("spent_24h_sats"), 0))
        counted = list(local.get("counted_details", [])) if isinstance(local.get("counted_details"), list) else []
        unknown_ts = max(0, self._parse_int(local.get("skipped_without_timestamp"), 0))

        external = self._get_external_liquidity_costs()
        external_spent = max(0, self._parse_int(external.get("spent_24h_sats"), 0))
        external_reserved = max(0, self._parse_int(external.get("reserved_24h_sats"), 0))
        local_reserved = max(0, self._parse_int(local.get("reserved_24h_sats"), 0))
        total_spent = boltz_spent + external_spent
        total_reserved = local_reserved + external_reserved

        boltz_remaining = max(0, budget - boltz_spent - local_reserved)
        remaining = max(0, budget - total_spent - total_reserved)
        return {
            "daily_budget_sats": budget,
            # Unified liquidity-cost accounting used for gating swaps.
            "spent_24h_sats_estimate": total_spent,
            "remaining_24h_sats_estimate": remaining,
            "reserved_24h_sats_estimate": total_reserved,
            # Component breakdowns preserved for visibility and debugging.
            "boltz_spent_24h_sats_estimate": boltz_spent,
            "boltz_remaining_24h_sats_estimate": boltz_remaining,
            "boltz_reserved_24h_sats_estimate": local_reserved,
            "external_liquidity_costs": external,
            "budget_source": budget_info.get("source"),
            "budget_info": budget_info,
            "counted_swaps": len(counted),
            "skipped_without_timestamp": unknown_ts,
            "enforce_budget": bool(self.cfg.enforce_budget),
            "window_seconds": 86400,
            "counted_details": counted[:20],
        }

    def _enforce_budget_for_quote(self, quote: Dict[str, Any],
                                  extra_fee_sats: int = 0) -> Dict[str, Any]:
        # Use the same fee estimation as spend accounting to prevent asymmetry
        # where the gate underestimates and allows budget overruns.
        # extra_fee_sats (E-4.4): plan-time components NOT in the raw boltzcli
        # quote payload — currently the reverse-swap LN routing estimate.
        fee_sats = self._estimate_swap_fee_sats(quote) + max(0, self._parse_int(extra_fee_sats, 0))
        budget = self.get_budget_status()
        allowed = True
        reason = None
        if self.cfg.enforce_budget and fee_sats > budget.get("remaining_24h_sats_estimate", 0):
            allowed = False
            reason = (
                f"Estimated swap fee {fee_sats} sats exceeds remaining Boltz daily budget "
                f"{budget.get('remaining_24h_sats_estimate', 0)} sats"
            )
        return {
            "allowed": allowed,
            "estimated_fee_sats": fee_sats,
            "budget": budget,
            "reason": reason,
        }

    # ---------------------------------------------------------------------
    # P4-014 / DD1: atomic unified-budget reservation around swap creation
    # ---------------------------------------------------------------------
    # Phase 2G: optional governor/ledger plumbing, injected at plugin
    # init (BoltzCliManager holds its own BoltzCliConfig, so the main
    # config's flag arrives via provider like the other cross-module
    # hooks). None = legacy behavior.
    econ_shadow = None
    econ_governor_enabled_provider = None
    # Phase 4: callable -> bool, True when authority level permits swaps
    # ('capital'). None = ungated (legacy).
    econ_authority_check_provider = None

    def _boltz_governor_enabled(self) -> bool:
        try:
            provider = self.econ_governor_enabled_provider
            return bool(provider is not None and provider() is True)
        except Exception:
            return False

    def _governed_open_reservation(self, *, reservation_id, fee, channel_id,
                                   subcat, effective_budget, intent_type,
                                   structural):
        """Phase 2G: GovernorFacade.authorize() gates the pre-create swap
        reservation. The delegate performs the IDENTICAL
        reserve_boltz_swap_budget call. FAILS CLOSED on internal error
        (returns False = swap rejected) — a deliberate, flag-gated
        strengthening of the legacy infra-error fail-open; flag off
        restores legacy exactly. No pause gate (matching legacy: Boltz
        is gated by its own enable flags)."""
        try:
            import time as _time

            from .econ_intents import Explanation, make_intent
            from .econ_types import Micro, Msat, SignedMsat, UnixTime
            from .governor_facade import GovernorFacade

            now = int(_time.time())

            def _boltz_reserve_delegate(**_facade_kwargs):
                return bool(self._capex_engine.reserve_boltz_swap_budget(
                    reservation_id=reservation_id,
                    estimated_fee_sats=fee,
                    channel_id=channel_id,
                    effective_budget_sats=effective_budget,
                    subcategory=subcat,
                ))

            ledger = None
            registry = None
            if self.econ_shadow is not None:
                try:
                    ledger = self.econ_shadow.ledger_for_reconciliation()
                except Exception:
                    ledger = None
                try:
                    registry = self.econ_shadow.arbitration_registry()
                except Exception:
                    registry = None

            facade = GovernorFacade(
                reserve_spend=_boltz_reserve_delegate,
                release_spend=lambda rid: bool(
                    self._capex_engine.release_boltz_swap_reservation(rid)),
                is_paused=lambda: False,
                ledger=ledger,
                registry=registry,
                authority_check=self.econ_authority_check_provider,
            )
            env = make_intent(
                intent_type=intent_type,
                snapshot_id=f"boltz-swap-{now}",
                created_at=UnixTime(now),
                expires_at=UnixTime(now + 600),
                target=str(channel_id or "onchain"),
                amount_msat=None,
                expected_benefit_msat=SignedMsat(0),
                max_cost_msat=Msat(int(fee) * 1000),
                capital_committed_msat=Msat(0),
                confidence_micro=Micro(0),
                reason_codes=(),
                explanation=Explanation("boltz_swap_reservation", (
                    ("subcategory", str(subcat)),
                    ("structural", bool(structural)),
                    ("estimated_fee_sats", int(fee)),
                )),
                preconditions=(),
                priority=50,
                budget_bucket="boltz",
                origin_policy="boltz_manager_governed",
                reversible=False,
            )
            if ledger is not None:
                try:
                    ledger.append(
                        event_type="intent_proposed",
                        intent_id=env.intent_id.value,
                        idempotency_key=env.idempotency_key,
                        cycle_id=env.snapshot_id,
                        at=now,
                        details={"target": env.target, "governed": True,
                                 "subcategory": str(subcat)},
                    )
                except Exception:
                    pass
            decision = facade.authorize(env, now,
                                        reservation_id=reservation_id)
            if not decision.authorized:
                try:
                    self.plugin.log(
                        f"BOLTZ: governor blocked swap reservation "
                        f"({subcat}): {decision.reason_code}", level="warn")
                except Exception:
                    pass
                return False
            return reservation_id
        except Exception as e:
            try:
                self.plugin.log(
                    f"BOLTZ: governed reservation failed closed: {e}",
                    level="warn")
            except Exception:
                pass
            return False

    def _open_swap_budget_reservation(
        self,
        estimated_fee_sats: int,
        channel_id: Optional[str],
        structural: bool = False,
        intent_type: str = "SWAP_OUT",
    ):
        """Reserve the swap's estimated cost against the unified cross-category
        budget BEFORE the swap is created, so a concurrent rebalance reserve both
        sees and is counted by this boltz commitment (DD1). Returns:
          * a reservation id (str) when a reservation was placed,
          * ``None`` when enforcement does not apply (no capex engine / fee<=0 /
            enforce_budget off / no unified budget) — behaviour unchanged, and
          * ``False`` when the unified budget WOULD be exceeded — the caller must
            reject the swap without creating it.
        """
        if self._capex_engine is None or not self.cfg.enforce_budget:
            return None
        try:
            fee = max(0, int(estimated_fee_sats or 0))
        except (TypeError, ValueError):
            return None
        if fee <= 0:
            return None
        try:
            budget_info = self._get_global_budget_limit()
            effective_budget = max(0, self._parse_int(budget_info.get("budget_sats"), 0))
        except Exception:
            return None
        if effective_budget <= 0:
            # No unified budget wired: leave gating to the (already-applied)
            # advisory checks; fail-open preserves prior behaviour.
            return None
        reservation_id = f"boltz-swap:{uuid.uuid4().hex}"
        subcat = "structural" if structural else "swap_fee"
        if self._boltz_governor_enabled():
            # Phase 2G: governor-gated reservation (same
            # reserve_boltz_swap_budget underneath; fail-closed).
            return self._governed_open_reservation(
                reservation_id=reservation_id, fee=fee,
                channel_id=channel_id, subcat=subcat,
                effective_budget=effective_budget,
                intent_type=intent_type, structural=structural)
        try:
            ok = self._capex_engine.reserve_boltz_swap_budget(
                reservation_id=reservation_id,
                estimated_fee_sats=fee,
                channel_id=channel_id,
                effective_budget_sats=effective_budget,
                subcategory=subcat,
            )
        except Exception as e:
            try:
                self.plugin.log(f"BOLTZ: unified budget reserve error: {e}", level="warn")
            except Exception:
                pass
            return None  # fail-open on infra error (advisory gate still applied)
        return reservation_id if ok else False

    def _finalize_swap_budget_reservation(
        self,
        reservation_id,
        result: Any,
        estimated_fee_sats: int,
        channel_id: Optional[str],
        structural: bool = False,
    ) -> None:
        """Settle (swap created) or release (swap failed) the pre-create
        reservation. Called on every swap-method exit."""
        if not reservation_id or self._capex_engine is None:
            return
        try:
            fee = max(0, int(estimated_fee_sats or 0))
        except (TypeError, ValueError):
            fee = 0
        subcat = "structural" if structural else "swap_fee"
        sid = ""
        created = False
        try:
            primary = self._primary_swap_entry(result) if result is not None else None
            if primary is not None:
                sid = str(primary.get("id") or "").strip()
                created = bool(sid) and not self._is_error_swap(primary)
        except Exception:
            sid, created = "", False
        try:
            if created and sid:
                self._capex_engine.settle_boltz_swap_reservation(
                    reservation_id=reservation_id,
                    swap_id=sid,
                    estimated_fee_sats=fee,
                    channel_id=channel_id,
                    subcategory=subcat,
                )
            else:
                self._capex_engine.release_boltz_swap_reservation(reservation_id)
        except Exception as e:
            try:
                self.plugin.log(f"BOLTZ: swap budget finalize error: {e}", level="warn")
            except Exception:
                pass

    # ---------------------------------------------------------------------
    # Public operations
    # ---------------------------------------------------------------------
    def wallet_balances(self) -> Dict[str, Any]:
        data = self._wallet_list()
        selected = {}
        for c in ("BTC", "LBTC"):
            w = self._resolve_wallet(c)
            if w:
                selected[c] = {"name": w.get("name"), "id": w.get("id"), "currency": w.get("currency")}
        return {"wallets": data.get("wallets", []), "selected_wallets": selected, "datadir": self.cfg.datadir}

    def quote(self, amount_sats: int, swap_type: str = "reverse", currency: Optional[str] = None,
              routing_fee_limit_ppm: Optional[int] = None) -> Dict[str, Any]:
        st = (swap_type or "reverse").strip().lower()
        amount_sats = int(amount_sats)
        if amount_sats <= 0:
            raise BoltzCliError("amount_sats must be > 0")

        if st == "reverse":
            target_cur = self._norm_currency(currency, "BTC")
            args = ["quote", "--json", "--send", str(amount_sats), "--to", target_cur, "reverse"]
        elif st in ("submarine", "normal"):
            source_cur = self._norm_currency(currency, "LBTC")
            args = ["quote", "--json", "--receive", str(amount_sats), "--from", source_cur, "submarine"]
        elif st == "chain":
            from_cur = self._norm_currency(currency, "LBTC")
            to_cur = "BTC" if from_cur == "LBTC" else "LBTC"
            args = ["quote", "--json", "--send", str(amount_sats), "--from", from_cur, "--to", to_cur, "chain"]
        else:
            raise BoltzCliError("swap_type must be reverse, submarine, or chain")

        data = self._run_json(args)
        # E-4.4: a reverse swap also pays LN routing to Boltz — include the
        # routing_fee_limit_ppm-based estimate in the plan-time total ONCE
        # (the boltzcli quote covers only service + on-chain fees).
        routing_fee_sats = (
            self._estimate_reverse_routing_fee_sats(
                amount_sats, routing_fee_limit_ppm=routing_fee_limit_ppm
            )
            if st == "reverse" else 0
        )
        return {
            "swap_type": st,
            "amount_sats": amount_sats,
            "currency": self._norm_currency(currency, "BTC" if st == "reverse" else "LBTC"),
            "quote": data,
            "estimated_routing_fee_sats": routing_fee_sats,
            "estimated_total_fee_sats": self._estimate_swap_fee_sats(data) + routing_fee_sats,
        }

    def loop_in(self, amount_sats: int, channel_id: Optional[str] = None, peer_id: Optional[str] = None,
                currency: Optional[str] = None) -> Dict[str, Any]:
        """Create a submarine (loop-in) swap: on-chain funds in, local balance up.

        channel_id/peer_id are trigger metadata only — boltzcli v2.11.0 cannot
        pin submarine swaps to a channel. Budget gates (swap quote, tactical,
        per-channel capex) run under the swap-creation lock and may reject the
        swap before any subprocess call.
        """
        amount_sats = int(amount_sats)
        if amount_sats <= 0:
            raise BoltzCliError("amount_sats must be > 0")
        source_cur = self._norm_currency(currency, "LBTC")
        wallet_name = self._resolve_wallet_name(source_cur)

        # P0-1 FIX: Serialize budget-check + swap-create to prevent TOCTOU race
        with self._swap_creation_lock:
            quote = self.quote(amount_sats=amount_sats, swap_type="submarine", currency=source_cur)
            budget_check = self._enforce_budget_for_quote(quote.get("quote", {}))
            if not budget_check["allowed"]:
                return {
                    "status": "rejected",
                    "error": budget_check["reason"],
                    "quote": quote,
                    "budget": budget_check["budget"],
                }

            # Tactical budget gate for pure treasury swaps
            tactical_check = self.check_tactical_budget(
                estimated_fee_sats=budget_check.get("estimated_fee_sats", 0),
                channel_id=channel_id,
            )
            if not tactical_check["allowed"]:
                return {
                    "status": "rejected",
                    "reason": tactical_check["reason"],
                    "budget": budget_check.get("budget"),
                }

            # Channel-targeted swaps draw on the channel's own capex budget.
            channel_check = self.check_channel_capex_budget(
                estimated_fee_sats=budget_check.get("estimated_fee_sats", 0),
                channel_id=channel_id,
            )
            if not channel_check["allowed"]:
                return {
                    "status": "rejected",
                    "reason": channel_check["reason"],
                    "budget": budget_check.get("budget"),
                    "channel_budget_check": channel_check,
                }

            warnings: List[str] = []
            if channel_id or peer_id:
                warnings.append(
                    "boltzcli createswap (submarine/loop-in) on v2.11.0 does not support channel pinning; channel_id/peer_id used only as trigger metadata"
                )

            # P4-014 / DD1: atomically reserve the swap's committed cost against
            # the unified cross-category budget before creating it.
            est_fee = int(budget_check.get("estimated_fee_sats", 0) or 0)
            reservation_id = self._open_swap_budget_reservation(
                est_fee, channel_id, structural=False,
                intent_type="SWAP_IN")
            if reservation_id is False:
                return {
                    "status": "rejected",
                    "error": "unified_budget_exceeded",
                    "quote": quote,
                    "budget": budget_check.get("budget"),
                }

            result = None
            try:
                args = ["createswap", "--json", "--from-wallet", wallet_name, self._swap_cli_currency(source_cur, source_cur), str(amount_sats)]
                result = self._run_json(args, timeout=max(self.cfg.timeout_seconds, 120))
                self._record_swap_result(
                    result,
                    source="loop_in",
                    metadata={"trigger_channel_id": channel_id, "trigger_peer_id": peer_id},
                )
            finally:
                self._finalize_swap_budget_reservation(
                    reservation_id, result, est_fee, channel_id, structural=False
                )

        primary = self._primary_swap_entry(result)
        status = "error" if self._is_error_swap(primary) else "accepted"
        return {
            "status": status,
            "swap_type": "submarine",
            "amount_sats": amount_sats,
            "funding_currency": source_cur,
            "wallet": wallet_name,
            "trigger_channel_id": channel_id,
            "trigger_peer_id": peer_id,
            "quote": quote,
            "budget_check": budget_check,
            "warnings": warnings,
            "result": result,
        }

    def loop_out(self, amount_sats: int, address: Optional[str] = None, channel_id: Optional[str] = None,
                 peer_id: Optional[str] = None, currency: Optional[str] = None,
                 routing_fee_limit_ppm: Optional[int] = None, structural: bool = False) -> Dict[str, Any]:
        """Create a reverse (loop-out) swap: local balance out, on-chain funds in.

        `structural` is True when the candidate's profit-guard pass depended on
        the inbound-scarcity credit; it routes the recorded fee into the
        boltz/structural envelope partition (category="boltz",
        subcategory="structural") so the daily structural envelope gate can
        account for it. Budget gates run under the swap-creation lock.
        """
        amount_sats = int(amount_sats)
        if amount_sats <= 0:
            raise BoltzCliError("amount_sats must be > 0")
        target_cur = self._norm_currency(currency, "BTC")

        # P4-005: validate the operator-supplied on-chain destination for the
        # settlement network BEFORE any subprocess call. A wrong-network/typo'd
        # address on a reverse swap sends proceeds to an unrecoverable dest.
        # (When no address is given, proceeds go to the internal wallet.)
        if address and not _validate_onchain_address(address, target_cur):
            raise BoltzCliError(
                f"invalid {target_cur} on-chain destination address: refusing loop-out"
            )

        # P0-1 FIX: Serialize budget-check + swap-create to prevent TOCTOU race
        with self._swap_creation_lock:
            # P4-014: _loop_out_locked reserves the swap's committed cost against
            # the unified cross-category budget before creation and stashes the
            # reservation context on self; settle (created) or release (failed /
            # exception) it on every exit so the budget is never held after the
            # swap resolves and a boltz commitment is always counted while live.
            self._pending_swap_budget = None
            out = None
            try:
                out = self._loop_out_locked(amount_sats, address, channel_id, peer_id, currency, target_cur,
                                            routing_fee_limit_ppm, structural=structural)
                ctx = self._pending_swap_budget
                if ctx and ctx.get("reservation_id"):
                    self._finalize_swap_budget_reservation(
                        ctx["reservation_id"],
                        out.get("result") if isinstance(out, dict) else None,
                        ctx["estimated_fee_sats"], ctx["channel_id"], ctx["structural"],
                    )
                return out
            except Exception:
                ctx = self._pending_swap_budget
                if ctx and ctx.get("reservation_id"):
                    # Unknown result → release (no committed cost recorded).
                    self._finalize_swap_budget_reservation(
                        ctx["reservation_id"], None,
                        ctx["estimated_fee_sats"], ctx["channel_id"], ctx["structural"],
                    )
                raise
            finally:
                self._pending_swap_budget = None

    def _loop_out_locked(self, amount_sats: int, address: Optional[str], channel_id: Optional[str],
                         peer_id: Optional[str], currency: Optional[str], target_cur: str,
                         routing_fee_limit_ppm: Optional[int] = None, structural: bool = False) -> Dict[str, Any]:
        # E-4.4: quote with the SAME routing-fee limit the swap will be
        # created with, so the plan-time total includes the LN routing leg.
        quote = self.quote(
            amount_sats=amount_sats, swap_type="reverse", currency=target_cur,
            routing_fee_limit_ppm=(
                int(routing_fee_limit_ppm) if routing_fee_limit_ppm
                else (self.cfg.routing_fee_limit_ppm or 0)
            ),
        )
        budget_check = self._enforce_budget_for_quote(
            quote.get("quote", {}),
            extra_fee_sats=quote.get("estimated_routing_fee_sats", 0),
        )
        if not budget_check["allowed"]:
            return {
                "status": "rejected",
                "error": budget_check["reason"],
                "quote": quote,
                "budget": budget_check["budget"],
            }

        # Tactical budget gate for pure treasury swaps
        tactical_check = self.check_tactical_budget(
            estimated_fee_sats=budget_check.get("estimated_fee_sats", 0),
            channel_id=channel_id,
        )
        if not tactical_check["allowed"]:
            return {
                "status": "rejected",
                "reason": tactical_check["reason"],
                "budget": budget_check.get("budget"),
            }

        # Channel-targeted swaps draw on the channel's own capex budget —
        # EXCEPT structural drains when the daily envelope is configured.
        # Source-heavy decayed channels carry bootstrap-scale (<=200 sat)
        # 30d capex budgets that rejected exactly the drains the structural
        # feature exists for; those swaps are already gated by the dedicated
        # daily envelope (fail-closed, balance cycle) + the unified budget,
        # which remain the binding constraints.
        if structural and self._structural_envelope_sats() > 0:
            try:
                self.plugin.log(
                    "BOLTZ: structural swap: envelope-gated, per-channel capex bypassed",
                    level="info",
                )
            except Exception:
                pass
        else:
            channel_check = self.check_channel_capex_budget(
                estimated_fee_sats=budget_check.get("estimated_fee_sats", 0),
                channel_id=channel_id,
            )
            if not channel_check["allowed"]:
                return {
                    "status": "rejected",
                    "reason": channel_check["reason"],
                    "budget": budget_check.get("budget"),
                    "channel_budget_check": channel_check,
                }

        # P4-014 / DD1: atomically reserve the swap's committed cost against the
        # unified cross-category budget before creating it. Stash the context so
        # the loop_out wrapper settles (created) or releases (failed/exception).
        est_fee = int(budget_check.get("estimated_fee_sats", 0) or 0)
        reservation_id = self._open_swap_budget_reservation(
            est_fee, channel_id, structural=structural,
            intent_type="SWAP_OUT")
        if reservation_id is False:
            return {
                "status": "rejected",
                "error": "unified_budget_exceeded",
                "quote": quote,
                "budget": budget_check.get("budget"),
            }
        self._pending_swap_budget = {
            "reservation_id": reservation_id,
            "estimated_fee_sats": est_fee,
            "channel_id": channel_id,
            "structural": structural,
        }

        target_channel_id = (str(channel_id).replace(':', 'x') if channel_id else None)
        target_peer_id = str(peer_id).strip() if peer_id else None
        chan_ids: List[str] = []
        if target_channel_id:
            chan_ids.append(target_channel_id)
        elif target_peer_id:
            chan_ids.extend(self._resolve_peer_channel_ids(target_peer_id))

        warnings: List[str] = []
        wallet_name = None if address else self._resolve_wallet_name(target_cur)
        reverse_chanids_supported = self._detect_reverse_chanids_support()
        effective_routing_fee_limit = int(routing_fee_limit_ppm) if routing_fee_limit_ppm else (self.cfg.routing_fee_limit_ppm or 0)

        # CLN backends that reject reverse chanIds can still be routed through a desired first hop by
        # creating the reverse swap with --external-pay and then paying the invoice from CLN while
        # excluding all other local peers.
        use_external_routed = bool(target_channel_id or target_peer_id) and (reverse_chanids_supported is False)
        if use_external_routed:
            resolved_peer, resolved_channel, target_warnings = self._resolve_first_hop_target(
                channel_id=target_channel_id,
                peer_id=target_peer_id,
            )
            warnings.extend(target_warnings)
            warnings.append(
                "CLN boltz backend does not support reverse-swap chanIds; using external-pay with CLN first-hop constrained payment"
            )
            cmd: List[str] = ["createreverseswap", "--json", "--external-pay"]
            if effective_routing_fee_limit > 0:
                cmd.extend(["--routing-fee-limit-ppm", str(effective_routing_fee_limit)])
            if address:
                pass
            else:
                cmd.extend(["--to-wallet", str(wallet_name)])
            # `--` terminates option parsing so the free-form on-chain address
            # (and other positionals) cannot be reparsed by boltzcli as flags.
            cmd.append("--")
            cmd.extend([self._swap_cli_currency(target_cur, target_cur), str(amount_sats)])
            if address:
                cmd.append(address)

            result = self._run_json(cmd, timeout=max(self.cfg.timeout_seconds, 120))
            self._record_swap_result(
                result,
                source="loop_out_external_create",
                metadata={
                    "peer_id": resolved_peer,
                    "requested_channel_ids": [resolved_channel] if resolved_channel else None,
                    # None (not False) so the journal-merge None-filter drops it
                    "structural": True if structural else None,
                },
            )
            invoice = self._extract_reverse_swap_invoice(result)
            if not invoice:
                return {
                    "status": "error",
                    "error": "reverse swap created but no invoice was found in boltz response for external-pay",
                    "swap_type": "reverse",
                    "amount_sats": amount_sats,
                    "settlement_currency": target_cur,
                    "channel_ids": [resolved_channel] if resolved_channel else [],
                    "peer_id": resolved_peer,
                    "address": address,
                    "quote": quote,
                    "budget_check": budget_check,
                    "warnings": warnings,
                    "result": result,
                }
            cln_payment = self._pay_invoice_via_first_hop(
                invoice,
                preferred_peer_id=resolved_peer,
                preferred_channel_id=resolved_channel,
                retry_for=120,
                expected_amount_sats=amount_sats,
            )
            primary = self._primary_swap_entry(result)
            created_id = str((primary or {}).get("id") or "").strip()
            status_probe = None
            if created_id:
                try:
                    time.sleep(0.5)
                    status_probe = self.swap_status(created_id)
                except Exception as e:
                    warnings.append(f"swap status probe failed after external-pay submit: {e}")
            outer_status = "accepted"
            if isinstance(cln_payment, dict) and cln_payment.get("status") in ("timeout", "error"):
                outer_status = str(cln_payment.get("status"))
            try:
                pay_summary = None
                if isinstance(cln_payment, dict):
                    pay_summary = {
                        "status": cln_payment.get("status"),
                        "preferred_peer_id": cln_payment.get("preferred_peer_id"),
                        "preferred_channel_id": cln_payment.get("preferred_channel_id"),
                        "exclude_count": cln_payment.get("exclude_count"),
                        "warnings": cln_payment.get("warnings"),
                    }
                    if cln_payment.get("pay_error"):
                        pay_summary["pay_error"] = cln_payment.get("pay_error")
                self._record_swap_result(
                    result,
                    source="loop_out_external_pay",
                    metadata={
                        "peer_id": resolved_peer,
                        "requested_channel_ids": [resolved_channel] if resolved_channel else None,
                        "requested_first_hop_peer_id": resolved_peer,
                        "requested_first_hop_channel_id": resolved_channel,
                        "routing_mode": "external_pay_first_hop_pinned",
                        "cln_payment_status": (cln_payment.get("status") if isinstance(cln_payment, dict) else None),
                        "cln_payment_error": (cln_payment.get("pay_error") if isinstance(cln_payment, dict) else None),
                        "cln_pay_lookup": (cln_payment.get("pay_lookup") if isinstance(cln_payment, dict) else None),
                        "cln_pay_summary": pay_summary,
                    },
                )
                if isinstance(status_probe, dict):
                    self._record_swap_result(result, source="loop_out_external_pay_status_probe", metadata={
                        "status_probe": status_probe,
                    })
            except Exception:
                pass
            return {
                "status": outer_status,

                "swap_type": "reverse",
                "amount_sats": amount_sats,
                "settlement_currency": target_cur,
                "channel_ids": [resolved_channel] if resolved_channel else [],
                "peer_id": resolved_peer,
                "address": address,
                "quote": quote,
                "budget_check": budget_check,
                "warnings": warnings,
                "routing_mode": "external_pay_first_hop_pinned",
                "cln_payment": cln_payment,
                "result": result,
                "status_probe": status_probe,
            }

        include_chanids_initial = bool(chan_ids) and (reverse_chanids_supported is not False)
        if chan_ids and not include_chanids_initial:
            warnings.append("CLN boltz backend does not support reverse-swap chanIds; submitted without channel pinning")

        def _build_args(include_chanids: bool) -> List[str]:
            cmd: List[str] = ["createreverseswap", "--json"]
            if include_chanids:
                for scid in chan_ids:
                    cmd.extend(["--chan-id", scid])
            if effective_routing_fee_limit > 0:
                cmd.extend(["--routing-fee-limit-ppm", str(effective_routing_fee_limit)])
            if address:
                pass
            else:
                cmd.extend(["--to-wallet", str(wallet_name)])
            # `--` terminates option parsing so the free-form on-chain address
            # (and other positionals) cannot be reparsed by boltzcli as flags.
            cmd.append("--")
            cmd.extend([self._swap_cli_currency(target_cur, target_cur), str(amount_sats)])
            if address:
                cmd.append(address)
            return cmd

        try:
            result = self._run_json(_build_args(include_chanids=include_chanids_initial), timeout=max(self.cfg.timeout_seconds, 120))
            if chan_ids and self._contains_chanids_cln_error(result):
                self._reverse_chanids_supported = False
                warnings.append("CLN boltz backend rejected chanIds in JSON result; retried reverse swap without channel pinning")
                # Re-check budget before creating second swap (first swap may have consumed fees).
                # Pass the nested quote (not the wrapper) so the fee estimate is not double-counted.
                retry_budget = self._enforce_budget_for_quote(
                    quote.get("quote", {}),
                    extra_fee_sats=quote.get("estimated_routing_fee_sats", 0),
                )
                if not retry_budget["allowed"]:
                    warnings.append(f"Budget exhausted after chanId rejection: {retry_budget.get('reason')}")
                else:
                    result = self._run_json(_build_args(include_chanids=False), timeout=max(self.cfg.timeout_seconds, 120))
            elif chan_ids and include_chanids_initial:
                # Some CLN/Boltz versions accept create-reverse-swap first, then surface the chanIds rejection asynchronously in swapinfo.
                primary_created = self._primary_swap_entry(result)
                created_id = str((primary_created or {}).get("id") or "").strip()
                if created_id and not self._is_error_swap(primary_created):
                    probe_timeout = max(self.cfg.timeout_seconds, 120)
                    for _ in range(3):
                        try:
                            probe = self._run_json(["swapinfo", created_id], timeout=probe_timeout)
                        except Exception:
                            probe = None
                        if isinstance(probe, dict):
                            try:
                                self._record_swap_result(probe, source="loop_out_probe")
                            except Exception:
                                pass
                            if self._contains_chanids_cln_error(probe):
                                self._reverse_chanids_supported = False
                                warnings.append("CLN boltz backend rejected chanIds asynchronously; retried reverse swap without channel pinning")
                                # Re-check budget before creating second swap (nested quote, not wrapper)
                                retry_budget = self._enforce_budget_for_quote(
                    quote.get("quote", {}),
                    extra_fee_sats=quote.get("estimated_routing_fee_sats", 0),
                )
                                if not retry_budget["allowed"]:
                                    warnings.append(f"Budget exhausted after async chanId rejection: {retry_budget.get('reason')}")
                                else:
                                    result = self._run_json(_build_args(include_chanids=False), timeout=probe_timeout)
                                break
                            probe_primary = self._primary_swap_entry(probe)
                            if probe_primary and not self._is_error_swap(probe_primary):
                                # No async rejection observed yet; avoid extra blocking if swap is proceeding normally.
                                break
                        time.sleep(0.5)
        except BoltzCliError as e:
            msg = str(e)
            if chan_ids and "chanIds are not supported for cln" in msg:
                # CLN backends may reject chanIds even though boltzcli accepts the flag.
                self._reverse_chanids_supported = False
                warnings.append("CLN boltz backend rejected chanIds; retried reverse swap without channel pinning")
                # Re-check budget before creating second swap (nested quote, not wrapper)
                retry_budget = self._enforce_budget_for_quote(
                    quote.get("quote", {}),
                    extra_fee_sats=quote.get("estimated_routing_fee_sats", 0),
                )
                if not retry_budget["allowed"]:
                    warnings.append(f"Budget exhausted after chanId exception: {retry_budget.get('reason')}")
                else:
                    result = self._run_json(_build_args(include_chanids=False), timeout=max(self.cfg.timeout_seconds, 120))
            else:
                raise
        self._record_swap_result(
            result,
            source="loop_out",
            metadata={
                "peer_id": target_peer_id,
                "requested_channel_ids": chan_ids or None,
                # None (not False) so the journal-merge None-filter drops it
                "structural": True if structural else None,
            },
        )
        primary = self._primary_swap_entry(result)
        status = "accepted"
        if self._is_error_swap(primary):
            status = "error"
        return {
            "status": status,
            "swap_type": "reverse",
            "amount_sats": amount_sats,
            "settlement_currency": target_cur,
            "channel_ids": chan_ids,
            "peer_id": target_peer_id,
            "address": address,
            "quote": quote,
            "budget_check": budget_check,
            "warnings": warnings,
            "result": result,
        }

    def swap_status(self, swap_id: str) -> Dict[str, Any]:
        swap_id = str(swap_id)
        # `--` terminates option parsing so a swap_id beginning with '-' is
        # treated as a positional value, not reparsed by boltzcli as a flag.
        raw = self._run(["swapinfo", "--", swap_id], timeout=max(self.cfg.timeout_seconds, 120))
        list_json = None
        list_match = None
        try:
            list_json = self._run_json(["listswaps", "--json"])
            for s in self._extract_swap_list(list_json):
                if str(s.get("id")) == swap_id:
                    list_match = s
                    break
        except Exception:
            list_json = None
        swapinfo_json = None
        swapinfo_entry = None
        try:
            swapinfo_json = json.loads(raw)
            extracted = self._extract_swap_list(swapinfo_json)
            if extracted:
                swapinfo_entry = extracted[0]
                self._record_swap_result({"swaps": extracted}, source="swap_status_lookup")
        except Exception:
            swapinfo_json = None
        ignores = self._load_ignored_external_swaps()
        journal_index = self._load_swap_journal_index()
        annotated_swapinfo = self._annotate_journal_swap(self._annotate_ignored_swap(swapinfo_entry, ignores), journal_index)
        annotated_list = self._annotate_journal_swap(self._annotate_ignored_swap(list_match, ignores), journal_index)
        ignore_meta = ignores.get(swap_id)
        journal_meta = journal_index.get(swap_id)
        return {
            "swap_id": swap_id,
            "swapinfo_raw": raw,
            "swapinfo_entry": annotated_swapinfo,
            "listswaps_entry": annotated_list,
            "ignored_external_swap": bool(ignore_meta),
            "ignored_external_swap_meta": ignore_meta,
            "journal_meta": journal_meta,
        }

    def swap_history(self, limit: Optional[int] = None) -> Dict[str, Any]:
        data = self._listswaps_json()
        swaps = self._extract_swap_list(data)
        swaps = self._augment_with_swap_journal(swaps, limit_hint=limit)
        journal_index = self._load_swap_journal_index()
        swaps = [self._annotate_journal_swap(s, journal_index) for s in swaps]
        ignores = self._load_ignored_external_swaps()
        swaps = [self._annotate_ignored_swap(s, ignores) for s in swaps]
        # sort newest first (best effort)
        swaps.sort(key=lambda s: self._swap_created_ts(s) or 0, reverse=True)
        if limit is not None:
            try:
                lim = max(0, int(limit))
                swaps = swaps[:lim]
            except Exception:
                pass
        cost_summary = {
            "swap_count": len(swaps),
            "estimated_total_fee_sats": sum(self._estimate_swap_fee_sats(s) for s in swaps),
            "completed_count": sum(1 for s in swaps if self._is_completed_swap(s)),
        }
        return {"swaps": swaps, "cost_summary": cost_summary}

    def budget(self) -> Dict[str, Any]:
        return self.get_budget_status()

    def refund(self, swap_id: str, destination: Optional[str] = None) -> Dict[str, Any]:
        dest = destination or "wallet"
        # P4-005: validate an explicit on-chain destination before the
        # subprocess call (the "wallet" keyword targets the internal wallet).
        if dest != "wallet" and not _validate_swap_destination(dest):
            raise BoltzCliError(
                "invalid on-chain refund destination address: refusing refund"
            )
        # `--` terminates option parsing so a swap_id/dest beginning with '-'
        # is treated as a positional value, not reparsed by boltzcli as a flag.
        raw = self._run(["refundswap", "--", str(swap_id), str(dest)], timeout=max(self.cfg.timeout_seconds, 120))
        return {"swap_id": swap_id, "destination": dest, "result_raw": raw}

    def claim(self, swap_ids: List[str], destination: Optional[str] = None) -> Dict[str, Any]:
        ids = [str(x) for x in (swap_ids or []) if str(x).strip()]
        if not ids:
            raise BoltzCliError("swap_ids is required")
        dest = destination or "wallet"
        # P4-005: validate an explicit on-chain destination before the
        # subprocess call (the "wallet" keyword targets the internal wallet).
        if dest != "wallet" and not _validate_swap_destination(dest):
            raise BoltzCliError(
                "invalid on-chain claim destination address: refusing claim"
            )
        # `--` terminates option parsing so a dest/swap-id beginning with '-'
        # is treated as a positional value, not reparsed by boltzcli as a flag.
        raw = self._run(["claimswaps", "--", str(dest)] + ids, timeout=max(self.cfg.timeout_seconds, 120))
        return {"swap_ids": ids, "destination": dest, "result_raw": raw}

    def chainswap(self, amount_sats: int, from_currency: Optional[str] = None, to_currency: Optional[str] = None,
                  to_address: Optional[str] = None) -> Dict[str, Any]:
        amount_sats = int(amount_sats)
        if amount_sats <= 0:
            raise BoltzCliError("amount_sats must be > 0")
        from_cur = self._norm_currency(from_currency, "LBTC")
        to_cur = self._norm_currency(to_currency, "BTC")
        if from_cur == to_cur:
            raise BoltzCliError("from_currency and to_currency must differ")

        # P4-005: validate the operator-supplied to_address for the destination
        # network BEFORE any subprocess call. (No to_address => internal wallet.)
        if to_address and not _validate_onchain_address(to_address, to_cur):
            raise BoltzCliError(
                f"invalid {to_cur} on-chain to_address: refusing chainswap"
            )

        # P0-1 FIX: Serialize budget-check + swap-create to prevent TOCTOU race
        with self._swap_creation_lock:
            quote = self._run_json([
                "quote", "--json", "--send", str(amount_sats), "--from", from_cur, "--to", to_cur, "chain"
            ])
            budget_check = self._enforce_budget_for_quote(quote)
            if not budget_check["allowed"]:
                return {
                    "status": "rejected",
                    "error": budget_check["reason"],
                    "quote": quote,
                    "budget": budget_check["budget"],
                }

            # P4-023 / DD1: chainswap is the 6th swap-create and was the ONLY one
            # with no atomic pre-create reservation — its sole gate
            # (_enforce_budget_for_quote above) is a soft advisory read (no
            # reserve_spend / BEGIN IMMEDIATE, a no-op when enforce_budget=false)
            # and the actual cost was recorded only after an up-to-180s
            # subprocess, invisible to a concurrent daemon reserve for that whole
            # window. Reserve the swap's estimated cost against the unified
            # cross-category budget BEFORE createchainswap so a concurrent
            # rebalance/boltz/capex reserve both sees and is counted by this
            # commitment; reject the swap if the unified budget would be
            # exceeded (mirrors loop_in / loop_out, P4-014).
            est_fee = int(budget_check.get("estimated_fee_sats", 0) or 0)
            # Chainswap (BTC<->LBTC): closest wire intent type is SWAP_OUT;
            # the explanation carries subcategory for the distinction.
            reservation_id = self._open_swap_budget_reservation(
                est_fee, channel_id=None, structural=False,
                intent_type="SWAP_OUT"
            )
            if reservation_id is False:
                return {
                    "status": "rejected",
                    "error": "unified_budget_exceeded",
                    "quote": quote,
                    "budget": budget_check.get("budget"),
                }

            args: List[str] = ["createchainswap", "--json", "--from-wallet", self._resolve_wallet_name(from_cur)]
            if to_address:
                args.extend(["--to-address", str(to_address)])
            else:
                args.extend(["--to-wallet", self._resolve_wallet_name(to_cur)])
            # P4-006: `--` terminates option parsing so the positional amount
            # (and any future free-form positional) is passed as data, never
            # reparsed by boltzcli as a flag — matching the sibling P1-015
            # commands (withdraw/refund/claim/loop_out).
            args.append("--")
            args.append(str(amount_sats))
            result = None
            try:
                result = self._run_json(args, timeout=max(self.cfg.timeout_seconds, 180))
                self._record_swap_result(result, source="chainswap")
            finally:
                # Settle (swap created) or release (failed/exception) the
                # pre-create reservation on every exit so the boltz commitment
                # is counted while live and never held after it resolves.
                self._finalize_swap_budget_reservation(
                    reservation_id, result, est_fee, channel_id=None, structural=False
                )
        primary = self._primary_swap_entry(result)
        status = "error" if self._is_error_swap(primary) else "accepted"
        return {
            "status": status,
            "amount_sats": amount_sats,
            "from_currency": from_cur,
            "to_currency": to_cur,
            "quote": quote,
            "budget_check": budget_check,
            "result": result,
        }

    def withdraw(self, amount_sats: Optional[int], destination: str, currency: Optional[str] = None,
                 sat_per_vbyte: Optional[int] = None, sweep: bool = False,
                 confirm_sweep: bool = False) -> Dict[str, Any]:
        if not destination:
            raise BoltzCliError("destination is required")
        cur = self._norm_currency(currency, "LBTC")
        # P1-006: validate the destination address format for the wallet
        # currency/network BEFORE any subprocess call. A malformed or
        # wrong-network destination is rejected cleanly and never reaches
        # boltzcli.
        if not _validate_onchain_address(destination, cur):
            raise BoltzCliError(
                f"invalid {cur} on-chain destination address: refusing to withdraw"
            )
        wallet_name = self._resolve_wallet_name(cur)
        amt = 0 if sweep else int(amount_sats or 0)
        if not sweep and amt <= 0:
            raise BoltzCliError("amount_sats must be > 0 unless sweep=true")
        # DD3 / P1-006: enforce the hard single-withdraw cap BEFORE the
        # subprocess call so a typo/automation bug cannot drain the wallet.
        try:
            max_withdraw = int(getattr(self.cfg, "max_withdraw_sats", 0) or 0)
        except (ValueError, TypeError):
            max_withdraw = 0
        if sweep:
            # Sweep intentionally empties the wallet and cannot be bounded by
            # the amount cap (the amount is unknown until boltzcli runs), so it
            # is the exact "drain the wallet" action the cap guards against.
            # Safer choice: require an explicit confirmation flag.
            if not confirm_sweep:
                raise BoltzCliError(
                    "sweep withdraws the entire wallet balance and bypasses the "
                    "max_withdraw_sats cap; pass confirm_sweep=true to proceed"
                )
        elif max_withdraw > 0 and amt > max_withdraw:
            raise BoltzCliError(
                f"withdraw amount {amt} exceeds max_withdraw_sats cap {max_withdraw}: "
                "refusing to withdraw"
            )
        # Flags first, then a `--` terminator before the free-form positional
        # args (wallet/destination/amount). This keeps a destination beginning
        # with '-' from being reparsed by boltzcli as a flag.
        args: List[str] = ["wallet", "send"]
        if sat_per_vbyte is not None:
            args.extend(["--sat-per-vbyte", str(int(sat_per_vbyte))])
        if sweep:
            args.append("--sweep")
        args.extend(["--", wallet_name, destination, str(amt)])
        raw = self._run(args, timeout=max(self.cfg.timeout_seconds, 180))
        return {
            "wallet": wallet_name,
            "currency": cur,
            "destination": destination,
            "amount_sats": amt,
            "sweep": bool(sweep),
            "result_raw": raw,
        }

    def deposit_address(self, currency: Optional[str] = None) -> Dict[str, Any]:
        cur = self._norm_currency(currency, "LBTC")
        wallet_name = self._resolve_wallet_name(cur)
        raw = self._run(["wallet", "receive", wallet_name])
        # Usually returns a single address line
        address = raw.splitlines()[-1].strip() if raw else ""
        return {"wallet": wallet_name, "currency": cur, "address": address, "raw": raw}

    def backup(self, include_mnemonic: bool = False) -> Dict[str, Any]:
        wallets = self.wallet_balances()
        pending = self._listswaps_json(pending_only=True)
        result: Dict[str, Any] = {
            "wallets": wallets.get("wallets", []),
            "pending_swaps": self._extract_swap_list(pending),
        }
        if include_mnemonic:
            mnemonic = self._run(["swapmnemonic", "get"])
            result["swap_mnemonic"] = mnemonic.strip()
            result["warning"] = "Contains plaintext swap mnemonic. Store securely."
        else:
            result["note"] = "Swap mnemonic omitted. Pass include_mnemonic=true to include."
        return result

    def backup_verify(self, swap_mnemonic: str) -> Dict[str, Any]:
        current = self._run(["swapmnemonic", "get"]).strip()
        provided = " ".join(str(swap_mnemonic or "").split())
        actual = " ".join(current.split())
        return {
            "matches": provided == actual,
            "provided_word_count": len(provided.split()),
            "current_word_count": len(actual.split()),
        }
