"""
Boltz CLI integration for cl-revenue-ops.

Provides a synchronous manager used by plugin RPC methods to call `boltzcli`
against a local `boltzd` instance. Designed as a best-effort integration that
works even when boltzd autoswap is unstable; revenue ops can still perform
manual quotes/swaps and enforce a local daily fee budget for swap fees.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


class BoltzCliError(RuntimeError):
    """Raised when boltzcli execution fails."""


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


class BoltzCliManager:
    def __init__(self, plugin, rpc, config: BoltzCliConfig):
        self.plugin = plugin
        self.rpc = rpc
        self.cfg = config

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
            result = self.rpc.listpeerchannels(peer_id)
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
            for key in ('swaps', 'list'):
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

        # Fallback: recursively sum fee-like integer fields (excluding percentages/rates).
        def rec(obj: Any, path: str = "") -> int:
            subtotal = 0
            if isinstance(obj, dict):
                for k, v in obj.items():
                    lk = str(k).lower()
                    if "fee" in lk and not any(x in lk for x in ("percent", "percentage", "ppm", "rate")):
                        if isinstance(v, (int, float, str)):
                            subtotal += max(0, self._parse_int(v, 0))
                            continue
                    subtotal += rec(v, path + "." + str(k))
            elif isinstance(obj, list):
                for v in obj:
                    subtotal += rec(v, path)
            return subtotal

        return rec(swap)

    def _swap_created_ts(self, swap: Dict[str, Any]) -> Optional[int]:
        for key in ("createdAt", "updatedAt", "created_at", "updated_at"):
            ts = self._parse_timestamp(swap.get(key))
            if ts:
                return ts
        return None

    def _swap_status_text(self, swap: Dict[str, Any]) -> str:
        return str(swap.get('state') or swap.get('status') or '').lower()

    def _is_completed_swap(self, swap: Dict[str, Any]) -> bool:
        st = self._swap_status_text(swap)
        return any(token in st for token in ("success", "completed", "claimed", "done"))

    # ---------------------------------------------------------------------
    # Budget helpers
    # ---------------------------------------------------------------------
    def get_budget_status(self) -> Dict[str, Any]:
        budget = max(0, int(self.cfg.daily_budget_sats))
        swaps_json = self._listswaps_json(manual_only=True)
        swaps = self._extract_swap_list(swaps_json)
        now = int(time.time())
        cutoff = now - 86400

        spent = 0
        counted: List[Dict[str, Any]] = []
        unknown_ts = 0
        for s in swaps:
            ts = self._swap_created_ts(s)
            if ts is None:
                unknown_ts += 1
                continue
            if ts < cutoff:
                continue
            if not self._is_completed_swap(s):
                continue
            fee_sats = self._estimate_swap_fee_sats(s)
            spent += max(0, fee_sats)
            counted.append({
                "id": s.get("id"),
                "created_at": ts,
                "fee_sats_estimate": fee_sats,
                "state": s.get("state"),
                "status": s.get("status"),
            })

        remaining = max(0, budget - spent)
        return {
            "daily_budget_sats": budget,
            "spent_24h_sats_estimate": spent,
            "remaining_24h_sats_estimate": remaining,
            "counted_swaps": len(counted),
            "skipped_without_timestamp": unknown_ts,
            "enforce_budget": bool(self.cfg.enforce_budget),
            "window_seconds": 86400,
            "counted_details": counted[:20],
        }

    def _enforce_budget_for_quote(self, quote: Dict[str, Any]) -> Dict[str, Any]:
        fee_sats = max(0, self._parse_int(quote.get("boltzFee"), 0)) + max(0, self._parse_int(quote.get("networkFee"), 0))
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

    def quote(self, amount_sats: int, swap_type: str = "reverse", currency: Optional[str] = None) -> Dict[str, Any]:
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
        return {
            "swap_type": st,
            "amount_sats": amount_sats,
            "currency": self._norm_currency(currency, "BTC" if st == "reverse" else "LBTC"),
            "quote": data,
            "estimated_total_fee_sats": max(0, self._parse_int(data.get("boltzFee"), 0)) + max(0, self._parse_int(data.get("networkFee"), 0)),
        }

    def loop_in(self, amount_sats: int, channel_id: Optional[str] = None, peer_id: Optional[str] = None,
                currency: Optional[str] = None) -> Dict[str, Any]:
        amount_sats = int(amount_sats)
        if amount_sats <= 0:
            raise BoltzCliError("amount_sats must be > 0")
        source_cur = self._norm_currency(currency, "LBTC")
        wallet_name = self._resolve_wallet_name(source_cur)

        quote = self.quote(amount_sats=amount_sats, swap_type="submarine", currency=source_cur)
        budget_check = self._enforce_budget_for_quote(quote.get("quote", {}))
        if not budget_check["allowed"]:
            return {
                "status": "rejected",
                "error": budget_check["reason"],
                "quote": quote,
                "budget": budget_check["budget"],
            }

        warnings: List[str] = []
        if channel_id or peer_id:
            warnings.append(
                "boltzcli createswap (submarine/loop-in) on v2.11.0 does not support channel pinning; channel_id/peer_id used only as trigger metadata"
            )

        args = ["createswap", "--json", "--from-wallet", wallet_name, self._swap_cli_currency(source_cur, source_cur), str(amount_sats)]
        result = self._run_json(args, timeout=max(self.cfg.timeout_seconds, 120))

        return {
            "status": "accepted",
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
                 peer_id: Optional[str] = None, currency: Optional[str] = None) -> Dict[str, Any]:
        amount_sats = int(amount_sats)
        if amount_sats <= 0:
            raise BoltzCliError("amount_sats must be > 0")
        target_cur = self._norm_currency(currency, "BTC")

        quote = self.quote(amount_sats=amount_sats, swap_type="reverse", currency=target_cur)
        budget_check = self._enforce_budget_for_quote(quote.get("quote", {}))
        if not budget_check["allowed"]:
            return {
                "status": "rejected",
                "error": budget_check["reason"],
                "quote": quote,
                "budget": budget_check["budget"],
            }

        args: List[str] = ["createreverseswap", "--json"]
        chan_ids: List[str] = []
        if channel_id:
            chan_ids.append(str(channel_id).replace(':', 'x'))
        elif peer_id:
            chan_ids.extend(self._resolve_peer_channel_ids(peer_id))
        for scid in chan_ids:
            args.extend(["--chan-id", scid])

        if address:
            # address is positional after amount
            pass
        else:
            wallet_name = self._resolve_wallet_name(target_cur)
            args.extend(["--to-wallet", wallet_name])

        args.extend([self._swap_cli_currency(target_cur, target_cur), str(amount_sats)])
        if address:
            args.append(address)

        warnings: List[str] = []
        try:
            result = self._run_json(args, timeout=max(self.cfg.timeout_seconds, 120))
        except BoltzCliError as e:
            msg = str(e)
            if chan_ids and "chanIds are not supported for cln" in msg:
                # CLN backends may reject chanIds even though boltzcli accepts the flag.
                retry_args: List[str] = ["createreverseswap", "--json"]
                if address:
                    pass
                else:
                    wallet_name = self._resolve_wallet_name(target_cur)
                    retry_args.extend(["--to-wallet", wallet_name])
                retry_args.extend([self._swap_cli_currency(target_cur, target_cur), str(amount_sats)])
                if address:
                    retry_args.append(address)
                warnings.append("CLN boltz backend rejected chanIds; retried reverse swap without channel pinning")
                result = self._run_json(retry_args, timeout=max(self.cfg.timeout_seconds, 120))
            else:
                raise
        return {
            "status": "accepted",
            "swap_type": "reverse",
            "amount_sats": amount_sats,
            "settlement_currency": target_cur,
            "channel_ids": chan_ids,
            "peer_id": peer_id,
            "address": address,
            "quote": quote,
            "budget_check": budget_check,
            "warnings": warnings,
            "result": result,
        }

    def swap_status(self, swap_id: str) -> Dict[str, Any]:
        swap_id = str(swap_id)
        raw = self._run(["swapinfo", swap_id], timeout=max(self.cfg.timeout_seconds, 120))
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
        except Exception:
            swapinfo_json = None
        return {
            "swap_id": swap_id,
            "swapinfo_raw": raw,
            "swapinfo_entry": swapinfo_entry,
            "listswaps_entry": list_match,
        }

    def swap_history(self, limit: Optional[int] = None) -> Dict[str, Any]:
        data = self._run_json(["listswaps", "--json"])
        swaps = self._extract_swap_list(data)
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
        raw = self._run(["refundswap", str(swap_id), str(dest)], timeout=max(self.cfg.timeout_seconds, 120))
        return {"swap_id": swap_id, "destination": dest, "result_raw": raw}

    def claim(self, swap_ids: List[str], destination: Optional[str] = None) -> Dict[str, Any]:
        ids = [str(x) for x in (swap_ids or []) if str(x).strip()]
        if not ids:
            raise BoltzCliError("swap_ids is required")
        dest = destination or "wallet"
        raw = self._run(["claimswaps", str(dest)] + ids, timeout=max(self.cfg.timeout_seconds, 120))
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

        args: List[str] = ["createchainswap", "--json", "--from-wallet", self._resolve_wallet_name(from_cur)]
        if to_address:
            args.extend(["--to-address", str(to_address)])
        else:
            args.extend(["--to-wallet", self._resolve_wallet_name(to_cur)])
        args.append(str(amount_sats))
        result = self._run_json(args, timeout=max(self.cfg.timeout_seconds, 180))
        return {
            "status": "accepted",
            "amount_sats": amount_sats,
            "from_currency": from_cur,
            "to_currency": to_cur,
            "quote": quote,
            "budget_check": budget_check,
            "result": result,
        }

    def withdraw(self, amount_sats: Optional[int], destination: str, currency: Optional[str] = None,
                 sat_per_vbyte: Optional[int] = None, sweep: bool = False) -> Dict[str, Any]:
        if not destination:
            raise BoltzCliError("destination is required")
        cur = self._norm_currency(currency, "LBTC")
        wallet_name = self._resolve_wallet_name(cur)
        amt = 0 if sweep else int(amount_sats or 0)
        if not sweep and amt <= 0:
            raise BoltzCliError("amount_sats must be > 0 unless sweep=true")
        args: List[str] = ["wallet", "send", wallet_name, destination, str(amt)]
        if sat_per_vbyte is not None:
            args.extend(["--sat-per-vbyte", str(int(sat_per_vbyte))])
        if sweep:
            args.append("--sweep")
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

    def backup(self) -> Dict[str, Any]:
        mnemonic = self._run(["swapmnemonic", "get"])  # plaintext by design (caller warning)
        wallets = self.wallet_balances()
        pending = self._listswaps_json(pending_only=True)
        return {
            "swap_mnemonic": mnemonic.strip(),
            "wallets": wallets.get("wallets", []),
            "pending_swaps": self._extract_swap_list(pending),
            "warning": "Contains plaintext swap mnemonic. Store securely.",
        }

    def backup_verify(self, swap_mnemonic: str) -> Dict[str, Any]:
        current = self._run(["swapmnemonic", "get"]).strip()
        provided = " ".join(str(swap_mnemonic or "").split())
        actual = " ".join(current.split())
        return {
            "matches": provided == actual,
            "provided_word_count": len(provided.split()),
            "current_word_count": len(actual.split()),
        }
