"""
Boltz swap module for cl-revenue-ops.

Uses the local boltzcli binary (talking to co-located boltzd gRPC daemon)
for all swap operations. boltzd handles crypto, claim/refund logic, status
tracking, and wallet management.

Implements:
- Loop-out (reverse swap): Lightning -> on-chain BTC or LBTC wallet
- Loop-in (submarine swap): on-chain BTC or LBTC wallet -> Lightning

Supports both BTC (on-chain) and LBTC (Liquid) as the swap currency.
LBTC swaps route through boltzd's Liquid wallet, providing ~50-90% lower
fees. When LBTC is configured but wallet balance is insufficient for a
loop-in, auto-fallback to BTC occurs.

Tracks costs and swap state in SQLite for P&L integration.
"""

import time
import json
import subprocess
from uuid import uuid4
from typing import Dict, Any, Optional, List, Tuple


class BoltzSwapManager:
    DEFAULT_SWAP_DAILY_BUDGET_SATS = 50_000
    DEFAULT_SWAP_MAX_FEE_PPM = 5_000
    DEFAULT_SWAP_MIN_AMOUNT_SATS = 100_000
    DEFAULT_SWAP_MAX_AMOUNT_SATS = 10_000_000
    DEFAULT_BUDGET_RESERVATION_TTL_SECS = 86_400

    def __init__(self, database, safe_plugin, config):
        self.db = database
        self.plugin = safe_plugin
        self.rpc = safe_plugin.rpc
        self.config = config
        self._wallet_names: Dict[str, str] = {}  # currency -> wallet name cache

        self._ensure_tables()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str, level: str = "info"):
        try:
            self.plugin.log(f"Boltz: {msg}", level=level)
        except Exception:
            pass

    def _cfg_int(self, key: str, default: int) -> int:
        value: Any = default
        if isinstance(self.config, dict):
            value = self.config.get(key, default)
        else:
            value = getattr(self.config, key, default)
        try:
            return int(value)
        except Exception:
            return default

    def _cfg_str(self, key: str, default: str) -> str:
        if isinstance(self.config, dict):
            return str(self.config.get(key, default))
        return str(getattr(self.config, key, default))

    def _now_ts(self) -> int:
        return int(time.time())

    def _cli_token(self, value: Any, field_name: str, allow_wallet_keyword: bool = False) -> Tuple[Optional[str], Optional[str]]:
        """
        Validate an untrusted token passed to boltzcli.

        Prevents option-style values (e.g. '--foo') and control/whitespace chars
        that can alter CLI parsing semantics for positional parameters.
        """
        if not isinstance(value, str):
            return None, f"{field_name} must be a string"

        token = value.strip()
        if not token:
            return None, f"{field_name} cannot be empty"
        if token.startswith("-"):
            return None, f"{field_name} cannot start with '-'"
        if any(ch.isspace() for ch in token):
            return None, f"{field_name} cannot contain whitespace"
        if any(ch in token for ch in ("\n", "\r", "\x00")):
            return None, f"{field_name} contains invalid control characters"
        if allow_wallet_keyword and token == "wallet":
            return token, None
        return token, None

    # ------------------------------------------------------------------
    # Currency & wallet helpers
    # ------------------------------------------------------------------

    def _get_currency(self, override: Optional[str] = None) -> str:
        """Get swap currency, normalized to lowercase.

        Args:
            override: If provided, use this instead of config value.
        """
        raw = override if override else self._cfg_str("swap_currency", "lbtc")
        return raw.lower() if raw.lower() in ("btc", "lbtc") else "lbtc"

    def _get_wallet_name(self, currency: str) -> str:
        """Get the boltzd wallet name for a given currency.

        Discovers wallet names dynamically from ``wallet list`` and caches
        them.  For LBTC, auto-creates the wallet if none exists.

        Raises RuntimeError if no wallet is found (BTC) or creation fails (LBTC).
        """
        cur_upper = currency.upper()
        if cur_upper in self._wallet_names:
            return self._wallet_names[cur_upper]

        wallets = self._run_boltzcli("wallet", "list")
        for w in wallets.get("wallets", []):
            if w.get("currency", "").upper() == cur_upper:
                self._wallet_names[cur_upper] = w["name"]
                return self._wallet_names[cur_upper]

        # No wallet found for this currency
        if cur_upper == "LBTC":
            # Auto-create LBTC wallet
            self._run_boltzcli_raw("wallet", "create", "liquid", "LBTC")
            self._log("Created LBTC wallet 'liquid' in boltzd")
            self._wallet_names[cur_upper] = "liquid"
            return "liquid"

        raise RuntimeError(f"No {cur_upper} wallet found in boltzd")

    # Backward-compatible alias
    def _ensure_lbtc_wallet(self) -> str:
        """Ensure an LBTC wallet exists in boltzd. Returns wallet name."""
        return self._get_wallet_name("LBTC")

    def _get_lbtc_balance(self, wallet_name: str) -> int:
        """Get confirmed LBTC balance in sats."""
        wallets = self._run_boltzcli("wallet", "list")
        for w in wallets.get("wallets", []):
            if w.get("name") == wallet_name:
                return int(w.get("balance", {}).get("confirmed", 0))
        return 0

    def get_wallet_balances(self) -> Dict[str, Any]:
        """Get all boltzd wallet balances (BTC and LBTC)."""
        return self._run_boltzcli("wallet", "list")

    def wallet_receive(self, currency: str = "lbtc") -> Dict[str, Any]:
        """Get a deposit address for a boltzd wallet.

        Args:
            currency: 'btc' or 'lbtc'. Defaults to 'lbtc'.

        Returns deposit address for receiving funds into the boltzd wallet.
        """
        cur = currency.lower() if currency else "lbtc"
        try:
            wallet_name = self._get_wallet_name(cur)
        except RuntimeError as e:
            return {"error": f"{cur.upper()} wallet unavailable: {e}"}

        try:
            result = self._run_boltzcli_auto("wallet", "receive", wallet_name)
        except RuntimeError as e:
            return {"error": str(e)}

        self._record_audit_event(
            "wallet_receive_address",
            f"Generated deposit address for {cur} wallet '{wallet_name}'",
            details={"currency": cur, "wallet": wallet_name},
        )
        return {"currency": cur, "wallet": wallet_name, "result": result}

    def wallet_send(
        self,
        destination: str,
        amount_sats: int,
        currency: str = "lbtc",
        sat_per_vbyte: Optional[int] = None,
        sweep: bool = False,
    ) -> Dict[str, Any]:
        """Send funds from a boltzd wallet to an external address.

        Args:
            destination: Target address (BTC or Liquid address).
            amount_sats: Amount in sats to send (ignored if sweep=True).
            currency: 'btc' or 'lbtc'. Defaults to 'lbtc'.
            sat_per_vbyte: Optional fee rate override.
            sweep: If True, send entire wallet balance.
        """
        destination_token, destination_err = self._cli_token(destination, "destination")
        if destination_err:
            return {"error": destination_err}
        if not sweep and int(amount_sats) <= 0:
            return {"error": "amount_sats must be > 0 unless sweep=true"}
        if sat_per_vbyte is not None and int(sat_per_vbyte) <= 0:
            return {"error": "sat_per_vbyte must be > 0"}

        cur = currency.lower() if currency else "lbtc"
        try:
            wallet_name = self._get_wallet_name(cur)
        except RuntimeError as e:
            return {"error": f"{cur.upper()} wallet unavailable: {e}"}

        cmd_args = ["wallet", "send", wallet_name, destination_token, str(amount_sats)]
        if sat_per_vbyte:
            cmd_args.extend(["--sat-per-vbyte", str(sat_per_vbyte)])
        if sweep:
            cmd_args.append("--sweep")

        try:
            result = self._run_boltzcli_auto(*cmd_args)
        except RuntimeError as e:
            self._record_audit_event(
                "wallet_send_failed", str(e),
                level="error",
                details={"destination": destination_token, "amount_sats": amount_sats, "currency": cur},
            )
            return {"error": str(e)}

        self._record_audit_event(
            "wallet_send_initiated",
            f"Wallet send: {amount_sats} sats ({cur}) to {destination_token}"
            + (" [sweep]" if sweep else ""),
            details={
                "destination": destination_token,
                "amount_sats": amount_sats,
                "currency": cur,
                "sweep": sweep,
            },
        )
        return {"status": "sent", "currency": cur, "amount_sats": amount_sats, "destination": destination_token, "result": result}

    def create_chain_swap(
        self,
        amount_sats: int,
        from_currency: str = "lbtc",
        to_currency: str = "btc",
        to_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a chain swap between BTC and LBTC via Boltz.

        This converts between BTC and LBTC on-chain (no Lightning involved).
        Primary use case: exit LBTC back to BTC when needed.

        Args:
            amount_sats: Amount to swap.
            from_currency: Source currency ('btc' or 'lbtc').
            to_currency: Destination currency ('btc' or 'lbtc').
            to_address: Optional destination address (otherwise uses boltzd wallet).
        """
        from_cur = from_currency.lower()
        to_cur = to_currency.lower()

        if from_cur == to_cur:
            return {"error": f"Cannot chain swap {from_cur} to itself"}
        if from_cur not in ("btc", "lbtc") or to_cur not in ("btc", "lbtc"):
            return {"error": f"Invalid currencies: {from_cur} -> {to_cur}"}
        if to_address:
            to_address, address_err = self._cli_token(to_address, "to_address")
            if address_err:
                return {"error": address_err}

        # Get quote for budget check
        try:
            q_args = ["quote", "--send", str(amount_sats),
                      "--from", from_cur.upper(), "--to", to_cur.upper(), "chain"]
            q_data = self._run_boltzcli(*q_args)
            q = self._parse_quote(q_data, amount_sats, "chain")
        except RuntimeError as e:
            return {"error": f"Chain swap quote failed: {e}"}

        reservation_id, budget_err = self._reserve_budget(q["total_fee_sats"], q["fee_ppm"], amount_sats)
        if budget_err:
            self._record_audit_event(
                "chain_swap_budget_blocked", budget_err,
                level="warn",
                details={"amount_sats": amount_sats, "from": from_cur, "to": to_cur},
            )
            return {"error": budget_err, "budget": self.get_budget_status()}

        # Resolve source wallet
        try:
            from_wallet = self._get_wallet_name(from_cur)
        except RuntimeError as e:
            self._release_budget_reservation(reservation_id, "chain_swap_wallet_unavailable")
            return {"error": f"{from_cur.upper()} wallet unavailable: {e}"}

        # Check source balance for LBTC (with fee margin)
        if from_cur == "lbtc":
            balance = self._get_lbtc_balance(from_wallet)
            required = int(amount_sats * 1.02)
            if balance < required:
                self._release_budget_reservation(reservation_id, "chain_swap_insufficient_lbtc_balance")
                return {
                    "error": f"LBTC wallet balance {balance} < {required} "
                    f"(amount {amount_sats} + 2% fee margin)",
                    "balance": balance,
                }

        # Build command — --to-wallet and --to-address are mutually exclusive
        cmd_args = [
            "createchainswap", str(amount_sats),
            "--from-wallet", from_wallet,
        ]
        if to_address:
            cmd_args.extend(["--to-address", to_address])
        else:
            try:
                to_wallet = self._get_wallet_name(to_cur)
            except RuntimeError as e:
                self._release_budget_reservation(reservation_id, "chain_swap_wallet_unavailable")
                return {"error": f"{to_cur.upper()} wallet unavailable: {e}"}
            cmd_args.extend(["--to-wallet", to_wallet])

        try:
            result = self._run_boltzcli(*cmd_args)
        except RuntimeError as e:
            self._release_budget_reservation(reservation_id, "chain_swap_create_failed")
            self._record_audit_event(
                "chain_swap_failed", str(e),
                level="error",
                details={
                    "amount_sats": amount_sats,
                    "from": from_cur, "to": to_cur,
                },
            )
            return {"error": str(e)}

        swap_id = result.get("id", "")
        if not swap_id:
            self._release_budget_reservation(reservation_id, "chain_swap_missing_id")
            return {"error": f"No swap ID in response: {result}"}

        fees = self._extract_fees_from_response(result, amount_sats, q)
        self._attach_budget_reservation(
            reservation_id,
            swap_id,
            max(q["total_fee_sats"], fees["total_fee_sats"]),
        )

        # Record in DB as a chain swap
        if swap_id:
            rec = {
                "id": swap_id,
                "created_at": self._now_ts(),
                "updated_at": self._now_ts(),
                "swap_type": "chain",
                "amount_sats": amount_sats,
                "received_sats": fees["received_sats"],
                "boltz_fee_sats": fees["boltz_fee_sats"],
                "network_fee_sats": fees["network_fee_sats"],
                "total_fee_sats": fees["total_fee_sats"],
                "fee_ppm": fees["fee_ppm"],
                "status": "created",
                "error": None,
                "target_channel_id": None,
                "target_peer_id": None,
                "currency": f"{from_cur}->{to_cur}",
            }
            self._record_swap(rec)

        self._record_audit_event(
            "chain_swap_created",
            f"Chain swap: {amount_sats} sats {from_cur} -> {to_cur}",
            swap_id=swap_id,
            details={
                "amount_sats": amount_sats,
                "from": from_cur, "to": to_cur,
                "to_address": to_address,
            },
        )

        return {
            "status": "created",
            "swap_id": swap_id,
            "amount_sats": amount_sats,
            "from_currency": from_cur,
            "to_currency": to_cur,
            "boltzd_response": result,
        }

    # ------------------------------------------------------------------
    # boltzcli interface
    # ------------------------------------------------------------------

    # Wallet subcommands that accept --json (others like send/receive don't)
    _WALLET_JSON_SUBCMDS = frozenset({"list", "balances", "transactions"})

    def _run_boltzcli(self, *args: str, timeout: int = 60) -> Dict[str, Any]:
        """
        Run a boltzcli command with --json flag and return parsed output.

        --json is a per-command flag in boltzcli v2.x, so it's inserted
        after the command/subcommand name, not as a global flag.

        For wallet subcommands, --json goes after the subcommand name
        (e.g., ``wallet list --json``), not after ``wallet``.

        Raises RuntimeError on non-zero exit or unparseable output.
        """
        if not args:
            cmd = ["boltzcli", "--json"]
        elif (args[0] == "wallet" and len(args) > 1
              and args[1] in self._WALLET_JSON_SUBCMDS):
            # wallet subcommands: --json belongs to the subcommand
            cmd = ["boltzcli", "wallet", args[1], "--json", *args[2:]]
        else:
            cmd = ["boltzcli", args[0], "--json", *args[1:]]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout,
                text=True,
            )
        except FileNotFoundError:
            raise RuntimeError("boltzcli not found in PATH")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"boltzcli timed out after {timeout}s: {' '.join(cmd)}")

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(f"boltzcli error (exit {result.returncode}): {stderr}")

        stdout = (result.stdout or "").strip()
        if not stdout:
            return {}
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            raise RuntimeError(f"boltzcli returned non-JSON output: {stdout[:200]}")

    def _run_boltzcli_auto(self, *args: str, timeout: int = 60) -> Dict[str, Any]:
        """
        Run boltzcli without --json and attempt to parse JSON from output.

        Use this for commands that don't support --json but may still
        return JSON natively (e.g., getinfo, wallet receive).
        Falls back to {"raw_output": ...} if output isn't JSON.

        Raises RuntimeError on non-zero exit.
        """
        raw = self._run_boltzcli_raw(*args, timeout=timeout)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw_output": raw}

    def _run_boltzcli_raw(self, *args: str, timeout: int = 60) -> str:
        """
        Run boltzcli without --json and return raw stdout.
        Used for commands that may not support --json.
        """
        cmd = ["boltzcli", *args]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout,
                text=True,
            )
        except FileNotFoundError:
            raise RuntimeError("boltzcli not found in PATH")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"boltzcli timed out after {timeout}s: {' '.join(cmd)}")

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(f"boltzcli error (exit {result.returncode}): {stderr}")

        return (result.stdout or "").strip()

    # ------------------------------------------------------------------
    # Database schema
    # ------------------------------------------------------------------

    def _table_exists(self, conn, table_name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        return bool(row)

    def _ensure_tables(self):
        conn = self.db._get_connection()

        # Rename old table if it exists (one-time migration)
        if self._table_exists(conn, "boltz_swaps") and not self._table_exists(conn, "boltz_swaps_v1"):
            try:
                conn.execute("""
                    ALTER TABLE boltz_swaps RENAME TO boltz_swaps_v1
                """)
                self._log("Migrated old boltz_swaps table to boltz_swaps_v1")
            except Exception:
                pass

        conn.execute("""
            CREATE TABLE IF NOT EXISTS boltz_swaps (
                id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                swap_type TEXT NOT NULL,
                amount_sats INTEGER NOT NULL,
                received_sats INTEGER NOT NULL DEFAULT 0,
                boltz_fee_sats INTEGER NOT NULL DEFAULT 0,
                network_fee_sats INTEGER NOT NULL DEFAULT 0,
                total_fee_sats INTEGER NOT NULL DEFAULT 0,
                fee_ppm INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                error TEXT,
                target_channel_id TEXT,
                target_peer_id TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_boltz_swaps_status ON boltz_swaps(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_boltz_swaps_created ON boltz_swaps(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_boltz_swaps_type ON boltz_swaps(swap_type)")

        # Keep existing audit log table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS boltz_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                swap_id TEXT,
                event_type TEXT NOT NULL,
                level TEXT NOT NULL DEFAULT 'info',
                message TEXT NOT NULL,
                details_json TEXT,
                created_at INTEGER NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_boltz_audit_swap_created ON boltz_audit_log(swap_id, created_at)")

        # swap_costs table for P&L integration
        conn.execute("""
            CREATE TABLE IF NOT EXISTS swap_costs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                swap_id TEXT NOT NULL,
                channel_id TEXT,
                peer_id TEXT,
                cost_sats INTEGER NOT NULL,
                amount_sats INTEGER NOT NULL,
                swap_type TEXT NOT NULL,
                timestamp INTEGER NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_swap_costs_timestamp ON swap_costs(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_swap_costs_channel ON swap_costs(channel_id)")

        # Migration: add currency column to boltz_swaps (existing rows default to 'btc')
        try:
            conn.execute("ALTER TABLE boltz_swaps ADD COLUMN currency TEXT NOT NULL DEFAULT 'btc'")
        except Exception:
            pass  # Column already exists

        # Migration: add currency column to swap_costs
        try:
            conn.execute("ALTER TABLE swap_costs ADD COLUMN currency TEXT NOT NULL DEFAULT 'btc'")
        except Exception:
            pass  # Column already exists

        conn.execute("""
            CREATE TABLE IF NOT EXISTS swap_budget_reservations (
                reservation_id TEXT PRIMARY KEY,
                swap_id TEXT UNIQUE,
                fee_sats INTEGER NOT NULL,
                state TEXT NOT NULL DEFAULT 'active',
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                note TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_swap_budget_reservations_state ON swap_budget_reservations(state, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_swap_budget_reservations_swap_id ON swap_budget_reservations(swap_id)")

        self._backfill_legacy_swaps(conn)

    def _backfill_legacy_swaps(self, conn) -> None:
        """
        Copy legacy rows into the current swap schema so in-flight swaps
        are still monitored after migration.
        """
        if not self._table_exists(conn, "boltz_swaps_v1"):
            return

        cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(boltz_swaps_v1)").fetchall()
        }
        if "id" not in cols:
            return

        rows = conn.execute("SELECT * FROM boltz_swaps_v1").fetchall()
        if not rows:
            return

        def _pick_int(data: Dict[str, Any], keys: List[str], default: int = 0) -> int:
            for k in keys:
                if k in data and data[k] is not None:
                    try:
                        return int(data[k])
                    except Exception:
                        continue
            return default

        swap_type_map = {
            "loop_out": "reverse",
            "loop-out": "reverse",
            "loop_in": "submarine",
            "loop-in": "submarine",
            "loopin": "submarine",
        }

        migrated = 0
        now = self._now_ts()

        for row in rows:
            data = dict(row)
            swap_id = str(data.get("id", "")).strip()
            if not swap_id:
                continue

            exists = conn.execute(
                "SELECT 1 FROM boltz_swaps WHERE id = ?",
                (swap_id,),
            ).fetchone()
            if exists:
                continue

            raw_type = str(data.get("swap_type", "") or "").lower()
            swap_type = swap_type_map.get(raw_type, raw_type or "legacy")

            amount_sats = _pick_int(data, ["amount_sats", "invoice_amount_sats", "onchain_amount_sats"], 0)
            boltz_fee_sats = _pick_int(data, ["boltz_fee_sats"], 0)
            network_fee_sats = _pick_int(
                data,
                ["network_fee_sats"],
                _pick_int(data, ["miner_fee_lockup_sats"], 0) + _pick_int(data, ["miner_fee_claim_sats"], 0),
            )
            total_fee_sats = _pick_int(data, ["total_fee_sats", "total_cost_sats"], boltz_fee_sats + network_fee_sats)
            if total_fee_sats <= 0 and (boltz_fee_sats or network_fee_sats):
                total_fee_sats = boltz_fee_sats + network_fee_sats

            fee_ppm = _pick_int(
                data,
                ["fee_ppm", "cost_ppm"],
                int(total_fee_sats * 1_000_000 / amount_sats) if amount_sats > 0 else 0,
            )
            received_sats = _pick_int(data, ["received_sats"], max(0, amount_sats - total_fee_sats))

            currency = str(data.get("currency", "") or "").lower()
            if not currency:
                address = str(data.get("address", "") or "").lower()
                currency = "lbtc" if address.startswith("lq") else "btc"

            created_at = _pick_int(data, ["created_at"], now)
            updated_at = _pick_int(data, ["updated_at"], created_at)

            conn.execute(
                """
                INSERT OR IGNORE INTO boltz_swaps
                (id, created_at, updated_at, swap_type, amount_sats, received_sats,
                 boltz_fee_sats, network_fee_sats, total_fee_sats, fee_ppm, status,
                 error, target_channel_id, target_peer_id, currency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    swap_id,
                    created_at,
                    updated_at,
                    swap_type,
                    amount_sats,
                    received_sats,
                    boltz_fee_sats,
                    network_fee_sats,
                    total_fee_sats,
                    fee_ppm,
                    str(data.get("status", "created") or "created"),
                    data.get("error"),
                    data.get("target_channel_id"),
                    data.get("target_peer_id"),
                    currency,
                ),
            )
            migrated += 1

        if migrated:
            self._log(f"Backfilled {migrated} legacy Boltz swap rows into current schema", level="warn")

    def _record_swap(self, rec: Dict[str, Any]):
        conn = self.db._get_connection()
        fields = [
            "id", "created_at", "updated_at", "swap_type",
            "amount_sats", "received_sats", "boltz_fee_sats",
            "network_fee_sats", "total_fee_sats", "fee_ppm",
            "status", "error", "target_channel_id", "target_peer_id",
            "currency",
        ]
        values = [rec.get(f) for f in fields]
        placeholders = ",".join(["?"] * len(fields))
        conn.execute(
            f"INSERT OR REPLACE INTO boltz_swaps ({','.join(fields)}) VALUES ({placeholders})",
            values,
        )

    def _get_swap(self, swap_id: str) -> Optional[Dict[str, Any]]:
        conn = self.db._get_connection()
        row = conn.execute(
            "SELECT * FROM boltz_swaps WHERE id = ?", (swap_id,)
        ).fetchone()
        if not row:
            return None
        return dict(row)

    def _record_audit_event(
        self,
        event_type: str,
        message: str,
        swap_id: Optional[str] = None,
        level: str = "info",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        conn = self.db._get_connection()
        now = self._now_ts()
        details_json = None
        if details is not None:
            try:
                details_json = json.dumps(details, sort_keys=True, separators=(",", ":"))
            except Exception:
                details_json = json.dumps({"details": str(details)})
        conn.execute(
            """
            INSERT INTO boltz_audit_log
            (swap_id, event_type, level, message, details_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (swap_id, event_type, level, message, details_json, now),
        )
        self._log(f"{event_type}: {message} (swap_id={swap_id})", level=level)

    def _record_swap_cost(
        self,
        swap_id: str,
        cost_sats: int,
        amount_sats: int,
        swap_type: str,
        channel_id: Optional[str] = None,
        peer_id: Optional[str] = None,
        currency: str = "btc",
    ) -> None:
        conn = self.db._get_connection()
        conn.execute(
            """
            INSERT INTO swap_costs
            (swap_id, channel_id, peer_id, cost_sats, amount_sats, swap_type, timestamp, currency)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (swap_id, channel_id, peer_id, cost_sats, amount_sats, swap_type, self._now_ts(), currency),
        )

    # ------------------------------------------------------------------
    # Budget enforcement
    # ------------------------------------------------------------------

    def _get_daily_completed_swap_fee_spend(self) -> int:
        """Sum completed swap fees in the rolling 24h window."""
        conn = self.db._get_connection()
        since = self._now_ts() - 86400
        row = conn.execute(
            """
            SELECT COALESCE(SUM(total_fee_sats), 0) AS total
            FROM boltz_swaps
            WHERE updated_at >= ? AND status = 'completed'
            """,
            (since,),
        ).fetchone()
        return int(row["total"]) if row else 0

    def _get_daily_reserved_swap_fee_spend(self) -> int:
        """Sum active budget reservations in the rolling 24h window."""
        conn = self.db._get_connection()
        since = self._now_ts() - 86400
        row = conn.execute(
            """
            SELECT COALESCE(SUM(fee_sats), 0) AS total
            FROM swap_budget_reservations
            WHERE created_at >= ? AND state = 'active'
            """,
            (since,),
        ).fetchone()
        return int(row["total"]) if row else 0

    def _get_daily_swap_fee_spend(self) -> int:
        """Effective 24h spend = completed fees + active reservations."""
        return self._get_daily_completed_swap_fee_spend() + self._get_daily_reserved_swap_fee_spend()

    def _check_budget_limits(self, fee_ppm: int, amount_sats: int) -> Optional[str]:
        """Validate static per-swap budget limits."""
        max_fee_ppm = self._cfg_int("swap_max_fee_ppm", self.DEFAULT_SWAP_MAX_FEE_PPM)
        min_amount = self._cfg_int("swap_min_amount_sats", self.DEFAULT_SWAP_MIN_AMOUNT_SATS)
        max_amount = self._cfg_int("swap_max_amount_sats", self.DEFAULT_SWAP_MAX_AMOUNT_SATS)

        if amount_sats < min_amount:
            return f"Amount {amount_sats} below minimum {min_amount} sats"
        if amount_sats > max_amount:
            return f"Amount {amount_sats} above maximum {max_amount} sats"
        if fee_ppm > max_fee_ppm:
            return f"Fee rate {fee_ppm} ppm exceeds maximum {max_fee_ppm} ppm"
        return None

    def _check_budget(self, quoted_fee_sats: int, fee_ppm: int, amount_sats: int) -> Optional[str]:
        """
        Check swap budget constraints. Returns error string if blocked, None if OK.
        """
        self._cleanup_stale_budget_reservations()
        daily_budget = self._cfg_int("swap_daily_budget_sats", self.DEFAULT_SWAP_DAILY_BUDGET_SATS)
        static_err = self._check_budget_limits(fee_ppm, amount_sats)
        if static_err:
            return static_err

        daily_spent = self._get_daily_swap_fee_spend()
        if daily_spent + quoted_fee_sats > daily_budget:
            return (
                f"Daily swap budget exceeded: {daily_spent} + {quoted_fee_sats} "
                f"> {daily_budget} sats"
            )

        return None

    def _cleanup_stale_budget_reservations(self, ttl_seconds: Optional[int] = None) -> int:
        """Release stale active reservations to avoid orphan budget locks."""
        conn = self.db._get_connection()
        ttl = int(ttl_seconds or self.DEFAULT_BUDGET_RESERVATION_TTL_SECS)
        cutoff = self._now_ts() - max(1, ttl)
        stale = conn.execute(
            """
            SELECT reservation_id
            FROM swap_budget_reservations
            WHERE state = 'active' AND created_at < ?
            """,
            (cutoff,),
        ).fetchall()
        if not stale:
            return 0

        now = self._now_ts()
        conn.execute(
            """
            UPDATE swap_budget_reservations
            SET state = 'released', updated_at = ?, note = 'stale_timeout'
            WHERE state = 'active' AND created_at < ?
            """,
            (now, cutoff),
        )
        return len(stale)

    def _reserve_budget(self, quoted_fee_sats: int, fee_ppm: int, amount_sats: int) -> Tuple[Optional[str], Optional[str]]:
        """
        Atomically reserve daily budget before submitting a swap.
        Returns (reservation_id, error).
        """
        static_err = self._check_budget_limits(fee_ppm, amount_sats)
        if static_err:
            return None, static_err

        quoted_fee = max(0, int(quoted_fee_sats))
        daily_budget = self._cfg_int("swap_daily_budget_sats", self.DEFAULT_SWAP_DAILY_BUDGET_SATS)
        reservation_id = f"swap-rsv-{uuid4().hex}"
        since = self._now_ts() - 86400
        now = self._now_ts()

        self._cleanup_stale_budget_reservations()
        conn = self.db._get_connection()
        savepoint_name: Optional[str] = None
        try:
            if getattr(conn, "in_transaction", False):
                savepoint_name = f"swap_budget_{uuid4().hex[:12]}"
                conn.execute(f"SAVEPOINT {savepoint_name}")
            else:
                conn.execute("BEGIN IMMEDIATE")
            completed_row = conn.execute(
                """
                SELECT COALESCE(SUM(total_fee_sats), 0) AS total
                FROM boltz_swaps
                WHERE updated_at >= ? AND status = 'completed'
                """,
                (since,),
            ).fetchone()
            reserved_row = conn.execute(
                """
                SELECT COALESCE(SUM(fee_sats), 0) AS total
                FROM swap_budget_reservations
                WHERE created_at >= ? AND state = 'active'
                """,
                (since,),
            ).fetchone()

            completed = int(completed_row["total"]) if completed_row else 0
            reserved = int(reserved_row["total"]) if reserved_row else 0
            projected = completed + reserved + quoted_fee
            if projected > daily_budget:
                if savepoint_name:
                    conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                else:
                    conn.execute("ROLLBACK")
                return None, (
                    f"Daily swap budget exceeded: {completed} (completed) + "
                    f"{reserved} (reserved) + {quoted_fee} > {daily_budget} sats"
                )

            conn.execute(
                """
                INSERT INTO swap_budget_reservations
                (reservation_id, swap_id, fee_sats, state, created_at, updated_at, note)
                VALUES (?, NULL, ?, 'active', ?, ?, 'reserved_pre_create')
                """,
                (reservation_id, quoted_fee, now, now),
            )
            if savepoint_name:
                conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
            else:
                conn.execute("COMMIT")
            return reservation_id, None
        except Exception as e:
            try:
                if savepoint_name:
                    conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                    conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                else:
                    conn.execute("ROLLBACK")
            except Exception:
                pass
            return None, f"Budget reservation failed: {e}"

    def _attach_budget_reservation(self, reservation_id: Optional[str], swap_id: str, fee_sats: int) -> None:
        """Attach a pre-create reservation to a concrete swap ID."""
        if not reservation_id or not swap_id:
            return
        conn = self.db._get_connection()
        now = self._now_ts()
        conn.execute(
            """
            UPDATE swap_budget_reservations
            SET swap_id = ?, fee_sats = ?, updated_at = ?, note = 'attached_to_swap'
            WHERE reservation_id = ? AND state = 'active'
            """,
            (swap_id, max(0, int(fee_sats)), now, reservation_id),
        )

    def _release_budget_reservation(self, reservation_id: Optional[str], note: str) -> None:
        """Release an active reservation by reservation ID."""
        if not reservation_id:
            return
        conn = self.db._get_connection()
        conn.execute(
            """
            UPDATE swap_budget_reservations
            SET state = 'released', updated_at = ?, note = ?
            WHERE reservation_id = ? AND state = 'active'
            """,
            (self._now_ts(), note, reservation_id),
        )

    def _finalize_budget_reservation_for_swap(self, swap_id: str, final_fee_sats: Optional[int] = None) -> None:
        """Finalize reservation when swap reaches completed state."""
        if not swap_id:
            return
        conn = self.db._get_connection()
        now = self._now_ts()
        if final_fee_sats is None:
            conn.execute(
                """
                UPDATE swap_budget_reservations
                SET state = 'finalized', updated_at = ?, note = 'swap_completed'
                WHERE swap_id = ? AND state = 'active'
                """,
                (now, swap_id),
            )
        else:
            conn.execute(
                """
                UPDATE swap_budget_reservations
                SET state = 'finalized', fee_sats = ?, updated_at = ?, note = 'swap_completed'
                WHERE swap_id = ? AND state = 'active'
                """,
                (max(0, int(final_fee_sats)), now, swap_id),
            )

    def _release_budget_reservation_for_swap(self, swap_id: str, note: str) -> None:
        """Release reservation when swap fails/refunds before completion."""
        if not swap_id:
            return
        conn = self.db._get_connection()
        conn.execute(
            """
            UPDATE swap_budget_reservations
            SET state = 'released', updated_at = ?, note = ?
            WHERE swap_id = ? AND state = 'active'
            """,
            (self._now_ts(), note, swap_id),
        )

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current swap budget usage."""
        self._cleanup_stale_budget_reservations()
        daily_budget = self._cfg_int("swap_daily_budget_sats", self.DEFAULT_SWAP_DAILY_BUDGET_SATS)
        completed_spent = self._get_daily_completed_swap_fee_spend()
        reserved_spent = self._get_daily_reserved_swap_fee_spend()
        daily_spent = completed_spent + reserved_spent
        return {
            "daily_budget_sats": daily_budget,
            "daily_spent_sats": daily_spent,
            "daily_completed_sats": completed_spent,
            "daily_reserved_sats": reserved_spent,
            "daily_remaining_sats": max(0, daily_budget - daily_spent),
            "max_fee_ppm": self._cfg_int("swap_max_fee_ppm", self.DEFAULT_SWAP_MAX_FEE_PPM),
            "min_amount_sats": self._cfg_int("swap_min_amount_sats", self.DEFAULT_SWAP_MIN_AMOUNT_SATS),
            "max_amount_sats": self._cfg_int("swap_max_amount_sats", self.DEFAULT_SWAP_MAX_AMOUNT_SATS),
        }

    # ------------------------------------------------------------------
    # Quote helpers
    # ------------------------------------------------------------------

    def _parse_quote(self, data: Dict[str, Any], amount_sats: int, swap_type: str) -> Dict[str, Any]:
        """Parse boltzcli quote output into a standardized format."""
        # boltzcli v2.x quote fields: boltzFee, networkFee, sendAmount, receiveAmount
        boltz_fee = int(data.get("boltzFee", 0) or data.get("serviceFee", 0) or 0)
        network_fee = int(data.get("networkFee", 0) or data.get("onchainFee", 0) or 0)
        total_fee = boltz_fee + network_fee

        # Use boltzd's calculated receive amount if available
        received = int(data.get("receiveAmount", 0) or 0)
        if not received:
            received = max(0, amount_sats - total_fee)

        fee_ppm = int(total_fee * 1_000_000 / amount_sats) if amount_sats else 0

        return {
            "swap_type": swap_type,
            "amount_sats": amount_sats,
            "received_sats": received,
            "boltz_fee_sats": boltz_fee,
            "network_fee_sats": network_fee,
            "total_fee_sats": total_fee,
            "fee_ppm": fee_ppm,
            "raw": data,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _extract_fees_from_response(
        self, result: Dict[str, Any], amount_sats: int, quote: Dict[str, Any]
    ) -> Dict[str, int]:
        """Extract actual fees from swap creation response, falling back to quote.

        The creation response may include the actual negotiated fees which can
        differ from the quote (TOCTOU gap). Prefer creation response values.
        """
        # boltzd creation responses may include fee fields directly
        boltz_fee = int(result.get("serviceFee", 0) or result.get("boltzFee", 0) or 0)
        network_fee = int(result.get("onchainFee", 0) or result.get("networkFee", 0) or 0)

        # Fall back to quote if creation response didn't include fee breakdown
        if not boltz_fee and not network_fee:
            boltz_fee = quote.get("boltz_fee_sats", 0)
            network_fee = quote.get("network_fee_sats", 0)

        total_fee = boltz_fee + network_fee
        received = max(0, amount_sats - total_fee)
        fee_ppm = int(total_fee * 1_000_000 / amount_sats) if amount_sats else 0

        return {
            "boltz_fee_sats": boltz_fee,
            "network_fee_sats": network_fee,
            "total_fee_sats": total_fee,
            "received_sats": received,
            "fee_ppm": fee_ppm,
        }

    def _parse_swapinfo_raw(self, raw: str) -> Dict[str, Any]:
        """Parse raw swapinfo text output into a dict.

        swapinfo doesn't support --json, so we parse the key: value lines.
        """
        parsed: Dict[str, Any] = {}
        for line in raw.splitlines():
            line = line.strip()
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()
                if value:
                    parsed[key] = value
        return parsed

    def get_info(self) -> Dict[str, Any]:
        """Check boltzd connectivity."""
        return self._run_boltzcli_auto("getinfo")

    def quote_reverse(self, amount_sats: int, currency: Optional[str] = None) -> Dict[str, Any]:
        """Get reverse swap (loop-out) fee quote."""
        args = ["quote", "--send", str(amount_sats)]
        cur = self._get_currency(currency)
        if cur == "lbtc":
            args.extend(["--to", "LBTC"])
        args.append("reverse")
        data = self._run_boltzcli(*args)
        q = self._parse_quote(data, amount_sats, "reverse")
        q["currency"] = cur
        return q

    def quote_submarine(self, amount_sats: int, currency: Optional[str] = None) -> Dict[str, Any]:
        """Get submarine swap (loop-in) fee quote."""
        args = ["quote", "--receive", str(amount_sats)]
        cur = self._get_currency(currency)
        if cur == "lbtc":
            args.extend(["--from", "LBTC"])
        args.append("submarine")
        data = self._run_boltzcli(*args)
        q = self._parse_quote(data, amount_sats, "submarine")
        q["currency"] = cur
        return q

    def quote(self, amount_sats: int, swap_type: str = "reverse", currency: Optional[str] = None) -> Dict[str, Any]:
        """Get fee quote for a swap.

        Args:
            amount_sats: Swap amount.
            swap_type: 'reverse' or 'submarine'.
            currency: 'btc', 'lbtc', or 'both'. None = use config default.
        """
        if currency == "both":
            btc_q = self.quote(amount_sats, swap_type=swap_type, currency="btc")
            lbtc_q = self.quote(amount_sats, swap_type=swap_type, currency="lbtc")
            return {"btc": btc_q, "lbtc": lbtc_q}

        if swap_type == "submarine":
            q = self.quote_submarine(amount_sats, currency=currency)
        else:
            q = self.quote_reverse(amount_sats, currency=currency)
        q["budget"] = self.get_budget_status()
        return q

    def create_reverse_swap(
        self,
        amount_sats: int,
        address: Optional[str] = None,
        channel_id: Optional[str] = None,
        peer_id: Optional[str] = None,
        currency: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a reverse swap (loop-out: LN -> on-chain/LBTC).
        boltzd handles invoice payment, lockup monitoring, and claiming.

        When currency is 'lbtc' and no explicit address is given, funds are
        routed to the boltzd LBTC wallet (auto-created if needed).
        """
        cur = self._get_currency(currency)
        if address:
            address, address_err = self._cli_token(address, "address")
            if address_err:
                return {"error": address_err}
        if channel_id:
            channel_id, channel_err = self._cli_token(channel_id, "channel_id")
            if channel_err:
                return {"error": channel_err}

        # Get quote for budget check
        q = self.quote_reverse(amount_sats, currency=cur)
        if "error" in q:
            return q

        reservation_id, budget_err = self._reserve_budget(q["total_fee_sats"], q["fee_ppm"], amount_sats)
        if budget_err:
            self._record_audit_event(
                "reverse_swap_budget_blocked", budget_err,
                level="warn",
                details={"amount_sats": amount_sats, "fee_ppm": q["fee_ppm"], "currency": cur},
            )
            return {"error": budget_err, "budget": self.get_budget_status()}

        # Build command
        cmd_args = ["createreverseswap", cur, str(amount_sats)]
        if address:
            cmd_args.append(address)
        elif cur == "lbtc":
            # Route to LBTC wallet when no explicit address
            try:
                wallet_name = self._ensure_lbtc_wallet()
            except RuntimeError as e:
                self._release_budget_reservation(reservation_id, "reverse_swap_lbtc_wallet_unavailable")
                self._record_audit_event(
                    "lbtc_wallet_error", f"Failed to ensure LBTC wallet: {e}",
                    level="error",
                    details={"amount_sats": amount_sats, "operation": "reverse_swap"},
                )
                return {"error": f"LBTC wallet unavailable: {e}"}
            cmd_args.extend(["--to-wallet", wallet_name])
        if channel_id:
            cmd_args.extend(["--chan-id", channel_id])

        try:
            result = self._run_boltzcli(*cmd_args)
        except RuntimeError as e:
            self._release_budget_reservation(reservation_id, "reverse_swap_create_failed")
            self._record_audit_event(
                "reverse_swap_failed", str(e),
                level="error",
                details={"amount_sats": amount_sats, "currency": cur},
            )
            return {"error": str(e)}

        swap_id = result.get("id", "")
        if not swap_id:
            self._release_budget_reservation(reservation_id, "reverse_swap_missing_id")
            return {"error": f"No swap ID in response: {result}"}

        # Extract actual fees from creation response, fall back to quote (fix #4)
        fees = self._extract_fees_from_response(result, amount_sats, q)
        self._attach_budget_reservation(
            reservation_id,
            swap_id,
            max(q["total_fee_sats"], fees["total_fee_sats"]),
        )

        # Record swap
        rec = {
            "id": swap_id,
            "created_at": self._now_ts(),
            "updated_at": self._now_ts(),
            "swap_type": "reverse",
            "amount_sats": amount_sats,
            "received_sats": fees["received_sats"],
            "boltz_fee_sats": fees["boltz_fee_sats"],
            "network_fee_sats": fees["network_fee_sats"],
            "total_fee_sats": fees["total_fee_sats"],
            "fee_ppm": fees["fee_ppm"],
            "status": "created",
            "error": None,
            "target_channel_id": channel_id,
            "target_peer_id": peer_id,
            "currency": cur,
        }
        self._record_swap(rec)

        self._record_audit_event(
            "reverse_swap_created",
            f"Reverse swap created for {amount_sats} sats ({cur})",
            swap_id=swap_id,
            details={
                "amount_sats": amount_sats,
                "total_fee_sats": fees["total_fee_sats"],
                "fee_ppm": fees["fee_ppm"],
                "currency": cur,
                "address": address,
                "channel_id": channel_id,
            },
        )

        return {
            "status": "created",
            "swap_id": swap_id,
            "amount_sats": amount_sats,
            "received_sats": fees["received_sats"],
            "total_fee_sats": fees["total_fee_sats"],
            "fee_ppm": fees["fee_ppm"],
            "currency": cur,
            "budget": self.get_budget_status(),
            "boltzd_response": result,
        }

    def create_submarine_swap(
        self,
        amount_sats: int,
        channel_id: Optional[str] = None,
        peer_id: Optional[str] = None,
        currency: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a submarine swap (loop-in: on-chain/LBTC -> LN).
        boltzd handles invoice creation, on-chain funding, and claiming.

        When currency is 'lbtc', spends from the boltzd LBTC wallet.
        If the LBTC wallet has insufficient balance, auto-falls back to BTC.
        """
        if channel_id:
            channel_id, channel_err = self._cli_token(channel_id, "channel_id")
            if channel_err:
                return {"error": channel_err}
        if peer_id:
            peer_id, peer_err = self._cli_token(peer_id, "peer_id")
            if peer_err:
                return {"error": peer_err}

        cur = self._get_currency(currency)
        wallet_name = None
        fallback_used = False

        # For LBTC, check wallet balance and auto-fallback if insufficient
        if cur == "lbtc":
            try:
                wallet_name = self._ensure_lbtc_wallet()
                balance = self._get_lbtc_balance(wallet_name)
            except RuntimeError as e:
                self._record_audit_event(
                    "lbtc_wallet_error", f"Failed to access LBTC wallet: {e}",
                    level="error",
                    details={"amount_sats": amount_sats, "operation": "submarine_swap"},
                )
                self._log(f"LBTC wallet error, falling back to BTC: {e}", level="warn")
                cur = "btc"
                wallet_name = None
                balance = 0
                fallback_used = True

            if cur == "lbtc":
                # Include 2% fee margin so swap doesn't fail mid-flight
                required = int(amount_sats * 1.02)
                if balance < required:
                    self._log(
                        f"LBTC wallet balance {balance} < {required} "
                        f"(amount {amount_sats} + 2% fee margin), "
                        f"falling back to BTC for submarine swap",
                        level="warn",
                    )
                    cur = "btc"
                    wallet_name = None
                    fallback_used = True

        # Get quote for budget check (use actual currency after possible fallback)
        q = self.quote_submarine(amount_sats, currency=cur)
        if "error" in q:
            return q

        reservation_id, budget_err = self._reserve_budget(q["total_fee_sats"], q["fee_ppm"], amount_sats)
        if budget_err:
            self._record_audit_event(
                "submarine_swap_budget_blocked", budget_err,
                level="warn",
                details={"amount_sats": amount_sats, "fee_ppm": q["fee_ppm"], "currency": cur},
            )
            return {"error": budget_err, "budget": self.get_budget_status()}

        # Build command
        if cur == "lbtc":
            cmd_args = ["createswap", "--from-wallet", wallet_name, "--refund", "wallet", "lbtc", str(amount_sats)]
        else:
            cmd_args = ["createswap", "--refund", "wallet", "btc", str(amount_sats)]

        try:
            result = self._run_boltzcli(*cmd_args)
        except RuntimeError as e:
            self._release_budget_reservation(reservation_id, "submarine_swap_create_failed")
            self._record_audit_event(
                "submarine_swap_failed", str(e),
                level="error",
                details={"amount_sats": amount_sats, "currency": cur},
            )
            return {"error": str(e)}

        swap_id = result.get("id", "")
        if not swap_id:
            self._release_budget_reservation(reservation_id, "submarine_swap_missing_id")
            return {"error": f"No swap ID in response: {result}"}

        # Extract actual fees from creation response, fall back to quote (fix #4)
        fees = self._extract_fees_from_response(result, amount_sats, q)
        self._attach_budget_reservation(
            reservation_id,
            swap_id,
            max(q["total_fee_sats"], fees["total_fee_sats"]),
        )

        # Record swap
        rec = {
            "id": swap_id,
            "created_at": self._now_ts(),
            "updated_at": self._now_ts(),
            "swap_type": "submarine",
            "amount_sats": amount_sats,
            "received_sats": fees["received_sats"],
            "boltz_fee_sats": fees["boltz_fee_sats"],
            "network_fee_sats": fees["network_fee_sats"],
            "total_fee_sats": fees["total_fee_sats"],
            "fee_ppm": fees["fee_ppm"],
            "status": "created",
            "error": None,
            "target_channel_id": channel_id,
            "target_peer_id": peer_id,
            "currency": cur,
        }
        self._record_swap(rec)

        self._record_audit_event(
            "submarine_swap_created",
            f"Submarine swap created for {amount_sats} sats ({cur})"
            + (" [fallback from lbtc]" if fallback_used else ""),
            swap_id=swap_id,
            details={
                "amount_sats": amount_sats,
                "total_fee_sats": fees["total_fee_sats"],
                "fee_ppm": fees["fee_ppm"],
                "currency": cur,
                "fallback_used": fallback_used,
            },
        )

        resp = {
            "status": "created",
            "swap_id": swap_id,
            "amount_sats": amount_sats,
            "received_sats": fees["received_sats"],
            "total_fee_sats": fees["total_fee_sats"],
            "fee_ppm": fees["fee_ppm"],
            "currency": cur,
            "budget": self.get_budget_status(),
            "boltzd_response": result,
        }
        if fallback_used:
            resp["fallback"] = "lbtc_insufficient_balance"
        return resp

    # Backward-compatible aliases
    def loop_out(self, amount_sats: int, address: Optional[str] = None,
                 channel_id: Optional[str] = None, peer_id: Optional[str] = None,
                 currency: Optional[str] = None) -> Dict[str, Any]:
        """Execute a reverse swap (loop-out). Alias for create_reverse_swap."""
        return self.create_reverse_swap(amount_sats, address=address,
                                        channel_id=channel_id, peer_id=peer_id,
                                        currency=currency)

    def loop_in(self, amount_sats: int,
                channel_id: Optional[str] = None,
                peer_id: Optional[str] = None,
                currency: Optional[str] = None) -> Dict[str, Any]:
        """Execute a submarine swap (loop-in). Alias for create_submarine_swap."""
        if channel_id and peer_id:
            return {"error": "Provide either channel_id or peer_id, not both"}
        return self.create_submarine_swap(amount_sats, channel_id=channel_id,
                                          peer_id=peer_id, currency=currency)

    def get_swap_info(self, swap_id: str) -> Dict[str, Any]:
        """Get swap status from boltzd + local DB.

        swapinfo does NOT support --json, so we use raw output and parse it.
        """
        swap_id, swap_id_err = self._cli_token(swap_id, "swap_id")
        if swap_id_err:
            return {"error": swap_id_err}

        local = self._get_swap(swap_id)

        # swapinfo doesn't support --json — go straight to raw parsing (fix #5)
        remote: Dict[str, Any] = {}
        try:
            raw = self._run_boltzcli_raw("swapinfo", swap_id)
            remote = self._parse_swapinfo_raw(raw)
            remote["raw_output"] = raw
        except RuntimeError as e:
            remote = {"error": str(e)}

        # Sync status if remote has an update
        remote_status = remote.get("status")
        if local and remote_status and remote_status != local.get("status"):
            local["updated_at"] = self._now_ts()
            if self._is_completed_status(remote_status):
                local["status"] = "completed"
                self._record_swap(local)
                self._finalize_budget_reservation_for_swap(swap_id, final_fee_sats=local.get("total_fee_sats", 0))
                # Record cost if not already recorded
                self._maybe_record_completion_cost(local)
            elif self._is_failed_status(remote_status):
                local["status"] = "failed"
                local["error"] = remote_status
                self._record_swap(local)
                self._release_budget_reservation_for_swap(swap_id, "swap_failed")
                # Auto-refund if needed (fix #9)
                if (local.get("swap_type") in ("submarine", "chain")
                        and self._is_refundable_status(remote_status)):
                    self._try_auto_refund(swap_id, remote_status)
            else:
                local["status"] = remote_status
                self._record_swap(local)

        return {
            "local": local,
            "boltzd": remote,
        }

    # Backward-compatible alias
    def status(self, swap_id: str) -> Dict[str, Any]:
        return self.get_swap_info(swap_id)

    def refund_swap(self, swap_id: str, destination: str = "wallet") -> Dict[str, Any]:
        """Refund a failed submarine/chain swap to recover locked on-chain funds.

        Args:
            swap_id: The boltzd swap ID to refund.
            destination: 'wallet' (boltzd internal) or a BTC address.

        This is critical for recovering funds from submarine swaps that failed
        after on-chain lockup (invoice.failedToPay, swap.expired, transaction.lockupFailed).
        """
        swap_id, swap_id_err = self._cli_token(swap_id, "swap_id")
        if swap_id_err:
            return {"error": swap_id_err}
        destination, destination_err = self._cli_token(destination, "destination", allow_wallet_keyword=True)
        if destination_err:
            return {"error": destination_err}

        try:
            result = self._run_boltzcli_auto("refundswap", swap_id, destination)
        except RuntimeError as e:
            self._record_audit_event(
                "refund_failed", str(e),
                swap_id=swap_id,
                level="error",
                details={"destination": destination},
            )
            return {"error": str(e)}

        self._record_audit_event(
            "refund_initiated",
            f"Refund initiated for swap {swap_id} to {destination}",
            swap_id=swap_id,
            details={"destination": destination, "result": result},
        )

        # Update local record
        local = self._get_swap(swap_id)
        if local:
            local["status"] = "refunded"
            local["updated_at"] = self._now_ts()
            self._record_swap(local)
            self._release_budget_reservation_for_swap(swap_id, "swap_refunded")

        return {"status": "refund_initiated", "swap_id": swap_id, "result": result}

    def claim_swaps(self, swap_ids: List[str], destination: str = "wallet") -> Dict[str, Any]:
        """Manually claim reverse/chain swaps that failed to auto-claim.

        Args:
            swap_ids: List of boltzd swap IDs to claim.
            destination: 'wallet' (boltzd internal) or a BTC address.

        Use this when boltzd's automatic claim fails (e.g., crash mid-claim,
        claim tx didn't confirm). The funds are in the HTLC and need active claiming.
        """
        if isinstance(swap_ids, str):
            swap_ids = [swap_ids]
        elif not isinstance(swap_ids, (list, tuple, set)):
            return {"error": "swap_ids must be a list of swap IDs"}

        destination, destination_err = self._cli_token(destination, "destination", allow_wallet_keyword=True)
        if destination_err:
            return {"error": destination_err}

        validated_ids: List[str] = []
        for sid in list(swap_ids):
            sid_token, sid_err = self._cli_token(sid, "swap_id")
            if sid_err:
                return {"error": sid_err}
            validated_ids.append(sid_token)
        swap_ids = validated_ids

        if not swap_ids:
            return {"error": "No swap IDs provided"}

        try:
            result = self._run_boltzcli_auto("claimswaps", destination, *swap_ids)
        except RuntimeError as e:
            self._record_audit_event(
                "manual_claim_failed", str(e),
                level="error",
                details={"swap_ids": swap_ids, "destination": destination},
            )
            return {"error": str(e)}

        for sid in swap_ids:
            self._record_audit_event(
                "manual_claim_initiated",
                f"Manual claim initiated for swap {sid} to {destination}",
                swap_id=sid,
                details={"destination": destination},
            )

        return {"status": "claim_initiated", "swap_ids": swap_ids, "result": result}

    def _try_auto_refund(self, swap_id: str, failed_status: str) -> bool:
        """Attempt automatic refund for a submarine swap with locked funds.

        Called when a submarine swap enters a refundable failure state.
        Refunds to boltzd's internal wallet for safety.
        """
        self._log(
            f"Attempting auto-refund for submarine swap {swap_id} "
            f"(status: {failed_status})",
            level="warn",
        )
        try:
            result = self.refund_swap(swap_id, "wallet")
            if "error" not in result:
                self._record_audit_event(
                    "auto_refund_success",
                    f"Auto-refund succeeded for swap {swap_id}",
                    swap_id=swap_id,
                    details={"failed_status": failed_status},
                )
                return True
            else:
                # Refund may not be possible yet (timelock not expired).
                # Will retry on next monitoring cycle.
                self._record_audit_event(
                    "auto_refund_pending",
                    f"Auto-refund not yet possible for {swap_id}: {result['error']}",
                    swap_id=swap_id,
                    level="warn",
                    details={"failed_status": failed_status, "error": result["error"]},
                )
                return False
        except Exception as e:
            self._record_audit_event(
                "auto_refund_error",
                f"Auto-refund error for {swap_id}: {e}",
                swap_id=swap_id,
                level="error",
                details={"failed_status": failed_status},
            )
            return False

    def _maybe_record_completion_cost(self, local: Dict[str, Any]) -> None:
        """Record swap cost for P&L if not already recorded for this swap."""
        swap_id = local["id"]
        self._finalize_budget_reservation_for_swap(swap_id, final_fee_sats=local.get("total_fee_sats", 0))
        conn = self.db._get_connection()
        existing = conn.execute(
            "SELECT 1 FROM swap_costs WHERE swap_id = ?", (swap_id,)
        ).fetchone()
        if existing:
            return
        self._record_swap_cost(
            swap_id=swap_id,
            cost_sats=local.get("total_fee_sats", 0),
            amount_sats=local.get("amount_sats", 0),
            swap_type=local.get("swap_type", "reverse"),
            channel_id=local.get("target_channel_id"),
            peer_id=local.get("target_peer_id"),
            currency=local.get("currency", "btc"),
        )

    def list_swaps(self, pending: bool = False) -> Dict[str, Any]:
        """List swaps from boltzd."""
        args = ["listswaps"]
        if pending:
            args.append("--pending")
        return self._run_boltzcli(*args)

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate swap statistics from boltzd."""
        return self._run_boltzcli("stats")

    def history(self, limit: int = 20) -> Dict[str, Any]:
        """Get swap history from local DB merged with boltzd stats."""
        conn = self.db._get_connection()
        rows = conn.execute(
            "SELECT * FROM boltz_swaps ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        swaps = [dict(r) for r in rows]

        completed = [s for s in swaps if s.get("status") == "completed"]
        reverse_completed = [s for s in completed if s.get("swap_type") == "reverse"]
        submarine_completed = [s for s in completed if s.get("swap_type") == "submarine"]

        total_fees = sum(s.get("total_fee_sats", 0) for s in completed)
        total_amount = sum(s.get("amount_sats", 0) for s in completed)

        # Get boltzd stats if available
        try:
            boltzd_stats = self.get_stats()
        except Exception:
            boltzd_stats = None

        return {
            "swaps": swaps,
            "totals": {
                "count": len(swaps),
                "completed": len(completed),
                "reverse_completed": len(reverse_completed),
                "submarine_completed": len(submarine_completed),
                "total_fees_sats": total_fees,
                "total_amount_sats": total_amount,
                "avg_fee_ppm": int(total_fees * 1_000_000 / total_amount) if total_amount else 0,
            },
            "budget": self.get_budget_status(),
            "boltzd_stats": boltzd_stats,
        }

    # ------------------------------------------------------------------
    # Swap monitoring (called from background loop)
    # ------------------------------------------------------------------

    def check_pending_swaps(self) -> Dict[str, Any]:
        """
        Check status of pending swaps and update local records.
        Handles completion (cost recording), failure detection, and
        auto-refund for submarine swaps with locked funds.
        Returns summary of updates.
        """
        conn = self.db._get_connection()
        rows = conn.execute(
            """
            SELECT *
            FROM boltz_swaps
            WHERE status NOT IN ('completed', 'refunded')
              AND (
                status != 'failed'
                OR (
                  swap_type IN ('submarine', 'chain')
                  AND LOWER(COALESCE(error, '')) IN (
                    'invoice.failedtopay',
                    'swap.expired',
                    'transaction.lockupfailed'
                  )
                )
              )
            """
        ).fetchall()
        pending = [dict(r) for r in rows]

        if not pending:
            return {"checked": 0, "updated": 0, "completed": 0, "failed": 0, "refund_attempts": 0}

        # Get current swap list from boltzd
        try:
            boltzd_swaps = self.list_swaps()
        except RuntimeError as e:
            self._log(f"Failed to list swaps from boltzd: {e}", level="warn")
            return {"checked": len(pending), "updated": 0, "error": str(e)}

        # Build lookup by ID from ALL boltzd response lists (fix #7)
        boltzd_by_id = {}
        for key in ["reverseSwaps", "swaps", "chainSwaps", "channelCreations", "allSwaps"]:
            for s in boltzd_swaps.get(key, []):
                sid = s.get("id")
                if sid and sid not in boltzd_by_id:
                    boltzd_by_id[sid] = s

        updated = 0
        completed = 0
        failed = 0
        refund_attempts = 0

        for local in pending:
            swap_id = local["id"]
            remote = boltzd_by_id.get(swap_id)
            if not remote:
                continue

            remote_status = remote.get("status", "")
            if not remote_status:
                continue

            # Keep retrying auto-refund for failed swaps in refundable states.
            # Failed swaps are intentionally included in the query above so
            # timelock-dependent refunds can succeed in later monitor cycles.
            if local.get("status") == "failed" and self._is_failed_status(remote_status):
                if (local.get("swap_type") in ("submarine", "chain")
                        and self._is_refundable_status(remote_status)):
                    if self._try_auto_refund(swap_id, remote_status):
                        updated += 1
                    refund_attempts += 1
                continue

            if remote_status == local.get("status"):
                continue

            local["updated_at"] = self._now_ts()

            if self._is_completed_status(remote_status):
                local["status"] = "completed"
                self._record_swap(local)
                self._finalize_budget_reservation_for_swap(swap_id, final_fee_sats=local.get("total_fee_sats", 0))
                # Record cost for P&L (idempotent)
                self._maybe_record_completion_cost(local)
                self._record_audit_event(
                    "swap_completed",
                    f"Swap completed: {swap_id}",
                    swap_id=swap_id,
                    details={"remote_status": remote_status, "fee_sats": local.get("total_fee_sats", 0)},
                )
                completed += 1
            elif self._is_failed_status(remote_status):
                local["status"] = "failed"
                local["error"] = remote_status
                self._record_swap(local)
                self._release_budget_reservation_for_swap(swap_id, "swap_failed")
                self._record_audit_event(
                    "swap_failed",
                    f"Swap failed: {swap_id}",
                    swap_id=swap_id,
                    level="warn",
                    details={"remote_status": remote_status},
                )
                failed += 1
                # Auto-refund submarine/chain swaps with locked funds (fix #9)
                if (local.get("swap_type") in ("submarine", "chain")
                        and self._is_refundable_status(remote_status)):
                    self._try_auto_refund(swap_id, remote_status)
                    refund_attempts += 1
            else:
                local["status"] = remote_status
                self._record_swap(local)

            updated += 1

        return {
            "checked": len(pending),
            "updated": updated,
            "completed": completed,
            "failed": failed,
            "refund_attempts": refund_attempts,
        }

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _is_failed_status(self, status: str) -> bool:
        """Check if a boltzd status indicates the swap has failed.

        Covers all failure terminal states from the Boltz protocol:
        - transaction.failed: Boltz unable to lock agreed amount
        - transaction.refunded: Boltz auto-refunded unclaimed bitcoin
        - transaction.lockupFailed: Invalid lockup amount sent
        - swap.error: Generic swap error
        - swap.expired: Swap timed out
        - invoice.failedToPay: Submarine swap LN payment failed (funds need refund!)
        - invoice.expired: Reverse swap invoice expired
        """
        st = (status or "").lower()
        return (
            st.startswith("transaction.failed")
            or st == "transaction.refunded"
            or st == "transaction.lockupfailed"
            or st.startswith("swap.error")
            or st.startswith("invoice.failed")
            or st == "invoice.expired"
            or st == "swap.expired"
            or st == "failed"
            or st == "error"
        )

    def _is_refundable_status(self, status: str) -> bool:
        """Check if a failed submarine/chain swap may have locked funds needing refund.

        These states mean our on-chain lockup tx exists but the swap didn't complete,
        so the locked funds need to be reclaimed via refundswap after timelock expiry.
        """
        st = (status or "").lower()
        return st in (
            "invoice.failedtopay",
            "swap.expired",
            "transaction.lockupfailed",
        )

    def _is_completed_status(self, status: str) -> bool:
        """Check if a boltzd status indicates the swap completed successfully.

        Only truly terminal success states:
        - transaction.claimed: Boltz claimed on-chain BTC (submarine final success)
        - invoice.settled: Client preimage revealed, Boltz settled (reverse final success)

        NOTE: invoice.paid is NOT a final state for submarine swaps — it means
        the LN payment succeeded but Boltz hasn't claimed the on-chain funds yet.
        The swap could still fail between invoice.paid and transaction.claimed.
        """
        st = (status or "").lower()
        return st in (
            "completed",
            "invoice.settled",
            "transaction.claimed",
        )
