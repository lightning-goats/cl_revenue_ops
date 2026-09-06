"""Native settled-forward identity and transactional replay receipts.

Not wired into operational ingestion yet. This module has no RPC, clock,
learning, or migration authority. Callers must verify source continuity and
reconcile legacy accounting before admitting events to a receipt ledger.
Receipts are idempotency metadata, not another revenue/accounting source.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import sqlite3
from typing import Any, Mapping, Optional

from .forward_archive import parse_cln_time_ns


class ForwardIdentityError(ValueError):
    """Malformed, conflicting, or unbound ingestion evidence."""


@dataclass(frozen=True)
class ForwardSource:
    """Caller-verified wallet generation; never generated from event time.

    A daemon/plugin restart does not change generation. A wallet restore or
    replacement needs explicit continuity/reconciliation, not a fresh value
    supplied merely to get past a duplicate or conflict.
    """

    node_id: str
    network: str
    generation: str

    def key(self) -> str:
        if not isinstance(self.node_id, str) or not re.fullmatch(
            r"0[23][0-9a-f]{64}", self.node_id
        ):
            raise ForwardIdentityError("invalid source node")
        if self.network not in ("bitcoin", "testnet", "testnet4", "signet", "regtest"):
            raise ForwardIdentityError("invalid source network")
        if not isinstance(self.generation, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,128}", self.generation
        ):
            raise ForwardIdentityError("invalid source generation")
        return json.dumps([self.network, self.node_id, self.generation], separators=(",", ":"))


def _integer(value: Any, field: str, maximum: int, *, amount: bool = False) -> int:
    if amount and hasattr(value, "millisatoshis"):
        value = value.millisatoshis
    if amount and isinstance(value, str) and value.endswith("msat"):
        value = value[:-4]
    if isinstance(value, str) and len(value) <= 20 and re.fullmatch(r"[0-9]+", value):
        value = int(value)
    if type(value) is not int or not 0 <= value <= maximum:
        raise ForwardIdentityError(f"invalid {field}")
    return value


def _scid(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 24:
        raise ForwardIdentityError("invalid channel")
    parts = re.split("[x:]", value)
    if len(parts) != 3:
        raise ForwardIdentityError("invalid channel")
    values = [_integer(v, "channel", maximum) for v, maximum in zip(
        parts, (2**24 - 1, 2**24 - 1, 2**16 - 1)
    )]
    return "x".join(str(v) for v in values)


def _index(payload: Mapping[str, Any], name: str) -> Optional[int]:
    value = payload.get(name)
    if value is None:
        return None
    # CLN uses zero as an absent-index sentinel, unlike incoming HTLC ID 0.
    return _integer(value, name, 2**64 - 1) or None


@dataclass(frozen=True)
class SettledForwardIdentity:
    source_key: str
    in_channel: str
    in_htlc_id: int
    out_channel: str
    in_msat: int
    out_msat: int
    fee_msat: int
    received_time_ns: int
    resolved_time_ns: int
    created_index: Optional[int]
    updated_index: Optional[int]

    def validate(self) -> None:
        try:
            network, node_id, generation = json.loads(self.source_key)
            if ForwardSource(node_id, network, generation).key() != self.source_key:
                raise ForwardIdentityError("noncanonical source binding")
            if _scid(self.in_channel) != self.in_channel or _scid(self.out_channel) != self.out_channel:
                raise ForwardIdentityError("noncanonical channel")
            for field in ("in_htlc_id", "in_msat", "out_msat", "fee_msat",
                          "received_time_ns", "resolved_time_ns"):
                value = getattr(self, field)
                maximum = 2**64 - 1 if field == "in_htlc_id" else 2**63 - 1
                if type(value) is not int:
                    raise ForwardIdentityError("noncanonical integer")
                _integer(value, field, maximum)
            if self.out_msat == 0 or self.in_msat - self.out_msat != self.fee_msat:
                raise ForwardIdentityError("inconsistent amounts")
            if not 0 < self.received_time_ns <= self.resolved_time_ns:
                raise ForwardIdentityError("invalid canonical times")
            for field in ("created_index", "updated_index"):
                value = getattr(self, field)
                if value is not None and (type(value) is not int or not 0 < value <= 2**64 - 1):
                    raise ForwardIdentityError("noncanonical index")
        except (ValueError, TypeError, AttributeError) as exc:
            raise ForwardIdentityError("invalid normalized settlement") from exc

    def payload_digest(self) -> str:
        # Indices are optional enrichment, not economic payload or identity.
        values = [self.in_channel, self.in_htlc_id, self.out_channel,
                  self.in_msat, self.out_msat, self.fee_msat,
                  self.received_time_ns, self.resolved_time_ns]
        return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class IdentityObservation:
    status: str
    record: Optional[SettledForwardIdentity] = None
    reason: str = ""


def observe_settled_identity(
    payload: Any, source: Optional[ForwardSource],
) -> IdentityObservation:
    """Normalize without inventing identity, zero earnings, or receipt times.

    ``usable`` only means identity/payload are locally well formed; it does
    not prove source continuity, payer exposure, or permission to train.
    No raw payload is retained in error responses.
    """
    if not isinstance(payload, Mapping):
        return IdentityObservation("invalid", reason="expected forward object")
    try:
        status = payload.get("status")
        if status in ("offered", "failed", "local_failed"):
            return IdentityObservation("not_settled")
        if status != "settled":
            raise ForwardIdentityError("invalid status")
        if source is None:
            return IdentityObservation("unknown", reason="missing source binding")
        if not isinstance(source, ForwardSource):
            raise ForwardIdentityError("invalid source binding")
        source_key = source.key()
        if payload.get("in_htlc_id") is None:
            return IdentityObservation("unknown", reason="missing incoming HTLC identity")
        htlc_id = _integer(payload["in_htlc_id"], "incoming HTLC identity", 2**64 - 1)
        amounts = []
        for name in ("in_msat", "out_msat", "fee_msat"):
            legacy_name = name.replace("_msat", "_msatoshi")
            value = payload[name] if name in payload else payload.get(legacy_name)
            amounts.append(_integer(value, name, 2**63 - 1, amount=True))
        in_msat, out_msat, fee_msat = amounts
        if out_msat == 0 or in_msat - out_msat != fee_msat:
            raise ForwardIdentityError("inconsistent settlement amounts")
        received = parse_cln_time_ns(payload.get("received_time"))
        resolved = parse_cln_time_ns(payload.get("resolved_time"))
        if received is None or resolved is None or received == 0:
            return IdentityObservation("unknown", reason="missing canonical time")
        if not 0 < received <= resolved <= 2**63 - 1:
            raise ForwardIdentityError("invalid canonical time order or range")
        return IdentityObservation("usable", SettledForwardIdentity(
            source_key, _scid(payload.get("in_channel")), htlc_id,
            _scid(payload.get("out_channel")), in_msat, out_msat, fee_msat,
            received, resolved, _index(payload, "created_index"),
            _index(payload, "updated_index"),
        ))
    except Exception:
        return IdentityObservation("invalid", reason="malformed settlement evidence")


@dataclass(frozen=True)
class ReceiptClaim:
    receipt_id: int
    inserted: bool


class ForwardReceiptLedger:
    """A caller-transaction-owned replay ledger, intentionally not auto-wired.

    Claim AND operational/reputation writes must commit in the SAME SQLite
    transaction on this connection. An insertion failure must roll back the
    whole transaction. ``inserted=False`` never authorizes another reward.
    Never prune receipts merely because operational raw rows were pruned.
    This primitive does not reconcile pre-ledger legacy rows/rollups, validate
    a live wallet generation, or detect restoration of the entire database.
    """

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def _require_transaction(self) -> None:
        if not self.connection.in_transaction:
            raise ForwardIdentityError("caller transaction required")

    def initialize(self, source: ForwardSource) -> None:
        self._require_transaction()
        if not isinstance(source, ForwardSource):
            raise ForwardIdentityError("verified source binding required")
        source_key = source.key()
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS forward_receipt_source_v1 (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                source_key TEXT NOT NULL
            )
        """)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS forward_receipts_v1 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_key TEXT NOT NULL,
                in_channel TEXT NOT NULL,
                in_htlc_id TEXT NOT NULL,
                payload_digest TEXT NOT NULL,
                created_index TEXT,
                updated_index TEXT,
                UNIQUE(source_key, in_channel, in_htlc_id),
                UNIQUE(source_key, created_index),
                UNIQUE(source_key, updated_index)
            )
        """)
        row = self.connection.execute(
            "SELECT source_key FROM forward_receipt_source_v1 WHERE singleton = 1"
        ).fetchone()
        if row is None:
            if self.connection.execute("SELECT 1 FROM forward_receipts_v1 LIMIT 1").fetchone():
                raise ForwardIdentityError("missing source binding requires explicit reconciliation")
            self.connection.execute(
                "INSERT INTO forward_receipt_source_v1 VALUES (1, ?)", (source_key,)
            )
        elif row[0] != source_key:
            raise ForwardIdentityError("source continuity requires explicit reconciliation")

    def claim(self, observation: IdentityObservation) -> ReceiptClaim:
        self._require_transaction()
        if not isinstance(observation, IdentityObservation) or observation.status != "usable":
            raise ForwardIdentityError("usable identity evidence required")
        record = observation.record
        if not isinstance(record, SettledForwardIdentity):
            raise ForwardIdentityError("normalized settlement required")
        record.validate()
        c = self.connection
        binding = c.execute(
            "SELECT source_key FROM forward_receipt_source_v1 WHERE singleton = 1"
        ).fetchone()
        if binding is None or binding[0] != record.source_key:
            raise ForwardIdentityError("source binding mismatch")
        key = (record.source_key, record.in_channel, str(record.in_htlc_id))
        row = c.execute("""
            SELECT id, payload_digest, created_index, updated_index
            FROM forward_receipts_v1
            WHERE source_key = ? AND in_channel = ? AND in_htlc_id = ?
        """, key).fetchone()
        created = str(record.created_index) if record.created_index is not None else None
        updated = str(record.updated_index) if record.updated_index is not None else None
        digest = record.payload_digest()
        if row is not None:
            if row[1] != digest or (created is not None and row[2] is not None and created != row[2]):
                raise ForwardIdentityError("conflicting native settlement")
            created = row[2] if row[2] is not None else created
            updates = [int(v) for v in (row[3], updated) if v is not None]
            updated = str(max(updates)) if updates else None
            try:
                c.execute("""
                    UPDATE forward_receipts_v1 SET created_index = ?, updated_index = ?
                    WHERE id = ?
                """, (created, updated, row[0]))
            except sqlite3.IntegrityError as exc:
                raise ForwardIdentityError("native index belongs to another identity") from exc
            return ReceiptClaim(row[0], False)
        try:
            cursor = c.execute("""
                INSERT INTO forward_receipts_v1
                (source_key, in_channel, in_htlc_id, payload_digest, created_index, updated_index)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (*key, digest, created, updated))
        except sqlite3.IntegrityError as exc:
            raise ForwardIdentityError("native index belongs to another identity") from exc
        return ReceiptClaim(cursor.lastrowid, True)
