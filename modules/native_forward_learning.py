"""Explicit staging of source-bound learning over canonical receipt payloads.

No runtime caller, RPC, automatic migration or source/model admission. This
does not grant fee authority or change accounting-cutover admission guards.
Receipts are never replaced by local operational IDs or daily aggregates.
"""

import hashlib
import json
import re

from .forward_identity import ForwardSource, SettledForwardIdentity

MAX_STATE_BYTES = 1024 * 1024
MAX_BATCH = 1000
TABLE = "native_forward_learning_v1"


class LearningError(ValueError):
    pass


def _encoded(state):
    try:
        value = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, RecursionError):
        raise LearningError("model state must be bounded finite JSON") from None
    if not isinstance(state, dict) or len(value.encode()) > MAX_STATE_BYTES:
        raise LearningError("model state must be a bounded object")
    return value


def _digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class NativeForwardLearning:
    """Atomic bounded replay with a separate cursor for each model version.

    Caller must independently verify source/alias/coverage continuity before
    explicitly staging a model. A stored source key is not proof of continuity.
    The reducer receives only state and normalized records, never a connection.
    It must be deterministic, bounded and side-effect-free; timestamps describe
    settlement, not historical notification receipt or price exposure.

    Native receipt IDs must form a retained, append-only prefix. A missing
    receipt or archive payload is an error, never skipped learning. The archive
    may lag a newly received notification: retry that same batch after sync.
    Entire-database restoration needs external continuity reconciliation; a
    cursor/checksum cannot independently detect restoration of all evidence.
    """

    def __init__(self, connection, source, model_version, archive_generation=1):
        if not isinstance(source, ForwardSource):
            raise LearningError("explicit verified source required")
        self.source = source.key()
        if not isinstance(model_version, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]{1,128}", model_version):
            raise LearningError("explicit model version required")
        if type(archive_generation) is not int or archive_generation <= 0:
            raise LearningError("explicit archive generation required")
        self.connection = connection
        self.model = model_version
        self.generation = archive_generation

    def _binding(self):
        rows = self.connection.execute("SELECT source_key FROM forward_receipt_source_v1").fetchall()
        if len(rows) != 1 or rows[0][0] != self.source:
            raise LearningError("receipt source continuity mismatch")
        row = self.connection.execute("SELECT seq FROM sqlite_sequence WHERE name='forward_receipts_v1'").fetchone()
        high = row[0] if row else 0
        if type(high) is not int or high < 0:
            raise LearningError("invalid receipt watermark")
        return high

    def initialize(self, initial_state):
        """Explicit new-model staging only; never reset an existing model."""
        encoded = _encoded(initial_state)
        c = self.connection
        c.execute("BEGIN IMMEDIATE")
        try:
            high = self._binding()
            count, maximum = c.execute("SELECT COUNT(*),COALESCE(MAX(id),0) FROM forward_receipts_v1").fetchone()
            if count != high or maximum != high:
                raise LearningError("receipt prefix missing; reconcile before bootstrap")
            c.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (
                model_version TEXT PRIMARY KEY, source_key TEXT NOT NULL,
                archive_generation INTEGER NOT NULL, bootstrap_through INTEGER NOT NULL,
                through_id INTEGER NOT NULL, after_id INTEGER NOT NULL,
                anchor_digest TEXT, state_json TEXT NOT NULL, state_digest TEXT NOT NULL,
                revision INTEGER NOT NULL)""")
            if c.execute(f"SELECT 1 FROM {TABLE} WHERE model_version=?", (self.model,)).fetchone():
                raise LearningError("model already exists; resume instead of resetting")
            c.execute(f"INSERT INTO {TABLE} VALUES (?,?,?,?,?,0,NULL,?,?,0)",
                      (self.model, self.source, self.generation, high, high, encoded, _digest(encoded)))
            c.execute("COMMIT")
        except Exception:
            if c.in_transaction:
                c.execute("ROLLBACK")
            raise

    def _load(self):
        high = self._binding()
        cursor = self.connection.execute(f"SELECT * FROM {TABLE} WHERE model_version=?", (self.model,))
        row = cursor.fetchone()
        if row is None:
            raise LearningError("model not staged")
        value = dict(zip((col[0] for col in cursor.description), row))
        if value["source_key"] != self.source or value["archive_generation"] != self.generation:
            raise LearningError("model/source epoch mismatch")
        ints = [value[key] for key in ("bootstrap_through", "through_id", "after_id", "revision")]
        if (any(type(v) is not int or v < 0 for v in ints)
                or not value["after_id"] <= value["through_id"] <= high
                or value["bootstrap_through"] > value["through_id"]):
            raise LearningError("model cursor exceeds receipt evidence")
        encoded = value["state_json"]
        if not isinstance(encoded, str) or _digest(encoded) != value["state_digest"]:
            raise LearningError("model state checksum mismatch")
        state = json.loads(encoded)
        if _encoded(state) != encoded:
            raise LearningError("noncanonical model state")
        if value["after_id"]:
            anchor = self.connection.execute("SELECT payload_digest FROM forward_receipts_v1 WHERE id=? AND source_key=?",
                                             (value["after_id"], self.source)).fetchone()
            if anchor is None or anchor[0] != value["anchor_digest"]:
                raise LearningError("consumed receipt anchor changed")
        elif value["anchor_digest"] is not None:
            raise LearningError("unexpected empty model anchor")
        return value, state, high

    def status(self):
        """Read-only: no schema creation, cursor movement or model update."""
        c = self.connection
        c.execute("BEGIN")
        try:
            value, state, _ = self._load()
            return {"model_version": self.model, "state": state,
                    "after_id": value["after_id"], "through_id": value["through_id"],
                    "revision": value["revision"],
                    "bootstrap_complete": value["after_id"] >= value["bootstrap_through"],
                    "historical_admission_eligible": False}
        finally:
            c.execute("ROLLBACK")

    def advance(self, reducer, *, limit=500):
        """Commit model and receipt cursor once, or roll the entire batch back.

        A complete frozen window is followed by a new receipt high watermark.
        Late settlements with old event times but new receipt IDs are included.
        Reducers must handle receipt-ingestion order, not assume event-time order.
        """
        if type(limit) is not int or not 1 <= limit <= MAX_BATCH or not callable(reducer):
            raise LearningError("bounded batch and pure reducer required")
        c = self.connection
        c.execute("BEGIN IMMEDIATE")
        try:
            value, state, high = self._load()
            after = value["after_id"]
            through = high if after == value["through_id"] else value["through_id"]
            end = min(through, after + limit)
            rows = c.execute("SELECT id,source_key,in_channel,in_htlc_id,payload_digest,created_index "
                             "FROM forward_receipts_v1 WHERE id>? AND id<=? ORDER BY id", (after, end)).fetchall()
            if [row[0] for row in rows] != list(range(after+1, end+1)):
                raise LearningError("missing receipt in frozen learning window")
            records = []
            for receipt in rows:
                if receipt[1] != self.source:
                    raise LearningError("mixed receipt source")
                created = receipt[5]
                if not isinstance(created, str) or not re.fullmatch(r"[1-9][0-9]{0,18}", created) or int(created) > 2**63-1:
                    raise LearningError("archive lookup needs enriched created identity")
                archive = c.execute("SELECT in_channel,in_htlc_id,out_channel,in_msat,out_msat,fee_msat,"
                                    "received_time_ns,resolved_time_ns,created_index,updated_index,status,schema_version "
                                    "FROM forward_archive_v1 WHERE archive_generation=? AND created_index=?",
                                    (self.generation, int(created))).fetchone()
                if archive is None:
                    raise LearningError("canonical payload unavailable; retry after archive sync")
                if archive[10:] != ("settled", 1):
                    # sqlite3.Row slicing also returns a tuple.
                    raise LearningError("canonical settlement/schema required")
                record = SettledForwardIdentity(self.source, *archive[:10])
                record.validate()
                if (record.in_channel != receipt[2] or str(record.in_htlc_id) != receipt[3]
                        or record.payload_digest() != receipt[4]):
                    raise LearningError("receipt/archive payload conflict")
                records.append(record)
            if not records:
                c.execute("ROLLBACK")
                return {"consumed": 0, "after_id": after, "through_id": through, "complete": True}
            encoded = _encoded(reducer(state, tuple(records)))
            c.execute(f"UPDATE {TABLE} SET through_id=?,after_id=?,anchor_digest=?,state_json=?,state_digest=?,revision=revision+1 WHERE model_version=?",
                      (through, end, rows[-1][4], encoded, _digest(encoded), self.model))
            c.execute("COMMIT")
            return {"consumed": len(records), "after_id": end, "through_id": through,
                    "complete": end == through}
        except Exception:
            if c.in_transaction:
                c.execute("ROLLBACK")
            raise
