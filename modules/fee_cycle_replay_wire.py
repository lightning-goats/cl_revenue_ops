import hashlib
import hmac
import json
import math
from typing import Any, Dict

SCHEMA_NAME = "fee_cycle_replay"
SCHEMA_VERSION = 0
MAX_ENVELOPE_BYTES = 32 * 1024 * 1024


def tag_floats(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        rendered = repr(value)
        if math.isnan(value):
            rendered = "nan"
        elif value == math.inf:
            rendered = "inf"
        elif value == -math.inf:
            rendered = "-inf"
        return {"__f__": rendered}
    if isinstance(value, dict):
        return {str(key): tag_floats(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tag_floats(item) for item in value]
    if value is None or isinstance(value, (str, int)):
        return value
    raise TypeError(f"unsupported replay wire type: {type(value).__name__}")


def canonical_body_bytes(body: Dict[str, Any]) -> bytes:
    return json.dumps(
        tag_floats(body),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def seal_envelope(body: Dict[str, Any]) -> Dict[str, Any]:
    if body.get("schema_name") != SCHEMA_NAME:
        raise ValueError("wrong schema_name")
    if body.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("wrong schema_version")
    tagged = tag_floats(body)
    payload = canonical_body_bytes(tagged)
    if len(payload) > MAX_ENVELOPE_BYTES:
        raise ValueError("envelope exceeds 32 MiB")
    return {
        **tagged,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }


def verify_envelope(envelope: Dict[str, Any]) -> None:
    supplied = envelope.get("payload_sha256")
    if not isinstance(supplied, str) or len(supplied) != 64:
        raise ValueError("missing payload digest")
    body = dict(envelope)
    del body["payload_sha256"]
    expected = hashlib.sha256(canonical_body_bytes(body)).hexdigest()
    if not hmac.compare_digest(supplied, expected):
        raise ValueError("payload digest mismatch")
