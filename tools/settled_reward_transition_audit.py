"""Read-only, aggregate-only reward-transition diagnostic; never a release gate.

Compare the incumbent's current-price reward with settled fees in a single
operational DB view. Does not replay controller actions or repair saved models.
Retained forwards may be incomplete; quotes and DB are not an atomic CLN view.
"""

import argparse
import json
import math
from pathlib import Path
import socket
import sqlite3
import time
from urllib.parse import quote

MAX_CHANNELS = 1000
MAX_STATE_BYTES = 1024 * 1024
MAX_SECONDS = 10


class AuditError(ValueError):
    """Sanitized refusal; never include private rows or paths."""


def _integer(value, low=0, high=2**63 - 1):
    return type(value) is int and low <= value <= high


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _quotes(rpc):
    try:
        response = rpc("listpeerchannels", {})
    except Exception:
        raise AuditError("channel evidence unavailable") from None
    if not isinstance(response, dict) or not isinstance(response.get("channels"), list):
        raise AuditError("invalid channel response")
    if len(response["channels"]) > MAX_CHANNELS:
        raise AuditError("channel budget exceeded")
    result = {}
    for channel in response["channels"]:
        if not isinstance(channel, dict):
            raise AuditError("invalid channel entry")
        if channel.get("state") != "CHANNELD_NORMAL":
            continue
        scid, ppm = channel.get("short_channel_id"), channel.get("fee_proportional_millionths")
        if not isinstance(scid, str) or not scid or not _integer(ppm, high=2**32 - 1):
            raise AuditError("missing active channel quote")
        if scid in result:
            raise AuditError("ambiguous active channel identity")
        result[scid] = ppm
    return result


def _branches(rate, previous):
    # Pins the incumbent and candidate's normal hysteresis predicates, NOT
    # their entire control flow (which also gates on time, inventory, etc.).
    ratio = abs(rate - previous) / max(1.0, previous) if previous > 0 else (1.5 if rate > 0 else 0)
    wake_ratio = abs(rate - previous) / previous if previous > 0 else (1.0 if rate > 0 else 0)
    return ratio > .50, ratio < .01, wake_ratio > .20


def audit(database, rpc, *, now=None, fee_interval=1800):
    now = int(time.time()) if now is None else now
    if not _integer(now, 1) or not _integer(fee_interval, 1, 86400):
        raise AuditError("invalid observation time or interval")
    deadline = time.monotonic() + MAX_SECONDS
    conn = None
    try:
        quotes = _quotes(rpc)
        path = Path(database).resolve(strict=True)
        conn = sqlite3.connect(f"file:{quote(str(path), safe='/')}?mode=ro", uri=True,
                               timeout=1, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        conn.execute("BEGIN")
        rows = conn.execute("SELECT channel_id,last_update,last_revenue_rate,is_sleeping,sleep_until,"
                            "length(CAST(v2_state_json AS BLOB)) AS state_bytes FROM fee_strategy_state LIMIT ?",
                            (MAX_CHANNELS + 1,)).fetchall()
        if len(rows) > MAX_CHANNELS:
            raise AuditError("strategy budget exceeded")
        counts = {key: 0 for key in (
            "active_channels", "strategy_rows", "active_without_strategy", "inactive_strategy_rows",
            "unknown_windows", "bootstrap_windows", "zero_volume_windows", "positive_volume_windows",
            "reward_changed_windows", "normal_volatility_predicate_changes", "normal_stability_predicate_changes",
            "sleep_wake_predicate_changes", "shortfall_windows", "shortfall_forwards",
            "stored_observations", "stored_positive_observations", "stored_contexts",
            "models_without_reward_source_marker", "models_missing_thompson_state",
            "positive_reference_models", "earned_msat", "forward_count")}
        counts["active_channels"], counts["strategy_rows"] = len(quotes), len(rows)
        seen = set()
        max_relative_delta = 0.0
        for row in rows:
            if time.monotonic() >= deadline:
                raise AuditError("audit time budget exceeded")
            scid = row["channel_id"]
            if scid not in quotes:
                counts["inactive_strategy_rows"] += 1
                continue
            if scid in seen:
                raise AuditError("ambiguous strategy identity")
            seen.add(scid)
            if not _integer(row["state_bytes"], 2, MAX_STATE_BYTES):
                raise AuditError("invalid or oversized saved model")
            raw = conn.execute("SELECT v2_state_json FROM fee_strategy_state WHERE channel_id=?", (scid,)).fetchone()[0]
            state = json.loads(raw)
            if not isinstance(state, dict):
                raise AuditError("invalid saved model")
            # Match FeeController._extract_fee_state_payload: canonical nested
            # state takes precedence over the legacy flat compatibility mirror.
            payload = state.get("fee_state")
            payload = payload if isinstance(payload, dict) else state
            model = payload.get("thompson_state", {})
            if not isinstance(model, dict):
                raise AuditError("invalid saved model")
            counts["models_missing_thompson_state"] += int("thompson_state" not in payload)
            reference = model.get("positive_rate_ref", 0)
            if not _number(reference):
                raise AuditError("invalid saved positive reference")
            counts["positive_reference_models"] += int(reference > 0)
            observations, contexts = model.get("observations", []), model.get("contextual_posteriors", {})
            if not isinstance(observations, list) or not isinstance(contexts, dict):
                raise AuditError("invalid saved observations")
            for obs in observations:
                if not isinstance(obs, list) or len(obs) < 4 or not all(_number(v) for v in obs[:4]):
                    raise AuditError("invalid saved observation")
                counts["stored_positive_observations"] += int(obs[1] > 0)
            counts["stored_observations"] += len(observations)
            counts["stored_contexts"] += len(contexts)
            counts["models_without_reward_source_marker"] += int("reward_source" not in model)
            cycle = state.get("cycle_state", {})
            if not isinstance(cycle, dict) or any(
                key in cycle and cycle[key] != row[key]
                for key in ("last_update", "last_revenue_rate", "is_sleeping", "sleep_until")
            ):
                # Runtime cycle loading prefers nested values, while the DTS
                # loader uses row scalars. Do not invent one common window if
                # these normally-mirrored fields disagree.
                counts["unknown_windows"] += 1
                continue
            cursor, previous = row["last_update"], row["last_revenue_rate"]
            if (not _integer(cursor) or not _number(previous) or row["is_sleeping"] not in (0, 1)
                    or not _integer(row["sleep_until"])):
                counts["unknown_windows"] += 1
                continue
            bootstrap = cursor == 0
            since = now - fee_interval if bootstrap else cursor
            if not 0 <= since < now:
                counts["unknown_windows"] += 1
                continue
            # Same directional exclusive cursor as the production reader. A
            # future timestamp is refused: incumbent volume has no upper bound.
            forwards = conn.execute("""
                SELECT COUNT(*) AS n, COALESCE(SUM(fee_msat),0) AS earned,
                       COALESCE(SUM(out_msat),0) AS volume,
                       COALESCE(SUM(CASE WHEN typeof(in_msat)!='integer' OR typeof(out_msat)!='integer'
                         OR typeof(fee_msat)!='integer' OR out_msat<0 OR fee_msat<0
                         OR in_msat<out_msat OR in_msat-out_msat!=fee_msat
                         OR typeof(timestamp)!='integer' OR timestamp>? THEN 1 ELSE 0 END),0) AS invalid,
                       COALESCE(SUM(CASE WHEN fee_msat < (out_msat/1000000)*? +
                         ((out_msat%1000000)*?)/1000000 THEN 1 ELSE 0 END),0) AS shortfalls
                FROM forwards WHERE out_channel=? AND timestamp>?
            """, (now, quotes[scid], quotes[scid], scid, since)).fetchone()
            if forwards["invalid"] or not all(_integer(forwards[k]) for k in ("n", "earned", "volume", "shortfalls")):
                counts["unknown_windows"] += 1
                continue
            counts["bootstrap_windows"] += int(bootstrap)
            counts["zero_volume_windows" if forwards["volume"] == 0 else "positive_volume_windows"] += 1
            counts["earned_msat"] += forwards["earned"]
            counts["forward_count"] += forwards["n"]
            counts["shortfall_windows"] += int(forwards["shortfalls"] > 0)
            counts["shortfall_forwards"] += forwards["shortfalls"]
            elapsed_hours = (now - since) / 3600
            proxy_sats = (forwards["volume"] // 1000) * quotes[scid] / 1_000_000
            old_rate = proxy_sats / (1.0 if bootstrap else elapsed_hours)
            new_rate = forwards["earned"] * 3.6 / (now - since)
            counts["reward_changed_windows"] += int(not math.isclose(old_rate, new_rate, rel_tol=1e-12, abs_tol=1e-12))
            max_relative_delta = max(max_relative_delta, abs(new_rate-old_rate)/max(1.0, old_rate))
            old_branches, new_branches = _branches(old_rate, previous), _branches(new_rate, previous)
            if not bootstrap:
                counts["normal_volatility_predicate_changes"] += int(old_branches[0] != new_branches[0])
                counts["normal_stability_predicate_changes"] += int(old_branches[1] != new_branches[1])
            if row["is_sleeping"] and row["sleep_until"] > now:
                old_sleep_rate = proxy_sats / max(.1, 1.0 if bootstrap else elapsed_hours)
                counts["sleep_wake_predicate_changes"] += int(_branches(old_sleep_rate, previous)[2] != new_branches[2])
        counts["active_without_strategy"] = len(quotes.keys() - seen)
        if _quotes(rpc) != quotes:
            raise AuditError("active quotes changed during observation")
        if time.monotonic() >= deadline:
            raise AuditError("audit time budget exceeded")
        return {"schema_version": 1, "observed_at": now, **counts,
                "max_reward_delta_over_max_one_proxy_sats_per_hour": max_relative_delta,
                "production_admission_eligible": False,
                "limitations": ["single_snapshot_not_full_controller_replay", "operational_history_may_be_incomplete",
                                "live_quotes_not_atomic_with_database", "no_historical_reward_relabeling",
                                "no_causal_exposure_or_net_yield_evidence"]}
    except AuditError:
        raise
    except (OSError, sqlite3.Error, ValueError, TypeError, OverflowError, RecursionError):
        raise AuditError("reward transition evidence unavailable") from None
    finally:
        if conn is not None:
            conn.close()


class ChannelReader:
    """Bounded read-only transport; no configurable action RPC."""

    def __init__(self, path):
        self.path = str(path)

    def __call__(self, method, params):
        if method != "listpeerchannels" or params != {}:
            raise AuditError("RPC outside read-only audit surface")
        try:
            deadline = time.monotonic() + 2
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                conn.settimeout(2)
                conn.connect(self.path)
                conn.sendall(json.dumps({"jsonrpc": "2.0", "id": "reward-audit", "method": method, "params": {}}).encode()+b"\n\n")
                data = bytearray()
                while b"\n\n" not in data:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise AuditError("channel RPC timeout")
                    conn.settimeout(remaining)
                    chunk = conn.recv(65536)
                    if not chunk:
                        raise AuditError("incomplete channel RPC response")
                    data.extend(chunk)
                    if len(data) > 4 * 1024 * 1024:
                        raise AuditError("channel RPC byte budget exceeded")
                response = json.loads(bytes(data).split(b"\n\n", 1)[0])
                if not isinstance(response, dict) or response.get("id") != "reward-audit" or "error" in response:
                    raise AuditError("invalid channel RPC response")
                return response["result"]
        except (OSError, ValueError, KeyError, TypeError):
            raise AuditError("read-only channel RPC failed") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--rpc-file", required=True)
    parser.add_argument("--fee-interval", required=True, type=int)
    args = parser.parse_args()
    try:
        result = audit(args.database, ChannelReader(args.rpc_file), fee_interval=args.fee_interval)
    except AuditError as exc:
        print(json.dumps({"error": str(exc), "production_admission_eligible": False}))
        return 1
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
