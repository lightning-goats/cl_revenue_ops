"""Isolated DTS reward-boundary probe, not a full-controller/economic replay.

No controller object, DB initializer, capture writer or action RPC is invoked.
Private saved state is copied in memory and only aggregate results are returned.
"""

import __future__
import ast
from collections import Counter
import copy
import hashlib
import inspect
import json
import random
from pathlib import Path
import sqlite3
import sys
import time
import types
from urllib.parse import quote

from tools import settled_reward_transition_audit as audit

MODEL_CLASSES = ("GaussianThompsonState", "PIDState", "ChannelFeeState")
MAX_SECONDS = 30
MAX_SEEDS = 32


def model_digest(source):
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in MODEL_CLASSES]
    if tuple(node.name for node in nodes) != MODEL_CLASSES:
        raise audit.AuditError("model class inventory differs")
    # AST identifies boundaries only. ast.dump's optional-field formatting
    # differs between Python 3.12 and 3.13, so it is not a portable source pin.
    # Hash exact decorated class source, including comments and whitespace.
    lines = source.splitlines(keepends=True)
    sections = []
    for node in nodes:
        first = min([node.lineno] + [item.lineno for item in node.decorator_list])
        sections.append([node.name, "".join(lines[first-1:node.end_lineno])])
    return hashlib.sha256(json.dumps(sections, separators=(",", ":")).encode()).hexdigest()


class Entropy:
    def __init__(self, now):
        self.now = now
        self.reset(0)

    def reset(self, seed):
        self.seed, self.calls = seed, Counter()

    def gauss(self, label, mean, std):
        ordinal = self.calls[label]
        self.calls[label] += 1
        # Common standard-normal noise by semantic label and per-label ordinal;
        # a branch's extra calls under another label don't shift future draws.
        key = json.dumps([self.seed, label, ordinal], separators=(",", ":"))
        return mean + std * random.Random(key).gauss(0, 1)


def isolated_classes(controller, entropy):
    """Compile unchanged class AST in a private namespace with fixed entropy.

    The caller must pin the imported controller source before using production
    state. Dependencies retain their existing definitions; no live module's
    clock, globals or random generator are patched.
    """
    source = inspect.getsource(controller)
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in MODEL_CLASSES]
    if tuple(node.name for node in nodes) != MODEL_CLASSES:
        raise audit.AuditError("model class inventory differs")
    name = "_revenue_reward_model_probe"
    if name in sys.modules:
        raise audit.AuditError("concurrent model probe refused")
    module = types.ModuleType(name)
    module.__dict__.update(vars(controller))
    module.__name__ = name
    module.decision_now = lambda _label: entropy.now
    module.decision_gauss = entropy.gauss
    sys.modules[name] = module
    try:
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<isolated revenue models>", "exec",
                     flags=__future__.annotations.compiler_flag, dont_inherit=True), module.__dict__)
    except Exception:
        del sys.modules[name]
        raise
    return module


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _roundtrip(state, cls, row):
    raw = _json(state.to_v2_dict())
    restored = cls.from_v2_dict(json.loads(raw), dict(row))
    if _json(restored.to_v2_dict()) != raw:
        raise audit.AuditError("model serialization is not idempotent")
    return restored


def compare_window(controller, classes, entropy, row, model_payload, *,
                   volume_msat, earned_msat, since, until, fee_ppm, floor, ceiling, seeds):
    """Probe one real window; unknown evidence leaves the model alone.

    Rates are undemand-adjusted component stimuli, not alleged full-pipeline
    observations. The current fee label is held fixed in both arms; no causal
    price/exposure claim follows from this comparison.
    """
    if (not audit._integer(since) or not audit._integer(until, 1) or since >= until
            or not audit._integer(fee_ppm, high=2**32-1)
            or not audit._integer(floor) or not audit._integer(ceiling) or not floor <= ceiling <= 1200
            or not audit._integer(seeds, 1, MAX_SEEDS)):
        raise audit.AuditError("invalid model probe window or rails")
    if not audit._integer(volume_msat) or not audit._integer(earned_msat):
        return {"unknown": 1}
    original = _json(model_payload)
    payload = controller.FeeController._extract_fee_state_payload(dict(row), copy.deepcopy(model_payload))
    if payload.get("algorithm_version") not in ("dts_pid_v1", "thompson_aimd_v1"):
        raise audit.AuditError("unsupported saved model version")
    if "thompson_state" not in payload:
        raise audit.AuditError("saved model missing")
    cls = classes.ChannelFeeState
    base = cls.from_v2_dict(copy.deepcopy(payload), dict(row))
    baseline = _roundtrip(base, cls, row)
    baseline_wire = _json(baseline.to_v2_dict())
    for bad in (None, -1, float("nan"), float("inf"), True):
        baseline.thompson.update_posterior(fee_ppm, bad, (until-since)/3600)
        baseline.thompson.update_contextual("normal:normal:P", fee_ppm, bad, "normal")
        if _json(baseline.to_v2_dict()) != baseline_wire:
            raise audit.AuditError("unknown reward changed saved learning")
    elapsed = (until - since) / 3600
    bootstrap = row["last_update"] == 0
    proxy_rate = (volume_msat // 1000) * fee_ppm / 1_000_000 / (1 if bootstrap else elapsed)
    earned_rate = earned_msat * 3.6 / (until - since)
    arms = []
    for rate in (proxy_rate, earned_rate):
        state = cls.from_v2_dict(json.loads(baseline_wire), dict(row))
        if fee_ppm > 0:  # Preserve the production zero-proportional-fee learning guard.
            state.thompson.update_posterior(fee_ppm, rate, elapsed, state.last_time_bucket)
            if state.last_context_key:
                state.thompson.update_contextual(state.last_context_key, fee_ppm, rate, state.last_time_bucket)
        # A common component stimulus, not a claim about dynamic profile selection.
        state.thompson.apply_dts_discount(gamma=.98)
        restarted = _roundtrip(state, cls, row)
        # Source-only rollback of d16f223 reads this unchanged state format using
        # the identical production class; this is not DB/accounting rollback.
        rollback = controller.ChannelFeeState.from_v2_dict(json.loads(_json(state.to_v2_dict())), dict(row))
        if _json(rollback.to_v2_dict()) != _json(state.to_v2_dict()):
            raise audit.AuditError("incumbent cannot round-trip candidate model")
        draws = []
        for seed in range(seeds):
            entropy.reset(seed)
            draw = state.thompson.sample_fee_contextual(state.last_context_key, floor, ceiling)
            entropy.reset(seed)
            after_restart = restarted.thompson.sample_fee_contextual(state.last_context_key, floor, ceiling)
            if draw != after_restart or not floor <= draw <= ceiling:
                raise audit.AuditError("restart changed bounded model proposal")
            draws.append(draw)
        arms.append((state, draws))
    if _json(model_payload) != original:
        raise audit.AuditError("source model was modified")
    old, new = arms
    differences = [abs(a-b) for a, b in zip(old[1], new[1])]
    return {"unknown": 0, "windows": 1, "positive_windows": int(earned_msat > 0),
            "reward_changed": int(proxy_rate != earned_rate),
            "positive_reference_changed": int(old[0].thompson.positive_rate_ref != new[0].thompson.positive_rate_ref),
            "zero_streak_changed": int(old[0].thompson.zero_revenue_streak != new[0].thompson.zero_revenue_streak),
            "paired_proposals": seeds, "changed_proposals": sum(delta > 0 for delta in differences),
            "max_proposal_delta_ppm": max(differences), "sum_proposal_delta_ppm": sum(differences)}


def probe(database, rpc, controller, *, expected_controller_sha, expected_model_digest,
          floor, ceiling, fee_interval=1800, now=None, seeds=MAX_SEEDS):
    """One read-only source snapshot, current windows only, no raw export."""
    conn = classes = None
    now = int(time.time()) if now is None else now
    deadline = time.monotonic() + MAX_SECONDS
    try:
        source = inspect.getsource(controller)
        if hashlib.sha256(source.encode()).hexdigest() != expected_controller_sha or model_digest(source) != expected_model_digest:
            raise audit.AuditError("controller/model source pin differs")
        if not audit._integer(now, 1) or not audit._integer(fee_interval, 1, 86400):
            raise audit.AuditError("invalid probe time")
        quotes = audit._quotes(rpc)
        entropy = Entropy(now)
        classes = isolated_classes(controller, entropy)
        path = Path(database).resolve(strict=True)
        conn = sqlite3.connect(f"file:{quote(str(path), safe='/')}?mode=ro", uri=True,
                               timeout=1, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        conn.execute("BEGIN")
        rows = conn.execute("SELECT channel_id,last_update,last_revenue_rate,last_fee_ppm,"
                            "length(CAST(v2_state_json AS BLOB)) AS size FROM fee_strategy_state LIMIT ?",
                            (audit.MAX_CHANNELS+1,)).fetchall()
        if len(rows) > audit.MAX_CHANNELS:
            raise audit.AuditError("model count budget exceeded")
        totals, seen = Counter(), set()
        for row in rows:
            if time.monotonic() >= deadline:
                raise audit.AuditError("model probe time budget exceeded")
            scid = row["channel_id"]
            if scid not in quotes:
                continue
            if scid in seen or not audit._integer(row["size"], 2, audit.MAX_STATE_BYTES):
                raise audit.AuditError("ambiguous or oversized saved model")
            seen.add(scid)
            cursor = row["last_update"]
            if not audit._integer(cursor) or cursor >= now:
                totals["unknown"] += 1
                continue
            since = cursor or now-fee_interval
            # Preserve all legacy scalars (PID/sleep/broadcast state included)
            # when invoking the real loader, not a four-field approximation.
            row = conn.execute("SELECT * FROM fee_strategy_state WHERE channel_id=?", (scid,)).fetchone()
            raw = row["v2_state_json"]
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise audit.AuditError("malformed saved model")
            nested = payload.get("cycle_state", {})
            if not isinstance(nested, dict) or any(key in nested and nested[key] != row[key] for key in ("last_update", "last_revenue_rate")):
                totals["unknown"] += 1
                continue
            data = conn.execute("""SELECT COALESCE(SUM(out_msat),0),COALESCE(SUM(fee_msat),0),
                COALESCE(SUM(CASE WHEN typeof(in_msat)!='integer' OR typeof(out_msat)!='integer'
                  OR typeof(fee_msat)!='integer' OR out_msat<0 OR fee_msat<0
                  OR in_msat<out_msat OR in_msat-out_msat!=fee_msat
                  OR typeof(timestamp)!='integer' OR timestamp>? THEN 1 ELSE 0 END),0)
                FROM forwards WHERE out_channel=? AND timestamp>?""", (now, scid, since)).fetchone()
            if data[2]:
                totals["unknown"] += 1
                continue
            result = compare_window(controller, classes, entropy, row, payload,
                                    volume_msat=data[0], earned_msat=data[1], since=since, until=now,
                                    fee_ppm=quotes[scid], floor=floor, ceiling=ceiling, seeds=seeds)
            maximum = result.pop("max_proposal_delta_ppm", 0)
            totals.update(result)
            totals["max_proposal_delta_ppm"] = max(totals["max_proposal_delta_ppm"], maximum)
        if audit._quotes(rpc) != quotes:
            raise audit.AuditError("quotes changed during model probe")
        if time.monotonic() >= deadline:
            raise audit.AuditError("model probe time budget exceeded")
        return {"schema_version": 1, "observed_at": now, "active_channels": len(quotes),
                "missing_strategy": len(quotes.keys()-seen), **dict(totals),
                "model_digest": expected_model_digest, "production_admission_eligible": False,
                "scope": "undemand_adjusted_DTS_component_current_windows_only",
                "limitations": ["not_full_controller_replay", "no_policy_exposure_proof", "not_atomic_CLN_DB_view",
                                "no_net_yield_evidence", "no_post_upgrade_arrival_or_accounting_rollback_test"]}
    except audit.AuditError:
        raise
    except Exception:
        raise audit.AuditError("saved-model probe refused") from None
    finally:
        if conn is not None:
            conn.close()
        if classes is not None:
            sys.modules.pop(classes.__name__, None)
