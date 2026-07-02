# Rebalancer Upstream-Patterns Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three upstream-`rebalance`-inspired behaviors to the cl_revenue_ops rebalancer — a per-channel realized-utilization EV basis (#2), size-tiered ideal-ratio targets (#3), and a live-activity score penalty (#1) — without touching any spend/budget path.

**Architecture:** One shared, pure data unit (`ChannelFlowFacts`, computed once per cycle from the `forwards` table) is attached to each `ChannelState`, then threaded through `PairCandidate` into the engine's EV gate and the planner's band classification. No spend path, reservation, or budget math is modified — the audited atomic budget rail still bounds all spend, so the worst case is suboptimal pair selection within budget.

**Tech Stack:** Python 3, pytest, SQLite (`modules/database.py`), the v2 rebalance pipeline (`rebalance_state_v2.py` → `rebalance_planner_v2.py` → `rebalance_engine_v2.py`), frozen dataclasses (`rebalance_types_v2.py`).

**Spec:** `docs/planning/2026-07-02-rebalancer-upstream-patterns-design.md` (approved 2026-07-02).

**Branch:** `rebalancer-upstream-patterns` (already created; the spec is committed there).

---

## File Structure

- **Create** `modules/rebalance_flow_facts.py` — `ChannelFlowFacts` dataclass + `compute_channel_flow_facts()` pure function. One responsibility: turn DB forward-window rows + config into per-channel facts.
- **Create** `tests/test_rebalance_flow_facts.py` — unit tests for the facts math.
- **Modify** `modules/database.py` — add one read method `get_channel_flow_window()`.
- **Modify** `modules/config.py` — add the new config knobs (OPTION_TYPES + dataclass defaults).
- **Modify** `cl-revenue-ops.py` — register the new options via `add_option` (avoid the P6-002 unregistered-option trap).
- **Modify** `modules/rebalance_state_v2.py` — add fields to `ChannelState`; thread a `flow_facts` map + per-channel target bands into `build_state_snapshot`.
- **Modify** `modules/rebalance_types_v2.py` — add utilization fields to `PairCandidate`.
- **Modify** `modules/rebalance_planner_v2.py` — consume per-channel bands; map utilization onto pairs.
- **Modify** `modules/rebalance_engine_v2.py` — use per-channel utilization in EV; add the activity penalty; extend `score_decomposition`.
- **Tests:** `tests/test_rebalance_flow_facts.py`, plus additions to `tests/test_rebalance_planner_v2.py`, `tests/test_rebalance_engine_v2.py` (or per-feature `tests/test_p_upstream_*.py`).

**Regression gate after every task:** `python3 -m pytest tests/ -q` and `python3 tools/audit/scorecard.py --deep-only` must stay green; `python3 -m pytest tests/test_all_spenders_atomic.py -q` proves the budget invariant is untouched.

---

## Task 1: DB read method for windowed directional forward facts

**Files:**
- Modify: `modules/database.py` (add method near `get_volume_since`, ~5114)
- Test: `tests/test_rebalance_flow_facts.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rebalance_flow_facts.py
import time
from modules.database import Database

def _seed(db, out_channel, in_channel, out_msat, in_msat, ts):
    conn = db._get_connection()
    conn.execute(
        "INSERT INTO forwards (in_channel, out_channel, in_msat, out_msat, fee_msat, status, timestamp) "
        "VALUES (?, ?, ?, ?, ?, 'settled', ?)",
        (in_channel, out_channel, in_msat, out_msat, 0, ts),
    )

def test_get_channel_flow_window_sums_directional(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    now = 1_000_000
    # 2 outbound forwards and 1 inbound for channel "A"
    _seed(db, out_channel="A", in_channel="B", out_msat=30_000_000, in_msat=30_100_000, ts=now - 10)
    _seed(db, out_channel="A", in_channel="C", out_msat=20_000_000, in_msat=20_050_000, ts=now - 20)
    _seed(db, out_channel="D", in_channel="A", out_msat=15_000_000, in_msat=15_030_000, ts=now - 30)
    # one row OUTSIDE the window must be excluded
    _seed(db, out_channel="A", in_channel="B", out_msat=99_000_000, in_msat=99_000_000, ts=now - 10_000)

    out_sats, in_sats, count = db.get_channel_flow_window("A", since_timestamp=now - 100)
    assert out_sats == 50_000  # (30_000_000 + 20_000_000) msat -> sats
    assert in_sats == 15_030   # 15_030_000 msat -> sats (floor)
    assert count == 3          # 2 out + 1 in within window
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_rebalance_flow_facts.py::test_get_channel_flow_window_sums_directional -v`
Expected: FAIL with `AttributeError: 'Database' object has no attribute 'get_channel_flow_window'`

- [ ] **Step 3: Write minimal implementation**

Add to `modules/database.py` (after `get_forward_count_since`, ~5155). Follow the existing `get_volume_since` pattern (uses `base_to_sats_floor`, `_get_connection`, `timestamp` column):

```python
    def get_channel_flow_window(self, channel_id: str, since_timestamp: int):
        """Directional forwarded volume + count for a channel since a timestamp.

        Returns (out_sats, in_sats, forward_count) where out_sats is volume
        forwarded OUT of the channel and in_sats is volume forwarded INTO it,
        over settled forwards with timestamp > since_timestamp. Used by the
        rebalancer's ChannelFlowFacts (realized-utilization + live-activity).
        """
        conn = self._get_connection()
        out_row = conn.execute(
            "SELECT COALESCE(SUM(out_msat), 0) AS m, COUNT(*) AS c FROM forwards "
            "WHERE out_channel = ? AND timestamp > ?",
            (channel_id, since_timestamp),
        ).fetchone()
        in_row = conn.execute(
            "SELECT COALESCE(SUM(in_msat), 0) AS m, COUNT(*) AS c FROM forwards "
            "WHERE in_channel = ? AND timestamp > ?",
            (channel_id, since_timestamp),
        ).fetchone()
        out_sats = base_to_sats_floor(out_row["m"]) if out_row else 0
        in_sats = base_to_sats_floor(in_row["m"]) if in_row else 0
        count = (out_row["c"] if out_row else 0) + (in_row["c"] if in_row else 0)
        return out_sats, in_sats, count
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_rebalance_flow_facts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modules/database.py tests/test_rebalance_flow_facts.py
git commit -m "feat(rebalance): DB windowed directional forward facts (get_channel_flow_window)"
```

---

## Task 2: `ChannelFlowFacts` pure compute unit

**Files:**
- Create: `modules/rebalance_flow_facts.py`
- Test: `tests/test_rebalance_flow_facts.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rebalance_flow_facts.py`:

```python
from modules.rebalance_flow_facts import ChannelFlowFacts, compute_channel_flow_facts

class _Cfg:
    rebalance_activity_window_seconds = 3600
    rebalance_utilization_window_days = 7
    rebalance_utilization_floor = 0.05
    rebalance_utilization_ceiling = 1.0
    rebalance_utilization_min_forwards = 5

def test_realized_utilization_clamped_and_ratio(tmp_path):
    db = Database(str(tmp_path / "u.db"))
    now = 2_000_000
    # 6 outbound forwards over the long window, 600k sats out of a 1M-sat channel
    for i in range(6):
        _seed(db, out_channel="A", in_channel="B", out_msat=100_000_000, in_msat=100_000_000, ts=now - 100 * (i + 1))
    facts = compute_channel_flow_facts(db, "A", capacity_sats=1_000_000, now=now, cfg=_Cfg())
    # 600k/1M = 0.6, above floor/below ceiling, >= min_forwards -> realized
    assert abs(facts.realized_utilization - 0.6) < 1e-6
    assert facts.utilization_is_realized is True
    assert facts.forward_count_window == 6

def test_thin_history_falls_back_to_prior(tmp_path):
    db = Database(str(tmp_path / "t2.db"))
    now = 2_000_000
    _seed(db, out_channel="A", in_channel="B", out_msat=10_000_000, in_msat=10_000_000, ts=now - 50)
    facts = compute_channel_flow_facts(db, "A", capacity_sats=1_000_000, now=now, cfg=_Cfg())
    # only 1 forward < min_forwards(5) -> not realized; utilization is the prior sentinel 0.5
    assert facts.utilization_is_realized is False
    assert facts.realized_utilization == 0.5

def test_zero_capacity_is_safe(tmp_path):
    db = Database(str(tmp_path / "z.db"))
    facts = compute_channel_flow_facts(db, "A", capacity_sats=0, now=2_000_000, cfg=_Cfg())
    assert facts.realized_utilization == 0.5
    assert facts.utilization_is_realized is False
    assert facts.out_sats_window == 0 and facts.in_sats_window == 0

def test_util_clamped_to_ceiling(tmp_path):
    db = Database(str(tmp_path / "c.db"))
    now = 2_000_000
    for i in range(6):
        _seed(db, out_channel="A", in_channel="B", out_msat=500_000_000, in_msat=500_000_000, ts=now - 10 * (i + 1))
    facts = compute_channel_flow_facts(db, "A", capacity_sats=1_000_000, now=now, cfg=_Cfg())
    assert facts.realized_utilization == 1.0  # 3M/1M clamped to ceiling
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_rebalance_flow_facts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modules.rebalance_flow_facts'`

- [ ] **Step 3: Write the module**

```python
# modules/rebalance_flow_facts.py
"""Per-channel windowed forwarding facts for the rebalancer.

Pure adapter over Database.get_channel_flow_window. One responsibility:
turn (channel_id, capacity, now, config) into a ChannelFlowFacts bundle that
feeds three consumers — realized-utilization EV (#2), live-activity penalty
(#1), and (indirectly) size-tiered targets read capacity elsewhere.

Fail-open: on any DB error or zero capacity, returns neutral facts
(utilization = the 0.5 prior, zero net-flow) so the rebalancer degrades to
current behavior and never crashes.
"""
from dataclasses import dataclass

UTILIZATION_PRIOR = 0.5


@dataclass(frozen=True)
class ChannelFlowFacts:
    channel_id: str
    out_sats_window: int          # short activity window
    in_sats_window: int           # short activity window
    forward_count_window: int     # long utilization window count
    realized_utilization: float   # clamped ratio, or the prior when thin
    utilization_is_realized: bool  # False => value is the prior fallback


def _neutral(channel_id: str) -> ChannelFlowFacts:
    return ChannelFlowFacts(
        channel_id=channel_id,
        out_sats_window=0,
        in_sats_window=0,
        forward_count_window=0,
        realized_utilization=UTILIZATION_PRIOR,
        utilization_is_realized=False,
    )


def compute_channel_flow_facts(db, channel_id: str, capacity_sats: int, now: int, cfg) -> ChannelFlowFacts:
    if capacity_sats <= 0:
        return _neutral(channel_id)
    try:
        activity_window = int(getattr(cfg, "rebalance_activity_window_seconds", 3600) or 3600)
        util_days = int(getattr(cfg, "rebalance_utilization_window_days", 7) or 7)
        floor = float(getattr(cfg, "rebalance_utilization_floor", 0.05) or 0.05)
        ceiling = float(getattr(cfg, "rebalance_utilization_ceiling", 1.0) or 1.0)
        min_forwards = int(getattr(cfg, "rebalance_utilization_min_forwards", 5) or 5)

        short_since = now - activity_window
        long_since = now - util_days * 86_400

        short_out, short_in, _ = db.get_channel_flow_window(channel_id, short_since)
        long_out, _long_in, long_count = db.get_channel_flow_window(channel_id, long_since)

        if long_count >= min_forwards:
            raw = long_out / float(capacity_sats)
            realized = max(floor, min(ceiling, raw))
            is_realized = True
        else:
            realized = UTILIZATION_PRIOR
            is_realized = False

        return ChannelFlowFacts(
            channel_id=channel_id,
            out_sats_window=max(0, int(short_out)),
            in_sats_window=max(0, int(short_in)),
            forward_count_window=int(long_count),
            realized_utilization=realized,
            utilization_is_realized=is_realized,
        )
    except Exception:
        return _neutral(channel_id)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_rebalance_flow_facts.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```bash
git add modules/rebalance_flow_facts.py tests/test_rebalance_flow_facts.py
git commit -m "feat(rebalance): ChannelFlowFacts pure compute unit (realized utilization + activity net-flow)"
```

---

## Task 3: Config knobs (all three features)

**Files:**
- Modify: `modules/config.py` (OPTION_TYPES dict ~64-129; dataclass defaults ~336)
- Modify: `cl-revenue-ops.py` (register via `add_option`, mirror an existing `revenue-ops-rebalance-*` option)
- Test: `tests/test_rebalancer_module.py` (extend — reuse the P6-010 default-alignment test file)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rebalancer_module.py  (append)
from modules.config import Config

def test_upstream_pattern_defaults():
    c = Config()
    assert c.rebalance_activity_window_seconds == 3600
    assert c.rebalance_activity_penalty_coeff == 0.5
    assert c.rebalance_activity_penalty_cap_frac == 0.5
    assert c.rebalance_utilization_window_days == 7
    assert c.rebalance_utilization_floor == 0.05
    assert c.rebalance_utilization_ceiling == 1.0
    assert c.rebalance_utilization_min_forwards == 5
    assert c.rebalance_size_tiered_targets is True
    assert c.rebalance_size_reference_percentile == 0.5
    assert c.rebalance_small_channel_band_half_width == 0.15
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_rebalancer_module.py::test_upstream_pattern_defaults -v`
Expected: FAIL with `AttributeError` on the first missing field.

- [ ] **Step 3: Add the fields**

In `modules/config.py`, add to the `OPTION_TYPES` dict (near the other `rebalance_*` entries ~114-121):

```python
    'rebalance_activity_window_seconds': int,
    'rebalance_activity_penalty_coeff': float,
    'rebalance_activity_penalty_cap_frac': float,
    'rebalance_utilization_window_days': int,
    'rebalance_utilization_floor': float,
    'rebalance_utilization_ceiling': float,
    'rebalance_utilization_min_forwards': int,
    'rebalance_size_tiered_targets': bool,
    'rebalance_size_reference_percentile': float,
    'rebalance_small_channel_band_half_width': float,
```

And add dataclass defaults (near `rebalance_interval: int = 900` ~336):

```python
    rebalance_activity_window_seconds: int = 3600
    rebalance_activity_penalty_coeff: float = 0.5
    rebalance_activity_penalty_cap_frac: float = 0.5
    rebalance_utilization_window_days: int = 7
    rebalance_utilization_floor: float = 0.05
    rebalance_utilization_ceiling: float = 1.0
    rebalance_utilization_min_forwards: int = 5
    rebalance_size_tiered_targets: bool = True
    rebalance_size_reference_percentile: float = 0.5
    rebalance_small_channel_band_half_width: float = 0.15
```

- [ ] **Step 4: Register the plugin options** in `cl-revenue-ops.py`

Find an existing `plugin.add_option("revenue-ops-rebalance-...", ...)` call and add alongside it (repeat the pattern for each; example for two):

```python
    plugin.add_option(
        "revenue-ops-rebalance-activity-window-seconds", "3600",
        "Rebalancer: live-forwarding activity window (seconds) for the activity penalty.",
        opt_type="int",
    )
    plugin.add_option(
        "revenue-ops-rebalance-size-tiered-targets", "true",
        "Rebalancer: use per-channel size-tiered target bands (false = flat band).",
        opt_type="bool",
    )
    # ...repeat add_option for the remaining 8 knobs, matching names to the
    #    config fields (dash-form option name -> underscore config field).
```

- [ ] **Step 5: Run to verify it passes**

Run: `python3 -m pytest tests/test_rebalancer_module.py::test_upstream_pattern_defaults -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add modules/config.py cl-revenue-ops.py tests/test_rebalancer_module.py
git commit -m "feat(rebalance): config knobs for the three upstream patterns (registered options)"
```

---

## Task 4: Add flow-facts + target-band fields to `ChannelState`; thread into `build_state_snapshot`

**Files:**
- Modify: `modules/rebalance_state_v2.py` (`ChannelState` ~43-69; `build_state_snapshot` ~266-330)
- Test: `tests/test_rebalance_state_v2.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rebalance_state_v2.py  (append)
from modules.rebalance_state_v2 import ChannelState

def test_channel_state_has_flow_fact_fields_with_neutral_defaults():
    ch = ChannelState(
        channel_id="A", peer_id="B", capacity_sats=1_000_000, local_ratio=0.5,
        actual_inbound_fee_ppm=0, value_class="neutral", is_valuable=False,
        remaining_budget_sats=0, cooldown_active=False,
    )
    assert ch.realized_utilization == 0.5
    assert ch.utilization_is_realized is False
    assert ch.activity_out_sats == 0
    assert ch.activity_in_sats == 0
    assert ch.target_band_low == 0.35
    assert ch.target_band_high == 0.65
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_rebalance_state_v2.py::test_channel_state_has_flow_fact_fields_with_neutral_defaults -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword` or `AttributeError`.

- [ ] **Step 3: Add fields to `ChannelState`**

In `modules/rebalance_state_v2.py`, add to `ChannelState` (after `historical_sourced_fee_ppm` ~69):

```python
    # Upstream-pattern fields (#1 activity, #2 realized util, #3 per-channel band).
    realized_utilization: float = 0.5
    utilization_is_realized: bool = False
    activity_out_sats: int = 0
    activity_in_sats: int = 0
    target_band_low: float = 0.35
    target_band_high: float = 0.65
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_rebalance_state_v2.py::test_channel_state_has_flow_fact_fields_with_neutral_defaults -v`
Expected: PASS

- [ ] **Step 5: Populate the fields in `build_state_snapshot`**

`build_state_snapshot` (~266) gains an optional `flow_facts` map and an optional per-channel `target_bands` map (both default None → neutral). At the `ChannelState(...)` construction (~321), add:

```python
        facts = (flow_facts or {}).get(channel.channel_id)
        band = (target_bands or {}).get(channel.channel_id)
        # ...inside ChannelState(...):
                realized_utilization=(facts.realized_utilization if facts else 0.5),
                utilization_is_realized=(facts.utilization_is_realized if facts else False),
                activity_out_sats=(facts.out_sats_window if facts else 0),
                activity_in_sats=(facts.in_sats_window if facts else 0),
                target_band_low=(band[0] if band else _as_nonnegative_float(getattr(cfg, "low_liquidity_threshold", 0.35), 0.35)),
                target_band_high=(band[1] if band else _as_nonnegative_float(getattr(cfg, "high_liquidity_threshold", 0.65), 0.65)),
```

Update the `build_state_snapshot` signature to accept `flow_facts=None, target_bands=None, cfg=None` (keep backward-compatible defaults). Add a test that passing a `flow_facts` dict populates the fields.

```python
# tests/test_rebalance_state_v2.py  (append)
def test_build_state_snapshot_injects_flow_facts():
    from modules.rebalance_flow_facts import ChannelFlowFacts
    facts = {"A": ChannelFlowFacts("A", out_sats_window=1000, in_sats_window=0,
                                   forward_count_window=9, realized_utilization=0.7,
                                   utilization_is_realized=True)}
    # build a minimal channel input list per the existing helper in this test file,
    # call build_state_snapshot(channels, flow_facts=facts), assert channel A's
    # ChannelState.realized_utilization == 0.7 and utilization_is_realized is True.
```

- [ ] **Step 6: Run + commit**

Run: `python3 -m pytest tests/test_rebalance_state_v2.py -v`
Expected: PASS

```bash
git add modules/rebalance_state_v2.py tests/test_rebalance_state_v2.py
git commit -m "feat(rebalance): ChannelState carries flow facts + per-channel target band"
```

---

## Task 5: Compute the per-cycle flow-facts map (wire DB → snapshot)

**Files:**
- Modify: `modules/rebalance_engine_v2.py` (the snapshot-build call site that constructs `build_state_snapshot`, near the `low_liquidity_threshold` reads ~982 / ~1389)
- Test: `tests/test_rebalance_engine_v2.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rebalance_engine_v2.py  (append)
def test_engine_builds_flow_facts_map_from_db(monkeypatch):
    # Given an engine with a db exposing get_channel_flow_window, when the cycle
    # builds the snapshot, each ChannelState carries realized utilization from the db.
    # (Construct the engine per this file's existing fixtures; assert a channel's
    #  ChannelState.realized_utilization reflects a stubbed db.get_channel_flow_window.)
    ...
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/test_rebalance_engine_v2.py::test_engine_builds_flow_facts_map_from_db -v` → FAIL.

- [ ] **Step 3: Implement the wiring** — before the `build_state_snapshot(...)` call, compute the facts map:

```python
        from modules.rebalance_flow_facts import compute_channel_flow_facts
        now_ts = self._now()  # existing monotonic/wall-clock helper in the engine
        flow_facts = {}
        for ch in raw_channels:  # the same channel iterable feeding build_state_snapshot
            flow_facts[ch.channel_id] = compute_channel_flow_facts(
                self._db, ch.channel_id, int(getattr(ch, "capacity_sats", 0) or 0), now_ts, cfg
            )
```

Pass `flow_facts=flow_facts, cfg=cfg` into `build_state_snapshot(...)`. (Task 7 adds `target_bands`.)

- [ ] **Step 4: Run + Step 5: Commit**

```bash
git add modules/rebalance_engine_v2.py tests/test_rebalance_engine_v2.py
git commit -m "feat(rebalance): compute per-cycle ChannelFlowFacts map and inject into snapshot"
```

---

## Task 6: Feature #2 — realized utilization in the EV gate

**Files:**
- Modify: `modules/rebalance_types_v2.py` (`PairCandidate` ~47)
- Modify: `modules/rebalance_planner_v2.py` (`_generate_pairs` `PairCandidate(...)` ~332)
- Modify: `modules/rebalance_engine_v2.py` (EV terms ~502-520; `score_decomposition` ~571)
- Test: `tests/test_rebalance_engine_v2.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rebalance_engine_v2.py  (append)
def test_realized_utilization_replaces_constant_for_hot_channel():
    # A pair whose dest has realized utilization 0.8 must value its refill term
    # at amount * fee_ppm/1e6 * 0.8 (not the 0.5 constant). Build the decision dict
    # via the engine's score-decomposition path with a PairCandidate carrying
    # dest_realized_utilization=0.8, dest_value_fee_ppm implied by fees, and assert
    # decision["destination_refill_value_sats"] uses 0.8 and
    # decision["expected_utilization"] == 0.8 and decision["utilization_source"] == "realized".
    ...

def test_thin_history_pair_uses_prior():
    # dest_utilization_is_realized=False -> term uses 0.5 and utilization_source == "prior".
    ...
```

- [ ] **Step 2: Run to verify it fails** — FAIL (fields/behavior absent).

- [ ] **Step 3a: Add fields to `PairCandidate`** (`rebalance_types_v2.py` after ~47):

```python
    dest_realized_utilization: float = 0.5
    source_realized_utilization: float = 0.5
    dest_utilization_is_realized: bool = False
    source_utilization_is_realized: bool = False
```

- [ ] **Step 3b: Map them in `_generate_pairs`** (`rebalance_planner_v2.py`, inside the `PairCandidate(...)` ~332, alongside `dest_out_fee_ppm=`):

```python
                    dest_realized_utilization=float(getattr(dest, "realized_utilization", 0.5) or 0.5),
                    source_realized_utilization=float(getattr(src, "realized_utilization", 0.5) or 0.5),
                    dest_utilization_is_realized=bool(getattr(dest, "utilization_is_realized", False)),
                    source_utilization_is_realized=bool(getattr(src, "utilization_is_realized", False)),
```

- [ ] **Step 3c: Use them in the EV terms** (`rebalance_engine_v2.py` ~502-520). Replace the three `EXPECTED_UTILIZATION` usages with per-channel values (keep `EXPECTED_UTILIZATION` as the fallback constant):

```python
        dest_u = (
            float(getattr(pair, "dest_realized_utilization", EXPECTED_UTILIZATION))
            if getattr(pair, "dest_utilization_is_realized", False)
            else EXPECTED_UTILIZATION
        )
        source_u = (
            float(getattr(pair, "source_realized_utilization", EXPECTED_UTILIZATION))
            if getattr(pair, "source_utilization_is_realized", False)
            else EXPECTED_UTILIZATION
        )
        destination_refill_value_sats = (
            amount_sats * dest_value_fee_ppm / 1_000_000.0 * dest_u
        )
        source_drain_value_sats = (
            amount_sats * source_historical_sourced_fee_ppm / 1_000_000.0 * source_u
        )
        # source_opportunity keeps the source utilization (was EXPECTED_UTILIZATION):
        source_opportunity_sats = (
            amount_sats * source_opportunity_fee_ppm / 1_000_000.0
            * source_u * SOURCE_UTILIZATION_DISCOUNT
        )
```

- [ ] **Step 3d: Extend `score_decomposition`** (~571): change `"expected_utilization": EXPECTED_UTILIZATION,` to reflect the effective value and add the source:

```python
            "expected_utilization": round(dest_u, 6),
            "utilization_source": ("realized" if getattr(pair, "dest_utilization_is_realized", False) else "prior"),
            "source_utilization": round(source_u, 6),
```

- [ ] **Step 4: Run the tests** — `pytest tests/test_rebalance_engine_v2.py -q` → PASS.

- [ ] **Step 5: Mutation check**

Temporarily change `dest_u` selection to always use `EXPECTED_UTILIZATION`; run `test_realized_utilization_replaces_constant_for_hot_channel` → it MUST fail. Revert.

- [ ] **Step 6: Commit**

```bash
git add modules/rebalance_types_v2.py modules/rebalance_planner_v2.py modules/rebalance_engine_v2.py tests/test_rebalance_engine_v2.py
git commit -m "feat(rebalance): #2 per-channel realized-utilization EV basis (closes RE-H1/H2)"
```

---

## Task 7: Feature #3 — size-tiered target bands

**Files:**
- Modify: `modules/rebalance_state_v2.py` (add `compute_size_tiered_bands()` helper)
- Modify: `modules/rebalance_engine_v2.py` (build `target_bands` map, pass to `build_state_snapshot`)
- Modify: `modules/rebalance_planner_v2.py` (classify against per-channel bands)
- Test: `tests/test_rebalance_planner_v2.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rebalance_planner_v2.py  (append)
from modules.rebalance_state_v2 import compute_size_tiered_bands

def test_small_channel_keeps_flat_band():
    # capacities: three 1M channels + one 20M. reference percentile 0.5 -> ~1M.
    caps = {"s1": 1_000_000, "s2": 1_000_000, "s3": 1_000_000, "big": 20_000_000}
    bands = compute_size_tiered_bands(caps, percentile=0.5, small_half_width=0.15,
                                      flat_low=0.35, flat_high=0.65)
    assert bands["s1"] == (0.35, 0.65)          # small -> flat band preserved

def test_large_channel_gets_wider_asymmetric_band():
    caps = {"s1": 1_000_000, "s2": 1_000_000, "s3": 1_000_000, "big": 20_000_000}
    bands = compute_size_tiered_bands(caps, percentile=0.5, small_half_width=0.15,
                                      flat_low=0.35, flat_high=0.65)
    low, high = bands["big"]
    assert low < 0.35 and high > 0.65           # big channel allowed to hold residual
    assert 0.0 < low < high < 1.0

def test_toggle_off_restores_flat_band_in_planner():
    # With rebalance_size_tiered_targets False, every ChannelState.target_band_* stays
    # 0.35/0.65 and planner classification matches the pre-feature behavior exactly.
    ...
```

- [ ] **Step 2: Run to verify it fails** — FAIL (`compute_size_tiered_bands` absent).

- [ ] **Step 3a: Implement `compute_size_tiered_bands`** in `rebalance_state_v2.py`:

```python
def compute_size_tiered_bands(capacities, percentile=0.5, small_half_width=0.15,
                              flat_low=0.35, flat_high=0.65):
    """Per-channel (band_low, band_high) from the node's capacity distribution.

    Channels at/below the reference capacity keep the flat band (0.5 +/- half_width).
    Larger channels get a proportionally wider, downward-skewed band so they act as
    liquidity buffers (target away from 50/50) rather than being force-balanced.
    Returns {channel_id: (low, high)} with bounds clamped to (0.05, 0.95).
    """
    if not capacities:
        return {}
    caps_sorted = sorted(capacities.values())
    idx = min(len(caps_sorted) - 1, max(0, int(percentile * (len(caps_sorted) - 1))))
    reference = max(1, caps_sorted[idx])
    bands = {}
    for cid, cap in capacities.items():
        cap = max(1, int(cap))
        if cap <= reference:
            low, high = round(0.5 - small_half_width, 6), round(0.5 + small_half_width, 6)
        else:
            # widen with size, capped; skew downward so a big channel holds outbound residual
            scale = min(2.0, cap / float(reference))
            widen = small_half_width * scale
            low = 0.5 - widen * 1.3   # more room below 0.5 (buffer holds local)
            high = 0.5 + widen * 0.7
        bands[cid] = (round(max(0.05, min(low, 0.45)), 6),
                      round(min(0.95, max(high, 0.55)), 6))
    return bands
```

- [ ] **Step 3b: Build the map in the engine** (near the flow-facts map, Task 5):

```python
        target_bands = None
        if bool(getattr(cfg, "rebalance_size_tiered_targets", True)):
            from modules.rebalance_state_v2 import compute_size_tiered_bands
            caps = {ch.channel_id: int(getattr(ch, "capacity_sats", 0) or 0) for ch in raw_channels}
            target_bands = compute_size_tiered_bands(
                caps,
                percentile=float(getattr(cfg, "rebalance_size_reference_percentile", 0.5)),
                small_half_width=float(getattr(cfg, "rebalance_small_channel_band_half_width", 0.15)),
                flat_low=float(getattr(cfg, "low_liquidity_threshold", 0.35)),
                flat_high=float(getattr(cfg, "high_liquidity_threshold", 0.65)),
            )
```

Pass `target_bands=target_bands` into `build_state_snapshot(...)`.

- [ ] **Step 3c: Planner uses per-channel bands** (`rebalance_planner_v2.py` ~129/139). Replace `self.target_band_high`/`self.target_band_low` in the classification loop with the channel's own band:

```python
        for ch in snapshot.channels:
            band_high = getattr(ch, "target_band_high", self.target_band_high)
            band_low = getattr(ch, "target_band_low", self.target_band_low)
            if ch.local_ratio > band_high:
                ...
            elif ch.local_ratio < band_low:
                ...
```

(The scalar `self.target_band_*` remains the fallback for channels without a per-channel band.)

- [ ] **Step 4: Run** — `pytest tests/test_rebalance_planner_v2.py tests/test_rebalance_state_v2.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/rebalance_state_v2.py modules/rebalance_engine_v2.py modules/rebalance_planner_v2.py tests/test_rebalance_planner_v2.py
git commit -m "feat(rebalance): #3 size-tiered ideal-ratio target bands (flat-band fallback via toggle)"
```

---

## Task 8: Feature #1 — live-activity score penalty

**Files:**
- Modify: `modules/rebalance_engine_v2.py` (score composition ~521-530; `score_decomposition` ~569)
- Test: `tests/test_rebalance_engine_v2.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rebalance_engine_v2.py  (append)
def test_activity_penalty_lowers_score_for_helpful_live_flow():
    # Two identical pairs; the one whose source has activity_out_sats>0 (live traffic
    # already draining it the helpful way) must score LOWER by activity_penalty_sats.
    ...

def test_activity_penalty_is_capped():
    # Even with huge activity_out_sats, penalty <= cap_frac * gross value term, so a
    # strongly-EV pair still has final_score_sats >= 0 (beats_do_nothing True).
    ...
```

- [ ] **Step 2: Run to verify it fails** — FAIL.

- [ ] **Step 3: Implement the penalty** (`rebalance_engine_v2.py`, before `final_score_sats` ~524). "Helpful" flow = source being drained outbound (`source_activity_out_sats`) — supply these on `PairCandidate` (add `source_activity_out_sats: int = 0`, `dest_activity_in_sats: int = 0` in `rebalance_types_v2.py`, and map from ChannelState in `_generate_pairs`, mirroring Task 6 Step 3b):

```python
        activity_coeff = float(getattr(self._cfg, "rebalance_activity_penalty_coeff", 0.5) or 0.0)
        activity_cap_frac = float(getattr(self._cfg, "rebalance_activity_penalty_cap_frac", 0.5) or 0.0)
        helpful_flow_sats = int(getattr(pair, "source_activity_out_sats", 0) or 0) + \
                            int(getattr(pair, "dest_activity_in_sats", 0) or 0)
        raw_penalty = activity_coeff * helpful_flow_sats * dest_value_fee_ppm / 1_000_000.0
        penalty_cap = activity_cap_frac * max(0.0, expected_future_value_sats)
        activity_penalty_sats = round(min(raw_penalty, penalty_cap), 6)

        final_score_sats = round(
            p_success * expected_future_value_sats
            - expected_fee_sats
            - source_opportunity_sats
            - failure_penalty_sats
            - activity_penalty_sats,
            6,
        )
```

Add to `score_decomposition`: `"activity_penalty_sats": activity_penalty_sats,`.

- [ ] **Step 4: Run** → PASS. **Mutation check:** set `activity_penalty_sats = 0.0`; the first test must fail. Revert.

- [ ] **Step 5: Commit**

```bash
git add modules/rebalance_types_v2.py modules/rebalance_planner_v2.py modules/rebalance_engine_v2.py tests/test_rebalance_engine_v2.py
git commit -m "feat(rebalance): #1 capped live-activity score penalty"
```

---

## Task 9: Full regression + live-node verification

**Files:** none (verification only)

- [ ] **Step 1: Full suite green**

Run: `python3 -m pytest tests/ -q`
Expected: all pass (no regressions; the new tests included).

- [ ] **Step 2: Budget invariant untouched**

Run: `python3 -m pytest tests/test_all_spenders_atomic.py tests/test_cross_category_budget_atomicity.py -q`
Expected: PASS — proves no spend/budget path changed.

- [ ] **Step 3: Deep-audit standing gate green**

Run: `python3 tools/audit/scorecard.py --deep-only`
Expected: `deep summary: PASS=7`.

- [ ] **Step 4: Merge to main + push** (only on operator go-ahead)

```bash
git checkout main && git merge --no-ff rebalancer-upstream-patterns
python3 -m pytest tests/ -q   # re-confirm on main
git push origin main
```

- [ ] **Step 5: Deploy + observe on hive-nexus-02**

After deploy (plugin restart, per the established procedure), inspect the live decision surface:
```
docker exec 943474cc1057 lightning-cli --lightning-dir=/data/lightning/bitcoin revenue-status
```
Confirm in a rebalance decision's `score_decomposition`: `utilization_source` shows "realized" for active channels, `activity_penalty_sats` appears, and `expected_utilization` varies per channel (no longer a constant 0.5). Confirm `revenue-health.budget` is unchanged and no daemon loop stalled.

---

## Self-Review

**Spec coverage:**
- #2 realized-utilization EV → Tasks 1,2,4,5,6 ✓ (DB → facts → ChannelState → PairCandidate → EV + decomposition).
- #3 size-tiered targets → Task 7 ✓ (compute + wire + planner consume + toggle).
- #1 activity penalty → Tasks 1,2 (net-flow facts) + 8 ✓ (capped penalty + decomposition).
- Shared `ChannelFlowFacts` foundation → Tasks 1–5 ✓.
- Config knobs, all tunable, registered → Task 3 ✓.
- Safety invariant (no spend/budget path touched) → asserted by Task 9 Steps 2–3 ✓.
- Fail-open degradation → `_neutral()` in Task 2 ✓.

**Placeholder scan:** The `...` bodies in Task 5/6/7/8 test stubs describe the assertion precisely but omit fixture boilerplate specific to each test file's existing helpers; the implementer fills these using the same-file fixtures (the *behavior* asserted is fully specified). All implementation-code steps contain complete code.

**Type consistency:** Field names are consistent across tasks — `realized_utilization` / `utilization_is_realized` / `activity_out_sats` / `activity_in_sats` / `target_band_low` / `target_band_high` on `ChannelState`; `dest_realized_utilization` / `source_realized_utilization` / `dest_utilization_is_realized` / `source_utilization_is_realized` / `source_activity_out_sats` / `dest_activity_in_sats` on `PairCandidate`; `compute_channel_flow_facts` / `compute_size_tiered_bands` / `get_channel_flow_window` used identically where referenced.

**Build order:** foundation (1–5) precedes consumers (6,7,8); each feature is independently mergeable after the shared foundation.
