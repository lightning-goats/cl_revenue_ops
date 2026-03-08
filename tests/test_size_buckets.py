"""Tests for payment size bucket profiling."""
import random
import sqlite3
import time


def test_classify_size_bucket_micro():
    from modules.fee_controller import classify_size_bucket
    assert classify_size_bucket(0) == 0
    assert classify_size_bucket(9_999) == 0


def test_classify_size_bucket_small():
    from modules.fee_controller import classify_size_bucket
    assert classify_size_bucket(10_000) == 1
    assert classify_size_bucket(99_999) == 1


def test_classify_size_bucket_medium():
    from modules.fee_controller import classify_size_bucket
    assert classify_size_bucket(100_000) == 2
    assert classify_size_bucket(499_999) == 2


def test_classify_size_bucket_large():
    from modules.fee_controller import classify_size_bucket
    assert classify_size_bucket(500_000) == 3
    assert classify_size_bucket(4_999_999) == 3


def test_classify_size_bucket_whale():
    from modules.fee_controller import classify_size_bucket
    assert classify_size_bucket(5_000_000) == 4
    assert classify_size_bucket(100_000_000) == 4


def test_bucket_posterior_defaults():
    from modules.fee_controller import BucketPosterior
    bp = BucketPosterior()
    assert bp.mu == 200.0
    assert bp.precision == 0.1
    assert bp.n_obs == 0
    assert bp.revenue_share == 0.0
    assert not bp.graduated


def test_bucket_posterior_update():
    from modules.fee_controller import BucketPosterior
    bp = BucketPosterior(mu=200.0, precision=0.1, n_obs=0)
    bp.update(observed_fee=300.0, noise_variance=1000.0)
    assert bp.n_obs == 1
    assert bp.precision > 0.1  # precision increased
    # Posterior mean should shift toward 300
    assert bp.mu > 200.0


def test_bucket_posterior_graduation():
    from modules.fee_controller import BucketPosterior, SIZE_BUCKET_GRADUATION_THRESHOLD
    bp = BucketPosterior(n_obs=SIZE_BUCKET_GRADUATION_THRESHOLD - 1)
    assert not bp.graduated
    bp.n_obs += 1
    assert bp.graduated


def test_bucket_posterior_sample_clamped():
    from modules.fee_controller import BucketPosterior
    random.seed(42)
    bp = BucketPosterior(mu=50.0, precision=100.0)
    # With high precision centered at 50, samples near 50
    # But clamped to [100, 500]
    fee = bp.sample(floor=100, ceiling=500)
    assert 100 <= fee <= 500


def test_bucket_posterior_serialization():
    from modules.fee_controller import BucketPosterior
    bp = BucketPosterior(mu=150.0, precision=3.0, n_obs=25, revenue_share=0.4)
    d = bp.to_dict()
    bp2 = BucketPosterior.from_dict(d)
    assert bp2.mu == 150.0
    assert bp2.precision == 3.0
    assert bp2.n_obs == 25
    assert bp2.revenue_share == 0.4


def test_size_bucket_state_defaults():
    from modules.fee_controller import SizeBucketState, SIZE_BUCKET_LABELS
    state = SizeBucketState()
    assert len(state.buckets) == 5
    for label in SIZE_BUCKET_LABELS:
        assert label in state.buckets
        assert not state.buckets[label].graduated


def test_size_bucket_state_update():
    from modules.fee_controller import SizeBucketState
    state = SizeBucketState()
    state.update_bucket(amount_sats=50_000, fee_ppm=200.0)  # small bucket
    assert state.buckets["small"].n_obs == 1
    assert state.buckets["micro"].n_obs == 0  # others unchanged


def test_size_bucket_state_serialization():
    from modules.fee_controller import SizeBucketState
    state = SizeBucketState()
    for _ in range(15):
        state.update_bucket(amount_sats=50_000, fee_ppm=200.0)
    d = state.to_dict()
    state2 = SizeBucketState.from_dict(d)
    assert state2.buckets["small"].n_obs == 15
    assert state2.buckets["small"].graduated


def test_size_bucket_state_revenue_shares():
    from modules.fee_controller import SizeBucketState
    state = SizeBucketState()
    state.update_revenue_shares({"micro": 0.1, "small": 0.4, "whale": 0.5})
    assert state.buckets["micro"].revenue_share == 0.1
    assert state.buckets["small"].revenue_share == 0.4
    assert state.buckets["whale"].revenue_share == 0.5
    assert state.buckets["medium"].revenue_share == 0.0  # unchanged


def _create_test_db():
    """Create an in-memory database with forwards table for testing."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE forwards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            in_channel TEXT NOT NULL,
            out_channel TEXT NOT NULL,
            in_msat INTEGER NOT NULL,
            out_msat INTEGER NOT NULL,
            fee_msat INTEGER NOT NULL,
            resolution_time REAL DEFAULT 0,
            timestamp INTEGER NOT NULL,
            resolved_time INTEGER DEFAULT 0
        )
    """)
    return conn


def _insert_forward(conn, out_channel, out_msat, fee_msat, timestamp):
    """Insert a test forward."""
    conn.execute(
        "INSERT INTO forwards (in_channel, out_channel, in_msat, out_msat, fee_msat, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("in_ch", out_channel, out_msat, out_msat, fee_msat, timestamp),
    )


def test_get_revenue_by_size_bucket_basic():
    from modules.database import _revenue_by_size_bucket_sql
    conn = _create_test_db()
    now = int(time.time())
    # Insert forwards in different size buckets (out_msat is in millisatoshis)
    _insert_forward(conn, "ch1", 5_000_000, 100, now - 3600)      # micro: 5k sats
    _insert_forward(conn, "ch1", 50_000_000, 500, now - 3600)     # small: 50k sats
    _insert_forward(conn, "ch1", 50_000_000, 500, now - 3600)     # small: another
    _insert_forward(conn, "ch1", 200_000_000, 2000, now - 3600)   # medium: 200k sats
    conn.commit()
    result = _revenue_by_size_bucket_sql(conn, "ch1", window_days=7)
    assert result["micro"] == 100
    assert result["small"] == 1000  # 500 + 500
    assert result["medium"] == 2000
    assert result["large"] == 0
    assert result["whale"] == 0


def test_get_revenue_by_size_bucket_window():
    from modules.database import _revenue_by_size_bucket_sql
    conn = _create_test_db()
    now = int(time.time())
    _insert_forward(conn, "ch1", 50_000_000, 500, now - 3600)         # within 7 days
    _insert_forward(conn, "ch1", 50_000_000, 500, now - 8 * 86400)    # outside 7 days
    conn.commit()
    result = _revenue_by_size_bucket_sql(conn, "ch1", window_days=7)
    assert result["small"] == 500  # only the recent one


def test_revenue_shares_from_totals():
    from modules.fee_controller import compute_revenue_shares
    totals = {"micro": 100, "small": 400, "medium": 300, "large": 200, "whale": 0}
    shares = compute_revenue_shares(totals)
    assert abs(shares["micro"] - 0.1) < 1e-6
    assert abs(shares["small"] - 0.4) < 1e-6
    assert abs(shares["medium"] - 0.3) < 1e-6
    assert abs(shares["large"] - 0.2) < 1e-6
    assert shares["whale"] == 0.0
    assert abs(sum(shares.values()) - 1.0) < 1e-6


def test_revenue_shares_all_zero():
    from modules.fee_controller import compute_revenue_shares
    totals = {"micro": 0, "small": 0, "medium": 0, "large": 0, "whale": 0}
    shares = compute_revenue_shares(totals)
    for label in shares:
        assert shares[label] == 0.0


def test_size_bucket_labels_consistent():
    """Verify database and fee_controller agree on bucket labels."""
    from modules.database import _SIZE_BUCKET_LABELS
    from modules.fee_controller import SIZE_BUCKET_LABELS
    assert list(_SIZE_BUCKET_LABELS) == SIZE_BUCKET_LABELS


def test_size_bucket_sql_boundaries_match_python():
    """Verify SQL CASE boundaries agree with Python classify_size_bucket."""
    from modules.fee_controller import classify_size_bucket, SIZE_BUCKET_BOUNDARIES, SIZE_BUCKET_LABELS
    from modules.database import _revenue_by_size_bucket_sql
    conn = _create_test_db()
    now = int(time.time())
    # Test exact boundary values through SQL
    test_amounts_sats = [9_999, 10_000, 99_999, 100_000, 499_999, 500_000, 4_999_999, 5_000_000]
    for amt_sats in test_amounts_sats:
        out_msat = amt_sats * 1000
        _insert_forward(conn, f"ch_{amt_sats}", out_msat, 100, now - 3600)
    conn.commit()
    for amt_sats in test_amounts_sats:
        result = _revenue_by_size_bucket_sql(conn, f"ch_{amt_sats}", window_days=7)
        python_idx = classify_size_bucket(amt_sats)
        python_label = SIZE_BUCKET_LABELS[python_idx]
        assert result[python_label] == 100, f"SQL/Python disagree for {amt_sats} sats: SQL={result}, Python={python_label}"


def test_composite_fee_all_cold():
    """No graduated buckets -> returns channel-wide sample unchanged."""
    from modules.fee_controller import SizeBucketState
    state = SizeBucketState()
    # All buckets have 0 observations -> not graduated
    result = state.composite_sample(
        channel_wide_sample=250.0, floor=10, ceiling=5000
    )
    assert result == 250.0  # channel-wide passthrough


def test_composite_fee_partial_graduation():
    """Some graduated buckets weighted by revenue share, rest to channel-wide."""
    from modules.fee_controller import SizeBucketState
    random.seed(99)
    state = SizeBucketState()
    # Graduate "small" bucket with tight posterior around 300
    state.buckets["small"].n_obs = 50
    state.buckets["small"].mu = 300.0
    state.buckets["small"].precision = 10000.0  # very tight -> samples ~300
    state.buckets["small"].revenue_share = 0.6

    # Graduate "medium" bucket with tight posterior around 150
    state.buckets["medium"].n_obs = 20
    state.buckets["medium"].mu = 150.0
    state.buckets["medium"].precision = 10000.0
    state.buckets["medium"].revenue_share = 0.3

    # Ungraduated buckets have 0.1 combined share
    channel_wide = 200.0
    result = state.composite_sample(channel_wide, floor=10, ceiling=5000)

    # graduated_share = 0.6 + 0.3 = 0.9
    # Expected: ~300*0.6 + ~150*0.3 + 200*0.1 = 180 + 45 + 20 = 245
    assert 200 < result < 290  # reasonable range given tight posteriors


def test_composite_fee_full_graduation():
    """All buckets graduated -> zero channel-wide weight."""
    from modules.fee_controller import SizeBucketState, SIZE_BUCKET_LABELS
    random.seed(42)
    state = SizeBucketState()
    target_fees = [100.0, 200.0, 300.0, 250.0, 150.0]
    shares = [0.05, 0.40, 0.30, 0.20, 0.05]
    for i, label in enumerate(SIZE_BUCKET_LABELS):
        state.buckets[label].n_obs = 50
        state.buckets[label].mu = target_fees[i]
        state.buckets[label].precision = 10000.0
        state.buckets[label].revenue_share = shares[i]

    result = state.composite_sample(
        channel_wide_sample=999.0, floor=10, ceiling=5000
    )
    # channel_wide should have zero influence
    # Expected: ~100*0.05 + ~200*0.40 + ~300*0.30 + ~250*0.20 + ~150*0.05
    #         = 5 + 80 + 90 + 50 + 7.5 = 232.5
    assert 190 < result < 275
    assert result != 999.0  # channel_wide NOT used


def test_composite_fee_clamped():
    """Composite result clamped to floor/ceiling."""
    from modules.fee_controller import SizeBucketState
    random.seed(42)
    state = SizeBucketState()
    state.buckets["small"].n_obs = 50
    state.buckets["small"].mu = 5.0  # very low
    state.buckets["small"].precision = 10000.0
    state.buckets["small"].revenue_share = 1.0
    result = state.composite_sample(channel_wide_sample=5.0, floor=100, ceiling=5000)
    assert result >= 100  # clamped to floor


def test_v2_state_json_without_size_buckets():
    """Existing v2_state_json without size_buckets key works (graceful absence)."""
    from modules.fee_controller import ThompsonAIMDState
    # Simulate loading old state that has no size_buckets key
    v2_data = {
        "algorithm_version": "thompson_aimd_v1",
        "thompson_state": {},
        "aimd_state": {},
    }
    db_state = {
        "last_revenue_rate": 0.0,
        "last_fee_ppm": 200,
        "trend_direction": 1,
        "step_ppm": 50,
        "consecutive_same_direction": 0,
        "last_update": 0,
        "v2_state_json": "{}",
    }
    state = ThompsonAIMDState.from_v2_dict(v2_data, db_state)
    # size_buckets should exist with defaults
    assert state.size_buckets is not None
    assert not any(b.graduated for b in state.size_buckets.buckets.values())


def test_v2_state_json_roundtrip_with_size_buckets():
    """Size bucket state survives serialization roundtrip."""
    from modules.fee_controller import ThompsonAIMDState
    state = ThompsonAIMDState()
    state.size_buckets.update_bucket(50_000, 200.0)  # small bucket
    state.size_buckets.update_bucket(50_000, 250.0)
    state.size_buckets.buckets["small"].revenue_share = 0.7

    v2 = state.to_v2_dict()
    assert "size_buckets" in v2
    assert v2["size_buckets"]["small"]["n_obs"] == 2

    # Roundtrip
    db_state = {"v2_state_json": "{}", "last_revenue_rate": 0.0,
                "last_fee_ppm": 200, "trend_direction": 1, "step_ppm": 50,
                "consecutive_same_direction": 0, "last_update": 0}
    state2 = ThompsonAIMDState.from_v2_dict(v2, db_state)
    assert state2.size_buckets.buckets["small"].n_obs == 2
    assert state2.size_buckets.buckets["small"].revenue_share == 0.7
