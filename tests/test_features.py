"""Unit tests for the streaming feature engineering (src/features.py).
Run with: pytest tests/ -v"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from features import GAP_CAP_SECONDS, CustomerState, compute_features, seed_state_from_baseline


def test_new_customer_has_no_history_signal():
    state = CustomerState(customer_id=1)
    feats = compute_features(amount=50.0, quantity=1, ts_epoch=1000.0, hour_of_day=14, state=state)
    assert feats["is_new_customer"] == 1
    assert feats["seconds_since_last_txn"] == GAP_CAP_SECONDS
    assert feats["velocity_5min"] == 0


def test_amount_zscore_reflects_deviation_from_customer_mean():
    state = seed_state_from_baseline(customer_id=1, baseline_mean=100.0, baseline_std=10.0, prior_count=30)
    normal_feats = compute_features(amount=100.0, quantity=1, ts_epoch=1000.0, hour_of_day=14, state=state)
    spike_feats = compute_features(amount=1000.0, quantity=1, ts_epoch=1000.0, hour_of_day=14, state=state)
    assert abs(normal_feats["amount_zscore"]) < 1
    assert spike_feats["amount_zscore"] > 10  # a 10x spend for this customer should stand out sharply


def test_velocity_counts_only_recent_transactions():
    state = CustomerState(customer_id=1)
    for ts in [1_735_000_000, 1_735_000_060, 1_735_000_120, 1_735_004_000]:
        state.update(amount=50.0, ts=ts)  # first 3 within 5 min of the query, last one far outside
    feats = compute_features(amount=50.0, quantity=1, ts_epoch=1_735_000_150.0, hour_of_day=14, state=state)
    assert feats["velocity_5min"] == 3


def test_velocity_ignores_out_of_order_future_timestamps():
    """A timestamp after the query time shouldn't count as 'recent' just
    because the raw gap is small in magnitude (regression guard)."""
    state = CustomerState(customer_id=1)
    state.update(amount=50.0, ts=1_735_000_500.0)  # 500s after the query below
    feats = compute_features(amount=50.0, quantity=1, ts_epoch=1_735_000_000.0, hour_of_day=14, state=state)
    assert feats["velocity_5min"] == 0


def test_seconds_since_last_txn_capped_not_unbounded():
    state = CustomerState(customer_id=1)
    state.update(amount=50.0, ts=0.0)
    feats = compute_features(amount=50.0, quantity=1, ts_epoch=1_000_000.0, hour_of_day=14, state=state)
    assert feats["seconds_since_last_txn"] == GAP_CAP_SECONDS  # capped, not the raw (huge) gap


def test_rapid_succession_gives_small_gap():
    state = CustomerState(customer_id=1)
    state.update(amount=50.0, ts=1000.0)
    feats = compute_features(amount=50.0, quantity=1, ts_epoch=1003.0, hour_of_day=14, state=state)
    assert feats["seconds_since_last_txn"] == 3.0


def test_is_night_flag():
    state = CustomerState(customer_id=1)
    day_feats = compute_features(amount=50.0, quantity=1, ts_epoch=1000.0, hour_of_day=14, state=state)
    night_feats = compute_features(amount=50.0, quantity=1, ts_epoch=1000.0, hour_of_day=3, state=state)
    assert day_feats["is_night"] == 0
    assert night_feats["is_night"] == 1


def test_customer_state_update_is_order_independent_of_features_call():
    """compute_features must reflect state BEFORE update — the model should
    never see an event's own outcome baked into its own features."""
    state = CustomerState(customer_id=1)
    state.update(amount=100.0, ts=0.0)
    state.update(amount=100.0, ts=10.0)
    feats_before_third = compute_features(amount=9999.0, quantity=1, ts_epoch=20.0, hour_of_day=14, state=state)
    assert state.count == 2  # not yet incremented by the amount=9999 event
    assert feats_before_third["amount_zscore"] > 0
