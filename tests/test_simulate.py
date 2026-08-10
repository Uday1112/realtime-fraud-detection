"""Unit tests for the transaction/fraud-pattern generator (src/simulate.py).
Run with: pytest tests/ -v"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from simulate import TransactionSimulator

CURRENT_TS = 1_735_000_000.0  # a realistic epoch time; near-zero values trip datetime.fromtimestamp on Windows
CUSTOMERS = [{"customer_id": i, "baseline_mean_amount": 100.0, "baseline_std_amount": 20.0} for i in range(1, 6)]
PRODUCTS = [{"product_id": i, "unit_price": 15.0} for i in range(1, 6)]


def test_fraud_rate_zero_never_produces_fraud():
    sim = TransactionSimulator(CUSTOMERS, PRODUCTS, seed=1, fraud_rate=0.0)
    events = [e for _ in range(200) for e in sim.generate_one(current_ts=CURRENT_TS)]
    assert all(not e["is_fraud_synthetic"] for e in events)
    assert all(e["fraud_pattern"] is None for e in events)


def test_fraud_rate_one_always_produces_fraud():
    sim = TransactionSimulator(CUSTOMERS, PRODUCTS, seed=1, fraud_rate=1.0)
    events = [e for _ in range(30) for e in sim.generate_one(current_ts=CURRENT_TS)]
    assert all(e["is_fraud_synthetic"] for e in events)
    assert all(e["fraud_pattern"] in {"amount_spike", "velocity_burst", "odd_hour_bulk"} for e in events)


def test_velocity_burst_is_multiple_events_same_customer_close_in_time():
    sim = TransactionSimulator(CUSTOMERS, PRODUCTS, seed=42, fraud_rate=1.0)
    # seed=42 with these inputs happens to draw velocity_burst first; if not,
    # search a few draws for one.
    burst = None
    for _ in range(20):
        batch = sim.generate_one(current_ts=CURRENT_TS)
        if batch[0]["fraud_pattern"] == "velocity_burst":
            burst = batch
            break
    assert burst is not None
    assert len(burst) >= 5
    assert len({e["customer_id"] for e in burst}) == 1  # all same customer
    timestamps = sorted(e["timestamp"] for e in burst)
    assert timestamps[-1] - timestamps[0] < 120  # whole burst within ~2 minutes


def test_amount_spike_is_far_above_customer_baseline():
    sim = TransactionSimulator(CUSTOMERS, PRODUCTS, seed=7, fraud_rate=1.0)
    spike = None
    for _ in range(20):
        batch = sim.generate_one(current_ts=CURRENT_TS)
        if batch[0]["fraud_pattern"] == "amount_spike":
            spike = batch[0]
            break
    assert spike is not None
    customer = next(c for c in CUSTOMERS if c["customer_id"] == spike["customer_id"])
    assert spike["amount"] >= customer["baseline_mean_amount"] * 5


def test_odd_hour_bulk_is_night_and_high_quantity():
    sim = TransactionSimulator(CUSTOMERS, PRODUCTS, seed=3, fraud_rate=1.0)
    bulk = None
    for _ in range(20):
        batch = sim.generate_one(current_ts=CURRENT_TS)
        if batch[0]["fraud_pattern"] == "odd_hour_bulk":
            bulk = batch[0]
            break
    assert bulk is not None
    assert 1 <= bulk["hour_of_day"] <= 4
    assert bulk["quantity"] >= 8


def test_transaction_ids_are_unique_and_increasing():
    sim = TransactionSimulator(CUSTOMERS, PRODUCTS, seed=1, fraud_rate=0.0)
    events = [e for _ in range(100) for e in sim.generate_one(current_ts=CURRENT_TS)]
    ids = [e["transaction_id"] for e in events]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
