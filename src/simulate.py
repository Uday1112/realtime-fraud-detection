"""
Transaction event generator shared by offline historical generation and the
live stream. Emits mostly normal transactions plus three injected fraud
patterns, each modeled on a real card-fraud signature:

- amount_spike     a single purchase far outside the customer's own normal
                    spend (account takeover / stolen card testing a big buy)
- velocity_burst    many transactions from the same customer in a short
                    window (card testing / bot checkout abuse)
- odd_hour_bulk     a large-quantity purchase at 1-5am (reseller / bulk
                    fraud pattern flagged by real fraud teams)

The ground-truth `is_fraud_synthetic` label is carried on every event but is
used ONLY for offline evaluation — the model itself is unsupervised and
never sees it, which mirrors production fraud detection where confirmed
fraud labels are scarce and delayed.
"""

from datetime import datetime, timedelta

import numpy as np

# Weekday shopping activity: low overnight, rising through the day,
# peaking in the evening. Used to make normal-event timestamps realistic.
HOUR_WEIGHTS = np.array(
    [0.2, 0.15, 0.1, 0.1, 0.1, 0.15,
     0.4, 0.8, 1.2, 1.4, 1.5, 1.6,
     1.7, 1.6, 1.5, 1.4, 1.4, 1.5,
     1.8, 1.9, 1.7, 1.3, 0.9, 0.5]
)
HOUR_WEIGHTS = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()

FRAUD_PATTERNS = ["amount_spike", "velocity_burst", "odd_hour_bulk"]


class TransactionSimulator:
    def __init__(self, customers: list[dict], products: list[dict], seed: int = 123, fraud_rate: float = 0.025):
        self.customer_by_id = {c["customer_id"]: c for c in customers}
        self.product_by_id = {p["product_id"]: p for p in products}
        self.customer_ids = list(self.customer_by_id.keys())
        self.product_ids = list(self.product_by_id.keys())
        self.rng = np.random.default_rng(seed)
        self.fraud_rate = fraud_rate
        self._next_txn_id = 1

    def _new_id(self) -> int:
        tid = self._next_txn_id
        self._next_txn_id += 1
        return tid

    def _sample_hour(self) -> int:
        return int(self.rng.choice(np.arange(24), p=HOUR_WEIGHTS))

    def _random_timestamp(self, days_back_max: int) -> float:
        day_offset = int(self.rng.integers(0, days_back_max))
        dt = (datetime.now() - timedelta(days=day_offset)).replace(
            hour=self._sample_hour(),
            minute=int(self.rng.integers(0, 60)),
            second=int(self.rng.integers(0, 60)),
            microsecond=0,
        )
        return dt.timestamp()

    def _build(self, ts: float, cid: int, pid: int, quantity: int, amount: float, is_fraud: bool, pattern: str | None) -> dict:
        dt = datetime.fromtimestamp(ts)
        return {
            "transaction_id": self._new_id(),
            "customer_id": cid,
            "product_id": pid,
            "quantity": quantity,
            "amount": round(amount, 2),
            "timestamp": ts,
            "datetime": dt.isoformat(timespec="seconds"),
            "hour_of_day": dt.hour,
            "is_fraud_synthetic": is_fraud,
            "fraud_pattern": pattern,
        }

    def _make_normal(self, ts: float) -> dict:
        cid = int(self.rng.choice(self.customer_ids))
        pid = int(self.rng.choice(self.product_ids))
        customer = self.customer_by_id[cid]
        quantity = int(self.rng.choice([1, 1, 1, 1, 2, 2, 3]))
        amount = max(3.0, float(self.rng.normal(customer["baseline_mean_amount"], customer["baseline_std_amount"])))
        return self._build(ts, cid, pid, quantity, amount, False, None)

    def _make_amount_spike(self, ts: float) -> dict:
        cid = int(self.rng.choice(self.customer_ids))
        pid = int(self.rng.choice(self.product_ids))
        customer = self.customer_by_id[cid]
        multiplier = float(self.rng.uniform(8, 20))
        amount = customer["baseline_mean_amount"] * multiplier
        quantity = int(self.rng.choice([1, 1, 2]))
        return self._build(ts, cid, pid, quantity, amount, True, "amount_spike")

    def _make_odd_hour_bulk(self, ts: float) -> dict:
        cid = int(self.rng.choice(self.customer_ids))
        pid = int(self.rng.choice(self.product_ids))
        quantity = int(self.rng.integers(8, 20))
        unit_price = self.product_by_id[pid]["unit_price"]
        amount = unit_price * quantity
        night_hour = int(self.rng.integers(1, 5))
        dt = datetime.fromtimestamp(ts).replace(hour=night_hour)
        return self._build(dt.timestamp(), cid, pid, quantity, amount, True, "odd_hour_bulk")

    def _make_velocity_burst(self, ts: float) -> list[dict]:
        cid = int(self.rng.choice(self.customer_ids))
        customer = self.customer_by_id[cid]
        burst_size = int(self.rng.integers(5, 11))
        events = []
        t = ts
        for _ in range(burst_size):
            pid = int(self.rng.choice(self.product_ids))
            quantity = int(self.rng.choice([1, 1, 2]))
            amount = max(3.0, float(self.rng.normal(customer["baseline_mean_amount"] * 0.8, customer["baseline_std_amount"])))
            events.append(self._build(t, cid, pid, quantity, amount, True, "velocity_burst"))
            t += float(self.rng.uniform(2, 8))
        return events

    def generate_one(self, current_ts: float | None = None, historical_days_back: int = 30) -> list[dict]:
        """Returns 1+ events (a velocity_burst yields several). If
        current_ts is None, a random historical timestamp is used (offline
        generation); otherwise events are anchored to the given time (live
        streaming)."""
        anchor_ts = current_ts if current_ts is not None else self._random_timestamp(historical_days_back)

        if self.rng.random() < self.fraud_rate:
            pattern = self.rng.choice(FRAUD_PATTERNS)
            if pattern == "amount_spike":
                return [self._make_amount_spike(anchor_ts)]
            if pattern == "odd_hour_bulk":
                return [self._make_odd_hour_bulk(anchor_ts)]
            return self._make_velocity_burst(anchor_ts)

        return [self._make_normal(anchor_ts)]
