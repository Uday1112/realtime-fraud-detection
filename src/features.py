"""
Streaming feature engineering, shared by offline training (generate_historical.py)
and live scoring (app.py) so the model sees identical feature semantics in
both places — a common real-world bug is training/serving skew where the
notebook computes features differently than the production scorer.

CustomerState tracks running statistics per customer incrementally (Welford's
algorithm for mean/variance, a rolling timestamp window for velocity) so
scoring a live event never has to rescan transaction history.
"""

from collections import deque
from dataclasses import dataclass, field


@dataclass
class CustomerState:
    customer_id: int
    count: int = 0
    mean_amount: float = 0.0
    _m2: float = 0.0  # Welford's running sum of squared deviations
    recent_timestamps: deque = field(default_factory=lambda: deque(maxlen=50))
    last_ts: float | None = None

    def std_amount(self) -> float:
        if self.count < 2:
            return 0.0
        return (self._m2 / (self.count - 1)) ** 0.5

    def update(self, amount: float, ts: float) -> None:
        self.count += 1
        delta = amount - self.mean_amount
        self.mean_amount += delta / self.count
        delta2 = amount - self.mean_amount
        self._m2 += delta * delta2
        self.recent_timestamps.append(ts)
        self.last_ts = ts

    def velocity_last_n_seconds(self, ts: float, window_seconds: float = 300) -> int:
        # Guard against t > ts (shouldn't happen in normal chronological
        # processing, but a negative gap would otherwise satisfy `<=` trivially).
        return sum(1 for t in self.recent_timestamps if 0 <= ts - t <= window_seconds)


def seed_state_from_baseline(customer_id: int, baseline_mean: float, baseline_std: float, prior_count: int = 30) -> CustomerState:
    """Bootstrap a CustomerState from a customer's known historical baseline
    (rather than starting cold at mean=0), so day-one live scoring already
    knows what "normal" looks like for this customer."""
    state = CustomerState(customer_id=customer_id)
    state.count = prior_count
    state.mean_amount = baseline_mean
    state._m2 = (baseline_std ** 2) * (prior_count - 1) if prior_count > 1 else 0.0
    return state


FEATURE_COLUMNS = [
    "amount",
    "quantity",
    "amount_zscore",
    "velocity_5min",
    "seconds_since_last_txn",
    "hour_of_day",
    "is_night",
    "is_new_customer",
]

# Gaps are capped at 1 hour: once a customer's last purchase was over an
# hour ago, the exact number stops mattering for "is this rapid succession?"
# — an uncapped sentinel (e.g. 999999) would dominate IsolationForest's
# random split selection and distort isolation for every other feature.
GAP_CAP_SECONDS = 3600.0


def compute_features(amount: float, quantity: int, ts_epoch: float, hour_of_day: int, state: CustomerState) -> dict:
    std = state.std_amount()
    if std > 1e-6:
        zscore = (amount - state.mean_amount) / std
    else:
        zscore = 0.0 if state.count == 0 else (amount - state.mean_amount) / max(state.mean_amount, 1.0)

    if state.last_ts is None:
        gap = GAP_CAP_SECONDS
    else:
        gap = min(GAP_CAP_SECONDS, max(0.0, ts_epoch - state.last_ts))

    return {
        "amount": amount,
        "quantity": quantity,
        "amount_zscore": zscore,
        "velocity_5min": state.velocity_last_n_seconds(ts_epoch, window_seconds=300),
        "seconds_since_last_txn": gap,
        "hour_of_day": hour_of_day,
        "is_night": int(hour_of_day < 6 or hour_of_day >= 23),
        "is_new_customer": int(state.count < 3),
    }
