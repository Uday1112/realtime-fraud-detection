"""
The live streaming engine: generates transactions in (simulated) real time,
computes features incrementally exactly like the historical replay did,
scores each one with the trained model, and writes results to the DuckDB
warehouse. This is the piece the Streamlit dashboard drives on a timer.

Micro-batch design: each `tick()` call generates however many events should
have occurred since the last tick (Poisson-distributed around a target
rate) and scores them together as one small batch — the same micro-batch
pattern Spark Structured Streaming uses internally, chosen here over a
per-event round trip because scoring in small batches is both simpler and
lower-overhead than a strict one-event-at-a-time pipeline.
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import warehouse
from entities import generate_customers, generate_products
from features import FEATURE_COLUMNS, CustomerState, compute_features, seed_state_from_baseline
from simulate import TransactionSimulator

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

WAREHOUSE_COLUMNS = [
    "transaction_id", "customer_id", "product_id", "quantity", "amount", "ts", "event_time",
    "hour_of_day", "amount_zscore", "velocity_5min", "seconds_since_last_txn", "is_night",
    "is_new_customer", "anomaly_score", "is_flagged", "is_fraud_synthetic", "fraud_pattern",
]


class LiveStreamEngine:
    def __init__(self, events_per_tick: float = 6.0, fraud_rate: float = 0.03, seed: int = 999):
        self.model = joblib.load(MODELS_DIR / "anomaly_model.joblib")
        self.scaler = joblib.load(MODELS_DIR / "feature_scaler.joblib")

        customers = pd.read_parquet(DATA_DIR / "customers.parquet")
        products = pd.read_parquet(DATA_DIR / "products.parquet")
        baselines = pd.read_parquet(MODELS_DIR / "customer_baselines.parquet")

        self.simulator = TransactionSimulator(
            customers.to_dict("records"), products.to_dict("records"), seed=seed, fraud_rate=fraud_rate
        )
        self.events_per_tick = events_per_tick
        self.rng = np.random.default_rng(seed)

        # Bootstrap each customer's running stats from the historical baseline
        # computed offline, instead of starting the live stream cold.
        self.states: dict[int, CustomerState] = {}
        for row in baselines.itertuples():
            self.states[row.customer_id] = seed_state_from_baseline(
                row.customer_id, row.mean_amount, max(row.std_amount, 1.0), prior_count=max(int(row.count), 2)
            )

        self.conn = warehouse.get_connection()
        warehouse.reset_schema(self.conn)
        warehouse.load_dimensions(self.conn, customers, products)

    def _score(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURE_COLUMNS]
        X_scaled = self.scaler.transform(X)
        df["anomaly_score"] = -self.model.decision_function(X_scaled)
        df["is_flagged"] = self.model.predict(X_scaled) == -1
        return df

    def tick(self) -> pd.DataFrame:
        """Generate + score + persist one micro-batch. Returns the batch
        (with scores) for immediate use by the caller (e.g. UI feedback)."""
        now = time.time()
        target_n = int(self.rng.poisson(self.events_per_tick))

        rows = []
        while len(rows) < target_n:
            batch = self.simulator.generate_one(current_ts=now)
            for e in batch:
                cid = e["customer_id"]
                state = self.states.setdefault(cid, CustomerState(customer_id=cid))
                feats = compute_features(e["amount"], e["quantity"], e["timestamp"], e["hour_of_day"], state)
                state.update(e["amount"], e["timestamp"])
                rows.append({**e, **feats})

        if not rows:
            return pd.DataFrame(columns=WAREHOUSE_COLUMNS)

        df = pd.DataFrame(rows)
        df = self._score(df)
        df["ts"] = df["timestamp"]
        df["event_time"] = pd.to_datetime(df["timestamp"], unit="s")

        out = df[WAREHOUSE_COLUMNS].copy()
        warehouse.insert_transactions(self.conn, out)
        return out
