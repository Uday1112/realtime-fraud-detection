"""
Generates a 30-day historical transaction dataset (with fraud patterns
injected and labeled) for offline model training/evaluation, and persists
the resulting per-customer baseline statistics for the live stream to
bootstrap from — so live scoring on day one already knows what "normal"
looks like for each customer instead of starting cold.

Run with: python src/generate_historical.py
"""

from pathlib import Path

import pandas as pd

from entities import generate_customers, generate_products
from features import CustomerState, compute_features
from simulate import TransactionSimulator

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

TARGET_EVENTS = 18000
HISTORICAL_DAYS_BACK = 30


def main():
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    customers = generate_customers()
    products = generate_products()
    pd.DataFrame(customers).to_parquet(DATA_DIR / "customers.parquet", index=False)
    pd.DataFrame(products).to_parquet(DATA_DIR / "products.parquet", index=False)

    sim = TransactionSimulator(customers, products, seed=123, fraud_rate=0.025)

    print(f"Generating ~{TARGET_EVENTS} historical events over {HISTORICAL_DAYS_BACK} days...")
    events = []
    while len(events) < TARGET_EVENTS:
        events.extend(sim.generate_one(current_ts=None, historical_days_back=HISTORICAL_DAYS_BACK))
    events.sort(key=lambda e: e["timestamp"])
    print(f"Generated {len(events)} events ({sum(e['is_fraud_synthetic'] for e in events)} labeled fraud)")

    # Replay in chronological order, computing features from state BEFORE
    # each event (so the model never sees the event's own outcome) then
    # updating state AFTER — same order of operations the live scorer uses.
    states: dict[int, CustomerState] = {}
    rows = []
    for e in events:
        cid = e["customer_id"]
        state = states.setdefault(cid, CustomerState(customer_id=cid))
        feats = compute_features(e["amount"], e["quantity"], e["timestamp"], e["hour_of_day"], state)
        state.update(e["amount"], e["timestamp"])

        rows.append({**e, **feats})

    df = pd.DataFrame(rows)
    df.to_parquet(DATA_DIR / "historical_transactions.parquet", index=False)
    print(f"wrote {DATA_DIR / 'historical_transactions.parquet'} ({len(df)} rows, {len(df.columns)} cols)")

    baselines = pd.DataFrame(
        [
            {
                "customer_id": cid,
                "count": s.count,
                "mean_amount": s.mean_amount,
                "std_amount": s.std_amount(),
            }
            for cid, s in states.items()
        ]
    )
    baselines.to_parquet(MODELS_DIR / "customer_baselines.parquet", index=False)
    print(f"wrote {MODELS_DIR / 'customer_baselines.parquet'} ({len(baselines)} customers)")


if __name__ == "__main__":
    main()
