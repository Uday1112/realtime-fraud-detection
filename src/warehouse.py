"""
DuckDB-backed warehouse for the live transaction stream. DuckDB (an
embedded OLAP engine) is used instead of a toy row-store because the
dashboard's job is exactly what it's built for: fast aggregate queries
(revenue over time, flagged-rate by hour) over an append-only fact table,
without standing up a separate database server for a portfolio project.

`is_fraud_synthetic` / `fraud_pattern` are carried through so the live
dashboard can show real-time precision/recall as the demo runs. In a real
system these wouldn't exist at ingest time — confirmed fraud is a delayed,
separate feedback signal — they're included here purely because this is a
controlled simulation with known ground truth, which is called out in the UI.
"""

from pathlib import Path

import duckdb
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
WAREHOUSE_PATH = BASE_DIR / "warehouse" / "warehouse.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS dim_customer (
    customer_id INTEGER PRIMARY KEY,
    region VARCHAR,
    baseline_mean_amount DOUBLE,
    baseline_std_amount DOUBLE
);

CREATE TABLE IF NOT EXISTS dim_product (
    product_id INTEGER PRIMARY KEY,
    product_name VARCHAR,
    category VARCHAR,
    unit_price DOUBLE
);

CREATE TABLE IF NOT EXISTS fact_transactions (
    transaction_id BIGINT PRIMARY KEY,
    customer_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    amount DOUBLE,
    ts DOUBLE,
    event_time TIMESTAMP,
    hour_of_day INTEGER,
    amount_zscore DOUBLE,
    velocity_5min INTEGER,
    seconds_since_last_txn DOUBLE,
    is_night INTEGER,
    is_new_customer INTEGER,
    anomaly_score DOUBLE,
    is_flagged BOOLEAN,
    is_fraud_synthetic BOOLEAN,
    fraud_pattern VARCHAR
);
"""


def get_connection(db_path: Path = WAREHOUSE_PATH) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    conn.execute(SCHEMA)
    return conn


def reset_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("DROP TABLE IF EXISTS fact_transactions")
    conn.execute("DROP TABLE IF EXISTS dim_customer")
    conn.execute("DROP TABLE IF EXISTS dim_product")
    conn.execute(SCHEMA)


def load_dimensions(conn: duckdb.DuckDBPyConnection, customers: pd.DataFrame, products: pd.DataFrame) -> None:
    conn.execute("DELETE FROM dim_customer")
    conn.execute("DELETE FROM dim_product")
    conn.register("customers_df", customers)
    conn.register("products_df", products)
    conn.execute("INSERT INTO dim_customer SELECT customer_id, region, baseline_mean_amount, baseline_std_amount FROM customers_df")
    conn.execute("INSERT INTO dim_product SELECT product_id, product_name, category, unit_price FROM products_df")
    conn.unregister("customers_df")
    conn.unregister("products_df")


def insert_transactions(conn: duckdb.DuckDBPyConnection, rows: pd.DataFrame) -> None:
    if rows.empty:
        return
    conn.register("new_rows", rows)
    conn.execute("INSERT INTO fact_transactions SELECT * FROM new_rows")
    conn.unregister("new_rows")


def recent_transactions(conn: duckdb.DuckDBPyConnection, limit: int = 25) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT t.event_time, t.customer_id, p.product_name, t.quantity, t.amount,
               t.anomaly_score, t.is_flagged, t.fraud_pattern
        FROM fact_transactions t
        LEFT JOIN dim_product p USING (product_id)
        ORDER BY t.ts DESC
        LIMIT ?
        """,
        [limit],
    ).df()


def recent_alerts(conn: duckdb.DuckDBPyConnection, limit: int = 15) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT t.event_time, t.customer_id, p.product_name, t.amount,
               t.anomaly_score, t.fraud_pattern, t.is_fraud_synthetic
        FROM fact_transactions t
        LEFT JOIN dim_product p USING (product_id)
        WHERE t.is_flagged
        ORDER BY t.ts DESC
        LIMIT ?
        """,
        [limit],
    ).df()


def summary_stats(conn: duckdb.DuckDBPyConnection) -> dict:
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS n_transactions,
            COALESCE(SUM(amount), 0) AS total_revenue,
            COALESCE(SUM(CASE WHEN is_flagged THEN 1 ELSE 0 END), 0) AS n_flagged,
            COALESCE(SUM(CASE WHEN is_fraud_synthetic THEN 1 ELSE 0 END), 0) AS n_true_fraud,
            COALESCE(SUM(CASE WHEN is_flagged AND is_fraud_synthetic THEN 1 ELSE 0 END), 0) AS n_true_positive
        FROM fact_transactions
        """
    ).fetchone()
    n_transactions, total_revenue, n_flagged, n_true_fraud, n_true_positive = row
    live_precision = n_true_positive / n_flagged if n_flagged else 0.0
    live_recall = n_true_positive / n_true_fraud if n_true_fraud else 0.0
    return {
        "n_transactions": n_transactions,
        "total_revenue": total_revenue,
        "n_flagged": n_flagged,
        "n_true_fraud": n_true_fraud,
        "n_true_positive": n_true_positive,
        "live_precision": live_precision,
        "live_recall": live_recall,
    }


def revenue_by_category(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT
            p.category,
            COUNT(*) AS n_transactions,
            SUM(t.amount) AS revenue
        FROM fact_transactions t
        LEFT JOIN dim_product p USING (product_id)
        GROUP BY 1
        ORDER BY revenue DESC
        """
    ).df()


def revenue_by_region(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT
            c.region,
            COUNT(*) AS n_transactions,
            SUM(t.amount) AS revenue,
            SUM(CASE WHEN t.is_flagged THEN 1 ELSE 0 END) AS n_flagged
        FROM fact_transactions t
        LEFT JOIN dim_customer c USING (customer_id)
        GROUP BY 1
        ORDER BY revenue DESC
        """
    ).df()


def revenue_timeseries(conn: duckdb.DuckDBPyConnection, seconds_back: int = 300, bucket_seconds: int = 5) -> pd.DataFrame:
    return conn.execute(
        """
        SELECT
            to_timestamp(CAST(ts / ? AS BIGINT) * ?) AS bucket_time,
            COUNT(*) AS n_transactions,
            SUM(amount) AS revenue,
            SUM(CASE WHEN is_flagged THEN 1 ELSE 0 END) AS n_flagged
        FROM fact_transactions
        WHERE ts >= (epoch(now()) - ?)
        GROUP BY 1
        ORDER BY 1
        """,
        [bucket_seconds, bucket_seconds, seconds_back],
    ).df()
