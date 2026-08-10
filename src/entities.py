"""Static dimensions (customers, products) shared by historical generation,
live streaming, and the warehouse. Each customer gets its own baseline
spend distribution so "deviation from this customer's normal behavior" is a
meaningful feature instead of a global average."""

import numpy as np

N_CUSTOMERS = 500
N_PRODUCTS = 80

# Major Canadian cities spanning every province — customers are modeled as
# shoppers nationwide rather than a single metro area, since this project
# is framed as a Canada-wide e-commerce marketplace.
REGIONS = [
    "Toronto", "Ottawa", "Mississauga", "Hamilton",
    "Montreal", "Quebec City",
    "Vancouver", "Victoria",
    "Calgary", "Edmonton",
    "Winnipeg", "Saskatoon",
    "Halifax", "Moncton",
    "St. John's", "Charlottetown",
]
CATEGORIES = ["Electronics", "Apparel", "Home & Kitchen", "Grocery", "Office", "Gift Cards"]


def generate_customers(seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    customers = []
    for cid in range(1, N_CUSTOMERS + 1):
        # Each customer's "normal" spend is log-normal so most are modest
        # spenders with a long tail of higher-value shoppers.
        baseline_mean = float(np.clip(rng.lognormal(mean=3.6, sigma=0.6), 15, 800))
        customers.append(
            {
                "customer_id": cid,
                "region": REGIONS[rng.integers(0, len(REGIONS))],
                "baseline_mean_amount": round(baseline_mean, 2),
                "baseline_std_amount": round(baseline_mean * 0.25, 2),
            }
        )
    return customers


def generate_products(seed: int = 11) -> list[dict]:
    rng = np.random.default_rng(seed)
    products = []
    for pid in range(1, N_PRODUCTS + 1):
        category = CATEGORIES[rng.integers(0, len(CATEGORIES))]
        products.append(
            {
                "product_id": pid,
                "product_name": f"{category} Item {pid}",
                "category": category,
                "unit_price": round(float(rng.uniform(5, 300)), 2),
            }
        )
    return products
