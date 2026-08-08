"""
STAGE 3 — DATA PREPROCESSING & FEATURE ENGINEERING
======================================================
Cleans Stage 2 output and derives the feature set used by:
  - Stage 4/5 architecture simulators (routing complexity proxy)
  - Stage 6 ML intelligent router (supervised features)

Output: data/features.csv -> consumed by Stages 4, 5, 6
"""
import numpy as np
import pandas as pd
from config import DATA_DIR

def load_raw():
    df = pd.read_csv(f"{DATA_DIR}/transactions.csv", parse_dates=["timestamp"])
    return df

def engineer(df):
    df = df.copy()

    # 1. Missing-value / integrity checks (none expected, but validated)
    assert df["amount_usd"].isna().sum() == 0
    df = df.drop_duplicates(subset="transaction_id")

    # 2. Categorical encodings
    df["channel_code"] = df["channel"].astype("category").cat.codes
    df["txn_type_code"] = df["txn_type"].astype("category").cat.codes
    df["merchant_cat_code"] = df["merchant_category"].astype("category").cat.codes

    # 3. Derived complexity / risk-proxy features
    df["is_high_value"] = (df["amount_usd"] > df["amount_usd"].quantile(0.90)).astype(int)
    df["is_cross_bank"] = (df["issuer_bank_id"] != df["acquirer_bank_id"]).astype(int)
    df["is_peak_hour"] = df["hour_of_day"].isin([9, 10, 11, 12, 18, 19]).astype(int)
    df["amount_log"] = np.log1p(df["amount_usd"])

    # 4. Message-complexity proxy (more fields required => higher processing cost)
    complexity_map = {"BALANCE_INQUIRY": 1, "WITHDRAWAL": 2, "PURCHASE": 2,
                       "BILL_PAYMENT": 3, "FUND_TRANSFER": 3, "REVERSAL": 4}
    df["message_complexity"] = df["txn_type"].map(complexity_map)
    df["message_complexity"] += df["cross_border"].astype(int)  # +1 for FX/cross-border

    # 5. Load normalization (z-score) for downstream latency modeling
    df["load_z"] = (df["simulated_network_load_tps"] -
                     df["simulated_network_load_tps"].mean()) / \
                    df["simulated_network_load_tps"].std()

    feature_cols = [
        "transaction_id", "timestamp", "hour_of_day", "channel", "channel_code",
        "txn_type", "txn_type_code", "merchant_category", "merchant_cat_code",
        "amount_usd", "amount_log", "cross_border", "is_cross_bank",
        "is_high_value", "is_peak_hour", "message_complexity",
        "simulated_network_load_tps", "load_z", "issuer_host_available",
    ]
    return df[feature_cols]

if __name__ == "__main__":
    raw = load_raw()
    feat = engineer(raw)
    out = f"{DATA_DIR}/features.csv"
    feat.to_csv(out, index=False)
    print(f"[Stage 3] Engineered {feat.shape[1]} features for {len(feat):,} rows -> {out}")
    print(feat.describe(include="all").transpose().head(12).to_string())
