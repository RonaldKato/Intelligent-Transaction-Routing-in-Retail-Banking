"""
STAGE 2 — SYNTHETIC TRANSACTION DATASET GENERATION
=====================================================
Generates a transaction-level dataset that mimics real interbank switching
logs (ISO 8583 style fields) at N_TRANSACTIONS scale. Because genuine bank
switch logs are confidential, this stage produces a *parameterized
stochastic* dataset: every distribution is anchored to the priors declared
in config.py (Stage 1), not to arbitrary numbers, and is documented as
synthetic in the resulting paper.

Output: data/transactions.csv  -> consumed by Stage 3
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import (SEED, N_TRANSACTIONS, N_BANKS, CHANNELS, TXN_TYPES,
                     MTI_MAP, MERCHANT_CATEGORIES, HOURLY_LOAD_MULTIPLIER,
                     DATA_DIR)

rng = np.random.default_rng(SEED)


def random_timestamps(n):
    start = datetime(2026, 1, 1)
    days = rng.integers(0, 180, size=n)
    hours = rng.choice(24, size=n, p=np.array(HOURLY_LOAD_MULTIPLIER) /
                        sum(HOURLY_LOAD_MULTIPLIER))
    minutes = rng.integers(0, 60, size=n)
    seconds = rng.integers(0, 60, size=n)
    return [start + timedelta(days=int(d), hours=int(h), minutes=int(m),
                               seconds=int(s))
            for d, h, m, s in zip(days, hours, minutes, seconds)]


def build_dataset():
    n = N_TRANSACTIONS
    ts = random_timestamps(n)

    txn_type = rng.choice(TXN_TYPES, size=n,
                           p=[0.42, 0.18, 0.14, 0.14, 0.09, 0.03])
    channel = rng.choice(CHANNELS, size=n, p=[0.30, 0.16, 0.27, 0.24, 0.03])
    issuer_bank = rng.integers(1, N_BANKS + 1, size=n)
    acquirer_bank = rng.integers(1, N_BANKS + 1, size=n)
    merchant_cat = rng.choice(MERCHANT_CATEGORIES, size=n)

    # Amount distribution: lognormal, channel-dependent scale
    channel_scale = {"POS": 3.6, "ATM": 4.1, "ECOM": 3.8,
                      "MOBILE_APP": 3.3, "USSD": 2.6}
    amount = np.array([
        rng.lognormal(mean=channel_scale[c], sigma=0.9) for c in channel
    ]).round(2)
    amount = np.clip(amount, 1.0, 500_000.0)

    # Cross-border flag (issuer != acquirer bank AND ecom channel more likely)
    cross_border = ((issuer_bank != acquirer_bank) &
                     (rng.random(n) < np.where(channel == "ECOM", 0.35, 0.10)))

    mti = [MTI_MAP[t] for t in txn_type]

    # network load proxy: hour-of-day multiplier drives instantaneous tps
    hour = np.array([t.hour for t in ts])
    load_multiplier = np.array(HOURLY_LOAD_MULTIPLIER)[hour]
    simulated_tps = (load_multiplier * 38 + rng.normal(0, 3, size=n)).clip(2, None)

    df = pd.DataFrame({
        "transaction_id": [f"TXN{100000+i}" for i in range(n)],
        "timestamp": ts,
        "hour_of_day": hour,
        "txn_type": txn_type,
        "mti_code": mti,
        "channel": channel,
        "issuer_bank_id": issuer_bank,
        "acquirer_bank_id": acquirer_bank,
        "merchant_category": merchant_cat,
        "amount_usd": amount,
        "cross_border": cross_border,
        "simulated_network_load_tps": simulated_tps.round(1),
    })

    # Host-availability flag: brief simulated outages for stand-in testing
    outage_windows = rng.random(n) < 0.015
    df["issuer_host_available"] = ~outage_windows

    return df


if __name__ == "__main__":
    df = build_dataset()
    out_path = f"{DATA_DIR}/transactions.csv"
    df.to_csv(out_path, index=False)
    print(f"[Stage 2] Generated {len(df):,} synthetic transactions -> {out_path}")
    print(df.head(3).to_string())
    print("\nChannel distribution:\n", df["channel"].value_counts(normalize=True).round(3))
    print("\nTxn type distribution:\n", df["txn_type"].value_counts(normalize=True).round(3))
