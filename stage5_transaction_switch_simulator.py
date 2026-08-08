"""
STAGE 5 — TRANSACTION SWITCH ARCHITECTURE SIMULATOR
=======================================================
Mirrors Stage 4 but for a traditional ISO 8583 host-to-host Transaction
Switch: persistent socket pools, fixed-length binary frames, STAN-based
sequencing, and store-and-forward queuing under load. Parameters are
anchored to ARCHITECTURE_PRIORS['transaction_switch'].

Output: results/transaction_switch_simulation.csv -> consumed by Stage 7
"""
import numpy as np
import pandas as pd
from config import SEED, DATA_DIR, RESULTS_DIR, ARCHITECTURE_PRIORS, COST_MODEL

rng = np.random.default_rng(SEED + 2)
P = ARCHITECTURE_PRIORS["transaction_switch"]
COST = COST_MODEL["transaction_switch"]


def simulate(df):
    n = len(df)

    queueing = (df["simulated_network_load_tps"] / 100.0) * P["queueing_penalty_per_100tps"]
    complexity_penalty = (df["message_complexity"] - 1) * 9.0   # binary re-parse cost
    cross_border_penalty = df["cross_border"].astype(int) * 55   # extra hop via correspondent
    jitter = rng.normal(0, P["latency_jitter_ms"], size=n)

    latency = (P["base_latency_ms"] + queueing + complexity_penalty +
               cross_border_penalty + jitter).clip(40, None)

    host_down = ~df["issuer_host_available"]
    success_prob = np.where(host_down, P["stand_in_recovery_rate"],
                             P["base_success_rate"])
    success_prob = success_prob - df["is_high_value"] * 0.012 - df["cross_border"].astype(int) * 0.01
    success = rng.random(n) < success_prob

    timed_out = (latency > P["timeout_ms"]) | (~success & (rng.random(n) < 0.14))
    status = np.where(success, "APPROVED",
                       np.where(timed_out, "TIMEOUT", "DECLINED"))

    infra_cost = COST["infra_per_1k"] / 1000.0
    incident_cost = np.where(status != "APPROVED", COST["incident_cost_per_failure"], 0.0)
    total_cost = infra_cost + incident_cost

    out = pd.DataFrame({
        "transaction_id": df["transaction_id"],
        "architecture": "TRANSACTION_SWITCH",
        "latency_ms": latency.round(2),
        "status": status,
        "success": success,
        "cost_usd": total_cost.round(4),
        "protocol_overhead_bytes": P["protocol_overhead_bytes"],
    })
    return out


if __name__ == "__main__":
    feat = pd.read_csv(f"{DATA_DIR}/features.csv")
    result = simulate(feat)
    out_path = f"{RESULTS_DIR}/transaction_switch_simulation.csv"
    result.to_csv(out_path, index=False)
    print(f"[Stage 5] Transaction Switch simulation complete -> {out_path}")
    print(f"  Mean latency: {result['latency_ms'].mean():.2f} ms | "
          f"Success rate: {result['success'].mean()*100:.2f}% | "
          f"Mean cost/txn: ${result['cost_usd'].mean():.5f}")
