"""
STAGE 4 — API GATEWAY ARCHITECTURE SIMULATOR
================================================
Simulates routing every transaction in data/features.csv through a
modern REST/HTTPS API-gateway architecture (stateless pool, circuit
breaker, JSON payloads). Latency and success/failure are stochastic,
anchored to ARCHITECTURE_PRIORS['api_gateway'] from Stage 1, and modulated
by per-transaction complexity/load features engineered in Stage 3.

Output: results/api_gateway_simulation.csv -> consumed by Stage 7
"""
import numpy as np
import pandas as pd
from config import SEED, DATA_DIR, RESULTS_DIR, ARCHITECTURE_PRIORS, COST_MODEL

rng = np.random.default_rng(SEED + 1)
P = ARCHITECTURE_PRIORS["api_gateway"]
COST = COST_MODEL["api_gateway"]


def simulate(df):
    n = len(df)

    # Latency = base + jitter + queueing penalty (load) + complexity penalty
    queueing = (df["simulated_network_load_tps"] / 100.0) * P["queueing_penalty_per_100tps"]
    complexity_penalty = (df["message_complexity"] - 1) * 6.5   # ms per extra field-set
    cross_border_penalty = df["cross_border"].astype(int) * 40   # FX/compliance hop
    jitter = rng.normal(0, P["latency_jitter_ms"], size=n)

    # Cold-start / TLS-handshake penalty: unlike the switch's persistent
    # socket pool, a fraction of API calls (new/idle client sessions,
    # low-traffic channels such as USSD, or the first call after connection
    # pool eviction) must pay a fresh TLS handshake + auth-token round trip.
    cold_start = rng.random(n) < np.where(df["channel"] == "USSD", 0.55, 0.22)
    cold_start_penalty = cold_start * rng.normal(170, 35, size=n).clip(60, None)

    latency = (P["base_latency_ms"] + queueing + complexity_penalty +
               cross_border_penalty + cold_start_penalty + jitter).clip(30, None)

    # Success probability: reduced by host outages, but API gateway has a
    # richer stand-in/circuit-breaker recovery path (cached auth, retries)
    host_down = ~df["issuer_host_available"]
    success_prob = np.where(host_down, P["stand_in_recovery_rate"],
                             P["base_success_rate"])
    # High-value & cross-border transactions face slightly stricter checks
    success_prob = success_prob - df["is_high_value"] * 0.01 - df["cross_border"].astype(int) * 0.008
    success = rng.random(n) < success_prob

    timed_out = (latency > P["timeout_ms"]) | (~success & (rng.random(n) < 0.10))
    status = np.where(success, "APPROVED",
                       np.where(timed_out, "TIMEOUT", "DECLINED"))

    infra_cost = COST["infra_per_1k"] / 1000.0
    incident_cost = np.where(status != "APPROVED", COST["incident_cost_per_failure"], 0.0)
    total_cost = infra_cost + incident_cost

    out = pd.DataFrame({
        "transaction_id": df["transaction_id"],
        "architecture": "API_GATEWAY",
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
    out_path = f"{RESULTS_DIR}/api_gateway_simulation.csv"
    result.to_csv(out_path, index=False)
    print(f"[Stage 4] API Gateway simulation complete -> {out_path}")
    print(f"  Mean latency: {result['latency_ms'].mean():.2f} ms | "
          f"Success rate: {result['success'].mean()*100:.2f}% | "
          f"Mean cost/txn: ${result['cost_usd'].mean():.5f}")
