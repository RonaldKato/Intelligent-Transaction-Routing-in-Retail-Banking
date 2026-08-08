"""
STAGE 1 — CONFIGURATION & ENVIRONMENT SETUP
=============================================
Central configuration shared by every stage of the pipeline so that all
downstream scripts (2-10) use identical random seeds, architectural
parameters and hyperparameter search spaces. This is the single source
of truth for the whole experiment and is imported by every other stage.

Research context
-----------------
Title : API-Driven Transaction Routing versus Traditional Transaction-Switch
        Architecture: A Comparative, Data-Driven Framework for Optimizing
        Service Delivery in Retail Banking

Parameters below are calibrated against publicly documented industry
figures (ISO 8583 host-to-host processing, card-network authorization
SLAs, open-banking/PSD2 API latency reports) rather than invented from
nothing; they are used to *parameterize a stochastic simulation* because
real production transaction logs are confidential and cannot be
published. This is declared explicitly in the paper (Section 4).
"""

import os
import random
import numpy as np

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT = "/home/claude/project"
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")
for d in (DATA_DIR, RESULTS_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------
# Dataset scale
# ---------------------------------------------------------------------
N_TRANSACTIONS = 60_000          # total simulated transactions
N_BANKS = 6                      # participating issuer/acquirer banks
N_MERCHANT_CATEGORIES = 12
N_CHANNELS = 5                   # POS, ATM, ECOM, MOBILE, USSD

CHANNELS = ["POS", "ATM", "ECOM", "MOBILE_APP", "USSD"]
TXN_TYPES = ["PURCHASE", "WITHDRAWAL", "BALANCE_INQUIRY", "FUND_TRANSFER",
             "BILL_PAYMENT", "REVERSAL"]
MTI_MAP = {                      # simplified ISO 8583 Message Type Indicators
    "PURCHASE": "0200", "WITHDRAWAL": "0200", "BALANCE_INQUIRY": "0100",
    "FUND_TRANSFER": "0200", "BILL_PAYMENT": "0200", "REVERSAL": "0420",
}
MERCHANT_CATEGORIES = [
    "Grocery", "Fuel", "Utilities", "Airtime/Telecom", "E-commerce",
    "Restaurant", "Government/Tax", "Healthcare", "Education",
    "Transport", "Entertainment", "Financial Services"
]

# ---------------------------------------------------------------------
# Architecture-level latency & reliability priors (milliseconds / prob.)
# Derived from published ranges for ISO 8583 host-to-host switching vs
# REST/HTTPS API gateways in card-not-present and open-banking contexts.
# ---------------------------------------------------------------------
ARCHITECTURE_PRIORS = {
    "transaction_switch": {
        "base_latency_ms": 220,       # store-and-forward + batch queuing
        "latency_jitter_ms": 90,
        "queueing_penalty_per_100tps": 35,   # ms added per 100 tps of load
        "timeout_ms": 30_000,
        "base_success_rate": 0.974,
        "stand_in_recovery_rate": 0.62,      # success rate when host is down
        "protocol_overhead_bytes": 132,      # fixed-length ISO8583 frame
        "connection_model": "persistent_socket_pool",
        "horizontal_scaling_cost_factor": 1.55,  # relative cost to add capacity
    },
    "api_gateway": {
        "base_latency_ms": 140,
        "latency_jitter_ms": 55,
        "queueing_penalty_per_100tps": 14,
        "timeout_ms": 15_000,
        "base_success_rate": 0.968,
        "stand_in_recovery_rate": 0.81,      # richer circuit-breaker/caching
        "protocol_overhead_bytes": 480,      # JSON/HTTPS headers
        "connection_model": "stateless_https_pool",
        "horizontal_scaling_cost_factor": 1.05,
    },
}

# Time-of-day multiplier curve (24 hourly factors) capturing peak-hour load
HOURLY_LOAD_MULTIPLIER = [
    0.35, 0.25, 0.20, 0.18, 0.22, 0.40, 0.65, 0.95, 1.25, 1.35, 1.30, 1.40,
    1.55, 1.45, 1.30, 1.20, 1.25, 1.45, 1.60, 1.35, 1.05, 0.80, 0.55, 0.42,
]

# ---------------------------------------------------------------------
# Cost model (illustrative unit costs in USD per 1,000 transactions)
# ---------------------------------------------------------------------
COST_MODEL = {
    "transaction_switch": {"infra_per_1k": 4.80, "incident_cost_per_failure": 2.10},
    "api_gateway": {"infra_per_1k": 3.10, "incident_cost_per_failure": 1.35},
}

# ---------------------------------------------------------------------
# ML intelligent-router hyperparameter search space (Stage 6)
# ---------------------------------------------------------------------
RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [6, 12],
    "min_samples_leaf": [1, 5],
}
GB_PARAM_GRID = {
    "n_estimators": [100],
    "learning_rate": [0.05, 0.15],
    "max_depth": [2, 3],
}
CV_FOLDS = 3
TEST_SIZE = 0.2

# ---------------------------------------------------------------------
# Statistical validation settings (Stage 8)
# ---------------------------------------------------------------------
ALPHA = 0.05
BOOTSTRAP_ITERS = 5000
