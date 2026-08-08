"""
STAGE 9 — VISUALIZATION & RESULTS ANALYSIS
==============================================
Produces the figure set referenced in the paper's Results section.
Output: figures/*.png
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import RESULTS_DIR, FIG_DIR

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

df = pd.read_csv(f"{RESULTS_DIR}/benchmark_full.csv")
summary = pd.read_csv(f"{RESULTS_DIR}/benchmark_summary.csv")

# 1. Latency distribution (violin/hist overlay)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(df["sw_latency"], bins=80, alpha=0.55, label="Transaction Switch", color="#c0392b")
ax.hist(df["api_latency"], bins=80, alpha=0.55, label="API Gateway", color="#2471a3")
ax.set_xlabel("Latency (ms)"); ax.set_ylabel("Transaction count")
ax.set_title("Latency Distribution: API Gateway vs Transaction Switch")
ax.legend(); fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig1_latency_distribution.png"); plt.close(fig)

# 2. Percentile latency bar chart
fig, ax = plt.subplots(figsize=(7, 4.5))
metrics = ["mean_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms"]
labels = ["Mean", "P50", "P95", "P99"]
x = np.arange(len(labels)); width = 0.25
for i, arch in enumerate(summary["architecture"]):
    vals = summary.loc[summary["architecture"] == arch, metrics].values.flatten()
    ax.bar(x + i * width, vals, width, label=arch)
ax.set_xticks(x + width); ax.set_xticklabels(labels)
ax.set_ylabel("Latency (ms)"); ax.set_title("Latency Percentiles by Architecture")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig2_latency_percentiles.png"); plt.close(fig)

# 3. Success rate comparison
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.bar(summary["architecture"], summary["success_rate_pct"],
       color=["#c0392b", "#2471a3", "#27ae60"])
ax.set_ylim(90, 100); ax.set_ylabel("Success rate (%)")
ax.set_title("Transaction Success Rate by Architecture")
for i, v in enumerate(summary["success_rate_pct"]):
    ax.text(i, v + 0.15, f"{v:.2f}%", ha="center")
fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig3_success_rate.png"); plt.close(fig)

# 4. Latency by hour of day (peak-load behaviour)
hourly = df.groupby("hour_of_day")[["sw_latency", "api_latency"]].mean()
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(hourly.index, hourly["sw_latency"], marker="o", label="Transaction Switch", color="#c0392b")
ax.plot(hourly.index, hourly["api_latency"], marker="o", label="API Gateway", color="#2471a3")
ax.set_xlabel("Hour of day"); ax.set_ylabel("Mean latency (ms)")
ax.set_title("Mean Latency by Hour of Day (Load-Dependent Queuing)")
ax.legend(); fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig4_latency_by_hour.png"); plt.close(fig)

# 5. ML feature importance
feat_imp = pd.read_csv(f"{RESULTS_DIR}/ml_router_model_features.csv", index_col=0)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
feat_imp["importance"].sort_values().plot(kind="barh", ax=ax, color="#8e44ad")
ax.set_xlabel("Importance"); ax.set_title("Intelligent Router — Feature Importances")
fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig5_feature_importance.png"); plt.close(fig)

# 6. Cost per 1,000 transactions
cost_summary = pd.read_csv(f"{RESULTS_DIR}/api_gateway_simulation.csv")["cost_usd"].sum(), \
               pd.read_csv(f"{RESULTS_DIR}/transaction_switch_simulation.csv")["cost_usd"].sum()
fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.bar(["API Gateway", "Transaction Switch"], cost_summary, color=["#2471a3", "#c0392b"])
ax.set_ylabel("Total simulated cost (USD)")
ax.set_title(f"Total Cost across {len(df):,} Transactions")
fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig6_cost_comparison.png"); plt.close(fig)

print("[Stage 9] Figures written to", FIG_DIR)
for f in ["fig1_latency_distribution.png", "fig2_latency_percentiles.png",
          "fig3_success_rate.png", "fig4_latency_by_hour.png",
          "fig5_feature_importance.png", "fig6_cost_comparison.png"]:
    print("  -", f)
