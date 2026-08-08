"""
STAGE 6 — ML-BASED INTELLIGENT ROUTING MODEL
================================================
Trains a supervised classifier that learns, from transaction features
alone, which architecture (API_GATEWAY vs TRANSACTION_SWITCH) would have
produced the lower-latency *successful* outcome for that transaction
(label derived from Stages 4+5 ground truth). This models a "smart API
gateway" that can dynamically pick a downstream rail per transaction —
the core proposed improvement over a statically-wired switch.

Two candidate model families are tuned by 5-fold grid search:
  - RandomForestClassifier
  - GradientBoostingClassifier
Best model selected by cross-validated F1 score, then evaluated on a
held-out test split with full metrics + hyperparameters reported.

Output: results/ml_router_metrics.json, results/ml_router_model_features.csv
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score, confusion_matrix)
from config import (SEED, DATA_DIR, RESULTS_DIR, RF_PARAM_GRID, GB_PARAM_GRID,
                     CV_FOLDS, TEST_SIZE)

FEATURE_COLS = [
    "channel_code", "txn_type_code", "merchant_cat_code", "amount_log",
    "cross_border", "is_cross_bank", "is_high_value", "is_peak_hour",
    "message_complexity", "load_z",
]


def build_training_table():
    feat = pd.read_csv(f"{DATA_DIR}/features.csv")
    api = pd.read_csv(f"{RESULTS_DIR}/api_gateway_simulation.csv")
    sw = pd.read_csv(f"{RESULTS_DIR}/transaction_switch_simulation.csv")

    merged = feat.merge(api[["transaction_id", "latency_ms", "success"]]
                         .rename(columns={"latency_ms": "api_latency", "success": "api_success"}),
                         on="transaction_id")
    merged = merged.merge(sw[["transaction_id", "latency_ms", "success"]]
                           .rename(columns={"latency_ms": "sw_latency", "success": "sw_success"}),
                           on="transaction_id")

    # label = 1 if API_GATEWAY is the better choice (succeeds AND is faster,
    # or switch fails while API succeeds), else 0 (switch preferred)
    def label_row(r):
        if r.api_success and not r.sw_success:
            return 1
        if r.sw_success and not r.api_success:
            return 0
        if r.api_success and r.sw_success:
            return 1 if r.api_latency <= r.sw_latency else 0
        return 1 if r.api_latency <= r.sw_latency else 0

    merged["label_prefer_api"] = merged.apply(label_row, axis=1)
    return merged


def tune_and_evaluate(merged):
    X = merged[FEATURE_COLS]
    y = merged["label_prefer_api"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

    results = {}

    rf = GridSearchCV(RandomForestClassifier(random_state=SEED),
                       RF_PARAM_GRID, cv=CV_FOLDS, scoring="f1", n_jobs=-1)
    rf.fit(X_train, y_train)
    results["random_forest"] = {
        "best_params": rf.best_params_,
        "cv_best_f1": round(rf.best_score_, 4),
    }

    gb = GridSearchCV(GradientBoostingClassifier(random_state=SEED),
                       GB_PARAM_GRID, cv=CV_FOLDS, scoring="f1", n_jobs=-1)
    gb.fit(X_train, y_train)
    results["gradient_boosting"] = {
        "best_params": gb.best_params_,
        "cv_best_f1": round(gb.best_score_, 4),
    }

    # pick winner by CV F1
    winner_name = max(results, key=lambda k: results[k]["cv_best_f1"])
    winner_model = rf.best_estimator_ if winner_name == "random_forest" else gb.best_estimator_

    y_pred = winner_model.predict(X_test)
    y_proba = winner_model.predict_proba(X_test)[:, 1]

    test_metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    feat_importance = None
    if hasattr(winner_model, "feature_importances_"):
        feat_importance = dict(zip(FEATURE_COLS,
                                    [round(float(v), 4) for v in winner_model.feature_importances_]))

    summary = {
        "candidate_models": results,
        "selected_model": winner_name,
        "test_set_metrics": test_metrics,
        "feature_importances": feat_importance,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "class_balance_train": y_train.value_counts(normalize=True).round(3).to_dict(),
    }
    return summary, feat_importance


if __name__ == "__main__":
    merged = build_training_table()
    summary, feat_importance = tune_and_evaluate(merged)

    with open(f"{RESULTS_DIR}/ml_router_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    if feat_importance:
        pd.Series(feat_importance).sort_values(ascending=False).to_csv(
            f"{RESULTS_DIR}/ml_router_model_features.csv", header=["importance"])

    print("[Stage 6] ML intelligent router tuned and evaluated.")
    print(f"  Selected model: {summary['selected_model']}")
    print(f"  Test F1: {summary['test_set_metrics']['f1']} | "
          f"Accuracy: {summary['test_set_metrics']['accuracy']} | "
          f"ROC-AUC: {summary['test_set_metrics']['roc_auc']}")
    print(f"  Best hyperparameters: "
          f"{summary['candidate_models'][summary['selected_model']]['best_params']}")
