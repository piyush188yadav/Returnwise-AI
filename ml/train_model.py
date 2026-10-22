"""
ReturnWise AI - Machine Learning Model Training Pipeline
=========================================================
Trains a scikit-learn classification pipeline to predict e-commerce return risk.
Uses a realistic synthetic dataset (>= 500 records), reproducible seed,
train/test split with leakage prevention, and model evaluation metrics
(Precision, Recall, F1-Score, ROC-AUC).
"""

import os
import sys
import json
import csv
import numpy as np

# Ensure root dir is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
import joblib

from ml.pipeline import ReturnRiskPipeline, CATEGORIES, SIZES, PAYMENT_TYPES, RANDOM_STATE

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "model")
CSV_PATH = os.path.join(DATA_DIR, "orders_synthetic_train.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "return_risk_pipeline.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def generate_synthetic_orders(num_samples: int = 1200) -> list:
    """
    Generates realistic fictional orders with business logic:
    - Clothing & Shoes have sizing fit risks
    - Cash on Delivery (COD) orders have a higher cancellation/refusal rate
    - High previous return count strongly correlates with future return risk
    - Extended delivery times increase buyer remorse and returns
    - Low product ratings correlate with return rates
    - Electronics and Home return rates are driven by rating and high price
    """
    category_weights = [0.35, 0.25, 0.25, 0.15]
    records = []

    for i in range(num_samples):
        cat = np.random.choice(CATEGORIES, p=category_weights)

        if cat in ["Clothing", "Shoes"]:
            size = np.random.choice(["XS", "S", "M", "L", "XL", "XXL"], p=[0.1, 0.2, 0.3, 0.2, 0.12, 0.08])
            price = round(float(np.random.gamma(shape=3.0, scale=20.0) + 15), 2)
            price = min(price, 350.0)
        else:
            size = "Not applicable"
            if cat == "Electronics":
                price = round(float(np.random.gamma(shape=4.0, scale=28.0) + 25), 2)
                price = min(price, 600.0)
            else:  # Home
                price = round(float(np.random.gamma(shape=3.0, scale=18.0) + 18), 2)
                price = min(price, 300.0)

        payment = np.random.choice(PAYMENT_TYPES, p=[0.45, 0.55])
        delivery_days = int(np.clip(np.random.normal(loc=4.5, scale=2.0), 1, 14))
        prev_returns = int(np.clip(np.random.exponential(scale=1.2), 0, 8))
        rating = round(float(np.clip(np.random.normal(loc=4.0, scale=0.8), 1.0, 5.0)), 1)

        # Baseline log-odds
        log_odds = -2.1
        log_odds += prev_returns * 0.45

        if payment == "COD":
            log_odds += 0.55

        if delivery_days >= 6:
            log_odds += (delivery_days - 5) * 0.25

        if rating < 3.5:
            log_odds += (3.5 - rating) * 0.70
        elif rating >= 4.5:
            log_odds -= 0.35

        if cat in ["Clothing", "Shoes"]:
            log_odds += 0.30
            if size in ["XS", "XXL"]:
                log_odds += 0.40
        elif cat == "Electronics":
            if price > 200:
                log_odds += 0.25
        elif cat == "Home":
            log_odds -= 0.20

        prob = 1.0 / (1.0 + np.exp(-log_odds))
        prob = np.clip(prob + np.random.normal(0, 0.03), 0.02, 0.98)
        returned = 1 if np.random.rand() < prob else 0

        records.append({
            "order_id": f"SYN-{10000 + i}",
            "product_category": cat,
            "product_price": price,
            "size": size,
            "delivery_days": delivery_days,
            "customer_previous_returns": prev_returns,
            "product_rating": rating,
            "payment_type": payment,
            "returned": returned
        })

    return records


def save_dataset_csv(records: list, path: str):
    """Saves records to CSV using standard library."""
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def extract_features_and_target(records: list):
    """Converts record dicts into feature lists/arrays for training."""
    cats = []
    nums = []
    y = []

    for r in records:
        cats.append([r["product_category"], r["size"], r["payment_type"]])
        nums.append([
            float(r["product_price"]),
            float(r["delivery_days"]),
            float(r["customer_previous_returns"]),
            float(r["product_rating"])
        ])
        y.append(int(r["returned"]))

    return np.array(cats, dtype=object), np.array(nums, dtype=float), np.array(y, dtype=int)


def train_and_evaluate_model():
    print("[INFO] Generating synthetic e-commerce orders dataset (N=1200)...")
    records = generate_synthetic_orders(num_samples=1200)

    # Save to CSV
    save_dataset_csv(records, CSV_PATH)
    print(f"[INFO] Saved dataset to {CSV_PATH} ({len(records)} records).")

    X_cat, X_num, y = extract_features_and_target(records)
    print(f"[INFO] Overall dataset return rate: {y.mean() * 100:.2f}%")

    indices = np.arange(len(records))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    X_cat_train, X_cat_test = X_cat[train_idx], X_cat[test_idx]
    X_num_train, X_num_test = X_num[train_idx], X_num[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pipeline = ReturnRiskPipeline()
    print("[INFO] Fitting scikit-learn pipeline on training set (leakage-free)...")
    pipeline.fit(X_cat_train, X_num_train, y_train)

    # Evaluate on test set
    y_pred = pipeline.predict(X_cat_test, X_num_test)
    y_prob = pipeline.predict_proba(X_cat_test, X_num_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    metrics = {
        "model_name": "ReturnWise-CalibratedLogisticRegressionPipeline",
        "sample_size": len(records),
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "random_state": RANDOM_STATE,
        "classes": ["Kept (0)", "Returned (1)"],
        "categorical_features": ["product_category", "size", "payment_type"],
        "numeric_features": ["product_price", "delivery_days", "customer_previous_returns", "product_rating"]
    }

    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Kept", "Returned"]))

    # Save model pipeline
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[INFO] Trained pipeline successfully serialized to: {MODEL_PATH}")

    # Save metrics metadata
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Evaluation metrics saved to: {METRICS_PATH}")

    return metrics


if __name__ == "__main__":
    train_and_evaluate_model()
