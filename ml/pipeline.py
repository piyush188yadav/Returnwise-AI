"""
ReturnWise AI - ML Pipeline Definition
=======================================
Defines the ReturnRiskPipeline class so it has a stable, fixed module path
(ml.pipeline.ReturnRiskPipeline) across training, testing, and production inference.
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
CATEGORIES = ["Clothing", "Shoes", "Electronics", "Home"]
SIZES = ["Not applicable", "XS", "S", "M", "L", "XL", "XXL"]
PAYMENT_TYPES = ["COD", "Prepaid"]


class ReturnRiskPipeline:
    """
    Unified scikit-learn pipeline container combining categorical OneHotEncoder,
    numeric StandardScaler, and calibrated LogisticRegression classifier.
    """
    def __init__(self):
        self.num_scaler = StandardScaler()
        self.cat_encoder = OneHotEncoder(
            categories=[CATEGORIES, SIZES, PAYMENT_TYPES],
            handle_unknown="ignore",
            sparse_output=False
        )
        self.classifier = LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        )
        self.is_fitted = False

    def fit(self, X_cat, X_num, y):
        X_cat_encoded = self.cat_encoder.fit_transform(X_cat)
        X_num_scaled = self.num_scaler.fit_transform(X_num)
        X_combined = np.hstack([X_num_scaled, X_cat_encoded])
        self.classifier.fit(X_combined, y)
        self.is_fitted = True
        return self

    def transform_features(self, X_cat, X_num):
        X_cat_encoded = self.cat_encoder.transform(X_cat)
        X_num_scaled = self.num_scaler.transform(X_num)
        return np.hstack([X_num_scaled, X_cat_encoded])

    def predict(self, X_cat, X_num):
        X_combined = self.transform_features(X_cat, X_num)
        return self.classifier.predict(X_combined)

    def predict_proba(self, X_cat, X_num):
        X_combined = self.transform_features(X_cat, X_num)
        return self.classifier.predict_proba(X_combined)

    def predict_single_order(self, order_dict: dict) -> float:
        cat_row = np.array([[
            order_dict.get("product_category", "Clothing"),
            order_dict.get("size", "M"),
            order_dict.get("payment_type", "Prepaid")
        ]], dtype=object)

        num_row = np.array([[
            float(order_dict.get("product_price", 50.0)),
            float(order_dict.get("delivery_days", 4)),
            float(order_dict.get("customer_previous_returns", 0)),
            float(order_dict.get("product_rating", 4.0))
        ]], dtype=float)

        proba = self.predict_proba(cat_row, num_row)
        return float(proba[0, 1])
