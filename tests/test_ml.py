"""
ReturnWise AI - Python Machine Learning Unit Tests
===================================================
Tests dataset generation, pipeline training, serialization, inference,
category-specific logic, risk classification, contributing factor extraction,
and transparent fallback modes.
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.pipeline import ReturnRiskPipeline
from ml.train_model import (
    generate_synthetic_orders,
    extract_features_and_target,
    CSV_PATH,
    MODEL_PATH
)
from ml.predict import (
    sanitize_input,
    get_risk_level,
    compute_contributing_factors,
    generate_recommendations,
    predict_return_risk,
    fallback_prediction,
    load_model
)


class TestMLPipeline(unittest.TestCase):
    def test_01_synthetic_dataset_generation(self):
        """Verifies synthetic dataset contains >= 500 records with all required fields."""
        orders = generate_synthetic_orders(num_samples=600)
        self.assertGreaterEqual(len(orders), 500)

        required_keys = [
            "order_id",
            "product_category",
            "product_price",
            "size",
            "delivery_days",
            "customer_previous_returns",
            "product_rating",
            "payment_type",
            "returned"
        ]
        for key in required_keys:
            self.assertIn(key, orders[0])

        # Verify binary target labels
        labels = {o["returned"] for o in orders}
        self.assertTrue(labels.issubset({0, 1}))

    def test_02_category_specific_logic(self):
        """Verifies category-specific rules (size ignored for non-apparel)."""
        # Case A: Electronics with explicit size should be coerced to 'Not applicable'
        electronics_order = {
            "product_category": "Electronics",
            "product_price": 120.0,
            "size": "XL",
            "delivery_days": 3,
            "customer_previous_returns": 0,
            "product_rating": 4.5,
            "payment_type": "Prepaid"
        }
        sanitized = sanitize_input(electronics_order)
        self.assertEqual(sanitized["size"], "Not applicable")

        # Case B: Home goods with explicit size should be coerced to 'Not applicable'
        home_order = {
            "product_category": "Home",
            "product_price": 45.0,
            "size": "M",
            "delivery_days": 2,
            "customer_previous_returns": 1,
            "product_rating": 4.2,
            "payment_type": "Prepaid"
        }
        sanitized_home = sanitize_input(home_order)
        self.assertEqual(sanitized_home["size"], "Not applicable")

        # Case C: Clothing should preserve valid apparel size
        clothing_order = {
            "product_category": "Clothing",
            "product_price": 60.0,
            "size": "XXL",
            "delivery_days": 5,
            "customer_previous_returns": 1,
            "product_rating": 4.0,
            "payment_type": "COD"
        }
        sanitized_clothing = sanitize_input(clothing_order)
        self.assertEqual(sanitized_clothing["size"], "XXL")

    def test_03_risk_level_thresholds(self):
        """Verifies strict boundary classification for Low, Medium, and High risk."""
        self.assertEqual(get_risk_level(10.0), "Low")
        self.assertEqual(get_risk_level(34.9), "Low")
        self.assertEqual(get_risk_level(35.0), "Medium")
        self.assertEqual(get_risk_level(50.0), "Medium")
        self.assertEqual(get_risk_level(65.0), "Medium")
        self.assertEqual(get_risk_level(65.1), "High")
        self.assertEqual(get_risk_level(95.0), "High")

    def test_04_contributing_factors_and_recommendations(self):
        """Verifies exactly 3 contributing factors and exactly 2 recommendations."""
        order = {
            "product_category": "Clothing",
            "product_price": 99.0,
            "size": "XXL",
            "delivery_days": 8,
            "customer_previous_returns": 3,
            "product_rating": 3.0,
            "payment_type": "COD"
        }
        factors = compute_contributing_factors(order, 85)
        self.assertEqual(len(factors), 3, "Must return exactly 3 top contributing factors")
        for f in factors:
            self.assertIn("feature", f)
            self.assertIn("impact", f)
            self.assertIn("description", f)

        recs = generate_recommendations(order, 85)
        self.assertEqual(len(recs), 2, "Must return exactly 2 actionable recommendations")
        self.assertTrue(isinstance(recs[0], str))
        self.assertTrue(isinstance(recs[1], str))

    def test_05_model_loading_and_prediction(self):
        """Verifies serialized model loads and generates calibrated predictions."""
        pipeline = load_model()
        self.assertIsNotNone(pipeline, "Trained pipeline must exist and load successfully")

        high_risk_order = {
            "product_category": "Clothing",
            "product_price": 120.0,
            "size": "XXL",
            "delivery_days": 8,
            "customer_previous_returns": 4,
            "product_rating": 3.1,
            "payment_type": "COD"
        }
        result = predict_return_risk(high_risk_order)

        self.assertFalse(result["is_fallback"], "Should use trained ML pipeline")
        self.assertGreaterEqual(result["risk_score"], 1)
        self.assertLessEqual(result["risk_score"], 99)
        self.assertEqual(result["risk_level"], "High")
        self.assertEqual(len(result["contributing_factors"]), 3)
        self.assertEqual(len(result["recommendations"]), 2)

    def test_06_fallback_mode_transparency(self):
        """Verifies rule-based fallback is honestly labeled and never fakes ML inference."""
        order = {
            "product_category": "Home",
            "product_price": 40.0,
            "size": "Not applicable",
            "delivery_days": 3,
            "customer_previous_returns": 0,
            "product_rating": 4.7,
            "payment_type": "Prepaid"
        }
        fallback = fallback_prediction(order)
        self.assertTrue(fallback["is_fallback"])
        self.assertIn("Fallback", fallback["model_type"])
        self.assertGreaterEqual(fallback["risk_score"], 1)
        self.assertLessEqual(fallback["risk_score"], 99)
        self.assertEqual(len(fallback["recommendations"]), 2)


if __name__ == "__main__":
    unittest.main()
