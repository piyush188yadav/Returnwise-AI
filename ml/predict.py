"""
ReturnWise AI - ML Return Prediction Engine
============================================
Loads the trained scikit-learn pipeline, validates incoming order parameters,
performs calibrated inference, computes top contributing factors, and returns
exactly two actionable seller recommendations.
"""

import os
import sys
import json
import argparse
import numpy as np

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# Add current project root to sys.path so ReturnRiskPipeline class is resolved
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.pipeline import ReturnRiskPipeline

MODEL_PATH = os.path.join(BASE_DIR, "ml", "model", "return_risk_pipeline.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "ml", "model", "model_metrics.json")

# In-memory cached model
_LOADED_MODEL = None


def load_model():
    """Loads and caches the serialized model pipeline."""
    global _LOADED_MODEL
    if _LOADED_MODEL is not None:
        return _LOADED_MODEL

    if not HAS_JOBLIB or not os.path.exists(MODEL_PATH):
        return None

    try:
        _LOADED_MODEL = joblib.load(MODEL_PATH)
        return _LOADED_MODEL
    except Exception as e:
        sys.stderr.write(f"[WARN] Failed to load model pipeline: {e}\n")
        return None


def get_risk_level(score: float) -> str:
    """Classifies risk score into Low, Medium, or High risk tiers."""
    if score < 35.0:
        return "Low"
    elif score <= 65.0:
        return "Medium"
    else:
        return "High"


def sanitize_input(order_data: dict) -> dict:
    """
    Sanitizes and validates order input data, enforcing category-specific logic.
    For non-apparel categories, size is coerced to 'Not applicable'.
    """
    category = str(order_data.get("product_category", "Clothing")).strip()
    if category not in ["Clothing", "Shoes", "Electronics", "Home"]:
        category = "Clothing"

    size = str(order_data.get("size", "Not applicable")).strip()
    if category not in ["Clothing", "Shoes"]:
        size = "Not applicable"
    elif size not in ["XS", "S", "M", "L", "XL", "XXL", "Not applicable"]:
        size = "M"

    try:
        price = max(1.0, float(order_data.get("product_price", 50.0)))
    except (ValueError, TypeError):
        price = 50.0

    try:
        delivery_days = max(1, int(order_data.get("delivery_days", 4)))
    except (ValueError, TypeError):
        delivery_days = 4

    try:
        prev_returns = max(0, int(order_data.get("customer_previous_returns", 0)))
    except (ValueError, TypeError):
        prev_returns = 0

    try:
        rating = min(5.0, max(1.0, float(order_data.get("product_rating", 4.0))))
    except (ValueError, TypeError):
        rating = 4.0

    payment = str(order_data.get("payment_type", "Prepaid")).strip()
    if payment not in ["COD", "Prepaid"]:
        payment = "Prepaid"

    return {
        "product_category": category,
        "product_price": round(price, 2),
        "size": size,
        "delivery_days": delivery_days,
        "customer_previous_returns": prev_returns,
        "product_rating": round(rating, 1),
        "payment_type": payment
    }


def compute_contributing_factors(order: dict, risk_score: float) -> list:
    """
    Evaluates individual feature impact relative to typical e-commerce baselines.
    Returns the top 3 most influential factors driving the prediction.
    """
    factors = []

    # 1. Previous returns impact
    prev_returns = order["customer_previous_returns"]
    if prev_returns >= 3:
        factors.append({
            "feature": "Customer Return History",
            "impact": "High Risk Driver",
            "description": f"Customer has {prev_returns} previous returns, indicating habitual return behavior."
        })
    elif prev_returns in [1, 2]:
        factors.append({
            "feature": "Customer Return History",
            "impact": "Moderate Risk Factor",
            "description": f"Customer has recorded {prev_returns} prior return(s)."
        })
    else:
        factors.append({
            "feature": "Customer Return History",
            "impact": "Risk Mitigator",
            "description": "Zero previous returns from this customer profile."
        })

    # 2. Payment Type
    if order["payment_type"] == "COD":
        factors.append({
            "feature": "Payment Method (COD)",
            "impact": "High Risk Driver",
            "description": "Cash on Delivery orders demonstrate higher refusal and buyer-remorse return rates."
        })
    else:
        factors.append({
            "feature": "Payment Method (Prepaid)",
            "impact": "Risk Mitigator",
            "description": "Prepaid transaction signals verified buyer commitment."
        })

    # 3. Delivery Duration
    delivery = order["delivery_days"]
    if delivery >= 7:
        factors.append({
            "feature": "Estimated Delivery Time",
            "impact": "High Risk Driver",
            "description": f"Long delivery window ({delivery} days) increases cancellation and buyer impatience."
        })
    elif delivery >= 5:
        factors.append({
            "feature": "Estimated Delivery Time",
            "impact": "Moderate Risk Factor",
            "description": f"Delivery timeframe of {delivery} days is slightly above standard average."
        })
    else:
        factors.append({
            "feature": "Estimated Delivery Time",
            "impact": "Risk Mitigator",
            "description": f"Fast transit time ({delivery} days) reduces likelihood of customer change of mind."
        })

    # 4. Product Rating
    rating = order["product_rating"]
    if rating < 3.5:
        factors.append({
            "feature": "Product Satisfaction Rating",
            "impact": "High Risk Driver",
            "description": f"Sub-par customer rating ({rating}/5.0) signals quality or expectation mismatch."
        })
    elif rating >= 4.5:
        factors.append({
            "feature": "Product Satisfaction Rating",
            "impact": "Risk Mitigator",
            "description": f"High product rating ({rating}/5.0) confirms strong buyer satisfaction."
        })

    # 5. Category & Size Fit (Applicable only to Clothing/Shoes)
    category = order["product_category"]
    size = order["size"]
    if category in ["Clothing", "Shoes"]:
        if size in ["XS", "XXL"]:
            factors.append({
                "feature": "Extreme Sizing Variance",
                "impact": "Moderate Risk Factor",
                "description": f"Size {size} has higher statistical variance in sizing charts across brands."
            })
        elif size in ["S", "M", "L", "XL"]:
            factors.append({
                "feature": "Standard Sizing Category",
                "impact": "Neutral / Slight Risk",
                "description": f"{category} items inherently have higher try-on return frequency than hard goods."
            })

    # 6. Price tier
    price = order["product_price"]
    if category == "Electronics" and price > 150:
        factors.append({
            "feature": "High-Value Electronic Item",
            "impact": "Moderate Risk Factor",
            "description": f"High-ticket electronic (${price:.2f}) invites stricter buyer scrutiny upon receipt."
        })

    # Sort: Prioritize High Risk Drivers when risk is medium/high; prioritize Mitigators if risk is low
    if risk_score >= 50:
        priority_map = {"High Risk Driver": 3, "Moderate Risk Factor": 2, "Neutral / Slight Risk": 1, "Risk Mitigator": 0}
    else:
        priority_map = {"Risk Mitigator": 3, "Neutral / Slight Risk": 2, "Moderate Risk Factor": 1, "High Risk Driver": 0}

    factors.sort(key=lambda x: priority_map.get(x["impact"], 0), reverse=True)
    return factors[:3]


def generate_recommendations(order: dict, risk_score: float) -> list:
    """
    Returns exactly two concrete, actionable seller recommendations based on order parameters.
    """
    recs = []
    category = order["product_category"]
    payment = order["payment_type"]
    delivery = order["delivery_days"]
    prev_returns = order["customer_previous_returns"]
    rating = order["product_rating"]

    # Rule 1: High customer return history or COD
    if prev_returns >= 3 or (payment == "COD" and risk_score >= 60):
        recs.append("Trigger an automated WhatsApp/SMS order verification request to confirm customer intent prior to dispatch.")
    elif payment == "COD":
        recs.append("Offer a $2-$5 prepaid instant discount incentive at confirmation to convert the order to prepaid.")

    # Rule 2: Sizing for clothing / shoes
    if category in ["Clothing", "Shoes"]:
        recs.append("Include an interactive size-fit guide confirmation link in the order dispatch email to verify sizing measurements.")

    # Rule 3: Extended delivery times
    if delivery >= 6 and len(recs) < 2:
        recs.append(f"Upgrade to expedited priority carrier shipping to compress the {delivery}-day transit and prevent buyer remorse.")

    # Rule 4: Low rating or quality concern
    if rating < 3.8 and len(recs) < 2:
        recs.append("Conduct an extra pre-shipment quality and packaging check, and enclose a dedicated customer satisfaction card.")

    # Fallback recommendations if fewer than 2 matched
    if len(recs) < 1:
        recs.append("Standard dispatch recommended. Send active milestone tracking notifications to keep buyer informed.")
    if len(recs) < 2:
        recs.append("Package securely with clear return/exchange instructions to streamline resolution if required.")

    return recs[:2]


def fallback_prediction(order: dict) -> dict:
    """
    Honest heuristic fallback calculation when the ML pipeline is unavailable.
    Clearly labeled so the system never falsely claims ML inference.
    """
    base_score = 25.0
    base_score += order["customer_previous_returns"] * 12.0
    if order["payment_type"] == "COD":
        base_score += 18.0
    if order["delivery_days"] >= 6:
        base_score += (order["delivery_days"] - 4) * 4.0
    if order["product_rating"] < 3.5:
        base_score += (4.0 - order["product_rating"]) * 14.0
    if order["product_category"] in ["Clothing", "Shoes"]:
        base_score += 8.0
        if order["size"] in ["XS", "XXL"]:
            base_score += 7.0

    score = int(round(np.clip(base_score, 5.0, 95.0)))
    level = get_risk_level(score)
    factors = compute_contributing_factors(order, score)
    recs = generate_recommendations(order, score)

    return {
        "risk_score": score,
        "risk_level": level,
        "probability": round(score / 100.0, 4),
        "contributing_factors": factors,
        "recommendations": recs,
        "explanation": f"Fallback rule-based risk calculated at {score}/100 ({level} Risk).",
        "is_fallback": True,
        "model_type": "Heuristic Rule-Based Fallback (ML Pipeline Artifact Offline)",
        "sanitized_input": order
    }


def predict_return_risk(raw_order: dict) -> dict:
    """
    Main prediction function. Sanitizes input, runs ML inference via scikit-learn
    pipeline, or falls back transparently.
    """
    order = sanitize_input(raw_order)
    pipeline = load_model()

    if pipeline is None:
        return fallback_prediction(order)

    try:
        prob = float(pipeline.predict_single_order(order))
        score = int(round(prob * 100))
        score = max(1, min(99, score))
        level = get_risk_level(score)

        factors = compute_contributing_factors(order, score)
        recs = generate_recommendations(order, score)

        return {
            "risk_score": score,
            "risk_level": level,
            "probability": round(prob, 4),
            "contributing_factors": factors,
            "recommendations": recs,
            "explanation": f"ML model estimated {score}% return probability ({level} Risk) based on trained behavioral patterns.",
            "is_fallback": False,
            "model_type": "scikit-learn Logistic Regression (Calibrated Pipeline)",
            "sanitized_input": order
        }
    except Exception as e:
        sys.stderr.write(f"[ERROR] Inference failed: {e}. Switching to fallback.\n")
        fb = fallback_prediction(order)
        fb["fallback_notice"] = f"Inference exception: {str(e)}"
        return fb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReturnWise AI Prediction CLI")
    parser.add_argument("--json", type=str, help="JSON order payload string")
    args = parser.parse_args()

    if args.json:
        try:
            payload = json.loads(args.json)
            result = predict_return_risk(payload)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(json.dumps({"error": f"Invalid JSON input: {str(e)}"}))
            sys.exit(1)
    else:
        sample = {
            "product_category": "Clothing",
            "product_price": 75.0,
            "size": "XXL",
            "delivery_days": 7,
            "customer_previous_returns": 3,
            "product_rating": 3.2,
            "payment_type": "COD"
        }
        print(json.dumps(predict_return_risk(sample), indent=2))
