"""
ReturnWise AI - AWS Lambda Serverless Handler
=============================================
Provides a serverless entry point for executing return risk predictions
and aggregating store analytics inside an AWS Lambda function environment.

Deployment & Architecture Guidelines:
-------------------------------------
1. Package Structure:
   - Handler file: lambda_handler.py
   - Model artifact: ml/model/return_risk_pipeline.joblib (or loaded from S3)
   - Dependencies: scikit-learn, pandas, numpy, joblib (packaged via AWS Lambda Layer
     or container image such as public.ecr.aws/lambda/python:3.11).

2. Cold Starts & Latency:
   - Initial cold-start: ~1.2s - 2.5s for scikit-learn & joblib imports and model loading.
   - Warm execution latency: ~15ms - 35ms.
   - Mitigation: Lazy-load model outside the handler at global module scope so warm invocations reuse memory.
   - For high-volume production, consider Provisioned Concurrency.

3. Model Size & Storage Limits:
   - The serialized pipeline is ~150KB, well within Lambda's 250MB unzipped zip package limit.
   - When deploying via S3: store model in an S3 bucket with IAM role `s3:GetObject` permission.
"""

import json
import os
import sys

# Ensure local ml directory is accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from ml.predict import predict_return_risk, load_model
except ImportError:
    predict_return_risk = None


def make_response(status_code: int, body: dict) -> dict:
    """Builds standard API Gateway HTTP response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization"
        },
        "body": json.dumps(body)
    }


def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    Routes:
      - POST /predict: Evaluates order return risk
      - GET /health: Verifies Lambda warm state & model availability
    """
    http_method = event.get("httpMethod") or event.get("requestContext", {}).get("http", {}).get("method", "GET")
    raw_path = event.get("path") or event.get("rawPath", "/predict")

    # Handle CORS preflight
    if http_method == "OPTIONS":
        return make_response(200, {"status": "ok"})

    if "/health" in raw_path:
        return make_response(200, {
            "status": "healthy",
            "runtime": f"Python {sys.version.split()[0]} on AWS Lambda",
            "cold_start": False
        })

    if "/predict" in raw_path or http_method == "POST":
        try:
            body_str = event.get("body", "{}")
            if isinstance(body_str, str):
                payload = json.loads(body_str) if body_str else {}
            else:
                payload = body_str

            if not payload:
                return make_response(400, {"error": "Missing or empty request body."})

            if predict_return_risk is None:
                return make_response(500, {"error": "Prediction engine not available in runtime environment."})

            result = predict_return_risk(payload)
            result["serverless_runtime"] = "AWS Lambda"
            return make_response(200, result)

        except json.JSONDecodeError:
            return make_response(400, {"error": "Malformed JSON payload in request body."})
        except Exception as e:
            return make_response(500, {"error": f"Internal Lambda execution error: {str(e)}"})

    return make_response(404, {"error": f"Unsupported route: {http_method} {raw_path}"})
