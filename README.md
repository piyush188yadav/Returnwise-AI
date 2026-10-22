# ReturnWise AI — Full-Stack Machine Learning E-Commerce Return Predictor

> **Predict E-Commerce Returns Before Dispatch** — Cut avoidable logistics losses, optimize size-fit guidance, and protect merchant profit margins with calibrated scikit-learn machine learning, real-time analytics, and optional AWS cloud intelligence.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Architecture & System Flow](#-architecture--system-flow)
3. [Technology Stack](#-technology-stack)
4. [Dataset & Fictional Benchmark Notice](#-dataset--fictional-benchmark-notice)
5. [Key Features](#-key-features)
6. [REST API Documentation](#-rest-api-documentation)
7. [Beginner-Friendly Windows Setup Guide](#-beginner-friendly-windows-setup-guide)
8. [Automated Testing & Verification](#-automated-testing--verification)
9. [Optional Cloud Deployments (AWS Bedrock, Lambda, S3)](#-optional-cloud-deployments)
10. [GitHub Repository Setup Commands](#-github-repository-setup-commands)
11. [Screenshots & UI Progression](#-screenshots--ui-progression)
12. [Known Limitations & Future Improvements](#-known-limitations--future-improvements)

---

## 🚀 Project Overview

High e-commerce return rates (often exceeding 20–30% in apparel and footwear) erode seller profitability through reverse shipping costs, restocking fees, and unsellable inventory.

**ReturnWise AI** solves this problem by:
- Ingesting multi-signal order data at checkout (category, price, sizing, delivery days, customer return history, star rating, and payment method).
- Applying a trained **scikit-learn** classification pipeline to compute a calibrated **return-risk score (0–100)** and assign **Low**, **Medium**, or **High** risk levels.
- Diagnosing the **top 3 contributing factors** driving the risk score.
- Recommending **exactly two targeted pre-dispatch seller interventions** (e.g. verifying sizing fit, offering prepaid discount incentives, or upgrading shipping).
- Providing an executive **Seller Analytics Dashboard** calculating return rates by category, frequent return reasons, and risk distributions from a 50-order benchmark dataset.
- Providing an optional **Amazon Bedrock** adapter for automated generative seller strategy insights, with transparent deterministic local fallback.

---

## 🏛️ Architecture & System Flow

```
                      ┌──────────────────────────────────────────────┐
                      │             ReturnWise AI Web UI             │
                      │   (Semantic HTML5, Responsive CSS3, JS)      │
                      └──────────────────────┬───────────────────────┘
                                             │ HTTP REST
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │          Node.js / Express Server            │
                      │                (Port: 3000)                  │
                      │  - Serves static assets                      │
                      │  - Input validation & CORS                   │
                      │  - Computes store analytics                  │
                      │  - Bedrock & S3 cloud adapters               │
                      └───────┬──────────────────────────────┬───────┘
                              │                              │
         Dual-Channel Dispatch│ (HTTP / CLI)                 │ Aggregated Stats
                              ▼                              ▼
    ┌──────────────────────────────────┐        ┌─────────────────────────┐
    │     Python ML Prediction Service │        │      Amazon Bedrock     │
    │            (Port: 5001)          │        │  (Claude 3 / Titan)     │
    │  - scikit-learn Pipeline         │        │  * Optional - with      │
    │  - StandardScaler + OneHotEncoder│        │    transparent fallback │
    │  - Logistic Regression Classifier│        └─────────────────────────┘
    │  - Calibrated Probability (0-100)│
    │  - Feature Diagnostics           │
    └──────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Technology | Role in ReturnWise AI |
|---|---|
| **HTML5 & CSS3** | Modern semantic layout, blue/white/navy theme, custom risk gauge meter, status badges, and mobile-responsive grid. |
| **Vanilla JavaScript & Chart.js** | Client-side input validation, dynamic tab routing, REST API communication, and interactive dashboard charts. |
| **Node.js & Express** | Main web server on port 3000, REST API routing, input schema validation, order filtering, and analytics calculations. |
| **Python 3.10+** | Scientific machine learning pipeline execution and lightweight microservice server (`ml/predict_service.py`). |
| **scikit-learn** | ColumnTransformer preprocessing, StandardScaler, OneHotEncoder, and calibrated LogisticRegression classification. |
| **NumPy & Pandas** | Synthetic order dataset generation (1200+ samples), feature array transformations, and statistical metrics. |
| **Amazon Bedrock (Optional)** | Generative AI integration using `@aws-sdk/client-bedrock-runtime` for executive store recommendations with deterministic fallback. |
| **Amazon S3 (Optional)** | Cloud storage adapter for model artifacts and dataset backup via `@aws-sdk/client-s3`. |
| **AWS Lambda (Optional)** | Serverless prediction handler (`aws/lambda_handler.py`) replicating the scikit-learn pipeline for cloud deployments. |

---

## 📊 Dataset & Fictional Benchmark Notice

> **⚠️ Fictional Demo Data Notice:**
> ReturnWise AI uses synthetic, demonstrative order records (`data/orders.json` and `data/orders_synthetic_train.csv`). It does **not** claim access to live Amazon customer accounts or private merchant databases. All customer IDs, names, and purchase figures are fictional and illustrative.

- **Training Dataset (`data/orders_synthetic_train.csv`):** 1,200 synthetic order records reflecting real-world e-commerce dynamics (e.g. COD orders having higher return rates, sizing fit variance in apparel, and delayed shipping heightening cancellations).
- **Seller Analytics Dataset (`data/orders.json`):** Exactly 50 structured fictional orders used by the Express backend to compute real-time metrics (return rates, reasons breakdown, and high-risk queues).

---

## 🎯 Key Features

### 1. Home Page & Overview
- Modern branding, problem statement, and live benchmark stats ribbon.
- Direct quick-action CTAs for "Check Order Risk" and "Seller Dashboard".
- Step-by-step workflow explanation and technical feature highlights.

### 2. Order Risk Checker
- Validated input fields: Category, Price, Size, Estimated Delivery Days, Customer's Previous Return Count, Product Rating, and Payment Type.
- **Category-Aware Logic:** Sizing options automatically disable and set to "Not applicable" when selecting *Electronics* or *Home*, reflecting realistic domain behavior.
- Quick Presets: Instant one-click demo payloads (*High-Risk Apparel*, *Low-Risk Electronics*, *Medium-Risk Shoes*).
- Visual Risk Meter with animated pointer and color-coded risk badge (Low: Green, Medium: Amber, High: Red).
- **Top 3 Contributing Factors:** Explains exactly which features drove the score.
- **Exactly 2 Actionable Recommendations:** Concrete steps to prevent avoidable returns.

### 3. Machine-Learning Engine
- Reproducible random seed (`RANDOM_STATE = 42`).
- Prevents data leakage: Preprocessing transformers are fitted exclusively on training splits.
- Model evaluation: Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics serialized to `ml/model/model_metrics.json`.
- Dual-channel Node-to-Python interface: Connects via local HTTP microservice or CLI subprocess, backed by a transparent rule-based fallback.

### 4. Seller Analytics Dashboard
- Interactive **Chart.js** charts for category return rates and frequent return reasons.
- Real-time multi-criteria filtering (Category, Risk Level, Payment Type, and Search query).
- Comprehensive orders table with sorting and risk indicators.
- AI Seller Insights card with one-click refresh powered by Bedrock or deterministic heuristic rules.

---

## 📡 REST API Documentation

### `GET /api/health`
Checks overall application health and service connectivity.
```json
{
  "status": "healthy",
  "application": "ReturnWise AI",
  "version": "1.0.0",
  "services": {
    "webServer": "active (Node.js/Express)",
    "pythonMlService": "connected",
    "amazonBedrock": "optional (unconfigured)",
    "amazonS3": "optional (unconfigured)"
  }
}
```

### `POST /api/predict`
Evaluates return risk for an incoming order.

**Request Payload:**
```json
{
  "product_category": "Clothing",
  "product_price": 89.00,
  "size": "XL",
  "delivery_days": 7,
  "customer_previous_returns": 3,
  "product_rating": 3.2,
  "payment_type": "COD"
}
```

**Response (Status: 200 OK):**
```json
{
  "success": true,
  "prediction": {
    "risk_score": 86,
    "risk_level": "High",
    "probability": 0.8624,
    "contributing_factors": [
      {
        "feature": "Customer Return History",
        "impact": "High Risk Driver",
        "description": "Customer has 3 previous returns, indicating habitual return behavior."
      },
      {
        "feature": "Payment Method (COD)",
        "impact": "High Risk Driver",
        "description": "Cash on Delivery orders demonstrate higher refusal and buyer-remorse return rates."
      },
      {
        "feature": "Estimated Delivery Time",
        "impact": "High Risk Driver",
        "description": "Long delivery window (7 days) increases cancellation and buyer impatience."
      }
    ],
    "recommendations": [
      "Trigger an automated WhatsApp/SMS order verification request to confirm customer intent prior to dispatch.",
      "Include an interactive size-fit guide confirmation link in the order dispatch email to verify sizing measurements."
    ],
    "explanation": "ML model estimated 86% return probability (High Risk) based on trained behavioral patterns.",
    "is_fallback": false,
    "model_type": "scikit-learn Logistic Regression (Calibrated Pipeline)"
  }
}
```

### `GET /api/orders`
Returns demo orders with optional query filters:
- `category`: `Clothing` | `Shoes` | `Electronics` | `Home` | `all`
- `risk_level`: `Low` | `Medium` | `High` | `all`
- `payment_type`: `COD` | `Prepaid` | `all`
- `search`: string filter for order ID, customer, or product name.

### `GET /api/analytics`
Returns aggregate statistics computed directly from `data/orders.json` (total orders, return rate, category breakdown, reason frequencies).

### `POST /api/bedrock-insights`
Generates strategic recommendations from aggregated metrics. Uses Amazon Bedrock if configured, or returns deterministic local recommendations.

---

## 💻 Beginner-Friendly Windows Setup Guide

Follow these exact step-by-step commands in **Windows PowerShell** or **Command Prompt**.

### Step 1: Verify Prerequisites
Check that Node.js and Python are installed:
```powershell
node -v
npm -v
python --version
```
*(Recommended: Node.js 18+ and Python 3.10+)*

### Step 2: Navigate to Project Directory
```powershell
cd "c:\Users\piyus\Downloads\Returnwise AI"
```

### Step 3: Install Node.js Dependencies
```powershell
npm install
```

### Step 4: Install Python Dependencies
```powershell
python -m pip install -r ml/requirements.txt
```

### Step 5: Train the Machine-Learning Model
Generate the synthetic dataset (1,200 records) and train the scikit-learn pipeline:
```powershell
npm run train
```
*Expected output:* You will see model evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC) and confirmation that `ml/model/return_risk_pipeline.joblib` was created.

---

### Step 6: Launch the Application

For optimal performance, use **two terminals**:

#### Terminal 1 — Start the Python ML Microservice:
```powershell
npm run ml-service
```
*Output: `[ML-SERVICE] ReturnWise AI ML Service listening on http://127.0.0.1:5001`*

#### Terminal 2 — Start the Node.js Web Server:
```powershell
npm start
```
*Output: `ReturnWise AI Server running at http://localhost:3000`*

> **Tip:** If you prefer running only one terminal, simply run `npm start` in Terminal 1. ReturnWise AI includes an automatic Python CLI adapter that invokes `ml/predict.py` directly on demand!

### Step 7: Open the Application
Open your browser and navigate to:
```
http://localhost:3000
```

---

## 🧪 Automated Testing & Verification

Run the full automated test suites to verify system integrity:

### Run Backend API & Analytics Tests:
```powershell
npm test
```
*Tests verified:*
- Validates that `data/orders.json` contains exactly 50 orders.
- Confirms mathematical accuracy of computed return rates.
- Enforces strict input validation on all order fields.
- Checks health and filtering endpoints.
- Verifies ML prediction schema (score range 1–99, 3 factors, 2 recommendations).
- Validates deterministic Bedrock fallback.

### Run Python ML Unit Tests:
```powershell
npm run test:ml
```
*Tests verified:*
- Synthetic dataset generation (>= 500 records).
- Category-specific sizing logic.
- Risk boundary thresholds (Low < 35, Medium 35–65, High > 65).
- Pipeline serialization and unpickling consistency.
- Transparent fallback labeling.

---

## ☁️ Optional Cloud Deployments

ReturnWise AI is **100% functional locally without an AWS account**. The cloud adapters below are optional enhancements:

### 1. Amazon Bedrock Setup
To enable live AI seller recommendations:
1. Copy `.env.example` to `.env` (already created).
2. Add your AWS IAM credentials:
   ```ini
   AWS_REGION=us-east-1
   AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
   AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
   BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
   ```
3. Ensure your IAM identity has `bedrock:InvokeModel` permissions.
4. If credentials are empty or invalid, the app automatically serves deterministic local insights.

### 2. Amazon S3 Model Backup
To back up trained models to an S3 bucket:
```ini
S3_BUCKET_NAME=my-returnwise-bucket
S3_REGION=us-east-1
```

### 3. AWS Lambda Serverless Handler
The serverless entry point is available in `aws/lambda_handler.py`:
- Deployable with an AWS Lambda Python 3.11/3.12 runtime.
- Package with standard dependencies via Lambda Layer or container image.
- Cold-start latency: ~1.5s for initial model import; warm execution: ~20ms.

---

## 🐙 GitHub Repository Setup Commands

To initialize a new Git repository and publish to GitHub:

```powershell
# 1. Initialize git in workspace
git init

# 2. Add all files (respects .gitignore)
git add .

# 3. Commit initial project version
git commit -m "feat: initial release of ReturnWise AI platform"

# 4. Set default branch to main
git branch -M main

# 5. Link to your personal GitHub repository (replace with your repo URL)
git remote add origin https://github.com/<your-username>/returnwise-ai.git

# 6. Push code to GitHub
git push -u origin main
```

---

## 📸 Screenshots & UI Progression

You can capture and add screenshots to your repository:
1. **Landing Page (`docs/screenshots/home.png`):** The hero section, value proposition, and statistics ribbon.
2. **Order Risk Checker (`docs/screenshots/checker.png`):** The form with category-aware size fields, animated gauge meter, and 2 recommendations.
3. **Seller Analytics Dashboard (`docs/screenshots/dashboard.png`):** Chart.js charts, filters, high-risk queue table, and AI insights.

---

## 🔍 Known Limitations & Future Improvements

- **Synthetic vs. Real-World Variance:** The classification model is trained on a synthetic distribution. In a commercial deployment, fine-tune the pipeline on at least 10,000 historical merchant orders.
- **Seasonality & Holiday Surges:** Current model features do not yet account for Black Friday / Cyber Monday impulse buying dynamics.
- **Multi-Store Support:** Future versions will introduce tenant isolation for multi-brand Shopify / WooCommerce integrations.
- **Automated Carrier API Webhooks:** Direct integrations with carriers (e.g. FedEx, DHL, Delhivery) to trigger automatic SMS customer verification upon transit delays.

---

*ReturnWise AI — Built with precision for sustainable, high-margin e-commerce.*
