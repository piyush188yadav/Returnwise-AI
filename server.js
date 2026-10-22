/**
 * ReturnWise AI - Node.js Express Application Server
 * ==================================================
 * Serves the frontend web interface, REST API endpoints, analytics calculations,
 * Python ML prediction service interface, and optional AWS Bedrock / S3 integrations.
 */

require("dotenv").config();
const express = require("express");
const cors = require("cors");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");

const bedrockService = require("./services/bedrockService");
const s3Service = require("./services/s3Service");

const app = express();
const PORT = process.env.PORT || 3000;
const ML_SERVICE_URL = process.env.PYTHON_ML_SERVICE_URL || "http://127.0.0.1:5001";
const PYTHON_PATH = process.env.PYTHON_PATH || "python";

// Middleware
app.use(cors());
app.use(express.json({ limit: "1mb" }));
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, "public")));

// Load demo orders dataset
const ORDERS_FILE = path.join(__dirname, "data", "orders.json");

function loadDemoOrders() {
  try {
    if (!fs.existsSync(ORDERS_FILE)) {
      return [];
    }
    const raw = fs.readFileSync(ORDERS_FILE, "utf-8");
    return JSON.parse(raw);
  } catch (err) {
    console.error("[SERVER] Failed to read orders.json:", err.message);
    return [];
  }
}

/**
 * Validates order risk check input
 */
function validateOrderInput(body) {
  const errors = [];
  const validCategories = ["Clothing", "Shoes", "Electronics", "Home"];
  const validSizes = ["Not applicable", "XS", "S", "M", "L", "XL", "XXL"];
  const validPayments = ["COD", "Prepaid"];

  if (!body.product_category || !validCategories.includes(body.product_category)) {
    errors.push(`Invalid product_category. Allowed: ${validCategories.join(", ")}`);
  }

  const price = parseFloat(body.product_price);
  if (isNaN(price) || price <= 0 || price > 50000) {
    errors.push("product_price must be a positive number up to 50,000");
  }

  if (body.size && !validSizes.includes(body.size)) {
    errors.push(`Invalid size. Allowed: ${validSizes.join(", ")}`);
  }

  const deliveryDays = parseInt(body.delivery_days, 10);
  if (isNaN(deliveryDays) || deliveryDays < 1 || deliveryDays > 60) {
    errors.push("delivery_days must be an integer between 1 and 60");
  }

  const prevReturns = parseInt(body.customer_previous_returns, 10);
  if (isNaN(prevReturns) || prevReturns < 0 || prevReturns > 100) {
    errors.push("customer_previous_returns must be an integer between 0 and 100");
  }

  const rating = parseFloat(body.product_rating);
  if (isNaN(rating) || rating < 1.0 || rating > 5.0) {
    errors.push("product_rating must be a number between 1.0 and 5.0");
  }

  if (!body.payment_type || !validPayments.includes(body.payment_type)) {
    errors.push(`Invalid payment_type. Allowed: ${validPayments.join(", ")}`);
  }

  return errors;
}

/**
 * Communicates with the Python ML prediction service.
 * Strategy:
 * 1. Attempts HTTP request to local Python microservice (default port 5001)
 * 2. If microservice is unavailable, executes python CLI script as fallback
 * 3. If python execution fails, returns a clearly documented heuristic fallback
 */
async function getPredictionFromPythonService(orderData) {
  // Strategy 1: Local HTTP microservice
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);

    const response = await fetch(`${ML_SERVICE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(orderData),
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    if (response.ok) {
      const result = await response.json();
      result.channel = "HTTP Microservice (Port 5001)";
      return result;
    }
  } catch (httpErr) {
    // Microservice not responding or timed out; fall back to CLI runner
  }

  // Strategy 2: Python CLI runner
  return new Promise((resolve) => {
    const scriptPath = path.join(__dirname, "ml", "predict.py");
    const jsonArg = JSON.stringify(orderData);

    const pyProcess = spawn(PYTHON_PATH, [scriptPath, "--json", jsonArg]);

    let stdout = "";
    let stderr = "";

    pyProcess.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    pyProcess.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    pyProcess.on("close", (code) => {
      if (code === 0 && stdout.trim()) {
        try {
          const parsed = JSON.parse(stdout);
          parsed.channel = "Python CLI Subprocess Adapter";
          return resolve(parsed);
        } catch (e) {
          // JSON parsing failed
        }
      }

      // Strategy 3: Built-in Node heuristic fallback
      console.warn("[PREDICT] Python inference offline; using documented rule fallback. Stderr:", stderr);
      resolve(createNodeFallbackPrediction(orderData, stderr));
    });

    pyProcess.on("error", (err) => {
      console.warn("[PREDICT] Failed to spawn Python process:", err.message);
      resolve(createNodeFallbackPrediction(orderData, err.message));
    });
  });
}

function createNodeFallbackPrediction(order, errorDetail) {
  let score = 25;
  score += (parseInt(order.customer_previous_returns, 10) || 0) * 12;
  if (order.payment_type === "COD") score += 18;
  const delivery = parseInt(order.delivery_days, 10) || 4;
  if (delivery >= 6) score += (delivery - 4) * 4;
  const rating = parseFloat(order.product_rating) || 4.0;
  if (rating < 3.5) score += (4.0 - rating) * 14;
  if (["Clothing", "Shoes"].includes(order.product_category)) {
    score += 8;
    if (["XS", "XXL"].includes(order.size)) score += 7;
  }
  score = Math.max(5, Math.min(95, Math.round(score)));

  const level = score < 35 ? "Low" : score <= 65 ? "Medium" : "High";

  return {
    risk_score: score,
    risk_level: level,
    probability: Number((score / 100).toFixed(4)),
    contributing_factors: [
      {
        feature: "Customer Return History",
        impact: order.customer_previous_returns >= 2 ? "High Risk Driver" : "Risk Mitigator",
        description: `Customer account has ${order.customer_previous_returns} recorded prior return(s).`
      },
      {
        feature: "Payment Method",
        impact: order.payment_type === "COD" ? "High Risk Driver" : "Risk Mitigator",
        description: `${order.payment_type} payment method influences customer refusal frequency.`
      },
      {
        feature: "Delivery Window",
        impact: delivery >= 6 ? "Moderate Risk Factor" : "Risk Mitigator",
        description: `Estimated fulfillment of ${delivery} day(s).`
      }
    ],
    recommendations: [
      order.payment_type === "COD"
        ? "Offer an instant prepaid checkout discount to convert payment type."
        : "Trigger standard shipment tracking notifications to maintain customer awareness.",
      ["Clothing", "Shoes"].includes(order.product_category)
        ? "Attach an interactive sizing guide in the confirmation email."
        : "Conduct a pre-dispatch quality verification check."
    ],
    explanation: `Fallback rule calculation evaluated this order at ${score}/100 (${level} Risk).`,
    is_fallback: true,
    channel: "Node.js Heuristic Fallback Engine",
    fallback_notice: "Python service was unreachable; calculated via documented rule fallback.",
    error_detail: errorDetail || null
  };
}

/**
 * Calculates dashboard metrics from the orders array
 */
function computeAnalytics(orders) {
  const totalOrders = orders.length;
  if (totalOrders === 0) {
    return {
      totalOrders: 0,
      totalReturns: 0,
      returnRate: "0.0",
      highRiskCount: 0,
      mediumRiskCount: 0,
      lowRiskCount: 0,
      categoryStats: {},
      returnReasons: {},
      paymentStats: {},
      highRiskOrders: []
    };
  }

  let totalReturns = 0;
  let highRiskCount = 0;
  let mediumRiskCount = 0;
  let lowRiskCount = 0;

  const categoryMap = {};
  const returnReasonsMap = {};
  const paymentMap = {
    COD: { total: 0, returned: 0 },
    Prepaid: { total: 0, returned: 0 }
  };

  orders.forEach((order) => {
    const isReturned = Boolean(order.returned || order.return_status === "Returned");
    if (isReturned) totalReturns++;

    if (order.risk_level === "High" || order.risk_score > 65) highRiskCount++;
    else if (order.risk_level === "Medium" || order.risk_score >= 35) mediumRiskCount++;
    else lowRiskCount++;

    // Category stats
    const cat = order.product_category || "Other";
    if (!categoryMap[cat]) {
      categoryMap[cat] = { total: 0, returned: 0, sumPrice: 0 };
    }
    categoryMap[cat].total++;
    categoryMap[cat].sumPrice += Number(order.product_price) || 0;
    if (isReturned) categoryMap[cat].returned++;

    // Return reasons
    if (isReturned && order.return_reason) {
      returnReasonsMap[order.return_reason] = (returnReasonsMap[order.return_reason] || 0) + 1;
    }

    // Payment stats
    const pType = order.payment_type === "COD" ? "COD" : "Prepaid";
    paymentMap[pType].total++;
    if (isReturned) paymentMap[pType].returned++;
  });

  const categoryStats = Object.keys(categoryMap).map((cat) => {
    const item = categoryMap[cat];
    const rate = item.total > 0 ? (item.returned / item.total) * 100 : 0;
    return {
      category: cat,
      totalOrders: item.total,
      returnedOrders: item.returned,
      returnRate: Number(rate.toFixed(1)),
      avgPrice: Number((item.sumPrice / item.total).toFixed(2))
    };
  });

  const returnReasons = Object.entries(returnReasonsMap)
    .map(([reason, count]) => ({ reason, count }))
    .sort((a, b) => b.count - a.count);

  const paymentStats = Object.keys(paymentMap).map((pt) => {
    const item = paymentMap[pt];
    const rate = item.total > 0 ? (item.returned / item.total) * 100 : 0;
    return {
      paymentType: pt,
      totalOrders: item.total,
      returnedOrders: item.returned,
      returnRate: Number(rate.toFixed(1))
    };
  });

  const overallReturnRate = Number(((totalReturns / totalOrders) * 100).toFixed(1));

  const highRiskOrders = orders
    .filter((o) => o.risk_level === "High" || o.risk_score > 65)
    .slice(0, 10);

  return {
    totalOrders,
    totalReturns,
    overallReturnRate,
    highRiskCount,
    mediumRiskCount,
    lowRiskCount,
    categoryStats,
    returnReasons,
    paymentStats,
    highRiskOrders
  };
}

// ---------------- REST API ROUTES ----------------

// GET /api/health
app.get("/api/health", async (req, res) => {
  let mlHealth = "checking";
  try {
    const controller = new AbortController();
    const tId = setTimeout(() => controller.abort(), 1000);
    const resp = await fetch(`${ML_SERVICE_URL}/health`, { signal: controller.signal });
    clearTimeout(tId);
    mlHealth = resp.ok ? "connected" : "unreachable";
  } catch (e) {
    mlHealth = "offline (using local CLI adapter or fallback)";
  }

  res.json({
    status: "healthy",
    application: "ReturnWise AI",
    version: "1.0.0",
    environment: process.env.NODE_ENV || "development",
    timestamp: new Date().toISOString(),
    services: {
      webServer: "active (Node.js/Express)",
      pythonMlService: mlHealth,
      amazonBedrock: bedrockService.isConfigured ? "configured" : "optional (unconfigured)",
      amazonS3: s3Service.isConfigured ? "configured" : "optional (unconfigured)"
    }
  });
});

// POST /api/predict
app.post("/api/predict", async (req, res) => {
  const errors = validateOrderInput(req.body);
  if (errors.length > 0) {
    return res.status(400).json({
      error: "Validation Failed",
      details: errors
    });
  }

  try {
    const prediction = await getPredictionFromPythonService(req.body);
    res.json({
      success: true,
      timestamp: new Date().toISOString(),
      order: req.body,
      prediction
    });
  } catch (err) {
    console.error("[PREDICT] Unexpected error:", err);
    res.status(500).json({
      error: "Prediction processing failed",
      message: err.message
    });
  }
});

// GET /api/orders
app.get("/api/orders", (req, res) => {
  let orders = loadDemoOrders();
  const { category, risk_level, payment_type, search } = req.query;

  if (category && category !== "all") {
    orders = orders.filter((o) => (o.product_category || "").toLowerCase() === category.toLowerCase());
  }

  if (risk_level && risk_level !== "all") {
    orders = orders.filter((o) => (o.risk_level || "").toLowerCase() === risk_level.toLowerCase());
  }

  if (payment_type && payment_type !== "all") {
    orders = orders.filter((o) => (o.payment_type || "").toLowerCase() === payment_type.toLowerCase());
  }

  if (search && search.trim()) {
    const q = search.trim().toLowerCase();
    orders = orders.filter((o) =>
      (o.order_id || "").toLowerCase().includes(q) ||
      (o.product_name || "").toLowerCase().includes(q) ||
      (o.customer_id || "").toLowerCase().includes(q)
    );
  }

  res.json({
    success: true,
    count: orders.length,
    orders
  });
});

// GET /api/analytics
app.get("/api/analytics", (req, res) => {
  const orders = loadDemoOrders();
  const analytics = computeAnalytics(orders);
  res.json({
    success: true,
    calculatedAt: new Date().toISOString(),
    isDemoDataset: true,
    analytics
  });
});

// GET /api/insights
app.get("/api/insights", (req, res) => {
  const orders = loadDemoOrders();
  const analytics = computeAnalytics(orders);
  const fallback = bedrockService.getDeterministicFallbackInsights({
    totalOrders: analytics.totalOrders,
    totalReturns: analytics.totalReturns,
    returnRate: analytics.overallReturnRate,
    highRiskCount: analytics.highRiskCount,
    topReturnReasons: analytics.returnReasons.slice(0, 3),
    categoryReturnRates: analytics.categoryStats
  });

  res.json({
    success: true,
    ...fallback
  });
});

// POST /api/bedrock-insights
app.post("/api/bedrock-insights", async (req, res) => {
  try {
    const orders = loadDemoOrders();
    const analytics = computeAnalytics(orders);

    const summaryMetrics = {
      totalOrders: analytics.totalOrders,
      totalReturns: analytics.totalReturns,
      returnRate: analytics.overallReturnRate,
      highRiskCount: analytics.highRiskCount,
      topReturnReasons: analytics.returnReasons.slice(0, 3),
      categoryReturnRates: analytics.categoryStats
    };

    const result = await bedrockService.generateInsights(summaryMetrics);
    res.json({
      success: true,
      timestamp: new Date().toISOString(),
      ...result
    });
  } catch (err) {
    res.status(500).json({
      error: "Bedrock insights generation failed",
      message: err.message
    });
  }
});

// Catch-all route to serve index.html for SPA
app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

// Start Server
if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`====================================================`);
    console.log(` ReturnWise AI Server running at http://localhost:${PORT}`);
    console.log(` Mode: ${process.env.NODE_ENV || "development"}`);
    console.log(` Machine Learning Service URL: ${ML_SERVICE_URL}`);
    console.log(` AWS Bedrock: ${bedrockService.isConfigured ? "Configured" : "Offline / Deterministic Fallback"}`);
    console.log(` Amazon S3: ${s3Service.isConfigured ? "Configured" : "Offline / Local Storage"}`);
    console.log(`====================================================`);
  });
}

module.exports = { app, computeAnalytics, validateOrderInput };
