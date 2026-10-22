/**
 * ReturnWise AI - Automated Backend & REST API Tests
 * ==================================================
 * Validates:
 * 1. Health check endpoint response structure
 * 2. Order dataset verification (exactly 50 demo orders)
 * 3. Analytics metric calculations & accuracy
 * 4. Prediction API validation (rejects invalid/missing fields, accepts valid input)
 * 5. Python ML prediction integration & calibrated score ranges
 * 6. Bedrock insights endpoint and graceful deterministic fallback
 */

const { test, describe, before } = require("node:test");
const assert = require("node:assert");
const fs = require("fs");
const path = require("path");

const { app, computeAnalytics, validateOrderInput } = require("../server");

// Helper to simulate HTTP requests without external network listeners
async function simulateRequest(app, method, url, body = null) {
  // Use Node.js http test server
  return new Promise((resolve, reject) => {
    const server = app.listen(0, async () => {
      const port = server.address().port;
      try {
        const options = {
          method,
          headers: { "Content-Type": "application/json" }
        };
        if (body) {
          options.body = JSON.stringify(body);
        }
        const res = await fetch(`http://127.0.0.1:${port}${url}`, options);
        const data = await res.json();
        server.close(() => resolve({ status: res.status, data }));
      } catch (err) {
        server.close(() => reject(err));
      }
    });
  });
}

describe("ReturnWise AI - REST API & Analytics Verification", () => {
  const ordersFilePath = path.join(__dirname, "..", "data", "orders.json");

  test("1. Dataset Verification: data/orders.json contains exactly 50 fictional orders", () => {
    assert.strictEqual(fs.existsSync(ordersFilePath), true, "data/orders.json must exist");
    const rawData = fs.readFileSync(ordersFilePath, "utf-8");
    const orders = JSON.parse(rawData);

    assert.strictEqual(Array.isArray(orders), true, "orders.json must contain a JSON array");
    assert.strictEqual(orders.length, 50, "orders.json must contain EXACTLY 50 demo orders");

    // Verify required fields on each order
    orders.forEach((order, idx) => {
      assert.ok(order.order_id, `Order at index ${idx} missing order_id`);
      assert.ok(order.product_category, `Order ${order.order_id} missing product_category`);
      assert.ok(typeof order.product_price === "number", `Order ${order.order_id} price must be a number`);
      assert.ok(["COD", "Prepaid"].includes(order.payment_type), `Order ${order.order_id} invalid payment_type`);
      assert.ok(typeof order.risk_score === "number", `Order ${order.risk_score} risk_score must be a number`);
      assert.ok(["Low", "Medium", "High"].includes(order.risk_level), `Order ${order.risk_level} invalid risk_level`);
      assert.ok(typeof order.returned === "boolean", `Order ${order.order_id} returned must be boolean`);
      assert.ok(typeof order.delivery_days === "number", `Order ${order.order_id} delivery_days must be number`);
      assert.ok(typeof order.product_rating === "number", `Order ${order.order_id} product_rating must be number`);
    });
  });

  test("2. Analytics Calculation: accurate rates computed from demo data", () => {
    const rawData = fs.readFileSync(ordersFilePath, "utf-8");
    const orders = JSON.parse(rawData);
    const analytics = computeAnalytics(orders);

    assert.strictEqual(analytics.totalOrders, 50, "Analytics totalOrders must equal 50");
    assert.strictEqual(typeof analytics.totalReturns, "number");
    assert.ok(analytics.totalReturns > 0, "Must have returned orders in dataset");
    assert.strictEqual(typeof analytics.overallReturnRate, "number");
    assert.ok(analytics.overallReturnRate > 0 && analytics.overallReturnRate < 100);

    // Verify category return rates
    assert.ok(Array.isArray(analytics.categoryStats));
    assert.strictEqual(analytics.categoryStats.length, 4, "Must cover all 4 categories");

    // Verify sum of categories equals 50
    const catOrdersSum = analytics.categoryStats.reduce((sum, c) => sum + c.totalOrders, 0);
    assert.strictEqual(catOrdersSum, 50, "Sum of category orders must equal 50");
  });

  test("3. Input Validation: rejects invalid categories, out-of-range ratings, negative prices", () => {
    // Missing required fields
    const err1 = validateOrderInput({});
    assert.ok(err1.length > 0, "Empty payload must produce validation errors");

    // Invalid category
    const err2 = validateOrderInput({
      product_category: "Automotive",
      product_price: 50,
      delivery_days: 4,
      customer_previous_returns: 0,
      product_rating: 4.5,
      payment_type: "Prepaid"
    });
    assert.ok(err2.some((e) => e.includes("product_category")));

    // Out of range rating
    const err3 = validateOrderInput({
      product_category: "Clothing",
      product_price: 50,
      delivery_days: 4,
      customer_previous_returns: 0,
      product_rating: 6.5,
      payment_type: "Prepaid"
    });
    assert.ok(err3.some((e) => e.includes("product_rating")));

    // Negative price
    const err4 = validateOrderInput({
      product_category: "Clothing",
      product_price: -10,
      delivery_days: 4,
      customer_previous_returns: 0,
      product_rating: 4.0,
      payment_type: "Prepaid"
    });
    assert.ok(err4.some((e) => e.includes("product_price")));
  });

  test("4. GET /api/health: reports healthy status and service configurations", async () => {
    const res = await simulateRequest(app, "GET", "/api/health");
    assert.strictEqual(res.status, 200);
    assert.strictEqual(res.data.status, "healthy");
    assert.strictEqual(res.data.application, "ReturnWise AI");
    assert.ok(res.data.services);
  });

  test("5. GET /api/orders: supports category and risk-level filtering", async () => {
    const resAll = await simulateRequest(app, "GET", "/api/orders");
    assert.strictEqual(resAll.status, 200);
    assert.strictEqual(resAll.data.count, 50);

    const resClothing = await simulateRequest(app, "GET", "/api/orders?category=Clothing");
    assert.strictEqual(resClothing.status, 200);
    resClothing.data.orders.forEach((o) => {
      assert.strictEqual(o.product_category, "Clothing");
    });

    const resHigh = await simulateRequest(app, "GET", "/api/orders?risk_level=High");
    assert.strictEqual(resHigh.status, 200);
    resHigh.data.orders.forEach((o) => {
      assert.strictEqual(o.risk_level, "High");
    });
  });

  test("6. POST /api/predict: returns calibrated score, risk level, 3 factors, 2 recommendations", async () => {
    const validOrder = {
      product_category: "Clothing",
      product_price: 89.0,
      size: "XL",
      delivery_days: 7,
      customer_previous_returns: 3,
      product_rating: 3.2,
      payment_type: "COD"
    };

    const res = await simulateRequest(app, "POST", "/api/predict", validOrder);
    assert.strictEqual(res.status, 200);
    assert.strictEqual(res.data.success, true);

    const pred = res.data.prediction;
    assert.ok(typeof pred.risk_score === "number", "risk_score must be number");
    assert.ok(pred.risk_score >= 1 && pred.risk_score <= 99, "risk_score must be between 1 and 99");
    assert.ok(["Low", "Medium", "High"].includes(pred.risk_level), "risk_level must be Low, Medium, or High");

    // Contributing factors
    assert.ok(Array.isArray(pred.contributing_factors), "contributing_factors must be array");
    assert.strictEqual(pred.contributing_factors.length, 3, "must return exactly 3 top contributing factors");

    // Actionable recommendations
    assert.ok(Array.isArray(pred.recommendations), "recommendations must be array");
    assert.strictEqual(pred.recommendations.length, 2, "must return exactly 2 actionable recommendations");
  });

  test("7. POST /api/bedrock-insights: provides recommendations with deterministic fallback", async () => {
    const res = await simulateRequest(app, "POST", "/api/bedrock-insights");
    assert.strictEqual(res.status, 200);
    assert.strictEqual(res.data.success, true);
    assert.ok(Array.isArray(res.data.insights));
    assert.strictEqual(res.data.insights.length, 4, "Must provide 4 recommendations");
    assert.ok(res.data.source);
  });
});
