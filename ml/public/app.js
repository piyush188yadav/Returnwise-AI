/**
 * ReturnWise AI - Frontend Application Client
 * ===========================================
 * Handles view navigation, order risk form validation, ML prediction API requests,
 * Chart.js visualizations, dashboard filters, and Bedrock seller insights.
 */

class ReturnWiseApp {
  constructor() {
    this.currentOrders = [];
    this.analyticsData = null;
    this.categoryChart = null;
    this.reasonsChart = null;
    this.init();
  }

  init() {
    this.bindNavigation();
    this.bindFormEvents();
    this.checkHealth();
    this.loadInitialDashboard();
  }

  // Navigation & Tab Switching
  bindNavigation() {
    document.querySelectorAll(".nav-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetView = btn.getAttribute("data-target");
        this.switchView(targetView);
      });
    });

    // Brand link clicks
    const brandLink = document.getElementById("brand-link");
    if (brandLink) {
      brandLink.addEventListener("click", (e) => {
        e.preventDefault();
        this.switchView("home-view");
      });
    }

    // Hash navigation support
    window.addEventListener("hashchange", () => {
      const hash = window.location.hash.replace("#", "");
      if (hash === "checker") this.switchView("checker-view");
      else if (hash === "dashboard") this.switchView("dashboard-view");
      else this.switchView("home-view");
    });
  }

  switchView(viewId) {
    document.querySelectorAll(".page-view").forEach((view) => {
      view.classList.remove("active");
    });
    document.querySelectorAll(".nav-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-target") === viewId);
    });

    const targetEl = document.getElementById(viewId);
    if (targetEl) {
      targetEl.classList.add("active");
      window.scrollTo({ top: 0, behavior: "smooth" });
    }

    // If switching to dashboard, ensure charts resize properly
    if (viewId === "dashboard-view") {
      setTimeout(() => {
        if (this.categoryChart) this.categoryChart.resize();
        if (this.reasonsChart) this.reasonsChart.resize();
      }, 150);
    }
  }

  // System Health Monitoring
  async checkHealth() {
    const textEl = document.getElementById("health-status-text");
    const pillEl = document.getElementById("system-health-pill");

    try {
      const res = await fetch("/api/health");
      if (res.ok) {
        const data = await res.json();
        const mlActive = data.services?.pythonMlService?.includes("connected");
        textEl.textContent = mlActive ? "ML Online" : "System Ready";
        pillEl.title = `Application: ${data.application} | ML: ${data.services.pythonMlService} | Bedrock: ${data.services.amazonBedrock}`;
      } else {
        textEl.textContent = "Offline";
      }
    } catch (e) {
      textEl.textContent = "Offline";
    }
  }

  // Form Interactions & Category-Specific Logic
  bindFormEvents() {
    const categorySelect = document.getElementById("product_category");
    const sizeSelect = document.getElementById("size");
    const sizeHint = document.getElementById("size-hint");

    // Category-specific sizing logic
    categorySelect.addEventListener("change", () => {
      const cat = categorySelect.value;
      if (cat === "Electronics" || cat === "Home") {
        sizeSelect.value = "Not applicable";
        sizeSelect.disabled = true;
        sizeHint.textContent = "(Not applicable for " + cat + ")";
      } else {
        sizeSelect.disabled = false;
        if (sizeSelect.value === "Not applicable") {
          sizeSelect.value = "M";
        }
        sizeHint.textContent = "(Applies to " + cat.toLowerCase() + ")";
      }
    });

    // Payment Segmented Control
    const btnPrepaid = document.getElementById("btn-pay-prepaid");
    const btnCod = document.getElementById("btn-pay-cod");
    const hiddenPayment = document.getElementById("payment_type");

    btnPrepaid.addEventListener("click", () => {
      btnPrepaid.classList.add("active");
      btnCod.classList.remove("active");
      hiddenPayment.value = "Prepaid";
    });

    btnCod.addEventListener("click", () => {
      btnCod.classList.add("active");
      btnPrepaid.classList.remove("active");
      hiddenPayment.value = "COD";
    });

    // Form Submit
    const form = document.getElementById("order-risk-form");
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      this.handleFormSubmit();
    });
  }

  // Load Presets for instant user testing
  loadPreset(type) {
    const categorySelect = document.getElementById("product_category");
    const priceInput = document.getElementById("product_price");
    const sizeSelect = document.getElementById("size");
    const deliveryInput = document.getElementById("delivery_days");
    const returnsInput = document.getElementById("customer_previous_returns");
    const ratingInput = document.getElementById("product_rating");
    const btnPrepaid = document.getElementById("btn-pay-prepaid");
    const btnCod = document.getElementById("btn-pay-cod");
    const hiddenPayment = document.getElementById("payment_type");

    if (type === "high-risk-clothing") {
      categorySelect.value = "Clothing";
      categorySelect.dispatchEvent(new Event("change"));
      priceInput.value = "115.00";
      sizeSelect.value = "XXL";
      deliveryInput.value = "8";
      returnsInput.value = "4";
      ratingInput.value = "3.1";
      btnCod.click();
    } else if (type === "low-risk-electronics") {
      categorySelect.value = "Electronics";
      categorySelect.dispatchEvent(new Event("change"));
      priceInput.value = "129.50";
      sizeSelect.value = "Not applicable";
      deliveryInput.value = "2";
      returnsInput.value = "0";
      ratingInput.value = "4.8";
      btnPrepaid.click();
    } else if (type === "med-risk-shoes") {
      categorySelect.value = "Shoes";
      categorySelect.dispatchEvent(new Event("change"));
      priceInput.value = "89.00";
      sizeSelect.value = "XL";
      deliveryInput.value = "5";
      returnsInput.value = "2";
      ratingInput.value = "3.8";
      btnCod.click();
    }

    // Smooth scroll to submit button
    document.getElementById("predict-submit-btn").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // Handle Order Risk Prediction Submission
  async handleFormSubmit() {
    const errorBanner = document.getElementById("form-error-banner");
    errorBanner.style.display = "none";
    errorBanner.textContent = "";

    const payload = {
      product_category: document.getElementById("product_category").value,
      product_price: parseFloat(document.getElementById("product_price").value),
      size: document.getElementById("size").value,
      delivery_days: parseInt(document.getElementById("delivery_days").value, 10),
      customer_previous_returns: parseInt(document.getElementById("customer_previous_returns").value, 10),
      product_rating: parseFloat(document.getElementById("product_rating").value),
      payment_type: document.getElementById("payment_type").value
    };

    // Client-side validation
    if (isNaN(payload.product_price) || payload.product_price <= 0) {
      this.showFormError("Please enter a valid positive price.");
      return;
    }
    if (isNaN(payload.delivery_days) || payload.delivery_days < 1) {
      this.showFormError("Delivery days must be at least 1.");
      return;
    }
    if (isNaN(payload.customer_previous_returns) || payload.customer_previous_returns < 0) {
      this.showFormError("Customer previous returns cannot be negative.");
      return;
    }
    if (isNaN(payload.product_rating) || payload.product_rating < 1.0 || payload.product_rating > 5.0) {
      this.showFormError("Product rating must be between 1.0 and 5.0.");
      return;
    }

    // UI Loading state
    document.getElementById("results-empty").style.display = "none";
    document.getElementById("results-content").style.display = "none";
    document.getElementById("results-loading").style.display = "block";

    const submitBtn = document.getElementById("predict-submit-btn");
    submitBtn.disabled = true;

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.details?.join(", ") || data.error || "Prediction request failed.");
      }

      this.renderPredictionResults(data.prediction);
    } catch (err) {
      this.showFormError(err.message || "Failed to communicate with prediction service.");
      document.getElementById("results-loading").style.display = "none";
      document.getElementById("results-empty").style.display = "block";
    } finally {
      submitBtn.disabled = false;
    }
  }

  showFormError(msg) {
    const errorBanner = document.getElementById("form-error-banner");
    errorBanner.textContent = msg;
    errorBanner.style.display = "block";
    errorBanner.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // Render Risk Meter & Diagnostic Breakdown
  renderPredictionResults(prediction) {
    document.getElementById("results-loading").style.display = "none";
    document.getElementById("results-content").style.display = "block";

    const score = Math.max(0, Math.min(100, prediction.risk_score || 0));
    const level = prediction.risk_level || (score < 35 ? "Low" : score <= 65 ? "Medium" : "High");

    // Score & Gauge Pointer
    document.getElementById("score-value").textContent = score;
    const pointer = document.getElementById("meter-pointer");
    pointer.style.left = `${score}%`;

    // Risk Badge
    const badge = document.getElementById("risk-badge");
    badge.className = "badge";
    if (level === "High") {
      badge.classList.add("badge-high");
      badge.textContent = "🚨 High Risk";
    } else if (level === "Medium") {
      badge.classList.add("badge-med");
      badge.textContent = "⚠️ Medium Risk";
    } else {
      badge.classList.add("badge-low");
      badge.textContent = "✅ Low Risk";
    }

    // Explanation
    document.getElementById("prediction-explanation").textContent =
      prediction.explanation || `Estimated return risk evaluated at ${score}/100.`;

    // Contributing Factors (3)
    const factorsList = document.getElementById("factors-list");
    factorsList.innerHTML = "";

    const factors = prediction.contributing_factors || [];
    factors.slice(0, 3).forEach((f) => {
      const item = document.createElement("div");
      item.className = "factor-item";

      let pillClass = "badge-med";
      if (f.impact?.includes("High")) pillClass = "badge-high";
      else if (f.impact?.includes("Mitigator")) pillClass = "badge-low";

      item.innerHTML = `
        <div class="factor-top">
          <span class="factor-name">${f.feature || "Signal"}</span>
          <span class="factor-pill badge ${pillClass}">${f.impact || "Evaluated"}</span>
        </div>
        <div class="factor-desc">${f.description || ""}</div>
      `;
      factorsList.appendChild(item);
    });

    // Recommendations (Exactly 2)
    const recsList = document.getElementById("recommendations-list");
    recsList.innerHTML = "";

    const recs = prediction.recommendations || [];
    recs.slice(0, 2).forEach((recText, idx) => {
      const card = document.createElement("div");
      card.className = "rec-card";
      card.innerHTML = `
        <div class="rec-icon">⚡</div>
        <div class="rec-text">
          <strong>Action ${idx + 1}:</strong> ${recText}
        </div>
      `;
      recsList.appendChild(card);
    });

    // Engine Info
    const modelTag = document.getElementById("engine-model-type");
    const statusFlag = document.getElementById("engine-status-flag");

    if (prediction.is_fallback) {
      modelTag.textContent = "Rule-Based Heuristic Fallback";
      statusFlag.textContent = "● Fallback Mode";
      statusFlag.style.color = "#b45309";
    } else {
      modelTag.textContent = prediction.model_type || "scikit-learn Pipeline";
      statusFlag.textContent = "● Calibrated ML Inference";
      statusFlag.style.color = "#10b981";
    }
  }

  // Load Analytics Data & Render Dashboard
  async loadInitialDashboard() {
    try {
      const [analyticsRes, ordersRes] = await Promise.all([
        fetch("/api/analytics"),
        fetch("/api/orders")
      ]);

      if (analyticsRes.ok && ordersRes.ok) {
        const analyticsJson = await analyticsRes.json();
        const ordersJson = await ordersRes.json();

        this.analyticsData = analyticsJson.analytics;
        this.currentOrders = ordersJson.orders || [];

        this.renderStatsRibbon();
        this.renderCharts();
        this.renderOrdersTable(this.currentOrders);
        this.fetchBedrockInsights(); // Initial insights load
      }
    } catch (err) {
      console.warn("[DASHBOARD] Could not load dashboard data:", err);
    }
  }

  // Render Metric Counters
  renderStatsRibbon() {
    const a = this.analyticsData;
    if (!a) return;

    // Home view stats
    const homeOrders = document.getElementById("home-stat-orders");
    const homeRate = document.getElementById("home-stat-rate");
    const homeHighRisk = document.getElementById("home-stat-highrisk");
    if (homeOrders) homeOrders.textContent = a.totalOrders;
    if (homeRate) homeRate.textContent = `${a.overallReturnRate}%`;
    if (homeHighRisk) homeHighRisk.textContent = a.highRiskCount;

    // Dashboard view stats
    document.getElementById("dash-stat-total").textContent = a.totalOrders;
    document.getElementById("dash-stat-returned").textContent = a.totalReturns;
    document.getElementById("dash-stat-rate").textContent = `${a.overallReturnRate}%`;
    document.getElementById("dash-stat-highrisk").textContent = a.highRiskCount;
  }

  // Render Chart.js Visualizations
  renderCharts() {
    if (!this.analyticsData) return;

    // 1. Category Return Rates
    const catCtx = document.getElementById("categoryChart")?.getContext("2d");
    if (catCtx) {
      const catStats = this.analyticsData.categoryStats || [];
      const labels = catStats.map((c) => c.category);
      const rates = catStats.map((c) => c.returnRate);

      if (this.categoryChart) this.categoryChart.destroy();

      this.categoryChart = new Chart(catCtx, {
        type: "bar",
        data: {
          labels,
          datasets: [
            {
              label: "Return Rate (%)",
              data: rates,
              backgroundColor: ["#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"],
              borderRadius: 6,
              borderSkipped: false
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (ctx) => ` Return Rate: ${ctx.parsed.y}%`
              }
            }
          },
          scales: {
            y: {
              beginAtZero: true,
              max: 50,
              ticks: { callback: (val) => `${val}%` }
            }
          }
        }
      });
    }

    // 2. Return Reasons Distribution
    const reasonsCtx = document.getElementById("reasonsChart")?.getContext("2d");
    if (reasonsCtx) {
      const reasons = this.analyticsData.returnReasons || [];
      const labels = reasons.map((r) => r.reason);
      const counts = reasons.map((r) => r.count);

      if (this.reasonsChart) this.reasonsChart.destroy();

      this.reasonsChart = new Chart(reasonsCtx, {
        type: "bar",
        data: {
          labels,
          datasets: [
            {
              label: "Reported Cases",
              data: counts,
              backgroundColor: "#2563eb",
              borderRadius: 6,
              borderSkipped: false
            }
          ]
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (ctx) => ` Cases: ${ctx.parsed.x}`
              }
            }
          },
          scales: {
            x: {
              beginAtZero: true,
              ticks: { stepSize: 1 }
            }
          }
        }
      });
    }
  }

  // Filter Orders Queue Table
  applyOrderFilters() {
    const cat = document.getElementById("filter-category").value;
    const risk = document.getElementById("filter-risk").value;
    const payment = document.getElementById("filter-payment").value;
    const query = document.getElementById("filter-search").value.trim().toLowerCase();

    let filtered = this.currentOrders;

    if (cat !== "all") {
      filtered = filtered.filter((o) => (o.product_category || "").toLowerCase() === cat.toLowerCase());
    }

    if (risk !== "all") {
      filtered = filtered.filter((o) => (o.risk_level || "").toLowerCase() === risk.toLowerCase());
    }

    if (payment !== "all") {
      filtered = filtered.filter((o) => (o.payment_type || "").toLowerCase() === payment.toLowerCase());
    }

    if (query) {
      filtered = filtered.filter((o) =>
        (o.order_id || "").toLowerCase().includes(query) ||
        (o.product_name || "").toLowerCase().includes(query) ||
        (o.customer_id || "").toLowerCase().includes(query)
      );
    }

    this.renderOrdersTable(filtered);
  }

  // Populate Table Body
  renderOrdersTable(orders) {
    const tbody = document.getElementById("orders-table-body");
    const countEl = document.getElementById("orders-shown-count");
    tbody.innerHTML = "";
    countEl.textContent = orders.length;

    if (orders.length === 0) {
      tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; color: var(--text-muted); padding: 30px;">No matching orders found.</td></tr>`;
      return;
    }

    orders.forEach((o) => {
      const tr = document.createElement("tr");

      let badgeClass = "badge-low";
      if (o.risk_level === "High") badgeClass = "badge-high";
      else if (o.risk_level === "Medium") badgeClass = "badge-med";

      const outcomeClass = o.returned ? "badge-high" : "badge-low";
      const outcomeText = o.returned ? "Returned" : "Kept";

      tr.innerHTML = `
        <td style="font-weight: 700; color: var(--navy);">${o.order_id}</td>
        <td>
          <div style="font-weight: 600;">${o.product_name || "Demo Product"}</div>
          <div style="font-size: 0.75rem; color: var(--text-muted);">${o.customer_id || ""} ${o.size && o.size !== "Not applicable" ? `• Size: ${o.size}` : ""}</div>
        </td>
        <td>${o.product_category}</td>
        <td>$${Number(o.product_price).toFixed(2)}</td>
        <td>
          <span style="font-size: 0.78rem; font-weight: 600; padding: 2px 6px; border-radius: 4px; background: ${o.payment_type === "COD" ? "#fef3c7; color: #92400e" : "#f1f5f9; color: #334155"}">
            ${o.payment_type}
          </span>
        </td>
        <td>${o.delivery_days}d</td>
        <td>★ ${o.product_rating}</td>
        <td style="font-weight: 700;">${o.risk_score}</td>
        <td><span class="badge ${badgeClass}">${o.risk_level}</span></td>
        <td>
          <span class="badge ${outcomeClass}">${outcomeText}</span>
          ${o.return_reason ? `<div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 2px;">${o.return_reason}</div>` : ""}
        </td>
      `;
      tbody.appendChild(tr);
    });
  }

  // Fetch AI Recommendations (Bedrock or Local Fallback)
  async fetchBedrockInsights() {
    const container = document.getElementById("ai-insights-container");
    const statusPill = document.getElementById("bedrock-status-pill");
    const btn = document.getElementById("btn-generate-ai");

    container.innerHTML = `
      <div style="grid-column: 1 / -1; text-align: center; padding: 30px;">
        <div class="spinner"></div>
        <p style="margin-top: 10px; font-size: 0.85rem; color: #cbd5e1;">Synthesizing store return patterns...</p>
      </div>
    `;
    btn.disabled = true;

    try {
      const res = await fetch("/api/bedrock-insights", { method: "POST" });
      const data = await res.json();

      if (data.isLiveAI) {
        statusPill.textContent = "● Amazon Bedrock Active";
        statusPill.style.background = "rgba(16, 185, 129, 0.2)";
        statusPill.style.borderColor = "rgba(16, 185, 129, 0.4)";
      } else {
        statusPill.textContent = "● Local Deterministic Heuristics";
        statusPill.style.background = "rgba(255, 255, 255, 0.15)";
      }

      container.innerHTML = "";
      const insights = data.insights || [];

      insights.forEach((item) => {
        const card = document.createElement("div");
        card.className = "ai-insight-card";
        card.innerHTML = `
          <div class="ai-card-title">${item.area || "Store Strategy"}</div>
          <div class="ai-card-text">${item.recommendation || ""}</div>
          <div class="ai-card-footer">
            <span>Impact: <strong>${item.impact || "High"}</strong></span>
            <span>${item.timeframe || "1-2 weeks"}</span>
          </div>
        `;
        container.appendChild(card);
      });
    } catch (err) {
      container.innerHTML = `
        <div style="grid-column: 1 / -1; color: #fca5a5; font-size: 0.85rem; padding: 10px;">
          Failed to load AI recommendations. Falling back to default suggestions.
        </div>
      `;
    } finally {
      btn.disabled = false;
    }
  }
}

// Instantiate on DOM content loaded
document.addEventListener("DOMContentLoaded", () => {
  window.app = new ReturnWiseApp();
});
