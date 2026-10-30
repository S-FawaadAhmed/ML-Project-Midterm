const API_BASE_URL = "http://127.0.0.1:8000";

let REPORT = null;

const $ = (id) => document.getElementById(id);

const FIELDS = {
    Quantity:        { label: "Quantity", type: "number", value: 2, min: 1, max: 100, step: 1 },
    Unit_Price_INR:  { label: "Unit Price (INR)", type: "number", value: 1500, min: 0.5, step: 0.5 },
    Discount:        { label: "Discount Rate (0.00 - 1.00)", type: "number", value: 0.1, min: 0, max: 1, step: 0.01 },
    Category:        { label: "Category", type: "select", options: "Category" },
    Subcategory:     { label: "Subcategory", type: "select", options: "Subcategory" },
    Region:          { label: "Region", type: "select", options: "Region" },
    Segment:         { label: "Segment", type: "select", options: "Segment" },
    Channel:         { label: "Channel", type: "select", options: "Channel" },
    Payment_Method:  { label: "Payment Method", type: "select", options: "Payment_Method" },
    Order_Month:     { label: "Order Month", type: "number", value: 10, min: 1, max: 12, step: 1 },
    Order_DayOfWeek: { label: "Day of Week (0=Mon)", type: "number", value: 4, min: 0, max: 6, step: 1 }
};

const METRIC_LABELS = {
    r2: "R²", mae: "MAE (₹)", rmse: "RMSE (₹)",
    accuracy: "Accuracy", precision: "Precision", recall: "Recall", f1: "F1 Score", roc_auc: "ROC AUC",
    macro_precision: "Macro Precision", macro_recall: "Macro Recall", macro_f1: "Macro F1"
};

document.addEventListener("DOMContentLoaded", async () => {
    await checkBackendHealth();
    await loadReport();

    $("model-menu").addEventListener("change", onModelChange);
    $("predict-form").addEventListener("submit", handleSubmit);
    $("reset-btn").addEventListener("click", () => {
        renderFields(REPORT.models[$("model-menu").value]);
        $("result").classList.add("hidden");
    });
});

async function checkBackendHealth() {
    const badge = $("api-status-badge");
    try {
        const res = await fetch(`${API_BASE_URL}/health`);
        const data = await res.json();
        if (data.status === "healthy") {
            badge.className = "badge badge-online";
            badge.innerText = "Backend Connected & Models Loaded";
        } else {
            badge.className = "badge badge-loading";
            badge.innerText = "Backend Degraded";
        }
    } catch (err) {
        badge.className = "badge badge-offline";
        badge.innerText = "Backend Offline";
    }
}

async function loadReport() {
    try {
        const res = await fetch(`${API_BASE_URL}/models`);
        if (!res.ok) throw new Error("report missing");
        REPORT = await res.json();
    } catch (err) {
        $("insights").innerHTML = "<p class='card-desc'>Could not load model report. Run train_models.py and restart the API.</p>";
        return;
    }

    const menu = $("model-menu");
    menu.innerHTML = "";
    Object.entries(REPORT.models).forEach(([key, m]) => {
        const opt = document.createElement("option");
        opt.value = key;
        opt.textContent = `Predict ${m.target} | Model: ${m.model}`;
        menu.appendChild(opt);
    });
    onModelChange();
}

function onModelChange() {
    const m = REPORT.models[$("model-menu").value];
    $("model-desc").innerText = m.description;
    $("form-title").innerText = `${m.target} using ${m.model}`;
    $("result").classList.add("hidden");
    renderFields(m);
    renderInsights(m);
}

function renderFields(model) {
    const box = $("fields");
    box.innerHTML = "";
    model.features.forEach((name) => {
        const f = FIELDS[name];
        const group = document.createElement("div");
        group.className = "form-group";

        const label = document.createElement("label");
        label.textContent = f.label;
        group.appendChild(label);

        let input;
        if (f.type === "select") {
            input = document.createElement("select");
            input.addEventListener("change", () => {
                if (name === "Category") fillSubcategory();
            });
        } else {
            input = document.createElement("input");
            input.type = "number";
            input.value = f.value;
            input.required = true;
            if (f.min !== undefined) input.min = f.min;
            if (f.max !== undefined) input.max = f.max;
            if (f.step !== undefined) input.step = f.step;
        }
        input.id = `f-${name}`;
        group.appendChild(input);
        box.appendChild(group);

        if (f.type === "select" && name !== "Subcategory") {
            fillSelect(input, REPORT.options[f.options]);
        }
    });
    fillSubcategory();
}

function fillSelect(el, values) {
    el.innerHTML = "";
    values.forEach((v) => {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        el.appendChild(opt);
    });
}

function fillSubcategory() {
    const sub = $("f-Subcategory");
    if (!sub) return;
    const cat = $("f-Category");
    const category = cat ? cat.value : Object.keys(REPORT.subcategories)[0];
    fillSelect(sub, REPORT.subcategories[category] || []);
}

async function handleSubmit(e) {
    e.preventDefault();
    const key = $("model-menu").value;
    const model = REPORT.models[key];

    const payload = {};
    model.features.forEach((name) => {
        const el = $(`f-${name}`);
        const f = FIELDS[name];
        payload[name] = f.type === "select"
            ? el.value
            : (f.step === 1 ? parseInt(el.value) : parseFloat(el.value));
    });

    try {
        const res = await fetch(`${API_BASE_URL}/predict/${key}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) {
            alert(`API Error: ${typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)}`);
            return;
        }
        showResult(data);
    } catch (err) {
        alert("Failed to connect to the prediction endpoint.");
    }
}

function showResult(data) {
    const out = $("result-output");
    const prob = $("result-prob");
    $("result-title").innerText = `${data.target} (${data.model_name})`;
    out.style.color = "";

    if (data.task === "regression") {
        out.innerText = `₹ ${data.predicted_profit_inr.toLocaleString("en-IN")}`;
        prob.innerText = "Estimated profit for this order.";
    } else if (data.model_key === "return_logistic") {
        const high = data.predicted_class === "Yes";
        out.innerText = high ? "RETURN RISK: HIGH" : "RETURN RISK: LOW";
        out.style.color = high ? "#dc2626" : "#16a34a";
        prob.innerText = `Return Probability: ${(data.probabilities.Yes * 100).toFixed(1)}%`;
    } else {
        out.innerText = data.predicted_class;
        const rows = Object.entries(data.probabilities)
            .sort((a, b) => b[1] - a[1])
            .map(([c, p]) => `${c}: ${(p * 100).toFixed(1)}%`);
        prob.innerHTML = rows.join("<br>");
    }
    $("result").classList.remove("hidden");
}

function renderInsights(m) {
    let html = "<h3>Inputs used</h3><p class='card-desc'>" + m.features.join(", ") + "</p>";
    html += `<h3>Target</h3><p class="card-desc">${m.target_column}</p>`;

    html += "<h3>Metrics</h3><table class='metric-table'>";
    Object.entries(m.metrics).forEach(([k, v]) => {
        html += `<tr><td>${METRIC_LABELS[k] || k}</td><td>${v}</td></tr>`;
    });
    html += "</table>";

    if (m.confusion_matrix) {
        const classes = m.classes;
        html += "<h3>Confusion Matrix</h3><div class='table-scroll'><table class='cm-table'><tr><th>Actual \\ Predicted</th>";
        classes.forEach((c) => { html += `<th>${c}</th>`; });
        html += "</tr>";
        m.confusion_matrix.forEach((row, i) => {
            html += `<tr><th>${classes[i]}</th>`;
            row.forEach((v, j) => {
                html += `<td class="${i === j ? "cm-good" : ""}">${v}</td>`;
            });
            html += "</tr>";
        });
        html += "</table></div>";
    }

    if (m.feature_importance) {
        const max = Math.max(...m.feature_importance.map((f) => f.importance)) || 1;
        html += "<h3>Feature Importance</h3><div class='bars'>";
        m.feature_importance.forEach((f) => {
            const w = (f.importance / max) * 100;
            html += `<div class="bar-row">
                <span class="bar-label">${f.feature}</span>
                <div class="bar-track"><div class="bar-fill" style="width:${w}%"></div></div>
                <span class="bar-value">${f.importance}</span>
            </div>`;
        });
        html += "</div>";
    }

    if (m.rules) {
        html += `<h3>Decision Rules</h3><pre class="rules">${escapeHtml(m.rules)}</pre>`;
    }
    if (m.tree_image) {
        html += `<h3>Decision Tree</h3>
        <a href="${API_BASE_URL}/artifacts/${m.tree_image}" target="_blank">
            <img class="tree-img" src="${API_BASE_URL}/artifacts/${m.tree_image}" alt="Decision tree">
        </a>`;
    }

    $("insights").innerHTML = html;
}

function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
