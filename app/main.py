import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Security, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

# App setup
app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn from RFM behavioural features.",
    version="1.0.0",
)

# API key auth
API_KEY = os.environ.get("API_KEY", "dev-key-aig-churn")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return key

# Load artifacts at startup
BASE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE, "models")

model = joblib.load(os.path.join(MODELS_DIR, "churn_model.joblib"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib"))

# Request / response 
class CustomerFeatures(BaseModel):
    recency: float = Field(..., description="Days since last purchase before cutoff", example=30)
    frequency: float = Field(..., description="Number of unique invoices", example=5)
    monetary: float = Field(..., description="Total revenue (GBP)", example=1200.0)
    avg_order_value: float = Field(..., description="Mean revenue per transaction", example=45.0)
    total_items: float = Field(..., description="Total quantity purchased", example=300)
    unique_products: float = Field(..., description="Number of distinct products bought", example=20)

class PredictionResponse(BaseModel):
    churned: int = Field(..., description="1 = churned, 0 = active")
    churn_probability: float = Field(..., description="Probability of churn (0.0 to 1.0)")
    model: str = Field(..., description="Model used for prediction")

# HTML UI
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Churn Prediction</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #fafaf8;
      color: #1a1a1a;
      min-height: 100vh;
      display: flex;
      align-items: flex-start;
      justify-content: center;
      padding: 48px 16px;
    }

    .container {
      width: 100%;
      max-width: 520px;
    }

    .header {
      margin-bottom: 36px;
    }

    .header h1 {
      font-size: 22px;
      font-weight: 600;
      letter-spacing: -0.3px;
      color: #111;
    }

    .header p {
      margin-top: 6px;
      font-size: 14px;
      color: #777;
      line-height: 1.5;
    }

    .card {
      background: #fff;
      border: 1px solid #e8e8e4;
      border-radius: 10px;
      padding: 28px;
    }

    .fields {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px 20px;
    }

    .field label {
      display: block;
      font-size: 12px;
      font-weight: 500;
      color: #555;
      margin-bottom: 6px;
      letter-spacing: 0.2px;
      text-transform: uppercase;
    }

    .field input {
      width: 100%;
      padding: 9px 12px;
      font-size: 14px;
      font-family: inherit;
      border: 1px solid #ddd;
      border-radius: 6px;
      background: #fafaf8;
      color: #111;
      transition: border-color 0.15s;
      outline: none;
    }

    .field input:focus {
      border-color: #aaa;
      background: #fff;
    }

    .field .hint {
      font-size: 11px;
      color: #aaa;
      margin-top: 4px;
    }

    .divider {
      border: none;
      border-top: 1px solid #f0f0ec;
      margin: 24px 0;
    }

    .api-row {
      display: flex;
      gap: 10px;
      align-items: flex-end;
    }

    .api-row .field {
      flex: 1;
    }

    button {
      width: 100%;
      padding: 11px;
      font-size: 14px;
      font-weight: 500;
      font-family: inherit;
      background: #111;
      color: #fff;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      margin-top: 20px;
      transition: background 0.15s;
      letter-spacing: 0.1px;
    }

    button:hover { background: #333; }
    button:disabled { background: #bbb; cursor: not-allowed; }

    .result {
      margin-top: 20px;
      padding: 18px 20px;
      border-radius: 8px;
      display: none;
    }

    .result.active {
      background: #f0faf4;
      border: 1px solid #c6e8d1;
    }

    .result.churned {
      background: #fdf4f4;
      border: 1px solid #f0c8c8;
    }

    .result-label {
      font-size: 13px;
      font-weight: 600;
      letter-spacing: 0.2px;
      text-transform: uppercase;
      margin-bottom: 6px;
    }

    .result.active .result-label { color: #2d7a4f; }
    .result.churned .result-label { color: #b84040; }

    .result-main {
      font-size: 20px;
      font-weight: 600;
      color: #111;
      margin-bottom: 4px;
    }

    .result-sub {
      font-size: 13px;
      color: #666;
    }

    .error {
      margin-top: 16px;
      padding: 12px 16px;
      background: #fdf4f4;
      border: 1px solid #f0c8c8;
      border-radius: 6px;
      font-size: 13px;
      color: #b84040;
      display: none;
    }

    .footer {
      margin-top: 20px;
      font-size: 12px;
      color: #bbb;
      text-align: center;
    }

    .footer a {
      color: #aaa;
      text-decoration: none;
    }

    .footer a:hover { color: #777; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Customer Churn Predictor</h1>
      <p>Enter a customer's RFM behavioural features to predict their churn risk.</p>
    </div>

    <div class="card">
      <div class="fields">
        <div class="field">
          <label>Recency</label>
          <input type="number" id="recency" placeholder="e.g. 30" min="0" />
          <div class="hint">Days since last purchase</div>
        </div>
        <div class="field">
          <label>Frequency</label>
          <input type="number" id="frequency" placeholder="e.g. 5" min="0" />
          <div class="hint">Number of orders</div>
        </div>
        <div class="field">
          <label>Monetary</label>
          <input type="number" id="monetary" placeholder="e.g. 1200" min="0" step="0.01" />
          <div class="hint">Total spend (GBP)</div>
        </div>
        <div class="field">
          <label>Avg Order Value</label>
          <input type="number" id="avg_order_value" placeholder="e.g. 45" min="0" step="0.01" />
          <div class="hint">Mean spend per order</div>
        </div>
        <div class="field">
          <label>Total Items</label>
          <input type="number" id="total_items" placeholder="e.g. 300" min="0" />
          <div class="hint">Total quantity purchased</div>
        </div>
        <div class="field">
          <label>Unique Products</label>
          <input type="number" id="unique_products" placeholder="e.g. 20" min="0" />
          <div class="hint">Distinct products bought</div>
        </div>
      </div>

      <hr class="divider" />

      <div class="field">
        <label>API Key</label>
        <input type="password" id="api_key" placeholder="Enter your API key" />
      </div>

      <button id="btn" onclick="predict()">Predict</button>

      <div class="result" id="result">
        <div class="result-label" id="result-label"></div>
        <div class="result-main" id="result-main"></div>
        <div class="result-sub" id="result-sub"></div>
      </div>

      <div class="error" id="error"></div>
    </div>

    <div class="footer">
      <a href="/docs">API docs</a> &nbsp;·&nbsp; AIG Spring 2026
    </div>
  </div>

  <script>
    async function predict() {
      const btn = document.getElementById('btn');
      const resultEl = document.getElementById('result');
      const errorEl = document.getElementById('error');

      // hide previous results
      resultEl.style.display = 'none';
      errorEl.style.display = 'none';
      resultEl.className = 'result';

      // read values
      const fields = ['recency', 'frequency', 'monetary', 'avg_order_value', 'total_items', 'unique_products'];
      const body = {};
      for (const f of fields) {
        const val = document.getElementById(f).value;
        if (val === '') {
          errorEl.textContent = 'Please fill in all fields.';
          errorEl.style.display = 'block';
          return;
        }
        body[f] = parseFloat(val);
      }

      const apiKey = document.getElementById('api_key').value;
      if (!apiKey) {
        errorEl.textContent = 'Please enter your API key.';
        errorEl.style.display = 'block';
        return;
      }

      btn.disabled = true;
      btn.textContent = 'Predicting...';

      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-API-Key': apiKey,
          },
          body: JSON.stringify(body),
        });

        const data = await res.json();

        if (!res.ok) {
          errorEl.textContent = data.detail || 'Something went wrong.';
          errorEl.style.display = 'block';
          return;
        }

        const prob = (data.churn_probability * 100).toFixed(1);
        const isChurned = data.churned === 1;

        resultEl.classList.add(isChurned ? 'churned' : 'active');
        document.getElementById('result-label').textContent = isChurned ? 'At Risk' : 'Active';
        document.getElementById('result-main').textContent = isChurned
          ? `${prob}% churn probability`
          : `${prob}% churn probability`;
        document.getElementById('result-sub').textContent = isChurned
          ? 'This customer is likely to churn.'
          : 'This customer is likely to remain active.';

        resultEl.style.display = 'block';

      } catch (e) {
        errorEl.textContent = 'Request failed. Check your connection and try again.';
        errorEl.style.display = 'block';
      } finally {
        btn.disabled = false;
        btn.textContent = 'Predict';
      }
    }

    // allow Enter key to submit
    document.addEventListener('keydown', e => {
      if (e.key === 'Enter') predict();
    });
  </script>
</body>
</html>
"""

# Endpoints
@app.get("/", response_class=HTMLResponse)
def ui():
    """Simple prediction UI"""
    return HTML


@app.get("/health")
def health():
    """Health check [200 if the API is up and model is loaded]"""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures, _: str = Depends(verify_api_key)):
    """
    Predict customer churn from RFM features.
    Returns churn prediction (0 or 1) and churn probability.
    Requires X-API-Key header.
    """
    values = [[
        features.recency,
        features.frequency,
        features.monetary,
        features.avg_order_value,
        features.total_items,
        features.unique_products,
    ]]

    scaled = scaler.transform(values)
    prediction = int(model.predict(scaled)[0])
    probability = float(model.predict_proba(scaled)[0][1])

    return PredictionResponse(
        churned=prediction,
        churn_probability=round(probability, 4),
        model=type(model).__name__,
    )