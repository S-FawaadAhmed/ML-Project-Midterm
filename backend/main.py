import os
import json

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

try:
    from .schemas import OrderPredictionInput, PredictionResponse, HealthResponse
except ImportError:
    from schemas import OrderPredictionInput, PredictionResponse, HealthResponse

app = FastAPI(
    title="Sales Prediction API",
    description="Six models with their own prediction targets",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
ARTIFACT_DIR = os.path.join(MODEL_DIR, "artifacts")
REPORT_PATH = os.path.join(ARTIFACT_DIR, "report.json")

MODEL_KEYS = [
    "profit_linear",
    "return_logistic",
    "channel_tree",
    "segment_forest",
    "payment_svm",
    "category_bayes",
]

MODELS = {}
REPORT = {}

app.mount("/artifacts", StaticFiles(directory=ARTIFACT_DIR, check_dir=False), name="artifacts")


@app.on_event("startup")
def load_models():
    global REPORT
    for key in MODEL_KEYS:
        path = os.path.join(MODEL_DIR, f"{key}.joblib")
        if os.path.exists(path):
            MODELS[key] = joblib.load(path)
            print(f"Loaded {key}.")
        else:
            print(f"Warning: {path} not found. Run train_models.py.")

    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            REPORT = json.load(f)


@app.get("/health", response_model=HealthResponse)
def health_check():
    loaded = {key: key in MODELS for key in MODEL_KEYS}
    return {
        "status": "healthy" if all(loaded.values()) else "degraded",
        "models_loaded": loaded,
    }


@app.get("/models")
def list_models():
    if not REPORT:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Report not found. Run train_models.py.",
        )
    return REPORT


@app.post("/predict/{model_key}", response_model=PredictionResponse)
def predict(model_key: str, payload: OrderPredictionInput):
    if model_key not in MODEL_KEYS:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_key}")
    model = MODELS.get(model_key)
    info = REPORT.get("models", {}).get(model_key)
    if model is None or info is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model {model_key} is not loaded.",
        )

    data = payload.model_dump()
    missing = [f for f in info["features"] if data.get(f) is None]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required fields: {', '.join(missing)}")

    try:
        input_df = pd.DataFrame([{f: data[f] for f in info["features"]}])
        base = {"model_key": model_key, "model_name": info["model"], "target": info["target"]}

        if info["task"] == "regression":
            value = model.predict(input_df)[0]
            return {**base, "task": "regression", "predicted_profit_inr": round(float(value), 2)}

        label = model.predict(input_df)[0]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
        else:
            d = model.decision_function(input_df)[0]
            e = np.exp(d - d.max())
            probs = e / e.sum()
        classes = [str(c) for c in model.named_steps["model"].classes_]
        label_text = "Yes" if str(label) == "1" else "No" if str(label) == "0" else str(label)
        return {
            **base,
            "task": "classification",
            "predicted_class": label_text,
            "probabilities": {
                ("Yes" if c == "1" else "No" if c == "0" else c): round(float(p), 4)
                for c, p in zip(classes, probs)
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)