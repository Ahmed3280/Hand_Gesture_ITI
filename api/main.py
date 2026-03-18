"""
api/main.py
-----------
FastAPI service that serves both trained models via a single /predict endpoint.

What this does:
  - Loads XGBoost and MLP models once at startup (lifespan event)
  - POST /predict  accepts a flat list of 63 landmark floats
                   returns both model predictions in one JSON response
  - GET  /health   liveness check

Run locally:
  uvicorn api.main:app --reload --port 8000

Example request:
  curl -X POST http://localhost:8000/predict \
       -H "Content-Type: application/json" \
       -d '{"landmarks": [0.1, -0.2, 0.0, ...]}'   # 63 floats
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
XGB_MODEL_PATH = ROOT / "api" / "models" / "xgb_model.json"
MLP_MODEL_PATH = ROOT / "api" / "models" / "mlp_model.pt"
CLASSES_PATH   = ROOT / "api" / "models" / "classes.txt"

FEATURE_DIM = 63


# ── MLP architecture (must match train_mlp.py) ────────────────────────────────

class HandGestureMLP(nn.Module):
    def __init__(self, n_classes: int = 18, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Global model store ────────────────────────────────────────────────────────

_models: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup, release on shutdown."""
    classes = CLASSES_PATH.read_text().strip().splitlines()
    n_classes = len(classes)

    # XGBoost
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(XGB_MODEL_PATH))

    # MLP
    ckpt = torch.load(str(MLP_MODEL_PATH), map_location="cpu", weights_only=False)
    mlp_model = HandGestureMLP(
        n_classes=ckpt.get("n_classes", n_classes),
        dropout=ckpt.get("dropout", 0.3),
    )
    mlp_model.load_state_dict(ckpt["state_dict"])
    mlp_model.eval()

    _models["xgb"] = xgb_model
    _models["mlp"] = mlp_model
    _models["classes"] = classes

    print(f"[startup] Loaded XGBoost + MLP | {n_classes} classes")
    yield
    _models.clear()
    print("[shutdown] Models released.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Hand Gesture Classifier API",
    description="Predict hand gesture from 63 normalized landmark features using XGBoost and MLP.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    landmarks: list[float]

    @field_validator("landmarks")
    @classmethod
    def check_length(cls, v: list[float]) -> list[float]:
        if len(v) != FEATURE_DIM:
            raise ValueError(
                f"landmarks must have exactly {FEATURE_DIM} elements, got {len(v)}"
            )
        return v


class ModelPrediction(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    xgboost: ModelPrediction
    mlp: ModelPrediction


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "models_loaded": str(bool(_models))}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if not _models:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    classes: list[str] = _models["classes"]
    x = np.array(request.landmarks, dtype=np.float32).reshape(1, -1)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_proba = _models["xgb"].predict_proba(x)[0]
    xgb_idx   = int(xgb_proba.argmax())
    xgb_pred  = ModelPrediction(
        label=classes[xgb_idx],
        confidence=round(float(xgb_proba[xgb_idx]), 4),
        probabilities={c: round(float(p), 4) for c, p in zip(classes, xgb_proba)},
    )

    # ── MLP ───────────────────────────────────────────────────────────────────
    tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        mlp_proba = torch.softmax(_models["mlp"](tensor), dim=-1).squeeze().numpy()
    mlp_idx  = int(mlp_proba.argmax())
    mlp_pred = ModelPrediction(
        label=classes[mlp_idx],
        confidence=round(float(mlp_proba[mlp_idx]), 4),
        probabilities={c: round(float(p), 4) for c, p in zip(classes, mlp_proba)},
    )

    return PredictResponse(xgboost=xgb_pred, mlp=mlp_pred)
