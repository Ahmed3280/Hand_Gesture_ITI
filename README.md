# Hand Gesture Classification

End-to-end pipeline for real-time hand gesture recognition using MediaPipe landmarks, XGBoost, and a PyTorch MLP — with MLflow experiment tracking, a FastAPI inference endpoint, and a Docker image.

---

## Overview

| Component | Detail |
|---|---|
| **Dataset** | 25,675 samples · 18 gesture classes · 63 landmark features |
| **Features** | 21 hand landmarks × (x, y, z) extracted by MediaPipe, normalized for position/scale invariance |
| **XGBoost** | 98.36% test accuracy |
| **PyTorch MLP** | 98.75% test accuracy |
| **Tracking** | MLflow — params, per-epoch metrics, confusion matrices, model registry |
| **API** | FastAPI `/predict` — accepts 63 floats, returns both model predictions in one JSON |
| **Inference** | Real-time webcam with dual-model overlay HUD |

### Gesture Classes (18)

`call` · `dislike` · `fist` · `four` · `like` · `mute` · `ok` · `one` · `palm` · `peace` · `peace_inverted` · `rock` · `stop` · `stop_inverted` · `three` · `three2` · `two_up` · `two_up_inverted`

---

## Project Structure

```
hand-gesture-clf/
├── data/
│   ├── raw/
│   │   └── hand_landmarks_data.csv   # pre-extracted landmark dataset
│   └── processed/                    # normalized features.csv (optional)
├── src/
│   ├── landmarks.py                  # MediaPipe extraction & normalization
│   ├── train_xgb.py                  # XGBoost training + MLflow logging
│   ├── train_mlp.py                  # PyTorch MLP training + MLflow logging
│   └── inference.py                  # Real-time webcam inference
├── api/
│   ├── main.py                       # FastAPI /predict endpoint
│   └── models/
│       ├── xgb_model.json            # saved XGBoost model
│       ├── mlp_model.pt              # saved MLP weights + metadata
│       └── classes.txt               # ordered class names
├── models/
│   └── hand_landmarker.task          # MediaPipe model (auto-downloaded)
├── mlflow/                           # MLflow tracking store
├── notebooks/
├── Dockerfile
├── .dockerignore
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train both models

```bash
# XGBoost (~30 seconds)
python src/train_xgb.py

# PyTorch MLP (~2 minutes, early stopping)
python src/train_mlp.py
```

Both runs are logged to MLflow. View the comparison dashboard:

```bash
mlflow ui --backend-store-uri file:./mlflow
# open http://127.0.0.1:5000
```

### 3. Real-time webcam inference

```bash
python src/inference.py

# Non-default camera or resolution
python src/inference.py --camera 1 --width 1280 --height 720
```

The HUD overlays both model predictions and confidence bars simultaneously.
Press **Q** or **Esc** to quit.

> The MediaPipe hand landmarker model (~8 MB) is downloaded automatically on first run.

### 4. FastAPI endpoint

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive docs at **http://localhost:8000/docs**

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"landmarks": [0.0, 0.1, -0.05, ...]}'   # 63 floats
```

**Response:**

```json
{
  "xgboost": {
    "label": "peace",
    "confidence": 0.9123,
    "probabilities": { "call": 0.001, "peace": 0.9123, ... }
  },
  "mlp": {
    "label": "peace",
    "confidence": 0.9451,
    "probabilities": { "call": 0.0008, "peace": 0.9451, ... }
  }
}
```

### 5. Docker

```bash
# Build
docker build -t gesture-api .

# Run
docker run -p 8000:8000 gesture-api
```

---

## Pipeline Architecture

```
Raw Image / Webcam Frame
        │
        ▼
  MediaPipe HandLandmarker
  (21 landmarks × x,y,z = 63 raw features)
        │
        ▼
  Normalization
  • Translate: wrist → origin
  • Scale: divide by max landmark distance
        │
        ├──────────────────┬
        ▼                  ▼
   XGBoost            PyTorch MLP
  (200 trees)    63→256→BN→ReLU→DO
                   →128→BN→ReLU→DO
                        →18
        │                  │
        └──────────┬───────┘
                   ▼
          Predicted Gesture
          + Confidence Score
```

---

## Model Details

### Normalization

Each 21-landmark hand is normalized before feeding into either model:

1. **Translate** — subtract wrist (landmark 0), so the wrist is always at the origin
2. **Scale** — divide by the max L2-norm among the 21 translated landmarks, mapping the hand into the unit sphere

This makes predictions invariant to hand position in the frame and distance from the camera.

### XGBoost

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 6 |
| `learning_rate` | 0.1 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `objective` | `multi:softprob` |

### PyTorch MLP

| Layer | Output | Notes |
|---|---|---|
| Input | 63 | normalized landmarks |
| Linear | 256 | |
| BatchNorm + ReLU + Dropout(0.3) | 256 | |
| Linear | 128 | |
| BatchNorm + ReLU + Dropout(0.3) | 128 | |
| Linear | 18 | logits → softmax at inference |

Training: Adam (lr=1e-3, weight_decay=1e-4), ReduceLROnPlateau scheduler, early stopping (patience=10).

### Results

| Model | Test Accuracy | Weighted F1 |
|---|---|---|
| XGBoost | 98.36% | 98.37% |
| PyTorch MLP | **98.75%** | **98.76%** |

---

## Retraining with Custom Hyperparameters

```bash
python src/train_xgb.py --n-estimators 300 --max-depth 8 --lr 0.05

python src/train_mlp.py --epochs 100 --lr 5e-4 --dropout 0.4 --batch-size 128
```

All runs are automatically tracked in MLflow for comparison.

---

## Tech Stack

- **Python 3.11**
- **MediaPipe 0.10** — hand landmark detection (Tasks API)
- **XGBoost** — gradient boosted classifier
- **PyTorch** — MLP with BatchNorm and Dropout
- **Scikit-learn** — train/test split, metrics
- **MLflow** — experiment tracking and model registry
- **OpenCV** — webcam capture and HUD rendering
- **FastAPI + Uvicorn** — REST inference endpoint
- **Docker** — multi-stage containerization
