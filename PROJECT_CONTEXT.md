# Hand Gesture Classification — Full Project Context

> Paste this entire file into a new conversation to resume with full context.

---

## What This Project Is

End-to-end hand gesture classification pipeline. Takes a pre-extracted landmark CSV,
trains two models, tracks experiments in MLflow, serves predictions via FastAPI,
runs real-time webcam inference, and packages everything in Docker.

---

## Environment

- **OS:** Windows 11
- **Python:** `C:\Program Files\Python311\python.exe` (3.11.9) ← always use this
- **Conda base (Python 3.13) is the default `python` command — do NOT use it**
- **Working directory:** `D:\akrooom\Projects\Hand_Gesture_ITI`
- **Run commands as:** `& "C:\Program Files\Python311\python.exe" src/train_xgb.py`
  or add Python 3.11 to PATH first:
  ```powershell
  $env:PATH = "C:\Program Files\Python311;C:\Program Files\Python311\Scripts;" + $env:PATH
  ```

---

## Installed Packages (Python 3.11)

```
mediapipe==0.10.32
opencv-contrib-python==4.13.0.92
xgboost==3.2.0
torch==2.9.1+cpu
mlflow==3.10.1
fastapi==0.128.0
uvicorn
scikit-learn==1.8.0
pandas==2.3.3
numpy
tqdm
matplotlib
```

---

## Dataset

- **File:** `data/raw/hand_landmarks_data.csv`
- **Shape:** 25,675 rows × 64 columns
- **Features:** `x1,y1,z1, x2,y2,z2, ..., x21,y21,z21` (1-indexed, raw pixel coords)
- **Label column:** `label`
- **Classes (18):**
  `call, dislike, fist, four, like, mute, ok, one, palm, peace,
   peace_inverted, rock, stop, stop_inverted, three, three2, two_up, two_up_inverted`
- **Class balance:** roughly 945–1653 samples per class, no nulls

---

## Project Structure

```
D:\akrooom\Projects\Hand_Gesture_ITI\
├── data/
│   ├── raw/hand_landmarks_data.csv       # source dataset
│   └── processed/                        # optional normalized CSV output
├── src/
│   ├── landmarks.py                      # DONE
│   ├── train_xgb.py                      # DONE
│   ├── train_mlp.py                      # DONE
│   └── inference.py                      # DONE
├── api/
│   ├── main.py                           # DONE
│   └── models/
│       ├── xgb_model.json                # DONE — trained XGBoost
│       ├── mlp_model.pt                  # DONE — trained MLP weights
│       └── classes.txt                   # DONE — 18 class names
├── models/
│   └── hand_landmarker.task              # auto-downloaded on first run (~8MB)
├── mlflow/                               # MLflow tracking store (file-based)
├── notebooks/
├── Dockerfile                            # DONE
├── .dockerignore                         # DONE
├── requirements.txt                      # DONE
└── README.md                             # DONE
```

---

## Source Files — What Each Does

### `src/landmarks.py`
MediaPipe Tasks API (v0.10+) hand landmark extraction and normalization.

Key exports:
```python
load_csv_and_normalize(csv_path)
# -> X: (N,63) float32, y: (N,) int64, classes: list[str]
# Reads CSV with columns x1..x21,y1..y21,z1..z21,label
# Applies: translate wrist->origin, scale by max landmark distance

RealtimeHandExtractor(min_detection_confidence=0.5)
# Context manager for VIDEO-mode webcam use
# .extract(frame_bgr, timestamp_ms) -> (63,) float32 or None
# .extract_with_results(frame_bgr, timestamp_ms) -> (features, raw_result)

extract_landmarks_from_image(path) -> (63,) or None
extract_landmarks_from_frame(bgr, ...) -> (63,) or None
```

**Important:** MediaPipe 0.10+ uses Tasks API (`mp.tasks.vision.HandLandmarker`).
The old `mp.solutions.hands` does NOT exist in this version.
Model file `models/hand_landmarker.task` is auto-downloaded from Google on first call.

---

### `src/train_xgb.py`
Trains XGBoost on normalized landmarks, logs to MLflow.

What it does:
1. `load_csv_and_normalize("data/raw/hand_landmarks_data.csv")`
2. Stratified 80/20 train-test split
3. Fits `XGBClassifier(n_estimators=200, max_depth=6, lr=0.1, objective="multi:softprob")`
4. Evaluates: accuracy, weighted F1, classification report, confusion matrix
5. Logs params + metrics + confusion matrix PNG to MLflow
6. Saves model to `api/models/xgb_model.json` and `api/models/classes.txt`

Run: `python src/train_xgb.py [--n-estimators 200] [--max-depth 6] [--lr 0.1]`

**Results:** Test accuracy **98.36%**, Weighted F1 **98.37%**

---

### `src/train_mlp.py`
Trains PyTorch MLP on normalized landmarks, logs to MLflow.

Architecture:
```
Input(63) -> Linear(256) -> BatchNorm -> ReLU -> Dropout(0.3)
          -> Linear(128) -> BatchNorm -> ReLU -> Dropout(0.3)
          -> Linear(18)
```
CrossEntropyLoss during training; softmax applied at inference.

What it does:
1. `load_csv_and_normalize(...)` — same normalization as XGBoost
2. Stratified 80/10/10 train-val-test split
3. Trains with Adam (lr=1e-3, weight_decay=1e-4) + ReduceLROnPlateau
4. Early stopping (patience=10 on val loss)
5. Logs per-epoch train/val loss+acc to MLflow
6. Saves checkpoint to `api/models/mlp_model.pt`:
   `{"state_dict": ..., "classes": [...], "n_classes": 18, "dropout": 0.3}`

Run: `python src/train_mlp.py [--epochs 60] [--lr 1e-3] [--dropout 0.3]`

**Results:** Test accuracy **98.75%**, Weighted F1 **98.76%**, stopped at epoch 52

---

### `src/inference.py`
Real-time webcam inference with dual-model HUD.

What it does:
1. Loads `xgb_model.json` + `mlp_model.pt` + `classes.txt`
2. Opens webcam with OpenCV
3. Each frame: extracts landmarks via `RealtimeHandExtractor`
4. Feeds (63,) feature vector into both models simultaneously
5. Draws hand skeleton (cyan lines, white dots, red wrist)
6. Overlays HUD panel at bottom: both predictions + confidence bars + FPS

HUD layout:
```
[ Press Q to quit ]                     [ FPS: 28.3 ]
    [hand skeleton drawn on frame]
┌─────────────────────────────────────────────────────┐
│ XGBoost  peace     [████████░░]  81.2%              │
│ MLP      peace     [█████████░]  89.4%              │
└─────────────────────────────────────────────────────┘
```

Run: `python src/inference.py [--camera 0] [--width 1280] [--height 720]`

**Known issue:** Must use Python 3.11, not conda default (Python 3.13 lacks packages).

---

### `api/main.py`
FastAPI REST endpoint serving both models.

Routes:
- `GET /health` → `{"status": "ok", "models_loaded": "True"}`
- `POST /predict` → accepts `{"landmarks": [63 floats]}`, returns both predictions

Response schema:
```json
{
  "xgboost": { "label": "peace", "confidence": 0.9123,
               "probabilities": {"call": 0.001, "peace": 0.9123, ...} },
  "mlp":     { "label": "peace", "confidence": 0.9451,
               "probabilities": {"call": 0.0008, "peace": 0.9451, ...} }
}
```

Models loaded once at startup via FastAPI `lifespan` event.
Run: `uvicorn api.main:app --reload --port 8000`
Swagger UI: `http://localhost:8000/docs`

---

### `Dockerfile`
Multi-stage build:
- **Stage 1 (builder):** `python:3.11-slim` + build tools → installs all deps into `/opt/venv`
- **Stage 2 (runtime):** `python:3.11-slim` + runtime libs only → copies venv + source code
- Serves FastAPI via Uvicorn on port 8000
- **Note:** `api/models/` (trained models) and `models/` (MediaPipe `.task` file)
  must exist before building — they're copied into the image

---

## MLflow

Tracking URI: `file:D:\akrooom\Projects\Hand_Gesture_ITI\mlflow`
Experiment: `hand-gesture-clf`
Runs:
- `xgboost` run — params, test_accuracy=0.9836, confusion matrix PNG, registered model v2
- `pytorch-mlp` run — per-epoch metrics, test_accuracy=0.9875, confusion matrix PNG, registered model v1

View UI:
```bash
mlflow ui --backend-store-uri file:D:\akrooom\Projects\Hand_Gesture_ITI\mlflow
# http://127.0.0.1:5000
```

---

## All Working Commands

```powershell
# Set correct Python for the session
$env:PATH = "C:\Program Files\Python311;C:\Program Files\Python311\Scripts;" + $env:PATH

# Train
python src/train_xgb.py
python src/train_mlp.py

# Webcam inference
python src/inference.py

# API (local)
uvicorn api.main:app --reload --port 8000

# MLflow UI
mlflow ui --backend-store-uri file:./mlflow

# Docker
docker build -t gesture-api .
docker run -p 8000:8000 gesture-api
```

---

## What Is NOT Done Yet (Next Steps)

### 1. Deployment (priority)
The FastAPI container needs to be deployed to the cloud.
The webcam script runs locally only — it cannot be deployed (needs physical camera).
What CAN be deployed: the `/predict` API + optionally a web demo UI.

**Recommended deployment path:**

#### Option A — Render (easiest, free tier)
1. Push code + models to GitHub
2. Connect repo to [render.com](https://render.com)
3. Set build command: `docker build` or point to Dockerfile
4. Render auto-deploys on every push
5. Free tier gives a public HTTPS URL

#### Option B — Railway
1. `railway init` → `railway up`
2. Detects Dockerfile automatically
3. Free $5/month credit, public URL in seconds

#### Option C — Fly.io
```bash
fly launch          # detects Dockerfile, asks region
fly deploy          # builds + deploys
fly open            # opens public URL
```

#### Option D — AWS / GCP / Azure (production-grade)
- AWS: ECR (push image) → ECS Fargate (run container) → ALB (HTTPS)
- GCP: Artifact Registry → Cloud Run (serverless, scales to zero)
- Azure: ACR → Container Apps

**Cloud Run (GCP) is the simplest cloud option:**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/gesture-api
gcloud run deploy gesture-api --image gcr.io/PROJECT_ID/gesture-api --platform managed
```

### 2. Web Demo UI (optional but impressive)
A browser-based demo where users point their webcam at their hand and see predictions.

**Stack:** Streamlit or Gradio frontend → calls the deployed `/predict` API

Streamlit approach:
- Use `streamlit-webrtc` to access webcam in browser
- Run MediaPipe in Python on each frame
- Send landmarks to `/predict` and display results

Gradio approach (fastest to build):
- Gradio has built-in webcam input component
- 20-30 lines of code for a working demo
- Can be hosted for free on HuggingFace Spaces

### 3. Notebook (optional)
An exploratory notebook in `notebooks/` showing:
- Class distribution plots
- Sample landmark visualizations
- Training curves from MLflow
- Confusion matrix side-by-side comparison

### 4. CI/CD (optional)
GitHub Actions workflow:
- On push: run tests → build Docker image → push to registry → deploy

---

## Deployment Decision Guide

| Goal | Best Option |
|---|---|
| Quickest public URL | Railway or Render |
| Free + permanent | Render (free tier, sleeps after inactivity) |
| Serverless, scales to zero | GCP Cloud Run |
| Full production | AWS ECS Fargate |
| Demo with webcam in browser | HuggingFace Spaces + Gradio |
| Portfolio project | HuggingFace Spaces (most visible) |

---

## Key Technical Decisions Made

| Decision | Reason |
|---|---|
| MediaPipe Tasks API (not solutions) | `mp.solutions` removed in 0.10+ |
| Normalize per-sample not global | Invariance to position/scale at inference time |
| CSV columns are 1-indexed (x1..x21) | Source dataset convention, handled in `CSV_FEATURE_COLS` |
| MLP uses CrossEntropyLoss, no softmax in forward | PyTorch convention; softmax applied at inference only |
| MLflow file store (not SQLite) | Simpler local setup; can migrate to SQLite for production |
| `mlp_model.pt` saves full checkpoint dict | Stores classes + dropout so inference needs no config file |
| `/tmp` replaced with `tempfile.gettempdir()` | Windows has no /tmp |
| Arrow `->` not `→` in print() | Windows cp1252 terminal encoding |
