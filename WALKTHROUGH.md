# Hand Gesture Classifier — Complete Project Walkthrough

> A deep-dive guide to every file, every design decision, and how the full pipeline works from raw data to live deployment.

---

## Quick Answer: Are the Two Models Combined?

**No — they run in parallel, independently.** Both XGBoost and MLP receive the exact same 63-number input, make their own predictions, and their results are displayed side-by-side. There is no ensemble, no averaging, no voting. You see both predictions simultaneously so you can compare them. This is an intentional design choice for a demo/research project — it lets you observe when the two models agree or disagree.

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FULL PIPELINE                            │
│                                                                 │
│  [Webcam / Image]                                               │
│       │                                                         │
│       ▼                                                         │
│  MediaPipe HandLandmarker                                       │
│  Detects hand → 21 keypoints (x,y,z each) = 63 raw numbers     │
│       │                                                         │
│       ▼                                                         │
│  Normalization                                                  │
│  • Translate: wrist moves to (0,0,0)                            │
│  • Scale: divide by max distance → fits in unit sphere          │
│  Result: 63 float32 values, position/scale invariant            │
│       │                                                         │
│       ├────────────────────┬────────────────────┐               │
│       ▼                    ▼                    │               │
│  XGBoost Model         MLP Model (PyTorch)      │               │
│  200 trees             63→256→128→18            │               │
│  98.36% accuracy       98.75% accuracy          │               │
│       │                    │                    │               │
│       ▼                    ▼                    │               │
│  18 probabilities      18 probabilities         │               │
│  → pick argmax         → softmax → argmax       │               │
│       │                    │                    │               │
│       └────────────────────┘                    │               │
│                    │                            │               │
│              Show BOTH results                  │               │
│              side-by-side                       │               │
└─────────────────────────────────────────────────────────────────┘
```

---

## File-by-File Walkthrough

### Reading Order (recommended)

1. `src/landmarks.py` — the foundation everything else depends on
2. `src/train_xgb.py` — how the XGBoost model is trained
3. `src/train_mlp.py` — how the MLP is trained
4. `src/inference.py` — local real-time webcam demo
5. `api/main.py` — the REST API wrapper
6. `app.py` — the Streamlit cloud deployment

---

### 1. `src/landmarks.py` — The Foundation

**What it does:** Everything related to extracting numbers from a hand image.

This is the most important file. If you understand this one, you understand the core of the project.

#### The problem it solves

A neural network or XGBoost model cannot look at a raw camera image. It needs numbers. MediaPipe solves this by detecting the hand and returning 21 keypoints — specific anatomical landmarks like fingertip, knuckle, wrist.

```
MediaPipe's 21 landmarks:
  0 = Wrist
  1-4 = Thumb (base to tip)
  5-8 = Index finger
  9-12 = Middle finger
  13-16 = Ring finger
  17-20 = Pinky
```

Each landmark has x, y, z coordinates. That's 21 × 3 = **63 numbers total**.

#### Why normalization matters

If you feed raw pixel coordinates directly into the model, it would learn "the gesture is in the top-left corner" instead of learning "this is a peace sign." The model would fail when you move your hand.

The normalization (lines 73–92) solves this:

```python
def _normalize_landmarks(landmarks_xyz):
    lm = landmarks_xyz.copy()
    lm -= lm[0]          # Step 1: subtract wrist → wrist becomes (0,0,0)
    max_norm = np.linalg.norm(lm, axis=1).max()
    lm /= max_norm        # Step 2: divide by max distance → fits in unit sphere
    return lm
```

**Step 1 (translation):** Landmark 0 is the wrist. Subtracting it from all 21 points makes the wrist the origin. Now all coordinates are *relative to the wrist*, not the screen.

**Step 2 (scaling):** Find the farthest landmark from the wrist (usually a fingertip). Divide all coordinates by that distance. Now the hand always "fits" in a 1-unit sphere regardless of how close or far your hand is from the camera.

After normalization: the same gesture produces nearly the same 63 numbers whether your hand is in the top-left or bottom-right, near or far.

#### The MediaPipe version trap

This project uses MediaPipe **0.10+**, which switched to the **Tasks API**. The old API (`mp.solutions.hands`) was removed. This is why you see:

```python
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
landmarker = mp_vision.HandLandmarker.create_from_options(opts)
```

instead of the old `mp.solutions.hands.Hands()`. If you find old tutorials online, they will NOT work with this version.

#### The `.task` model file

MediaPipe needs its own trained model (`hand_landmarker.task`, ~8MB) to detect hands. The code auto-downloads it from Google on first use. For cloud deployment, it must be pre-committed to the repo because the Streamlit server has no internet access at runtime.

#### VIDEO mode vs IMAGE mode

There are two running modes:

- **IMAGE mode** — for processing individual static images. Each call is independent. Used in `build_feature_dataset()` and `extract_landmarks_from_image()`.
- **VIDEO mode** — for webcam streams. MediaPipe uses temporal tracking between frames (smoother, faster, more accurate). **Requires timestamps to be monotonically increasing** — if you pass timestamp 100 then 90, it crashes. Used in `RealtimeHandExtractor`.

#### Key exports you'll use

```python
# For loading the training CSV:
X, y, classes = load_csv_and_normalize("data/raw/hand_landmarks_data.csv")
# X: (25675, 63) float32 — normalized features
# y: (25675,) int64 — integer labels (0-17)
# classes: ['call', 'dislike', ..., 'two_up_inverted']  ← 18 names

# For webcam:
with RealtimeHandExtractor() as ex:
    features = ex.extract(frame_bgr, timestamp_ms)  # (63,) or None

# features is None when no hand is detected in the frame
```

---

### 2. `src/train_xgb.py` — Training the XGBoost Model

**What it does:** Trains the first classifier and saves it to `api/models/xgb_model.json`.

#### What is XGBoost?

XGBoost (Extreme Gradient Boosting) builds an ensemble of **decision trees** sequentially. Each new tree corrects the errors of the previous ones. It works extremely well on tabular data (which our 63 normalized coordinates are). It's fast to train and often achieves top performance without requiring normalization of inputs.

#### The training process (line by line)

```python
# 1. Load data
X, y, classes = load_csv_and_normalize("data/raw/hand_landmarks_data.csv")
# 25,675 samples, 63 features, 18 classes

# 2. Split: 80% train, 20% test (stratified = equal class proportions in both)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Define model
model = xgb.XGBClassifier(
    n_estimators=200,        # build 200 trees
    max_depth=6,             # each tree up to 6 levels deep
    learning_rate=0.1,       # how much each tree contributes
    subsample=0.8,           # use 80% of rows per tree (prevents overfitting)
    colsample_bytree=0.8,    # use 80% of features per tree
    objective="multi:softprob",  # outputs probability for each of 18 classes
    num_class=18,
)

# 4. Train
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

# 5. Evaluate
y_pred = model.predict(X_test)
# → accuracy: 98.36%, weighted F1: 98.37%
```

#### What is `multi:softprob`?

The `objective="multi:softprob"` setting means XGBoost outputs a probability distribution across all 18 classes (not just the winning class). After prediction, `model.predict_proba(x)` returns 18 numbers that sum to 1.0. This is needed to show confidence percentages in the HUD.

#### MLflow logging

Every training run is tracked in MLflow (a local experiment tracker):
- Hyperparameters (n_estimators, max_depth, etc.)
- Test accuracy and F1 score
- A confusion matrix image artifact
- The model itself (registered as `xgboost-hand-gesture`)

The tracking data lives in `mlflow/` folder locally. View it with:
```bash
mlflow ui --backend-store-uri file:./mlflow
# Opens at http://127.0.0.1:5000
```

#### Saved outputs

- `api/models/xgb_model.json` — the trained model (XGBoost's native format)
- `api/models/classes.txt` — the 18 class names, one per line

---

### 3. `src/train_mlp.py` — Training the MLP Model

**What it does:** Trains the second classifier (a small neural network) and saves it to `api/models/mlp_model.pt`.

#### What is an MLP?

A Multi-Layer Perceptron is a basic feed-forward neural network. Our architecture:

```
Input: 63 numbers (normalized landmarks)
    │
    ▼
Linear(63 → 256)   ← fully connected layer, 63×256 = 16,128 weights
BatchNorm1d(256)   ← normalizes activations, stabilizes training
ReLU               ← non-linearity: max(0, x)
Dropout(0.3)       ← randomly zeroes 30% of neurons during training
    │
    ▼
Linear(256 → 128)
BatchNorm1d(128)
ReLU
Dropout(0.3)
    │
    ▼
Linear(128 → 18)   ← one output per class (raw "logits", not probabilities)
```

#### Why no softmax in `forward()`?

```python
def forward(self, x):
    return self.net(x)   # raw logits, NO softmax here
```

This is a PyTorch convention. `CrossEntropyLoss` (used during training) internally applies softmax. If you put softmax in `forward()`, you'd be applying it twice. At **inference time**, softmax is applied explicitly:

```python
proba = torch.softmax(model(x), dim=-1)  # in predict_mlp()
```

#### The 3-way split

Unlike XGBoost (which only uses train/test), the MLP uses a **3-way split**:
- **80% train** — gradient descent updates happen here
- **10% validation** — used to monitor overfitting and trigger early stopping
- **10% test** — final evaluation, never touched during training

```python
# First split off 10% test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1)
# Then split remaining 90% into 80% train + 10% val
X_train, X_val = train_test_split(X_temp, y_temp, test_size=0.1/0.9)
```

#### Early stopping

Training stops automatically if the validation loss doesn't improve for 10 consecutive epochs (`patience=10`). When it stops, the model weights are **restored to the best checkpoint** (not the final weights):

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    wait = 0
else:
    wait += 1
    if wait >= patience:
        break

model.load_state_dict(best_state)  # restore best
```

This model stopped at epoch 52 (out of 60 max).

#### Learning rate scheduler

`ReduceLROnPlateau` halves the learning rate when validation loss plateaus (no improvement for 5 epochs). This gives the optimizer finer control late in training.

#### Saved checkpoint format

```python
torch.save({
    "state_dict": model.state_dict(),  # all weights
    "classes": classes,                # ['call', 'dislike', ...]
    "n_classes": 18,
    "dropout": 0.3,                    # needed to recreate architecture
}, "api/models/mlp_model.pt")
```

Storing `classes`, `n_classes`, and `dropout` inside the checkpoint means the inference code never needs a separate config file — it can reconstruct the exact model from the `.pt` file alone.

---

### 4. `src/inference.py` — Local Webcam Demo

**What it does:** Opens your webcam and runs real-time gesture recognition. This is the local-only script (requires a physical camera).

#### The main loop

```
For each frame:
  1. cap.read()                    → grab frame from webcam
  2. cv2.flip(frame, 1)            → mirror it (natural interaction)
  3. extractor.extract_with_results(frame, ts_ms)
                                   → (63 features, raw landmark result)
  4. If hand detected:
       _draw_skeleton(frame, ...)  → draw cyan lines + white dots on frame
       _predict_xgb(...)           → (label, confidence)
       _predict_mlp(...)           → (label, confidence)
  5. _draw_hud(frame, ...)         → draw bottom panel with both predictions
  6. cv2.imshow(...)               → display the annotated frame
  7. FPS calculation every 10 frames
```

#### The skeleton drawing

```python
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),    # index
    ...
    (5, 9), (9, 13), (13, 17),         # palm across knuckles
]
```

This hardcodes the anatomical connections between MediaPipe's 21 landmarks. The drawing uses the **raw (unnormalized)** landmark coordinates from the MediaPipe result — because we need actual pixel positions on screen. The normalized coordinates (used for prediction) have the wrist at (0,0,0) so they can't be used for drawing.

#### HUD design

```
[ Press Q to quit ]                         [ FPS: 28.3 ]
    [hand skeleton drawn on top of video]
┌──────────────────────────────────────────────────────────┐
│ XGBoost  peace     [████████░░]  81.2%                   │
│ MLP      peace     [█████████░]  89.4%                   │
└──────────────────────────────────────────────────────────┘
```

The confidence bar is drawn by `_draw_model_row()`:
- Dark background rectangle (full width)
- Colored fill rectangle (proportional to confidence)
- If confidence < 60%, the prediction text turns blue-ish (low confidence indicator)

#### Last prediction caching

```python
last_xgb = ("---", 0.0)
last_mlp = ("---", 0.0)

if hand_detected:
    last_xgb = _predict_xgb(...)
    last_mlp = _predict_mlp(...)

_draw_hud(frame, last_xgb[0], last_xgb[1], last_mlp[0], last_mlp[1], ...)
```

When the hand briefly disappears (between frames), the HUD shows the **last known prediction** instead of flickering to "---". This makes the display much smoother.

---

### 5. `api/main.py` — The REST API

**What it does:** Wraps both models in a FastAPI web server. You send 63 numbers via HTTP POST, get back both predictions as JSON.

#### The lifespan pattern

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code here runs ONCE at startup
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(XGB_MODEL_PATH))
    # ... load MLP ...
    _models["xgb"] = xgb_model
    _models["mlp"] = mlp_model
    _models["classes"] = classes
    yield
    # Code here runs at shutdown
    _models.clear()
```

This is the modern FastAPI pattern for startup/shutdown events. Models are loaded **once** when the server starts, not on every request. They're stored in a module-level dict `_models` so they're accessible from any endpoint.

#### The `/predict` endpoint

```
POST /predict
Content-Type: application/json

{
  "landmarks": [0.0, 0.0, 0.0, -0.12, 0.05, 0.02, ...]  ← exactly 63 floats
}
```

The `@field_validator` on `PredictRequest` immediately rejects requests with the wrong number of values — returns HTTP 422 before any model inference happens.

Response:
```json
{
  "xgboost": {
    "label": "peace",
    "confidence": 0.9123,
    "probabilities": {"call": 0.001, "peace": 0.9123, "fist": 0.003, ...}
  },
  "mlp": {
    "label": "peace",
    "confidence": 0.9451,
    "probabilities": {"call": 0.0008, "peace": 0.9451, ...}
  }
}
```

The API runs locally or in Docker. It does **not** run on Streamlit Cloud (the Streamlit app loads models directly without going through the API).

---

### 6. `app.py` — The Streamlit Deployment

**What it does:** The cloud-deployable version. Uses `streamlit-webrtc` to stream webcam video from the browser to the server, process it with MediaPipe + both models, and stream the annotated video back.

#### Why this is architecturally different from `inference.py`

`inference.py` is single-threaded: one thread reads frames, processes them, and displays them. Streamlit is multi-threaded: the WebRTC video thread and the Streamlit UI thread run independently.

#### The video processor factory pattern

```python
def make_video_processor(xgb_model, mlp_model, classes):
    class GestureVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self._landmarker = _build_landmarker(...)
            self._frame_idx = 0
            self._lock = threading.Lock()
            self.result = {...}   # shared state

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            # This runs in the WebRTC thread for EVERY video frame
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            # ... MediaPipe + models ...
            # Write result under lock (thread-safe)
            with self._lock:
                self.result = {...}
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    return GestureVideoProcessor
```

The outer function (`make_video_processor`) is called once to "bake in" the already-loaded models. The inner class's `recv()` method is called by `streamlit-webrtc` for every incoming video frame. The `threading.Lock()` protects `self.result` from race conditions between the video thread (writing) and the Streamlit UI thread (reading).

#### `@st.cache_resource`

```python
@st.cache_resource(show_spinner="Loading models — please wait...")
def load_models():
    ...
```

This decorator is Streamlit's mechanism to run a function **once** and cache the result across all user sessions and re-runs. Without it, models would be reloaded from disk every time the user interacts with a widget. With it, models are loaded once at startup and held in memory.

#### WebRTC and STUN

WebRTC establishes a peer-to-peer-like connection between the browser and server. STUN (Session Traversal Utilities for NAT) servers help the two sides find each other through NAT/firewalls. We use Google's free STUN servers:

```python
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]
})
```

#### Snapshot fallback

If WebRTC fails (strict corporate firewall), the app falls back to `st.camera_input` — a built-in Streamlit component that lets users take a single photo. It's less smooth (no real-time video) but always works.

---

## The Dataset

`data/raw/hand_landmarks_data.csv`

- **25,675 rows** — each is one hand gesture sample
- **64 columns** — `x1,y1,z1, x2,y2,z2, ..., x21,y21,z21, label`
- Column naming is **1-indexed** (`x1`..`x21`) in the CSV, but the code uses **0-indexed** internally after loading
- Labels are strings: `call`, `dislike`, `fist`, `four`, `like`, `mute`, `ok`, `one`, `palm`, `peace`, `peace_inverted`, `rock`, `stop`, `stop_inverted`, `three`, `three2`, `two_up`, `two_up_inverted`
- Class sizes range from ~945 to ~1,653 samples — not perfectly balanced, which is why **weighted F1** is used alongside accuracy

---

## Why Two Models?

| | XGBoost | MLP (PyTorch) |
|---|---|---|
| Type | Gradient-boosted trees | Neural network |
| Input normalization | Not required (trees split on thresholds) | Not required (BatchNorm handles scale) |
| Training time | ~10 seconds | ~2 minutes (52 epochs) |
| Test accuracy | 98.36% | 98.75% |
| Inference speed | Very fast (~0.1ms) | Fast (~1ms) |
| Interpretability | Can inspect feature importance | Black box |
| Why used here | Strong baseline, fast, robust | Slightly higher accuracy, learns non-linear patterns differently |

Both models produce nearly identical results on this dataset. The project runs them both to demonstrate that different ML approaches can achieve similar performance on well-structured feature data.

---

## Data Flow Summary

```
data/raw/hand_landmarks_data.csv
        │
        ▼
load_csv_and_normalize()   [landmarks.py]
        │  reads CSV columns x1..x21, y1..y21, z1..z21
        │  reshapes each row → (21,3)
        │  normalizes: translate + scale
        │  flattens → (63,)
        ▼
X: (25675, 63) float32
y: (25675,) int64
        │
        ├──────────────────────┐
        ▼                      ▼
train_xgb.py              train_mlp.py
80/20 split               80/10/10 split
XGBClassifier             HandGestureMLP
.fit(X_train, y_train)    Adam + EarlyStopping
        │                      │
        ▼                      ▼
api/models/               api/models/
xgb_model.json            mlp_model.pt
classes.txt
        │
        ├─────────────────────────────────┐
        ▼                                 ▼
inference.py (local)              app.py (cloud)
OpenCV webcam                     streamlit-webrtc
RealtimeHandExtractor             GestureVideoProcessor.recv()
  → (63,) features                  → (63,) features
        │                                 │
        ├───────────┬─────────────┬───────┴────────────┐
        ▼           ▼             ▼                    ▼
  XGBoost      MLP           XGBoost              MLP
  predict      predict       predict              predict
        │           │             │                    │
        └───────────┘             └────────────────────┘
             │                              │
        HUD overlay                   HUD overlay
        on cv2 window                 on WebRTC frame
```

---

## Normalization: The Most Important Detail

This is the single design decision that makes the whole system work. Without it, the models would overfit to hand position and size in the training data.

**What the numbers look like before normalization** (raw pixel coords from a 1280×720 camera):
```
Wrist:  x=640, y=600, z=-0.05
Index tip: x=680, y=400, z=-0.12
...
```

**After normalization:**
```
Wrist:  x=0.0, y=0.0, z=0.0       ← always zero
Index tip: x=0.17, y=-0.83, z=-0.07  ← relative to wrist, scaled
...
```

The same gesture from a different person, different camera resolution, different hand position → nearly the same 63 numbers → same prediction.

**Important:** The CSV uses raw pixel coordinates. `load_csv_and_normalize()` applies the normalization during loading. The same normalization function is used at inference time (in `landmarks.py`). **They must always match** — if you change normalization logic for training, you must change it for inference too, or the models will break silently (wrong predictions with high confidence).

---

## The Dockerfile

A multi-stage build to minimize image size:

**Stage 1 (builder):** Install all Python packages (including build tools, compilers needed for some packages) into `/opt/venv`.

**Stage 2 (runtime):** Start fresh with a clean `python:3.11-slim` image. Copy only `/opt/venv` (the built packages) and the source code. No compilers, no build artifacts. Result: a smaller, cleaner production image.

The API is served by `uvicorn api.main:app --host 0.0.0.0 --port 8000`.

---

## Common Questions

**Q: Can I retrain with new gestures?**
Add new rows to the CSV with a new label string. Re-run `train_xgb.py` and `train_mlp.py`. The models will automatically pick up the new class. Update `classes.txt` is handled automatically.

**Q: Why does the model file path use `ROOT / "api" / "models"`?**
The trained models are stored *inside the api folder* so the Docker image can find them at the correct relative path. It's a convention to co-locate model files with the serving code.

**Q: What happens when `features` is `None`?**
MediaPipe returns `None` when it doesn't detect a hand. The code always checks `if features is not None` before calling the models. The HUD shows "No hand detected."

**Q: Why is `classes.txt` needed if it's also inside `mlp_model.pt`?**
`classes.txt` is shared by both models. XGBoost's `.json` format doesn't store metadata like class names, so `classes.txt` is the canonical source. The MLP checkpoint also stores classes as a backup.

**Q: What is `ts_ms` (timestamp in milliseconds)?**
MediaPipe's VIDEO mode requires timestamps to know the order of frames and apply temporal smoothing. In `inference.py` it comes from `cap.get(cv2.CAP_PROP_POS_MSEC)`. In `app.py` it's approximated as `frame_idx * 33` (33ms ≈ 30fps). It must be strictly increasing.

---

## File Dependencies Map

```
landmarks.py          ← no project dependencies (only mediapipe, cv2, numpy)
    │
    ├── train_xgb.py  ← imports load_csv_and_normalize
    ├── train_mlp.py  ← imports load_csv_and_normalize
    ├── inference.py  ← imports RealtimeHandExtractor, NUM_LANDMARKS
    └── app.py        ← imports _build_landmarker, _result_to_features,
                                 extract_landmarks_from_frame

api/main.py           ← loads model files, does NOT import landmarks.py
                         (landmarks extraction happens before the API call)
```

`landmarks.py` is the only shared utility. Everything else is independent. You can change `train_xgb.py` without affecting `inference.py` as long as the model file format stays the same.
