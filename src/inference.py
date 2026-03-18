"""
inference.py
------------
Real-time webcam hand-gesture inference using both trained models.

What this script does:
  1. Opens the webcam with OpenCV
  2. Extracts hand landmarks each frame via MediaPipe (RealtimeHandExtractor)
  3. Feeds the (63,) feature vector into XGBoost and MLP simultaneously
  4. Draws the hand skeleton on the frame
  5. Overlays an HUD with both predictions, confidence bars, and FPS

Run:
  python src/inference.py
  python src/inference.py --camera 1          # non-default camera index
  python src/inference.py --width 1280 --height 720
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from landmarks import RealtimeHandExtractor, NUM_LANDMARKS

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
XGB_MODEL_PATH  = ROOT / "api" / "models" / "xgb_model.json"
MLP_MODEL_PATH  = ROOT / "api" / "models" / "mlp_model.pt"
CLASSES_PATH    = ROOT / "api" / "models" / "classes.txt"

# ── MediaPipe hand skeleton connections (21 landmarks) ────────────────────────
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17),
]


# ── MLP model definition (must match train_mlp.py) ────────────────────────────

class HandGestureMLP(torch.nn.Module):
    def __init__(self, n_classes: int = 18, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(63, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Model loader ──────────────────────────────────────────────────────────────

def load_models() -> tuple[xgb.XGBClassifier, HandGestureMLP, list[str]]:
    """Load both models and the class list from api/models/."""
    # Classes
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

    print(f"Loaded XGBoost : {XGB_MODEL_PATH.name}")
    print(f"Loaded MLP     : {MLP_MODEL_PATH.name}")
    print(f"Classes ({n_classes}): {classes}")
    return xgb_model, mlp_model, classes


# ── Inference helpers ─────────────────────────────────────────────────────────

def _predict_xgb(
    model: xgb.XGBClassifier,
    features: np.ndarray,
    classes: list[str],
) -> tuple[str, float]:
    """Return (class_name, confidence) for XGBoost."""
    proba = model.predict_proba(features.reshape(1, -1))[0]
    idx = int(proba.argmax())
    return classes[idx], float(proba[idx])


def _predict_mlp(
    model: HandGestureMLP,
    features: np.ndarray,
    classes: list[str],
) -> tuple[str, float]:
    """Return (class_name, confidence) for MLP."""
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, 63)
    with torch.no_grad():
        proba = torch.softmax(model(x), dim=-1).squeeze().numpy()
    idx = int(proba.argmax())
    return classes[idx], float(proba[idx])


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_skeleton(
    frame: np.ndarray,
    raw_landmarks,   # list of NormalizedLandmark from MediaPipe result
    h: int,
    w: int,
) -> None:
    """Draw hand skeleton on frame using raw (unnormalized) pixel coordinates."""
    pts = [
        (int(lm.x * w), int(lm.y * h)) for lm in raw_landmarks
    ]
    # Connections
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 220, 255), 2, cv2.LINE_AA)
    # Landmark dots
    for i, (x, y) in enumerate(pts):
        color = (0, 0, 255) if i == 0 else (255, 255, 255)  # red wrist
        cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)


def _draw_hud(
    frame: np.ndarray,
    xgb_label: str,
    xgb_conf: float,
    mlp_label: str,
    mlp_conf: float,
    fps: float,
    hand_detected: bool,
) -> None:
    """Overlay prediction panel and FPS on the frame."""
    h, w = frame.shape[:2]

    # ── Semi-transparent panel (bottom strip) ─────────────────────────────────
    panel_h = 130
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if not hand_detected:
        cv2.putText(
            frame, "No hand detected",
            (20, h - panel_h // 2 + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 255), 2, cv2.LINE_AA,
        )
    else:
        # ── Row 1: XGBoost ────────────────────────────────────────────────────
        _draw_model_row(
            frame,
            label="XGBoost",
            pred=xgb_label,
            conf=xgb_conf,
            y=h - panel_h + 28,
            bar_color=(0, 200, 100),
            w=w,
        )
        # ── Row 2: MLP ────────────────────────────────────────────────────────
        _draw_model_row(
            frame,
            label="MLP    ",
            pred=mlp_label,
            conf=mlp_conf,
            y=h - panel_h + 80,
            bar_color=(0, 150, 255),
            w=w,
        )

    # ── FPS (top-right) ───────────────────────────────────────────────────────
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (w - 140, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA,
    )

    # ── Quit hint (top-left) ──────────────────────────────────────────────────
    cv2.putText(
        frame, "Press Q to quit",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA,
    )


def _draw_model_row(
    frame: np.ndarray,
    label: str,
    pred: str,
    conf: float,
    y: int,
    bar_color: tuple,
    w: int,
) -> None:
    """Draw one model's prediction row: [Model] PredClass  [====    ] 82%"""
    # Model tag
    cv2.putText(
        frame, label,
        (15, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA,
    )
    # Prediction class
    cv2.putText(
        frame, pred,
        (130, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
        (255, 255, 255) if conf >= 0.6 else (80, 80, 255),
        2, cv2.LINE_AA,
    )
    # Confidence bar
    bar_x, bar_y = 320, y - 16
    bar_w, bar_h = 260, 20
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill = int(bar_w * conf)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)
    cv2.putText(
        frame, f"{conf * 100:.1f}%",
        (bar_x + bar_w + 10, y - 1),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 1, cv2.LINE_AA,
    )


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
    min_detection_confidence: float = 0.6,
) -> None:
    xgb_model, mlp_model, classes = load_models()

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}.")

    print("\nWebcam opened. Press Q to quit.\n")

    # Cache last prediction so HUD doesn't flicker when hand briefly lost
    last_xgb = ("---", 0.0)
    last_mlp = ("---", 0.0)

    fps_timer = time.perf_counter()
    fps = 0.0
    frame_count = 0

    with RealtimeHandExtractor(
        min_detection_confidence=min_detection_confidence
    ) as extractor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)   # mirror for natural interaction
            h, w = frame.shape[:2]
            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            # ── Extract landmarks ─────────────────────────────────────────────
            features, results = extractor.extract_with_results(frame, ts_ms)
            hand_detected = features is not None

            if hand_detected:
                # Draw skeleton using the raw (unnormalized) landmark coords
                raw_lm = results.hand_landmarks[0]
                _draw_skeleton(frame, raw_lm, h, w)

                # Run both models
                last_xgb = _predict_xgb(xgb_model, features, classes)
                last_mlp = _predict_mlp(mlp_model, features, classes)

            # ── HUD overlay ───────────────────────────────────────────────────
            _draw_hud(
                frame,
                xgb_label=last_xgb[0], xgb_conf=last_xgb[1],
                mlp_label=last_mlp[0], mlp_conf=last_mlp[1],
                fps=fps,
                hand_detected=hand_detected,
            )

            cv2.imshow("Hand Gesture Recognition", frame)

            # ── FPS ───────────────────────────────────────────────────────────
            frame_count += 1
            if frame_count % 10 == 0:
                now = time.perf_counter()
                fps = 10.0 / (now - fps_timer)
                fps_timer = now

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time hand gesture inference")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--confidence", type=float, default=0.6,
                        help="MediaPipe min detection confidence")
    args = parser.parse_args()

    run(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        min_detection_confidence=args.confidence,
    )
