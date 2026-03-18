"""
app.py
------
Streamlit Community Cloud deployment of the hand gesture classifier.

Real-time mode  : streamlit-webrtc captures webcam video, MediaPipe extracts
                  hand landmarks server-side, and both XGBoost + MLP models
                  predict the gesture.  All annotations are drawn directly on
                  the video frame so predictions update at full video framerate.

Snapshot fallback: st.camera_input for networks where WebRTC is blocked.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import torch
import xgboost as xgb
from mediapipe.tasks.python import vision as mp_vision
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

# ── src/ on the import path ───────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from landmarks import (  # noqa: E402
    _build_landmarker,
    _result_to_features,
    extract_landmarks_from_frame,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
XGB_MODEL_PATH = ROOT / "api" / "models" / "xgb_model.json"
MLP_MODEL_PATH = ROOT / "api" / "models" / "mlp_model.pt"
CLASSES_PATH   = ROOT / "api" / "models" / "classes.txt"

# ── Hand skeleton connections (MediaPipe 21-landmark topology) ─────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm knuckles
]

# ── MLP architecture (must mirror train_mlp.py exactly) ───────────────────────
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


# ── Model loading (cached across re-runs) ─────────────────────────────────────
@st.cache_resource(show_spinner="Loading models — please wait...")
def load_models() -> tuple[xgb.XGBClassifier, HandGestureMLP, list[str]]:
    classes = CLASSES_PATH.read_text().strip().splitlines()
    n_classes = len(classes)

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(XGB_MODEL_PATH))

    ckpt = torch.load(str(MLP_MODEL_PATH), map_location="cpu", weights_only=False)
    mlp = HandGestureMLP(
        n_classes=ckpt.get("n_classes", n_classes),
        dropout=ckpt.get("dropout", 0.3),
    )
    mlp.load_state_dict(ckpt["state_dict"])
    mlp.eval()

    return xgb_model, mlp, classes


# ── Prediction helpers ────────────────────────────────────────────────────────
def predict_xgb(
    model: xgb.XGBClassifier, features: np.ndarray, classes: list[str]
) -> tuple[str, float, np.ndarray]:
    proba = model.predict_proba(features.reshape(1, -1))[0]
    idx = int(proba.argmax())
    return classes[idx], float(proba[idx]), proba


def predict_mlp(
    model: HandGestureMLP, features: np.ndarray, classes: list[str]
) -> tuple[str, float, np.ndarray]:
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        proba = torch.softmax(model(x), dim=-1).squeeze().numpy()
    idx = int(proba.argmax())
    return classes[idx], float(proba[idx]), proba


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_skeleton(frame: np.ndarray, raw_landmarks, h: int, w: int) -> None:
    """Draw hand skeleton using raw (unnormalized) MediaPipe pixel coordinates."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in raw_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 220, 255), 2, cv2.LINE_AA)
    for i, (px, py) in enumerate(pts):
        color = (0, 0, 255) if i == 0 else (255, 255, 255)  # red wrist
        cv2.circle(frame, (px, py), 4, color, -1, cv2.LINE_AA)


def _draw_row(
    frame: np.ndarray,
    model_name: str,
    pred: str,
    conf: float,
    y: int,
    bar_color: tuple,
    w: int,
) -> None:
    cv2.putText(frame, model_name, (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, pred, (155, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255) if conf >= 0.6 else (80, 80, 255), 2, cv2.LINE_AA)
    bx, by, bw, bh = 340, y - 16, 240, 20
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
    fill = int(bw * conf)
    cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), bar_color, -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (120, 120, 120), 1)
    cv2.putText(frame, f"{conf * 100:.1f}%", (bx + bw + 10, y - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    xgb_label: str, xgb_conf: float,
    mlp_label: str, mlp_conf: float,
    hand_detected: bool,
) -> None:
    h, w = frame.shape[:2]
    panel_h = 130
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if not hand_detected:
        cv2.putText(frame, "No hand detected",
                    (20, h - panel_h // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 255), 2, cv2.LINE_AA)
        return

    _draw_row(frame, "XGBoost", xgb_label, xgb_conf, h - panel_h + 32,
              (0, 200, 100), w)
    _draw_row(frame, "MLP    ", mlp_label, mlp_conf, h - panel_h + 84,
              (0, 150, 255), w)


# ── Video processor factory ───────────────────────────────────────────────────
def make_video_processor(
    xgb_model: xgb.XGBClassifier,
    mlp_model: HandGestureMLP,
    classes: list[str],
):
    """
    Returns a VideoProcessorBase subclass that closes over pre-loaded models.
    Using a factory avoids re-loading models on every WebRTC restart.
    """

    class GestureVideoProcessor(VideoProcessorBase):
        def __init__(self) -> None:
            # Build a VIDEO-mode HandLandmarker (requires monotonic timestamps)
            self._landmarker = _build_landmarker(
                running_mode=mp_vision.RunningMode.VIDEO
            )
            self._frame_idx = 0
            self._lock = threading.Lock()
            # Shared result — written by video thread, read by Streamlit thread
            self.result: dict = {
                "xgb_label": "---", "xgb_conf": 0.0,
                "mlp_label": "---", "mlp_conf": 0.0,
                "hand_detected": False,
            }

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)  # mirror for natural interaction
            h, w = img.shape[:2]

            # Monotonically increasing timestamp required by VIDEO-mode MediaPipe
            self._frame_idx += 1
            ts_ms = self._frame_idx * 33  # ~30 fps approximation

            # Extract hand landmarks
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            mp_result = self._landmarker.detect_for_video(mp_image, ts_ms)
            features = _result_to_features(mp_result)

            hand_detected = features is not None
            xgb_label, xgb_conf = "---", 0.0
            mlp_label, mlp_conf = "---", 0.0

            if hand_detected:
                # Draw skeleton using raw (unnormalized) landmark coords
                draw_skeleton(img, mp_result.hand_landmarks[0], h, w)
                xgb_label, xgb_conf, _ = predict_xgb(xgb_model, features, classes)
                mlp_label, mlp_conf, _ = predict_mlp(mlp_model, features, classes)

            draw_hud(img, xgb_label, xgb_conf, mlp_label, mlp_conf, hand_detected)

            with self._lock:
                self.result = {
                    "xgb_label": xgb_label, "xgb_conf": xgb_conf,
                    "mlp_label": mlp_label, "mlp_conf": mlp_conf,
                    "hand_detected": hand_detected,
                }

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    return GestureVideoProcessor


# ── WebRTC ICE / STUN configuration ──────────────────────────────────────────
# Uses Google's free public STUN servers.  No TURN server is required for most
# networks; add a Twilio TURN entry here if users behind strict NAT report
# connection failures.
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]
})


# ── Streamlit UI ──────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Hand Gesture Recognition",
        layout="wide",
    )

    st.title("Hand Gesture Recognition")
    st.markdown(
        "Real-time classification using **XGBoost** (98.36% test accuracy) "
        "and **PyTorch MLP** (98.75% test accuracy) trained on 18 gesture classes.  \n"
        "Landmarks are extracted by **MediaPipe** (21 keypoints → 63 normalised features)."
    )

    # Load both models once (cached)
    xgb_model, mlp_model, classes = load_models()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("How to use")
        st.markdown(
            "1. Click **START** below\n"
            "2. Allow browser camera permission\n"
            "3. Hold your hand up — predictions appear in the video overlay\n"
            "4. Click **STOP** to end the session\n\n"
            "> If the stream does not connect, scroll down to use **Snapshot Mode**."
        )

        st.markdown("---")
        st.subheader("18 Gesture Classes")
        # Two-column layout for the class list
        half = len(classes) // 2
        col_a, col_b = st.columns(2)
        with col_a:
            for c in sorted(classes)[:half]:
                st.text(c)
        with col_b:
            for c in sorted(classes)[half:]:
                st.text(c)

        st.markdown("---")
        st.caption(
            "Dataset: 25,675 samples  \n"
            "Normalization: wrist → origin, scale by max landmark distance  \n"
            "Models trained with Python 3.11 + scikit-learn / PyTorch"
        )

    # ── Live WebRTC stream ────────────────────────────────────────────────────
    webrtc_ctx = webrtc_streamer(
        key="gesture",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=make_video_processor(xgb_model, mlp_model, classes),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # ── Snapshot fallback ─────────────────────────────────────────────────────
    if not webrtc_ctx.state.playing:
        st.markdown("---")
        st.subheader("Snapshot Mode")
        st.markdown(
            "WebRTC may be blocked by some corporate or restricted networks.  \n"
            "Use your device camera below to take a single photo and get a prediction."
        )
        img_file = st.camera_input("Point your camera at a hand gesture and capture")

        if img_file is not None:
            file_bytes = np.frombuffer(img_file.getvalue(), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with st.spinner("Detecting hand landmarks..."):
                features = extract_landmarks_from_frame(img_bgr)

            if features is None:
                st.warning(
                    "No hand detected. Try again with your full hand visible, "
                    "good lighting, and a plain background."
                )
            else:
                xgb_label, xgb_conf, xgb_proba = predict_xgb(
                    xgb_model, features, classes
                )
                mlp_label, mlp_conf, mlp_proba = predict_mlp(
                    mlp_model, features, classes
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("XGBoost")
                    st.metric("Prediction", xgb_label, f"{xgb_conf * 100:.1f}% confidence")
                    st.markdown("**Top-5 probabilities**")
                    top5 = sorted(zip(classes, xgb_proba), key=lambda x: -x[1])[:5]
                    for cls, p in top5:
                        st.progress(float(p), text=f"{cls}: {p * 100:.1f}%")

                with col2:
                    st.subheader("MLP (PyTorch)")
                    st.metric("Prediction", mlp_label, f"{mlp_conf * 100:.1f}% confidence")
                    st.markdown("**Top-5 probabilities**")
                    top5 = sorted(zip(classes, mlp_proba), key=lambda x: -x[1])[:5]
                    for cls, p in top5:
                        st.progress(float(p), text=f"{cls}: {p * 100:.1f}%")


if __name__ == "__main__":
    main()
