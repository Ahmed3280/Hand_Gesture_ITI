"""
landmarks.py
------------
MediaPipe Tasks-based hand landmark extraction and normalization.
Compatible with MediaPipe >= 0.10 (Tasks API).

Pipeline:
  1. Detect hand with MediaPipe HandLandmarker (Tasks API)
  2. Extract 21 landmarks × 3 coords (x, y, z) = 63 raw features
  3. Normalize for position/scale invariance:
       - Translate: subtract wrist (landmark 0) → wrist at origin
       - Scale: divide by max L2-norm among the 21 translated vectors
         → hand always fits in the unit sphere, rotation/scale invariant
  4. Flatten to a (63,) float32 array

Usage (dataset):
  df = build_feature_dataset("data/raw", save_to="data/processed/features.csv")

Usage (single image):
  features = extract_landmarks_from_image("path/to/img.jpg")

Usage (webcam):
  with RealtimeHandExtractor() as ex:
      features = ex.extract(frame_bgr)   # (63,) or None
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────

NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3  # 63

# Column names: x0,y0,z0, x1,y1,z1, … x20,y20,z20
FEATURE_COLS = [
    f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")
]

# MediaPipe Tasks model – downloaded automatically on first use
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = Path(__file__).parent.parent / "models" / "hand_landmarker.task"


# ── Model download ───────────────────────────────────────────────────────────

def _ensure_model() -> Path:
    """Download hand_landmarker.task if not already present."""
    if not _MODEL_PATH.exists():
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe hand landmarker model → {_MODEL_PATH} …")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Download complete.")
    return _MODEL_PATH


# ── Normalization ────────────────────────────────────────────────────────────

def _normalize_landmarks(landmarks_xyz: np.ndarray) -> np.ndarray:
    """
    Normalize a (21, 3) landmark array for position and scale invariance.

    Steps
    -----
    1. Translate: subtract wrist (index 0) → wrist at origin.
    2. Scale: divide by the max L2-norm of the 21 translated vectors so the
       farthest landmark has unit distance from the wrist.

    Returns
    -------
    np.ndarray shape (21, 3), dtype float32
    """
    lm = landmarks_xyz.copy().astype(np.float32)
    lm -= lm[0]                               # translate
    max_norm = np.linalg.norm(lm, axis=1).max()
    if max_norm > 1e-6:
        lm /= max_norm                        # scale
    return lm


# ── Low-level extractor (Tasks API) ─────────────────────────────────────────

def _build_landmarker(
    running_mode: mp_vision.RunningMode = mp_vision.RunningMode.IMAGE,
    min_detection_confidence: float = 0.5,
) -> mp_vision.HandLandmarker:
    """Create a HandLandmarker for IMAGE or VIDEO running mode."""
    model_path = _ensure_model()
    base_opts = mp_python.BaseOptions(model_asset_path=str(model_path))
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=running_mode,
        num_hands=1,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_detection_confidence,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def _result_to_features(
    result: mp_vision.HandLandmarkerResult,
) -> Optional[np.ndarray]:
    """Convert a HandLandmarkerResult → (63,) float32 or None."""
    if not result.hand_landmarks:
        return None
    lm_list = result.hand_landmarks[0]               # first hand
    lm_array = np.array(
        [[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32
    )                                                 # (21, 3)
    return _normalize_landmarks(lm_array).flatten()  # (63,)


# ── Public single-frame API ──────────────────────────────────────────────────

def extract_landmarks_from_image(
    image_path: str,
    min_detection_confidence: float = 0.5,
) -> Optional[np.ndarray]:
    """
    Extract and normalise hand landmarks from a single image file.

    Returns (63,) float32 or None if no hand detected.
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None
    return extract_landmarks_from_frame(
        img_bgr, min_detection_confidence=min_detection_confidence
    )


def extract_landmarks_from_frame(
    frame_bgr: np.ndarray,
    min_detection_confidence: float = 0.5,
    landmarker: Optional[mp_vision.HandLandmarker] = None,
    timestamp_ms: int = 0,
) -> Optional[np.ndarray]:
    """
    Extract and normalise hand landmarks from a BGR OpenCV frame.

    Pass a pre-built ``landmarker`` to avoid re-creating it on every call
    (critical for real-time inference).

    Returns (63,) float32 or None if no hand detected.
    """
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    if landmarker is not None:
        # VIDEO mode requires monotonically increasing timestamps
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
    else:
        with _build_landmarker(
            running_mode=mp_vision.RunningMode.IMAGE,
            min_detection_confidence=min_detection_confidence,
        ) as lm:
            result = lm.detect(mp_image)

    return _result_to_features(result)


# ── CSV loader (pre-extracted landmarks) ────────────────────────────────────

# CSV columns are 1-indexed: x1..x21, y1..y21, z1..z21
CSV_FEATURE_COLS = [
    f"{axis}{i}" for i in range(1, NUM_LANDMARKS + 1) for axis in ("x", "y", "z")
]


def load_csv_and_normalize(
    csv_path: str,
    label_col: str = "label",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load a pre-extracted landmark CSV and apply positional normalization.

    The CSV is expected to have columns x1,y1,z1,...,x21,y21,z21 (1-indexed)
    plus a label column. Coordinates may be raw pixel values.

    Returns
    -------
    X : np.ndarray shape (N, 63), float32 — normalized features
    y : np.ndarray shape (N,), int64    — integer-encoded labels
    classes : list[str]                 — class names (index → name)
    """
    df = pd.read_csv(csv_path)

    # Build feature matrix: reshape each row into (21,3), normalize, flatten
    raw = df[CSV_FEATURE_COLS].values.astype(np.float32)   # (N, 63)
    X = np.stack(
        [_normalize_landmarks(row.reshape(NUM_LANDMARKS, 3)).flatten()
         for row in raw]
    )  # (N, 63)

    classes = sorted(df[label_col].unique().tolist())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = df[label_col].map(label_to_idx).values.astype(np.int64)

    return X, y, classes


# ── Dataset builder (for raw image folders) ─────────────────────────────────

def build_feature_dataset(
    data_root: str,
    save_to: Optional[str] = "data/processed/features.csv",
    min_detection_confidence: float = 0.5,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp"),
) -> pd.DataFrame:
    """
    Walk a class-per-folder image dataset, extract landmarks for every image,
    and return (and optionally save) a tidy DataFrame.

    Expected layout
    ---------------
    data_root/
      class_A/   img1.jpg  img2.jpg  …
      class_B/   img1.jpg  …

    Returns
    -------
    pd.DataFrame with columns [x0,y0,z0, …, x20,y20,z20, label]
    """
    root = Path(data_root)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class sub-folders found in {data_root!r}.")

    class_names = [d.name for d in class_dirs]
    print(f"Found {len(class_names)} classes: {class_names}")

    rows, failed = [], 0

    # Re-use a single IMAGE-mode landmarker across all images for efficiency
    with _build_landmarker(
        running_mode=mp_vision.RunningMode.IMAGE,
        min_detection_confidence=min_detection_confidence,
    ) as landmarker:
        for class_dir in class_dirs:
            label = class_dir.name
            image_paths = [
                p for p in class_dir.rglob("*") if p.suffix.lower() in extensions
            ]
            for img_path in tqdm(image_paths, desc=label, leave=False):
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    failed += 1
                    continue

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                result = landmarker.detect(mp_image)
                feats = _result_to_features(result)

                if feats is None:
                    failed += 1
                    continue
                rows.append({**dict(zip(FEATURE_COLS, feats)), "label": label})

    df = pd.DataFrame(rows)
    print(
        f"\nExtraction complete: {len(df)} samples "
        f"({failed} images skipped – no hand detected or unreadable)."
    )

    if save_to:
        out = Path(save_to)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Saved → {out}")

    return df


# ── Real-time extractor ───────────────────────────────────────────────────────

class RealtimeHandExtractor:
    """
    Thin context-manager wrapper around MediaPipe HandLandmarker (VIDEO mode)
    for low-overhead webcam / video-stream inference.

    Example
    -------
    with RealtimeHandExtractor() as ex:
        while cap.isOpened():
            ret, frame = cap.read()
            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            features = ex.extract(frame, ts_ms)   # (63,) or None
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
    ):
        self._conf = min_detection_confidence
        self._landmarker: Optional[mp_vision.HandLandmarker] = None

    def __enter__(self) -> "RealtimeHandExtractor":
        self._landmarker = _build_landmarker(
            running_mode=mp_vision.RunningMode.VIDEO,
            min_detection_confidence=self._conf,
        )
        return self

    def __exit__(self, *_):
        if self._landmarker:
            self._landmarker.close()
        self._landmarker = None

    def extract(
        self, frame_bgr: np.ndarray, timestamp_ms: int = 0
    ) -> Optional[np.ndarray]:
        """Return (63,) float32 feature vector or None."""
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        return _result_to_features(result)

    def extract_with_results(
        self, frame_bgr: np.ndarray, timestamp_ms: int = 0
    ) -> tuple[Optional[np.ndarray], mp_vision.HandLandmarkerResult]:
        """
        Return (features, raw_result) so callers can draw landmarks.
        """
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        return _result_to_features(result), result


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract hand landmarks from dataset")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--output", default="data/processed/features.csv")
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()

    build_feature_dataset(
        data_root=args.data_root,
        save_to=args.output,
        min_detection_confidence=args.confidence,
    )
