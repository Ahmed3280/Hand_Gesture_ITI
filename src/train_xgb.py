"""
train_xgb.py
------------
Train an XGBoost classifier on normalized hand-landmark features and log
everything to MLflow.

What this script does:
  1. Load data/raw/hand_landmarks_data.csv and apply positional normalization
  2. Stratified 80/20 train-test split
  3. Fit XGBoostClassifier with tunable hyperparameters
  4. Evaluate on test set: accuracy, weighted F1, full classification report
  5. Plot and save confusion matrix as a PNG artifact
  6. Log params, metrics, confusion-matrix image, and the model to MLflow
  7. Save the fitted model to api/models/xgb_model.json for inference

Run:
  python src/train_xgb.py
  python src/train_xgb.py --n-estimators 300 --max-depth 8 --lr 0.05
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import tempfile
import mlflow.xgboost
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

# Make src importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))
from landmarks import load_csv_and_normalize

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "raw" / "hand_landmarks_data.csv"
MODEL_SAVE_PATH = ROOT / "api" / "models" / "xgb_model.json"
MLFLOW_DIR = ROOT / "mlflow"


# ── Confusion matrix helper ───────────────────────────────────────────────────

def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="XGBoost — Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    data_path: str = str(DATA_PATH),
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("Loading and normalizing data …")
    X, y, classes = load_csv_and_normalize(data_path)
    n_classes = len(classes)
    print(f"  {X.shape[0]} samples · {X.shape[1]} features · {n_classes} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    # ── 2. Model ──────────────────────────────────────────────────────────────
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=random_state,
        n_jobs=-1,
    )

    # ── 3. MLflow run ─────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
    mlflow.set_experiment("hand-gesture-clf")

    with mlflow.start_run(run_name="xgboost"):
        print("\nFitting XGBoost …")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # ── 4. Evaluate ───────────────────────────────────────────────────────
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(y_test, y_pred, target_names=classes)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\nTest accuracy : {acc:.4f}")
        print(f"Weighted F1   : {f1:.4f}")
        print("\nClassification report:")
        print(report)

        # ── 5. MLflow log params & metrics ───────────────────────────────────
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "test_size": test_size,
            "random_state": random_state,
            "n_classes": n_classes,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        })
        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_f1_weighted": f1,
        })

        # ── 6. Confusion matrix artifact ──────────────────────────────────────
        cm_path = str(Path(tempfile.gettempdir()) / "xgb_confusion_matrix.png")
        _plot_confusion_matrix(cm, classes, cm_path)
        mlflow.log_artifact(cm_path, artifact_path="plots")
        print(f"\nConfusion matrix saved.")

        # ── 7. Log model to MLflow ────────────────────────────────────────────
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="xgboost-hand-gesture",
        )

        # ── 8. Save model locally for API inference ───────────────────────────
        MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(MODEL_SAVE_PATH))
        print(f"Model saved -> {MODEL_SAVE_PATH}")

        # Save class list alongside model
        classes_path = MODEL_SAVE_PATH.parent / "classes.txt"
        classes_path.write_text("\n".join(classes))
        print(f"Classes saved -> {classes_path}")

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run id : {run_id}")
        print(f"MLflow UI     : mlflow ui --backend-store-uri file:{MLFLOW_DIR}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost hand-gesture classifier")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        data_path=args.data,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        test_size=args.test_size,
        random_state=args.seed,
    )
