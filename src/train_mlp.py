"""
train_mlp.py
------------
Train a PyTorch MLP on normalized hand-landmark features and log to MLflow.

Architecture:
  Input(63) -> Linear(256) -> BN -> ReLU -> Dropout
             -> Linear(128) -> BN -> ReLU -> Dropout
             -> Linear(18)
  (Softmax applied at inference; CrossEntropyLoss handles it during training)

What this script does:
  1. Load data/raw/hand_landmarks_data.csv and apply positional normalization
  2. Stratified 80/10/10 train-val-test split
  3. Train MLP with early stopping on val loss
  4. Evaluate on test set: accuracy, weighted F1, confusion matrix
  5. Log params, per-epoch metrics, confusion-matrix image, and model to MLflow
  6. Save model weights to api/models/mlp_model.pt for inference

Run:
  python src/train_mlp.py
  python src/train_mlp.py --epochs 50 --lr 1e-3 --dropout 0.4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import tempfile
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from landmarks import load_csv_and_normalize

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "raw" / "hand_landmarks_data.csv"
MODEL_SAVE_PATH = ROOT / "api" / "models" / "mlp_model.pt"
MLFLOW_DIR = ROOT / "mlflow"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ─────────────────────────────────────────────────────────────────────

class HandGestureMLP(nn.Module):
    """
    63 -> Linear(256) -> BN -> ReLU -> Dropout
       -> Linear(128) -> BN -> ReLU -> Dropout
       -> Linear(n_classes)
    """

    def __init__(self, n_classes: int = 18, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities (for inference)."""
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=-1)


# ── Confusion matrix helper ───────────────────────────────────────────────────

def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: str,
    title: str = "MLP — Confusion Matrix",
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
        title=title,
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


# ── Training loop ─────────────────────────────────────────────────────────────

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    train: bool,
) -> tuple[float, float]:
    """One epoch of train or eval. Returns (avg_loss, accuracy)."""
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += len(y_batch)

    return total_loss / total, correct / total


# ── Main training function ────────────────────────────────────────────────────

def train(
    data_path: str = str(DATA_PATH),
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    patience: int = 10,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
) -> None:
    torch.manual_seed(random_state)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print(f"Device: {DEVICE}")
    print("Loading and normalizing data …")
    X, y, classes = load_csv_and_normalize(data_path)
    n_classes = len(classes)
    print(f"  {X.shape[0]} samples · {X.shape[1]} features · {n_classes} classes")

    # 80 / 10 / 10 split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, stratify=y_temp, random_state=random_state
    )
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    def _loader(X_arr, y_arr, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.tensor(X_arr, dtype=torch.float32),
            torch.tensor(y_arr, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    train_loader = _loader(X_train, y_train, shuffle=True)
    val_loader   = _loader(X_val,   y_val,   shuffle=False)
    test_loader  = _loader(X_test,  y_test,  shuffle=False)

    # ── 2. Model / optimizer / scheduler ──────────────────────────────────────
    model = HandGestureMLP(n_classes=n_classes, dropout=dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ── 3. MLflow run ─────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
    mlflow.set_experiment("hand-gesture-clf")

    with mlflow.start_run(run_name="pytorch-mlp"):
        mlflow.log_params({
            "model": "MLP",
            "architecture": "63->256->BN->ReLU->DO->128->BN->ReLU->DO->18",
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "patience": patience,
            "n_classes": n_classes,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "device": str(DEVICE),
        })

        # ── 4. Training loop with early stopping ──────────────────────────────
        best_val_loss = float("inf")
        best_state = None
        wait = 0

        print(f"\nTraining for up to {epochs} epochs (early stop patience={patience}) …")
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = _run_epoch(
                model, train_loader, criterion, optimizer, train=True
            )
            val_loss, val_acc = _run_epoch(
                model, val_loader, criterion, None, train=False
            )
            scheduler.step(val_loss)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d} | "
                    f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                    f"val loss {val_loss:.4f} acc {val_acc:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs).")
                    break

        # Restore best weights
        model.load_state_dict(best_state)

        # ── 5. Test evaluation ────────────────────────────────────────────────
        print("\nEvaluating on test set …")
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                logits = model(X_batch.to(DEVICE))
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_true.extend(y_batch.numpy())

        y_pred = np.array(all_preds)
        y_true = np.array(all_true)
        acc  = accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred, average="weighted")
        report = classification_report(y_true, y_pred, target_names=classes)
        cm = confusion_matrix(y_true, y_pred)

        print(f"\nTest accuracy : {acc:.4f}")
        print(f"Weighted F1   : {f1:.4f}")
        print("\nClassification report:")
        print(report)

        mlflow.log_metrics({"test_accuracy": acc, "test_f1_weighted": f1})

        # ── 6. Confusion matrix + classification report artifacts ─────────────
        cm_path = str(Path(tempfile.gettempdir()) / "mlp_confusion_matrix.png")
        _plot_confusion_matrix(cm, classes, cm_path)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        report_path = str(Path(tempfile.gettempdir()) / "mlp_classification_report.txt")
        Path(report_path).write_text(report)
        mlflow.log_artifact(report_path, artifact_path="reports")

        # ── 7. Log model ──────────────────────────────────────────────────────
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name="mlp-hand-gesture",
        )

        # ── 8. Save locally for API ───────────────────────────────────────────
        MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "classes": classes,
                "n_classes": n_classes,
                "dropout": dropout,
            },
            MODEL_SAVE_PATH,
        )
        print(f"\nModel saved -> {MODEL_SAVE_PATH}")

        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run id : {run_id}")
        print(f"MLflow UI     : mlflow ui --backend-store-uri file:{MLFLOW_DIR}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PyTorch MLP hand-gesture classifier")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        random_state=args.seed,
    )
