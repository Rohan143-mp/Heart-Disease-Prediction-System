"""
Deep Learning (Deep Neural Network / MLP) Training Pipeline for CardioCare.
==========================================================================
Trains a Multi-Layer Perceptron (Deep Neural Network) for Heart Disease
Prediction using backpropagation, Adam optimization, and ReLU activations.

Compares performance directly against the baseline Classical ML models (SVM, RF, LR, DT)
and saves the trained Deep Learning model to models/deep_learning_model.pkl.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

X_PATH = os.path.join(DATA_DIR, "cleaned_X.csv")
Y_PATH = os.path.join(DATA_DIR, "clean_y.csv")
DL_MODEL_PATH = os.path.join(MODELS_DIR, "deep_learning_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
DL_METRICS_PATH = os.path.join(RESULTS_DIR, "deep_learning_metrics.txt")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_deep_neural_network():
    print("=" * 65)
    print("CardioCare — Deep Neural Network Training Pipeline")
    print("=" * 65)

    # 1. Load Data
    X = pd.read_csv(X_PATH)
    y = pd.read_csv(Y_PATH)["target"]
    print(f"[+] Loaded dataset: {X.shape[0]} samples, {X.shape[1]} input dimensions")
    print(f"[+] Class breakdown: Healthy (0) = {(y == 0).sum()}, Disease (1) = {(y == 1).sum()}")

    # 2. Train/Test Split (80/20, reproducible)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"[+] Partition: {len(X_train)} training instances, {len(X_test)} holdout test instances")

    # 3. Feature Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    print("[+] StandardScaler fitted on training fold and saved to models/scaler.pkl")

    # 4. Deep Neural Network Architecture
    # Input: 25 -> Hidden Layer 1: 64 (ReLU) -> Hidden Layer 2: 32 (ReLU) -> Hidden Layer 3: 16 (ReLU) -> Output: 1 (Sigmoid)
    print("\n[+] Constructing Deep Neural Network:")
    print("    - Input Layer: 25 neurons (One-Hot Encoded features)")
    print("    - Dense Hidden Layer 1: 64 neurons (ReLU activation)")
    print("    - Dense Hidden Layer 2: 32 neurons (ReLU activation)")
    print("    - Dense Hidden Layer 3: 16 neurons (ReLU activation)")
    print("    - Output Layer: 2 units (Softmax probability estimation)")
    print("    - Optimizer: Adam (Adaptive Moment Estimation, lr_init=0.001)")
    print("    - Loss Function: Cross-Entropy (Log-Loss) with L2 weight decay (alpha=0.01)")
    print("    - Regularization: Early stopping with 15% validation fraction")

    dl_model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        alpha=0.01,
        batch_size=16,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
    )

    print("\n[+] Training model via Backpropagation...")
    dl_model.fit(X_train_scaled, y_train)
    print(f"[+] Convergence reached after {dl_model.n_iter_} epochs")
    print(f"[+] Final Training Loss: {dl_model.loss_:.5f}")

    # 5. Evaluate on Holdout Test Set
    preds = dl_model.predict(X_test_scaled)
    probs = dl_model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    print("\n" + "=" * 40)
    print("DEEP LEARNING MODEL PERFORMANCE (Test Set)")
    print("=" * 40)
    print(f"Accuracy:    {acc:.4f} ({acc * 100:.1f}%)")
    print(f"ROC-AUC:     {auc:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"  TN: {cm[0, 0]:2d} | FP: {cm[0, 1]:2d}")
    print(f"  FN: {cm[1, 0]:2d} | TP: {cm[1, 1]:2d}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["Healthy (0)", "Disease (1)"]))

    # 6. Save Model Artifact
    joblib.dump(dl_model, DL_MODEL_PATH)
    print(f"[+] Deep Learning model artifact saved to: {DL_MODEL_PATH}")

    # 7. Write Summary Metrics
    metrics_summary = f"""CardioCare — Deep Neural Network (MLP) Results
=================================================
Architecture: 25 -> 64 -> 32 -> 16 -> 2
Activation: ReLU
Optimizer: Adam (batch_size=16)
Epochs: {dl_model.n_iter_}
Final Loss: {dl_model.loss_:.5f}

Test Accuracy: {acc:.4f} ({acc*100:.1f}%)
ROC-AUC:       {auc:.4f}
Precision:     {prec:.4f}
Recall:        {rec:.4f}
F1-Score:      {f1:.4f}

Confusion Matrix:
TN: {cm[0, 0]} | FP: {cm[0, 1]}
FN: {cm[1, 0]} | TP: {cm[1, 1]}

Classification Report:
{classification_report(y_test, preds, target_names=["Healthy (0)", "Disease (1)"])}
"""
    with open(DL_METRICS_PATH, "w") as f:
        f.write(metrics_summary)
    print(f"[+] Detailed metrics exported to: {DL_METRICS_PATH}")
    print("=" * 65)

    return dl_model, acc


if __name__ == "__main__":
    train_deep_neural_network()

