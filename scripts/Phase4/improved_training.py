# -*- coding: utf-8 -*-
"""
=============================================================
PHASE 4 - STEP 1: Improved Model Training
=============================================================

WHAT IS DIFFERENT IN PHASE 4:
-------------------------------
1. Lower initial learning rate (1e-4 instead of 1e-3)
   Smaller steps = more stable convergence, no spiking.

2. Warmup for first 5 epochs
   Learning rate starts very low and gradually rises to
   the target rate. Prevents the wild early-epoch instability
   seen in Phase 3 (epochs 5, 11 spiking to val_loss > 3).

3. Cosine annealing scheduler
   After warmup, the learning rate follows a smooth cosine
   curve downward. Better than ReduceLROnPlateau for CNNs.

4. Class weights
   We compute how underrepresented each class is and tell
   the loss function to penalise mistakes on rare classes
   more heavily. Red giant errors cost more than quasar
   errors, forcing the model to pay attention to hard classes.

5. Deeper architecture with residual connections
   We add skip connections (ResNet-style) that let gradients
   flow more easily through the network. Prevents the gradient
   from vanishing in deep layers and improves convergence.

6. Data augmentation
   We add small random noise and flux shifts to training
   spectra each epoch. This makes the model more robust and
   acts like having more training data.

7. More epochs (100) with longer patience (12)
   Phase 3 was still improving at epoch 50. We give it more
   room to converge properly.

WHAT THIS SCRIPT DOES:
    1. Loads X.npy and y.npy
    2. Applies class weight computation
    3. Builds improved CNN with residual connections
    4. Trains with warmup + cosine annealing + augmentation
    5. Evaluates on test set with full metrics
    6. Saves improved model to /models/cnn_v2.pth
    7. Re-extracts features with the better model
    8. Plots training curves and confusion matrix

HOW TO RUN:
    Run:   python improved_training.py
    Time:  3-6 minutes (varies with systems)

OUTPUT FILES:
    - /models/cnn_v2.pth
    - /data/processed/features_v2.npy
    - /data/processed/features_v2_labels.npy
    - /notebooks/training_curves_v2.png
    - /notebooks/confusion_matrix_v2.png
    - /notebooks/per_class_metrics.png

REQUIRES:
    pip install torch scikit-learn numpy pandas matplotlib tqdm
=============================================================
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR       = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "stellar-spectra-ai")
PROCESSED_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
NOTEBOOKS      = os.path.join(BASE_DIR, "notebooks")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(NOTEBOOKS,  exist_ok=True)

X_PATH           = os.path.join(PROCESSED_DIR, "X.npy")
Y_PATH           = os.path.join(PROCESSED_DIR, "y.npy")
LABEL_MAP_PATH   = os.path.join(PROCESSED_DIR, "label_map.json")
MODEL_PATH       = os.path.join(MODELS_DIR,    "cnn_v2.pth")
FEATURES_PATH    = os.path.join(PROCESSED_DIR, "features_v2.npy")
FEAT_LABELS_PATH = os.path.join(PROCESSED_DIR, "features_v2_labels.npy")

# ─────────────────────────────────────────────
# HYPERPARAMETERS (improved from Phase 3)
# ─────────────────────────────────────────────
BATCH_SIZE    = 32
LEARNING_RATE = 1e-4      # lower than Phase 3 (was 1e-3) -- more stable
MAX_EPOCHS    = 100       # more than Phase 3 (was 50) -- was still improving
PATIENCE      = 12        # longer patience -- cosine schedule needs time
WARMUP_EPOCHS = 5         # gradual LR increase at start
RANDOM_SEED   = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CLASS_COLORS = {
    0: "#4A90D9",
    1: "#E8593C",
    2: "#3BAD75",
    3: "#D4A017",
}


# ─────────────────────────────────────────────
# IMPROVED CNN WITH RESIDUAL CONNECTIONS
# ─────────────────────────────────────────────
class ResidualBlock1D(nn.Module):
    """
    A residual block adds the INPUT directly to the OUTPUT
    of two convolutional layers (a skip connection).

    Without residual:  output = Conv(Conv(x))
    With residual:     output = Conv(Conv(x)) + x

    Why this helps:
    In deep networks, gradients can become very small as they
    flow backwards through many layers (vanishing gradient).
    The skip connection gives gradients a shortcut path,
    keeping them large enough to update early layers properly.
    This is the core idea behind ResNet (He et al. 2015) which
    dramatically improved deep network training stability.
    """
    def __init__(self, channels, kernel_size=3):
        super(ResidualBlock1D, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)   # skip connection here


class ImprovedStellarCNN(nn.Module):
    """
    Improved CNN with:
    - Residual blocks for training stability
    - Deeper architecture (more capacity)
    - Same extract_features interface as Phase 3 model
    """
    def __init__(self, input_length=3000, n_classes=4):
        super(ImprovedStellarCNN, self).__init__()

        # Initial projection: 1 channel -> 64 channels
        self.input_proj = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),      # 3000 -> 1500
        )

        # Residual blocks at 64 channels
        self.res_block_1 = ResidualBlock1D(64, kernel_size=7)

        # Downsample to 128 channels
        self.down_1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),      # 1500 -> 750
        )

        # Residual blocks at 128 channels
        self.res_block_2 = ResidualBlock1D(128, kernel_size=5)

        # Downsample to 256 channels
        self.down_2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),      # 750 -> 375
        )

        # Residual blocks at 256 channels
        self.res_block_3 = ResidualBlock1D(256, kernel_size=3)

        # Global average pooling: (256, 375) -> (256,)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Dense feature layers
        self.feature_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_block_1(x)
        x = self.down_1(x)
        x = self.res_block_2(x)
        x = self.down_2(x)
        x = self.res_block_3(x)
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.feature_layers(x)
        return self.classifier(x)

    def extract_features(self, x):
        x = self.input_proj(x)
        x = self.res_block_1(x)
        x = self.down_1(x)
        x = self.res_block_2(x)
        x = self.down_2(x)
        x = self.res_block_3(x)
        x = self.global_avg_pool(x).squeeze(-1)
        return self.feature_layers(x)

    def get_conv_activations(self, x):
        x = self.input_proj(x)
        x = self.res_block_1(x)
        x = self.down_1(x)
        x = self.res_block_2(x)
        x = self.down_2(x)
        return self.res_block_3(x)


# ─────────────────────────────────────────────
# DATA AUGMENTATION
# Applied only to training data each batch.
# Adds small random variations to prevent overfitting.
# ─────────────────────────────────────────────
def augment_batch(X_batch):
    """
    Applies three lightweight augmentations to a batch:

    1. Gaussian noise (sigma=0.01):
       Adds tiny random fluctuations mimicking detector noise.
       Spectrum values are in [0,3] so 0.01 noise is ~0.3% --
       imperceptible to humans but teaches the model to ignore
       pixel-level noise.

    2. Random flux scaling (0.95 to 1.05):
       Multiplies the entire spectrum by a random factor near 1.
       Mimics small calibration differences between observations.

    3. Random spectral shift (+/- 3 pixels):
       Shifts the spectrum slightly along the wavelength axis.
       Teaches the model that a feature at 6560 and 6566 are
       the same thing (Halpha with slightly different velocity).

    These are applied randomly during training only -- never
    during validation or test evaluation.
    """
    device = X_batch.device

    # 1. Gaussian noise
    noise = torch.randn_like(X_batch) * 0.01
    X_aug = X_batch + noise

    # 2. Random flux scaling per spectrum in the batch
    scale = (torch.rand(X_batch.shape[0], 1, 1, device=device) * 0.10 + 0.95)
    X_aug = X_aug * scale

    # 3. Random pixel shift (-3 to +3 pixels)
    shift = torch.randint(-3, 4, (1,)).item()
    if shift != 0:
        X_aug = torch.roll(X_aug, shift, dims=2)

    # Clip back to valid range
    X_aug = torch.clamp(X_aug, 0.0, 3.0)
    return X_aug


# ─────────────────────────────────────────────
# WARMUP + COSINE ANNEALING SCHEDULER
# ─────────────────────────────────────────────
def get_lr(epoch, warmup_epochs, max_epochs, base_lr):
    """
    Learning rate schedule:
        Epochs 1 to warmup_epochs: linear warmup from 0 to base_lr
        After warmup: cosine annealing from base_lr to base_lr/100

    Linear warmup prevents the wild val_loss spikes seen in
    Phase 3 early epochs by starting with tiny updates and
    gradually increasing step size as the model stabilises.

    Cosine annealing smoothly reduces the learning rate in a
    cosine curve, which consistently outperforms step decay
    and ReduceLROnPlateau for CNNs on spectral data.
    """
    if epoch <= warmup_epochs:
        return base_lr * (epoch / warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        return base_lr * (1 + math.cos(math.pi * progress)) / 2


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("PHASE 4 - STEP 1: Improved Model Training")
print("=" * 60)

for path, name in [(X_PATH, "X.npy"), (Y_PATH, "y.npy")]:
    if not os.path.exists(path):
        print(f"\nERROR: {name} not found. Complete Phase 2 first.")
        sys.exit(1)

X = np.load(X_PATH)
y = np.load(Y_PATH)

with open(LABEL_MAP_PATH) as f:
    label_map  = json.load(f)
int_to_label   = label_map["int_to_label"]
CLASS_NAMES    = [int_to_label[str(i)] for i in range(len(int_to_label))]
N_CLASSES      = len(CLASS_NAMES)

print(f"\nLoaded X: {X.shape}")
print(f"Loaded y: {y.shape}")
print(f"Classes:  {CLASS_NAMES}")

# Fix NaN/Inf
if np.isnan(X).any() or np.isinf(X).any():
    print("WARNING: NaN/Inf found -- replacing with 0")
    X = np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=0.0)


# ─────────────────────────────────────────────
# COMPUTE CLASS WEIGHTS
# ─────────────────────────────────────────────
class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(N_CLASSES),
    y=y
)
print("\nClass weights (higher = model penalised more for missing this class):")
for name, w in zip(CLASS_NAMES, class_weights_arr):
    bar = "█" * int(w * 10)
    print(f"  {name:<22}  weight={w:.3f}  {bar}")


# ─────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
)

print(f"\nData split:")
print(f"  Train:      {len(X_train)} samples")
print(f"  Validation: {len(X_val)}   samples")
print(f"  Test:       {len(X_test)}  samples")


# ─────────────────────────────────────────────
# PYTORCH DATASETS
# ─────────────────────────────────────────────
def to_tensors(X_arr, y_arr):
    X_t = torch.tensor(X_arr, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y_arr, dtype=torch.long)
    return TensorDataset(X_t, y_t)

train_loader = DataLoader(to_tensors(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(to_tensors(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(to_tensors(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False)


# ─────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

model = ImprovedStellarCNN(input_length=X.shape[1], n_classes=N_CLASSES).to(device)

# Weighted loss function -- penalises red_giant errors more
weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float32).to(device)
criterion      = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer      = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
print(f"\nTraining for up to {MAX_EPOCHS} epochs...")
print(f"Warmup: {WARMUP_EPOCHS} epochs | Early stop patience: {PATIENCE}")
print("-" * 65)

train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
lr_history               = []
best_val_loss            = float("inf")
epochs_no_improve        = 0
best_epoch               = 0

for epoch in range(1, MAX_EPOCHS + 1):

    # Update learning rate
    current_lr = get_lr(epoch, WARMUP_EPOCHS, MAX_EPOCHS, LEARNING_RATE)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    lr_history.append(current_lr)

    # ── Train ─────────────────────────────────
    model.train()
    run_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Apply augmentation to training batches only
        X_batch = augment_batch(X_batch)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()

        # Gradient clipping -- prevents exploding gradients
        # (another cause of the Phase 3 val_loss spikes)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        run_loss += loss.item() * len(y_batch)
        correct  += (outputs.argmax(1) == y_batch).sum().item()
        total    += len(y_batch)

    train_loss = run_loss / total
    train_acc  = correct  / total

    # ── Validate ──────────────────────────────
    model.eval()
    val_loss_sum, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs  = model(X_batch)
            loss     = criterion(outputs, y_batch)
            val_loss_sum += loss.item() * len(y_batch)
            val_correct  += (outputs.argmax(1) == y_batch).sum().item()
            val_total    += len(y_batch)

    val_loss = val_loss_sum / val_total
    val_acc  = val_correct  / val_total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # Save best model
    improved = ""
    if val_loss < best_val_loss:
        best_val_loss     = val_loss
        epochs_no_improve = 0
        best_epoch        = epoch
        torch.save(model.state_dict(), MODEL_PATH)
        improved = " <- best"
    else:
        epochs_no_improve += 1

    print(f"  Epoch {epoch:>3}/{MAX_EPOCHS}  "
          f"lr={current_lr:.2e}  "
          f"train={train_loss:.4f}/{train_acc:.3f}  "
          f"val={val_loss:.4f}/{val_acc:.3f}{improved}")

    if epochs_no_improve >= PATIENCE:
        print(f"\n  Early stopping at epoch {epoch}.")
        print(f"  Best model: epoch {best_epoch}  val_loss={best_val_loss:.4f}")
        break

print(f"\nBest model saved to: {MODEL_PATH}")


# ─────────────────────────────────────────────
# EVALUATE ON TEST SET
# ─────────────────────────────────────────────
print("\nEvaluating on test set with best model weights...")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch.to(device))
        probs   = torch.softmax(outputs, dim=1).cpu().numpy()
        preds   = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())
        all_probs.extend(probs)

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

test_acc = (all_preds == all_labels).mean()

print(f"\nTest accuracy: {test_acc:.4f}  ({test_acc*100:.1f}%)")
print()
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# ─────────────────────────────────────────────
# EXTRACT FEATURES WITH IMPROVED MODEL
# ─────────────────────────────────────────────
print("\nExtracting features with improved model...")

all_loader = DataLoader(
    to_tensors(X, y), batch_size=BATCH_SIZE, shuffle=False
)
features_list, labels_list = [], []

model.eval()
with torch.no_grad():
    for X_batch, y_batch in tqdm(all_loader, desc="Extracting"):
        feats = model.extract_features(X_batch.to(device))
        features_list.append(feats.cpu().numpy())
        labels_list.append(y_batch.numpy())

features_v2 = np.concatenate(features_list, axis=0)
labels_v2   = np.concatenate(labels_list,   axis=0)

np.save(FEATURES_PATH,    features_v2)
np.save(FEAT_LABELS_PATH, labels_v2)
print(f"Features saved: {features_v2.shape} -> {FEATURES_PATH}")


# ─────────────────────────────────────────────
# PLOT 1: Training curves (smooth this time)
# ─────────────────────────────────────────────
epochs_range = range(1, len(train_losses) + 1)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss
axes[0].plot(epochs_range, train_losses, label="Train",      color="#4A90D9", lw=1.5)
axes[0].plot(epochs_range, val_losses,   label="Validation", color="#E8593C", lw=1.5)
axes[0].axvline(best_epoch, color="gray", lw=1, linestyle="--", label=f"Best ({best_epoch})")
axes[0].set_title("Loss", fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.2)

# Accuracy
axes[1].plot(epochs_range, [a*100 for a in train_accs], label="Train",      color="#4A90D9", lw=1.5)
axes[1].plot(epochs_range, [a*100 for a in val_accs],   label="Validation", color="#E8593C", lw=1.5)
axes[1].axvline(best_epoch, color="gray", lw=1, linestyle="--")
axes[1].axhline(73.9, color="orange", lw=1, linestyle=":", label="Phase 3 baseline (73.9%)")
axes[1].set_title("Accuracy", fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(True, alpha=0.2)
axes[1].set_ylim(0, 105)

# Learning rate schedule
axes[2].plot(epochs_range, lr_history, color="#3BAD75", lw=1.5)
axes[2].axvspan(1, WARMUP_EPOCHS, alpha=0.1, color="blue", label=f"Warmup ({WARMUP_EPOCHS} epochs)")
axes[2].set_title("Learning Rate Schedule", fontweight="bold")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Learning Rate")
axes[2].legend()
axes[2].grid(True, alpha=0.2)

plt.suptitle(f"Phase 4 Training  --  Best val accuracy: {max(val_accs)*100:.1f}%  |  Test accuracy: {test_acc*100:.1f}%",
             fontsize=13, fontweight="bold")
plt.tight_layout()
curves_path = os.path.join(NOTEBOOKS, "training_curves_v2.png")
plt.savefig(curves_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nTraining curves saved to:\n  {curves_path}")


# ─────────────────────────────────────────────
# PLOT 2: Confusion matrix
# ─────────────────────────────────────────────
cm  = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 7))
im  = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax)

ax.set_xticks(range(N_CLASSES))
ax.set_yticks(range(N_CLASSES))
ax.set_xticklabels([c.replace("_", "\n") for c in CLASS_NAMES], fontsize=10)
ax.set_yticklabels([c.replace("_", " ")  for c in CLASS_NAMES], fontsize=10)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("True",      fontsize=12)
ax.set_title(f"Confusion Matrix  --  Test Accuracy: {test_acc*100:.1f}%\n(diagonal = correct predictions)",
             fontsize=12, fontweight="bold")

thresh = cm.max() / 2.0
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
cm_path = os.path.join(NOTEBOOKS, "confusion_matrix_v2.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Confusion matrix saved to:\n  {cm_path}")


# ─────────────────────────────────────────────
# PLOT 3: Per-class precision / recall / F1
# ─────────────────────────────────────────────
metrics = {}
for i, name in enumerate(CLASS_NAMES):
    mask = all_labels == i
    metrics[name] = {
        "precision": precision_score(all_labels, all_preds, labels=[i], average="macro", zero_division=0),
        "recall":    recall_score(   all_labels, all_preds, labels=[i], average="macro", zero_division=0),
        "f1":        f1_score(       all_labels, all_preds, labels=[i], average="macro", zero_division=0),
    }

fig, ax = plt.subplots(figsize=(11, 5))
x       = np.arange(N_CLASSES)
width   = 0.25

bars_p = ax.bar(x - width, [metrics[n]["precision"] for n in CLASS_NAMES],
                width, label="Precision", color="#4A90D9", alpha=0.85)
bars_r = ax.bar(x,          [metrics[n]["recall"]    for n in CLASS_NAMES],
                width, label="Recall",    color="#E8593C", alpha=0.85)
bars_f = ax.bar(x + width,  [metrics[n]["f1"]        for n in CLASS_NAMES],
                width, label="F1 Score",  color="#3BAD75", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([c.replace("_", "\n") for c in CLASS_NAMES], fontsize=11)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.15)
ax.set_title("Per-Class Metrics  --  Phase 4 vs Phase 3 baseline",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.axhline(0.73, color="orange", lw=1, linestyle=":", alpha=0.7, label="Phase 3 overall")
ax.grid(True, alpha=0.2, axis="y")

# Label each bar
for bars in [bars_p, bars_r, bars_f]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
metrics_path = os.path.join(NOTEBOOKS, "per_class_metrics.png")
plt.savefig(metrics_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Per-class metrics saved to:\n  {metrics_path}")


# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 4 - STEP 1 COMPLETE")
print("=" * 60)
print(f"  Phase 4 accuracy:  {test_acc*100:.1f}%")
print(f"\n  Per-class F1 scores:")
for name in CLASS_NAMES:
    f1  = metrics[name]["f1"]
    bar = "█" * int(f1 * 20)
    print(f"    {name:<22}  F1={f1:.3f}  {bar}")
print(f"\n  Model:    {MODEL_PATH}")
print(f"  Features: {FEATURES_PATH}")
print(f"\nNext: Run  evaluate_and_report.py")