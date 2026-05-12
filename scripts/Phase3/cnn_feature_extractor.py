# -*- coding: utf-8 -*-
"""
=============================================================
PHASE 3 - STEP 1: Build the CNN Feature Extractor
=============================================================

WHAT IS A CNN AND WHY USE IT FOR SPECTRA?
------------------------------------------
A Convolutional Neural Network (CNN) was originally invented
for images, but it works beautifully on 1D data like spectra.

Think of a spectrum as a 1D image -- a row of 3000 pixels
where each pixel is a flux value at a specific wavelength.
A CNN slides small filters (called kernels) across this row,
learning to detect patterns at different scales:

    Small kernels (width ~5):  detect sharp absorption line dips
    Medium kernels (width ~21): detect line profiles and widths
    Large kernels (width ~51): detect broad continuum shapes

The key advantage over hand-crafted features:
    You do NOT need to tell the CNN which lines to look at.
    It discovers the most useful patterns by itself during
    training. It may find patterns you never thought of.

HOW THE CNN IS STRUCTURED:
----------------------------
Input: flux array of shape (3000,) -- one spectrum

Block 1:  Conv1D(filters=32,  kernel=11) -> BatchNorm -> ReLU -> MaxPool(2)
Block 2:  Conv1D(filters=64,  kernel=7)  -> BatchNorm -> ReLU -> MaxPool(2)
Block 3:  Conv1D(filters=128, kernel=5)  -> BatchNorm -> ReLU -> MaxPool(2)
Block 4:  Conv1D(filters=256, kernel=3)  -> BatchNorm -> ReLU -> GlobalAvgPool

Flatten -> Dense(128) -> Dropout(0.4) -> Dense(64) -> Dropout(0.3)
       -> Output Dense(4) -> Softmax

Output: 4 probability values (one per class)
        e.g. [0.02, 0.91, 0.04, 0.03]
        meaning: 91% chance this is a quasar

WHAT EACH LAYER DOES:
----------------------
Conv1D      -- slides filters across the spectrum to detect local patterns
BatchNorm   -- normalises activations so training is stable and fast
ReLU        -- activation function: keeps positive values, zeros negatives
MaxPool     -- downsamples by taking the max value in each window,
               reducing length and making features translation-invariant
GlobalAvgPool -- collapses the entire spatial dimension to one value per filter
Dense       -- fully connected layer that combines all detected features
Dropout     -- randomly zeros some neurons during training to prevent
               the model from memorising the training data (overfitting)
Softmax     -- converts raw scores to probabilities that sum to 1.0

WHAT IS THE FEATURE EXTRACTOR (for Phase 3)?
----------------------------------------------
We take the CNN above and CUT IT just before the final output layer.
The output of the Dense(64) layer is a 64-dimensional vector --
this is the LEARNED FEATURE REPRESENTATION of the spectrum.

Instead of raw 3000-point flux values, each spectrum is now
described by 64 numbers that the CNN has learned are most
useful for distinguishing stellar classes.

These 64-dimensional vectors are what we:
    1. Save as features.npy (Phase 3 output)
    2. Visualise with t-SNE and PCA (Step 2)
    3. Pass into the final classifier (Phase 4)

WHAT THIS SCRIPT DOES:
    1. Loads X.npy and y.npy from Phase 2
    2. Splits into train (70%) / validation (15%) / test (15%)
    3. Builds the CNN architecture in PyTorch
    4. Trains the CNN for up to 50 epochs with early stopping
    5. Saves the trained model weights
    6. Extracts 64-dimensional features for every spectrum
    7. Saves features.npy and the label array
    8. Plots training/validation loss and accuracy curves

HOW TO RUN:
    1. Make sure Phase 2 is complete (X.npy and y.npy exist)
    2. Install PyTorch:
           pip install torch torchvision
       If you have an NVIDIA GPU also run:
           pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    3. Run:
           python 10_cnn_feature_extractor.py

       Training takes 5-15 minutes on CPU, 1-3 minutes on GPU.

OUTPUT FILES:
    - /models/cnn_feature_extractor.pth   -- trained model weights
    - /data/processed/features.npy        -- shape (n_samples, 64)
    - /data/processed/features_labels.npy -- shape (n_samples,) integer labels
    - /notebooks/training_curves.png      -- loss and accuracy over epochs
    - /notebooks/confusion_matrix.png     -- per-class prediction accuracy

REQUIRES:
    pip install torch numpy pandas matplotlib scikit-learn tqdm
=============================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR      = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "stellar-spectra-ai")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
NOTEBOOKS     = os.path.join(BASE_DIR, "notebooks")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(NOTEBOOKS,  exist_ok=True)

X_PATH          = os.path.join(PROCESSED_DIR, "X.npy")
Y_PATH          = os.path.join(PROCESSED_DIR, "y.npy")
LABEL_MAP_PATH  = os.path.join(PROCESSED_DIR, "label_map.json")
MODEL_PATH      = os.path.join(MODELS_DIR, "cnn_feature_extractor.pth")
FEATURES_PATH   = os.path.join(PROCESSED_DIR, "features.npy")
FEAT_LABELS_PATH= os.path.join(PROCESSED_DIR, "features_labels.npy")

# ─────────────────────────────────────────────
# HYPERPARAMETERS
# These control how training behaves.
# You can tune them later but defaults work well.
# ─────────────────────────────────────────────
BATCH_SIZE    = 32      # how many spectra to process at once
LEARNING_RATE = 1e-3    # how fast the model updates its weights
MAX_EPOCHS    = 50      # maximum training rounds
PATIENCE      = 7       # stop if val loss doesnt improve for this many epochs
DROPOUT_1     = 0.4     # dropout rate after first dense layer
DROPOUT_2     = 0.3     # dropout rate after second dense layer
RANDOM_SEED   = 42      # for reproducible train/val/test splits

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CLASS_COLORS = {
    0: "#4A90D9",   # white_dwarf
    1: "#E8593C",   # quasar
    2: "#3BAD75",   # main_sequence
    3: "#D4A017",   # red_giant
}


# ─────────────────────────────────────────────
# CNN ARCHITECTURE
# ─────────────────────────────────────────────
class StellarCNN(nn.Module):
    """
    1D Convolutional Neural Network for stellar spectra classification.

    The architecture uses 4 convolutional blocks that progressively:
        - Increase the number of filters (32 -> 64 -> 128 -> 256)
        - Decrease the sequence length via MaxPooling
        - Learn features at different scales via decreasing kernel sizes

    The final layers produce:
        - A 64-dimensional feature vector (the learned representation)
        - A 4-class probability distribution (the classification output)

    We separate these two outputs so we can use the same model for
    both classification (Phase 4) and feature extraction (Phase 3).
    """

    def __init__(self, input_length=3000, n_classes=4):
        super(StellarCNN, self).__init__()

        # ── Convolutional blocks ──────────────────
        # Each block: Conv -> BatchNorm -> ReLU -> MaxPool
        # After 4 MaxPool(2) layers, sequence length = 3000/16 = 187

        self.conv_blocks = nn.Sequential(

            # Block 1: detect sharp absorption line dips
            nn.Conv1d(1, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2: detect line profiles and widths
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 3: detect multi-line patterns
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 4: detect broad continuum-level features
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # Global Average Pooling: collapses (256, L) -> (256,)
        # This makes the model independent of input length variations
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # ── Dense layers ──────────────────────────
        # These combine all the detected features into
        # a final classification decision

        self.feature_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_2),
        )

        # Final classification head
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        """
        Forward pass through the full network.
        x shape: (batch_size, 1, 3000)
        Returns: class logits of shape (batch_size, 4)
        """
        x = self.conv_blocks(x)
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)              # (batch, 256, 1) -> (batch, 256)
        x = self.feature_layers(x)     # (batch, 256) -> (batch, 64)
        return self.classifier(x)      # (batch, 64)  -> (batch, 4)

    def extract_features(self, x):
        """
        Forward pass that STOPS at the 64-dim feature layer.
        Used in Phase 3 to get the learned representation
        without the final classification step.
        x shape: (batch_size, 1, 3000)
        Returns: feature vectors of shape (batch_size, 64)
        """
        x = self.conv_blocks(x)
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        x = self.feature_layers(x)
        return x


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("PHASE 3 - STEP 1: CNN Feature Extractor")
print("=" * 60)

for path, name in [(X_PATH, "X.npy"), (Y_PATH, "y.npy")]:
    if not os.path.exists(path):
        print(f"\nERROR: {name} not found at:\n  {path}")
        print("Make sure Phase 2 is complete.")
        sys.exit(1)

X = np.load(X_PATH)
y = np.load(Y_PATH)

with open(LABEL_MAP_PATH) as f:
    label_map  = json.load(f)
int_to_label   = label_map["int_to_label"]
CLASS_NAMES    = [int_to_label[str(i)] for i in range(len(int_to_label))]

print(f"\nLoaded X: {X.shape}  (samples x wavelength points)")
print(f"Loaded y: {y.shape}  (integer class labels)")
print(f"Classes:  {CLASS_NAMES}")

# Check for NaN/Inf -- these will crash training
if np.isnan(X).any() or np.isinf(X).any():
    print("\nWARNING: NaN or Inf values found in X.")
    print("Replacing with 0 -- consider re-running Phase 2 if many affected.")
    X = np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=0.0)


# ─────────────────────────────────────────────
# TRAIN / VALIDATION / TEST SPLIT
#
# We split the data into three sets:
#   Train (70%)      -- model learns from these
#   Validation (15%) -- we check progress during training
#                       but NEVER train on these
#   Test (15%)       -- final evaluation AFTER training
#                       completely untouched until the end
#
# stratify=y means each split has the same class proportions
# ─────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
)

print(f"\nData split:")
print(f"  Train:      {len(X_train)} samples (70%)")
print(f"  Validation: {len(X_val)}   samples (15%)")
print(f"  Test:       {len(X_test)}  samples (15%)")


# ─────────────────────────────────────────────
# PREPARE PYTORCH DATASETS
#
# PyTorch needs data as Tensors (its own array format).
# We add a channel dimension: (n, 3000) -> (n, 1, 3000)
# This is because Conv1d expects (batch, channels, length).
# We have 1 channel (just flux) unlike images which have 3 (RGB).
# ─────────────────────────────────────────────
def to_tensors(X_arr, y_arr):
    X_t = torch.tensor(X_arr, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y_arr, dtype=torch.long)
    return TensorDataset(X_t, y_t)

train_ds = to_tensors(X_train, y_train)
val_ds   = to_tensors(X_val,   y_val)
test_ds  = to_tensors(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)


# ─────────────────────────────────────────────
# DEVICE SELECTION
# Uses GPU automatically if available, else CPU
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────
# BUILD MODEL, LOSS FUNCTION, OPTIMISER
# ─────────────────────────────────────────────
model     = StellarCNN(input_length=X.shape[1], n_classes=len(CLASS_NAMES)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler: reduce LR by half if val loss
# doesnt improve for 3 epochs -- helps fine-tune later training
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
print(f"\nTraining for up to {MAX_EPOCHS} epochs (early stop patience={PATIENCE})...")
print("-" * 60)

train_losses, val_losses     = [], []
train_accs,   val_accs       = [], []
best_val_loss                = float("inf")
epochs_no_improve            = 0
best_epoch                   = 0

for epoch in range(1, MAX_EPOCHS + 1):

    # ── Training phase ────────────────────────
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(y_batch)
        preds         = outputs.argmax(dim=1)
        correct      += (preds == y_batch).sum().item()
        total        += len(y_batch)

    train_loss = running_loss / total
    train_acc  = correct / total

    # ── Validation phase ──────────────────────
    model.eval()
    val_running_loss = 0.0
    val_correct      = 0
    val_total        = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs  = model(X_batch)
            loss     = criterion(outputs, y_batch)

            val_running_loss += loss.item() * len(y_batch)
            preds             = outputs.argmax(dim=1)
            val_correct      += (preds == y_batch).sum().item()
            val_total        += len(y_batch)

    val_loss = val_running_loss / val_total
    val_acc  = val_correct / val_total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    scheduler.step(val_loss)

    # ── Print progress ────────────────────────
    improved = ""
    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        epochs_no_improve = 0
        best_epoch       = epoch
        torch.save(model.state_dict(), MODEL_PATH)
        improved = " <- best"
    else:
        epochs_no_improve += 1

    print(f"  Epoch {epoch:>3}/{MAX_EPOCHS}  "
          f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
          f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}{improved}")

    # ── Early stopping ────────────────────────
    if epochs_no_improve >= PATIENCE:
        print(f"\n  Early stopping at epoch {epoch}.")
        print(f"  Best model was at epoch {best_epoch}.")
        break

print(f"\nBest validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
print(f"Model saved to: {MODEL_PATH}")


# ─────────────────────────────────────────────
# EVALUATE ON TEST SET
# Load the best saved weights before evaluating
# ─────────────────────────────────────────────
print("\nEvaluating on held-out test set...")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

all_preds  = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch.to(device))
        preds   = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

test_acc = (all_preds == all_labels).mean()
print(f"\nTest accuracy: {test_acc:.4f}  ({test_acc*100:.1f}%)")
print()
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))


# ─────────────────────────────────────────────
# EXTRACT FEATURES FOR ALL SPECTRA
#
# Now we use the trained model as a feature extractor.
# We pass every spectrum (train + val + test) through
# the CNN and collect the 64-dimensional output of the
# last dense layer BEFORE the classifier head.
# ─────────────────────────────────────────────
print("\nExtracting 64-dimensional features for all spectra...")

all_loader = DataLoader(
    to_tensors(X, y),
    batch_size=BATCH_SIZE,
    shuffle=False
)

features_list  = []
labels_list    = []

model.eval()
with torch.no_grad():
    for X_batch, y_batch in tqdm(all_loader, desc="Extracting"):
        feats = model.extract_features(X_batch.to(device))
        features_list.append(feats.cpu().numpy())
        labels_list.append(y_batch.numpy())

features = np.concatenate(features_list, axis=0)
feat_labels = np.concatenate(labels_list, axis=0)

np.save(FEATURES_PATH,    features)
np.save(FEAT_LABELS_PATH, feat_labels)

print(f"\nFeatures shape: {features.shape}  (samples x 64 feature dims)")
print(f"Saved to: {FEATURES_PATH}")


# ─────────────────────────────────────────────
# PLOT 1: Training and validation curves
# ─────────────────────────────────────────────
epochs_range = range(1, len(train_losses) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs_range, train_losses, label="Train loss",      color="#4A90D9", lw=1.5)
ax1.plot(epochs_range, val_losses,   label="Validation loss", color="#E8593C", lw=1.5)
ax1.axvline(best_epoch, color="gray", lw=1, linestyle="--", alpha=0.7, label=f"Best epoch ({best_epoch})")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.set_title("Training vs Validation Loss", fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.2)

ax2.plot(epochs_range, [a*100 for a in train_accs], label="Train accuracy",      color="#4A90D9", lw=1.5)
ax2.plot(epochs_range, [a*100 for a in val_accs],   label="Validation accuracy", color="#E8593C", lw=1.5)
ax2.axvline(best_epoch, color="gray", lw=1, linestyle="--", alpha=0.7)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Training vs Validation Accuracy", fontweight="bold")
ax2.legend()
ax2.grid(True, alpha=0.2)
ax2.set_ylim(0, 105)

plt.suptitle(f"CNN Training Curves  --  Best val accuracy: {max(val_accs)*100:.1f}%",
             fontsize=13, fontweight="bold")
plt.tight_layout()
curves_path = os.path.join(NOTEBOOKS, "training_curves.png")
plt.savefig(curves_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nTraining curves saved to:\n  {curves_path}")


# ─────────────────────────────────────────────
# PLOT 2: Confusion matrix
# ─────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax)

ax.set_xticks(range(len(CLASS_NAMES)))
ax.set_yticks(range(len(CLASS_NAMES)))
ax.set_xticklabels([c.replace("_", "\n") for c in CLASS_NAMES], fontsize=10)
ax.set_yticklabels([c.replace("_", " ") for c in CLASS_NAMES], fontsize=10)
ax.set_xlabel("Predicted Class", fontsize=12)
ax.set_ylabel("True Class",      fontsize=12)
ax.set_title(f"Confusion Matrix -- Test Accuracy: {test_acc*100:.1f}%",
             fontsize=13, fontweight="bold")

# Write counts inside each cell
thresh = cm.max() / 2.0
for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
cm_path = os.path.join(NOTEBOOKS, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Confusion matrix saved to:\n  {cm_path}")

print("\n" + "=" * 60)
print("STEP 1 COMPLETE")
print("=" * 60)
print(f"  Test accuracy:    {test_acc*100:.1f}%")
print(f"  Features shape:   {features.shape}")
print(f"  Model saved:      {MODEL_PATH}")
print(f"  Features saved:   {FEATURES_PATH}")
print("\nNext: Run  11_visualise_features.py")
