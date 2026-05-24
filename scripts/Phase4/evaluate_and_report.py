# -*- coding: utf-8 -*-
"""
=============================================================
PHASE 4 - STEP 2: Full Evaluation and Final Report
=============================================================

WHAT THIS SCRIPT DOES:
-----------------------
This is the final evaluation script for the classification
pipeline. It loads the best trained model and runs a
thorough analysis that goes beyond simple accuracy.

WHY ACCURACY ALONE IS NOT ENOUGH:
-----------------------------------
Imagine a dataset with 90% quasars and 10% white dwarfs.
A model that predicts "quasar" for everything would get
90% accuracy but be completely useless for finding white dwarfs.

We need metrics that reveal per-class performance:

PRECISION: Of all spectra the model CALLED class X,
           what fraction were actually class X?
           High precision = few false alarms.

RECALL:    Of all spectra that ARE class X,
           what fraction did the model FIND?
           High recall = few missed detections.

F1 SCORE:  Harmonic mean of precision and recall.
           Balances both -- a single number per class.
           F1=1.0 is perfect, F1=0.0 is useless.

ROC-AUC:   Area Under the ROC Curve. Measures how well
           the model separates a class from all others
           regardless of decision threshold. AUC=1.0 is
           perfect, AUC=0.5 is random guessing.

CALIBRATION: Are the model's confidence scores trustworthy?
             If the model says "80% sure this is a quasar",
             is it right ~80% of the time? A well-calibrated
             model's probabilities match its actual accuracy.

WHAT THIS SCRIPT PRODUCES:
    1. Full classification report (precision/recall/F1 per class)
    2. Confusion matrix with percentage labels
    3. ROC curves (one per class, one-vs-rest)
    4. Confidence distribution -- how sure is the model
       about correct vs incorrect predictions?
    5. Calibration plot -- are confidence scores trustworthy?
    6. Worst predictions -- which spectra confused the model
       most badly? (useful for finding labelling errors)
    7. A final summary printed to terminal

HOW TO RUN:
    Run: python evaluate_and_report.py

OUTPUT FILES:
    - /notebooks/roc_curves.png
    - /notebooks/confidence_distribution.png
    - /notebooks/calibration_plot.png
    - /notebooks/worst_predictions.png
    - /notebooks/final_report.txt   (plain text summary)

REQUIRES:
    pip install torch scikit-learn numpy matplotlib tqdm
=============================================================
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, f1_score,
    precision_score, recall_score
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR      = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "stellar-spectra-ai")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
NOTEBOOKS     = os.path.join(BASE_DIR, "notebooks")

X_PATH         = os.path.join(PROCESSED_DIR, "X.npy")
Y_PATH         = os.path.join(PROCESSED_DIR, "y.npy")
LABEL_MAP_PATH = os.path.join(PROCESSED_DIR, "label_map.json")
MODEL_PATH     = os.path.join(MODELS_DIR,    "cnn_v2.pth")
REPORT_PATH    = os.path.join(NOTEBOOKS,     "final_report.txt")

RANDOM_SEED = 42

CLASS_COLORS = {
    0: "#4A90D9",
    1: "#E8593C",
    2: "#3BAD75",
    3: "#D4A017",
}


# ─────────────────────────────────────────────
# REBUILD MODEL ARCHITECTURE
# Must match 12_improved_training.py exactly
# ─────────────────────────────────────────────
class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ResidualBlock1D, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class ImprovedStellarCNN(nn.Module):
    def __init__(self, input_length=3000, n_classes=4):
        super(ImprovedStellarCNN, self).__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.res_block_1 = ResidualBlock1D(64,  kernel_size=7)
        self.down_1      = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.res_block_2 = ResidualBlock1D(128, kernel_size=5)
        self.down_2      = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.res_block_3     = ResidualBlock1D(256, kernel_size=3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_layers  = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.3),
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


# ─────────────────────────────────────────────
# LOAD DATA AND MODEL
# ─────────────────────────────────────────────
print("=" * 60)
print("PHASE 4 - STEP 2: Full Evaluation and Report")
print("=" * 60)

for path, name in [(X_PATH, "X.npy"), (Y_PATH, "y.npy"), (MODEL_PATH, "cnn_v2.pth")]:
    if not os.path.exists(path):
        print(f"\nERROR: {name} not found.")
        if name == "cnn_v2.pth":
            print("Run improved_training.py first.")
        sys.exit(1)

X = np.load(X_PATH)
y = np.load(Y_PATH)
X = np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=0.0)

with open(LABEL_MAP_PATH) as f:
    label_map  = json.load(f)
int_to_label   = label_map["int_to_label"]
CLASS_NAMES    = [int_to_label[str(i)] for i in range(len(int_to_label))]
N_CLASSES      = len(CLASS_NAMES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = ImprovedStellarCNN(input_length=X.shape[1], n_classes=N_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"\nModel loaded from: {MODEL_PATH}")

# Use the test split (same seed as training scripts)
from sklearn.model_selection import train_test_split
_, X_temp, _, y_temp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y)
X_test, _, y_test, _ = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp)

print(f"Test set: {len(X_test)} samples")


# ─────────────────────────────────────────────
# RUN INFERENCE ON TEST SET
# ─────────────────────────────────────────────
test_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_test, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_test, dtype=torch.long)
    ),
    batch_size=64, shuffle=False
)

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
print(f"\nTest accuracy: {test_acc*100:.1f}%")


# ─────────────────────────────────────────────
# PLOT 1: Confusion matrix with percentages
# ─────────────────────────────────────────────
cm      = confusion_matrix(all_labels, all_preds)
cm_pct  = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for ax, data, fmt, title in [
    (ax1, cm,     "d",    "Counts"),
    (ax2, cm_pct, ".1f",  "Row %  (what % of true class X was predicted as Y)"),
]:
    im = ax.imshow(data, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels([c.replace("_", "\n") for c in CLASS_NAMES], fontsize=9)
    ax.set_yticklabels([c.replace("_", " ")  for c in CLASS_NAMES], fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    thresh = data.max() / 2.0
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            val = f"{data[i,j]:{fmt}}"
            if fmt == ".1f":
                val += "%"
            ax.text(j, i, val, ha="center", va="center", fontsize=10,
                    fontweight="bold",
                    color="white" if data[i, j] > thresh else "black")

plt.suptitle(f"Confusion Matrix  --  Test Accuracy: {test_acc*100:.1f}%",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(NOTEBOOKS, "confusion_matrix_v2.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Confusion matrix saved.")


# ─────────────────────────────────────────────
# PLOT 2: ROC curves (one per class)
# ─────────────────────────────────────────────
y_bin = label_binarize(all_labels, classes=list(range(N_CLASSES)))

fig, ax = plt.subplots(figsize=(9, 7))
auc_scores = {}

for i, name in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
    roc_auc     = auc(fpr, tpr)
    auc_scores[name] = roc_auc
    ax.plot(fpr, tpr, lw=2, color=CLASS_COLORS[i],
            label=f"{name.replace('_',' ').title()}  AUC={roc_auc:.3f}")

ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.5)")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate",  fontsize=12)
ax.set_title("ROC Curves  --  One vs Rest per Class\nAUC=1.0 is perfect, AUC=0.5 is random",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(NOTEBOOKS, "roc_curves.png"), dpi=150, bbox_inches="tight")
plt.close()
print("ROC curves saved.")


# ─────────────────────────────────────────────
# PLOT 3: Confidence distribution
# How confident is the model on correct vs wrong predictions
# ─────────────────────────────────────────────
confidence      = all_probs.max(axis=1)   # highest probability for each prediction
correct_mask    = all_preds == all_labels
conf_correct    = confidence[correct_mask]
conf_wrong      = confidence[~correct_mask]

fig, ax = plt.subplots(figsize=(10, 5))
bins = np.linspace(0, 1, 25)
ax.hist(conf_correct, bins=bins, alpha=0.7, color="#3BAD75", label=f"Correct ({correct_mask.sum()})")
ax.hist(conf_wrong,   bins=bins, alpha=0.7, color="#E8593C", label=f"Wrong   ({(~correct_mask).sum()})")
ax.set_xlabel("Model Confidence (max softmax probability)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Confidence Distribution: Correct vs Wrong Predictions\n"
             "Good model: correct predictions are high confidence, wrong are low confidence",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.2)
ax.axvline(0.5, color="gray", lw=1, linestyle="--", alpha=0.7, label="50% threshold")

plt.tight_layout()
plt.savefig(os.path.join(NOTEBOOKS, "confidence_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Confidence distribution saved.")


# ─────────────────────────────────────────────
# PLOT 4: Calibration plot
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")

for i, name in enumerate(CLASS_NAMES):
    prob_true, prob_pred = calibration_curve(
        (all_labels == i).astype(int),
        all_probs[:, i],
        n_bins=10
    )
    ax.plot(prob_pred, prob_true, lw=2, marker="o", markersize=5,
            color=CLASS_COLORS[i],
            label=name.replace("_", " ").title())

ax.set_xlabel("Mean Predicted Probability", fontsize=12)
ax.set_ylabel("Fraction of True Positives", fontsize=12)
ax.set_title("Calibration Plot\n"
             "Points on the diagonal = model confidence matches actual accuracy",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(NOTEBOOKS, "calibration_plot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Calibration plot saved.")


# ─────────────────────────────────────────────
# PLOT 5: Worst predictions
# Shows the 8 spectra the model was most wrong about
# ─────────────────────────────────────────────
wrong_indices   = np.where(~correct_mask)[0]
wrong_confs     = confidence[wrong_indices]
worst_indices   = wrong_indices[np.argsort(wrong_confs)[::-1][:8]]

WAVE_GRID = np.linspace(3800, 9200, 3000)

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for plot_idx, test_idx in enumerate(worst_indices):
    ax      = axes[plot_idx]
    flux    = X_test[test_idx]
    true_l  = CLASS_NAMES[all_labels[test_idx]].replace("_", " ")
    pred_l  = CLASS_NAMES[all_preds[test_idx]].replace("_", " ")
    conf    = confidence[test_idx]

    true_color = CLASS_COLORS[all_labels[test_idx]]
    pred_color = CLASS_COLORS[all_preds[test_idx]]

    ax.plot(WAVE_GRID, flux, lw=0.6, color="gray", alpha=0.8)
    ax.set_title(
        f"True: {true_l}\nPred: {pred_l}  ({conf*100:.0f}% conf)",
        fontsize=9, fontweight="bold",
        color=pred_color
    )
    ax.set_xlabel("Wavelength (A)", fontsize=8)
    ax.set_ylabel("Norm. Flux",     fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.15)

    for line_wl in [3934, 4861, 5893, 6563]:
        if 3800 < line_wl < 9200:
            ax.axvline(line_wl, color="lightgray", lw=0.5, linestyle="--")

plt.suptitle("8 Most Confidently Wrong Predictions\n"
             "(high model confidence but incorrect -- reveals hard cases and possible label errors)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(NOTEBOOKS, "worst_predictions.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Worst predictions plot saved.")


# ─────────────────────────────────────────────
# GENERATE PLAIN TEXT REPORT
# ─────────────────────────────────────────────
report_lines = []
report_lines.append("=" * 60)
report_lines.append("STELLAR SPECTRA AI -- PHASE 4 FINAL REPORT")
report_lines.append("=" * 60)
report_lines.append("")
report_lines.append(f"Test accuracy:  {test_acc*100:.1f}%")
report_lines.append(f"Test samples:   {len(all_labels)}")
report_lines.append(f"Classes:        {', '.join(CLASS_NAMES)}")
report_lines.append("")
report_lines.append("PER-CLASS METRICS:")
report_lines.append("-" * 40)
report_lines.append(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
report_lines.append("ROC-AUC SCORES:")
report_lines.append("-" * 40)
for name, score in auc_scores.items():
    bar = "█" * int(score * 20)
    report_lines.append(f"  {name:<22}  AUC={score:.3f}  {bar}")
report_lines.append("")
report_lines.append("CONFIDENCE ANALYSIS:")
report_lines.append("-" * 40)
report_lines.append(f"  Mean confidence (correct):   {conf_correct.mean()*100:.1f}%")
report_lines.append(f"  Mean confidence (incorrect): {conf_wrong.mean()*100:.1f}%  (lower = better)")
report_lines.append("")
report_lines.append("CONFUSION MATRIX (row=true, col=predicted):")
report_lines.append("-" * 40)
for i, row_name in enumerate(CLASS_NAMES):
    row_str = "  ".join([f"{cm[i,j]:>5}" for j in range(N_CLASSES)])
    report_lines.append(f"  {row_name:<22}  {row_str}")
report_lines.append("")
report_lines.append("HEADER: " + "  ".join([f"{c[:12]:>12}" for c in CLASS_NAMES]))
report_lines.append("")

# Go/no-go for Phase 5
macro_f1 = f1_score(all_labels, all_preds, average="macro")
min_auc  = min(auc_scores.values())

report_lines.append("")
report_lines.append("=" * 60)

report_text = "\n".join(report_lines)
print("\n" + report_text)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"\nReport saved to:\n  {REPORT_PATH}")

print("\n" + "=" * 60)
print("PHASE 4 COMPLETE")
print("=" * 60)
print("\nAll output files saved to /notebooks/:")
print("  confusion_matrix_v2.png")
print("  roc_curves.png")
print("  confidence_distribution.png")
print("  calibration_plot.png")
print("  worst_predictions.png")
print("  final_report.txt")
print("\nNext: commit to GitHub and proceed to web_interface.py")