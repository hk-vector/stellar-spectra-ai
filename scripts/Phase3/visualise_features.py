import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from tqdm import tqdm

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR       = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "stellar-spectra-ai")
PROCESSED_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
NOTEBOOKS      = os.path.join(BASE_DIR, "notebooks")

FEATURES_PATH      = os.path.join(PROCESSED_DIR, "features.npy")
FEAT_LABELS_PATH   = os.path.join(PROCESSED_DIR, "features_labels.npy")
LABEL_MAP_PATH     = os.path.join(PROCESSED_DIR, "label_map.json")
MODEL_PATH         = os.path.join(MODELS_DIR,    "cnn_feature_extractor.pth")
X_PATH             = os.path.join(PROCESSED_DIR, "X.npy")

CLASS_COLORS = {
    0: "#4A90D9",   # white_dwarf
    1: "#E8593C",   # quasar
    2: "#3BAD75",   # main_sequence
    3: "#D4A017",   # red_giant
}

# Key absorption/emission lines to mark on activation maps
KEY_LINES = {
    "Ca K":  3934,
    "Ca H":  3968,
    "Hd":    4102,
    "Hg":    4341,
    "Hb":    4861,
    "Mg b":  5175,
    "Na D":  5893,
    "Ha":    6563,
    "Ca IR": 8542,
}

WAVE_MIN   = 3800.0
WAVE_MAX   = 9200.0
N_POINTS   = 3000
TARGET_GRID = np.linspace(WAVE_MIN, WAVE_MAX, N_POINTS, dtype=np.float32)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("PHASE 3 - STEP 2: Visualise Learned Features")
print("=" * 60)

for path, name in [
    (FEATURES_PATH,    "features.npy"),
    (FEAT_LABELS_PATH, "features_labels.npy"),
    (LABEL_MAP_PATH,   "label_map.json"),
]:
    if not os.path.exists(path):
        print(f"\nERROR: {name} not found.")
        print("Run cnn_feature_extractor.py first.")
        sys.exit(1)

features = np.load(FEATURES_PATH)
labels   = np.load(FEAT_LABELS_PATH)

with open(LABEL_MAP_PATH) as f:
    label_map    = json.load(f)
int_to_label     = label_map["int_to_label"]
CLASS_NAMES      = [int_to_label[str(i)] for i in range(len(int_to_label))]
N_CLASSES        = len(CLASS_NAMES)

print(f"\nLoaded features: {features.shape}  (samples x 64 dims)")
print(f"Classes: {CLASS_NAMES}")

# Standardise features before PCA/t-SNE
# (zero mean, unit variance -- required for PCA to work correctly)
scaler   = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ─────────────────────────────────────────────
# VISUALISATION 1: PCA
# ─────────────────────────────────────────────
print("\nRunning PCA (64D -> 2D)...")

pca        = PCA(n_components=2, random_state=42)
features_2d_pca = pca.fit_transform(features_scaled)
explained  = pca.explained_variance_ratio_ * 100

fig, ax = plt.subplots(figsize=(10, 8))

for class_idx in range(N_CLASSES):
    mask  = labels == class_idx
    color = CLASS_COLORS[class_idx]
    name  = CLASS_NAMES[class_idx].replace("_", " ").title()
    ax.scatter(
        features_2d_pca[mask, 0],
        features_2d_pca[mask, 1],
        c=color, label=name,
        alpha=0.6, s=20, edgecolors="none"
    )

ax.set_xlabel(f"PC 1  ({explained[0]:.1f}% variance)", fontsize=12)
ax.set_ylabel(f"PC 2  ({explained[1]:.1f}% variance)", fontsize=12)
ax.set_title("PCA of CNN Features (64D -> 2D)\nWell-separated clusters = good feature learning",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11, markerscale=2)
ax.grid(True, alpha=0.2)

plt.tight_layout()
pca_path = os.path.join(NOTEBOOKS, "pca_features.png")
plt.savefig(pca_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  PCA plot saved to:\n    {pca_path}")
print(f"  Variance explained: PC1={explained[0]:.1f}%  PC2={explained[1]:.1f}%")

# ─────────────────────────────────────────────
# VISUALISATION 2: t-SNE
# ─────────────────────────────────────────────
print("\nRunning t-SNE (64D -> 2D) -- this takes 1-2 minutes...")

tsne = TSNE(
    n_components=2,
    perplexity=40,
    max_iter=1000,
    random_state=42,
    verbose=0
)
features_2d_tsne = tsne.fit_transform(features_scaled)

fig, ax = plt.subplots(figsize=(10, 8))

for class_idx in range(N_CLASSES):
    mask  = labels == class_idx
    color = CLASS_COLORS[class_idx]
    name  = CLASS_NAMES[class_idx].replace("_", " ").title()
    ax.scatter(
        features_2d_tsne[mask, 0],
        features_2d_tsne[mask, 1],
        c=color, label=name,
        alpha=0.6, s=20, edgecolors="none"
    )

ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
ax.set_title("t-SNE of CNN Features (64D -> 2D)\nTight separate clusters = excellent class discrimination",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11, markerscale=2)
ax.grid(True, alpha=0.2)

plt.tight_layout()
tsne_path = os.path.join(NOTEBOOKS, "tsne_features.png")
plt.savefig(tsne_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  t-SNE plot saved to:\n    {tsne_path}")

# ─────────────────────────────────────────────
# VISUALISATION 3: Per-class mean feature heatmap
#
# Shows the average value of each of the 64 features
# for each class. Classes with different patterns in
# this heatmap are well-separated in feature space.
# ─────────────────────────────────────────────
print("\nGenerating feature heatmap...")

mean_features = np.zeros((N_CLASSES, features.shape[1]))
for class_idx in range(N_CLASSES):
    mask = labels == class_idx
    mean_features[class_idx] = features[mask].mean(axis=0)

fig, ax = plt.subplots(figsize=(16, 4))
im = ax.imshow(mean_features, aspect="auto", cmap="RdBu_r", interpolation="nearest")
plt.colorbar(im, ax=ax, label="Mean Feature Activation")

ax.set_yticks(range(N_CLASSES))
ax.set_yticklabels([c.replace("_", " ").title() for c in CLASS_NAMES], fontsize=11)
ax.set_xlabel("Feature Dimension (0-63)", fontsize=12)
ax.set_title(
    "Per-Class Mean CNN Features\n"
    "Different row patterns = CNN found discriminative features. "
    "Identical rows = poor learning.",
    fontsize=12, fontweight="bold"
)
ax.grid(False)

plt.tight_layout()
heatmap_path = os.path.join(NOTEBOOKS, "feature_heatmap.png")
plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Feature heatmap saved to:\n    {heatmap_path}")

# ─────────────────────────────────────────────
# VISUALISATION 4: Activation maps
#
# Shows WHICH WAVELENGTHS the CNN focuses on for
# each class. We compute this by:
#   1. Taking the mean spectrum of each class
#   2. Passing it through the conv blocks
#   3. Computing the mean absolute activation
#      at each position across all filters
#   4. Upsampling back to 3000 points
#   5. Overlaying on the mean spectrum
# ─────────────────────────────────────────────
print("\nGenerating activation maps...")

if not os.path.exists(X_PATH):
    print("  WARNING: X.npy not found -- skipping activation maps.")
else:
    # Import the CNN class definition inline
    # (same architecture as in 10_cnn_feature_extractor.py)
    DROPOUT_1 = 0.4
    DROPOUT_2 = 0.3

    class StellarCNN(nn.Module):
        def __init__(self, input_length=3000, n_classes=4):
            super(StellarCNN, self).__init__()
            self.conv_blocks = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=11, padding=5),
                nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256), nn.ReLU(),
            )
            self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
            self.feature_layers = nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(DROPOUT_1),
                nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(DROPOUT_2),
            )
            self.classifier = nn.Linear(64, n_classes)

        def forward(self, x):
            x = self.conv_blocks(x)
            x = self.global_avg_pool(x)
            x = x.squeeze(-1)
            x = self.feature_layers(x)
            return self.classifier(x)

        def get_conv_activations(self, x):
            """Returns raw conv block output before global pooling."""
            return self.conv_blocks(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = StellarCNN(input_length=N_POINTS, n_classes=N_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    X_all = np.load(X_PATH)

    fig, axes = plt.subplots(N_CLASSES, 1, figsize=(14, 3.5 * N_CLASSES))

    for class_idx in range(N_CLASSES):
        ax    = axes[class_idx]
        color = CLASS_COLORS[class_idx]
        name  = CLASS_NAMES[class_idx].replace("_", " ").title()

        # Get all spectra of this class
        class_mask    = labels == class_idx
        class_spectra = X_all[class_mask]

        if len(class_spectra) == 0:
            continue

        # Compute mean spectrum for display
        mean_spectrum = class_spectra.mean(axis=0)

        # Compute activations for a batch of this class
        sample_size = min(50, len(class_spectra))
        sample      = class_spectra[:sample_size]
        tensor      = torch.tensor(sample, dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            # Get conv output: shape (batch, 256, L_reduced)
            conv_out = model.get_conv_activations(tensor)
            # Mean absolute activation across filters and batch
            # shape: (L_reduced,)
            activation_map = conv_out.abs().mean(dim=[0, 1]).cpu().numpy()

        # Upsample activation map back to N_POINTS using interpolation
        from scipy.interpolate import interp1d as scipy_interp1d
        act_x          = np.linspace(0, N_POINTS - 1, len(activation_map))
        full_x         = np.arange(N_POINTS)
        upsample_fn    = scipy_interp1d(act_x, activation_map, kind="linear",
                                         bounds_error=False, fill_value=0.0)
        activation_full = upsample_fn(full_x)

        # Normalise activation to 0-1 for overlay
        act_norm = (activation_full - activation_full.min())
        if act_norm.max() > 0:
            act_norm = act_norm / act_norm.max()

        # Plot mean spectrum
        ax2 = ax.twinx()   # second y-axis for activation overlay
        ax.plot(TARGET_GRID, mean_spectrum, lw=0.8, color=color, alpha=0.9, label="Mean spectrum")
        ax.set_ylabel("Normalised Flux", fontsize=9)
        ax.set_ylim(0, 1.8)

        # Plot activation as filled area on second axis
        ax2.fill_between(TARGET_GRID, 0, act_norm, alpha=0.25, color="black", label="CNN attention")
        ax2.set_ylabel("Activation (normalised)", fontsize=9, color="gray")
        ax2.set_ylim(0, 3)
        ax2.tick_params(axis="y", colors="gray")

        # Mark known spectral lines
        for line_name, wl in KEY_LINES.items():
            if WAVE_MIN < wl < WAVE_MAX:
                ax.axvline(wl, color="gray", lw=0.7, linestyle=":", alpha=0.6)
                ax.text(wl + 15, 1.65, line_name, fontsize=7, color="gray",
                        va="top", rotation=90)

        ax.set_title(f"{name}  --  CNN Activation Map (shaded = where CNN looks)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Wavelength (Angstroms)")
        ax.grid(True, alpha=0.15)

    plt.suptitle("CNN Activation Maps by Class\nPeaks near spectral lines = physically meaningful learning",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    act_path = os.path.join(NOTEBOOKS, "activation_maps.png")
    plt.savefig(act_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Activation maps saved to:\n    {act_path}")

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print()
print("=" * 60)
print("PHASE 3 COMPLETE")
print("=" * 60)
print()
print("  What you have now:")
print("    features.npy        -- 64-dim CNN features per spectrum")
print("    features_labels.npy -- corresponding class labels")
print("    4 visualisation plots in /notebooks/")
print()
print("  Next: Phase 4 -- Model Training")
print("  Your CNN is already trained (saved in /models/).")
print("  Phase 4 will fine-tune it and build the full")
print("  classification + evaluation pipeline.")