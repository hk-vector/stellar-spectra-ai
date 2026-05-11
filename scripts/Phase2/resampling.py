"""
=============================================================
PHASE 2 — STEP 4: Resampling to a Uniform Wavelength Grid
=============================================================

WHAT IS RESAMPLING AND WHY IS IT THE FINAL STEP?
─────────────────────────────────────────────────
After the first three steps your spectra are clean and
normalised — but they still have a critical problem:
every spectrum has a DIFFERENT number of data points.

Why? Because SDSS uses different spectrograph configurations
across its 20+ years of observations. A spectrum from 2002
might have 3,850 pixels. One from 2018 might have 4,632.
After redshift correction, the wavelength axes are also at
slightly different positions for each object.

Your neural network (CNN or Transformer) needs every single
input to be EXACTLY the same length. You cannot pass arrays
of different sizes into a model — it is mathematically
impossible to batch them together for training.

Resampling solves this by interpolating every spectrum onto
a single fixed wavelength grid that you define.

THE WAVELENGTH RANGE WE USE:
─────────────────────────────
We use 3800Å to 9200Å — the core optical range covered by
SDSS spectroscopy. Key features in this range:

    3934Å  — Ca II K  (strong in cool stars and white dwarfs)
    4861Å  — Hβ       (hydrogen Balmer series)
    5175Å  — Mg I b   (magnesium, strong in giants)
    5893Å  — Na I D   (sodium)
    6563Å  — Hα       (strongest hydrogen line)
    6717Å  — [S II]   (ionised sulphur, quasar indicator)

We use 3000 points across this range. That gives one point
every ~1.8Å — enough resolution to clearly resolve all
important absorption and emission features.

HOW INTERPOLATION WORKS:
─────────────────────────
Each spectrum's flux values are defined at its own original
wavelength positions. We use scipy's interp1d to fit a
cubic spline through those known (wavelength, flux) pairs,
then evaluate the spline at each of our 3000 target positions.

For wavelength positions outside the original spectrum's
range, we use the boundary value (extrapolation=False) to
avoid inventing data that was never observed.

WHAT THIS SCRIPT DOES:
    1. Defines the fixed target grid (3800–9200Å, 3000 points)
    2. Loads each normalised .npz from step3_normalised/
    3. Interpolates the flux onto the target grid
    4. Saves a fixed-length array to /data/processed/step4_resampled/
    5. Creates the FINAL dataset file: X.npy and y.npy
       X = all spectra stacked as a 2D array (n_samples × 3000)
       y = integer class labels (0, 1, 2, 3)
    6. Updates master_catalog.csv with 'resampled_filepath' column
    7. Saves a label_map.json so you always know which number
       maps to which class name

HOW TO RUN:
    1. Make sure 08_normalisation.py has finished
    2. Run:
           python 09_resampling.py

OUTPUT FILES:
    - /data/processed/step4_resampled/{label}/spec-XXXX.npz
    - /data/processed/X.npy      ← ALL spectra as 2D array
    - /data/processed/y.npy      ← ALL labels as 1D integer array
    - /data/processed/label_map.json  ← maps integer → class name
    - /notebooks/resampled_grid_comparison.png
    - /notebooks/final_dataset_summary.png
    - master_catalog.csv updated with 'resampled_filepath' column

REQUIRES:
    pip install numpy pandas matplotlib scipy tqdm
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
from scipy.interpolate import interp1d
from tqdm import tqdm

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR      = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "stellar-spectra-ai")
CATALOG_FILE  = os.path.join(BASE_DIR, "data", "catalog", "master_catalog.csv")
IN_DIR        = os.path.join(BASE_DIR, "data", "processed", "step3_normalised")
OUT_DIR       = os.path.join(BASE_DIR, "data", "processed", "step4_resampled")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
NOTEBOOKS     = os.path.join(BASE_DIR, "notebooks")
LOGS          = os.path.join(BASE_DIR, "logs")

os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(NOTEBOOKS, exist_ok=True)

CLASS_COLORS = {
    "white_dwarf":   "#4A90D9",
    "quasar":        "#E8593C",
    "main_sequence": "#3BAD75",
    "red_giant":     "#D4A017",
}

# ─────────────────────────────────────────────
# TARGET GRID DEFINITION
# This is the fixed wavelength grid every spectrum
# will be resampled onto. Change these values only
# if you have a strong reason (e.g. adding UV data).
# ─────────────────────────────────────────────
WAVE_MIN    = 3800.0    # Angstroms — blue end of SDSS optical range
WAVE_MAX    = 9200.0    # Angstroms — red end of SDSS optical range
N_POINTS    = 3000      # number of evenly spaced wavelength bins

TARGET_GRID = np.linspace(WAVE_MIN, WAVE_MAX, N_POINTS, dtype=np.float32)

# ─────────────────────────────────────────────
# LABEL MAPPING
# Maps class name strings to integer indices.
# The model outputs a number — this tells you
# what stellar class that number means.
# ─────────────────────────────────────────────
LABEL_MAP = {
    "white_dwarf":   0,
    "quasar":        1,
    "main_sequence": 2,
    "red_giant":     3,
}
# Reverse map: integer → class name (used for display)
REVERSE_MAP = {v: k for k, v in LABEL_MAP.items()}


# ─────────────────────────────────────────────
# HELPER: resample a single spectrum
# ─────────────────────────────────────────────
def resample_spectrum(wavelength, flux):
    """
    Interpolates the given (wavelength, flux) pair onto
    the fixed TARGET_GRID using cubic interpolation.

    For target wavelengths OUTSIDE the original spectrum's
    range, we use the nearest boundary value (fill_value=
    "extrapolate" is avoided — we never want to invent data).

    Args:
        wavelength (np.array): original wavelength axis
        flux (np.array): normalised flux values

    Returns:
        np.array: flux values at TARGET_GRID positions (length = N_POINTS)
    """
    # Sort by wavelength in case any spectra have reversed axes
    sort_idx   = np.argsort(wavelength)
    wavelength = wavelength[sort_idx]
    flux       = flux[sort_idx]

    # Remove duplicate wavelength values (causes interp1d to fail)
    _, unique_idx = np.unique(wavelength, return_index=True)
    wavelength    = wavelength[unique_idx]
    flux          = flux[unique_idx]

    if len(wavelength) < 4:
        # Too few points to interpolate — return flat array
        return np.ones(N_POINTS, dtype=np.float32)

    # Build interpolation function
    # bounds_error=False + fill_value=(boundary) means outside the
    # observed range we hold the edge value constant rather than guess
    lo_val = float(flux[0])
    hi_val = float(flux[-1])

    interp_fn = interp1d(
        wavelength,
        flux,
        kind="cubic",
        bounds_error=False,
        fill_value=(lo_val, hi_val)
    )

    resampled = interp_fn(TARGET_GRID).astype(np.float32)

    # Final safety clip — normalised flux should be in [0, 3]
    resampled = np.clip(resampled, 0.0, 3.0)

    return resampled


# ─────────────────────────────────────────────
# LOAD CATALOG
# ─────────────────────────────────────────────
print("=" * 60)
print("PHASE 2 — STEP 4: Resampling to Uniform Grid")
print("=" * 60)

if not os.path.exists(CATALOG_FILE):
    print(f"\nERROR: master_catalog.csv not found.")
    sys.exit(1)

master = pd.read_csv(CATALOG_FILE)
print(f"\nLoaded catalog: {len(master)} rows")
print(f"Target grid: {WAVE_MIN}–{WAVE_MAX} Å at {N_POINTS} points")
print(f"Resolution:  {(WAVE_MAX - WAVE_MIN) / N_POINTS:.2f} Å per pixel")

if "normalised_filepath" not in master.columns:
    print("\nERROR: 'normalised_filepath' column not found.")
    print("Run 08_normalisation.py first.")
    sys.exit(1)

for label in master["label"].unique():
    os.makedirs(os.path.join(OUT_DIR, label), exist_ok=True)


# ─────────────────────────────────────────────
# MAIN LOOP — resample every spectrum
# ─────────────────────────────────────────────
print("\nResampling all spectra to uniform grid...")
print("(Files already processed will be skipped)\n")

resampled_filepaths = []
failed              = []

for idx, row in tqdm(master.iterrows(), total=len(master), desc="Resampling"):
    in_path = row["normalised_filepath"]
    label   = row["label"]

    if pd.isna(in_path) or not os.path.exists(str(in_path)):
        resampled_filepaths.append(None)
        failed.append({"filepath": in_path, "error": "normalised file missing"})
        continue

    basename = os.path.basename(in_path)
    out_path = os.path.join(OUT_DIR, label, basename)

    if os.path.exists(out_path):
        resampled_filepaths.append(out_path)
        continue

    try:
        data       = np.load(in_path)
        wavelength = data["wavelength"]
        flux       = data["flux"]
    except Exception as e:
        resampled_filepaths.append(None)
        failed.append({"filepath": in_path, "error": str(e)})
        continue

    try:
        flux_resampled = resample_spectrum(wavelength, flux)
    except Exception as e:
        resampled_filepaths.append(None)
        failed.append({"filepath": in_path, "error": f"resampling failed: {e}"})
        continue

    np.savez_compressed(out_path, wavelength=TARGET_GRID, flux=flux_resampled)
    resampled_filepaths.append(out_path)

master["resampled_filepath"] = resampled_filepaths
master.to_csv(CATALOG_FILE, index=False)

if failed:
    pd.DataFrame(failed).to_csv(os.path.join(LOGS, "resampling_failed.csv"), index=False)
    print(f"\n  {len(failed)} files failed — see logs/resampling_failed.csv")

success = sum(1 for p in resampled_filepaths if p is not None)
print(f"\n  Resampled: {success} / {len(master)} spectra")


# ─────────────────────────────────────────────
# BUILD FINAL DATASET: X.npy and y.npy
#
# X.npy shape: (n_samples, N_POINTS) = (1782, 3000)
# y.npy shape: (n_samples,)          = (1782,)
#
# These are the files you will pass directly into
# your model training script in Phase 4.
# ─────────────────────────────────────────────
print("\nBuilding final X.npy and y.npy dataset arrays...")

valid_rows = master[master["resampled_filepath"].notna()].copy()

X_list = []
y_list = []
skipped = 0

for idx, row in tqdm(valid_rows.iterrows(), total=len(valid_rows), desc="Building dataset"):
    fp    = row["resampled_filepath"]
    label = row["label"]

    if label not in LABEL_MAP:
        skipped += 1
        continue

    try:
        data = np.load(fp)
        flux = data["flux"]

        if len(flux) != N_POINTS:
            skipped += 1
            continue

        X_list.append(flux)
        y_list.append(LABEL_MAP[label])

    except Exception:
        skipped += 1
        continue

X = np.stack(X_list, axis=0)   # shape: (n_samples, 3000)
y = np.array(y_list, dtype=np.int64)   # shape: (n_samples,)

# Save
X_path = os.path.join(PROCESSED_DIR, "X.npy")
y_path = os.path.join(PROCESSED_DIR, "y.npy")
np.save(X_path, X)
np.save(y_path, y)

# Save label map as JSON
label_map_path = os.path.join(PROCESSED_DIR, "label_map.json")
with open(label_map_path, "w") as f:
    json.dump({"label_to_int": LABEL_MAP, "int_to_label": REVERSE_MAP}, f, indent=2)

print(f"\n  X.npy shape: {X.shape}  (samples × wavelength points)")
print(f"  y.npy shape: {y.shape}  (integer class labels)")
print(f"  Skipped:     {skipped} spectra (missing files or unknown labels)")
print(f"\n  Saved:")
print(f"    {X_path}")
print(f"    {y_path}")
print(f"    {label_map_path}")

# Print class counts in the final array
print("\n  Class counts in final dataset:")
for label, idx_val in LABEL_MAP.items():
    count = int((y == idx_val).sum())
    bar   = "█" * (count // 10)
    print(f"    {label:<22} (label {idx_val})  {count:>5}  {bar}")


# ─────────────────────────────────────────────
# PLOT 1: overlaid resampled spectra per class
# Shows all spectra aligned on the same grid
# ─────────────────────────────────────────────
print("\nGenerating plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for plot_idx, (cls, label_int) in enumerate(LABEL_MAP.items()):
    ax    = axes[plot_idx]
    color = CLASS_COLORS.get(cls, "#555555")

    # Get indices of this class in X
    class_indices = np.where(y == label_int)[0]
    sample_indices = class_indices[:min(50, len(class_indices))]

    for si in sample_indices:
        ax.plot(TARGET_GRID, X[si], lw=0.3, color=color, alpha=0.2)

    # Plot the mean spectrum for this class
    mean_flux = X[class_indices].mean(axis=0)
    ax.plot(TARGET_GRID, mean_flux, lw=1.5, color="black", alpha=0.9, label="Mean spectrum")

    # Mark key absorption lines
    key_lines = {"Hα": 6563, "Hβ": 4861, "Ca K": 3934, "Na D": 5893}
    for name, wl in key_lines.items():
        if WAVE_MIN < wl < WAVE_MAX:
            ax.axvline(wl, color="gray", lw=0.8, linestyle="--", alpha=0.5)
            ax.text(wl + 20, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1.0,
                    name, fontsize=7, color="gray", va="top")

    ax.set_title(f"{cls.replace('_', ' ').title()}  (n={len(class_indices)})",
                 fontsize=12, fontweight="bold", color=color)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Normalised Flux")
    ax.set_ylim(0, 2.0)
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=8)

plt.suptitle("Resampled Spectra by Class (faint = individual, bold = mean)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plot1_path = os.path.join(NOTEBOOKS, "resampled_grid_comparison.png")
plt.savefig(plot1_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Spectra overlay plot saved to:\n    {plot1_path}")


# ─────────────────────────────────────────────
# PLOT 2: final dataset summary
# ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: class distribution bar chart
labels_list = [REVERSE_MAP[i] for i in range(len(LABEL_MAP))]
counts      = [(y == i).sum() for i in range(len(LABEL_MAP))]
colors_list = [CLASS_COLORS.get(l, "#888") for l in labels_list]

bars = ax1.bar(labels_list, counts, color=colors_list, edgecolor="white")
ax1.set_title("Final Dataset — Class Distribution", fontsize=12, fontweight="bold")
ax1.set_ylabel("Number of Samples")
for bar, count in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
             str(count), ha="center", va="bottom", fontsize=11, fontweight="bold")
ax1.set_ylim(0, max(counts) * 1.15)
ax1.grid(True, alpha=0.2, axis="y")

# Right: mean spectra of all classes overlaid
for cls, label_int in LABEL_MAP.items():
    class_indices = np.where(y == label_int)[0]
    if len(class_indices) == 0:
        continue
    mean_flux = X[class_indices].mean(axis=0)
    color     = CLASS_COLORS.get(cls, "#555555")
    ax2.plot(TARGET_GRID, mean_flux, lw=1.2, color=color,
             label=cls.replace("_", " ").title(), alpha=0.9)

ax2.set_title("Mean Spectrum per Class", fontsize=12, fontweight="bold")
ax2.set_xlabel("Wavelength (Å)")
ax2.set_ylabel("Normalised Flux")
ax2.legend(fontsize=9)
ax2.set_ylim(0, 1.8)
ax2.grid(True, alpha=0.2)

plt.suptitle(f"Phase 2 Complete — {len(X)} samples × {N_POINTS} wavelength points",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plot2_path = os.path.join(NOTEBOOKS, "final_dataset_summary.png")
plt.savefig(plot2_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Dataset summary plot saved to:\n    {plot2_path}")


# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 2 COMPLETE — ALL PREPROCESSING DONE")
print("=" * 60)
print()
print("  Pipeline completed:")
print("    Step 1  Redshift correction      06_redshift_correction.py")
print("    Step 2  Noise removal            07_noise_removal.py")
print("    Step 3  Continuum normalisation  08_normalisation.py")
print("    Step 4  Resampling               09_resampling.py  ← you are here")
print()
print("  Final dataset:")
print(f"    X.npy  shape: {X.shape}  — the spectra (input to your model)")
print(f"    y.npy  shape: {y.shape}  — the labels  (what the model predicts)")
print()
print("  How to load these in your training script (Phase 3/4):")
print("    import numpy as np")
print("    X = np.load('data/processed/X.npy')  # shape (n, 3000)")
print("    y = np.load('data/processed/y.npy')  # shape (n,)")
print()
print("  Label mapping (saved in data/processed/label_map.json):")
for cls, idx_val in LABEL_MAP.items():
    print(f"    {idx_val} = {cls}")
print()
print("  Next phase: Phase 3 — Feature Extraction")
print("  Or jump to: Phase 4 — Model Training (CNN on X.npy and y.npy)")
print()
