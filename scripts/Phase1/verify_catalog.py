"""
=============================================================
PHASE 1 — STEP 4: Verify and Inspect Your Data
=============================================================

WHAT THIS SCRIPT DOES:
    1. Loads master_catalog.csv
    2. Opens one .fits spectrum from EACH class and plots it
    3. Prints the class distribution (how many samples per class)
    4. Detects any corrupted .fits files that fail to open
    5. Removes corrupted entries from the catalog and saves a clean version
    6. Saves all plots as PNG images into /notebooks/ for your reference

HOW TO RUN:
    1. Make sure you have already run build_catalog.py (Step 3)
    2. Open your terminal
    3. Navigate to your scripts folder:
           cd Desktop/stellar-spectra-ai/scripts
    4. Run:
           python 04_verify_catalog.py

OUTPUT FILES:
    - /notebooks/spectra_samples.png     — plot of one spectrum per class
    - /notebooks/class_distribution.png  — bar chart of sample counts
    - /data/catalog/master_catalog.csv   — updated with bad rows removed
    - /logs/bad_files.txt                — list of any corrupted .fits files

REQUIRES:
    pip install astropy pandas matplotlib numpy tqdm
=============================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — saves to file instead of popup
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm

# ─────────────────────────────────────────────
# PATHS — adjust only if your folder layout differs
# ─────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "stellar-spectra-ai")
CATALOG_FILE = os.path.join(BASE_DIR, "data", "catalog", "master_catalog.csv")
NOTEBOOKS    = os.path.join(BASE_DIR, "notebooks")
LOGS         = os.path.join(BASE_DIR, "logs")
BAD_FILE     = os.path.join(LOGS, "bad_files.txt")

os.makedirs(NOTEBOOKS, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

# ─────────────────────────────────────────────
# COLOUR MAP — one colour per class for plots
# ─────────────────────────────────────────────
CLASS_COLORS = {
    "white_dwarf":   "#4A90D9",
    "quasar":        "#E8593C",
    "main_sequence": "#3BAD75",
    "red_giant":     "#D4A017",
}

# ─────────────────────────────────────────────
# HELPER: extract wavelength + flux from a .fits file
# ─────────────────────────────────────────────
def load_spectrum(filepath):
    """
    Opens an SDSS .fits file and returns (wavelength_array, flux_array).

    SDSS stores wavelengths as log10 values in the header (COEFF0, COEFF1).
    We convert them back to Angstroms using:
        wavelength = 10 ** (COEFF0 + COEFF1 * pixel_index)

    Returns:
        wavelength (np.array): wavelength in Angstroms
        flux       (np.array): flux values (arbitrary units)
    """
    with fits.open(filepath) as hdul:
        coadd  = hdul["COADD"]
        flux   = coadd.data["flux"]

        # Reconstruct wavelength axis from header coefficients
        header = coadd.header
        coeff0 = header.get("COEFF0", None)
        coeff1 = header.get("COEFF1", None)

        if coeff0 is not None and coeff1 is not None:
            npix       = len(flux)
            log_wave   = coeff0 + coeff1 * np.arange(npix)
            wavelength = 10 ** log_wave
        else:
            # Fallback: use pixel index as x-axis (rare but possible)
            wavelength = np.arange(len(flux))

    return wavelength, flux


# ─────────────────────────────────────────────
# STEP 4A — Load the master catalog
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 4: Verifying and Inspecting Your Data")
print("=" * 60)

if not os.path.exists(CATALOG_FILE):
    print(f"\nERROR: Could not find master_catalog.csv at:\n  {CATALOG_FILE}")
    print("Make sure you have run build_catalog.py first.")
    sys.exit(1)

master = pd.read_csv(CATALOG_FILE)
print(f"\nLoaded master catalog: {len(master)} total rows")
print(f"Classes found: {master['label'].unique().tolist()}")


# ─────────────────────────────────────────────
# STEP 4B — Print class distribution
# ─────────────────────────────────────────────
print("\n" + "─" * 40)
print("CLASS DISTRIBUTION")
print("─" * 40)
dist = master["label"].value_counts()
for label, count in dist.items():
    bar = "█" * (count // 10)      # simple ASCII bar
    print(f"  {label:<20} {count:>5}  {bar}")

# Warn if any class has fewer than 200 samples
for label, count in dist.items():
    if count < 200:
        print(f"\n  WARNING: '{label}' only has {count} samples.")
        print("  Consider downloading more rows from SDSS for this class.")


# ─────────────────────────────────────────────
# STEP 4C — Plot class distribution bar chart
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
colors  = [CLASS_COLORS.get(lbl, "#888888") for lbl in dist.index]

bars = ax.bar(dist.index, dist.values, color=colors, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Stellar Class", fontsize=12)
ax.set_ylabel("Number of Samples", fontsize=12)
ax.set_title("Class Distribution in Master Catalog", fontsize=14, fontweight="bold")
ax.set_ylim(0, dist.max() * 1.15)

# Label each bar with its count
for bar, val in zip(bars, dist.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        str(val),
        ha="center", va="bottom", fontsize=11, fontweight="bold"
    )

plt.tight_layout()
dist_plot_path = os.path.join(NOTEBOOKS, "class_distribution.png")
plt.savefig(dist_plot_path, dpi=150)
plt.close()
print(f"\nClass distribution chart saved to:\n  {dist_plot_path}")


# ─────────────────────────────────────────────
# STEP 4D — Verify every .fits file
#            (check it exists and opens cleanly)
# ─────────────────────────────────────────────
print("\n" + "─" * 40)
print("VERIFYING .FITS FILES")
print("─" * 40)

good_indices  = []
bad_filepaths = []

for idx, row in tqdm(master.iterrows(), total=len(master), desc="Checking files"):
    fp = row["filepath"]

    # Check file exists on disk
    if not os.path.exists(fp):
        bad_filepaths.append(fp)
        continue

    # Try opening with astropy — catches truncated / corrupt files
    try:
        with fits.open(fp) as hdul:
            _ = hdul["COADD"].data["flux"]   # access the actual data block
        good_indices.append(idx)

    except Exception as e:
        print(f"\n  Corrupted: {os.path.basename(fp)}  ({e})")
        bad_filepaths.append(fp)

# Save bad file list
if bad_filepaths:
    with open(BAD_FILE, "w") as f:
        for p in bad_filepaths:
            f.write(p + "\n")
    print(f"\n  {len(bad_filepaths)} bad/missing files logged to:\n    {BAD_FILE}")
else:
    print("\n  All files verified successfully — no corruption found.")

# Remove bad rows from master and save
master_clean = master.loc[good_indices].reset_index(drop=True)
master_clean.to_csv(CATALOG_FILE, index=False)

print(f"\n  Clean catalog: {len(master_clean)} rows")
print(f"  Removed:       {len(master) - len(master_clean)} bad rows")
print(f"  Saved to:      {CATALOG_FILE}")


# ─────────────────────────────────────────────
# STEP 4E — Plot one sample spectrum per class
# ─────────────────────────────────────────────
print("\n" + "─" * 40)
print("PLOTTING SAMPLE SPECTRA")
print("─" * 40)

classes = master_clean["label"].unique()
n       = len(classes)
fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n))

if n == 1:
    axes = [axes]   # make iterable if only one class

for ax, cls in zip(axes, classes):
    # Pick the first valid row for this class
    subset = master_clean[master_clean["label"] == cls]
    sample_row = subset.iloc[0]
    fp         = sample_row["filepath"]

    try:
        wavelength, flux = load_spectrum(fp)

        # Clip extreme outliers so the plot isn't dominated by spikes
        flux_clipped = np.clip(flux, np.percentile(flux, 2), np.percentile(flux, 98))

        color = CLASS_COLORS.get(cls, "#555555")
        ax.plot(wavelength, flux_clipped, lw=0.7, color=color, alpha=0.9)
        ax.set_title(f"{cls.replace('_', ' ').title()}  —  {os.path.basename(fp)}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Wavelength (Å)", fontsize=10)
        ax.set_ylabel("Flux", fontsize=10)

        # Mark key absorption lines
        key_lines = {
            "Hα":    6563,
            "Hβ":    4861,
            "Ca K":  3934,
            "Na D":  5893,
        }
        for name, wl in key_lines.items():
            if wavelength.min() < wl < wavelength.max():
                ax.axvline(wl, color="gray", lw=0.8, linestyle="--", alpha=0.6)
                ax.text(wl + 15, ax.get_ylim()[1] * 0.9, name,
                        fontsize=8, color="gray", va="top")

        ax.grid(True, alpha=0.2, lw=0.4)

    except Exception as e:
        ax.set_title(f"{cls} — Could not plot ({e})")

plt.suptitle("Sample Spectra by Class", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
spectra_plot_path = os.path.join(NOTEBOOKS, "spectra_samples.png")
plt.savefig(spectra_plot_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"  Sample spectra plot saved to:\n    {spectra_plot_path}")

print("\n" + "=" * 60)
print("STEP 4 COMPLETE")
print("=" * 60)
print(f"  Clean samples:      {len(master_clean)}")
print(f"  Bad files removed:  {len(master) - len(master_clean)}")
print(f"  Plots saved in:     {NOTEBOOKS}")
print("\nNext: Run  05_backup_and_version.py")
