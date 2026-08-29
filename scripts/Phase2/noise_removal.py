import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tqdm import tqdm

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR     = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "stellar-spectra-ai")
CATALOG_FILE = os.path.join(BASE_DIR, "data", "catalog", "master_catalog.csv")
IN_DIR       = os.path.join(BASE_DIR, "data", "processed", "step1_redshift")
OUT_DIR      = os.path.join(BASE_DIR, "data", "processed", "step2_noise")
NOTEBOOKS    = os.path.join(BASE_DIR, "notebooks")
LOGS         = os.path.join(BASE_DIR, "logs")
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(NOTEBOOKS, exist_ok=True)

CLASS_COLORS = {
    "white_dwarf":   "#4A90D9",
    "quasar":        "#E8593C",
    "main_sequence": "#3BAD75",
    "red_giant":     "#D4A017",
}

# ─────────────────────────────────────────────
# SAVITZKY-GOLAY PARAMETERS
# ─────────────────────────────────────────────
SG_WINDOW   = 11    # number of pixels in sliding window (must be odd)
SG_POLYORDER = 3    # polynomial degree (must be < SG_WINDOW)

# ─────────────────────────────────────────────
# KNOWN SKY EMISSION LINE WAVELENGTHS (Angstroms)
# ─────────────────────────────────────────────
SKY_LINES = [
    5577.3,   # OI atmospheric line — very common residual
    6300.3,   # OI atmospheric line
    6363.8,   # OI atmospheric line
    5890.0,   # Na D doublet (sodium street lights contamination)
    5896.0,
    6864.0,   # O2 band
    7620.0,   # O2 band
    7640.0,
]
SKY_MASK_WIDTH = 10.0   # Angstroms either side of each sky line to mask

# ─────────────────────────────────────────────
# Sigma-clip spike removal
# ─────────────────────────────────────────────
def remove_spikes(wavelength, flux, sigma_threshold=3.0, window=51):
    """
    Detects and removes sharp upward spikes (sky line residuals
    and cosmic rays) by comparing each point to its local median.

    For each flux value:
        1. Compute the median of the surrounding 'window' pixels
        2. Compute the standard deviation of those pixels
        3. If the point is more than sigma_threshold × std above
           the median, it is a spike — replace it with the median

    This preserves broad absorption features (which are real)
    but removes sharp single-pixel or few-pixel spikes (which are not).

    Args:
        wavelength (np.array): wavelength axis
        flux (np.array): raw flux
        sigma_threshold (float): how many sigma above median = spike
        window (int): local window size for median calculation (must be odd)

    Returns:
        np.array: flux with spikes replaced by local median
    """
    flux_clean = flux.copy()
    half       = window // 2

    for i in range(len(flux)):
        lo  = max(0, i - half)
        hi  = min(len(flux), i + half + 1)
        local = flux[lo:hi]

        local_median = np.median(local)
        local_std    = np.std(local)

        if local_std > 0 and (flux[i] - local_median) > sigma_threshold * local_std:
            flux_clean[i] = local_median

    return flux_clean

# ─────────────────────────────────────────────
# Mask known sky emission lines
# ─────────────────────────────────────────────
def mask_sky_lines(wavelength, flux):
    """
    Replaces flux values near known sky emission lines with
    linearly interpolated values from the surrounding region.

    This is called 'inpainting' — instead of leaving zeros or
    NaN values where the sky line was, we fill it smoothly so
    the Savitzky-Golay filter doesn't react to sudden gaps.

    Args:
        wavelength (np.array): wavelength axis in Angstroms
        flux (np.array): flux array

    Returns:
        np.array: flux with sky line regions interpolated over
    """
    flux_masked = flux.copy()

    for sky_wl in SKY_LINES:
        mask = np.abs(wavelength - sky_wl) < SKY_MASK_WIDTH

        if not mask.any():
            continue

        indices      = np.where(mask)[0]
        all_indices  = np.arange(len(flux))

        outside_mask = ~mask
        if outside_mask.sum() < 2:
            continue

        flux_masked[mask] = np.interp(
            wavelength[mask],
            wavelength[outside_mask],
            flux[outside_mask]
        )

    return flux_masked

# ─────────────────────────────────────────────
# Apply full noise removal pipeline
# ─────────────────────────────────────────────
def clean_spectrum(wavelength, flux):
    """
    Applies the full noise removal sequence

    Returns:
        np.array: cleaned flux (same length as input)
    """
    flux_step1 = remove_spikes(wavelength, flux)
    flux_step2 = mask_sky_lines(wavelength, flux_step1)

    window = min(SG_WINDOW, len(flux_step2) - 1)
    if window % 2 == 0:
        window -= 1
    if window < SG_POLYORDER + 1:
        return flux_step2

    flux_step3 = savgol_filter(flux_step2, window_length=window, polyorder=SG_POLYORDER)
    return flux_step3.astype(np.float32)

# ─────────────────────────────────────────────
# LOAD CATALOG
# ─────────────────────────────────────────────
print("=" * 60)
print("PHASE 2 — STEP 2: Noise Removal")
print("=" * 60)

if not os.path.exists(CATALOG_FILE):
    print(f"\nERROR: master_catalog.csv not found.")
    sys.exit(1)

master = pd.read_csv(CATALOG_FILE)
print(f"\nLoaded catalog: {len(master)} rows")

if "redshift_filepath" not in master.columns:
    print("\nERROR: 'redshift_filepath' column not found.")
    print("Run redshift_correction.py first.")
    sys.exit(1)

# Create output subfolders per class
for label in master["label"].unique():
    os.makedirs(os.path.join(OUT_DIR, label), exist_ok=True)

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
print("\nApplying noise removal to all spectra...")
print("(Files already processed will be skipped)\n")

noise_filepaths = []
failed          = []

for idx, row in tqdm(master.iterrows(), total=len(master), desc="Cleaning"):
    in_path = row["redshift_filepath"]
    label   = row["label"]

    if pd.isna(in_path) or not os.path.exists(str(in_path)):
        noise_filepaths.append(None)
        failed.append({"filepath": in_path, "error": "redshift file missing"})
        continue

    basename = os.path.basename(in_path)
    out_path = os.path.join(OUT_DIR, label, basename)

    if os.path.exists(out_path):
        noise_filepaths.append(out_path)
        continue

    try:
        data       = np.load(in_path)
        wavelength = data["wavelength"]
        flux       = data["flux"]
    except Exception as e:
        noise_filepaths.append(None)
        failed.append({"filepath": in_path, "error": str(e)})
        continue

    try:
        flux_clean = clean_spectrum(wavelength, flux)
    except Exception as e:
        noise_filepaths.append(None)
        failed.append({"filepath": in_path, "error": f"cleaning failed: {e}"})
        continue

    np.savez_compressed(out_path, wavelength=wavelength, flux=flux_clean)
    noise_filepaths.append(out_path)

# Update catalog
master["noise_filepath"] = noise_filepaths
master.to_csv(CATALOG_FILE, index=False)

if failed:
    pd.DataFrame(failed).to_csv(os.path.join(LOGS, "noise_failed.csv"), index=False)
    print(f"\n  {len(failed)} files failed — see logs/noise_failed.csv")

success = sum(1 for p in noise_filepaths if p is not None)
print(f"\n  Cleaned: {success} / {len(master)} spectra")

# ─────────────────────────────────────────────
# PLOT: before vs after noise removal
# one example per class
# ─────────────────────────────────────────────
print("\nGenerating noise removal comparison plots...")

classes = master["label"].unique()
fig, axes = plt.subplots(len(classes), 2, figsize=(16, 3.5 * len(classes)))

for row_idx, cls in enumerate(classes):
    subset = master[
        (master["label"] == cls) &
        (master["noise_filepath"].notna())
    ]
    if len(subset) == 0:
        continue

    sample  = subset.iloc[0]
    color   = CLASS_COLORS.get(cls, "#555555")

    display_name = str(cls).replace("_", " ").title()

    # Load before (redshift corrected, not yet cleaned)
    before_data = np.load(sample["redshift_filepath"])
    wl          = before_data["wavelength"]
    flux_before = before_data["flux"]

    # Load after (cleaned)
    after_data  = np.load(sample["noise_filepath"])
    flux_after  = after_data["flux"]

    clip_lo = np.percentile(flux_before, 2)
    clip_hi = np.percentile(flux_before, 98)

    ax_before = axes[row_idx, 0]
    ax_after  = axes[row_idx, 1]

    ax_before.plot(wl, np.clip(flux_before, clip_lo, clip_hi),
                   lw=0.6, color=color, alpha=0.8)
    ax_before.set_title(f"{display_name} — Before", fontsize=10, fontweight="bold")
    ax_before.set_ylabel("Flux")
    ax_before.grid(True, alpha=0.2)

    ax_after.plot(wl, np.clip(flux_after, clip_lo, clip_hi),
                  lw=0.6, color=color, alpha=0.8)
    ax_after.set_title(f"{display_name} — After (smoothed)", fontsize=10, fontweight="bold")
    ax_after.grid(True, alpha=0.2)

    for ax in [ax_before, ax_after]:
        ax.set_xlabel("Wavelength (Å)")

plt.suptitle("Noise Removal — Before vs After", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plot_path = os.path.join(NOTEBOOKS, "noise_removal_comparison.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Plot saved to: {plot_path}")

print("\n" + "=" * 60)
print("STEP 2 COMPLETE")
print("=" * 60)
print(f"  Catalog updated: noise_filepath column added")
print("\nNext: Run  normalisation.py")