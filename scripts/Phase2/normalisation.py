r"""
=============================================================
PHASE 2 — STEP 3: Continuum Normalisation
=============================================================

WHAT IS THE CONTINUUM AND WHY NORMALISE?
─────────────────────────────────────────
A stellar spectrum has two components:

1. THE CONTINUUM — the smooth background glow of the star,
   shaped like a broad curve. A hot white dwarf peaks in
   the blue (short wavelengths). A cool red giant peaks in
   the red (long wavelengths). This curve is determined by
   the star's temperature (Planck/blackbody radiation).

2. ABSORPTION LINES — narrow dips in the continuum at
   specific wavelengths where atoms in the star's atmosphere
   absorb photons. Hydrogen absorbs at Hα (6563Å), Hβ
   (4861Å), etc. Calcium absorbs at Ca K (3934Å). These
   patterns are the real fingerprint that identifies the
   stellar type.

THE PROBLEM:
A white dwarf 10 light years away and a red giant 1000 light
years away have completely different absolute flux values —
the nearby star is much brighter. But their absorption line
PATTERNS are what matter for classification, not their
overall brightness.

If we feed raw flux values into a model, it would partially
learn to classify based on brightness (a proxy for distance)
rather than the physics of the star itself. That is not what
we want.

THE SOLUTION — CONTINUUM NORMALISATION:
Estimate the smooth background continuum curve, then divide
every flux value by that continuum. The result is a spectrum
where:
    - The continuum becomes a flat line at 1.0
    - Absorption lines appear as dips below 1.0
    - Emission lines appear as peaks above 1.0
    - All spectra are on exactly the same scale regardless
      of the original brightness or distance

HOW WE ESTIMATE THE CONTINUUM:
We use an iterative sigma-clipping spline fit:
    1. Fit a cubic spline through the spectrum
    2. Find all points MORE than 1 sigma BELOW the spline
       (these are the absorption lines we want to exclude)
    3. Remove those points and refit the spline
    4. Repeat 3 times — the spline converges on the true
       continuum shape without being pulled down by lines
    5. Divide the spectrum by this final spline estimate

This method is standard in stellar spectroscopy and works
well across all four of our stellar classes.

WHAT THIS SCRIPT DOES:
    1. Loads each .npz from step2_noise/
    2. Estimates the continuum using iterative spline fitting
    3. Divides flux by the continuum to normalise
    4. Clips the result to a physically reasonable range
    5. Saves normalised spectrum to /data/processed/step3_normalised/
    6. Updates master_catalog.csv with 'normalised_filepath' column
    7. Plots the continuum fit and normalised result per class

HOW TO RUN:
    1. Make sure 07_noise_removal.py has finished
    2. Run:
           python 08_normalisation.py

OUTPUT FILES:
    - /data/processed/step3_normalised/{label}/spec-XXXX.npz
      Each .npz contains:
          wavelength  — same corrected wavelength
          flux        — normalised flux (continuum = 1.0)
          continuum   — the estimated continuum curve (for reference)
    - /notebooks/normalisation_comparison.png
    - master_catalog.csv updated with 'normalised_filepath' column

REQUIRES:
    pip install numpy pandas matplotlib scipy tqdm
=============================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR     = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "stellar-spectra-ai")
CATALOG_FILE = os.path.join(BASE_DIR, "data", "catalog", "master_catalog.csv")
IN_DIR       = os.path.join(BASE_DIR, "data", "processed", "step2_noise")
OUT_DIR      = os.path.join(BASE_DIR, "data", "processed", "step3_normalised")
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
# NORMALISATION PARAMETERS
# ─────────────────────────────────────────────
SPLINE_ITERATIONS = 3      # how many times to re-fit after clipping absorption lines
SIGMA_CLIP        = 1.0    # points this many sigma BELOW the fit are excluded (absorption)
SPLINE_SMOOTH     = 1e6    # spline smoothing factor — higher = smoother continuum
                           # if continuum looks too wavy, increase this value
NORM_CLIP_LO      = 0.0    # clip normalised flux below this (physically, flux >= 0)
NORM_CLIP_HI      = 3.0    # clip normalised flux above this (emission lines rarely > 3x continuum)


# ─────────────────────────────────────────────
# HELPER: estimate continuum with iterative
# sigma-clipping spline fit
# ─────────────────────────────────────────────
def estimate_continuum(wavelength, flux):
    """
    Fits a smooth spline to the spectrum, iteratively
    excluding absorption line pixels to get the true
    background continuum level.

    Returns:
        np.array: continuum estimate (same length as flux)
    """
    wl   = wavelength.astype(np.float64)
    fl   = flux.astype(np.float64)
    mask = np.ones(len(fl), dtype=bool)   # all pixels included initially

    for iteration in range(SPLINE_ITERATIONS):
        wl_fit = wl[mask]
        fl_fit = fl[mask]

        if len(wl_fit) < 10:
            # Too few points — return flat continuum at median
            return np.full_like(flux, np.median(flux))

        try:
            # Fit a cubic spline through the unmasked pixels
            spline    = UnivariateSpline(wl_fit, fl_fit, s=SPLINE_SMOOTH, k=3, ext=3)
            continuum = spline(wl)

            # Protect against zero or negative continuum values
            continuum = np.maximum(continuum, 1e-10)

            # Compute residuals: how much is each point below the spline?
            residuals = fl - continuum
            std_below = np.std(residuals[residuals < 0]) if (residuals < 0).any() else 1.0

            # Exclude pixels more than SIGMA_CLIP sigma below (absorption lines)
            # but keep pixels above (emission features, continuum itself)
            mask = residuals > -SIGMA_CLIP * std_below

        except Exception:
            # Spline fit failed — return flat continuum
            return np.full_like(flux, np.median(flux))

    return continuum.astype(np.float32)


# ─────────────────────────────────────────────
# HELPER: normalise a spectrum by its continuum
# ─────────────────────────────────────────────
def normalise_spectrum(wavelength, flux):
    """
    Estimates the continuum and divides the flux by it.
    Result: flux values where 1.0 = continuum level,
    < 1.0 = absorption, > 1.0 = emission.

    Also clips extreme values to the physically
    reasonable range [NORM_CLIP_LO, NORM_CLIP_HI].

    Returns:
        flux_norm  (np.array): normalised flux
        continuum  (np.array): the estimated continuum curve
    """
    continuum  = estimate_continuum(wavelength, flux)
    flux_norm  = flux / continuum
    flux_norm  = np.clip(flux_norm, NORM_CLIP_LO, NORM_CLIP_HI)
    return flux_norm.astype(np.float32), continuum


# ─────────────────────────────────────────────
# LOAD CATALOG
# ─────────────────────────────────────────────
print("=" * 60)
print("PHASE 2 — STEP 3: Continuum Normalisation")
print("=" * 60)

if not os.path.exists(CATALOG_FILE):
    print(f"\nERROR: master_catalog.csv not found.")
    sys.exit(1)

master = pd.read_csv(CATALOG_FILE)
print(f"\nLoaded catalog: {len(master)} rows")

if "noise_filepath" not in master.columns:
    print("\nERROR: 'noise_filepath' column not found.")
    print("Run 07_noise_removal.py first.")
    sys.exit(1)

for label in master["label"].unique():
    os.makedirs(os.path.join(OUT_DIR, label), exist_ok=True)


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
print("\nNormalising all spectra...")
print("(Files already processed will be skipped)\n")

norm_filepaths = []
failed         = []

for idx, row in tqdm(master.iterrows(), total=len(master), desc="Normalising"):
    in_path = row["noise_filepath"]
    label   = row["label"]

    if pd.isna(in_path) or not os.path.exists(str(in_path)):
        norm_filepaths.append(None)
        failed.append({"filepath": in_path, "error": "noise file missing"})
        continue

    basename = os.path.basename(in_path)
    out_path = os.path.join(OUT_DIR, label, basename)

    if os.path.exists(out_path):
        norm_filepaths.append(out_path)
        continue

    try:
        data       = np.load(in_path)
        wavelength = data["wavelength"]
        flux       = data["flux"]
    except Exception as e:
        norm_filepaths.append(None)
        failed.append({"filepath": in_path, "error": str(e)})
        continue

    try:
        flux_norm, continuum = normalise_spectrum(wavelength, flux)
    except Exception as e:
        norm_filepaths.append(None)
        failed.append({"filepath": in_path, "error": f"normalisation failed: {e}"})
        continue

    np.savez_compressed(
        out_path,
        wavelength=wavelength,
        flux=flux_norm,
        continuum=continuum
    )
    norm_filepaths.append(out_path)

master["normalised_filepath"] = norm_filepaths
master.to_csv(CATALOG_FILE, index=False)

if failed:
    pd.DataFrame(failed).to_csv(os.path.join(LOGS, "normalisation_failed.csv"), index=False)
    print(f"\n  {len(failed)} files failed — see logs/normalisation_failed.csv")

success = sum(1 for p in norm_filepaths if p is not None)
print(f"\n  Normalised: {success} / {len(master)} spectra")


# ─────────────────────────────────────────────
# PLOT: raw + continuum overlay, then normalised
# ─────────────────────────────────────────────
print("\nGenerating normalisation plots...")

classes = master["label"].unique()
fig, axes = plt.subplots(len(classes), 2, figsize=(16, 3.5 * len(classes)))

for row_idx, cls in enumerate(classes):
    subset = master[
        (master["label"] == cls) &
        (master["normalised_filepath"].notna())
    ]
    if len(subset) == 0:
        continue

    sample = subset.iloc[0]
    color  = CLASS_COLORS.get(cls, "#555555")

    # Load noise-cleaned (unnormalised) spectrum
    noise_data = np.load(sample["noise_filepath"])
    wl         = noise_data["wavelength"]
    flux_raw   = noise_data["flux"]

    # Load normalised spectrum + continuum
    norm_data  = np.load(sample["normalised_filepath"])
    flux_norm  = norm_data["flux"]
    continuum  = norm_data["continuum"]

    clip_lo = np.percentile(flux_raw, 2)
    clip_hi = np.percentile(flux_raw, 98)

    ax_raw  = axes[row_idx, 0]
    ax_norm = axes[row_idx, 1]

    # Left plot: raw spectrum with continuum overlaid
    ax_raw.plot(wl, np.clip(flux_raw, clip_lo, clip_hi),
                lw=0.6, color=color, alpha=0.7, label="Flux")
    ax_raw.plot(wl, np.clip(continuum, clip_lo, clip_hi),
                lw=1.5, color="black", linestyle="--", alpha=0.8, label="Continuum fit")
    ax_raw.set_title(f"{cls} — Raw + Continuum", fontsize=10, fontweight="bold")
    ax_raw.set_ylabel("Flux")
    ax_raw.legend(fontsize=8)
    ax_raw.grid(True, alpha=0.2)

    # Right plot: normalised spectrum (continuum = 1.0)
    ax_norm.plot(wl, flux_norm, lw=0.6, color=color, alpha=0.9)
    ax_norm.axhline(1.0, color="black", lw=0.8, linestyle="--", alpha=0.5, label="Continuum level")
    ax_norm.set_title(f"{cls} — Normalised", fontsize=10, fontweight="bold")
    ax_norm.set_ylabel("Normalised Flux")
    ax_norm.set_ylim(NORM_CLIP_LO - 0.05, min(NORM_CLIP_HI, 2.0))
    ax_norm.legend(fontsize=8)
    ax_norm.grid(True, alpha=0.2)

    for ax in [ax_raw, ax_norm]:
        ax.set_xlabel("Wavelength (Å)")

plt.suptitle("Continuum Normalisation — Before vs After", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plot_path = os.path.join(NOTEBOOKS, "normalisation_comparison.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Plot saved to: {plot_path}")

print("\n" + "=" * 60)
print("STEP 3 COMPLETE")
print("=" * 60)
print(f"  Normalised spectra: {success}")
print(f"  Output folder:      {OUT_DIR}")
print(f"  Catalog updated:    normalised_filepath column added")
print("\nNext: Run  09_resampling.py")
