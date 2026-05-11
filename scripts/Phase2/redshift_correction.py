r'''
=============================================================
PHASE 2 — STEP 1: Redshift Correction
=============================================================

WHAT IS REDSHIFT AND WHY DOES IT MATTER?
─────────────────────────────────────────
When an object in space is moving away from us, its light
gets stretched to longer (redder) wavelengths. This shift
is called redshift, measured as the value 'z' in your catalog.

Example: The hydrogen alpha absorption line (Hα) sits at
6563 Å in a stationary star. In a quasar with z=0.5, that
same line would appear at:
    6563 × (1 + 0.5) = 9844 Å

If you try to train a model on uncorrected spectra, the same
physical feature appears at a completely different wavelength
for every object — making it nearly impossible for any model
to learn consistent patterns.

Redshift correction (also called 'de-redshifting') shifts
every spectrum back to its rest-frame, so the same absorption
lines always appear at the same wavelength regardless of how
far away the object is.

The formula is simple:
    wavelength_rest = wavelength_observed / (1 + z)

Most stars in your dataset have z ≈ 0 (they are in our galaxy
and barely moving relative to us), so correction makes almost
no difference for them. But quasars can have z > 2, meaning
their features are shifted by 3x — correction is critical.

WHAT THIS SCRIPT DOES:
    1. Loads master_catalog.csv
    2. For each spectrum, reads the raw flux and wavelength
       from the .fits file
    3. Applies the redshift correction to the wavelength axis
    4. Saves the corrected (wavelength, flux) pair as a .npz
       file in /data/processed/step1_redshift/
    5. Updates master_catalog.csv with a 'redshift_filepath'
       column pointing to each corrected file
    6. Plots before/after comparison for one spectrum per class

HOW TO RUN:
    1. Make sure 04_verify_catalog.py has been run and your
       master_catalog.csv has 1782 clean rows
    2. Open terminal, navigate to scripts folder:
           cd "C:\Users\Harshit\OneDrive\Desktop\stellar-spectra-ai\scripts"
    3. Run:
           python 06_redshift_correction.py

OUTPUT FILES:
    - /data/processed/step1_redshift/{label}/spec-XXXX.npz
      Each .npz contains two arrays:
          wavelength — corrected wavelength axis in Angstroms
          flux       — raw flux values (not yet normalised)
    - /notebooks/redshift_comparison.png
    - master_catalog.csv updated with 'redshift_filepath' column

REQUIRES:
    pip install astropy numpy pandas matplotlib tqdm
=============================================================
'''

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR     = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "stellar-spectra-ai")
CATALOG_FILE = os.path.join(BASE_DIR, "data", "catalog", "master_catalog.csv")
OUT_DIR      = os.path.join(BASE_DIR, "data", "processed", "step1_redshift")
NOTEBOOKS    = os.path.join(BASE_DIR, "notebooks")
LOGS         = os.path.join(BASE_DIR, "logs")
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(NOTEBOOKS, exist_ok=True)
os.makedirs(LOGS,      exist_ok=True)

CLASS_COLORS = {
    "white_dwarf":   "#4A90D9",
    "quasar":        "#E8593C",
    "main_sequence": "#3BAD75",
    "red_giant":     "#D4A017",
}

# ─────────────────────────────────────────────
# HELPER: read raw wavelength + flux from .fits
# ─────────────────────────────────────────────
def read_fits(filepath):
    """
    Reads an SDSS .fits file and returns the raw observed
    wavelength array and flux array.

    SDSS encodes wavelengths logarithmically in the header:
        log10(wavelength) = COEFF0 + COEFF1 * pixel_index

    We invert this to get wavelength in Angstroms.
    """
    with fits.open(filepath) as hdul:
        coadd  = hdul["COADD"]
        flux   = coadd.data["flux"].astype(np.float32)
        header = coadd.header

        coeff0 = header.get("COEFF0")
        coeff1 = header.get("COEFF1")

        if coeff0 is not None and coeff1 is not None:
            npix       = len(flux)
            log_wave   = coeff0 + coeff1 * np.arange(npix)
            wavelength = (10 ** log_wave).astype(np.float32)
        else:
            # Fallback for files missing header coefficients
            wavelength = np.arange(len(flux), dtype=np.float32)

    return wavelength, flux


# ─────────────────────────────────────────────
# HELPER: apply redshift correction
# ─────────────────────────────────────────────
def correct_redshift(wavelength, z):
    """
    Shifts the observed wavelength axis back to rest-frame.

    Formula:  wavelength_rest = wavelength_observed / (1 + z)

    For a star with z=0 this changes nothing.
    For a quasar with z=2.0 this compresses the wavelength
    axis by a factor of 3, bringing all features back to
    where they would be if the quasar were stationary.

    Args:
        wavelength (np.array): observed wavelength in Angstroms
        z (float): redshift from SDSS catalog

    Returns:
        np.array: rest-frame wavelength in Angstroms
    """
    # Guard against bad z values (NaN, negative, unrealistically large)
    if not np.isfinite(z) or z < 0 or z > 10:
        z = 0.0
    return wavelength / (1.0 + z)


# ─────────────────────────────────────────────
# LOAD CATALOG
# ─────────────────────────────────────────────
print("=" * 60)
print("PHASE 2 — STEP 1: Redshift Correction")
print("=" * 60)

if not os.path.exists(CATALOG_FILE):
    print(f"\nERROR: master_catalog.csv not found at:\n  {CATALOG_FILE}")
    print("Run 04_verify_catalog.py first.")
    sys.exit(1)

master = pd.read_csv(CATALOG_FILE)
print(f"\nLoaded catalog: {len(master)} rows")

# Check the 'z' column exists
if "z" not in master.columns:
    print("\nERROR: 'z' column not found in catalog.")
    print("This column holds the redshift value from SDSS.")
    print("Re-check your SQL query included 'z' in the SELECT.")
    sys.exit(1)

# Fill missing z values with 0 (treat as no redshift)
missing_z = master["z"].isna().sum()
if missing_z > 0:
    print(f"  WARNING: {missing_z} rows have missing z values — treating as z=0")
    master["z"] = master["z"].fillna(0.0)


# ─────────────────────────────────────────────
# CREATE OUTPUT SUBFOLDERS PER CLASS
# ─────────────────────────────────────────────
for label in master["label"].unique():
    os.makedirs(os.path.join(OUT_DIR, label), exist_ok=True)


# ─────────────────────────────────────────────
# MAIN LOOP — process every spectrum
# ─────────────────────────────────────────────
print("\nApplying redshift correction to all spectra...")
print("(Files already processed will be skipped)\n")

redshift_filepaths = []
failed             = []

for idx, row in tqdm(master.iterrows(), total=len(master), desc="Correcting"):
    fits_path = row["filepath"]
    label     = row["label"]
    z         = float(row["z"])

    # Build output path — same filename but .npz extension
    basename   = os.path.splitext(os.path.basename(fits_path))[0]
    out_path   = os.path.join(OUT_DIR, label, basename + ".npz")

    # Skip if already done
    if os.path.exists(out_path):
        redshift_filepaths.append(out_path)
        continue

    # Read the raw spectrum
    try:
        wavelength, flux = read_fits(fits_path)
    except Exception as e:
        failed.append({"filepath": fits_path, "error": str(e)})
        redshift_filepaths.append(None)
        continue

    # Apply redshift correction to wavelength axis
    wavelength_rest = correct_redshift(wavelength, z)

    # Save as .npz — this stores multiple named arrays in one file
    # Load later with:  data = np.load(path); wl = data["wavelength"]
    np.savez_compressed(out_path, wavelength=wavelength_rest, flux=flux)
    redshift_filepaths.append(out_path)

# Add the new column to master catalog
master["redshift_filepath"] = redshift_filepaths
master.to_csv(CATALOG_FILE, index=False)

# Log failures
if failed:
    fail_df = pd.DataFrame(failed)
    fail_df.to_csv(os.path.join(LOGS, "redshift_failed.csv"), index=False)
    print(f"\n  {len(failed)} files failed — see logs/redshift_failed.csv")

success = sum(1 for p in redshift_filepaths if p is not None)
print(f"\n  Processed: {success} / {len(master)} spectra")
print(f"  Saved to:  {OUT_DIR}")


# ─────────────────────────────────────────────
# PLOT: before vs after redshift correction
# for one quasar (most visually dramatic class)
# ─────────────────────────────────────────────
print("\nGenerating before/after comparison plot...")

# Find a quasar with a meaningful redshift (z > 0.3)
quasars = master[(master["label"] == "quasar") & (master["z"] > 0.3)]

if len(quasars) == 0:
    print("  No high-z quasars found for comparison plot — skipping.")
else:
    sample = quasars.iloc[0]
    wl_obs, flux = read_fits(sample["filepath"])
    wl_rest      = correct_redshift(wl_obs, sample["z"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    # Before
    ax1.plot(wl_obs, np.clip(flux, np.percentile(flux, 2), np.percentile(flux, 98)),
             color="#E8593C", lw=0.7)
    ax1.set_title(f"BEFORE correction  —  Quasar  z={sample['z']:.3f}", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Observed Wavelength (Å)")
    ax1.set_ylabel("Flux")
    ax1.grid(True, alpha=0.2)

    # Mark Hα position in observed frame
    ha_observed = 6563 * (1 + sample["z"])
    if wl_obs.min() < ha_observed < wl_obs.max():
        ax1.axvline(ha_observed, color="gray", lw=1, linestyle="--")
        ax1.text(ha_observed + 30, ax1.get_ylim()[1] * 0.85,
                 f"Hα (shifted to {ha_observed:.0f}Å)", fontsize=8, color="gray")

    # After
    ax2.plot(wl_rest, np.clip(flux, np.percentile(flux, 2), np.percentile(flux, 98)),
             color="#4A90D9", lw=0.7)
    ax2.set_title("AFTER correction  —  rest-frame wavelengths restored", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Rest-frame Wavelength (Å)")
    ax2.set_ylabel("Flux")
    ax2.grid(True, alpha=0.2)

    # Mark Hα at rest position
    if wl_rest.min() < 6563 < wl_rest.max():
        ax2.axvline(6563, color="gray", lw=1, linestyle="--")
        ax2.text(6563 + 30, ax2.get_ylim()[1] * 0.85, "Hα (6563Å)", fontsize=8, color="gray")

    plt.suptitle("Redshift Correction — Before vs After", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(NOTEBOOKS, "redshift_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to: {plot_path}")

print("\n" + "=" * 60)
print("STEP 1 COMPLETE")
print("=" * 60)
print(f"  Corrected spectra: {success}")
print(f"  Output folder:     {OUT_DIR}")
print(f"  Catalog updated:   redshift_filepath column added")
print("\nNext: Run  07_noise_removal.py")
