
"""
=============================================================
PHASE 1 — STEP 3: Build Your Master Catalog
=============================================================

WHAT THIS SCRIPT DOES:
    1. Reads all four class CSVs you downloaded from SDSS
       (white_dwarfs.csv, quasars.csv, main_sequence.csv, red_giants.csv)
    2. Adds a 'label' column to each one identifying the class
    3. Adds a 'filepath' column pointing to each spectrum's
       local .fits file on your computer
    4. Merges all four into one single DataFrame
    5. Removes any duplicate spectra (same specobjid in two CSVs)
    6. Checks that every expected .fits file actually exists on disk
    7. Saves the final clean result as master_catalog.csv

WHY THIS FILE MATTERS:
    master_catalog.csv is the single source of truth for your
    entire project. Every later script — preprocessing, training,
    evaluation — reads from this one file. Each row is one
    spectrum. The 'label' column is what the AI will learn to
    predict. The 'filepath' column is how scripts know where to
    load the actual data from.

HOW TO RUN:
    1. Make sure you have:
           a) Downloaded your four CSVs from SDSS into /data/catalog/
           b) Run the download script so .fits files are in /data/raw/
    2. Open your terminal
    3. Navigate to your scripts folder:
           cd Desktop/stellar-spectra-ai/scripts
    4. Run:
           python 03_build_catalog.py

OUTPUT FILES:
    - /data/catalog/master_catalog.csv   ← the main output
    - /logs/missing_fits.txt             ← any .fits files not yet downloaded

REQUIRES:
    pip install pandas tqdm
=============================================================
"""

import os
import sys
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# ─────────────────────────────────────────────
# PATHS — only change BASE_DIR if your project
# folder is somewhere other than your Desktop
# ─────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "stellar-spectra-ai")
CATALOG_DIR  = os.path.join(BASE_DIR, "data", "catalog")
RAW_DIR      = os.path.join(BASE_DIR, "data", "raw")
LOGS_DIR     = os.path.join(BASE_DIR, "logs")
OUTPUT_FILE  = os.path.join(CATALOG_DIR, "master_catalog.csv")
MISSING_LOG  = os.path.join(LOGS_DIR, "missing_fits.txt")

os.makedirs(CATALOG_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)

# ─────────────────────────────────────────────
# SOURCE DEFINITIONS
# Each entry maps a CSV filename → class label
# → the raw subfolder its .fits files live in
# ─────────────────────────────────────────────
#
# If you downloaded additional classes (e.g. neutron stars),
# add another dictionary entry here following the same pattern.
#
SOURCES = [
    {
        "csv":    r"C:\Users\Harshit\OneDrive\Desktop\stellar-spectra-ai\data\catalog\white_dwarfs.csv",
        "label":  "white_dwarf",
        "folder": os.path.join(RAW_DIR, "white_dwarf"),
    },
    {
        "csv":    r"C:\Users\Harshit\OneDrive\Desktop\stellar-spectra-ai\data\catalog\quasars.csv",
        "label":  "quasar",
        "folder": os.path.join(RAW_DIR, "quasar"),
    },
    {
        "csv":    r"C:\Users\Harshit\OneDrive\Desktop\stellar-spectra-ai\data\catalog\main_sequence.csv",
        "label":  "main_sequence",
        "folder": os.path.join(RAW_DIR, "main_sequence"),
    },
    {
        "csv":    r"C:\Users\Harshit\OneDrive\Desktop\stellar-spectra-ai\data\catalog\red_giants.csv",
        "label":  "red_giant",
        "folder": os.path.join(RAW_DIR, "red_giant"),
    },
]


# ─────────────────────────────────────────────
# HELPER: build the .fits filename from plate,
# mjd, fiberid — this is SDSS naming convention
# ─────────────────────────────────────────────
def build_filename(plate, mjd, fiberid):
    """
    SDSS names every spectrum file like:
        spec-{plate:04d}-{mjd}-{fiberid:04d}.fits

    The :04d means 'pad with leading zeros to 4 digits'.
    Example: plate=266, mjd=51602, fiberid=1
        → spec-0266-51602-0001.fits

    This is the same pattern the download script used,
    so the filenames will always match.
    """
    return f"spec-{int(plate):04d}-{int(mjd)}-{int(fiberid):04d}.fits"


# ─────────────────────────────────────────────
# HELPER: validate that required columns exist
# in a CSV before we try to use them
# ─────────────────────────────────────────────
REQUIRED_COLS = {"specobjid", "plate", "mjd", "fiberid"}

def validate_columns(df, csv_path):
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        print(f"\n  ERROR in {os.path.basename(csv_path)}:")
        print(f"  Missing columns: {missing}")
        print(f"  Found columns:   {set(df.columns)}")
        print()
        print("  This usually means the SDSS SQL query did not include")
        print("  all required fields. Go back to SkyServer and make sure")
        print("  your SELECT includes: specobjid, plate, mjd, fiberid")
        sys.exit(1)


# ─────────────────────────────────────────────
# STEP 3A — Load and label each CSV
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 3: Building Master Catalog")
print("=" * 60)
print()

all_frames   = []
load_errors  = []

for source in SOURCES:
    csv_path = os.path.join(CATALOG_DIR, source["csv"])
    label    = source["label"]
    folder   = source["folder"]

    # Check the CSV file exists
    if not os.path.exists(csv_path):
        print(f"  MISSING CSV: {source['csv']}")
        print(f"  Expected at: {csv_path}")
        print(f"  Download this from SDSS SkyServer first.")
        print()
        load_errors.append(source["csv"])
        continue

    # Load it
    df = pd.read_csv(csv_path)

    # Validate required columns
    validate_columns(df, csv_path)

    # Strip any accidental whitespace in column names
    # (a common problem when copying SQL results)
    df.columns = df.columns.str.strip()

    # ── Add label column ──────────────────────
    # This is the class name the AI will predict.
    # Every row from white_dwarfs.csv gets label='white_dwarf', etc.
    df["label"] = label

    # ── Add filepath column ───────────────────
    # Build the full local path to each spectrum's .fits file.
    # This is what preprocessing scripts use to open the file.
    df["filepath"] = df.apply(
        lambda row: os.path.join(
            folder,
            build_filename(row["plate"], row["mjd"], row["fiberid"])
        ),
        axis=1
    )

    # ── Add source_csv column ─────────────────
    # Tracks which CSV each row originally came from.
    # Useful for debugging later.
    df["source_csv"] = source["csv"]

    print(f"  Loaded  {len(df):>5} rows  ←  {source['csv']}")
    all_frames.append(df)

# If any CSV was completely missing, stop here
if load_errors:
    print()
    print(f"  {len(load_errors)} CSV(s) could not be found.")
    print("  Fix the missing files and re-run this script.")
    sys.exit(1)

if not all_frames:
    print("  No data loaded. Exiting.")
    sys.exit(1)


# ─────────────────────────────────────────────
# STEP 3B — Merge all frames into one
# ─────────────────────────────────────────────
print()
print("─" * 40)
print("MERGING ALL CLASSES")
print("─" * 40)

master = pd.concat(all_frames, ignore_index=True)
print(f"  Total rows after merge:            {len(master):>6}")


# ─────────────────────────────────────────────
# STEP 3C — Remove duplicates
#
# A spectrum can appear in two CSVs if SDSS classifies
# it ambiguously (e.g. a cool star appearing in both
# red_giants.csv and main_sequence.csv). We deduplicate
# on specobjid, keeping the first occurrence.
# ─────────────────────────────────────────────
before_dedup = len(master)
master = master.drop_duplicates(subset=["specobjid"], keep="first")
after_dedup  = len(master)
dupes_removed = before_dedup - after_dedup

print(f"  Duplicates removed (same specobjid): {dupes_removed:>4}")
print(f"  Rows after deduplication:           {after_dedup:>6}")


# ─────────────────────────────────────────────
# STEP 3D — Check which .fits files exist on disk
#
# We do NOT remove missing files here — that is Step 4's job
# (04_verify_catalog.py also checks for corruption).
# Here we just warn you so you know if the download
# script needs to be re-run for any class.
# ─────────────────────────────────────────────
print()
print("─" * 40)
print("CHECKING .FITS FILES ON DISK")
print("─" * 40)

missing_fits    = []
present_count   = 0

print("  Scanning files", end="", flush=True)
for idx, row in master.iterrows():
    if os.path.exists(row["filepath"]):
        present_count += 1
    else:
        missing_fits.append({
            "specobjid": row["specobjid"],
            "label":     row["label"],
            "filepath":  row["filepath"],
        })
    # Print a dot every 100 rows so you know it is still running
    if idx % 100 == 0:
        print(".", end="", flush=True)

print(" done")
print()
print(f"  .fits files found on disk: {present_count:>5}")
print(f"  .fits files NOT found:     {len(missing_fits):>5}")

if missing_fits:
    # Save missing list to a log file
    missing_df = pd.DataFrame(missing_fits)
    missing_df.to_csv(MISSING_LOG.replace(".txt", ".csv"), index=False)

    with open(MISSING_LOG, "w") as f:
        f.write(f"Missing .fits files as of {datetime.today().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Total missing: {len(missing_fits)}\n\n")
        for m in missing_fits:
            f.write(f"{m['label']:<20}  {m['filepath']}\n")

    print()
    print(f"  WARNING: {len(missing_fits)} .fits files are listed in the catalog")
    print(f"  but were not found on disk.")
    print(f"  This means the download script has not run yet for those rows,")
    print(f"  or some downloads failed.")
    print()
    print(f"  Missing files logged to:")
    print(f"    {MISSING_LOG}")
    print(f"    {MISSING_LOG.replace('.txt', '.csv')}")
    print()
    print("  TO FIX: Re-run 02_download_spectra.py using the missing CSV:")
    print(f"    CSV_FILE = '{MISSING_LOG.replace('.txt', '.csv')}'")
    print(f"    Then re-run this script.")
    print()
    print("  The catalog will still be saved — missing rows are KEPT")
    print("  so you can download them later. Step 4 will remove any")
    print("  that remain missing after your retry.")

else:
    print("  All .fits files present on disk.")


# ─────────────────────────────────────────────
# STEP 3E — Print class distribution
# ─────────────────────────────────────────────
print()
print("─" * 40)
print("CLASS DISTRIBUTION")
print("─" * 40)

dist = master["label"].value_counts()
for label, count in dist.items():
    bar    = "█" * (count // 10)
    status = "  ⚠ LOW — consider downloading more" if count < 200 else ""
    print(f"  {label:<22} {count:>5}  {bar}{status}")

print()
print(f"  Total samples: {len(master)}")


# ─────────────────────────────────────────────
# STEP 3F — Standardise column order and types
#
# Puts columns in a logical order so the CSV
# is easy to read when you open it in Excel.
# Converts specobjid to string to avoid scientific
# notation (SDSS IDs are very large integers).
# ─────────────────────────────────────────────
master["specobjid"] = master["specobjid"].astype(str).str.strip()

# Define preferred column order
preferred_cols = [
    "specobjid",    # unique spectrum ID
    "label",        # class — what the AI predicts
    "ra",           # sky position (right ascension)
    "dec",          # sky position (declination)
    "z",            # redshift value
    "plate",        # SDSS plate number
    "mjd",          # observation date (Modified Julian Date)
    "fiberid",      # fibre number on the spectrograph
    "class",        # SDSS original classification (STAR / QSO / GALAXY)
    "subclass",     # SDSS subclass (WD, G, K, M ...)
    "filepath",     # local path to the .fits file
    "source_csv",   # which CSV this row came from
]

# Keep only columns that actually exist in the DataFrame
# (in case some CSVs didn't include all optional fields)
final_cols = [c for c in preferred_cols if c in master.columns]

# Add any extra columns not in our preferred list at the end
extra_cols = [c for c in master.columns if c not in final_cols]
final_cols += extra_cols

master = master[final_cols]


# ─────────────────────────────────────────────
# STEP 3G — Save master_catalog.csv
# ─────────────────────────────────────────────
print()
print("─" * 40)
print("SAVING MASTER CATALOG")
print("─" * 40)

master.to_csv(OUTPUT_FILE, index=False)

print(f"  Saved to: {OUTPUT_FILE}")
print(f"  Rows:     {len(master)}")
print(f"  Columns:  {list(master.columns)}")


# ─────────────────────────────────────────────
# STEP 3H — Print a preview of the first 3 rows
# so you can visually confirm it looks correct
# ─────────────────────────────────────────────
print()
print("─" * 40)
print("PREVIEW (first 3 rows)")
print("─" * 40)

preview_cols = ["specobjid", "label", "z", "plate", "mjd", "fiberid", "filepath"]
preview      = master[[c for c in preview_cols if c in master.columns]].head(3)

# Print each row as key:value pairs so nothing wraps awkwardly
for i, row in preview.iterrows():
    print(f"\n  Row {i}:")
    for col, val in row.items():
        # Truncate long filepaths for display
        display_val = str(val)
        if col == "filepath" and len(display_val) > 55:
            display_val = "..." + display_val[-52:]
        print(f"    {col:<12}: {display_val}")


# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 3 COMPLETE")
print("=" * 60)
print()
print(f"  master_catalog.csv — {len(master)} rows, {len(master.columns)} columns")
print(f"  Location: {OUTPUT_FILE}")
print()
print("  Column summary:")
print("    specobjid   — unique ID for each spectrum")
print("    label       — class label (what the AI will predict)")
print("    z           — redshift (used in Phase 2 correction)")
print("    plate/mjd/fiberid — used to locate the .fits file")
print("    filepath    — full path to the .fits file on your computer")
print()

if missing_fits:
    print(f"  ACTION NEEDED: {len(missing_fits)} .fits files still missing.")
    print("  Re-run 02_download_spectra.py for those files, then")
    print("  re-run this script before moving to Step 4.")
else:
    print("  No action needed. Proceed to:")
    print("  → python 04_verify_catalog.py")
print()