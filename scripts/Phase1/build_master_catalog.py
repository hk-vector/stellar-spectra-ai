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
# SOURCE DEFINITIONS (EDIT THESE!)
# ─────────────────────────────────────────────

# If you downloaded additional classes (e.g. neutron stars),
# add another dictionary entry here following the same pattern.
# Also change the CLASS_COLORS dictionary and any other dictionary
# which uses classes name

SOURCES = [
    {
        "csv":    r"Path to your CSV",
        "label":  "white_dwarf",
        "folder": os.path.join(RAW_DIR, "white_dwarf"),
    },
    {
        "csv":    r"Path to your CSV",
        "label":  "quasar",
        "folder": os.path.join(RAW_DIR, "quasar"),
    },
    {
        "csv":    r"Path to your CSV",
        "label":  "main_sequence",
        "folder": os.path.join(RAW_DIR, "main_sequence"),
    },
    {
        "csv":    r"Path to your CSV",
        "label":  "red_gaints",
        "folder": os.path.join(RAW_DIR, "red_giant"),
    },
]

# ─────────────────────────────────────────────
# Build the .fits filename from plate,
# mjd, fiberid — this is SDSS naming convention
# ─────────────────────────────────────────────
def build_filename(plate, mjd, fiberid):
    """
    SDSS names every spectrum file like:
        spec-{plate:04d}-{mjd}-{fiberid:04d}.fits

    The :04d means 'pad with leading zeros to 4 digits'.
    Example: plate=266, mjd=51602, fiberid=1
        → spec-0266-51602-0001.fits
    """
    return f"spec-{int(plate):04d}-{int(mjd)}-{int(fiberid):04d}.fits"

# ─────────────────────────────────────────────
# Validate that required columns exist
# in a CSV before we try to use them
# ─────────────────────────────────────────────
REQUIRED_COLS = {"specobjid", "plate", "mjd", "fiberid"}

def validate_columns(df, csv_path):
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        print(f"\n  ERROR in {os.path.basename(csv_path)}:")
        print(f"  Missing columns: {missing}")
        print(f"  Found columns:   {set(df.columns)}")
        sys.exit(1)

# ─────────────────────────────────────────────
# Load and label each CSV
# ─────────────────────────────────────────────
print("=" * 60)
print("Building Master Catalog")
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

    df = pd.read_csv(csv_path)

    validate_columns(df, csv_path)

    # Strip any accidental whitespace in column names
    df.columns = df.columns.str.strip()

    # ── Add label column ──────────────────────
    # This is the class name the AI will predict.
    df["label"] = label

    # ── Add filepath column ───────────────────
    df["filepath"] = df.apply(
        lambda row: os.path.join(
            folder,
            build_filename(row["plate"], row["mjd"], row["fiberid"])
        ),
        axis=1
    )

    # ── Add source_csv column ─────────────────
    # Tracks which CSV each row originally came from.
    df["source_csv"] = source["csv"]

    print(f"  Loaded  {len(df):>5} rows  <-  {source['csv']}")
    all_frames.append(df)

# If any CSV was completely missing, stop here
if load_errors:
    print()
    print(f"  {len(load_errors)} CSV(s) could not be found.")
    sys.exit(1)

if not all_frames:
    print("  No data loaded. Exiting.")
    sys.exit(1)

# ─────────────────────────────────────────────
# Merge all frames into one
# ─────────────────────────────────────────────
print()
print("─" * 40)
print("MERGING ALL CLASSES")
print("─" * 40)

master = pd.concat(all_frames, ignore_index=True)
print(f"  Total rows after merge:            {len(master):>6}")

# ─────────────────────────────────────────────
# Remove duplicates
# ─────────────────────────────────────────────
before_dedup = len(master)
master = master.drop_duplicates(subset=["specobjid"], keep="first")
after_dedup  = len(master)
dupes_removed = before_dedup - after_dedup

print(f"  Duplicates removed (same specobjid): {dupes_removed:>4}")
print(f"  Rows after deduplication:           {after_dedup:>6}")

# ─────────────────────────────────────────────
# Check which .fits files exist on disk
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
    print()
    print(f"  Missing files logged to:")
    print(f"    {MISSING_LOG}")
    print(f"    {MISSING_LOG.replace('.txt', '.csv')}")
    print()
    print("  TO FIX: Re-run download_spectra.py using the missing CSV:")
    print(f"    CSV_FILE = '{MISSING_LOG.replace('.txt', '.csv')}'")
    print(f"    Then re-run this script.")
    print()
    print("  The catalog will still be saved — missing rows are KEPT")
    print("  so you can download them later. Step 4 will remove any")
    print("  that remain missing after your retry.")

else:
    print("  All .fits files present on disk.")

# ─────────────────────────────────────────────
# Print class distribution
# ─────────────────────────────────────────────
print()
print("─" * 40)
print("CLASS DISTRIBUTION")
print("─" * 40)

dist = master["label"].value_counts()
for label, count in dist.items():
    bar    = "█" * (count // 10)
    status = "LOW — consider downloading more" if count < 200 else ""
    print(f"  {label:<22} {count:>5}  {bar}{status}")

print()
print(f"  Total samples: {len(master)}")

# ─────────────────────────────────────────────
# Standardise column order and types
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

final_cols = [c for c in preferred_cols if c in master.columns]

# Add any extra columns not in our preferred list at the end
extra_cols = [c for c in master.columns if c not in final_cols]
final_cols += extra_cols

master = master[final_cols]

# ─────────────────────────────────────────────
# Save master_catalog.csv
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
# FINAL SUMMARY
# ─────────────────────────────────────────────

if missing_fits:
    print(f"  ACTION NEEDED: {len(missing_fits)} .fits files still missing.")
    print("  Re-run download_spectra.py for those files, then")
    print("  re-run this script before moving to Step 4.")
print()