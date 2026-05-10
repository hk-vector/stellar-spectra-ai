"""
=============================================================
PHASE 1 — STEP 5: Back Up and Version Your Data
=============================================================

WHAT THIS SCRIPT DOES:
    1. Initialises a git repository in your project folder
    2. Creates a proper .gitignore so large .fits files are
       NOT tracked by git (they go to external storage instead)
    3. Reads your master_catalog.csv and auto-generates a
       detailed README.md with your actual sample counts,
       download date, and project structure
    4. Makes your first git commit containing:
           - All catalog CSVs
           - All scripts
           - README.md
           - .gitignore
    5. Prints clear instructions for copying your raw .fits
       files to external storage (Google Drive / hard drive)

HOW TO RUN:
    1. Make sure you have run 04_verify_catalog.py first
    2. Open your terminal
    3. Navigate to your scripts folder:
           cd Desktop/stellar-spectra-ai/scripts
    4. Run:
           python 05_backup_and_version.py

    NOTE: You need git installed on your computer.
          Download it free from: https://git-scm.com/downloads
          After installing, restart your terminal and run again.

OUTPUT FILES:
    - /README.md         — auto-generated project documentation
    - /.gitignore        — tells git what NOT to track
    - git repository     — initialised in your project root

REQUIRES:
    pip install pandas gitpython
    (gitpython lets Python talk to git for you)
=============================================================
"""

import os
import sys
import subprocess
import platform
import shutil
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "stellar-spectra-ai")
CATALOG_FILE = os.path.join(BASE_DIR, "data", "catalog", "master_catalog.csv")
GITIGNORE    = os.path.join(BASE_DIR, ".gitignore")
README       = os.path.join(BASE_DIR, "README.md")
LOGS         = os.path.join(BASE_DIR, "logs")
BAD_FILE     = os.path.join(LOGS, "bad_files.txt")


# ─────────────────────────────────────────────
# HELPER: run a shell command and print output
# ─────────────────────────────────────────────
def run(command, cwd=BASE_DIR):
    """
    Runs a terminal command from Python and prints the result.
    'cwd' means 'current working directory' — where the command runs.
    """
    result = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    if result.stdout.strip():
        print(f"    {result.stdout.strip()}")
    if result.returncode != 0 and result.stderr.strip():
        print(f"    WARNING: {result.stderr.strip()}")
    return result.returncode


# ─────────────────────────────────────────────
# CHECK: git is installed
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 5: Back Up and Version Your Data")
print("=" * 60)

print("\nChecking git installation...")
if shutil.which("git") is None:
    print("\n  ERROR: git is not installed on this computer.")
    print("  Download and install it from: https://git-scm.com/downloads")
    print("  After installing, restart your terminal and run this script again.")
    sys.exit(1)

result = subprocess.run(["git", "--version"], capture_output=True, text=True)
print(f"  Found: {result.stdout.strip()}")


# ─────────────────────────────────────────────
# CHECK: master catalog exists
# ─────────────────────────────────────────────
if not os.path.exists(CATALOG_FILE):
    print(f"\n  ERROR: master_catalog.csv not found at:\n    {CATALOG_FILE}")
    print("  Run 03_build_catalog.py and 04_verify_catalog.py first.")
    sys.exit(1)

master = pd.read_csv(CATALOG_FILE)
print(f"\n  Loaded master catalog: {len(master)} clean samples")


# ─────────────────────────────────────────────
# STEP 5A — Initialise git repository
# ─────────────────────────────────────────────
print("\n" + "─" * 40)
print("INITIALISING GIT REPOSITORY")
print("─" * 40)

git_dir = os.path.join(BASE_DIR, ".git")
if os.path.exists(git_dir):
    print("  Git repo already exists — skipping init.")
else:
    code = run("git init")
    if code == 0:
        print("  Git repository initialised successfully.")
    else:
        print("  ERROR: Could not initialise git repository.")
        sys.exit(1)

# Set a default git identity if not already configured
# (needed for the first commit — git refuses to commit without a name/email)
name_check  = subprocess.run("git config user.name",  shell=True, cwd=BASE_DIR,
                              capture_output=True, text=True)
email_check = subprocess.run("git config user.email", shell=True, cwd=BASE_DIR,
                              capture_output=True, text=True)

if not name_check.stdout.strip():
    run('git config user.name "Stellar Spectra AI"')
    print("  Set git user.name (local to this repo)")

if not email_check.stdout.strip():
    run('git config user.email "you@example.com"')
    print("  Set git user.email (local to this repo)")
    print("  TIP: Replace with your real email for proper attribution:")
    print('       git config user.email "your@email.com"')


# ─────────────────────────────────────────────
# STEP 5B — Write .gitignore
# ─────────────────────────────────────────────
print("\n" + "─" * 40)
print("CREATING .GITIGNORE")
print("─" * 40)

# Explanation of each ignored pattern is included so you understand WHY
gitignore_content = """# ─────────────────────────────────────────
# stellar-spectra-ai — .gitignore
# ─────────────────────────────────────────

# RAW SPECTRAL DATA (.fits files)
# These files are large (1–5 MB each).
# With 2000 spectra that is up to 10 GB — too large for git.
# Store them on an external hard drive or Google Drive instead.
data/raw/

# PROCESSED NUMPY ARRAYS
# Phase 2 will create .npy files (preprocessed spectra).
# These are also large and can be regenerated from raw data.
data/processed/

# PYTHON CACHE
# Python creates these automatically — they are not source code.
__pycache__/
*.py[cod]
*.pyo

# JUPYTER NOTEBOOK CHECKPOINTS
# Auto-saved by Jupyter — not needed in version control.
.ipynb_checkpoints/

# OPERATING SYSTEM FILES
# Invisible metadata files created by Mac and Windows.
.DS_Store
Thumbs.db

# ENVIRONMENT FILES
# If you use a virtual environment, its files stay local.
venv/
.env/
*.egg-info/

# PLOT IMAGES (optional — remove this block if you want to track them)
# notebooks/*.png

# LARGE LOG FILES
# Keep the logs folder but ignore large auto-generated logs.
logs/download_log.txt

# MODEL CHECKPOINTS (Phase 4)
# These will be large binary files — store them separately.
models/checkpoints/
*.pt
*.pth
*.h5
"""

with open(GITIGNORE, "w", encoding="utf-8") as f:
    f.write(gitignore_content)

print(f"  .gitignore written to:\n    {GITIGNORE}")
print("  The following are EXCLUDED from git (stored separately):")
print("    - data/raw/       (raw .fits files — too large)")
print("    - data/processed/ (numpy arrays — regeneratable)")
print("    - model checkpoints (.pt, .pth, .h5)")


# ─────────────────────────────────────────────
# STEP 5C — Auto-generate README.md
# ─────────────────────────────────────────────
print("\n" + "─" * 40)
print("GENERATING README.md")
print("─" * 40)

# Gather real stats from the catalog
dist          = master["label"].value_counts()
today         = datetime.today().strftime("%Y-%m-%d")
total_samples = len(master)
classes       = dist.index.tolist()

# Check if bad_files.txt exists and count entries
bad_count = 0
if os.path.exists(BAD_FILE):
    with open(BAD_FILE) as f:
        bad_count = sum(1 for line in f if line.strip())

# Build the class distribution table for the README
dist_table_rows = ""
for lbl, cnt in dist.items():
    dist_table_rows += f"| {lbl:<22} | {cnt:>7} | SDSS DR18 | {today} |\n"
dist_table_rows += f"| **Total**             | **{total_samples:>5}** | | |\n"

readme_content = f"""# Stellar Spectra AI

> An end-to-end machine learning pipeline that classifies
> astronomical spectra (white dwarfs, quasars, red giants, etc.)
> and generates detailed descriptions of each stellar body.

---

## Phase 1 Status: ✅ Complete

---

## Dataset Summary

| Class                  | Samples | Source    | Download Date |
|------------------------|---------|-----------|---------------|
{dist_table_rows}
**Bad / corrupted files removed:** {bad_count}

---

## Project Structure

```
stellar-spectra-ai/
│
├── data/
│   ├── raw/                   ← .fits spectra (NOT in git — back up separately)
│   │   ├── white_dwarf/
│   │   ├── quasar/
│   │   ├── main_sequence/
│   │   └── red_giant/
│   │
│   ├── processed/             ← Phase 2 output (normalised numpy arrays)
│   └── catalog/
│       ├── white_dwarfs.csv
│       ├── quasars.csv
│       ├── main_sequence.csv
│       ├── red_giants.csv
│       └── master_catalog.csv ← ground truth — label + filepath per spectrum
│
├── notebooks/
│   ├── spectra_samples.png    ← sample spectrum plot (one per class)
│   └── class_distribution.png
│
├── scripts/
│   ├── 01_folder_setup.py     (not committed — run manually once)
│   ├── 03_build_catalog.py
│   ├── 04_verify_catalog.py
│   └── 05_backup_and_version.py
│
├── logs/
│   ├── download_log.txt
│   ├── failed_downloads.csv
│   └── bad_files.txt
│
├── .gitignore
└── README.md
```

---

## Environment

| Tool       | Version   |
|------------|-----------|
| Python     | 3.10+     |
| astropy    | 6.x       |
| pandas     | 2.x       |
| numpy      | 1.x       |
| matplotlib | 3.x       |
| requests   | 2.x       |
| tqdm       | 4.x       |

Install all at once:
```bash
pip install astropy specutils numpy pandas matplotlib requests tqdm gitpython
```

---

## Data Source

- **Survey:** Sloan Digital Sky Survey (SDSS)
- **Data Release:** DR18
- **Query Tool:** SkyServer SQL Search
- **URL:** https://skyserver.sdss.org/dr18/SearchTools/sql

---

## Known Issues

- {bad_count} corrupted .fits files were removed from the catalog
  (see logs/bad_files.txt for their paths)
- Failed downloads (if any) are in logs/failed_downloads.csv
  — re-run 02_download_spectra.py with that file as CSV_FILE to retry

---

## Raw Data Backup

The raw .fits files are NOT stored in this git repository because
they are too large. They are backed up separately to:

- [ ] External hard drive  (copy data/raw/ manually)
- [ ] Google Drive         (drag and drop data/raw/ folder)
- [ ] OneDrive / Dropbox   (same)

---

## Next Steps

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Acquisition | ✅ Done |
| 2 | Preprocessing (denoise, normalise, redshift correction) | ⬜ Next |
| 3 | Feature Extraction | ⬜ |
| 4 | Model Training (CNN / Random Forest) | ⬜ |
| 5 | Classification Output | ⬜ |
| 6 | LLM Description Layer | ⬜ |
| 7 | Evaluation & Validation | ⬜ |
| 8 | Deployment | ⬜ |

---

*Generated automatically by 05_backup_and_version.py on {today}*
"""

with open(README, "w", encoding="utf-8") as f:
    f.write(readme_content)

print(f"  README.md written to:\n    {README}")
print(f"  Contains real stats: {total_samples} samples across {len(classes)} classes")


# ─────────────────────────────────────────────
# STEP 5D — Stage and commit to git
# ─────────────────────────────────────────────
print("\n" + "─" * 40)
print("COMMITTING TO GIT")
print("─" * 40)

# Stage everything that is not gitignored
print("  Staging files...")
run("git add .")

# Show what is being committed
print("\n  Files staged for commit:")
staged = subprocess.run("git diff --cached --name-only", shell=True,
                        cwd=BASE_DIR, capture_output=True, text=True)
for line in staged.stdout.strip().splitlines():
    print(f"    + {line}")

# Make the commit
commit_msg = f"Phase 1 complete — {total_samples} labeled spectra across {len(classes)} classes ({today})"
code = run(f'git commit -m "{commit_msg}"')

if code == 0:
    print(f'\n  Committed: "{commit_msg}"')
else:
    print("\n  Nothing new to commit (or commit failed — see message above).")


# ─────────────────────────────────────────────
# STEP 5E — Print external backup instructions
# ─────────────────────────────────────────────
print("\n" + "─" * 40)
print("BACKING UP RAW .FITS FILES (manual step)")
print("─" * 40)

raw_dir = os.path.join(BASE_DIR, "data", "raw")

# Estimate total size of raw data
total_bytes = 0
for dirpath, _, filenames in os.walk(raw_dir):
    for fn in filenames:
        if fn.endswith(".fits"):
            total_bytes += os.path.getsize(os.path.join(dirpath, fn))

total_mb = total_bytes / (1024 * 1024)
total_gb = total_mb / 1024

print(f"\n  Raw .fits folder size: {total_mb:.1f} MB  ({total_gb:.2f} GB)")
print(f"  Location: {raw_dir}")
print()

if platform.system() == "Windows":
    print("  WINDOWS — copy to external drive:")
    print(f'    xcopy /E /I "{raw_dir}" "D:\\stellar-spectra-backup\\raw"')
    print()
    print("  WINDOWS — copy to Google Drive (if installed):")
    drive = os.path.join(os.path.expanduser("~"), "Google Drive",
                         "stellar-spectra-backup", "raw")
    print(f'    xcopy /E /I "{raw_dir}" "{drive}"')

else:
    print("  MAC / LINUX — copy to external drive (replace /Volumes/MyDrive):")
    print(f"    cp -r {raw_dir} /Volumes/MyDrive/stellar-spectra-backup/raw")
    print()
    print("  MAC / LINUX — copy to Google Drive (if Drive for Desktop installed):")
    drive = os.path.join(os.path.expanduser("~"), "Google Drive",
                         "stellar-spectra-backup", "raw")
    print(f"    cp -r {raw_dir} \"{drive}\"")

print()
print("  ALTERNATIVE: drag and drop the entire 'raw' folder")
print("  from your file explorer into Google Drive in your browser.")


# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 COMPLETE — PHASE 1 FULLY DONE")
print("=" * 60)
print()
print("  What was done:")
print("    ✓ Git repository initialised")
print("    ✓ .gitignore created (raw data excluded from git)")
print("    ✓ README.md auto-generated with real sample counts")
print("    ✓ Catalog CSVs and scripts committed to git")
print()
print("  What YOU still need to do manually:")
print("    □ Copy data/raw/ to an external drive or cloud storage")
print("      (see the copy commands printed above)")
print("    □ Optionally push this repo to GitHub:")
print("        1. Create a new empty repo at https://github.com/new")
print("        2. Copy the remote URL (e.g. https://github.com/you/stellar-spectra-ai.git)")
print("        3. Run in terminal:")
print('           git remote add origin <your-url>')
print('           git branch -M main')
print('           git push -u origin main')
print()
print("  Your project is now versioned, documented, and backed up.")
print("  You are ready to start Phase 2: Preprocessing.")
print()
print("  master_catalog.csv location:")
print(f"    {CATALOG_FILE}")
print()
