import os
import time
import requests
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIGURATION — edit these two lines each run (EDIT THESE!)
# ─────────────────────────────────────────────
CSV_FILE   = r"Path to your CSV"
OUTPUT_DIR = r"Folder to save .fits files"
# ─────────────────────────────────────────────

LOG_FILE     = "../logs/download_log.txt"
FAILED_FILE  = "../logs/failed_downloads.txt"
BASE_URL = "https://dr18.sdss.org/sas/dr18/spectro/redux/v5_13_2/spectra/lite"
DELAY        = 0.5   # seconds to wait between requests

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("../logs", exist_ok=True)

# Load the CSV
df = pd.read_csv(CSV_FILE)
print(f"Loaded {len(df)} rows from {CSV_FILE}")

# Check required columns exist
required_cols = {"plate", "mjd", "fiberid"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV is missing columns. Need: {required_cols}. Found: {set(df.columns)}")

success_count = 0
fail_count    = 0
failed_rows   = []

with open(LOG_FILE, "a") as log:
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading spectra"):

        plate   = int(row["plate"])
        mjd     = int(row["mjd"])
        fiberid = int(row["fiberid"])

        # Build the filename and URL using SDSS naming convention
        filename = f"spec-{plate:04d}-{mjd}-{fiberid:04d}.fits"
        url = f"{BASE_URL}/{plate:04d}/{filename}"
        savepath = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(savepath):
            log.write(f"SKIP  {filename}\n")
            success_count += 1
            continue

        # Download with error handling
        try:
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                with open(savepath, "wb") as f:
                    f.write(response.content)
                log.write(f"OK    {filename}\n")
                success_count += 1

            else:
                log.write(f"FAIL  {filename}  status={response.status_code}\n")
                failed_rows.append(row)
                fail_count += 1

        except requests.exceptions.Timeout:
            log.write(f"TIMEOUT  {filename}\n")
            failed_rows.append(row)
            fail_count += 1

        except requests.exceptions.ConnectionError:
            log.write(f"CONNECTION ERROR  {filename}\n")
            failed_rows.append(row)
            fail_count += 1

        time.sleep(DELAY)

# Save failed rows to a CSV for retry
if failed_rows:
    pd.DataFrame(failed_rows).to_csv(FAILED_FILE.replace(".txt", ".csv"), index=False)

print(f"\nDone. Success: {success_count} | Failed: {fail_count}")
print(f"Log saved to: {LOG_FILE}")
if fail_count > 0:
    print(f"Failed rows saved to: logs/failed_downloads.csv. Re-run the script on that file to retry")