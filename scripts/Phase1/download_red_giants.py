# -*- coding: utf-8 -*-
import os
import time
import requests
import pandas as pd
from tqdm import tqdm

CSV_FILE   = r"C:\Users\Harshit\OneDrive\Desktop\stellar-spectra-ai\data\catalog\red_giants.csv"
OUTPUT_DIR = r"C:\Users\Harshit\OneDrive\Desktop\stellar-spectra-ai\data\raw\red_giant"

# BOSS spectra use v5_13_2 path instead of 26
BASE_URL = "https://dr18.sdss.org/sas/dr18/prior-surveys/sdss4-dr17-eboss/spectro/redux/v5_13_2/spectra/lite"
LOG_FILE = r"C:\Users\Harshit\OneDrive\Desktop\stellar-spectra-ai\logs\redgiant_v2_log.txt"
DELAY    = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

df = pd.read_csv(CSV_FILE)
print(f"Loaded {len(df)} rows")

success, failed, failed_rows = 0, 0, []

with open(LOG_FILE, "a") as log:
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        plate   = int(row["plate"])
        mjd     = int(row["mjd"])
        fiberid = int(row["fiberid"])

        filename = f"spec-{plate:04d}-{mjd}-{fiberid:04d}.fits"
        url      = f"{BASE_URL}/{plate:04d}/{filename}"
        savepath = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(savepath):
            success += 1
            log.write(f"SKIP  {filename}\n")
            continue

        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(savepath, "wb") as f:
                    f.write(r.content)
                log.write(f"OK    {filename}\n")
                success += 1
            else:
                # Try alternate run path for older plates
                alt_url = f"https://dr18.sdss.org/sas/dr18/spectro/redux/26/spectra/lite/{plate:04d}/{filename}"
                r2 = requests.get(alt_url, timeout=30)
                if r2.status_code == 200:
                    with open(savepath, "wb") as f:
                        f.write(r2.content)
                    log.write(f"OK-ALT {filename}\n")
                    success += 1
                else:
                    log.write(f"FAIL  {filename}  status={r.status_code}/{r2.status_code}\n")
                    failed_rows.append(row)
                    failed += 1
        except Exception as e:
            log.write(f"ERROR {filename} -- {e}\n")
            failed_rows.append(row)
            failed += 1

        time.sleep(DELAY)

if failed_rows:
    pd.DataFrame(failed_rows).to_csv(LOG_FILE.replace("_log.txt", "_failed.csv"), index=False)

print(f"\nDone. Success: {success} | Failed: {failed}")