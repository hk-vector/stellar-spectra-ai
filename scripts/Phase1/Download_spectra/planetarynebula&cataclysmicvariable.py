# -*- coding: utf-8 -*-
import os, time, requests, pandas as pd
from tqdm import tqdm

CSV_FILE   = r"C:\Users\Harshit\OneDrive\Desktop\stellar-spectra-ai\data\catalog\planetary_nebula.csv"
OUTPUT_DIR = r"C:\Users\Harshit\OneDrive\Desktop\stellar-spectra-ai\data\raw\planetary_nebula"
LOG_FILE   = r"C:\Users\Harshit\OneDrive\Desktop\stellar-spectra-ai\logs\pn_log.txt"

URLS = [
    "https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/lite",
    "https://dr18.sdss.org/sas/dr18/prior-surveys/sdss4-dr17-eboss/spectro/redux/v5_13_2/spectra/lite",
    "https://dr18.sdss.org/sas/dr18/spectro/redux/v5_13_2/spectra/lite",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

df = pd.read_csv(CSV_FILE, comment='#')
print(f"Loaded {len(df)} rows")

success, failed, failed_rows = 0, 0, []

with open(LOG_FILE, "a") as log:
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        plate   = int(row["plate"])
        mjd     = int(row["mjd"])
        fiberid = int(row["fiberid"])
        filename = f"spec-{plate:04d}-{mjd}-{fiberid:04d}.fits"
        savepath = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(savepath):
            success += 1
            continue

        downloaded = False
        for base_url in URLS:
            url = f"{base_url}/{plate:04d}/{filename}"
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 200:
                    with open(savepath, "wb") as f:
                        f.write(r.content)
                    log.write(f"OK  {filename}  {base_url}\n")
                    success += 1
                    downloaded = True
                    break
            except Exception:
                continue

        if not downloaded:
            log.write(f"FAIL  {filename}\n")
            failed_rows.append(row)
            failed += 1

        time.sleep(0.3)

if failed_rows:
    pd.DataFrame(failed_rows).to_csv(LOG_FILE.replace("_log.txt", "_failed.csv"), index=False)

print(f"\nDone. Success: {success} | Failed: {failed}")