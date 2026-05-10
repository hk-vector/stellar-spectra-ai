# Stellar Spectra AI

> An end-to-end machine learning pipeline that classifies
> astronomical spectra (white dwarfs, quasars, red giants, etc.)
> and generates detailed descriptions of each stellar body.

---

## Phase 1 Status: ✅ Complete

---

## Dataset Summary

| Class                  | Samples | Source    | Download Date |
|------------------------|---------|-----------|---------------|
| white_dwarf            |     493 | SDSS DR18 | 2026-05-10 |
| main_sequence          |     479 | SDSS DR18 | 2026-05-10 |
| quasar                 |     431 | SDSS DR18 | 2026-05-10 |
| red_giant              |     379 | SDSS DR18 | 2026-05-10 |
| **Total**             | ** 1782** | | |

**Bad / corrupted files removed:** 0

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

- 0 corrupted .fits files were removed from the catalog
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

*Generated automatically by 05_backup_and_version.py on 2026-05-10*
