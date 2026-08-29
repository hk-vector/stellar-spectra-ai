import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import trapezoid
import torch
import torch.nn as nn

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR       = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "stellar-spectra-ai")
PROCESSED_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
NOTEBOOKS      = os.path.join(BASE_DIR, "notebooks")
CATALOG_FILE   = os.path.join(BASE_DIR, "data", "catalog", "master_catalog.csv")
LABEL_MAP_PATH = os.path.join(PROCESSED_DIR, "label_map.json")
MODEL_PATH     = os.path.join(MODELS_DIR, "cnn_v2.pth")

WAVE_MIN    = 3800.0
WAVE_MAX    = 9200.0
N_POINTS    = 3000
TARGET_GRID = np.linspace(WAVE_MIN, WAVE_MAX, N_POINTS, dtype=np.float32)

# ─────────────────────────────────────────────
# CLASS INFO
# ─────────────────────────────────────────────

# Add another key:value pair here if u have addded some other class

CLASS_DESCRIPTIONS = {
    "white_dwarf": (
        "White dwarfs are the dense remnant cores of stars like our Sun\n"
        "    after they exhaust their nuclear fuel. Roughly Earth-sized but\n"
        "    containing ~0.6 solar masses. Broad hydrogen lines and a\n"
        "    blue-dominated continuum are the key identifying features."
    ),
    "quasar": (
        "Quasars are the brightest objects in the universe, powered by\n"
        "    supermassive black holes actively consuming surrounding matter.\n"
        "    Identified by broad emission lines and large redshift values\n"
        "    indicating extreme cosmological distances."
    ),
    "main_sequence": (
        "Main sequence stars are actively fusing hydrogen in their cores.\n"
        "    Our Sun is a G-type main sequence star. Identified by moderate\n"
        "    hydrogen absorption lines and a relatively flat continuum shape."
    ),
    "red_giant": (
        "Red giants are evolved stars that have expanded after exhausting\n"
        "    their core hydrogen. Identified by strong calcium and sodium\n"
        "    absorption lines and a red-dominated continuum."
    ),
    "symbiotic_binary": ("Symbiotic stars are binary systems where a cool\n"
    "Red Giant transfers material via winds or an accretion disk onto a hot\n"
    "White Dwarf companion. This creates an overlapping signature containing\n"
    "both hot and cold indicators."
    ),
}

CLASS_COLORS = {
    "white_dwarf":      "#4A90D9",
    "quasar":           "#E8593C",
    "main_sequence":    "#3BAD75",
    "red_giant":        "#D4A017",
    "symbiotic_binary": "#9B59B6",
}

SPECTRAL_LINES = {
    "Hydrogen Alpha": {"wl": 6563, "short": "Ha"},
    "Hydrogen Beta":  {"wl": 4861, "short": "Hb"},
    "Hydrogen Gamma": {"wl": 4341, "short": "Hg"},
    "Calcium K":      {"wl": 3934, "short": "CaK"},
    "Sodium D":       {"wl": 5893, "short": "NaD"},
    "Magnesium b":    {"wl": 5175, "short": "Mgb"},
    "Calcium IR":     {"wl": 8542, "short": "CaIR"},
}

# ─────────────────────────────────────────────
# CNN ARCHITECTURE
# ─────────────────────────────────────────────
class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        p = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=p),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=p),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class ImprovedStellarCNN(nn.Module):
    def __init__(self, input_length=3000, n_classes=4):
        super().__init__()
        self.input_proj  = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.res_block_1 = ResidualBlock1D(64,  7)
        self.down_1      = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.res_block_2 = ResidualBlock1D(128, 5)
        self.down_2      = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.res_block_3     = ResidualBlock1D(256, 3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_layers  = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_block_1(x)
        x = self.down_1(x)
        x = self.res_block_2(x)
        x = self.down_2(x)
        x = self.res_block_3(x)
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.feature_layers(x)
        return self.classifier(x)


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def read_fits(filepath):
    from astropy.io import fits
    with fits.open(filepath) as hdul:
        coadd  = hdul["COADD"]
        flux   = coadd.data["flux"].astype(np.float32)
        header = coadd.header
        c0     = header.get("COEFF0")
        c1     = header.get("COEFF1")
        if c0 and c1:
            wl = (10 ** (c0 + c1 * np.arange(len(flux)))).astype(np.float32)
        else:
            wl = np.arange(len(flux), dtype=np.float32)

        # Try COADD header first, then SPECOBJ table, then PRIMARY header
        z = header.get("Z", None)
        if z is None or z == 0.0:
            try:
                z = float(hdul["SPECOBJ"].data["Z"][0])
            except Exception:
                pass
        if z is None or z == 0.0:
            try:
                z = float(hdul[0].header.get("Z", 0.0))
            except Exception:
                z = 0.0

    return wl, flux, float(z)


def preprocess(wavelength, flux, z=0.0):
    # Step 1: Redshift correction
    if np.isfinite(z) and 0 <= z <= 10:
        wavelength = wavelength / (1.0 + z)

    # Step 2: Clip to valid optical range BEFORE normalisation
    # This prevents continuum fitting from running on out-of-range data
    mask = (wavelength >= 3500) & (wavelength <= 10000)
    if mask.sum() < 100:
        mask = np.ones(len(wavelength), dtype=bool)
    wavelength = wavelength[mask]
    flux       = flux[mask]

    # Step 3: Replace NaN and Inf in raw flux
    flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 4: Savitzky-Golay smoothing
    win = min(11, len(flux) - 1)
    if win % 2 == 0: win -= 1
    if win >= 5:
        flux = savgol_filter(flux, window_length=win, polyorder=3)

    # Step 5: Continuum normalisation
    wl   = wavelength.astype(np.float64)
    fl   = flux.astype(np.float64)
    mask2 = np.ones(len(fl), dtype=bool)
    cont  = np.full_like(fl, np.median(fl) if np.median(fl) > 0 else 1.0)

    for _ in range(3):
        try:
            if mask2.sum() < 10:
                break
            spl  = UnivariateSpline(wl[mask2], fl[mask2], s=1e6, k=3, ext=3)
            cont = np.maximum(spl(wl), 1e-10)
            res  = fl - cont
            neg  = res[res < 0]
            std  = np.std(neg) if len(neg) > 5 else 1.0
            mask2 = res > -1.0 * std
        except Exception:
            break

    flux_norm = np.clip(fl / cont, 0.0, 3.0).astype(np.float32)

    # Step 6: Resample to fixed grid
    sidx = np.argsort(wavelength)
    wl_s = wavelength[sidx]
    fl_s = flux_norm[sidx]
    _, ui = np.unique(wl_s, return_index=True)
    wl_s  = wl_s[ui]
    fl_s  = fl_s[ui]

    if len(wl_s) < 4:
        return np.ones(N_POINTS, dtype=np.float32)

    fn = interp1d(wl_s, fl_s, kind="linear", bounds_error=False,
                  fill_value=(float(fl_s[0]), float(fl_s[-1])))
    return np.clip(fn(TARGET_GRID), 0.0, 3.0).astype(np.float32)


# ─────────────────────────────────────────────
# PHYSICAL MEASUREMENTS
# ─────────────────────────────────────────────
def measure_ew(flux, line_wl, window_ang=30):
    lo = np.searchsorted(TARGET_GRID, line_wl - window_ang)
    hi = np.searchsorted(TARGET_GRID, line_wl + window_ang)
    if hi - lo < 5:
        return 0.0
    return round(float(trapezoid(1.0 - flux[lo:hi], TARGET_GRID[lo:hi])), 3)


def get_measurements(flux, z=0.0):
    blo = np.searchsorted(TARGET_GRID, 3850)
    bhi = np.searchsorted(TARGET_GRID, 4800)
    rlo = np.searchsorted(TARGET_GRID, 6000)
    rhi = np.searchsorted(TARGET_GRID, 7500)
    ci  = round(float(np.mean(flux[blo:bhi])) /
                max(float(np.mean(flux[rlo:rhi])), 1e-9), 4)
    slo    = np.searchsorted(TARGET_GRID, 5500)
    shi    = np.searchsorted(TARGET_GRID, 5600)
    region = flux[slo:shi]
    snr    = round(float(np.median(region)) /
                   max(float(np.std(region)), 1e-9), 2)
    emission = any(
        float(np.median(flux[
            np.searchsorted(TARGET_GRID, wl-15):
            np.searchsorted(TARGET_GRID, wl+15)
        ])) > 1.15
        for wl in [4861, 5007, 6563]
    )
    if ci > 1.8:    temp = "very hot  (>20000K)"
    elif ci > 1.3:  temp = "hot       (10000-20000K)"
    elif ci > 0.9:  temp = "intermediate (5500-10000K)"
    elif ci > 0.6:  temp = "cool      (4000-5500K)"
    else:            temp = "very cool (<4000K)"
    ews = {name: measure_ew(flux, info["wl"]) for name, info in SPECTRAL_LINES.items()}
    return {
        "colour_index": ci,
        "snr":          snr,
        "emission":     emission,
        "temp_class":   temp,
        "redshift":     z,
        "line_ews":     ews,
    }

# ─────────────────────────────────────────────
# DISPLAY FUNCTIONS
# ─────────────────────────────────────────────
def bar(value, max_value, width=20):
    filled = int(round(value / max_value * width)) if max_value > 0 else 0
    filled = max(0, min(filled, width))
    return "█" * filled + " " * (width - filled)


def print_result(filename, pred_class, confidence, probs,
                 measurements, class_names, true_label=None):
    W = 62
    print()
    print("=" * W)
    print("  STELLAR CLASSIFICATION RESULT")
    print("=" * W)
    print(f"  File:{os.path.basename(filename)}")
    if true_label:
        match = "CORRECT" if true_label == pred_class else "INCORRECT"
        print(f"  True label: {true_label.replace('_',' ').title()}  [{match}]")
    if pred_class == "symbiotic_binary":
        print(f"  Class:      EXOTIC SYMBIOTIC BINARY SYSTEM")
        rg_p = probs.get('red_giant', 0.0) * 100
        wd_p = probs.get('white_dwarf', 0.0) * 100
        print(f"  Details:    Red Giant ({rg_p:.1f}%) + White Dwarf ({wd_p:.1f}%)")
    else:
        print(f"  Class:      {pred_class.replace('_', ' ').upper()}")
    print(f"  Confidence: {confidence*100:.1f}%")
    print()
    print("  CLASS PROBABILITIES:")
    for cls in class_names:
        p    = probs[cls]
        b    = bar(p, 1.0, width=22)
        name = cls.replace("_", " ").title()
        if pred_class == "symbiotic_binary" and cls in ["red_giant", "white_dwarf"]:
            arrow = "  [BINARY COMPONENT]"
        else:
            arrow = "  <--" if cls == pred_class else ""
        print(f"    {name:<18}  {b}  {p*100:5.1f}%{arrow}")
    m = measurements
    print()
    print("  PHYSICAL MEASUREMENTS:")
    print(f"    Temperature class:    {m['temp_class']}")
    ci_desc = ("blue-dominated" if m['colour_index'] > 1.3 else
               "balanced"       if m['colour_index'] > 0.8 else
               "red-dominated")
    print(f"    Colour index:        {m['colour_index']:.3f}  ({ci_desc})")
    snr_desc = ("high quality" if m['snr'] > 20 else
                "moderate"     if m['snr'] > 10 else
                "low quality")
    print(f"    Signal-to-noise:     {m['snr']:.1f}  ({snr_desc})")
    print(f"    Emission lines:      {'Yes' if m['emission'] else 'No'}")
    if m['redshift'] > 0.01:
        dist = m['redshift'] * 13.8
        print(f"    Redshift:            z={m['redshift']:.4f}  (~{dist:.1f} billion light years)")
    else:
        print(f"    Redshift:            z={m['redshift']:.5f}  (local object)")
    print()
    print("  SPECTRAL LINE STRENGTHS (equivalent width in Angstroms):")
    max_ew = max(abs(v) for v in m["line_ews"].values()) or 1.0
    for name, ew in m["line_ews"].items():
        wl  = SPECTRAL_LINES[name]["wl"]
        b   = bar(abs(ew), max_ew, width=16)
        if abs(ew) > 15:    strength = "strong"
        elif abs(ew) > 5:   strength = "moderate"
        elif abs(ew) > 0.5: strength = "weak"
        else:                strength = "not detected"
        ltype  = "emission" if ew < -0.5 else "absorption" if ew > 0.5 else ""
        ew_str = f"{ew:+.1f} A" if abs(ew) > 0.1 else "  --  "
        print(f"    {name:<18} ({wl}A)  {b}  {ew_str:<8}  {strength} {ltype}")
    print()
    print("  WHAT THIS MEANS:")
    desc = CLASS_DESCRIPTIONS.get(pred_class, "")
    print(f"    {desc}")
    print()
    print("=" * W)
    print()


# ─────────────────────────────────────────────
# SPECTRUM PLOT
# ─────────────────────────────────────────────
def plot_spectrum(flux, pred_class, confidence, measurements, filename, save_path):
    color = CLASS_COLORS.get(pred_class, "#555555")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(TARGET_GRID, flux, lw=0.7, color=color, alpha=0.85)
    ax.axhline(1.0, color="gray", lw=0.8, linestyle="--", alpha=0.5)
    for name, info in SPECTRAL_LINES.items():
        wl = info["wl"]
        if WAVE_MIN < wl < WAVE_MAX:
            ax.axvline(wl, color="lightgray", lw=0.7, linestyle=":", alpha=0.8)
            ax.text(wl + 20, 0.05, info["short"], fontsize=7,
                    color="gray", va="bottom", rotation=90)
    ax.set_xlim(WAVE_MIN, WAVE_MAX)
    ax.set_ylim(0, min(2.5, float(np.percentile(flux, 99)) * 1.2))
    ax.set_xlabel("Wavelength (Angstroms)", fontsize=12)
    ax.set_ylabel("Normalised Flux", fontsize=12)
    ax.set_title(
        f"{pred_class.replace('_', ' ').upper()}  |  "
        f"Confidence: {confidence*100:.1f}%  |  "
        f"SNR: {measurements['snr']:.1f}  |  "
        f"Colour index: {measurements['colour_index']:.3f}\n"
        f"File: {os.path.basename(filename)}",
        fontsize=11, fontweight="bold", color=color
    )
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────
# CORE CLASSIFY FUNCTION
# ─────────────────────────────────────────────
def classify(fits_path, model, device, class_names, true_label=None, save_plot=True):
    try:
        wavelength, flux_raw, z = read_fits(fits_path)
    except Exception as e:
        print(f"\nERROR reading {fits_path}: {e}")
        return None

    flux = preprocess(wavelength, flux_raw, z)
    flux = np.nan_to_num(flux, nan=1.0, posinf=1.0, neginf=0.0)

    tensor = torch.tensor(flux, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    rg_score = prob_dict.get('red_giant', 0.0)
    wd_score = prob_dict.get('white_dwarf', 0.0)
    
    if rg_score > 0.35 and wd_score > 0.10:
        pred_class = "symbiotic_binary"
        confidence = float(rg_score + wd_score)
    else:
        pred_idx   = int(np.argmax(probs))
        pred_class = class_names[pred_idx]
        confidence = float(probs[pred_idx])

    measurements = get_measurements(flux, z)
    print_result(fits_path, pred_class, confidence, prob_dict,
                 measurements, class_names, true_label)

    if save_plot:
        plot_path = os.path.join(NOTEBOOKS, "spectrum_plot.png")
        plot_spectrum(flux, pred_class, confidence, measurements, fits_path, plot_path)
        print(f"  Spectrum plot saved to:\n    {plot_path}\n")

    return {
        "file":          os.path.basename(fits_path),
        "class":         pred_class,
        "confidence":    round(confidence, 4),
        "true_label":    true_label,
        "correct":       true_label == pred_class if true_label else None,
        "probabilities": prob_dict,
        "measurements":  {k: v for k, v in measurements.items() if k != "line_ews"},
        "line_ews":      measurements["line_ews"],
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stellar Spectra Classifier")
    parser.add_argument("fits_path", nargs="?", default=None,
                        help="Path to a .fits file (optional)")
    parser.add_argument("--batch", action="store_true",
                        help="Run on 5 random spectra from each class")
    parser.add_argument("--output", default="batch_results.csv",
                        help="Output CSV filename for batch mode")
    args = parser.parse_args()

    with open(LABEL_MAP_PATH) as f:
        lmap = json.load(f)
    int_to_label = lmap["int_to_label"]
    CLASS_NAMES  = [int_to_label[str(i)] for i in range(len(int_to_label))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ImprovedStellarCNN(input_length=N_POINTS,
                                n_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"\nModel loaded  |  Device: {device}")

    master = pd.read_csv(CATALOG_FILE)
    valid  = master[master["filepath"].apply(
        lambda p: os.path.exists(str(p)) if pd.notna(p) else False
    )].copy()

    os.makedirs(NOTEBOOKS, exist_ok=True)

    # Single file mode
    if args.fits_path:
        result = classify(args.fits_path, model, device, CLASS_NAMES, save_plot=True)
        if result:
            out = os.path.join(NOTEBOOKS, "classification_result.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"  Result saved to: {out}")

    # Batch mode
    elif args.batch:
        print(f"\nBatch mode: classifying 5 spectra from each class...\n")
        results = []
        for cls in CLASS_NAMES:
            subset = valid[valid["label"] == cls]
            sample = subset.sample(min(5, len(subset)), random_state=42)
            for _, row in sample.iterrows():
                result = classify(row["filepath"], model, device, CLASS_NAMES,
                                  true_label=row["label"], save_plot=False)
                if result:
                    results.append(result)

        batch_df = pd.DataFrame([{
            "file":       r["file"],
            "true_label": r["true_label"],
            "predicted":  r["class"],
            "confidence": r["confidence"],
            "correct":    r["correct"],
            **{f"prob_{k}": v for k, v in r["probabilities"].items()},
            **{f"ew_{k.replace(' ','_')}": v for k, v in r["line_ews"].items()},
        } for r in results])

        out_path = os.path.join(NOTEBOOKS, args.output)
        batch_df.to_csv(out_path, index=False)
        acc = batch_df["correct"].mean() * 100
        print(f"\nBatch complete: {len(batch_df)} spectra")
        print(f"Accuracy: {acc:.1f}%")
        print(f"Results saved to: {out_path}")

    # Auto mode -- one example per class
    else:
        print("\nNo file specified. Classifying one example from each class.\n")
        for cls in CLASS_NAMES:
            subset = valid[valid["label"] == cls]
            if len(subset) == 0:
                continue
            row = subset.sample(1, random_state=42).iloc[0]
            classify(row["filepath"], model, device, CLASS_NAMES,
                     true_label=row["label"],
                     save_plot=(cls == CLASS_NAMES[-1]))

        print("Done. Usage options:")
        print("  Single file:  python classify_and_display.py path/to/spectrum.fits")
        print("  Batch test:   python classify_and_display.py --batch")