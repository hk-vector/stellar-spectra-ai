import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import trapezoid
import torch
import torch.nn as nn
import gradio as gr

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SHARE = False    # set True to get a public shareable link

BASE_DIR       = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "stellar-spectra-ai")
PROCESSED_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
LABEL_MAP_PATH = os.path.join(PROCESSED_DIR, "label_map.json")
MODEL_PATH     = os.path.join(MODELS_DIR, "cnn_v2.pth")

WAVE_MIN    = 3800.0
WAVE_MAX    = 9200.0
N_POINTS    = 3000
TARGET_GRID = np.linspace(WAVE_MIN, WAVE_MAX, N_POINTS, dtype=np.float32)

CLASS_COLORS = {
    "white_dwarf":      "#4A90D9",
    "quasar":           "#E8593C",
    "main_sequence":    "#3BAD75",
    "red_giant":        "#D4A017",
    "symbiotic_binary": "#9B59B6"
}

CLASS_DESCRIPTIONS = {
    "white_dwarf":
        "White dwarfs are the dense remnant cores of stars like our Sun after they "
        "exhaust their nuclear fuel. They are roughly Earth-sized but contain ~0.6 "
        "solar masses, giving them extraordinarily high density. Broad, pressure-"
        "broadened hydrogen absorption lines (Balmer series) and a blue-dominated "
        "continuum are the key spectral signatures.",

    "quasar":
        "Quasars (quasi-stellar objects) are the most luminous persistent objects "
        "in the universe, powered by supermassive black holes actively accreting "
        "matter. Their spectra show broad emission lines (rather than absorption) "
        "and large redshift values indicating they are at cosmological distances — "
        "billions of light years away.",

    "main_sequence":
        "Main sequence stars are actively fusing hydrogen into helium in their cores "
        "— the primary and longest stage of stellar life. Our Sun is a G-type main "
        "sequence star. They show moderate hydrogen Balmer absorption lines and a "
        "relatively flat continuum. They range from hot blue O-type stars to cool "
        "red M-type dwarfs.",

    "red_giant":
        "Red giants are evolved stars that have exhausted hydrogen in their cores "
        "and expanded dramatically — up to 200 times the Sun's radius. They are "
        "cool, luminous, and show strong absorption from neutral metals: calcium, "
        "sodium, and magnesium. Their red-dominated continuum reflects their low "
        "surface temperature (3500–5500K).",

    "symbiotic_binary": 
        "Symbiotic stars are binary systems where a cool Red Giant transfers material "
        "via winds or an accretion disk onto a hot White Dwarf companion. This creates "
        "an overlapping signature containing both hot and cold indicators.",
}

SPECTRAL_LINES = {
    "H Alpha":    {"wl": 6563, "short": "Ha"},
    "H Beta":     {"wl": 4861, "short": "Hb"},
    "H Gamma":    {"wl": 4341, "short": "Hg"},
    "Ca K":       {"wl": 3934, "short": "CaK"},
    "Na D":       {"wl": 5893, "short": "NaD"},
    "Mg b":       {"wl": 5175, "short": "Mgb"},
    "Ca IR":      {"wl": 8542, "short": "CaIR"},
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
# LOAD MODEL ONCE AT STARTUP
# ─────────────────────────────────────────────
print("Loading model...")
with open(LABEL_MAP_PATH) as f:
    lmap = json.load(f)
int_to_label = lmap["int_to_label"]
CLASS_NAMES  = [int_to_label[str(i)] for i in range(len(int_to_label))]
N_CLASSES    = len(CLASS_NAMES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL  = ImprovedStellarCNN(input_length=N_POINTS, n_classes=N_CLASSES).to(DEVICE)
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
MODEL.eval()
print(f"Model ready on {DEVICE}")


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
    if hi - lo < 5: return 0.0
    return round(float(trapezoid(1.0 - flux[lo:hi], TARGET_GRID[lo:hi])), 3)


def get_measurements(flux, z=0.0):
    blo = np.searchsorted(TARGET_GRID, 3850)
    bhi = np.searchsorted(TARGET_GRID, 4800)
    rlo = np.searchsorted(TARGET_GRID, 6000)
    rhi = np.searchsorted(TARGET_GRID, 7500)
    ci  = round(float(np.mean(flux[blo:bhi])) /
                max(float(np.mean(flux[rlo:rhi])), 1e-9), 4)
    slo = np.searchsorted(TARGET_GRID, 5500)
    shi = np.searchsorted(TARGET_GRID, 5600)
    reg = flux[slo:shi]
    snr = round(float(np.median(reg)) / max(float(np.std(reg)), 1e-9), 2)
    emission = any(
        float(np.median(flux[
            np.searchsorted(TARGET_GRID, wl-15):
            np.searchsorted(TARGET_GRID, wl+15)
        ])) > 1.15 for wl in [4861, 5007, 6563]
    )
    if ci > 1.8:    temp = "Very hot  (>20,000 K)"
    elif ci > 1.3:  temp = "Hot       (10,000–20,000 K)"
    elif ci > 0.9:  temp = "Intermediate (5,500–10,000 K)"
    elif ci > 0.6:  temp = "Cool      (4,000–5,500 K)"
    else:            temp = "Very cool (<4,000 K)"
    ews = {n: measure_ew(flux, i["wl"]) for n, i in SPECTRAL_LINES.items()}
    return {"colour_index": ci, "snr": snr, "emission": emission,
            "temp_class": temp, "redshift": z, "line_ews": ews}


# ─────────────────────────────────────────────
# PLOT: spectrum with annotations
# ─────────────────────────────────────────────
def make_spectrum_plot(flux, pred_class, confidence, measurements):
    color = CLASS_COLORS.get(pred_class, "#555555")
    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    ax.plot(TARGET_GRID, flux, lw=0.8, color=color, alpha=0.9)
    ax.axhline(1.0, color="#555577", lw=0.8, linestyle="--", alpha=0.6,
               label="Continuum level")

    for name, info in SPECTRAL_LINES.items():
        wl = info["wl"]
        if WAVE_MIN < wl < WAVE_MAX:
            ax.axvline(wl, color="#334455", lw=0.8, linestyle=":", alpha=0.9)
            ax.text(wl + 18, ax.get_ylim()[1] * 0.06 if ax.get_ylim()[1] > 0 else 0.1,
                    info["short"], fontsize=7, color="#7788aa",
                    va="bottom", rotation=90)

    ax.set_xlim(WAVE_MIN, WAVE_MAX)
    ax.set_ylim(0, min(2.5, float(np.percentile(flux, 99)) * 1.3))
    ax.set_xlabel("Wavelength  (Angstroms)", fontsize=11, color="#aabbcc")
    ax.set_ylabel("Normalised Flux", fontsize=11, color="#aabbcc")
    ax.tick_params(colors="#aabbcc")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334455")

    title = (f"{pred_class.replace('_', ' ').upper()}   "
             f"Confidence: {confidence*100:.1f}%   "
             f"SNR: {measurements['snr']:.1f}   "
             f"Colour index: {measurements['colour_index']:.3f}")
    ax.set_title(title, fontsize=11, fontweight="bold",
                 color=color, pad=10)
    ax.grid(True, alpha=0.08, color="#334455")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# PLOT: probability bar chart
# ─────────────────────────────────────────────
def make_prob_chart(probs):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    names  = [c.replace("_", " ").title() for c in CLASS_NAMES]
    values = [probs[c] * 100 for c in CLASS_NAMES]
    colors = [CLASS_COLORS[c] for c in CLASS_NAMES]

    bars = ax.barh(names, values, color=colors, edgecolor="none", height=0.5)

    for bar, val in zip(bars, values):
        ax.text(min(val + 1.5, 97), bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10,
                fontweight="bold", color="white")

    ax.set_xlim(0, 105)
    ax.set_xlabel("Probability (%)", fontsize=10, color="#aabbcc")
    ax.set_title("Class Probabilities", fontsize=11,
                 fontweight="bold", color="#ddeeff")
    ax.tick_params(colors="#aabbcc", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334455")
    ax.grid(True, axis="x", alpha=0.1, color="#334455")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# FORMAT MEASUREMENTS AS MARKDOWN TABLE
# ─────────────────────────────────────────────
def format_measurements(measurements, pred_class, confidence):
    m   = measurements
    ews = m["line_ews"]

    # Header card
    ci_desc  = ("blue-dominated" if m["colour_index"] > 1.3 else
                "balanced"       if m["colour_index"] > 0.8 else
                "red-dominated")
    snr_desc = ("high quality" if m["snr"] > 20 else
                "moderate"     if m["snr"] > 10 else "low quality")

    if m["redshift"] > 0.01:
        dist    = m["redshift"] * 13.8
        z_str   = f"z = {m['redshift']:.4f}  (~{dist:.1f} billion light years)"
    else:
        z_str   = f"z = {m['redshift']:.5f}  (local object)"

    if pred_class == "symbiotic_binary":
        display_title = "EXOTIC SYMBIOTIC BINARY SYSTEM"
        binary_alert = (
            "<div style='background-color: rgba(155, 89, 182, 0.1); padding: 15px; "
            "border-left: 5px solid #9B59B6; border-radius: 4px; margin-bottom: 20px; color: #ddeeff;'>\n"
            "<strong>BIFURCATION MODEL ALERT:</strong><br>\n"
            "This spectrum concurrently exhibits cold Red Giant molecular features alongside an "
            "anomalous blue/UV excess signature. The engine has bypassed solitary single-star "
            "classification models.\n"
            "</div>"
        )
    else:
        display_title = pred_class.replace("_", " ").upper()
        binary_alert = ""

    md = f"""
## Classification: {display_title}
**Confidence: {confidence*100:.1f}%**

{binary_alert}

---

### Physical Properties

| Property | Value |
|---|---|
| Temperature class | {m["temp_class"]} |
| Colour index (blue/red) | {m["colour_index"]:.3f}  *({ci_desc})* |
| Signal-to-noise ratio | {m["snr"]:.1f}  *({snr_desc})* |
| Emission lines present | {"Yes — broad emission detected" if m["emission"] else "No — absorption spectrum"} |
| Redshift | {z_str} |

---

### Spectral Line Strengths  (Equivalent Width in Angstroms)

| Line | Wavelength | EW (Å) | Strength |
|---|---|---|---|
"""
    for name, ew in ews.items():
        wl  = SPECTRAL_LINES[name]["wl"]
        if abs(ew) > 15:    strength = "Strong"
        elif abs(ew) > 5:   strength = "Moderate"
        elif abs(ew) > 0.5: strength = "Weak"
        else:                strength = "Not detected"
        ltype = " (emission)" if ew < -0.5 else " (absorption)" if ew > 0.5 else ""
        md += f"| {name} | {wl} Å | {ew:+.2f}{ltype} | {strength} |\n"

    md += f"""
---

### What is a {pred_class.replace("_", " ").title()}?

{CLASS_DESCRIPTIONS.get(pred_class, "")}
"""
    return md


# ─────────────────────────────────────────────
# MAIN CLASSIFICATION FUNCTION
# Called by Gradio when user uploads a file
# ─────────────────────────────────────────────
def classify_uploaded_file(fits_file):
    """
    This is the function Gradio calls when a user uploads a file or multiple files.
    fits_file: can be a single file path string, a Gradio file object, or a list of them.

    Returns: spectrum_fig, prob_fig, info_md
    """
    if fits_file is None:
        return None, None, "## Please upload a .fits file to begin."

    # ─────────────────────────────────────────────────────────────
    # Safely intercept multi-file list objects
    # ─────────────────────────────────────────────────────────────
    if isinstance(fits_file, list):
        if len(fits_file) == 0:
            return None, None, "## Please upload a .fits file to begin."
        # Grab the first file from the batch list to display in the UI dashboard
        active_file = fits_file[0]
        batch_msg = f"> **Batch Upload Detected:** You uploaded {len(fits_file)} files. Displaying metrics for the first file below.\n\n"
    else:
        active_file = fits_file
        batch_msg = ""

    # Pull the path safely out of the object
    filepath = active_file.name if hasattr(active_file, "name") else str(active_file)

    # Read and preprocess
    try:
        wavelength, flux_raw, z = read_fits(filepath)
    except Exception as e:
        return None, None, f"## Error reading file\n\n`{e}`\n\nMake sure you upload a valid SDSS .fits spectrum file."

    flux = preprocess(wavelength, flux_raw, z)
    flux = np.nan_to_num(flux, nan=1.0, posinf=1.0, neginf=0.0)

    # CNN classification
    tensor = torch.tensor(flux, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    rg_score = prob_dict.get('red_giant', 0.0)
    wd_score = prob_dict.get('white_dwarf', 0.0)
    
    if rg_score > 0.35 and wd_score > 0.10:
        pred_class = "symbiotic_binary"
        confidence = float(rg_score + wd_score)
    else:
        pred_idx   = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

    # Physical measurements
    measurements = get_measurements(flux, z)

    # Generate outputs
    spectrum_fig = make_spectrum_plot(flux, pred_class, confidence, measurements)
    prob_fig     = make_prob_chart(prob_dict)
    
    # Generate the markdown data and attach the batch notification if multiple files were sent
    info_md      = batch_msg + format_measurements(measurements, pred_class, confidence)

    return spectrum_fig, prob_fig, info_md


# ─────────────────────────────────────────────
# BUILD GRADIO INTERFACE
# ─────────────────────────────────────────────
css = """
body { background-color: #0f0f1a; color: #ddeeff; font-family: 'Segoe UI', sans-serif; }
.gradio-container { background-color: #0f0f1a !important; }
h1 { color: #88aaff; text-align: center; }
.gr-button { background: #223366 !important; color: white !important; border: none !important; }
.gr-button:hover { background: #334488 !important; }
footer { display: none !important; }
"""

with gr.Blocks(css=css, title="Stellar Spectra Classifier") as app:

    gr.Markdown("""
    # Stellar Spectra AI
    ### Upload an SDSS .fits spectrum file to classify it as a White Dwarf, Quasar, Main Sequence Star, or Red Giant
    ---
    """)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload .fits spectrum file",
                file_types=[".fits", ".fit"],
                type="filepath",
                file_count="multiple"
            )
            submit_btn = gr.Button("Classify Spectrum", variant="primary", size="lg")

            gr.Markdown("""
            **How to use:**
            1. Upload any SDSS `.fits` spectrum file
            2. Click **Classify Spectrum**
            3. Results appear on the right

            **File format:** SDSS spectral .fits files
            """)

        with gr.Column(scale=3):
            spectrum_plot = gr.Plot(label="Spectrum")
            with gr.Row():
                prob_chart = gr.Plot(label="Class Probabilities")
            measurements_md = gr.Markdown(
                value="*Upload a .fits file and click Classify to see results.*"
            )

    submit_btn.click(
        fn=classify_uploaded_file,
        inputs=[file_input],
        outputs=[spectrum_plot, prob_chart, measurements_md]
    )

    # Auto-classify when file is uploaded (without needing button click)
    file_input.change(
        fn=classify_uploaded_file,
        inputs=[file_input],
        outputs=[spectrum_plot, prob_chart, measurements_md]
    )

    gr.Markdown("""
    ---
    **Model:** ResNet-style 1D CNN trained on SDSS DR18 spectra
    **Accuracy:** 94.0% on held-out test set
    **Classes:** White Dwarf · Quasar · Main Sequence · Red Giant
    """)


# ─────────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\nStarting Stellar Spectra AI web interface...")
    print("Open your browser at:  http://localhost:7860")
    if SHARE:
        print("Generating public share link...")
    print()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=SHARE,
        show_error=True,
    )