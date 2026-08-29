# Stellar Spectra AI

An end-to-end machine learning system that classifies astronomical spectra into stellar types and quasars, while automatically computing key astrophysical physical properties.

---

## Project Overview

Stellar Spectra AI processes raw 1D optical spectra from the Sloan Digital Sky Survey (SDSS DR18). Using a custom 1D Residual Convolutional Neural Network (1D ResNet), it classifies targets into four astronomical categories and extracts physical features from spectral line profiles:

* **White Dwarf:** Dense stellar remnants with broad hydrogen absorption lines.
* **Quasar:** High-redshift active galactic nuclei featuring prominent broad emission lines.
* **Main Sequence:** Hydrogen-fusing stars (e.g., G-type dwarfs).
* **Red Giant:** Evolved stars with strong molecular absorption bands.

Alongside deep learning predictions, the pipeline measures physical properties including Equivalent Width (EW), Signal-to-Noise Ratio (SNR), Color Index, and Emission Features.

---

## Model Performance

Evaluated on a held-out test split (70% Train, 15% Validation, 15% Test):

| Target Class            | Precision | Recall   | F1-Score | ROC-AUC   |
| :---------------------: | :-------: | :------: | :------: | :-------: |
| **White Dwarf**         | 0.94      | 0.86     | 0.90     | 0.987     |
| **Quasar**              | 0.94      | 1.00     | 0.97     | 0.997     |
| **Main Sequence**       | 0.88      | 0.90     | 0.89     | 0.989     |
| **Red Giant**           | 1.00      | 1.00     | 1.00     | 1.000     |
| **Overall (Macro Avg)** | **0.94**  | **0.94** | **0.94** | **0.993** |

---

## Data Preprocessing Pipeline

Raw FITS spectra undergo four sequential transformations prior to training:

1. **Rest-Frame Redshift Correction:** Shifts observed wavelengths to rest-frame using redshift (z) pulled from `SPECOBJ` FITS binary tables (λ_rest = λ_obs/(1+z)).
2. **Denoising:** Removes cosmic ray spikes via 3-sigma clipping over a 51-pixel sliding window, followed by Savitzky-Golay filtering (window = 11, polyorder = 3).
3. **Continuum Normalization:** Fits an iterative 3-pass cubic spline to the baseline, ignoring absorption dips >1σ below the fit, scaling the continuum baseline to 1.0.
4. **Resampling:** Interpolates each spectrum onto a uniform 3,000-point grid spanning 3800Å to 9200Å.

---

## Quickstart & Usage

### 1. Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/hk-vector/stellar-spectra-ai.git
cd stellar-spectra-ai
pip install -r requirements.txt