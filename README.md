# ANON4REID

**Privacy-Utility Trade-off Benchmark for Person Re-Identification**

Person re-identification (matching people across different camera views) is essential for modern surveillance but creates a significant conflict with privacy rights under the **GDPR**. While anonymization can protect identities, it often degrades the features ReID models rely on. This project explores the boundary of this conflict: **How much privacy can be gained before the ReID system becomes functionally useless?**

---

## Project Structure

```
anon4reid/
├── src/                          # Source code
│   ├── anonymizer.py             # Shared resolution handling (upscale/downscale)
│   ├── blur_anonymizer.py        # L1: Gaussian blur with MediaPipe face detection
│   ├── sd_inpaint.py             # L2: Stable Diffusion inpainting
│   ├── rad_pipeline.py           # L3: ControlNet + SD v1.5 + LCM-LoRA
│   ├── silhouette_anonymizer.py  # L4: Canny edge detection
│   ├── regenerate_blur_dataset.py    # Generate full L1 dataset (local)
│   ├── generate_gan_dataset.py       # Generate full L2 dataset (local, GPU)
│   ├── generate_blur-edge_datasets.ipynb   # Kaggle notebook: L1 + L4
│   ├── generate_sd_inpaint_dataset.ipynb   # Kaggle notebook: L2
│   ├── generate_rad_dataset.ipynb          # Kaggle notebook: L3
│   ├── baseline_osnet.py         # Baseline OSNet evaluation
│   ├── evaluate_utility.py       # Utility evaluation on anonymized datasets
│   ├── train_attacker.py         # Train MobileNetV2 identity classifier
│   ├── evaluate_privacy.py       # Privacy evaluation on anonymized queries
│   └── test_anonymization.py     # Dry run (10 sample images per pipeline)
├── app.py                        # Streamlit interactive dashboard
├── data/
│   ├── raw/Market-1501/          # Original dataset (not included)
│   └── processed/                # Anonymized dataset variants (generated)
│       ├── Market-1501-Blur/
│       ├── Market-1501-SDInpaint/
│       ├── Market-1501-RAD/
│       └── Market-1501-Edge/
├── models/                       # Trained model weights
│   ├── id_attacker.pth           # MobileNetV2 attacker weights
│   └── id_attacker_meta.json     # Attacker metadata (class mapping)
├── outputs/results/              # Evaluation results
│   ├── results_baseline.json     # Baseline OSNet metrics
│   ├── results_utility.json      # Utility metrics per anonymization level
│   └── results_privacy.json      # Privacy metrics per anonymization level
├── Report/ANON4REID/             # LaTeX report source
├── presentation.html             # reveal.js presentation
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Prerequisites

- **Python** 3.10+
- **GPU** with CUDA support (required for L2, L3 dataset generation and attacker training; CPU fallback available for L1, L4, and evaluation)
- ~10 GB disk space for datasets and model weights

---

## Setup

### Quick Setup (Recommended)

The complete project including all datasets (original + anonymized), trained model weights, and evaluation results is available for download:

**[Download from Proton Drive](https://drive.proton.me/urls/H1RSQA7RGR#hQPKySC3TDVp)**

Since the `data/` folder (~10 GB) cannot be hosted on GitLab, this is the easiest way to get a fully working setup. After downloading and extracting:

1. Copy the `data/` folder from the downloaded archive into the project root so that `data/raw/Market-1501/` and `data/processed/` exist.
2. Install dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

3. You can now skip directly to running the **dashboard** or any **evaluation script** — all datasets and model weights are already in place.

```bash
streamlit run app.py
```

---

### Manual Setup (From Scratch)

If you prefer to set up the project manually or regenerate the datasets yourself:

#### 1. Clone and create virtual environment

```bash
git clone https://github.com/C6Sucrose/anon4reid
cd anon4reid
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 3. Download Market-1501

Download the Market-1501 dataset from [its official source](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf) or a mirror, and extract it so the directory structure is:

```
data/raw/Market-1501/
├── bounding_box_train/    (12,936 images, 751 IDs)
├── bounding_box_test/     (19,732 images, 750 IDs)
└── query/                 (3,368 images, 750 IDs)
```

#### 4. Download OSNet pretrained weights

Download `osnet_x1_0_market1501.pth` from the [torchreid model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html) and place it in:

```
outputs/results/osnet_x1_0_market1501.pth
```

---

## Reproduction Guide

Run all scripts from the project root. Scripts use relative paths internally.

### Step 1: Generate anonymized datasets

Dataset generation uses Jupyter notebooks designed to run on **Kaggle** (free GPU tier). Each notebook attaches the Market-1501 dataset, processes all splits, and produces a downloadable zip archive.

| Notebook | Level(s) | Hardware | Est. Time |
|---|---|---|---|
| `src/generate_blur-edge_datasets.ipynb` | L1 (Blur) + L4 (Edge) | CPU | ~30 min |
| `src/generate_sd_inpaint_dataset.ipynb` | L2 (SD Inpainting) | GPU (T4) | ~6 hours |
| `src/generate_rad_dataset.ipynb` | L3 (RAD) | GPU (2x T4) | ~6 hours |

**To run on Kaggle:**

1. Create a new Kaggle notebook and attach the [Market-1501 dataset](https://www.kaggle.com/datasets/rayiooo/reid-market-1501).
2. Upload the notebook file (or paste its cells).
3. Enable GPU accelerator (for L2/L3 notebooks).
4. Run all cells. The notebook generates the anonymized dataset and zips it.
5. Download the zip from the Kaggle **Output** panel.
6. Extract into `data/processed/` locally:
   ```
   data/processed/Market-1501-Blur/
   data/processed/Market-1501-SDInpaint/
   data/processed/Market-1501-RAD/
   data/processed/Market-1501-Edge/
   ```

All notebooks support **resume-on-interrupt** — re-running skips already-processed images.

### Step 2: Run baseline evaluation

```bash
cd src
python baseline_osnet.py
```

Outputs baseline mAP and CMC metrics to `outputs/results/results_baseline.json`.

### Step 3: Run utility evaluation

```bash
cd src
python evaluate_utility.py
```

Evaluates OSNet on all anonymized dataset variants. Results saved to `outputs/results/results_utility.json`.

### Step 4: Train the adversarial attacker

```bash
cd src
python train_attacker.py
```

Fine-tunes MobileNetV2 on the original `bounding_box_test` split (750 IDs). Saves weights to `models/id_attacker.pth`.

### Step 5: Run privacy evaluation

```bash
cd src
python evaluate_privacy.py
```

Evaluates the attacker on anonymized query images. Results saved to `outputs/results/results_privacy.json`.

### Step 6: Launch the dashboard

```bash
streamlit run app.py
```

Opens an interactive Streamlit dashboard with three tabs:
- **Trade-off Analysis**: dual-axis charts comparing mAP and privacy leakage
- **Visual Comparison**: side-by-side query images across all anonymization levels
- **Detailed Metrics**: CMC curves and full data tables

---

## Optional: Dry run

To test all pipelines on 10 random images before full dataset generation:

```bash
cd src
python test_anonymization.py
```

Results are saved to `outputs/results/dry_run/` for visual inspection.

---

## Technical Approach

- **Baseline:** Pre-trained OSNet (`osnet_x1_0`, 2.2M params) with k-reciprocal re-ranking
- **Anonymization Pipeline — 4 levels of increasing strength:**
  1. **L1 (Gaussian Blur):** MediaPipe face detection + top-28% spatial fallback, kernel 15x15, sigma=30
  2. **L2 (SD Inpainting):** Stable Diffusion inpainting with 60%W x 80%H geometric mask, 20 steps
  3. **L3 (RAD):** ControlNet OpenPose + SD v1.5 + LCM-LoRA, 8 denoising steps
  4. **L4 (Edge Detection):** Canny edge detection (T_low=100, T_high=200)
- **Resolution Handling:** 64x128 → 256x512 (Lanczos) for L2/L3, then back to 64x128
- **Privacy Attack:** MobileNetV2 identity classifier (750 classes) trained on original test split
- **Dataset:** Market-1501 — 36,036 images, 1,501 identities, 6 cameras, 64x128 px
- **Metrics:** Utility (mAP, CMC R-1/5/10/20), Privacy (Top-1 attacker accuracy = leakage)

---

## Key Finding

The privacy-utility trade-off is a **cliff**, not a curve. Face blur retains 80.9% utility but leaks 88.66% of identities. Body-level methods achieve near-perfect privacy but destroy utility entirely. No sweet spot exists.

---
**Lecturer:** Robert Aufschlaeger — TH Deggendorf, AI Project, Summer Semester 2026
