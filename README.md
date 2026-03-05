This repository contains the code to reproduce the analyses and figures in the paper **"Internal state is at least two dimensional"** (Amjad, Hiratani & Latham).  
Preprint: [Internal state is at least two dimensional (Research Square)](https://assets-eu.researchsquare.com/files/rs-7749081/v1_covered_0e9653c4-13a2-473f-9694-819f65090418.pdf?c=1770805206).

## Data source

The behavioural and session data used in this analysis come from the IBL (International Brain Laboratory) Brain Wide Map release. Overview of the dataset, structure, and download instructions:

**[2025 – Brain Wide Map (IBL documentation)](https://docs.internationalbrainlab.org/notebooks_external/2025_data_release_brainwidemap.html#)**

Raw data can be downloaded via the ONE API (see the link above) using the IBL tag.

The GLM-HMM code was adapted from [int-brain-lab/GLM-HMM](https://github.com/int-brain-lab/GLM-HMM); running instructions for that pipeline can be found in that repository.

VAE encoder weights are provided in **models** and can be used to obtain latents via `src/vae/3-get_latents.py`.

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # or on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce manuscript figures

Processed **data** and **results** (from VAE and GLM-HMM pipelines) are hosted here: **[link to be added]**.

1. Download the data and results from the link above and extract them.
2. Create the **data** and **results** directories at the repo root (or your chosen paths).
3. Copy `config/paths.json.example` to `config/paths.json` and set **data_dir** and **results_dir** to point to those directories (e.g. `"data"` and `"results"` if using repo-root folders).
4. Run all figures from the repo root:
   ```bash
   python src/manuscript_figures/run_all_figures.py
   ```
   Figure PDFs will be written to **figures/** (or the path set as `figures_dir` in `config/paths.json`).

**Figure 8 (prior vs slope / prior vs noise):** Fig 8b uses a precompiled C program (`src/manuscript_figures/fig8b/gap`) that converts (z, gap, slope) to (z, sigma, prior). The binary is invoked automatically by `fig8.py`.
