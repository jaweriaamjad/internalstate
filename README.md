This repository contains the code to reproduce the analyses and figures in the paper **"Internal state is at least two dimensional"** (Amjad, Hiratani & Latham). Preprint: [Internal state is at least two dimensional (Research Square)](https://assets-eu.researchsquare.com/files/rs-7749081/v1_covered_0e9653c4-13a2-473f-9694-819f65090418.pdf?c=1770805206).

Code for the internal state analysis (VAE, GLM-HMM, pupil, manuscript figures).

## Layout

- **.gitignore**
- **config/** — Paths and paper settings (data_dir, results_dir, figures_dir, models_dir, VAE run name, etc.).
- **data/** — Input/processed data (e.g. glmhmm/bysubject, processed_for_vae).
- **figures/** — Manuscript figure outputs (PDFs: fig3–fig7, etc.). Writable; paths via `config/paths.json` `figures_dir`.
- **results/** — Pipeline outputs: vae_logs, pupil_logs, glmhmm_logs, fig5_cache, fig7_cache, etc.
- **src/** — Source code:
  - **src/vae/** — VAE training and latent extraction: `1-get_data4vae.py`, `2-fit_vae.py`, `3-get_latents.py`, `vae_utils.py`.
  - **src/glmhmm/** — GLM-HMM fitting, inference, and plotting.
  - **src/analysis/** — Analyses (e.g. pc_analysis, analysis_utils).
  - **src/manuscript_figures/** — Figure scripts (fig3–fig7, run_all_figures.py, plotting_utils.py) and SLURM script.
  - **src/data_preprocessing/** — Session list and filtering: `1-get_eiddf.py`, `2-filter_eiddf.py`, `preprocess_utils.py`.
  - **src/pupil/** — Pupil processing and session selection: `1_calculate_diameter.py`, `2_pupil_diameter_analysis.py`, `3_select_high_quality_sessions.py`, `pupil_utils.py`.
- **config_utils.py** (repo root) — Loads config; used by all modules.

## Utils (hybrid)

- **Shared (repo root):** `config_utils.py` loads `config/paths.json` and `config/paper_config.json`. Used by all modules.
- **Per folder, one utils file with a unique name** (no `utils/` subfolders):
  - **src/data_preprocessing/** — `preprocess_utils.py`
  - **src/vae/** — `vae_utils.py`
  - **src/glmhmm/** — Plotting and data utils (as in original adapted repo).
  - **src/pupil/** — `pupil_utils.py`
  - **src/analysis/** — `analysis_utils.py`
  - **src/manuscript_figures/** — `plotting_utils.py`

## Data source

The behavioural and session data used in this analysis come from the IBL (International Brain Laboratory) Brain Wide Map release. Overview of the dataset, structure, and download instructions are described here:

**[2025 – Brain Wide Map (IBL documentation)](https://docs.internationalbrainlab.org/notebooks_external/2025_data_release_brainwidemap.html#)**

Data can be downloaded via the ONE API (see the link above). After download, place the raw/processed data in the directory you set as **data_dir** (see below). Pipeline outputs (latents, GLM-HMM fits, caches, etc.) are written to **results_dir**. Both paths are set in **config/paths.json**.

## Data and results (where to put them)

1. Copy `config/paths.json.example` to `config/paths.json` and `config/paper_config.json.example` to `config/paper_config.json`, then edit.
2. Create the **data** and **results** directories (e.g. at repo root: `data/` and `results/`). In **config/paths.json**, set:
   - **data_dir** — Directory for input data (downloaded IBL data and any preprocessed files). Use `"data"` for the repo-root `data/` folder, or an absolute path.
   - **results_dir** — Directory for pipeline outputs (vae_logs, pupil_logs, glmhmm_logs, fig5_cache, fig7_cache, etc.). Use `"results"` for the repo-root `results/` folder, or an absolute path.
3. Add **figures_dir** and **models_dir** if needed (e.g. `"figures"` for manuscript figure PDFs, `"models"` for trained models). Paths are resolved relative to the repo root unless given as absolute.

Paths are read by `config_utils.py`; no paths are hardcoded in the scripts.

## Pipeline (order)

1. **Data preprocessing** — Run from repo root:  
   `python src/data_preprocessing/1-get_eiddf.py` then `python src/data_preprocessing/2-filter_eiddf.py`.  
   Uses `data_dir` from config; writes `1-eids_{tag}.npy`, `2-training_criteria_{tag}.pqt`, `3-sessions_intrainingorlater_{tag}.pqt`, `4-sessions_behav_filtered_{tag}.pqt`. Tag from `paper_config.json`. Requires ONE (IBL) and `config/paths.json` + `config/paper_config.json`.
2. **VAE** — Run `src/vae/1-get_data4vae.py` (reads `data_dir/4-sessions_behav_filtered_{tag}.pqt`, writes `data_dir/processed_for_vae/{tag}/`). Then train: `python src/vae/2-fit_vae.py <probL> <latent_dim> <epochs> <seed>` (e.g. `all 8 400 5312`); writes to `results_dir/vaelogs/`. Or use a provided model.  
3. **Extract latents** — `python src/vae/3-get_latents.py` (no args). Uses `vae_run_folder` and tag from `paper_config.json`; probL is fixed to `all`. Writes `results_dir/vaelogs/all_latents/<vae_run_folder>/data_with_vaelatents.pqt`.  
4. **PC analysis** — `python src/analysis/pc_analysis.py` to run PCA on latents and save `datalatents_3PC.pqt` (folder from config). No figure plotting here; use that .pqt in figure scripts (e.g. fig3.py) for scatterplot_latents, plot_wheelInPCspace, etc.
5. **GLM-HMM** — Design matrices, fitting, and inference live in **glmhmm/**. The **required pipeline** is scripts **1–9** (design matrix → global fit → per-subject fit → post-processing → predictive accuracy). Run them in order; use Slurm job scripts `4_submit.slurm` and `7_submit.slurm` with `4_job_array_pars.txt` and `7_job_array_pars.txt` for the array steps. Paths and tag come from `config_utils`; inputs use `data_dir/partially_processed` and `data_dir/processed_for_glmhmm/{tag}/`, outputs under `results_dir/glmhmmlogs/{tag}/`.  
   **Figures:** Scripts **10–13** and **script.slurm** (which runs `10_analysis_plots.py`) are for generating GLM-HMM figures only. You can skip them if you do not need those figures; the pipeline ends at step 9.
6. **Pupil** (optional) — Pupil pipeline for **paper figures** is in **src/pupil/**. Run in order: **1_calculate_diameter.py** → **2_pupil_diameter_analysis.py** (set `make_df = True` once to build prestim table) → **3_select_high_quality_sessions.py**. These produce `processed/` session parquets, `prestim_pupil_w_PCs.pqt`, and `high_quality_pupil_sessions.txt` used by fig4, fig5, fig6, fig7, etc. Paths and tag from `config_utils`; see **src/pupil/README.md** for details.  
7. Other analysis scripts  
8. **Figure scripts** — In **src/manuscript_figures/**. From repo root: `python src/manuscript_figures/fig3.py` → fig3b, fig3c; `python src/manuscript_figures/fig4.py` → fig4a–d; `python src/manuscript_figures/fig5.py` → fig5a–fig5f; `python src/manuscript_figures/fig6.py [K]` → fig6a–fig6c; `python src/manuscript_figures/fig7.py [K]` → fig7a–fig7c. Run all: `python src/manuscript_figures/run_all_figures.py`. Outputs go to **figures/** (figures_dir in config). SLURM: submit `src/manuscript_figures/run_all_figures.slurm` from repo root or from `src/manuscript_figures/`.
