# Pupil pipeline (for paper figures)

Only the **numbered scripts 1–3** are required for the data used by the paper’s figure scripts (fig4, fig5, fig6, psychometric, pupil_autocorr, etc.). Run them in order.

## Run order

| Step | Script | Produces | Used by paper |
|------|--------|----------|----------------|
| 1 | `1_calculate_diameter.py` | `data_dir/pupil/processed/{eid}_pupil_data.pqt` per session | Session-level pupil for step 2 |
| 2 | `2_pupil_diameter_analysis.py` | `prestim_pupil_w_PCs.pqt` / `prestim_pupil_w_PCs_.pqt` (trial-level prestim diameters + PCs). Set `make_df = True` once to build from VAE latents + processed pupil. | fig4, fig5, fig6, psychometric, pupil_autocorr, etc. |
| 3 | `3_select_high_quality_sessions.py` | `data_dir/pupil/high_quality_pupil_sessions.txt` | fig4, fig5, fig6 (session filter) |

**Prerequisites:** VAE latents. Step 2 expects an initial `data_dir/pupil/prestim_pupil_w_PCs.pqt` with VAE trial structure (eid, subject, stimOn_time, PC1, PC2, …); if missing, create it from your VAE latents export, then run step 2 with `make_df = True` to add prestim diameters.

Pupil–PC1 and pupil–GLM-weight correlations for the paper are computed in **manuscript** figure scripts, not here.

## Utils

- **`pupil_utils.py`** — Shared helpers (e.g. `calculate_pupil_dia`, `pc_rotation`). Used by the numbered scripts; do not run directly.
