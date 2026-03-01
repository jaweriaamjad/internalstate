#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=figures
#SBATCH --output=slurm_logs/run_all_figures_%j.out
#SBATCH --error=slurm_logs/run_all_figures_%j.err
#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH --time=4:00:00

# Run all figure scripts (fig3–fig7) on a compute node.
# REPO_ROOT: if submitted from src/manuscript_figures/, go up to repo root.
if [ -f "$SLURM_SUBMIT_DIR/run_all_figures.py" ]; then
  REPO_ROOT="$(cd "$SLURM_SUBMIT_DIR/../.." && pwd)"
else
  REPO_ROOT="$SLURM_SUBMIT_DIR"
fi
cd "$REPO_ROOT"

# Activate conda env if you use one (e.g. iblenv)
# source $(conda info --base)/etc/profile.d/conda.sh && conda activate iblenv

echo "Repo root: $REPO_ROOT"
python -u src/manuscript_figures/run_all_figures.py
