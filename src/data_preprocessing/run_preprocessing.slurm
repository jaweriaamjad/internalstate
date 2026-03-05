#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --output=slurm_logs/preprocess_%j.out
#SBATCH --error=slurm_logs/preprocess_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

# Run from repo root
REPO_ROOT="$SLURM_SUBMIT_DIR/../.."
cd "$REPO_ROOT"
echo "Repo root: $(pwd)"

# Activate environment if needed (adjust path as needed)
# source /path/to/venv/bin/activate

mkdir -p src/data_preprocessing/slurm_logs

# echo "Running 1-get_eiddf.py..."
# python -u src/data_preprocessing/1-get_eiddf.py

echo "Running 2-filter_eiddf.py..."
python -u src/data_preprocessing/2-filter_eiddf.py

echo "Done."
