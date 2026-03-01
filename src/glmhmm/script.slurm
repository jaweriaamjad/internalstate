#!/bin/bash

#SBATCH --partition=cpu
#SBATCH --job-name=hmm_ind
#SBATCH --output=analysis_plots_individual_%A.out
#SBATCH --error=analysis_plots_individual_%A.err
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH --time=6-00:00:00


module load cuda/11.8
python 10_analysis_plots.py 4