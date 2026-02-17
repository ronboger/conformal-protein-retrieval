#!/bin/bash
#SBATCH --job-name=clean-centroids
#SBATCH --partition=standard
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/clean_centroids_%j.log

# Compute CLEAN EC centroid embeddings (CPU-only)
# Usage: sbatch scripts/slurm_compute_clean_centroids.sh

set -euo pipefail

eval "$(/shared/software/miniconda3/latest/bin/conda shell.bash hook)"
conda activate conformal-s

cd /groups/doudna/projects/ronb/conformal-protein-retrieval

mkdir -p logs data/clean
python scripts/compute_clean_centroid_embeddings.py
