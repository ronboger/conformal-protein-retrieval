#!/bin/bash
#SBATCH --job-name=fdr-thresholds
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/groups/doudna/projects/ronb/conformal-protein-retrieval/logs/fdr_thresholds_%j.log
#SBATCH --error=/groups/doudna/projects/ronb/conformal-protein-retrieval/logs/fdr_thresholds_%j.err

# Compute FDR thresholds at standard alpha levels for the lookup table
# This uses the Learn-then-Test (LTT) calibration from the paper

set -e

# Setup environment
export HOME2=/groups/doudna/projects/ronb
eval "$(/shared/software/miniconda3/latest/bin/conda shell.bash hook)"
conda activate conformal-s

cd /groups/doudna/projects/ronb/conformal-protein-retrieval

echo "============================================"
echo "Computing FDR Thresholds at Standard Alpha Levels"
echo "============================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

python scripts/compute_fdr_table.py \
    --calibration data/pfam_new_proteins.npy \
    --output results/fdr_thresholds.csv \
    --n-trials 10 \
    --n-calib 1000 \
    --seed 42

echo ""
echo "============================================"
echo "Completed: $(date)"
echo "============================================"
