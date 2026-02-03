#!/bin/bash
#SBATCH --job-name=fnr-partial
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/groups/doudna/projects/ronb/conformal-protein-retrieval/logs/fnr_partial_%j.log
#SBATCH --error=/groups/doudna/projects/ronb/conformal-protein-retrieval/logs/fnr_partial_%j.err

# Compute FNR thresholds for PARTIAL matches only
# Use this if the main FNR job timed out before completing partial match computation

eval "$(/shared/software/miniconda3/latest/bin/conda shell.bash hook)"
conda activate conformal-s

cd /groups/doudna/projects/ronb/conformal-protein-retrieval

echo "============================================"
echo "Computing FNR Thresholds (Partial Match Only)"
echo "============================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

python scripts/compute_fnr_table.py \
    --calibration data/pfam_new_proteins.npy \
    --output results/fnr_thresholds_partial.csv \
    --n-trials 100 \
    --n-calib 1000 \
    --seed 42 \
    --partial

echo ""
echo "============================================"
echo "Completed: $(date)"
echo "============================================"
