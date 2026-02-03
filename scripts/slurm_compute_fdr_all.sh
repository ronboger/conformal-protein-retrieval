#!/bin/bash
#SBATCH --job-name=fdr-all
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/fdr_all_%j.out
#SBATCH --error=logs/fdr_all_%j.err

# Compute FDR thresholds at all alpha levels (both exact and partial matches)
# This uses the FIXED compute_fdr_table.py with correct argument order

set -e

echo "=== FDR Threshold Computation (All Alpha Levels) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo ""

# Setup environment
eval "$(/shared/software/miniconda3/latest/bin/conda shell.bash hook)"
conda activate conformal-s

cd /groups/doudna/projects/ronb/conformal-protein-retrieval

# Calibration data
CALIB_DATA="/groups/doudna/projects/ronb/conformal_backup/protein-conformal/data/pfam_new_proteins.npy"

# Alpha levels to compute
ALPHA_LEVELS="0.001,0.005,0.01,0.02,0.05,0.1,0.15,0.2"

echo "Calibration data: $CALIB_DATA"
echo "Alpha levels: $ALPHA_LEVELS"
echo ""

# Exact match FDR thresholds
echo "=== Computing EXACT match FDR thresholds ==="
python scripts/compute_fdr_table.py \
    --calibration "$CALIB_DATA" \
    --output results/fdr_thresholds.csv \
    --n-trials 100 \
    --n-calib 1000 \
    --seed 42 \
    --alpha-levels "$ALPHA_LEVELS"

echo ""
echo "=== Computing PARTIAL match FDR thresholds ==="
python scripts/compute_fdr_table.py \
    --calibration "$CALIB_DATA" \
    --output results/fdr_thresholds_partial.csv \
    --n-trials 100 \
    --n-calib 1000 \
    --seed 42 \
    --alpha-levels "$ALPHA_LEVELS" \
    --partial

echo ""
echo "=== FDR Computation Complete ==="
echo "Results:"
echo "  - results/fdr_thresholds.csv (exact match)"
echo "  - results/fdr_thresholds_partial.csv (partial match)"
echo ""
echo "Finished: $(date)"
