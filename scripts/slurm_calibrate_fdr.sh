#!/bin/bash
#SBATCH --job-name=cpr-calibrate-fdr
#SBATCH --output=logs/cpr-calibrate-fdr-%j.out
#SBATCH --error=logs/cpr-calibrate-fdr-%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# CPR FDR Calibration - Reproduces paper threshold computation
# Usage: sbatch scripts/slurm_calibrate_fdr.sh

set -e
mkdir -p logs results

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate conformal-s

echo "========================================"
echo "CPR FDR Calibration"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "========================================"

# IMPORTANT: Use pfam_new_proteins.npy - the CORRECT calibration dataset
# The backup dataset (conformal_pfam_with_lookup_dataset.npy) has DATA LEAKAGE:
#   - First 50 samples all have same Pfam family "PF01266;" repeated
#   - Positive rate is 3.00% vs 0.22% in correct dataset
#   - Results in different FDR threshold (~0.999965 vs paper's ~0.999980)
# See: scripts/quick_fdr_check.py for verification
CALIB_DATA="data/pfam_new_proteins.npy"

# Check if data exists
if [ ! -f "$CALIB_DATA" ]; then
    echo "ERROR: Calibration data not found at $CALIB_DATA"
    echo "Download from Zenodo: https://zenodo.org/records/14272215"
    exit 1
fi

echo "Using calibration data: $CALIB_DATA"
echo "NOTE: Using pfam_new_proteins.npy (correct dataset without leakage)"
echo ""

# Run calibration using the ORIGINAL generate_fdr.py script (LTT method)
# This matches what was used to generate the paper's threshold
python scripts/pfam/generate_fdr.py \
    --data_path "$CALIB_DATA" \
    --alpha 0.1 \
    --num_trials 100 \
    --n_calib 1000 \
    --delta 0.5 \
    --output results/pfam_fdr \
    --add_date True

echo ""
echo "========================================"
echo "Expected result: mean lhat ≈ 0.999980"
echo "Completed: $(date)"
echo "========================================"
