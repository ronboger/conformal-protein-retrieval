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

# Use the ORIGINAL calibration dataset from backup (what paper used)
CALIB_DATA="/groups/doudna/projects/ronb/conformal_backup/protein-conformal/data/conformal_pfam_with_lookup_dataset.npy"

# Check if data exists
if [ ! -f "$CALIB_DATA" ]; then
    echo "ERROR: Calibration data not found at $CALIB_DATA"
    exit 1
fi

echo "Using calibration data: $CALIB_DATA"
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
