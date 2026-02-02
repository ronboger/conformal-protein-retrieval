#!/bin/bash
#SBATCH --job-name=cpr-calibrate-fnr
#SBATCH --output=logs/cpr-calibrate-fnr-%j.out
#SBATCH --error=logs/cpr-calibrate-fnr-%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# CPR FNR Calibration - Computes FNR thresholds
# Usage: sbatch scripts/slurm_calibrate_fnr.sh

set -e
mkdir -p logs results

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate conformal-s

echo "========================================"
echo "CPR FNR Calibration"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "========================================"

# Use the ORIGINAL calibration dataset from backup
CALIB_DATA="/groups/doudna/projects/ronb/conformal_backup/protein-conformal/data/conformal_pfam_with_lookup_dataset.npy"

if [ ! -f "$CALIB_DATA" ]; then
    echo "ERROR: Calibration data not found at $CALIB_DATA"
    exit 1
fi

echo "Using calibration data: $CALIB_DATA"
echo ""

python scripts/pfam/generate_fnr.py \
    --data_path "$CALIB_DATA" \
    --alpha 0.1 \
    --num_trials 100 \
    --n_calib 1000 \
    --output results/pfam_fnr \
    --add_date True

echo ""
echo "========================================"
echo "Completed: $(date)"
echo "========================================"
