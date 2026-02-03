#!/bin/bash
#SBATCH --job-name=cpr-verify-probs
#SBATCH --output=logs/cpr-verify-probs-%j.out
#SBATCH --error=logs/cpr-verify-probs-%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

# CPR Precomputed Probability Verification
# Verifies that the sim->prob lookup table matches direct Venn-Abers computation
# Usage: sbatch scripts/slurm_verify_probs.sh

set -e
mkdir -p logs data

echo "========================================"
echo "CPR Probability Lookup Verification"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "========================================"
echo ""

# Activate conda environment
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate conformal-s

# Run the verification test
python scripts/test_precomputed_probs.py

echo ""
echo "========================================"
echo "Completed: $(date)"
echo "========================================"
