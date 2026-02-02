#!/bin/bash
#SBATCH --job-name=cpr-investigate
#SBATCH --output=logs/cpr-investigate-%j.out
#SBATCH --error=logs/cpr-investigate-%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# CPR Investigation - FDR calibration and precomputed probability verification
set -e
mkdir -p logs data

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate conformal-s

cd /groups/doudna/projects/ronb/conformal-protein-retrieval

echo "========================================"
echo "CPR Investigation"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "========================================"
echo ""

echo "=== Part 1: FDR Calibration Investigation ==="
python scripts/investigate_fdr.py
echo ""

echo "=== Part 2: Precomputed Probability Verification ==="
python scripts/test_precomputed_probs.py
echo ""

echo "========================================"
echo "Completed: $(date)"
echo "========================================"
