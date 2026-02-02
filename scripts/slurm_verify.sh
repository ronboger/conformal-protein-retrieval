#!/bin/bash
#SBATCH --job-name=cpr-verify
#SBATCH --output=logs/cpr-verify-%j.out
#SBATCH --error=logs/cpr-verify-%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
# Uncomment for GPU:
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:1

# CPR Verification Script for SLURM
# Usage: sbatch scripts/slurm_verify.sh [syn30|fdr]

set -e

# Create logs directory if needed
mkdir -p logs

# Activate conda environment
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate conformal-s

# Default to syn30 verification
CHECK=${1:-syn30}

echo "========================================"
echo "CPR Verification: $CHECK"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "========================================"

# Run verification
cpr verify --check $CHECK

echo "========================================"
echo "Completed: $(date)"
echo "========================================"
