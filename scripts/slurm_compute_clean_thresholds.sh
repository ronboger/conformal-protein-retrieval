#!/bin/bash
#SBATCH --job-name=clean-thresholds
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/groups/doudna/projects/ronb/conformal-protein-retrieval/logs/clean_thresholds_%j.log
#SBATCH --error=/groups/doudna/projects/ronb/conformal-protein-retrieval/logs/clean_thresholds_%j.err

# Compute CLEAN hierarchical thresholds at standard alpha levels
# Uses max hierarchical loss with euclidean distances (Paper Tables 1-2)

set -e

# Setup environment
export HOME2=/groups/doudna/projects/ronb
eval "$(/shared/software/miniconda3/latest/bin/conda shell.bash hook)"
conda activate conformal-s

cd /groups/doudna/projects/ronb/conformal-protein-retrieval

echo "============================================"
echo "Computing CLEAN Hierarchical Thresholds"
echo "============================================"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

python scripts/compute_clean_table.py \
    --calibration data/clean/clean_new_v_ec_cluster.npy \
    --output results/clean_thresholds.csv \
    --n-trials 20 \
    --n-calib 300 \
    --seed 42

echo ""
echo "============================================"
echo "Completed: $(date)"
echo "============================================"
