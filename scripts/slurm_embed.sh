#!/bin/bash
#SBATCH --job-name=cpr-embed
#SBATCH --output=logs/cpr-embed-%j.out
#SBATCH --error=logs/cpr-embed-%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# CPR Embedding Script for SLURM (GPU recommended)
# Usage: sbatch scripts/slurm_embed.sh input.fasta output.npy

set -e

INPUT_FASTA=${1:?"Usage: sbatch scripts/slurm_embed.sh input.fasta output.npy"}
OUTPUT_NPY=${2:?"Usage: sbatch scripts/slurm_embed.sh input.fasta output.npy"}

mkdir -p logs

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate conformal-s

echo "========================================"
echo "CPR Embedding"
echo "Input: $INPUT_FASTA"
echo "Output: $OUTPUT_NPY"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================================"

cpr embed --input "$INPUT_FASTA" --output "$OUTPUT_NPY"

echo "========================================"
echo "Completed: $(date)"
echo "========================================"
