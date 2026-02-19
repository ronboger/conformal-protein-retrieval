#!/bin/bash
#SBATCH --job-name=cpr-tests
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/groups/doudna/projects/ronb/conformal-protein-retrieval/logs/test_%j.log
#SBATCH --error=/groups/doudna/projects/ronb/conformal-protein-retrieval/logs/test_%j.err

cd /groups/doudna/projects/ronb/conformal-protein-retrieval

eval "$(/shared/software/miniconda3/latest/bin/conda shell.bash hook)"
conda activate conformal-s

echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "============================================"

pytest tests/ -v

echo "============================================"
echo "End time: $(date)"
